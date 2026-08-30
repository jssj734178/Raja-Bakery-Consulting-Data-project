"""
One-time interactive tool for calibrating the fixed invoice template:
records where each row's Quantity and Return cells sit as PROPORTIONS
of the table border's width/height, so extract_invoice.py can later
apply those same proportions to any new scan's own detected border,
regardless of exactly where/how big that border lands on the page.

Run this ONCE per template (the project currently has exactly one
fixed pre-printed form), against any one representative scan -- a
filled-in invoice works fine, since only the printed ruled-line
geometry matters, not what's written in the cells. Run it again only
if the physical paper form itself changes.

How it works: alignment.detect_border_corners() finds the table's
outer grid border automatically (same detector extract_invoice.py will
run per new invoice), drawn on screen so you can visually confirm it
looks right before marking anything. Then, for each row top to bottom,
drag a box around that row's Quantity cell and press 'q', then drag a
box around its Return cell and press 'r', then press 'n' to confirm
the row and move to the next one. Draw boxes a little GENEROUS (a few
pixels past the ruled lines) -- handwritten digits often overflow their
cell slightly, and a tight box risks clipping a digit's stroke.

Controls:
    Click and drag        -> draw a box around a cell
    q                      -> save the current box as this row's Quantity cell
    r                      -> save the current box as this row's Return cell
    n                      -> confirm the current row (needs both q and r) and advance
    Scroll wheel           -> zoom in / out
    Right-click and drag   -> pan around when zoomed in
    z                      -> undo the most recent action (a saved box, or a
                               confirmed row -- restored as pending so you can redo it)
    s                      -> save calibration to template_calibration.json

Run with:  python calibrate_template.py path/to/reference_scan.png
"""

import json
import os
import sys
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk

from alignment import corners_aspect_ratio, detect_border_corners, pixel_to_proportion, proportion_to_pixel

# 300 DPI full-page invoice scans routinely exceed Pillow's default
# decompression-bomb pixel-count guard (meant to protect against
# malicious/untrusted image uploads) even though they're legitimate,
# trusted, self-generated scans from this project's own pipeline --
# so that guard is disabled here rather than risking a real scan
# throwing a DecompressionBombError.
Image.MAX_IMAGE_PIXELS = None

CALIBRATION_OUTPUT = "template_calibration.json"
PRODUCT_ROWS_OUTPUT = "product_rows.json"

CANVAS_MAX_SIZE = 900
ZOOM_STEP = 1.25
MIN_ZOOM_MULTIPLIER = 1.0
MAX_ZOOM_MULTIPLIER = 8.0

# Distinct, high-contrast colors so a saved Quantity box, a saved
# Return box, and the box currently being dragged never look alike.
QUANTITY_COLOR = "#00aaff"
RETURN_COLOR = "#ff8800"
DRAG_COLOR = "red"
BORDER_COLOR = "#00ff00"


class CalibrationTool:
    """
    A tkinter app for marking, per row, where the Quantity and Return
    cells sit on the fixed invoice template -- see module docstring
    for the full interaction model.
    """

    def __init__(self, root: tk.Tk, image_path: str):
        """
        Detect the reference scan's table border, then start the
        interactive row-marking session.

        Args:
            root: the tkinter root window this app runs inside.
            image_path: path to a representative scan of the fixed
                invoice template (filled-in is fine).
        """
        self.root = root
        self.root.title("Template Calibration")
        self.image_path = image_path

        # Loaded via PIL (not cv2.imread) so paths with spaces/unicode
        # -- like this project's actual invoice filenames -- always
        # work; cv2.imread can silently fail on those on Windows.
        self.original_image = Image.open(image_path).convert("RGB")
        bgr = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2BGR)

        corners = detect_border_corners(bgr)
        if corners is None:
            raise SystemExit(
                f"Could not detect a table border in '{image_path}'. "
                f"Try a different, more clearly scanned reference image."
            )
        self.corners = corners
        print(
            f"Detected table border, aspect ratio "
            f"{corners_aspect_ratio(corners):.4f} -- verify it looks "
            f"right (green outline) before marking any rows."
        )

        # Rows already confirmed (via 'n'), each a dict of proportion
        # boxes. Pending boxes are the current row's in-progress work,
        # in ORIGINAL image pixel coordinates (converted to
        # proportions only once a row is confirmed).
        self.rows = []
        self.pending_quantity_box = None
        self.pending_return_box = None

        self.canvas = tk.Canvas(root, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.status = tk.Label(root, text="", font=("Helvetica", 12))
        self.status.pack(fill=tk.X)

        self.rect_id = None
        self.start_x = None
        self.start_y = None
        self.current_box = None  # (x1, y1, x2, y2) in displayed canvas coords

        self.zoom = 1.0
        self.fit_zoom = 1.0
        self._zoom_render_job = None

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<ButtonPress-3>", lambda e: self.canvas.scan_mark(e.x, e.y))
        self.canvas.bind("<B3-Motion>", lambda e: self.canvas.scan_dragto(e.x, e.y, gain=1))
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        root.bind("<Key>", self.on_key)

        self.fit_zoom = min(
            CANVAS_MAX_SIZE / self.original_image.width,
            CANVAS_MAX_SIZE / self.original_image.height,
            1.0,
        )
        self.zoom = self.fit_zoom
        viewport_w = max(1, int(self.original_image.width * self.fit_zoom))
        viewport_h = max(1, int(self.original_image.height * self.fit_zoom))
        self.canvas.config(width=viewport_w, height=viewport_h)

        self.render_at_current_zoom()
        self.update_status()

    def render_at_current_zoom(self):
        """Re-render the reference image plus all overlays at self.zoom."""
        self._zoom_render_job = None
        w = max(1, int(self.original_image.width * self.zoom))
        h = max(1, int(self.original_image.height * self.zoom))

        display_image = self.original_image.resize((w, h), Image.Resampling.NEAREST)
        self.tk_image = ImageTk.PhotoImage(display_image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.canvas.config(scrollregion=(0, 0, w, h))

        # Detected border, so it's always visible as a sanity check
        # while marking rows.
        border_pts = [(x * self.zoom, y * self.zoom) for x, y in self.corners]
        self.canvas.create_polygon(
            border_pts, outline=BORDER_COLOR, fill="", width=2
        )

        # Already-confirmed rows, drawn faint so the working area
        # doesn't get too cluttered on a template with many rows.
        for row in self.rows:
            self._draw_proportion_box(row["quantity_box"], QUANTITY_COLOR, width=1)
            self._draw_proportion_box(row["return_box"], RETURN_COLOR, width=1)

        # The current row's pending boxes, drawn bold.
        if self.pending_quantity_box:
            self._draw_pixel_box(self.pending_quantity_box, QUANTITY_COLOR, width=3)
        if self.pending_return_box:
            self._draw_pixel_box(self.pending_return_box, RETURN_COLOR, width=3)

    def _draw_pixel_box(self, box, color, width):
        """Draw a box given in original-image pixel coords, at the current zoom."""
        x1, y1, x2, y2 = box
        self.canvas.create_rectangle(
            x1 * self.zoom, y1 * self.zoom, x2 * self.zoom, y2 * self.zoom,
            outline=color, width=width,
        )

    def _draw_proportion_box(self, prop_box, color, width):
        """Draw a box given as [fx1, fy1, fx2, fy2] proportions of the border."""
        fx1, fy1, fx2, fy2 = prop_box
        x1, y1 = proportion_to_pixel(fx1, fy1, self.corners)
        x2, y2 = proportion_to_pixel(fx2, fy2, self.corners)
        self._draw_pixel_box((x1, y1, x2, y2), color, width)

    def on_mouse_wheel(self, event):
        """Mouse wheel scrolled: zoom in (delta > 0) or out (delta < 0)."""
        if event.delta > 0:
            new_zoom = min(self.zoom * ZOOM_STEP, self.fit_zoom * MAX_ZOOM_MULTIPLIER)
        else:
            new_zoom = max(self.zoom / ZOOM_STEP, self.fit_zoom * MIN_ZOOM_MULTIPLIER)
        if new_zoom == self.zoom:
            return
        self.zoom = new_zoom
        if self._zoom_render_job is not None:
            self.root.after_cancel(self._zoom_render_job)
        self._zoom_render_job = self.root.after(25, self.render_at_current_zoom)

    def update_status(self):
        """Refresh the label showing progress and instructions."""
        row_num = len(self.rows) + 1
        have_q = "Y" if self.pending_quantity_box else "n"
        have_r = "Y" if self.pending_return_box else "n"
        self.status.config(
            text=(
                f"row {row_num}  |  quantity marked: {have_q}  "
                f"return marked: {have_r}  |  "
                f"confirmed rows: {len(self.rows)}  |  "
                f"drag=box  q=save-qty  r=save-return  n=confirm-row  "
                f"z=undo  s=save-calibration"
            )
        )

    def on_press(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        if self.rect_id is not None:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline=DRAG_COLOR, width=2,
        )

    def on_drag(self, event):
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)
        self.canvas.coords(self.rect_id, self.start_x, self.start_y, cur_x, cur_y)

    def on_release(self, event):
        x1, y1 = self.start_x, self.start_y
        x2 = self.canvas.canvasx(event.x)
        y2 = self.canvas.canvasy(event.y)
        self.current_box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

    def on_key(self, event):
        if event.char == "q":
            self.save_pending_box("quantity")
        elif event.char == "r":
            self.save_pending_box("return")
        elif event.char == "n":
            self.confirm_row()
        elif event.char == "z":
            self.undo_last()
        elif event.char == "s":
            self.save_calibration()

    def save_pending_box(self, field: str):
        """
        Convert the currently drawn box from displayed canvas
        coordinates to original-image pixel coordinates, and stash it
        as the current row's pending quantity/return box.

        Args:
            field: "quantity" or "return".
        """
        if self.current_box is None:
            self.status.config(text="Draw a box first, then press q/r.")
            return

        x1, y1, x2, y2 = self.current_box
        # Same zoom-correction as label_tool.py's save_current_box:
        # canvas coordinates are at the CURRENT zoom level, so dividing
        # by self.zoom maps them back to the full-resolution original.
        pixel_box = (x1 / self.zoom, y1 / self.zoom, x2 / self.zoom, y2 / self.zoom)

        if field == "quantity":
            self.pending_quantity_box = pixel_box
        else:
            self.pending_return_box = pixel_box

        self.canvas.delete(self.rect_id)
        self.rect_id = None
        self.current_box = None
        self.render_at_current_zoom()
        self.update_status()

    def confirm_row(self):
        """
        Finalize the current row: convert its two pending pixel boxes
        to border-relative proportions (so they can later be applied
        to any new scan's own differently-sized/positioned border),
        store the row, and reset for the next one.
        """
        if not (self.pending_quantity_box and self.pending_return_box):
            self.status.config(
                text="Mark both quantity (q) and return (r) before confirming with n."
            )
            return

        def to_proportion_box(pixel_box):
            x1, y1, x2, y2 = pixel_box
            fx1, fy1 = pixel_to_proportion(x1, y1, self.corners)
            fx2, fy2 = pixel_to_proportion(x2, y2, self.corners)
            return [fx1, fy1, fx2, fy2]

        self.rows.append(
            {
                "row_index": len(self.rows),
                "quantity_box": to_proportion_box(self.pending_quantity_box),
                "return_box": to_proportion_box(self.pending_return_box),
            }
        )
        self.pending_quantity_box = None
        self.pending_return_box = None
        self.render_at_current_zoom()
        self.update_status()

    def undo_last(self):
        """
        Step back one action: clear a pending return box first, then a
        pending quantity box, then (if neither is pending) pop the
        most recently confirmed row back into pending state so it can
        be redrawn instead of lost outright.
        """
        if self.pending_return_box is not None:
            self.pending_return_box = None
        elif self.pending_quantity_box is not None:
            self.pending_quantity_box = None
        elif self.rows:
            last_row = self.rows.pop()

            def to_pixel_box(prop_box):
                fx1, fy1, fx2, fy2 = prop_box
                x1, y1 = proportion_to_pixel(fx1, fy1, self.corners)
                x2, y2 = proportion_to_pixel(fx2, fy2, self.corners)
                return (x1, y1, x2, y2)

            self.pending_quantity_box = to_pixel_box(last_row["quantity_box"])
            self.pending_return_box = to_pixel_box(last_row["return_box"])
        else:
            self.status.config(text="Nothing to undo.")
            return

        self.render_at_current_zoom()
        self.update_status()

    def save_calibration(self):
        """
        Write the confirmed rows out to template_calibration.json, and
        create a product_rows.json skeleton (row_index -> empty
        product_name) if one doesn't already exist -- product names
        are hand-edited separately, so an existing file with real
        names filled in is never overwritten here.
        """
        if not self.rows:
            self.status.config(text="No confirmed rows to save yet.")
            return

        calibration = {
            "reference_image": os.path.basename(self.image_path),
            "reference_corners": self.corners.tolist(),
            "reference_aspect_ratio": corners_aspect_ratio(self.corners),
            "rows": self.rows,
        }
        with open(CALIBRATION_OUTPUT, "w") as f:
            json.dump(calibration, f, indent=2)

        message = f"Saved {len(self.rows)} rows to {CALIBRATION_OUTPUT}."

        if not os.path.exists(PRODUCT_ROWS_OUTPUT):
            skeleton = {
                "_readme": (
                    "Fill in each row's product_name by hand, matching "
                    "row_index against template_calibration.json's rows "
                    "(same top-to-bottom order on the printed form). "
                    "This file is never overwritten by calibrate_template.py "
                    "once it exists, so re-running calibration is safe."
                ),
                "rows": [
                    {"row_index": row["row_index"], "product_name": ""}
                    for row in self.rows
                ],
            }
            with open(PRODUCT_ROWS_OUTPUT, "w") as f:
                json.dump(skeleton, f, indent=2)
            message += f" Created {PRODUCT_ROWS_OUTPUT} skeleton -- fill in product names by hand."
        else:
            message += (
                f" {PRODUCT_ROWS_OUTPUT} already exists, left untouched -- "
                f"check its row count still matches if you added/removed rows."
            )

        print(message)
        self.status.config(text=message)


def main():
    if len(sys.argv) != 2:
        print("Usage: python calibrate_template.py path/to/reference_scan.png")
        sys.exit(1)

    root = tk.Tk()
    CalibrationTool(root, sys.argv[1])
    root.mainloop()


if __name__ == "__main__":
    main()
