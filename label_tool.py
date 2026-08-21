"""
Interactive labeling tool for building the invoice digit dataset.

Displays one invoice image at a time. Drag a box around a single
handwritten digit, then press its value (0-9) on the keyboard to save
it -- cropped, converted to match MNIST's format, and sorted into the
right output folder.

Controls:
    Click and drag       -> draw a box around one digit
    0-9                  -> label the box you just drew and save it
    Scroll wheel         -> zoom in / out
    Right-click and drag -> pan around when zoomed in
    n                    -> move to the next invoice image
    z                    -> undo (delete) the last saved crop
    q                    -> quit

Run with:  python3 label_tool.py
"""

import glob
import os
import tkinter as tk

from PIL import Image, ImageOps, ImageTk

# --- Configuration ---
INVOICE_DIR = "invoices"        # folder of your scanned/photographed invoices
OUTPUT_DIR = "invoice_digits"   # where labeled digit crops get saved
CANVAS_MAX_SIZE = 900           # initial "fit whole page to window" viewport size
ZOOM_STEP = 1.25                # multiplier applied per scroll click
MIN_ZOOM_MULTIPLIER = 1.0       # can't zoom out past the initial fit-to-window level
MAX_ZOOM_MULTIPLIER = 8.0       # cap on how far in you can zoom


class LabelingTool:
    """
    A minimal tkinter app for cropping and labeling individual digits
    out of invoice photos, with zoom and pan for precise work on
    small or crowded handwriting.
    """

    def __init__(self, root: tk.Tk):
        """
        Set up the window, load the list of invoice images, and show
        the first one.

        Args:
            root: the tkinter root window this app runs inside.
        """
        self.root = root
        self.root.title("Invoice Digit Labeler")

        # Every invoice image we still need to go through.
        self.image_paths = sorted(
            glob.glob(os.path.join(INVOICE_DIR, "*.jpg"))
            + glob.glob(os.path.join(INVOICE_DIR, "*.jpeg"))
            + glob.glob(os.path.join(INVOICE_DIR, "*.png"))
        )
        if not self.image_paths:
            raise FileNotFoundError(
                f"No images found in '{INVOICE_DIR}/'. Put your invoice "
                f"photos there first."
            )

        self.index = 0               # which invoice we're currently on
        self.crop_count = 0          # digits saved this session, for on-screen feedback
        self.last_saved_path = None  # so 'z' can undo the most recent save

        # A single Canvas widget is both where we display the image
        # and where we draw the selection rectangle on top of it. It
        # acts as a fixed-size "viewport" -- scrollregion (set per
        # image/zoom level below) controls how much content can be
        # panned through inside that fixed window.
        self.canvas = tk.Canvas(root, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.status = tk.Label(root, text="", font=("Helvetica", 12))
        self.status.pack(fill=tk.X)

        # State for the rectangle currently being dragged.
        self.rect_id = None       # the canvas rectangle object's ID
        self.start_x = None
        self.start_y = None
        self.current_box = None   # (x1, y1, x2, y2) in *displayed* canvas coords

        # self.zoom is the current display scale; self.fit_zoom is the
        # "whole page visible" scale computed fresh for each image,
        # and doubles as the floor you can't zoom out past.
        self.zoom = 1.0
        self.fit_zoom = 1.0

        # Tracks a pending debounced re-render (see on_mouse_wheel) --
        # None means no render is currently scheduled.
        self._zoom_render_job = None

        # Mouse bindings: press to start a box, drag to resize it,
        # release to finalize its shape (but not yet its label). This
        # is event-driven programming -- these functions don't run in
        # any particular order, they run *whenever their event fires*.
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        # Right-click-drag pans the view -- scan_mark/scan_dragto are
        # tkinter Canvas's built-in support for exactly this "grab and
        # move the view" gesture, so we don't have to track scroll
        # offsets ourselves.
        self.canvas.bind("<ButtonPress-3>", lambda e: self.canvas.scan_mark(e.x, e.y))
        self.canvas.bind("<B3-Motion>", lambda e: self.canvas.scan_dragto(e.x, e.y, gain=1))

        # Mouse wheel zooms. <MouseWheel> + event.delta is the correct
        # binding on Windows specifically (Linux uses different event
        # names for this, but that's not the platform here).
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)

        # Keyboard bindings, same idea -- fires on_key whenever any
        # key is pressed while the window has focus.
        root.bind("<Key>", self.on_key)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        for digit in range(10):
            os.makedirs(os.path.join(OUTPUT_DIR, str(digit)), exist_ok=True)

        self.load_current_image()

    def load_current_image(self):
        """Load the invoice image at self.index and reset zoom/pan."""
        path = self.image_paths[self.index]

        # Some phone photos/scans store an EXIF orientation tag rather
        # than rotating the actual pixel data -- viewers like Windows'
        # File Explorer read that tag and display it correctly, but
        # PIL's Image.open() does NOT apply it automatically, showing
        # the raw, uncorrected pixels instead. exif_transpose() reads
        # that same tag and actually rotates the pixel data to match
        # -- without this, an image can look correct everywhere else
        # but wrong here specifically. A no-op if there's no such tag.
        self.original_image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")

        # The zoom level that fits the whole page inside
        # CANVAS_MAX_SIZE -- our starting point for every image, and
        # also the closest you're allowed to zoom back out to (no
        # reason to zoom out past "see the whole page").
        self.fit_zoom = min(
            CANVAS_MAX_SIZE / self.original_image.width,
            CANVAS_MAX_SIZE / self.original_image.height,
            1.0,  # never upscale a small image just to fill the window
        )
        self.zoom = self.fit_zoom

        # The canvas widget's on-screen size is fixed once per image,
        # based on this fit-to-window size -- it's the visible
        # "window" onto the page. Zooming changes how much of the
        # image that fixed window shows, not the window's own size.
        viewport_w = max(1, int(self.original_image.width * self.fit_zoom))
        viewport_h = max(1, int(self.original_image.height * self.fit_zoom))
        self.canvas.config(width=viewport_w, height=viewport_h)

        self.render_at_current_zoom()
        self.current_box = None
        self.update_status()

    def render_at_current_zoom(self):
        """Re-render the current invoice image at self.zoom onto the canvas."""
        self._zoom_render_job = None
        w = max(1, int(self.original_image.width * self.zoom))
        h = max(1, int(self.original_image.height * self.zoom))

        # NEAREST is a cheap, fast resize -- fine here since this is
        # only ever the on-screen preview. Saved crops in
        # save_current_box always come from self.original_image at
        # full resolution, completely independent of whatever quality
        # this live preview happens to render at.
        display_image = self.original_image.resize((w, h), Image.Resampling.NEAREST)

        self.tk_image = ImageTk.PhotoImage(display_image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        # scrollregion is the *virtual* content size -- when it's
        # bigger than the canvas widget's own fixed size, panning
        # (scan_mark/scan_dragto) has room to move around inside it.
        self.canvas.config(scrollregion=(0, 0, w, h))

    def on_mouse_wheel(self, event):
        """Mouse wheel scrolled: zoom in (delta > 0) or out (delta < 0)."""
        if event.delta > 0:
            new_zoom = min(self.zoom * ZOOM_STEP, self.fit_zoom * MAX_ZOOM_MULTIPLIER)
        else:
            new_zoom = max(self.zoom / ZOOM_STEP, self.fit_zoom * MIN_ZOOM_MULTIPLIER)

        if new_zoom == self.zoom:
            return

        self.zoom = new_zoom
        self.update_status()  # cheap -- fine to update immediately every tick

        # Debounce the actual (expensive) re-render: if more scroll
        # events arrive before this timer fires, cancel and push it
        # back again. A fast flurry of ticks ends up triggering ONE
        # render after scrolling settles, instead of a full resize on
        # every single tick -- that repeated work is what stuttering
        # scroll usually is.
        if self._zoom_render_job is not None:
            self.root.after_cancel(self._zoom_render_job)
        self._zoom_render_job = self.root.after(25, self.render_at_current_zoom)

    def update_status(self):
        """Refresh the label showing progress, zoom level, and instructions."""
        name = os.path.basename(self.image_paths[self.index])
        zoom_pct = int(round(self.zoom / self.fit_zoom * 100))
        self.status.config(
            text=(
                f"[{self.index + 1}/{len(self.image_paths)}] {name}   "
                f"|  zoom: {zoom_pct}%   "
                f"|  saved this session: {self.crop_count}   "
                f"|  drag=box  0-9=label  scroll=zoom  right-drag=pan  "
                f"n=next  z=undo  q=quit"
            )
        )

    def on_press(self, event):
        """Mouse button pressed: start a new selection box."""
        # canvasx/canvasy convert widget-relative click coordinates
        # into canvas-content coordinates, automatically accounting
        # for however far the view is currently panned/scrolled --
        # using raw event.x/event.y here would silently draw boxes in
        # the wrong place the moment you'd panned even slightly.
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        if self.rect_id is not None:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline="red", width=2,
        )

    def on_drag(self, event):
        """Mouse moved while held down: resize the box being drawn."""
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)
        self.canvas.coords(self.rect_id, self.start_x, self.start_y, cur_x, cur_y)

    def on_release(self, event):
        """Mouse released: box shape is finalized, waiting on a label."""
        x1, y1 = self.start_x, self.start_y
        x2 = self.canvas.canvasx(event.x)
        y2 = self.canvas.canvasy(event.y)
        # Normalize so (x1, y1) is always the top-left corner --
        # dragging up/left instead of down/right would otherwise
        # produce a box with reversed coordinates.
        self.current_box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

    def on_key(self, event):
        """Keyboard input: 0-9 labels the current box, n/z/q are commands."""
        if event.char in "0123456789" and self.current_box is not None:
            self.save_current_box(digit=int(event.char))
        elif event.char == "n":
            self.index = min(self.index + 1, len(self.image_paths) - 1)
            self.load_current_image()
        elif event.char == "z":
            self.undo_last()
        elif event.char == "q":
            self.root.quit()

    def save_current_box(self, digit: int):
        """
        Crop the current box out of the full-resolution image,
        preprocess it to match MNIST's format, and save it.

        Args:
            digit: the true digit (0-9) the person just typed for
                whatever is inside the current selection box.
        """
        x1, y1, x2, y2 = self.current_box

        # The box coordinates are in canvas-content space at the
        # CURRENT zoom level, so dividing by self.zoom maps them back
        # to the full-resolution original -- this works correctly at
        # any zoom level or pan position, not just the initial view.
        ox1, oy1 = x1 / self.zoom, y1 / self.zoom
        ox2, oy2 = x2 / self.zoom, y2 / self.zoom

        crop = self.original_image.crop((ox1, oy1, ox2, oy2))

        # Match MNIST's format: grayscale, then invert so dark ink on
        # light paper becomes light strokes on a dark background (the
        # convention the pretrained model expects).
        crop = crop.convert("L")
        crop = ImageOps.invert(crop)

        # Pad to a square BEFORE resizing, rather than resizing the
        # raw (often non-square) crop directly. A crowded double-digit
        # box is naturally wider than an isolated single digit's box --
        # resizing each straight to 28x28 would stretch them by
        # different amounts, distorting a digit's true proportions
        # (e.g. squashing or widening a "1" depending on how tight or
        # loose the box was). Padding to a square with black first,
        # then resizing, keeps every digit's real shape intact no
        # matter how the original box was drawn -- the same approach
        # MNIST itself used when it was built.
        w, h = crop.size
        side = max(w, h)
        padded = Image.new("L", (side, side), color=0)
        padded.paste(crop, ((side - w) // 2, (side - h) // 2))
        crop = padded.resize((28, 28))

        out_dir = os.path.join(OUTPUT_DIR, str(digit))
        invoice_name = os.path.splitext(os.path.basename(self.image_paths[self.index]))[0]
        # Embedding the source invoice's name in the filename means
        # you can always trace a crop back to which invoice it came
        # from -- useful later for splitting train/val/test by
        # invoice rather than by individual digit.
        filename = f"{invoice_name}_{self.crop_count:04d}.png"
        out_path = os.path.join(out_dir, filename)
        crop.save(out_path)

        self.last_saved_path = out_path
        self.crop_count += 1

        self.canvas.delete(self.rect_id)
        self.rect_id = None
        self.current_box = None
        self.update_status()

    def undo_last(self):
        """Delete the most recently saved crop, in case of a mislabel."""
        if self.last_saved_path and os.path.exists(self.last_saved_path):
            os.remove(self.last_saved_path)
            self.crop_count -= 1
            self.last_saved_path = None
            self.update_status()


def main():
    root = tk.Tk()
    LabelingTool(root)
    root.mainloop()


if __name__ == "__main__":
    main()
