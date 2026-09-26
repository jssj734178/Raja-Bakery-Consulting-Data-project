"""
Adds a total_price_box to every row of an already-calibrated
template_calibration.json, by finding the Total Price column's own
left divider automatically on the reference scan. The column's RIGHT
edge needs no detection at all -- it's simply the table's own outer
border, since Total Price is the rightmost column on the form with
nothing printed after it (see CLAUDE.md, "Pricing decision reversed").

Run this once, after calibrate_template.py has already recorded every
row's quantity_box/return_box -- this script needs those both to know
where to start looking for the Total Price column (just right of
Return) and to give each row's total_price_box the same top/bottom
edges its quantity_box already has, so all three boxes agree on which
row they belong to. Safe to re-run: it only ever adds or replaces
total_price_box on each row, never quantity_box or return_box.

How it works: reuses the calibration's own saved reference_corners
(no fresh border detection), rather than re-detecting them, so the new
proportions stay measured against the exact same coordinate system the
already-recorded boxes were. It crops one tall strip spanning every
row's height, from just left of the rightmost Return box out past the
table's own right edge, and finds the long vertical printed lines in
that strip with the same "how long is a straight run before it counts
as a real printed line, not handwriting or printed text" logic
extract_invoice.py already uses to snap cell boxes onto real lines on
every new scan (alignment.ruled_line_positions) -- just measured
against the whole table's height instead of one cell's. The first such
line found is the Return|Unit Price divider (used only to confirm the
search landed in the right place); the next one is the Unit
Price|Total Price divider this script is actually looking for.

Saves a preview image (see PREVIEW_OUTPUT) stacking the computed
total_price_box for the first, middle, and last row against the real
handwriting there. Open it and eyeball it before trusting the result --
the same way every geometry change in this project has been checked
against a real scan rather than assumed correct (see CLAUDE.md's
border-detection writeup for why that habit exists).

Run with:  python calibrate_total_price.py path/to/reference_scan.png
(the same reference scan calibrate_template.py was originally run
against -- check template_calibration.json's own "reference_image"
field if unsure which file that was).
"""

import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

from alignment import pixel_to_proportion, proportion_to_pixel, ruled_line_positions

# 300 DPI full-page invoice scans routinely exceed Pillow's default
# decompression-bomb pixel-count guard -- see the identical note in
# calibrate_template.py.
Image.MAX_IMAGE_PIXELS = None

CALIBRATION_PATH = "template_calibration.json"
PREVIEW_OUTPUT = "total_price_calibration_preview.png"

# How far left of Return's own right edge to start searching, and how
# far past the table's own right border to search to -- both as a
# fraction of the border's own width, generous enough to comfortably
# contain the Unit Price and Total Price columns whatever their exact
# width turns out to be, while starting just inside the Return|Unit
# Price divider so that divider itself is reliably the first line found.
SEARCH_LEFT_MARGIN_FRACTION = 0.01
SEARCH_RIGHT_MARGIN_FRACTION = 0.02


def _merge_nearby(positions: list, gap: float) -> list:
    """
    Collapse detected line positions that are only a few pixels apart
    into one -- ruled_line_positions() already merges adjacent
    pixel-rows of the SAME line, but real printed lines on these scans
    still sometimes come back as a couple of separate close detections
    (anti-aliasing, a slightly uneven scan), which this treats as one
    line rather than two.
    """
    merged = []
    for p in sorted(positions):
        if merged and p - merged[-1][-1] <= gap:
            merged[-1].append(p)
        else:
            merged.append([p])
    return [sum(group) / len(group) for group in merged]


def main():
    if len(sys.argv) != 2:
        print("Usage: python calibrate_total_price.py path/to/reference_scan.png")
        sys.exit(1)
    image_path = sys.argv[1]

    with open(CALIBRATION_PATH) as f:
        calibration = json.load(f)
    rows = calibration["rows"]
    corners = np.array(calibration["reference_corners"])

    if os.path.basename(image_path) != calibration["reference_image"]:
        print(
            f"Warning: '{os.path.basename(image_path)}' does not match this "
            f"calibration's own reference_image "
            f"('{calibration['reference_image']}'). The computed proportions "
            f"are only valid measured against the SAME scan "
            f"calibrate_template.py was originally run against -- double "
            f"check this is the right file before trusting the result."
        )

    image = cv2.cvtColor(np.array(Image.open(image_path).convert("RGB")), cv2.COLOR_RGB2BGR)

    top_fy = min(r["quantity_box"][1] for r in rows)
    bottom_fy = max(r["quantity_box"][3] for r in rows)
    return_right_fx = max(r["return_box"][2] for r in rows)

    x1, y1 = proportion_to_pixel(return_right_fx - SEARCH_LEFT_MARGIN_FRACTION, top_fy, corners)
    x2, y2 = proportion_to_pixel(1.0 + SEARCH_RIGHT_MARGIN_FRACTION, bottom_fy, corners)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    strip = image[y1:y2, x1:x2]

    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 10
    )
    # A real divider runs the height of the whole strip (every row at
    # once), so requiring a straight run at least half that long can't
    # be satisfied by handwriting or printed text -- same principle as
    # extract_invoice.py's DIVIDER_MIN_LENGTH_CELL_FRACTION, just
    # measured against the whole table instead of one cell.
    run_length = int(strip.shape[0] * 0.5)
    raw_positions = ruled_line_positions(binary, axis=0, run_length=run_length)
    positions = _merge_nearby(raw_positions, gap=strip.shape[1] * 0.02)

    if len(positions) < 2:
        raise SystemExit(
            f"Only found {len(positions)} candidate vertical line(s) in the "
            f"search strip -- expected at least 2 (Return|Unit Price, then "
            f"Unit Price|Total Price). Try a clearer/less crooked reference "
            f"scan."
        )

    # positions[0] is expected to be the Return|Unit Price divider (the
    # strip starts just left of it); positions[1] is the Unit
    # Price|Total Price divider this script is actually looking for.
    # Anything further right (e.g. the table's own outer border) is
    # ignored -- Total Price's right edge is defined as that border
    # directly, not detected.
    divider_x = positions[1]
    fx_left, _ = pixel_to_proportion(x1 + divider_x, y1, corners)

    for row in rows:
        fy1, fy2 = row["quantity_box"][1], row["quantity_box"][3]
        row["total_price_box"] = [fx_left, fy1, 1.0, fy2]

    with open(CALIBRATION_PATH, "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"Added total_price_box to {len(rows)} rows in {CALIBRATION_PATH}.")

    preview_rows = [rows[0], rows[len(rows) // 2], rows[-1]]
    crops = []
    for row in preview_rows:
        fx1, fy1, fx2, fy2 = row["total_price_box"]
        margin_x, margin_y = (fx2 - fx1) * 0.15, (fy2 - fy1) * 0.4
        px1, py1 = proportion_to_pixel(fx1 - margin_x, fy1 - margin_y, corners)
        px2, py2 = proportion_to_pixel(fx2 + margin_x, fy2 + margin_y, corners)
        crops.append(image[int(py1):int(py2), int(px1):int(px2)])
    max_w = max(c.shape[1] for c in crops)
    padded = [
        cv2.copyMakeBorder(c, 0, 10, 0, max_w - c.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 255))
        for c in crops
    ]
    cv2.imwrite(PREVIEW_OUTPUT, np.vstack(padded))
    print(
        f"Saved a preview of rows {preview_rows[0]['row_index']}, "
        f"{preview_rows[1]['row_index']}, {preview_rows[2]['row_index']} "
        f"(first/middle/last) to {PREVIEW_OUTPUT} -- open it and check the "
        f"box lands on the Total Price column before trusting this."
    )


if __name__ == "__main__":
    main()
