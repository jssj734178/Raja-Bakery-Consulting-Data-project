"""
Per-invoice extraction: given a NEW scanned invoice, applies the
one-time template calibration (template_calibration.json,
product_rows.json) to locate every row's Qty and Return cells, reads
each one with digit_reader, subtracts Return from Qty per row, and
writes out the resulting (product, quantity) pairs plus a per-field
confidence/flag status -- everything the (not yet built) human review
step will need to confirm or correct before anything reaches Odoo.

Run with:  python extract_invoice.py path/to/new_scan.png [more_scans.png ...]
(e.g. python extract_invoice.py invoices/*.png for a whole batch).
"""

import argparse
import glob
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from PIL import Image

from alignment import detect_border_corners, proportion_to_pixel, validate_aspect_ratio
from digit_reader import classify_blobs, load_model, segment_digit_blobs, threshold_cell

# 300 DPI full-page invoice scans routinely exceed Pillow's default
# decompression-bomb pixel-count guard even though they're legitimate,
# trusted, self-generated scans from this project's own pipeline --
# see the identical note in calibrate_template.py.
Image.MAX_IMAGE_PIXELS = None

CALIBRATION_PATH = "template_calibration.json"
PRODUCT_ROWS_PATH = "product_rows.json"
CHECKPOINT_PATH = "checkpoints/digit_cnn_finetuned.pt"

# Each cell is cropped twice, with different margins, because reading
# and reviewing want opposite things from the crop.
#
# The ANALYSIS crop is deliberately oversized -- a full cell height of
# extra room vertically. That looks wrong at first glance (it pulls in
# the rows above and below wholesale) but it is exactly what
# digit_reader's ownership test needs: it decides which ink belongs to
# this cell by asking what fraction of each blob lies inside the cell
# box, and that fraction is only meaningful if the blob is present in
# full. Crop tightly and a neighbouring row's digit arrives sliced off
# at the crop edge, where the visible sliver can sit mostly inside
# this cell and be claimed by it -- which is exactly how blank cells
# used to read as digits. Horizontal margin is smaller: it only has to
# contain a digit overflowing its column, and the Qty and Return
# columns sit close enough together that reaching further mostly just
# imports the neighbouring column's ink for the ownership test to
# throw away again.
ANALYSIS_VERTICAL_MARGIN_FRACTION = 1.0
ANALYSIS_HORIZONTAL_MARGIN_FRACTION = 0.3

# The REVIEW crop is the one saved to disk for a human to look at, so
# it is kept tight -- the cell plus enough margin that a digit sitting
# high or low in its row isn't clipped, and no more. A reviewer
# confirming a value wants the field itself, not its neighbours.
REVIEW_VERTICAL_MARGIN_FRACTION = 0.25
REVIEW_HORIZONTAL_MARGIN_FRACTION = 0.05

# How long a vertical stroke must be, as a multiple of the cell's
# height, to count as a column divider in snap_cell_box(). Must stay
# below 1 + ANALYSIS_VERTICAL_MARGIN_FRACTION (the analysis crop's
# full height), or no divider could ever qualify.
DIVIDER_MIN_LENGTH_CELL_FRACTION = 1.2


def _separate_overlapping_boxes(calibration: dict):
    """
    Nudge the calibrated cell boxes apart, in place, so that no two
    of them overlap.

    The cells on the printed form don't overlap -- they're divided by
    a single ruled line -- but the boxes in template_calibration.json
    were drawn by hand around them, and neighbouring boxes routinely
    run a little into each other (in the shipped calibration, the
    Return boxes for rows 1 and 2 share a band about 60px tall).

    That slop is harmless for cropping, but not for digit_reader's
    ownership test, which asks whether the majority of a blob's ink
    lies inside the cell box: ink sitting in a shared band is
    majority-owned by BOTH cells and gets read twice. The invoice
    where a Return of "1" was written low enough to be counted for its
    own row and the row beneath it is exactly that.

    Splitting each overlap down the middle restores what the form
    already guarantees -- that every point on the page belongs to at
    most one cell -- so the ownership test can stay a purely local
    decision per cell instead of needing to compare cells against each
    other.

    Args:
        calibration: the parsed template_calibration.json. Its row
            boxes are modified in place.
    """
    def split(low_box, low_index, high_box, high_index):
        """Pull two boxes apart to meet at the midpoint of their overlap."""
        if low_box[low_index] > high_box[high_index]:
            midpoint = (low_box[low_index] + high_box[high_index]) / 2
            low_box[low_index] = midpoint
            high_box[high_index] = midpoint

    rows = sorted(calibration["rows"], key=lambda r: r["quantity_box"][1])

    for row in rows:
        # Qty sits left of Return, so Qty's right edge meets Return's
        # left edge.
        split(row["quantity_box"], 2, row["return_box"], 0)

    # Within each column, a row's bottom edge meets the next row's top.
    for upper, lower in zip(rows, rows[1:]):
        for box_key in ("quantity_box", "return_box"):
            split(upper[box_key], 3, lower[box_key], 1)


def load_calibration(calibration_path: str = CALIBRATION_PATH, product_rows_path: str = PRODUCT_ROWS_PATH):
    """
    Load the saved template calibration and its row->product name mapping.

    Returns:
        (calibration, product_names) -- calibration is the parsed
        template_calibration.json, with any overlap between adjacent
        cell boxes resolved (see _separate_overlapping_boxes);
        product_names maps row_index to its product name string from
        product_rows.json.
    """
    with open(calibration_path) as f:
        calibration = json.load(f)
    # Adjusted on load rather than rewritten into the file: the saved
    # calibration stays exactly what the operator drew, and re-running
    # calibrate_template.py never has to know about this.
    _separate_overlapping_boxes(calibration)
    with open(product_rows_path) as f:
        product_rows = json.load(f)
    product_names = {row["row_index"]: row["product_name"] for row in product_rows["rows"]}
    return calibration, product_names


def crop_cell(
    image_bgr: np.ndarray,
    box_proportions: list,
    corners: np.ndarray,
    horizontal_margin: float,
    vertical_margin: float,
) -> tuple[np.ndarray, tuple]:
    """
    Crop one calibrated cell (Qty or Return) out of a new scan,
    rotated upright, given its saved proportion box and this scan's
    own detected border.

    Maps all 4 corners of the box individually through
    proportion_to_pixel (not just the 2 diagonal corners that are
    stored), then warps that quadrilateral to an axis-aligned crop
    rather than taking its bounding box. The rotation matters for more
    than tidiness: these scans sit around 1.4 degrees off square, and
    digit_reader isolates the form's printed ruled lines with
    axis-aligned morphological kernels, which cannot fit inside a line
    that drifts ~26px across the width of a cell. Handing it an
    already-upright crop is what lets that step work -- see the
    LINE_KERNEL_CELL_FRACTION note there.

    The angle comes from the box's own mapped top edge, so it tracks
    whatever local rotation this scan has where this particular cell
    sits, with no extra detection step and no full-page warp.

    Args:
        image_bgr: the full invoice scan.
        box_proportions: [fx1, fy1, fx2, fy2] as saved by
            calibrate_template.py (top-left and bottom-right corners,
            as proportions of the border).
        corners: this scan's own detected border corners, from
            alignment.detect_border_corners().
        horizontal_margin: extra width to include around the cell, as
            a fraction of the cell's own width, split evenly between
            the two sides.
        vertical_margin: the same, for height.

    Returns:
        (crop, cell_box) -- the upright BGR crop, and the cell's own
        box within it as (x1, y1, x2, y2) in crop pixel coordinates,
        i.e. the crop minus its margin. digit_reader needs that box to
        tell this cell's ink from a neighbouring row's.
    """
    fx1, fy1, fx2, fy2 = box_proportions
    top_left, top_right, bottom_right, bottom_left = (
        np.array(proportion_to_pixel(fx, fy, corners), dtype=float)
        for fx, fy in [(fx1, fy1), (fx2, fy1), (fx2, fy2), (fx1, fy2)]
    )

    center = (top_left + top_right + bottom_right + bottom_left) / 4.0
    angle = math.degrees(
        math.atan2(top_right[1] - top_left[1], top_right[0] - top_left[0])
    )
    # Average the two opposite edges: the mapped quadrilateral is only
    # approximately a rectangle, so neither edge alone is definitive.
    cell_width = (
        np.linalg.norm(top_right - top_left) + np.linalg.norm(bottom_right - bottom_left)
    ) / 2
    cell_height = (
        np.linalg.norm(bottom_left - top_left) + np.linalg.norm(bottom_right - top_right)
    ) / 2

    crop_width = int(round(cell_width * (1 + horizontal_margin)))
    crop_height = int(round(cell_height * (1 + vertical_margin)))

    # Rotate about the cell's centre, then shift that centre to the
    # centre of the output. warpAffine only evaluates the output
    # pixels it is asked for, so this stays cheap even though the
    # source is a full 300 DPI page. BORDER_REPLICATE keeps a cell at
    # the very edge of the scan from picking up a black margin that
    # would threshold as a huge block of ink.
    rotation = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle, 1.0)
    rotation[0, 2] += crop_width / 2 - center[0]
    rotation[1, 2] += crop_height / 2 - center[1]
    crop = cv2.warpAffine(
        image_bgr,
        rotation,
        (crop_width, crop_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    cell_box = (
        (crop_width - cell_width) / 2,
        (crop_height - cell_height) / 2,
        (crop_width + cell_width) / 2,
        (crop_height + cell_height) / 2,
    )
    return crop, cell_box


def _ruled_line_positions(binary: np.ndarray, axis: int, run_length: int) -> list:
    """
    Locate the form's printed ruled lines in a deskewed crop.

    Args:
        binary: thresholded crop, ink as foreground.
        axis: 0 to find vertical lines (returning x positions), 1 to
            find horizontal ones (returning y positions).
        run_length: how long a straight run has to be to count as a
            printed line rather than handwriting.

    Returns:
        The centre position of each detected line, in ascending order.
        Adjacent columns/rows of the same line are collapsed into one
        entry -- a printed line is several pixels thick, and what the
        caller wants is one number per line.
    """
    kernel_size = (1, run_length) if axis == 0 else (run_length, 1)
    lines = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    )

    # Count ink along each line's own direction, leaving one value per
    # candidate position, then keep the positions where a line
    # actually is. Half the run length is a deliberately forgiving
    # cut: it takes a real line to get near it, but a line broken up
    # where other rules cross it still clears it.
    profile = lines.sum(axis=axis) / 255
    present = profile > run_length / 2

    positions = []
    start = None
    for index, is_line in enumerate(present):
        if is_line and start is None:
            start = index
        elif not is_line and start is not None:
            positions.append((start + index - 1) / 2)
            start = None
    if start is not None:
        positions.append((start + len(present) - 1) / 2)
    return positions


def snap_cell_box(crop_bgr: np.ndarray, cell_box: tuple) -> tuple:
    """
    Move a calibrated cell box onto the ruled lines that actually
    bound that cell on this particular scan.

    The calibration records each cell as proportions of the table
    border, which puts the box in the right cell on any scan but not
    on exactly the right pixels: the proportion that lands on the
    column divider for the reference scan lands some way past it on
    another. That was not a small discrepancy in practice. On roughly
    a third of the pages sampled, the Return box overran the divider
    far enough to swallow the printed unit price beyond it, so the
    "$4.00" in the next column was read as the row's return quantity
    -- silently, and in nearly every row of the affected page.

    Rather than trust the proportions to the pixel, take from them
    only what they're reliable for -- WHICH cell this is -- and get
    the cell's actual bounds from the scan itself, snapping each edge
    to the nearest printed line beyond the box's centre. This is the
    same principle the calibration already rests on, that a new scan's
    own detected geometry beats anything remembered from the
    reference scan, applied one level down from the table border to
    the individual cell.

    Args:
        crop_bgr: the deskewed crop from crop_cell(), which must
            extend past the cell far enough to contain the lines
            bounding it.
        cell_box: (x1, y1, x2, y2) of the calibrated box within that
            crop.

    Returns:
        The snapped box, in the same form, never larger than the one
        passed in. Any edge with no printed line inside it keeps its
        calibrated position, so a faint or broken line degrades this
        back to current behaviour for that one edge rather than moving
        it somewhere wrong.
    """
    x1, y1, x2, y2 = cell_box
    width, height = x2 - x1, y2 - y1
    binary = threshold_cell(crop_bgr, height)

    # A horizontal line only counts if it runs at least half the cell's
    # width -- no handwriting or printed text is that long sideways.
    #
    # A vertical line has to be much longer than that: more than the
    # cell's full height. Half a cell's height is not enough, because
    # the printed "$4.20" in the Unit Price column just right of the
    # Return box has strokes that tall. Those were being taken for the
    # column divider, stopping the Return box's edge in the middle of
    # the price instead of on the divider, so the price was read as a
    # return quantity ("$4.2" as "831"). A real divider runs unbroken
    # through the rows above and below too, which the analysis crop
    # includes, so it clears this easily; no digit or printed
    # character does.
    verticals = _ruled_line_positions(
        binary, axis=0, run_length=int(height * DIVIDER_MIN_LENGTH_CELL_FRACTION)
    )
    horizontals = _ruled_line_positions(binary, axis=1, run_length=int(width * 0.5))

    def snap(edge, centre, candidates, limit):
        """Pull an edge in to the innermost printed line it has crossed."""
        # Only lines strictly INSIDE the box are candidates, so this can
        # shrink the box but never grow it. That asymmetry is the whole
        # point: a box reaching past its divider is reading a
        # neighbouring column and must be pulled back, whereas a box
        # sitting slightly shy of its divider is harmless, and letting
        # edges expand outward to find a line turned a correctly-read
        # "50" into "501" by reaching far enough to pick up a stray
        # mark. When in doubt, read less of the page, not more.
        inside = [
            p for p in candidates
            if min(centre, edge) < p < max(centre, edge) and abs(p - edge) <= limit
        ]
        return min(inside, key=lambda p: abs(p - edge)) if inside else edge

    # An edge may only move by a fraction of the cell, so a stray line
    # elsewhere in the crop -- the next row's rule, the far side of a
    # neighbouring column -- can never capture it.
    centre_x, centre_y = (x1 + x2) / 2, (y1 + y2) / 2
    return (
        snap(x1, centre_x, verticals, width * 0.4),
        snap(y1, centre_y, horizontals, height * 0.4),
        snap(x2, centre_x, verticals, width * 0.4),
        snap(y2, centre_y, horizontals, height * 0.4),
    )


def extract_invoice(image_path: str, output_dir: str, model, device) -> dict:
    """
    Run the full per-invoice extraction pipeline against one scanned
    invoice image: detect its border, apply the saved calibration to
    every row's Qty/Return cells, read each field, and subtract
    Return from Qty per row.

    Args:
        image_path: path to a new invoice scan (a PNG rendered by
            pdf_to_images.py, or any image of the same fixed template).
        output_dir: directory to write results.json and per-cell crop
            images into (created if missing) -- the crops are saved
            regardless of whether a field was flagged, since the
            (not yet built) review step needs to show every field's
            source image, not just the flagged ones.
        model: the fine-tuned DigitCNN, in eval mode.
        device: torch device to run inference on.

    Returns:
        The same dict that gets written to results.json -- see
        _save_results for its shape.
    """
    calibration, product_names = load_calibration()
    reference_ratio = calibration["reference_aspect_ratio"]

    image = cv2.cvtColor(np.array(Image.open(image_path).convert("RGB")), cv2.COLOR_RGB2BGR)
    corners = detect_border_corners(image)

    os.makedirs(output_dir, exist_ok=True)
    crops_dir = os.path.join(output_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    # Both reliability guards from the project plan: outright
    # detection failure, or a detected border whose shape doesn't
    # plausibly match the calibrated template. Either way, the whole
    # invoice is flagged for manual handling rather than extracting
    # from coordinates that can't be trusted.
    if corners is None:
        result = {"invoice_flagged": True, "reason": "border_not_detected", "rows": []}
        _save_results(result, output_dir)
        return result

    if not validate_aspect_ratio(corners, reference_ratio):
        result = {"invoice_flagged": True, "reason": "aspect_ratio_mismatch", "rows": []}
        _save_results(result, output_dir)
        return result

    def segment_cell(row, field, box_key):
        """Crop, segment and save one cell -- everything but the model."""
        analysis_crop, cell_box = crop_cell(
            image,
            row[box_key],
            corners,
            ANALYSIS_HORIZONTAL_MARGIN_FRACTION,
            ANALYSIS_VERTICAL_MARGIN_FRACTION,
        )
        cell_box = snap_cell_box(analysis_crop, cell_box)
        segmented = segment_digit_blobs(analysis_crop, cell_box)

        # Saved separately and tighter -- what a reviewer needs to
        # see is not what the reader needs to measure against.
        review_crop, _ = crop_cell(
            image,
            row[box_key],
            corners,
            REVIEW_HORIZONTAL_MARGIN_FRACTION,
            REVIEW_VERTICAL_MARGIN_FRACTION,
        )
        cv2.imwrite(
            os.path.join(crops_dir, f"row{row['row_index']:02d}_{field}.png"), review_crop
        )
        return segmented

    # The image work for each cell is independent and OpenCV releases
    # Python's lock while it runs, so cells are segmented on all cores
    # at once. The model then reads them one at a time in row order,
    # exactly as before, so results don't depend on thread timing.
    fields = (("qty", "quantity_box"), ("return", "return_box"))
    with ThreadPoolExecutor() as pool:
        segmented = {
            (row["row_index"], field): pool.submit(segment_cell, row, field, box_key)
            for row in calibration["rows"]
            for field, box_key in fields
        }

    rows_out = []
    for row in calibration["rows"]:
        row_index = row["row_index"]
        product_name = product_names.get(row_index, "")

        results = {
            field: classify_blobs(*segmented[(row_index, field)].result(), model, device)
            for field, _ in fields
        }
        qty_result, return_result = results["qty"], results["return"]

        # More returned than was ordered is almost never real on this
        # form -- it is what a misread Return box looks like (e.g. the
        # printed price next door read as a number). Checked here, not
        # in digit_reader, because it needs both fields of the row.
        if return_result["value"] > qty_result["value"]:
            return_result["flag_reasons"].append("return_exceeds_quantity")
            return_result["flagged"] = True

        rows_out.append(
            {
                "row_index": row_index,
                "product_name": product_name,
                "quantity": qty_result["value"],
                "quantity_confidences": qty_result["digit_confidences"],
                "quantity_flags": qty_result["flag_reasons"],
                "return": return_result["value"],
                "return_confidences": return_result["digit_confidences"],
                "return_flags": return_result["flag_reasons"],
                # Quantity minus Return, per row -- the actual
                # line-item quantity (see CLAUDE.md's design notes).
                # Computed even when a field is flagged, since it's
                # still the best-effort reading; "flagged" below is
                # what tells a reviewer not to trust it as-is.
                "line_quantity": qty_result["value"] - return_result["value"],
                "flagged": qty_result["flagged"] or return_result["flagged"],
            }
        )

    result = {"invoice_flagged": False, "reason": None, "rows": rows_out}
    _save_results(result, output_dir)
    return result


def _save_results(result: dict, output_dir: str):
    """Write the extraction result out as results.json in output_dir."""
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image_paths",
        nargs="+",
        help="One or more invoice scan images. Passing several in one run "
        "is much faster than one run each: loading PyTorch and the model "
        "takes ~20s and is then paid only once.",
    )
    parser.add_argument(
        "--output-dir",
        default="extractions",
        help="Folder that each invoice's own <image basename>/ results folder goes in.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(CHECKPOINT_PATH, device)

    # Windows shells hand "invoices/*.png" over literally rather than
    # expanding it, so expand patterns here.
    image_paths = [
        match for pattern in args.image_paths for match in (sorted(glob.glob(pattern)) or [pattern])
    ]
    for image_path in image_paths:
        output_dir = os.path.join(
            args.output_dir, os.path.splitext(os.path.basename(image_path))[0]
        )
        print(f"\n=== {image_path} ===")
        _print_result(extract_invoice(image_path, output_dir, model, device), output_dir)


def _print_result(result: dict, output_dir: str):
    """Print one invoice's extraction as a table on the console."""
    if result["invoice_flagged"]:
        print(f"INVOICE FLAGGED: {result['reason']} -- needs manual handling, no rows extracted.")
        return

    header = f"{'row':>4} {'product':<45} {'qty':>5} {'ret':>4} {'line':>5}  flags"
    print(header)
    print("-" * len(header))
    for row in result["rows"]:
        flags = ", ".join(row["quantity_flags"] + row["return_flags"])
        marker = "  <-- REVIEW" if row["flagged"] else ""
        print(
            f"{row['row_index']:>4} {row['product_name'][:45]:<45} "
            f"{row['quantity']:>5} {row['return']:>4} {row['line_quantity']:>5}  {flags}{marker}"
        )

    flagged_count = sum(1 for row in result["rows"] if row["flagged"])
    print(f"\n{flagged_count}/{len(result['rows'])} rows flagged for review.")
    print(f"saved detailed results + crops to {output_dir}/")


if __name__ == "__main__":
    main()
