"""
Per-invoice extraction: given a NEW scanned invoice, applies the
one-time template calibration (template_calibration.json,
product_rows.json) to locate every row's Qty and Return cells, reads
each one with digit_reader, subtracts Return from Qty per row, and
writes out the resulting (product, quantity) pairs plus a per-field
confidence/flag status -- everything the (not yet built) human review
step will need to confirm or correct before anything reaches Odoo.

Run with:  python extract_invoice.py path/to/new_scan.png
"""

import argparse
import json
import os

import cv2
import numpy as np
import torch
from PIL import Image

from alignment import detect_border_corners, proportion_to_pixel, validate_aspect_ratio
from digit_reader import load_model, read_number

# 300 DPI full-page invoice scans routinely exceed Pillow's default
# decompression-bomb pixel-count guard even though they're legitimate,
# trusted, self-generated scans from this project's own pipeline --
# see the identical note in calibrate_template.py.
Image.MAX_IMAGE_PIXELS = None

CALIBRATION_PATH = "template_calibration.json"
PRODUCT_ROWS_PATH = "product_rows.json"
CHECKPOINT_PATH = "checkpoints/digit_cnn_finetuned.pt"

# Margins added around each calibrated cell box before cropping, as a
# fraction of the box's own height/width. Vertical margin is generous
# -- real invoice digits routinely sit a bit high or low relative to
# their ruled row -- while horizontal margin is kept small, since
# reaching sideways risks pulling in the NEIGHBORING Qty/Return
# column's own digit as stray ink, corrupting both fields at once,
# rather than just clipping a wide one in this field alone.
VERTICAL_MARGIN_FRACTION = 0.25
HORIZONTAL_MARGIN_FRACTION = 0.05


def load_calibration(calibration_path: str = CALIBRATION_PATH, product_rows_path: str = PRODUCT_ROWS_PATH):
    """
    Load the saved template calibration and its row->product name mapping.

    Returns:
        (calibration, product_names) -- calibration is the parsed
        template_calibration.json; product_names maps row_index to
        its product name string from product_rows.json.
    """
    with open(calibration_path) as f:
        calibration = json.load(f)
    with open(product_rows_path) as f:
        product_rows = json.load(f)
    product_names = {row["row_index"]: row["product_name"] for row in product_rows["rows"]}
    return calibration, product_names


def crop_cell(image_bgr: np.ndarray, box_proportions: list, corners: np.ndarray) -> np.ndarray:
    """
    Crop one calibrated cell (Qty or Return) out of a new scan, given
    its saved proportion box and this scan's own detected border.

    Maps all 4 corners of the box individually through
    proportion_to_pixel (not just the 2 diagonal corners that are
    stored) so the crop correctly follows this particular scan's own
    rotation, then takes the axis-aligned bounding box of those 4
    mapped points plus a margin -- simpler than cropping a rotated
    quadrilateral directly, and the small rotations these scans
    actually have don't lose meaningful cell area this way.

    Args:
        image_bgr: the full invoice scan.
        box_proportions: [fx1, fy1, fx2, fy2] as saved by
            calibrate_template.py (top-left and bottom-right corners,
            as proportions of the border).
        corners: this scan's own detected border corners, from
            alignment.detect_border_corners().

    Returns:
        The cropped cell region as a BGR image array.
    """
    fx1, fy1, fx2, fy2 = box_proportions
    corner_points = [
        proportion_to_pixel(fx1, fy1, corners),
        proportion_to_pixel(fx2, fy1, corners),
        proportion_to_pixel(fx2, fy2, corners),
        proportion_to_pixel(fx1, fy2, corners),
    ]
    xs = [p[0] for p in corner_points]
    ys = [p[1] for p in corner_points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    width, height = x2 - x1, y2 - y1
    x1 -= width * HORIZONTAL_MARGIN_FRACTION / 2
    x2 += width * HORIZONTAL_MARGIN_FRACTION / 2
    y1 -= height * VERTICAL_MARGIN_FRACTION / 2
    y2 += height * VERTICAL_MARGIN_FRACTION / 2

    # Clamp to the actual image bounds -- a margin near the table's
    # outer edge could otherwise push the crop request outside the
    # scan entirely.
    img_h, img_w = image_bgr.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(img_w, int(x2)), min(img_h, int(y2))
    return image_bgr[y1:y2, x1:x2]


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

    rows_out = []
    for row in calibration["rows"]:
        row_index = row["row_index"]
        product_name = product_names.get(row_index, "")

        qty_crop = crop_cell(image, row["quantity_box"], corners)
        return_crop = crop_cell(image, row["return_box"], corners)

        cv2.imwrite(os.path.join(crops_dir, f"row{row_index:02d}_qty.png"), qty_crop)
        cv2.imwrite(os.path.join(crops_dir, f"row{row_index:02d}_return.png"), return_crop)

        qty_result = read_number(qty_crop, model, device)
        return_result = read_number(return_crop, model, device)

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
    parser.add_argument("image_path", help="Path to a new invoice scan image.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write results.json and crops/ (default: extractions/<image basename>/).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(CHECKPOINT_PATH, device)

    output_dir = args.output_dir or os.path.join(
        "extractions", os.path.splitext(os.path.basename(args.image_path))[0]
    )

    result = extract_invoice(args.image_path, output_dir, model, device)

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
