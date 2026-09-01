"""
Reads a handwritten quantity out of one cell crop (a Qty or Return
field cropped from a new invoice scan): finds individual digit blobs
via connected-component analysis, classifies each one with the
fine-tuned DigitCNN, and concatenates them into a number -- flagging
the field for human review wherever segmentation or classification
looks untrustworthy, rather than guessing.

This is the automated equivalent of what label_tool.py's human
operator did by hand during training-data collection: crop out one
digit, preprocess it to match MNIST's format, hand it to the model.
The difference is finding WHERE each digit is in the first place,
since a multi-digit quantity (e.g. "30") isn't pre-split the way
training crops were -- that's what segment_digit_blobs() does.

Two independent safeguards decide whether a field gets flagged,
deliberately covering different failure modes:
  - blob count outside the plausible 1-3 digit range, or an
    unusually wide single blob (touching/merged digits look like
    ONE blob wider than a real digit ever is);
  - low softmax confidence on any individual digit, even when the
    blob count looked perfectly normal -- catches things that don't
    even resemble a digit (e.g. a crossed-out correction's scribble)
    regardless of how many blobs it happened to produce.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from data import MNIST_MEAN, MNIST_STD
from model import DigitCNN

# A quantity/return field on this project's fixed invoice template is
# realistically 1-3 digits (the largest quantities seen in the
# training data were in the hundreds, e.g. "300"). More than that is
# far more likely to be stray ink (a crossed-out correction, a
# smudge, scanner noise) than a real 4+ digit quantity.
MAX_EXPECTED_DIGITS = 3

# A single handwritten digit, even a wide one like "0" or "8", is
# essentially never wider than it is tall. A blob wider than its own
# height is a strong signal of two touching/crowded digits that
# connected-component analysis merged into one blob -- exactly the
# "fewer blobs than actual digits" failure mode, detected via this
# proxy rather than needing to know the true digit count in advance.
MAX_SINGLE_DIGIT_ASPECT_RATIO = 1.0

# Blobs smaller than this fraction of the cell crop's own area are
# treated as noise (dust, faint grid-line remnants, JPEG artifacts)
# rather than a real digit stroke, and dropped before blob-count
# checks even see them -- otherwise a single dust speck could by
# itself trip "too many blobs" on an otherwise perfectly clean field.
MIN_BLOB_AREA_FRACTION = 0.003


# A digit's own softmax confidence below this is treated as the model
# not clearly recognizing it as any digit -- flagged for human review
# even when blob count looked clean (see module docstring).
LOW_CONFIDENCE_THRESHOLD = 0.7


def load_model(checkpoint_path: str, device) -> DigitCNN:
    """
    Load the fine-tuned DigitCNN checkpoint for inference.

    Args:
        checkpoint_path: path to a state_dict saved by finetune.py
            (typically checkpoints/digit_cnn_finetuned.pt).
        device: torch device to load the model onto.

    Returns:
        The model in eval mode (dropout disabled), ready for inference.
    """
    model = DigitCNN().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def _threshold_cell(cell_bgr: np.ndarray) -> np.ndarray:
    """
    Convert a raw cell crop (dark ink on light paper, as scanned) into
    a binary mask with ink as bright foreground -- the same convention
    label_tool.py's saved training crops use (grayscale, inverted).

    Args:
        cell_bgr: a cropped Qty or Return cell, as a BGR image array.

    Returns:
        A binary (0/255) mask, same height/width as the input crop.
    """
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    # THRESH_BINARY_INV + Otsu: ink (dark) becomes white/foreground,
    # paper (light) becomes black/background. Otsu picks the cutoff
    # automatically per crop, coping with lighting/shadow varying scan
    # to scan without needing one fixed brightness threshold.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def segment_digit_blobs(cell_bgr: np.ndarray) -> list[np.ndarray]:
    """
    Find individual digit-shaped ink blobs within one cell crop,
    ordered left to right (reading order).

    Args:
        cell_bgr: a cropped Qty or Return cell, as a BGR image array.

    Returns:
        A list of binary (0/255) blob crops, one per connected
        component surviving the noise-area filter, ordered left to
        right by horizontal position. Empty list if the cell has no
        ink at all -- a genuinely blank field.
    """
    binary = _threshold_cell(cell_bgr)
    crop_height, crop_width = cell_bgr.shape[:2]
    min_area = MIN_BLOB_AREA_FRACTION * crop_height * crop_width

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    components = []
    for label in range(1, num_labels):  # label 0 is the background
        x, y, w, h, area = stats[label]
        if area < min_area:
            continue
        # extract_invoice.py's crop margin exists specifically to give
        # a digit's ink breathing room, so real digit strokes never
        # reach the crop's outer edge pixels -- but the printed ruled
        # cell border does, by construction (the margin is measured
        # outward FROM that border). All 4 sides of that border
        # connect into one ring at the corners, so its bounding box
        # spans the crop's full width AND height, unlike a real digit
        # -- checking "touches the border" catches this reliably
        # regardless of exactly how much of the ring survived
        # thresholding (a full ring, a partial one, or just a stray
        # line segment all touch an edge the same way).
        touches_border = x <= 0 or y <= 0 or (x + w) >= crop_width or (y + h) >= crop_height
        if touches_border:
            continue
        components.append((x, y, w, h, centroids[label][0]))

    # Sort left to right by centroid x -- this is what makes
    # concatenating classified digits into a number correct (a "3"
    # then a "0" read left to right is "30", not "03").
    components.sort(key=lambda c: c[4])

    return [binary[y:y + h, x:x + w] for x, y, w, h, _ in components]


def _preprocess_blob(blob: np.ndarray) -> torch.Tensor:
    """
    Convert one segmented digit blob into the 28x28, MNIST-normalized
    tensor DigitCNN expects -- mirroring label_tool.py's own
    preprocessing (pad to square BEFORE resizing, so a digit's
    proportions aren't distorted) so the model sees the same kind of
    input it was fine-tuned on, not something subtly different.

    Args:
        blob: a binary (0/255) crop containing one digit's ink,
            already light-on-dark (see segment_digit_blobs).

    Returns:
        A (1, 1, 28, 28) tensor ready to feed to DigitCNN.
    """
    h, w = blob.shape
    side = max(h, w)
    padded = np.zeros((side, side), dtype=np.uint8)
    top, left = (side - h) // 2, (side - w) // 2
    padded[top:top + h, left:left + w] = blob
    resized = cv2.resize(padded, (28, 28), interpolation=cv2.INTER_AREA)

    arr = resized.astype(np.float32) / 255.0
    arr = (arr - MNIST_MEAN) / MNIST_STD
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def read_number(cell_bgr: np.ndarray, model: DigitCNN, device) -> dict:
    """
    Read a full quantity/return number out of one cell crop: segment
    it into digit blobs, classify each one, and concatenate left to
    right -- flagging the field for human review wherever segmentation
    or classification looks untrustworthy, rather than guessing.

    A flagged field still gets a best-effort "value" (every detected
    blob is classified and concatenated regardless of the flag) --
    the flag means "don't trust this without a human looking at the
    crop," not "no information was extracted." See CLAUDE.md's
    extraction pipeline notes for why that distinction matters.

    Args:
        cell_bgr: a cropped Qty or Return cell, as a BGR image array.
        model: the fine-tuned DigitCNN, in eval mode.
        device: torch device to run inference on.

    Returns:
        A dict:
          - "value": int. 0 for a genuinely blank cell (per this
            project's convention -- blank means 0, not a missing
            value). A best-effort reading otherwise, trustworthy
            exactly when "flagged" is False.
          - "digit_confidences": list of per-digit softmax
            confidences, in reading order.
          - "flagged": bool -- True if this field needs human review.
          - "flag_reasons": list of short strings explaining why, e.g.
            "too_many_blobs", "possible_merged_digits", "low_confidence".
    """
    blobs = segment_digit_blobs(cell_bgr)

    if not blobs:
        # A blank cell is a normal, valid reading (see CLAUDE.md:
        # "treat blank as 0, not as a missing/error value") -- not
        # flagged.
        return {"value": 0, "digit_confidences": [], "flagged": False, "flag_reasons": []}

    flag_reasons = []
    if len(blobs) > MAX_EXPECTED_DIGITS:
        flag_reasons.append("too_many_blobs")
    if any(blob.shape[1] / blob.shape[0] > MAX_SINGLE_DIGIT_ASPECT_RATIO for blob in blobs):
        flag_reasons.append("possible_merged_digits")

    digits = []
    confidences = []
    with torch.no_grad():
        for blob in blobs:
            tensor = _preprocess_blob(blob).to(device)
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)
            confidence, predicted = probs.max(dim=1)
            digits.append(str(predicted.item()))
            confidences.append(confidence.item())

    if any(c < LOW_CONFIDENCE_THRESHOLD for c in confidences):
        flag_reasons.append("low_confidence")

    return {
        "value": int("".join(digits)),
        "digit_confidences": confidences,
        "flagged": len(flag_reasons) > 0,
        "flag_reasons": flag_reasons,
    }
