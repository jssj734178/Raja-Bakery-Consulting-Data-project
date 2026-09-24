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

Segmentation has to survive three things that all hit at once on a
real scan of this ruled invoice form, and which between them caused
the long-standing "blank cell reads as 2 / cell containing 50 reads
as blank" bug (see CLAUDE.md for the full history):

  - the printed ruled cell border is inside the crop too, and the
    handwriting frequently CROSSES it, so plain connected-component
    analysis merges digit and border into one component -- deleting
    the border then deletes the digits along with it;
  - a digit's own strokes are often disconnected (a "4" drawn as two
    separate diagonals, a "5" whose flag doesn't touch its body),
    so one digit can arrive as several components;
  - the vertical crop margin deliberately overlaps the neighbouring
    rows, so ink belonging to the row above or below shows up here.

The order of operations below addresses each in turn: deskewed crops
(the caller's job -- see extract_invoice.crop_cell) let the printed
lines be isolated by LENGTH and subtracted, a reconnect step repairs
strokes severed by that subtraction, x-overlapping fragments merge
back into single digits, and an ink-ownership test decides which
blobs actually belong to this cell rather than a neighbouring one.

Independent safeguards then decide whether a field gets flagged,
deliberately covering different failure modes:
  - blob count outside the plausible 1-3 digit range, or an
    unusually wide single blob (touching/merged digits look like
    ONE blob wider than a real digit ever is);
  - ink too small to read as a digit but too big to dismiss, or a
    blob much smaller than the cell's tallest one -- both being what
    a digit fragmented into disconnected strokes looks like when the
    merge step didn't manage to reunite it;
  - low softmax confidence on any individual digit, even when the
    blob count and shapes looked perfectly normal -- catches things
    that don't even resemble a digit (e.g. a crossed-out
    correction's scribble) regardless of how they segmented.
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

# A blob wider than this multiple of its own height is treated as two
# touching/crowded digits that connected-component analysis merged
# into one -- the "fewer blobs than actual digits" failure mode,
# detected via this proxy rather than needing to know the true digit
# count in advance.
#
# Measured across 333 real blobs from a 24-page sample: the median
# blob is 0.59 wide-to-tall and the 90th percentile is 1.07, but the
# blobs above 1.0 are overwhelmingly ones the model then classifies
# with high confidence -- i.e. genuinely single, merely wide, digits.
# Flagging at 1.0 put ~35% of all non-blank cells into the review
# queue, mostly correct reads; 1.5 sits above the wide-single-digit
# population while still catching the side-by-side pairs this is for.
MAX_SINGLE_DIGIT_ASPECT_RATIO = 1.5

# How much of the cell a blob has to cover to be read as a digit.
# Everything here is measured against the CELL box rather than the
# crop, so no threshold moves when the caller changes how much margin
# it crops with.
#
# Three bands, because "too small to be a digit" and "not worth
# mentioning" are different things:
#   - below FRAGMENT_AREA_FRACTION is dust, a JPEG artifact or a
#     scrap of a printed line, ignored in silence; letting these
#     through means a single speck can trip "too many blobs" on an
#     otherwise clean field;
#   - between the two is a FRAGMENT: too small to read as a digit,
#     too big to have come from nothing. It is left out of the value
#     and raises a review flag instead -- that band is mostly the
#     loose strokes of a digit drawn in pieces, so silently dropping
#     it turns a "54" into a confident "51";
#   - at or above MIN_BLOB_AREA_FRACTION it is read as a digit.
#
# Measured over 541 blobs from a 12-page sample: below 0.006 the
# median blob is under half the cell's height, i.e. fragment-shaped,
# while real digits measure 0.02-0.05. Raising the cut to 0.010
# starts deleting real digits, and dropping it to the 0.003 it began
# at let line scraps into the values ("50" read as "501", "14" as
# "141").
FRAGMENT_AREA_FRACTION = 0.003
MIN_BLOB_AREA_FRACTION = 0.006

# ...except that area alone under-serves a narrow digit: a "1" is a
# single stroke and can cover less of the cell than a fragment of a
# "4" does, so it needs a second way to qualify. Any blob at least
# half the cell's height is full-height handwriting whatever its
# area, which a severed stroke or a line scrap never is. Judging on
# height ALONE is worse than area alone -- it deletes short digits
# like a flat "0" -- so the two run as alternatives.
MIN_BLOB_HEIGHT_FRACTION = 0.5

# ...and a mark that is the ONLY one in its box has to be at least this
# tall relative to the cell, whatever its area. A digit written on its
# own is full-height handwriting; what sits alone and short in an
# otherwise empty box is a speck of paper texture, a stroke from the
# next row, or a smudge. The area test above can't tell those apart
# once the box has been snapped tight to its printed lines, because a
# smaller box makes the same speck a larger share of it -- which is how
# empty Return boxes started reading "3" or "2". Checked by eye against
# every lone mark under 0.6 of the cell's height across all 96 sample
# pages: nearly all under 0.5 were junk, and the few real digits among
# them were multi-digit numbers that had already broken into pieces.
# The half-height trailing zeros described in CLAUDE.md are never
# alone -- they always sit beside a full-height digit -- so this
# doesn't affect them.
MIN_LONE_DIGIT_HEIGHT_FRACTION = 0.5

# A fragment only counts as evidence of a split digit if it is at
# least this tall relative to the cell -- tall enough to be a pen
# stroke rather than a scrap. Without this the flag was close to
# useless: most cells carry some small leftover in the fragment band,
# and flagging on any of them raised "possible_split_digit" on 110
# fields in a 24-page sample against 23 once stroke height is
# required, burying the handful that are actually a broken digit.
MIN_FRAGMENT_HEIGHT_FRACTION = 0.3

# How large a neighbourhood the local threshold estimates the paper's
# brightness over, as a fraction of the cell's height, and how much
# darker than that neighbourhood a pixel must be to count as ink.
# See threshold_cell() for why this is local rather than global.
ADAPTIVE_BLOCK_CELL_FRACTION = 0.6
ADAPTIVE_THRESHOLD_OFFSET = 10

# A digit's own softmax confidence below this is treated as the model
# not clearly recognizing it as any digit -- flagged for human review
# even when segmentation looked clean (see module docstring).
LOW_CONFIDENCE_THRESHOLD = 0.7

# Printed ruled lines are isolated by LENGTH: a morphological opening
# with a kernel this long (as a fraction of the cell's own width or
# height) survives only on strokes that run that far in a straight
# line, which the form's ruled lines do and handwriting does not.
#
# This only works because the caller hands us a DESKEWED crop. These
# scans sit about 1.4 degrees off square, which over a ~1100px-wide
# cell drops a "horizontal" ruled line by ~26px -- far more than its
# own ~7px thickness, so a flat 1px-tall kernel cannot fit inside it
# and the line survives in slanted fragments instead of being
# isolated. Deskewing first is what makes this step work at all.
LINE_KERNEL_CELL_FRACTION = 0.5

# The isolated line mask is dilated by this much before subtraction,
# so the anti-aliased soft edges of a printed line go with it rather
# than surviving as a thin ghost that reads as its own blob.
LINE_DILATION = 5

# Subtracting a horizontal ruled line severs any digit stroke that
# crossed it, splitting one digit into stacked pieces (a "5" becomes
# flag + body + tail). A closing with a TALL, 1px-WIDE kernel bridges
# those vertical gaps back together; being 1px wide, it cannot smear
# two side-by-side digits into each other the way a square kernel
# would. Expressed as a fraction of cell height so it tracks scan
# resolution instead of being pinned to 300 DPI.
#
# It has to clear the whole gap the subtraction opened up, which is
# the printed line's own thickness (~14-18px on these scans) plus
# LINE_DILATION on each side -- call it 30px against a ~270px cell.
# Sizing it any tighter is a real failure and not a graceful one:
# at 0.07 the flag of a "5" stayed a separate blob, fell outside the
# cell box on its own, was dropped as another row's ink, and the
# remaining body then read as a different digit entirely.
STROKE_RECONNECT_HEIGHT_FRACTION = 0.11

# Two blobs whose x-ranges overlap by more than this fraction of the
# narrower one are treated as strokes of a single digit stacked
# vertically (a "5" whose flag floats free of its body) and merged.
# Genuinely adjacent digits barely overlap in x -- a real "50"
# measures around 0.07 here -- so this separates the two cases with
# a lot of room to spare.
STROKE_X_OVERLAP_FRACTION = 0.5

# A blob both shorter and narrower than this fraction of the cell's
# tallest blob is a leftover fragment of a digit drawn in
# disconnected strokes that the x-overlap merge above didn't reunite
# (a "4" drawn as two separate diagonals is the case that motivated
# this). Reuniting those automatically is not safe -- the threshold
# that fixes the "4" is close enough to weld a legitimate "50" into
# one blob -- so this only raises a review flag.
SPLIT_FRAGMENT_FRACTION = 0.5

# A blob is claimed by this cell only if at least this fraction of
# its ink lies inside the cell's own box. The crop is deliberately
# taken with a generous margin, so ink from the row above or below
# is present in it; since the calibrated cell boxes tile the table
# without overlapping, "most of my ink is in this box" assigns each
# blob to exactly one cell.
#
# This test replaces an earlier "discard anything touching the crop
# edge" rule, which was correct in spirit but fired on real digits
# too -- ink that straddles a ruled line legitimately reaches the
# edge of a neighbouring cell's crop. Majority ownership makes the
# same distinction without that collateral damage, but it only works
# on an UNCLIPPED blob, which is why the caller must crop with
# enough margin to contain a neighbouring row's digits whole.
MIN_OWNED_INK_FRACTION = 0.5


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


def threshold_cell(cell_bgr: np.ndarray, cell_height: float) -> np.ndarray:
    """
    Convert a raw cell crop (dark ink on light paper, as scanned) into
    a binary mask with ink as bright foreground -- the same convention
    label_tool.py's saved training crops use (grayscale, inverted).

    Args:
        cell_bgr: a cropped Qty or Return cell, as a BGR image array.
        cell_height: height in pixels of the cell's own box, which
            sets how large a neighbourhood the local background is
            estimated over.

    Returns:
        A binary (0/255) mask, same height/width as the input crop.
    """
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    # Estimate the paper's brightness locally and call anything enough
    # darker than its own surroundings ink, rather than picking one
    # cutoff for the whole crop.
    #
    # This started as an Otsu threshold, which is the obvious choice
    # and was quietly costing more than everything else here put
    # together. Otsu wants a two-humped histogram; a cell crop is
    # ~95% blank paper with a few faint pencil strokes on it, so the
    # cutoff lands wherever the paper's own noise happens to sit.
    # On darker scans that merely coarsened the strokes, but on a
    # lightly-pencilled invoice it broke digits into pieces: trailing
    # zeros vanished, "30" read as "3", "40" as "4", "20" as "2",
    # across nearly a whole page. Switching to a local threshold fixed
    # that page almost entirely AND fixed errors on the darker pages
    # that had been blamed on segmentation and on the model -- a "20"
    # misread as "22", and the "54" whose split "4" no other change
    # here managed to reunite.
    #
    # The neighbourhood spans a bit over half a cell, big enough that
    # it is mostly paper even when centred on a stroke, and it scales
    # with the cell so it doesn't assume 300 DPI. OpenCV requires it
    # odd.
    block_size = max(3, int(cell_height * ADAPTIVE_BLOCK_CELL_FRACTION) | 1)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        ADAPTIVE_THRESHOLD_OFFSET,
    )


def _remove_printed_lines(binary: np.ndarray, cell_width: float, cell_height: float) -> np.ndarray:
    """
    Strip the form's printed ruled lines out of a thresholded cell,
    leaving only handwritten ink, then repair the digit strokes that
    subtraction severed.

    Assumes the crop has been DESKEWED by the caller -- the ruled
    lines must be axis-aligned for a length-based opening to isolate
    them (see LINE_KERNEL_CELL_FRACTION for why that matters).

    Args:
        binary: the thresholded cell mask from threshold_cell().
        cell_width: width in pixels of the cell's own box, excluding
            the crop margin -- the kernel lengths are measured
            against the cell rather than the crop so that changing
            the margin doesn't change what counts as "a long line".
        cell_height: height in pixels of the cell's own box.

    Returns:
        A binary (0/255) mask of handwritten ink only.
    """
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(5, int(cell_width * LINE_KERNEL_CELL_FRACTION)), 1)
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(5, int(cell_height * LINE_KERNEL_CELL_FRACTION)))
    )

    # An opening keeps only what the kernel fits entirely inside, so
    # each of these survives on the form's ruled lines and nowhere in
    # the handwriting.
    lines = cv2.bitwise_or(
        cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel),
        cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel),
    )
    lines = cv2.dilate(
        lines, cv2.getStructuringElement(cv2.MORPH_RECT, (LINE_DILATION, LINE_DILATION))
    )

    ink = cv2.bitwise_and(binary, cv2.bitwise_not(lines))

    return cv2.morphologyEx(
        ink,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, _severed_gap(cell_height))),
    )


def _severed_gap(cell_height: float) -> int:
    """
    How wide a gap subtracting one printed line leaves behind, in
    pixels -- the line's own thickness plus the dilation applied to
    it, which is the same in both directions.

    Args:
        cell_height: height in pixels of the cell's own box.

    Returns:
        The gap width, never smaller than 3px.
    """
    return max(3, int(cell_height * STROKE_RECONNECT_HEIGHT_FRACTION))


def _owned_ink(ink: np.ndarray, cell_box: tuple) -> np.ndarray:
    """
    Drop the ink in a cleaned crop that belongs to a neighbouring
    cell rather than this one.

    Ownership is decided per GLYPH, not per connected component, and
    the difference matters. Removing the vertical column divider cuts
    any glyph sitting across it in two, and the Return column's
    calibrated box ends right on that divider with the next column's
    printed "$" immediately beyond it -- so the "$" loses its left
    sliver into this cell, where a per-component test can only
    conclude the sliver belongs here, and it reads as a "1". Across a
    24-page sample that alone put ink in 115 Return cells against 56
    Quantity cells, on a form where returns are far rarer than
    quantities.

    So the pieces are provisionally grouped back together across the
    gaps line removal opened, in both directions, and each GROUP is
    judged as a whole. Grouping deliberately does not change the
    blobs that come out: closing horizontally by enough to rejoin a
    severed glyph also welds genuinely adjacent digits together, and
    a "50" merged into one blob reads as neither. Using the closed
    mask only to answer "whose ink is this?" gets the sliver rejected
    with the rest of its "$" while the digits stay separate.

    Args:
        ink: the handwriting-only mask from _remove_printed_lines().
        cell_box: (x1, y1, x2, y2) of the cell's own box, in the
            crop's pixel coordinates.

    Returns:
        A copy of ink with every disowned group cleared.
    """
    x1, y1, x2, y2 = cell_box
    gap = _severed_gap(y2 - y1)
    grouped = cv2.morphologyEx(
        ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (gap, 1))
    )

    num_groups, group_labels = cv2.connectedComponents(grouped, connectivity=8)

    # Count every group's ink, and how much of it lies inside the box,
    # in one pass over the crop. Looping over groups and re-scanning
    # the whole crop for each one cost ~7s per page.
    height, width = ink.shape
    rows_inside = (np.arange(height) >= y1) & (np.arange(height) <= y2)
    cols_inside = (np.arange(width) >= x1) & (np.arange(width) <= x2)
    inside_box = rows_inside[:, None] & cols_inside[None, :]
    total = np.bincount(group_labels.ravel(), minlength=num_groups)
    inside = np.bincount(group_labels[inside_box], minlength=num_groups)

    keep = np.zeros(num_groups, dtype=bool)
    keep[1:] = inside[1:] / total[1:] >= MIN_OWNED_INK_FRACTION
    owned = keep[group_labels]

    return np.where(owned, ink, 0).astype(np.uint8)


def _merge_stroke_fragments(blobs: list[dict]) -> list[dict]:
    """
    Merge blobs that are really separate strokes of one digit.

    A digit drawn without lifting the pen arrives as a single
    component, but plenty don't -- and subtracting a ruled line that
    crossed a digit splits it further. Vertically stacked pieces of
    one digit sit in the same horizontal band, whereas two adjacent
    digits sit side by side, so horizontal overlap separates the two
    cases cleanly (see STROKE_X_OVERLAP_FRACTION).

    Args:
        blobs: dicts with "x1"/"y1"/"x2"/"y2" bounds and a boolean
            "mask" over the full crop.

    Returns:
        The same list with overlapping entries combined. Runs to a
        fixed point, so a digit broken into three or more pieces
        merges in successive passes rather than only pairwise.
    """
    blobs = list(blobs)
    merged_any = True
    while merged_any:
        merged_any = False
        for i in range(len(blobs)):
            for j in range(i + 1, len(blobs)):
                first, second = blobs[i], blobs[j]
                overlap = min(first["x2"], second["x2"]) - max(first["x1"], second["x1"])
                narrower = min(first["x2"] - first["x1"], second["x2"] - second["x1"])
                if overlap > STROKE_X_OVERLAP_FRACTION * narrower:
                    blobs[i] = {
                        "x1": min(first["x1"], second["x1"]),
                        "y1": min(first["y1"], second["y1"]),
                        "x2": max(first["x2"], second["x2"]),
                        "y2": max(first["y2"], second["y2"]),
                        "area": first["area"] + second["area"],
                        "mask": first["mask"] | second["mask"],
                    }
                    blobs.pop(j)
                    merged_any = True
                    break
            if merged_any:
                break
    return blobs


def segment_digit_blobs(cell_bgr: np.ndarray, cell_box: tuple) -> list[np.ndarray]:
    """
    Find individual digit-shaped ink blobs belonging to one cell,
    ordered left to right (reading order).

    Args:
        cell_bgr: a DESKEWED crop around one Qty or Return cell, as a
            BGR image array. It must include enough margin around the
            cell that ink from the neighbouring rows appears whole
            rather than clipped -- see MIN_OWNED_INK_FRACTION.
        cell_box: (x1, y1, x2, y2) of the cell's own calibrated box
            in this crop's pixel coordinates, i.e. the crop minus its
            margin. Everything that distinguishes "this cell's ink"
            from "the neighbouring row's ink" is measured against it.

    Returns:
        (blobs, fragment_count) -- blobs is a list of binary (0/255)
        crops, one per digit found, ordered left to right by
        horizontal position, empty if the cell has no ink of its own
        (a genuinely blank field); fragment_count is how many pieces
        of ink were too small to read as digits but too large to
        dismiss, which the caller turns into a review flag rather
        than a value (see FRAGMENT_AREA_FRACTION).
    """
    cell_width = cell_box[2] - cell_box[0]
    cell_height = cell_box[3] - cell_box[1]

    ink = _remove_printed_lines(
        threshold_cell(cell_bgr, cell_height), cell_width, cell_height
    )
    # Ink from the row above or below reaches into this crop's margin
    # but leaves the bulk of itself outside the cell box, so this
    # drops it -- without the collateral damage of the old "discard
    # anything touching the crop edge" rule.
    ink = _owned_ink(ink, cell_box)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    cell_area = cell_width * cell_height

    blobs = []
    for label in range(1, num_labels):  # label 0 is the background
        bx, by, bw, bh, area = stats[label]
        if area < FRAGMENT_AREA_FRACTION * cell_area:
            continue
        blobs.append(
            {"x1": bx, "y1": by, "x2": bx + bw, "y2": by + bh,
             "area": area, "mask": labels == label}
        )

    # Merge first, size-test second: a digit split into strokes has to
    # be put back together before asking whether it is big enough to
    # be a digit, or every piece fails the test on its own.
    blobs = _merge_stroke_fragments(blobs)

    def is_digit(blob):
        return (
            blob["area"] >= MIN_BLOB_AREA_FRACTION * cell_area
            or (blob["y2"] - blob["y1"]) >= MIN_BLOB_HEIGHT_FRACTION * cell_height
        )

    def is_stroke_sized(blob):
        return (blob["y2"] - blob["y1"]) >= MIN_FRAGMENT_HEIGHT_FRACTION * cell_height

    fragment_count = sum(1 for b in blobs if not is_digit(b) and is_stroke_sized(b))
    blobs = [b for b in blobs if is_digit(b)]

    # A short mark with nothing beside it isn't read as a digit -- see
    # MIN_LONE_DIGIT_HEIGHT_FRACTION.
    if len(blobs) == 1 and (blobs[0]["y2"] - blobs[0]["y1"]) < MIN_LONE_DIGIT_HEIGHT_FRACTION * cell_height:
        fragment_count += is_stroke_sized(blobs[0])
        blobs = []

    # Sort left to right by box centre -- this is what makes
    # concatenating classified digits into a number correct (a "3"
    # then a "0" read left to right is "30", not "03").
    blobs.sort(key=lambda b: (b["x1"] + b["x2"]) / 2)

    return [
        (b["mask"][b["y1"]:b["y2"], b["x1"]:b["x2"]]).astype(np.uint8) * 255 for b in blobs
    ], fragment_count


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


def read_number(cell_bgr: np.ndarray, cell_box: tuple, model: DigitCNN, device) -> dict:
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
        cell_bgr: a deskewed, generously-margined crop around one Qty
            or Return cell (see segment_digit_blobs).
        cell_box: (x1, y1, x2, y2) of the cell's own box within that
            crop, in crop pixel coordinates.
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
            "too_many_blobs", "possible_merged_digits",
            "possible_split_digit", "unreadable_ink",
            "low_confidence".
    """
    return classify_blobs(*segment_digit_blobs(cell_bgr, cell_box), model, device)


def classify_blobs(blobs: list, fragment_count: int, model: DigitCNN, device) -> dict:
    """
    The second half of read_number(): read already-segmented blobs
    with the model and decide the flags.

    Split out so a caller can run the slow image work
    (segment_digit_blobs) for many cells in parallel threads, then
    pass each result through here one at a time.

    Args:
        blobs, fragment_count: exactly what segment_digit_blobs()
            returned for one cell.
        model: the fine-tuned DigitCNN, in eval mode.
        device: torch device to run inference on.

    Returns:
        The same dict as read_number().
    """
    if not blobs:
        # A blank cell is a normal, valid reading (see CLAUDE.md:
        # "treat blank as 0, not as a missing/error value") -- not
        # flagged, unless there was ink here that was simply too small
        # to read, in which case "blank" is a conclusion worth having
        # a human confirm.
        reasons = ["unreadable_ink"] if fragment_count else []
        return {
            "value": 0,
            "digit_confidences": [],
            "flagged": bool(reasons),
            "flag_reasons": reasons,
        }

    flag_reasons = []
    if len(blobs) > MAX_EXPECTED_DIGITS:
        flag_reasons.append("too_many_blobs")
    if any(blob.shape[1] / blob.shape[0] > MAX_SINGLE_DIGIT_ASPECT_RATIO for blob in blobs):
        flag_reasons.append("possible_merged_digits")

    # Two views of the same failure -- a digit drawn in strokes that
    # didn't get put back together. Either the leftover was too small
    # to read and got left out of the value entirely, or it was big
    # enough to be read as a digit of its own while being markedly
    # smaller in BOTH dimensions than the cell's tallest blob, which
    # no real neighbouring digit is.
    tallest = max(blob.shape[0] for blob in blobs)
    if fragment_count or any(
        blob.shape[0] < SPLIT_FRAGMENT_FRACTION * tallest
        and blob.shape[1] < SPLIT_FRAGMENT_FRACTION * tallest
        for blob in blobs
    ):
        flag_reasons.append("possible_split_digit")

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
