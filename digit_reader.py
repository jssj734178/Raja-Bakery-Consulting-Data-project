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

# A Total Price cell (e.g. "27.00", "130.0") additionally has a
# handwritten decimal point, which nothing else on this form does --
# Qty and Return are always whole numbers. Distinguishing "this small
# mark is the decimal point" from "this small mark is a leftover
# fragment of a digit written in disconnected strokes" (the case
# FRAGMENT_AREA_FRACTION/MIN_FRAGMENT_HEIGHT_FRACTION exist for) comes
# down to height: a decimal point is a short dot that never reaches
# anywhere near a digit's height, whereas even a small stroke fragment
# of a broken-up digit tends to be tall relative to the cell (see
# MIN_FRAGMENT_HEIGHT_FRACTION's own 0.3 cutoff). A mark shorter than
# that, and small in area, is read as the decimal point instead of a
# fragment.
#
# Measured against a real decimal point on an actual scan (a clean
# "29.20" total, PH 416-727-0623 invoice, page 2, row 10): the point
# itself measured 0.0025 of the cell's area and 0.18 of its height.
# These thresholds sit comfortably above that measurement (room for a
# slightly larger dot elsewhere) while staying well under
# MIN_BLOB_AREA_FRACTION/MIN_BLOB_HEIGHT_FRACTION, so a real digit is
# never mistaken for the point.
#
# These two were re-measured (2026-09-27) across every filled-in Total
# Price cell in all 96 real scans in invoices/, specifically to find out
# why 99% of them were getting flagged. The answer turned out not to be
# "these two numbers are wrong": a genuine decimal point and an ordinary
# fleck of paper grain measure THE SAME on this form -- both are just
# small, faint marks, and there is no area or height cutoff that tells
# one from the other. (Measured directly: across 1,214 Total Price cells
# with no handwriting in them at all, paper grain alone still produces
# small surviving specks in 95% of them, at almost exactly the same size
# as the one real decimal point measured here originally.) Narrowing
# these further would have started discarding real decimal points right
# alongside the noise, not separating the two.
#
# What DOES separate them is not size but WHERE they sit: a real decimal
# point has to fall somewhere inside the number it belongs to (between
# its leftmost and rightmost digit), because it's part of the same
# continuous handwriting, while a fleck of paper grain lands anywhere in
# the cell with no such preference. That positional test -- not these
# two constants -- is what actually resolves which candidate is the
# point; see the "positioned between the digits" logic in
# segment_price_blobs(). These two constants are kept only as a coarse
# first filter (too tall or too big to plausibly be a dot at all, e.g. a
# genuine digit fragment), not as the final decision.
DECIMAL_POINT_MAX_HEIGHT_FRACTION = 0.25
DECIMAL_POINT_MAX_AREA_FRACTION = 0.005

# How much wider than tall a mark can be and still plausibly be a dot or
# short dash rather than a sliver of leftover printed-line residue (see
# is_decimal_point in segment_price_blobs, and LINE_RESIDUE_MAX_ASPECT_RATIO
# below for the equivalent guard on digits). The widest real decimal
# point seen so far, a dash rather than a round dot, measured 2.6.
DECIMAL_POINT_MAX_ASPECT_RATIO = 3

# How far past the outermost digit a candidate can still sit and count
# as plausibly belonging to that number, as a fraction of the cell's own
# width -- covers a decimal point written after the last digit with no
# cents digits following it at all (a plain "$130 even", written as
# "130." with nothing after the dot). Measured on a real such example:
# the point sat 0.153 of the cell's width past its last digit. Set well
# above that (with room to spare) rather than exactly at it, since
# UNDER-shooting this turns a real trailing point into a no_decimal_point
# flag, which is a smaller loss than the alternative of setting it so
# wide that unrelated noise elsewhere in the cell starts qualifying.
DECIMAL_POINT_SPAN_MARGIN_FRACTION = 0.25

# When more than one small mark sits between the number's own digits --
# a real decimal point plus a stray fleck that also happens to land in
# that span -- the one actually closest to a digit is almost always the
# real point (measured: a real point sits a median 0.054 of the cell's
# width from its nearest digit, versus 0.114 when a noise speck is mixed
# in with it). This is only treated as a confident pick when the next-
# closest contender is at least this much farther away; when two
# candidates are nearly equally close, that's a genuine tie -- flagged
# as ambiguous_decimal_point rather than guessed.
CONTENDER_GAP_MARGIN_FRACTION = 0.05

# CONTENDER_GAP_MARGIN_FRACTION above only measures HORIZONTAL distance
# to the nearest digit, which means a mark can be called "close" purely
# by sharing a digit's x-range even while sitting far above or below it
# -- e.g. bleed-through ink from the row above. Measured across all 96
# real scans' ambiguous_decimal_point ties: in 48 of 134 (36%), the
# closer-by-horizontal-gap candidate is also genuinely close to a digit
# by true 2D distance (within this fraction of the cell's own height),
# while the tied runner-up is not close by ANY reasonable margin (median
# 0.162 of cell height away, nowhere near this threshold even when it's
# loosened well past it) -- confirmed by eye on real crops, e.g. a
# handwritten "48.00" whose real point sits flush against the "8" while
# the tied "contender" turns out to be a stray fleck of paper grain well
# outside the row entirely. Those 48 are a confident pick, not a genuine
# tie. Set well below where the tied runner-up cases start showing up
# (both candidates genuinely 2D-close only starts at 13 of 134 even at
# this same threshold) so a real second candidate still gets flagged.
DECIMAL_POINT_TOUCH_DISTANCE_FRACTION = 0.04

# The dust-vs-ink floor segment_digit_blobs() uses (FRAGMENT_AREA_FRACTION)
# was tuned for Qty/Return cells, where the smallest thing worth keeping
# is a fragment of a digit. A handwritten decimal point is smaller than
# that -- the same real "29.20" example measured above (0.0025 of the
# cell) would be discarded as dust by FRAGMENT_AREA_FRACTION (0.003) and
# never even reach decimal-point classification. segment_price_blobs()
# uses this lower floor instead, so a genuine decimal point survives.
#
# This floor does still let a fair amount of plain paper grain through
# too (measured: cells with no handwriting in them at all still produce
# small surviving specks most of the time) -- raising it further doesn't
# fix that, because real decimal points measure the same size as that
# grain (see DECIMAL_POINT_MAX_HEIGHT_FRACTION above). Telling a real
# point apart from grain that made it past this floor is handled instead
# by where it sits relative to the digits, not by raising this number.
PRICE_DUST_AREA_FRACTION = 0.0005

# Total Price runs higher than a plain Qty/Return quantity ever does (a
# bulk order's line total can run into the hundreds of dollars, e.g.
# 190 x $2.60 = $494.00) and always carries two more digits after the
# decimal point, so more raw digit characters are expected here than
# MAX_EXPECTED_DIGITS allows for a plain quantity field.
MAX_EXPECTED_PRICE_DIGITS = 6

# The Total Price equivalent of MAX_SINGLE_DIGIT_ASPECT_RATIO (used for
# the possible_merged_digits flag) -- kept as its own, much higher
# constant because Qty/Return's value (1.5) turned out not to transfer.
# This handwriting's cursive ending on a whole-dollar amount (a "00"
# cents suffix drawn as one connected loop rather than two separate
# zeros) is naturally much wider than any Qty/Return digit ever is.
# Measured directly: comparing every possible_merged_digits-flagged
# field's widest digit against whether that field's unit price matched
# its own product's usual price elsewhere in the same 96-scan corpus (a
# proxy for "this read is probably actually correct"), the aspect-ratio
# distributions of the "probably correct" and "probably not" groups were
# nearly identical up to about 3, and only started separating past that.
# At this threshold, exactly 0 of 129 "probably correct" reads still
# trip the flag, while 97% of the "probably not" group no longer does
# either -- the ones that still do are the genuinely extreme outliers
# (up to 11.8), not ordinary correct reads.
MAX_SINGLE_PRICE_DIGIT_ASPECT_RATIO = 5

# How much wider than tall a blob can be, as a straight ratio, and still
# plausibly be a single digit -- used only to decide whether something
# qualifies as a digit AT ALL (see is_digit in segment_price_blobs), not
# how confident to be about it once it does (that's the separate
# MAX_SINGLE_PRICE_DIGIT_ASPECT_RATIO just above, used for the
# possible_merged_digits flag). Set far above what any real digit reaches
# on purpose: measured
# across every blob classified as a digit in all 96 real scans, one that
# only qualifies by area (rather than by being tall enough on its own)
# has an aspect ratio under 3 in 90% of cases even at its most extreme,
# while a leftover fragment of the row's own printed line -- long and
# thin, the actual reason this constant exists -- routinely measures
# 8 to 70+. Anything past this is treated as line residue, not a digit,
# regardless of how much raw ink area it has.
LINE_RESIDUE_MAX_ASPECT_RATIO = 5

# The two pieces of a digit genuinely split by an erased printed line
# sit close together vertically -- a small multiple of that line's own
# removed thickness (see _severed_gap, ~0.11 of the cell's height).
# This caps how far apart, vertically, two blobs can be and still be
# merged as if they were split pieces of the same mark -- set with
# comfortable headroom above a genuine repair's own gap, while staying
# well under the one confirmed bad merge that motivated this: a decimal
# point and an unrelated leftover fragment of the row's own printed
# line, sitting over 100px apart vertically in a ~240px-tall cell
# (0.44 of the cell's height) purely because they happened to overlap
# sideways. Total Price's much wider cells give that kind of coincidence
# more room to happen than Qty/Return's narrower ones do.
STROKE_MAX_VERTICAL_GAP_FRACTION = 0.25

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


def _merge_stroke_fragments(blobs: list[dict], max_vertical_gap: float = float("inf")) -> list[dict]:
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
        max_vertical_gap: the two pieces of a digit genuinely split by
            an erased line sit close together vertically -- about as
            far apart as that line's own removed thickness (see
            _severed_gap). Two blobs further apart than this, even if
            they happen to overlap in x, are not a split digit; they're
            unrelated marks that coincidentally line up sideways (found
            on a real Total Price cell: a decimal point and a leftover
            fragment of the row's own printed bottom line, over 100px
            apart vertically in a ~240px cell, merging into one blob
            that read as neither). Left unbounded by default so
            segment_digit_blobs's existing, already-verified behaviour
            for Qty/Return is untouched unless a caller opts in.
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
                vertical_gap = max(0, max(first["y1"], second["y1"]) - min(first["y2"], second["y2"]))
                if overlap > STROKE_X_OVERLAP_FRACTION * narrower and vertical_gap <= max_vertical_gap:
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


def segment_price_blobs(cell_bgr: np.ndarray, cell_box: tuple):
    """
    Like segment_digit_blobs, but for a Total Price cell, which has a
    handwritten decimal point that no Qty/Return field ever does (see
    DECIMAL_POINT_MAX_HEIGHT_FRACTION for how it's told apart from a
    leftover digit fragment). Kept as its own function rather than an
    option on segment_digit_blobs so Qty/Return reading -- already
    checked against real invoices and trusted -- can't be affected by
    a change made for Total Price.

    Args:
        cell_bgr, cell_box: same as segment_digit_blobs.

    Returns:
        (digit_images, decimal_before_count, num_decimal_candidates,
        fragment_count) -- digit_images is the same as
        segment_digit_blobs's return (the decimal point itself is never
        included in it, since it isn't a digit); decimal_before_count is
        how many of those digit images sit to the left of the resolved
        decimal point, set whenever a decimal point was found at all
        (including an ambiguous one -- see below); num_decimal_candidates
        is 0 (no point found), 1 (one clear point, not flagged), or 2
        (more than one similarly-plausible point -- classify_price flags
        this as ambiguous_decimal_point, but decimal_before_count still
        holds its best guess rather than being left unused); fragment_count
        is as in segment_digit_blobs.
    """
    cell_width = cell_box[2] - cell_box[0]
    cell_height = cell_box[3] - cell_box[1]

    ink = _remove_printed_lines(
        threshold_cell(cell_bgr, cell_height), cell_width, cell_height
    )
    ink = _owned_ink(ink, cell_box)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    cell_area = cell_width * cell_height

    blobs = []
    for label in range(1, num_labels):  # label 0 is the background
        bx, by, bw, bh, area = stats[label]
        # A lower floor than segment_digit_blobs' FRAGMENT_AREA_FRACTION
        # -- a genuine decimal point is smaller than what that constant
        # treats as "dust" (see PRICE_DUST_AREA_FRACTION).
        if area < PRICE_DUST_AREA_FRACTION * cell_area:
            continue
        blobs.append(
            {"x1": bx, "y1": by, "x2": bx + bw, "y2": by + bh,
             "area": area, "mask": labels == label}
        )

    # Only merge pieces that are also close together vertically -- see
    # _merge_stroke_fragments's max_vertical_gap for why this matters
    # more here than for Qty/Return: Total Price's much wider cells give
    # a leftover fragment of the row's own printed line more room to
    # coincidentally overlap, in x, with a small mark (like a decimal
    # point) that's actually nowhere near it vertically.
    blobs = _merge_stroke_fragments(blobs, max_vertical_gap=STROKE_MAX_VERTICAL_GAP_FRACTION * cell_height)

    def is_digit(blob):
        height = blob["y2"] - blob["y1"]
        if height >= MIN_BLOB_HEIGHT_FRACTION * cell_height:
            return True
        # Qualifying by area alone still requires a shape a digit could
        # plausibly have. Real digits and the merges the fixes above still
        # miss occasionally never come close to LINE_RESIDUE_MAX_ASPECT_RATIO
        # wide-to-tall -- measured across 903 real Total Price cells, a
        # genuine short digit's own aspect ratio maxes out well under it,
        # while a leftover fragment of the row's own printed line (long
        # and thin) routinely measures 8-70+. Without this, that fragment
        # alone -- no merge needed -- has enough raw ink area to pass as a
        # "digit" purely because it's long, triggering possible_merged_digits
        # on a cell that has no touching digits in it at all.
        width = blob["x2"] - blob["x1"]
        return (
            blob["area"] >= MIN_BLOB_AREA_FRACTION * cell_area
            and width <= LINE_RESIDUE_MAX_ASPECT_RATIO * height
        )

    def is_decimal_point(blob):
        height = blob["y2"] - blob["y1"]
        width = blob["x2"] - blob["x1"]
        # A real decimal point is compact -- roughly as wide as it is
        # tall, whether drawn as a dot or a short dash (the widest real
        # example seen, a dash, measured 2.6x). Without this, the same
        # line-residue fragment that is_digit() now rejects by shape
        # (LINE_RESIDUE_MAX_ASPECT_RATIO) could still sneak in here
        # instead, since a thin sliver is small in both area and height
        # even while being far too wide to be a dot -- measured across a
        # 30-page sample, 11% of everything that passed the area/height
        # check alone was one of these slivers, not a real point.
        return (
            blob["area"] <= DECIMAL_POINT_MAX_AREA_FRACTION * cell_area
            and height <= DECIMAL_POINT_MAX_HEIGHT_FRACTION * cell_height
            and width <= DECIMAL_POINT_MAX_ASPECT_RATIO * height
        )

    def is_stroke_sized(blob):
        return (blob["y2"] - blob["y1"]) >= MIN_FRAGMENT_HEIGHT_FRACTION * cell_height

    decimal_candidates = [b for b in blobs if is_decimal_point(b)]
    digit_blobs = [b for b in blobs if is_digit(b) and not is_decimal_point(b)]
    other_blobs = [b for b in blobs if not is_digit(b) and not is_decimal_point(b)]

    fragment_count = sum(1 for b in other_blobs if is_stroke_sized(b))

    # A short mark with nothing beside it isn't read as a digit -- see
    # MIN_LONE_DIGIT_HEIGHT_FRACTION.
    if len(digit_blobs) == 1 and (digit_blobs[0]["y2"] - digit_blobs[0]["y1"]) < MIN_LONE_DIGIT_HEIGHT_FRACTION * cell_height:
        fragment_count += is_stroke_sized(digit_blobs[0])
        digit_blobs = []

    digit_blobs.sort(key=lambda b: (b["x1"] + b["x2"]) / 2)

    decimal_before_count = None
    num_decimal_candidates = 0
    if decimal_candidates and digit_blobs:
        min_digit_x = min(b["x1"] for b in digit_blobs)
        max_digit_x = max(b["x2"] for b in digit_blobs)

        def gap_to_digits(blob):
            """Horizontal distance from blob to the nearest digit, 0 if it overlaps one in x."""
            gaps = []
            for d in digit_blobs:
                if blob["x2"] < d["x1"]:
                    gaps.append(d["x1"] - blob["x2"])
                elif blob["x1"] > d["x2"]:
                    gaps.append(blob["x1"] - d["x2"])
                else:
                    gaps.append(0)
            return min(gaps)

        def touches_a_digit(blob):
            """
            True rectangle-to-rectangle distance to the nearest digit,
            unlike gap_to_digits above which only measures horizontal
            distance -- see DECIMAL_POINT_TOUCH_DISTANCE_FRACTION for
            why that matters here specifically.
            """
            best = None
            for d in digit_blobs:
                dx = max(0, max(blob["x1"], d["x1"]) - min(blob["x2"], d["x2"]))
                dy = max(0, max(blob["y1"], d["y1"]) - min(blob["y2"], d["y2"]))
                dist = (dx ** 2 + dy ** 2) ** 0.5
                if best is None or dist < best:
                    best = dist
            return best < DECIMAL_POINT_TOUCH_DISTANCE_FRACTION * cell_height

        # A real decimal point has to fall somewhere close to the number
        # it belongs to -- between two of its digits (e.g. "27.00"), or
        # just past the last one when no cents digits were written at
        # all (e.g. "130." as a plain "$130 even" -- confirmed on a real
        # scan, gap 0.153 of the cell's width past the last digit). A
        # fleck of paper grain has no such preference and lands anywhere
        # in the (often wide) cell. A margin past each end of the digits
        # -- generous enough to cover a genuine trailing point with room
        # to spare, per that measurement -- keeps the second case while
        # still ruling out most of the scattered noise; see
        # CONTENDER_GAP_MARGIN_FRACTION for how a genuine tie between
        # what's left is still told apart from a single confident answer.
        span_margin = DECIMAL_POINT_SPAN_MARGIN_FRACTION * cell_width

        def before_count(point_blob):
            point_x = (point_blob["x1"] + point_blob["x2"]) / 2
            return sum(1 for d in digit_blobs if (d["x1"] + d["x2"]) / 2 < point_x)

        def cents_penalty(point_blob):
            """
            How unusual the number of digits after this point would be
            (0 = the usual two cents digits, 1 = one, 2 = anything else,
            3 = ahead of every digit, which is never right).
            Among reads that came out matching a product's usual price
            (a proxy for "read correctly"), 82% had exactly two digits
            after the point and 14% had one, so a candidate that gives
            the usual layout beats one that doesn't -- even one sitting
            nearer a digit, since a speck of paper grain or a piece of a
            small handwritten "0" often sits closer to a digit than the
            real point does, which the writer leaves room around.
            """
            before = before_count(point_blob)
            if before == 0:
                # A point ahead of every digit would make the total
                # under $1, which no product on this form costs.
                return 3
            after = len(digit_blobs) - before
            return 0 if after == 2 else 1 if after == 1 else 2

        contenders = sorted(
            (b for b in decimal_candidates
             if min_digit_x - span_margin <= (b["x1"] + b["x2"]) / 2 <= max_digit_x + span_margin),
            key=lambda b: (cents_penalty(b), gap_to_digits(b)),
        )
        if contenders:
            num_decimal_candidates = 1
            if len(contenders) > 1:
                gap_difference = gap_to_digits(contenders[1]) - gap_to_digits(contenders[0])
                # A tie is only worth flagging if it actually changes the
                # answer. Two candidates -- often a real point and a
                # separate stray mark -- can both sit in the same gap
                # between the same two digits, placing the point in the
                # identical spot either way regardless of which one is
                # "real" (measured: 37% of ties across all 96 scans were
                # exactly this -- same decimal_before_count from either
                # candidate). Only treat it as genuinely unresolved when
                # the two candidates would actually split the digits
                # differently.
                # Even a horizontal-gap tie is still a confident pick,
                # not a genuine one, when the closer candidate is truly
                # (2D) touching a digit and the runner-up plainly isn't
                # -- see DECIMAL_POINT_TOUCH_DISTANCE_FRACTION. Measured
                # on 134 real ties: 48 were exactly this pattern (the
                # runner-up was noise sitting elsewhere in the cell, not
                # a second plausible point), and only 13 had BOTH
                # candidates genuinely close to a digit -- those stay
                # flagged, since two real-looking candidates is an
                # actual ambiguity, not a resolved one.
                if (
                    cents_penalty(contenders[0]) == cents_penalty(contenders[1])
                    and gap_difference / cell_width < CONTENDER_GAP_MARGIN_FRACTION
                    and before_count(contenders[0]) != before_count(contenders[1])
                    and not (touches_a_digit(contenders[0]) and not touches_a_digit(contenders[1]))
                ):
                    num_decimal_candidates = 2  # a genuine tie, not a confident pick
            point_x = (contenders[0]["x1"] + contenders[0]["x2"]) / 2
            decimal_before_count = sum(
                1 for b in digit_blobs if (b["x1"] + b["x2"]) / 2 < point_x
            )

    digit_images = [
        (b["mask"][b["y1"]:b["y2"], b["x1"]:b["x2"]]).astype(np.uint8) * 255 for b in digit_blobs
    ]
    return digit_images, decimal_before_count, num_decimal_candidates, fragment_count


def classify_price(
    digit_blobs: list, decimal_before_count, num_decimal_candidates: int,
    fragment_count: int, model: DigitCNN, device,
) -> dict:
    """
    The Total Price equivalent of classify_blobs(): reads already-
    segmented digit blobs plus where the decimal point landed, and
    turns them into a dollar amount.

    Args:
        digit_blobs, fragment_count: as classify_blobs().
        decimal_before_count, num_decimal_candidates: as returned by
            segment_price_blobs().
        model, device: as classify_blobs().

    Returns:
        A dict:
          - "value": float or None. Unlike Qty/Return, a blank cell is
            NOT read as 0 here -- every filled-in Qty row's Total Price
            box had something written in it in every real scan checked
            so far (see CLAUDE.md's "Checked against 7 real scans"), so
            a genuinely blank one is unexpected and worth a person's
            attention rather than a silent assumption of $0.
          - "digits", "digit_confidences": as classify_blobs(), for the
            digit images only -- the decimal point is never included,
            since it isn't a 0-9 class and must never be banked as one.
          - "flagged", "flag_reasons": as classify_blobs(), plus
            "no_decimal_point" (none of the segmented marks looked like
            one) and "ambiguous_decimal_point" (more than one did).
    """
    if not digit_blobs:
        reasons = ["unreadable_ink"] if fragment_count else []
        return {
            "value": None,
            "digits": [],
            "digit_confidences": [],
            "flagged": bool(reasons),
            "flag_reasons": reasons,
        }

    flag_reasons = []
    if len(digit_blobs) > MAX_EXPECTED_PRICE_DIGITS:
        flag_reasons.append("too_many_blobs")
    if any(blob.shape[1] / blob.shape[0] > MAX_SINGLE_PRICE_DIGIT_ASPECT_RATIO for blob in digit_blobs):
        flag_reasons.append("possible_merged_digits")

    # Unlike classify_blobs() (Qty/Return), this does NOT also flag a
    # digit that's merely small relative to the tallest one in the same
    # cell. That comparison assumes a cell's digits are naturally close
    # in height, which holds for a 1-3 digit quantity but not for a
    # 3-6 digit dollar amount: one genuinely tall, thin digit (a "1" is
    # the common case) routinely reaches well over double the height of
    # an entirely normal neighbouring digit, making it look like a
    # fragment purely by that comparison. Measured across all 96 scans:
    # of every case this flag could have fired on, 86% had no leftover
    # fragment at all (fragment_count == 0) -- it was only ever the
    # relative-size comparison, flagging correctly-read prices like a
    # clean "20 x $3.00 = $60.00". fragment_count alone -- genuine
    # leftover ink that never became a digit -- is a real, separate
    # measurement and is kept.
    if fragment_count:
        flag_reasons.append("possible_split_digit")

    digits = []
    confidences = []
    with torch.no_grad():
        for blob in digit_blobs:
            tensor = _preprocess_blob(blob).to(device)
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)
            confidence, predicted = probs.max(dim=1)
            digits.append(str(predicted.item()))
            confidences.append(confidence.item())

    if any(c < LOW_CONFIDENCE_THRESHOLD for c in confidences):
        flag_reasons.append("low_confidence")

    if num_decimal_candidates == 0:
        flag_reasons.append("no_decimal_point")
        value = float("".join(digits))
    else:
        # Even when the point's exact position is a genuine tie between
        # two similarly-plausible marks (num_decimal_candidates == 2),
        # decimal_before_count still holds the closer-to-a-digit guess
        # rather than being thrown away -- a flagged field still gets
        # the best answer the software could manage (see module
        # docstring), so this reads as e.g. "130.0" for a reviewer to
        # confirm or correct, not the much less useful "1300".
        if num_decimal_candidates > 1:
            flag_reasons.append("ambiguous_decimal_point")
        whole = "".join(digits[:decimal_before_count]) or "0"
        frac = "".join(digits[decimal_before_count:]) or "0"
        value = float(f"{whole}.{frac}")

    return {
        "value": value,
        "digits": digits,
        "digit_confidences": confidences,
        "flagged": len(flag_reasons) > 0,
        "flag_reasons": flag_reasons,
    }


def read_price(cell_bgr: np.ndarray, cell_box: tuple, model: DigitCNN, device) -> dict:
    """
    Read a full Total Price out of one cell crop in one call -- the
    Total Price equivalent of read_number(). See segment_price_blobs()
    and classify_price() for the two halves this combines.
    """
    return classify_price(*segment_price_blobs(cell_bgr, cell_box), model, device)


def prepare_digit_image(blob: np.ndarray) -> np.ndarray:
    """
    Pad one segmented digit blob to a square and resize it to 28x28 --
    mirroring label_tool.py's own preprocessing exactly (pad to square
    BEFORE resizing, so a digit's proportions aren't distorted, ink
    already light-on-dark from segment_digit_blobs).

    Shared by _preprocess_blob (which goes on to normalize this for
    the model) and by extract_invoice.py, which saves this same image
    to disk as a PNG -- pixel-for-pixel the same format label_tool.py
    itself saves, so a saved crop can later be dropped straight into a
    retraining bank with no reprocessing (see CLAUDE.md, "Banking
    verified-correct crops as future training data").

    Args:
        blob: a binary (0/255) crop containing one digit's ink,
            already light-on-dark (see segment_digit_blobs).

    Returns:
        A (28, 28) uint8 array, ink as light-on-dark.
    """
    h, w = blob.shape
    side = max(h, w)
    padded = np.zeros((side, side), dtype=np.uint8)
    top, left = (side - h) // 2, (side - w) // 2
    padded[top:top + h, left:left + w] = blob
    return cv2.resize(padded, (28, 28), interpolation=cv2.INTER_AREA)


def _preprocess_blob(blob: np.ndarray) -> torch.Tensor:
    """
    Convert one segmented digit blob into the 28x28, MNIST-normalized
    tensor DigitCNN expects, so the model sees the same kind of input
    it was fine-tuned on, not something subtly different.

    Args:
        blob: a binary (0/255) crop containing one digit's ink,
            already light-on-dark (see segment_digit_blobs).

    Returns:
        A (1, 1, 28, 28) tensor ready to feed to DigitCNN.
    """
    resized = prepare_digit_image(blob)
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
          - "digits": list of the model's predicted label for each
            blob, in reading order, as single-character strings (e.g.
            ["3", "0"]) -- kept separately from "value" because
            joining them into one string can silently drop a leading
            "0", which the caller needs to still pair each saved digit
            crop with its own predicted label.
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
            "digits": [],
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

    # A real quantity on this form never legitimately starts with "0"
    # -- a genuinely blank cell already reads as 0 with no blobs at
    # all (see the early return above), so a leading "0" ahead of at
    # least one more digit is always some other mark segmented as its
    # own blob, not a real digit. Concatenating digits into "value"
    # silently drops it (int("06") == 6), which used to make this
    # invisible: the field looked right, so nobody had a reason to
    # correct it, and it would have been banked as confirmed-correct
    # training data with a wrong "0" label attached (see
    # extract_invoice.py's digit crop saving). Flagging it instead
    # keeps it in front of a reviewer and out of the training bank
    # until it's actually looked at.
    if len(digits) > 1 and digits[0] == "0":
        flag_reasons.append("leading_zero_digit")

    return {
        "value": int("".join(digits)),
        "digits": digits,
        "digit_confidences": confidences,
        "flagged": len(flag_reasons) > 0,
        "flag_reasons": flag_reasons,
    }
