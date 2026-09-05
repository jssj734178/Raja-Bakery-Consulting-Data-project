# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Code line counts

Maintained via [line_counts.py](line_counts.py) — after any substantive edit to a tracked `.py` file, re-run `python line_counts.py` and paste its markdown table output back in here. "Code lines" means actual executable syntax only; comments and docstrings are counted separately (not lumped into "code") since this project's convention is full docstrings plus detailed inline comments explaining *why* — so a large share of most files' line counts is documentation, not logic, and this table is meant to make that visible rather than hide it inside one combined number.

| File | Total lines | Code lines | Comment/docstring lines | Blank lines |
|---|---|---|---|---|
| `alignment.py` | 448 | 119 | 284 | 45 |
| `calibrate_template.py` | 414 | 256 | 105 | 53 |
| `data.py` | 106 | 23 | 67 | 16 |
| `digit_reader.py` | 630 | 182 | 394 | 54 |
| `extract_invoice.py` | 497 | 211 | 233 | 53 |
| `finetune.py` | 499 | 213 | 224 | 62 |
| `label_tool.py` | 342 | 160 | 133 | 49 |
| `line_counts.py` | 122 | 59 | 47 | 16 |
| `model.py` | 122 | 22 | 84 | 16 |
| `pdf_to_images.py` | 77 | 35 | 25 | 17 |
| `review_screen.py` | 545 | 289 | 194 | 62 |
| `split_dataset.py` | 107 | 47 | 43 | 17 |
| `train.py` | 157 | 48 | 79 | 30 |
| **Total** | **4066** | **1664** | **1912** | **490** |

*Last updated: 2026-09-03.*

## What this is

A pipeline for training a digit-recognition model on handwritten digits found on scanned bakery invoices. The model architecture (`DigitCNN`) is pretrained on MNIST first (Stage 1), then fine-tuned on real invoice digits (Stage 2). Both stages are implemented; the current shipped checkpoint reaches 94.19% test accuracy on real invoice digits. See [FINETUNING_NOTES.md](FINETUNING_NOTES.md) for the full history of how that number was reached — the diagnostics tried, what worked, and what didn't.

## Environment

- Python 3.14, virtualenv at `venv/` (Windows: `venv\Scripts\python.exe`).
- Dependencies are pinned in `requirements.txt`: `torch`, `numpy`, `pillow`, `pymupdf`. Install with `venv\Scripts\python.exe -m pip install -r requirements.txt`.
- This is a git repository with a GitHub remote (`origin`, `main` branch). `checkpoints/`, `invoices/`, `invoice_digits/`, `data/`, and `*.pdf` are all gitignored (see `.gitignore` for why) — checkpoints are regeneratable via `train.py`/`finetune.py`, and the invoice/digit data is real business data that shouldn't sit in git history.
- No test suite, linter, or formatter is configured.

## Pipeline / architecture

The scripts form a sequential pipeline, each a standalone CLI entry point (`python <script>.py`), run roughly in this order:

1. **[pdf_to_images.py](pdf_to_images.py)** — renders each page of one or more input PDFs (scanned invoices) to a PNG at 300 DPI, saved into `invoices/`. Supports `--rotate 90|180|270` for scans with orientation issues.
2. **[label_tool.py](label_tool.py)** — a tkinter GUI for manually drawing boxes around handwritten digits in `invoices/*` images and labeling them 0-9 via keypress. Saves each crop into `invoice_digits/<digit>/`, preprocessed to match MNIST's format (grayscale, inverted so ink is light-on-dark, padded to square before resizing to 28x28 — padding before resize matters so a digit's proportions aren't distorted). Crop filenames encode their source invoice as `{invoice_name}_{4-digit counter}.png`, which `split_dataset.py` depends on.
3. **[split_dataset.py](split_dataset.py)** — moves crops from `invoice_digits/<digit>/` into `invoice_digits/<train|val|test>/<digit>/`. Splits are assigned **by invoice, not by individual digit** (via a deterministic MD5 hash of the invoice name), so digits from the same invoice never end up split across train/val/test — otherwise the model could partially memorize an invoice's handwriting and inflate test accuracy. Safe to re-run repeatedly as more invoices get labeled over time; already-split invoices don't move between splits.
4. **[data.py](data.py)** — `load_mnist()` loads `data/mnist.pkl.gz` (three pickled `(images, labels)` splits) and wraps each in a PyTorch `DataLoader`, reshaping flat 784-vectors to `(1, 28, 28)` and standardizing with MNIST's known mean/std (0.1307 / 0.3081).
5. **[model.py](model.py)** — `DigitCNN`: two conv+pool blocks (1→32→64 channels, 28x28→7x7) feeding two fully-connected layers (3136→128→10). Returns raw logits (no softmax — `CrossEntropyLoss` applies it).
6. **[train.py](train.py)** — Stage 1: trains `DigitCNN` on MNIST for 5 epochs with Adam (`lr=1e-3`), reports val accuracy per epoch and final test accuracy, saves weights (`state_dict`) to `checkpoints/digit_cnn_mnist.pt`. Run with `python train.py`. `checkpoints/` doesn't exist until this has been run at least once. Reaches ~99.1% MNIST test accuracy.
7. **[finetune.py](finetune.py)** — Stage 2: loads `checkpoints/digit_cnn_mnist.pt`, freezes `conv1` (kept frozen — its low-level features transfer fine from MNIST as-is), fine-tunes `conv2` at a low learning rate plus the FC head at a normal one, using a class-weighted loss (`invoice_digits` is imbalanced) and on-the-fly augmentation (rotation, independent width/height scaling, small translation) since no more labeled invoice data is coming in. Reports a zero-shot baseline (MNIST model with no fine-tuning) before training, then per-digit accuracy and a full confusion matrix after. Saves the best-validation-accuracy checkpoint to `checkpoints/digit_cnn_finetuned.pt`. Supports `--seed` (default `3`, matching the shipped checkpoint — reproducible bit-for-bit) and `--freeze-conv2` (reverts to the more conservative FC-only fine-tuning, which scores ~5 points lower). Run with `python finetune.py`.

## Production inference pipeline (in progress)

A second pipeline, separate from the training pipeline above, for applying the finished fine-tuned model to NEW invoices going forward — the system that automatically extracts quantities from a scanned invoice rather than the system that built the model in the first place. Design: fixed pre-printed invoice template, so product identity comes from row POSITION (not OCR); a one-time calibration records each row's Qty/Return cell positions as proportions of the table border, reused on every future scan via that scan's own freshly-detected border (no full geometric image warp needed); Return is subtracted from Qty per row; low-confidence/ambiguous reads are flagged for human review rather than guessed. Calibration is complete and verified (`template_calibration.json` + `product_rows.json`, both checked into git). Per-invoice extraction (`extract_invoice.py` + `digit_reader.py`) is built and working — the segmentation bug that previously made its output untrustworthy is fixed and verified against hand-read ground truth (see "Known issues" below for what was wrong and what residuals remain). The human review screen (`review_screen.py`) is also built and working, up to a person approving an invoice locally — see "The human review screen: requirements and decisions" below. Only getting this into Odoo remains — see "Next to build: getting this into Odoo" below.


### Plain-language glossary

The code and the notes below use some standard image-processing words. In this project they mean:

| Term | What it actually means here |
|---|---|
| **scan** / **page** | One photographed or scanned page of an invoice, saved as a PNG. |
| **cell** or **box** | One rectangle on the printed form — e.g. the Qty box on row 5. |
| **crop** | A small rectangle cut out of the scan, usually one cell plus a bit of margin. |
| **threshold** | Deciding, for every dot in the picture, whether it is pencil or blank paper. |
| **blob** / **connected component** | One joined-up piece of ink. Ideally one digit, but a digit can arrive as several blobs, and two touching digits can arrive as one. |
| **deskew** / **straighten** | Rotating a crop so the form's printed lines are level, because the scan is slightly crooked. |
| **bounding box** | The smallest upright rectangle that fits around a piece of ink. |
| **aspect ratio** | How wide something is compared to how tall. Above 1 means wider than tall. |
| **confidence** | How sure the model is about a digit it just read, from 0 to 1. Low confidence is treated as "don't trust this". |
| **flag** | A marker on a field meaning "a person should check this before it is trusted". The field still gets a best-guess value. |
| **calibration** | The one-time recording of where every Qty and Return box sits on the form, saved in `template_calibration.json`. |

1. **[alignment.py](alignment.py)** — shared geometry, used by every other file in this pipeline. `detect_border_corners()` finds the invoice table's outer grid border on a scan (see the detailed fix writeup below for how its precision was hardened). `proportion_to_pixel()`/`pixel_to_proportion()` convert between actual pixel coordinates and proportions of that border (bilinear interpolation across the border's 4 corners), which is the mechanism that lets one calibration be reused on any new scan regardless of exactly where/how big its own border lands. `validate_aspect_ratio()` is the reliability guard — flags a scan whose detected border shape doesn't plausibly match the calibrated template, rather than proceeding with bad coordinates.
2. **[calibrate_template.py](calibrate_template.py)** — one-time interactive tkinter tool (same drag-box / keypress-confirm / zoom / pan / undo conventions as `label_tool.py`) for recording each row's Qty and Return cell positions as proportions of the detected border. Run once against any representative scan of the fixed template (a filled-in invoice is fine — only the printed ruled-line geometry matters). Saves `template_calibration.json` (not gitignored, unlike the training pipeline's generated outputs — it holds only template geometry, not business data) plus a `product_rows.json` skeleton for hand-filling in each row's product name afterward, matched by row position — deliberately never overwritten by re-running calibration, so hand-edited product names are never at risk of being clobbered. **A row was skipped during the actual calibration session — see "Known issues" below.**
3. **[digit_reader.py](digit_reader.py)** — takes the small picture of one Qty or Return box and works out what number is handwritten in it.

   It works through the picture in steps, and the order matters because each step depends on the one before:

   1. **Decide which specks are pencil and which are paper.** It compares each spot to the paper immediately around it, rather than picking one brightness for the whole picture. (Doing this the simple way was the single biggest source of wrong answers — see "What was wrong" below.)
   2. **Erase the form's own printed lines.** The printed lines are recognised by being *long and straight*, which handwriting never is. This only works because the picture has already been straightened by `extract_invoice.py` — more on that below.
   3. **Repair the digits that erasing damaged.** People write over the printed lines, so rubbing a line out can cut a digit into pieces — the top of a "5" gets separated from its body. This step joins those pieces back up.
   4. **Throw away ink that belongs to a different box.** The picture deliberately includes some of the rows above and below (again, see below for why), so anything mostly sitting outside this box is discarded.
   5. **Join up the separate strokes of one digit.** Some people write a "4" as two separate strokes that never touch.
   6. **Read each remaining piece of ink** with the trained model and put the digits together left to right, so a "3" then a "0" becomes 30.

   Leftover specks are sorted by size: dust is ignored silently, something stroke-sized but too small to be a digit is left out of the number *but raises a review flag*, and anything digit-sized is read.

   Finally it decides whether the answer can be trusted, using several separate checks that deliberately overlap, so a problem missed by one is usually caught by another: more than 3 digits found (real quantities on this form aren't bigger than that); a piece of ink far wider than it is tall (usually two digits touching and being read as one); a leftover stroke (usually a digit written in disconnected pieces); or the model itself not being confident about a digit (which also catches scribbles and crossings-out that aren't digits at all).

   A flagged field still gets the best answer the software could manage rather than being left empty. The flag means "a person should look at this before trusting it", not "nothing could be read".
4. **[extract_invoice.py](extract_invoice.py)** — runs one whole invoice from start to finish: find the table's outer border on the new scan, check the border is a sensible shape (if not, the whole invoice is set aside for manual handling), then for every row cut out the Qty box and the Return box, read each one, and subtract Return from Qty. It writes `results.json` plus a picture of every single box — flagged or not — into `extractions/<invoice_name>/`, because the human review screen will need to show a picture of any field a person wants to check. Run with `python extract_invoice.py path/to/scan.png`.

   Three things this file does are worth explaining, because they look odd until you know why:

   - **It straightens each box before cutting it out.** These scans sit about 1.4 degrees crooked. That sounds tiny, but across the width of one box a "horizontal" printed line drifts down by about 26 pixels, while the line itself is only about 7 pixels thick — so the line is nowhere near level, and the step that erases printed lines simply fails on a crooked picture. Straightening costs nothing extra, because the tilt can be worked out from the corners of the box we already calculated.
   - **Each box is cut out twice, at two different sizes.** The copy used for *reading* is deliberately too big, taking in a whole row's height above and below. That seems wrong, but step 4 above needs to see a neighbouring row's digit in full to judge that it belongs to that row and not this one — if it's cut off at the edge of the picture, the visible sliver can look like it belongs here, and that is exactly why blank boxes used to be read as numbers. The copy *saved for a person to look at* is kept tight and tidy.
   - **It nudges each box's edges inwards onto the printed lines it can actually see on this scan.** The saved calibration reliably says *which* box we want, but not its exact position on every scan. On about a third of pages tested, the Return box reached past the column divider and swallowed the printed price in the next column, so "$4.00" was being read as that row's return quantity — quietly, on nearly every row of the page. The nudge only ever makes a box smaller, never bigger: allowing it to grow outwards turned a correctly-read "50" into "501".
5. **[review_screen.py](review_screen.py)** — the screen a person uses to check, correct, assign a customer to, and approve one invoice's extraction before anything goes to Odoo. Reads `results.json` and `crops/` from `extractions/<invoice_name>/` (both already produced by `extract_invoice.py`, above). Shows every row — not just flagged ones — with the product name, an editable box for its Qty and Return values, and a line quantity (Qty minus Return) that recalculates live as those are edited. A field the software flagged gets a pink background, a plain-language reason, and its crop image so the reason can be checked against the actual handwriting; an unflagged field skips the image (see "The human review screen" below for why) but stays just as editable. The customer picker at the top is one control doing both jobs the requirements called for: its dropdown offers the saved customer list, and it also accepts typing any name that isn't on it — and a typed name that does match one on the list (ignoring case/whitespace) is folded onto that customer's exact listed spelling rather than kept as separately-typed text. Approving an invoice writes `review.json` next to its `results.json` — the chosen customer, and every row's corrected value kept alongside what the software originally read, so a correction stays visible as a correction rather than overwriting the record of it. Deliberately does not talk to Odoo itself. Run with `python review_screen.py`, or `python review_screen.py "invoice folder name"` to open a specific invoice first.

## The human review screen: requirements and decisions

[review_screen.py](review_screen.py) (above) is built and working, up to
the point of a person approving an invoice locally. What it deliberately
does not do yet is talk to Odoo — see "Next to build" below for that.

**Customer selection (requested 2026-09-02, list arrived 2026-09-02).**
Both required ways of setting the customer are there: pick from the
regular list, saved in [customers.json](customers.json) (39 customers,
checked into git the same way `product_rows.json` is — plain business
names, not scan data), or type a name that isn't on it, for a one-off
customer who doesn't belong on the list.

**Typed-name matching (decided 2026-09-03).** A typed name that matches an
existing customer — ignoring case and surrounding whitespace — is folded
onto that customer's exact listed spelling, rather than kept as
separately-typed text or merely suggested. This happens as soon as the
customer field loses focus, so the correction is visible before
approving, and is applied again (defensively) at Approve & Save. A typed
name that doesn't match anything on the list is kept exactly as typed —
that's the free-text option working as intended, for a genuine one-off
customer.

**How much gets reviewed (decided 2026-09-02).** Every quantity and return
field is shown and editable, whether or not it was flagged — not just the
~39% of filled-in fields that raise a flag. Flagged-only review would be
faster but would miss a confident misread: the model is roughly 94%
accurate per digit, and when it is wrong it is often wrong confidently, so
nothing flags it (see "What is still not perfect" below). A flagged field
is shown with a pink background and the reason, but stays just as editable
as every other field.

**Crop images shown only for flagged fields (decided 2026-09-03).** Every
field is still editable regardless of flag state (per the decision just
above), but the handwriting photo itself is now only shown for a flagged
field. With up to 24 rows on screen and most fields unflagged, showing a
photo for all of them made the list too tall to scroll through
comfortably. A flagged field's photo is what a reviewer actually needs
open, to check the software's stated reason against the real handwriting;
an unflagged field's photo wasn't earning the space it cost.

## Next to build: getting this into Odoo

Not started yet. `review_screen.py`'s "Approve & Save" currently writes
`extractions/<invoice_name>/review.json` — the chosen customer plus every
row's corrected quantity/return/line quantity, each alongside what the
software originally read and whether that field had been flagged. That
file is the intended starting point for whatever pushes an approved
invoice into Odoo, but which specific approach to build was still being
decided — see below for where that landed (decided 2026-09-05).

**Decision: build a custom Odoo Community Edition module, not a
standalone web app (decided 2026-09-05).** Two other options were
weighed first:

- A desktop button added to `review_screen.py` (pick a PDF file, run it
  through the pipeline automatically) is the fastest thing to build, but
  only ever runs on the one Windows machine it's installed on — it can't
  be reached from a phone or any other computer, and every uploaded PDF
  and its extraction just accumulates as local files on that one
  machine's hard drive, the same way the 96 scans in `invoices/` do
  today, with no backup beyond whatever that machine's owner already has
  in place.
- A standalone web app (a small server plus a browser-based rewrite of
  the whole review screen) would fix the "only one machine" problem, but
  is pure extra work: it still leaves the Odoo push itself unbuilt
  afterward, as a separate fourth project.
- The Odoo module does both jobs in one build: Odoo already provides file
  upload, a browser-based UI framework, login/access control, and a
  central database, so this reuses that instead of building it from
  scratch — and because it's built inside Odoo, "Approve" can create the
  real Odoo record directly instead of producing a `review.json` for some
  future separate script to consume. It also means the tool is reachable
  by any device with a browser, phones included, unlike a desktop-only
  build.

**Approve, precisely (decided 2026-09-05):**

- **No stock/inventory movement.** This business does not track
  warehouse quantities in Odoo for these products, so approving an
  invoice never needs to touch stock levels — only billing.
- **Pricing comes from Odoo, not the paper.** The invoice form has a
  printed price column, but it is not used — each invoice line is
  created with no price set, so Odoo fills it in from whatever price
  list already applies to that product and customer.
- **Creates a draft invoice, not a finalized one.** Approve creates a
  Customer Invoice (Odoo's `account.move`) in **draft** status, with one
  line per product row (quantity = Qty minus Return, matched to an
  existing Odoo product by name and an existing Odoo contact by
  customer) — left as a draft so a person still gives it a final look
  and posts/sends it from inside Odoo itself, rather than Approve being
  the last checkpoint.

**Project structure: keep this project as the shared engine; the Odoo
module is a separate project (decided 2026-09-05).** This repository
stays as the "engine" — `alignment.py`, `digit_reader.py`,
`extract_invoice.py`, `model.py`, and the trained checkpoint — since a
tkinter project and an Odoo module (which requires its own specific
folder shape: a manifest file, XML views, etc.) don't share a structure.
The Odoo module should be built to *reference* this project's code
(installed as a dependency) rather than have its files copied in —
copying would work initially, but a fix or model improvement made here
later would then have to be manually re-applied inside the Odoo module
too, and the two copies would quietly drift apart over time.

**Also planned, not yet started:**

- **A desktop "import PDF" button on `review_screen.py`.** Not the
  long-term intake path (see above for why), but worth adding anyway as
  a quick way to feed new scans through the pipeline locally without
  typing commands, while the Odoo module is being built.
- **Banking verified-correct crops as future training data.** When a
  reviewer leaves a field unflagged and uncorrected, that's a
  high-confidence signal the model's digit-by-digit read was right, so
  those digit crops and the model's own labels could be saved into a
  growing bank for retraining later — free labeled data, without anyone
  hand-labeling anything new. The one prerequisite: `extract_invoice.py`
  only saves each whole Qty/Return cell as one crop today, not each
  individual digit inside it, so it would need to also save the
  individual digit crops `digit_reader.py` already segments internally
  before this can work. A *corrected* field is harder to bank this way —
  if the software had mis-split the digits in the first place, the
  corrected number can't always be cleanly matched back onto individual
  digit images — so this would start with unflagged/uncorrected fields
  only.

## Earlier fix: one corner of the detected table border was in the wrong place (`alignment.py`)

**In plain terms:** the software draws a box around the invoice's printed table, and everything else it does is measured from that box's four corners. One of those four corners was landing outside the real table edge — by roughly 70 to 160 pixels on a 10,000-pixel-wide scan — because the method being used had to stretch the box to cover every last stray speck of ink, and one speck sticking out was enough to drag a corner with it. Three different approaches were tried and measured before one worked. The full write-up is kept below because the reasoning is easy to lose and expensive to rediscover; the terminology in it is heavier than the rest of this document.

A detailed record of a real bug found and fixed in `detect_border_corners()` while calibration was first being tested for real, kept for the same reason as [FINETUNING_NOTES.md](FINETUNING_NOTES.md) — so the reasoning behind the current implementation doesn't have to be rediscovered later.

**What happened.** `detect_border_corners()` is the anchor the entire production inference pipeline is built on — every row/column proportion, both during calibration and later extraction, is defined relative to whatever 4 corners it returns. During the first real test of `calibrate_template.py`, one of its four detected corners was visibly outside the table's actual printed edge, while the other three landed correctly. This was caught by eye, comparing the tool's green border overlay against the real ruled table on a genuine scan — not by any automated check, since the existing aspect-ratio reliability guard (`validate_aspect_ratio()`) only flags a border whose overall *shape* is implausible, and a single-corner error small enough to still look roughly rectangular doesn't necessarily move the aspect ratio enough to trip it.

**Where it came from.** The original implementation isolated the table's ruled lines via morphological line-extraction (erode+dilate with a wide/tall kernel to keep only long horizontal/vertical strokes, discarding handwriting and printed text), combined the horizontal and vertical results into one mask, took its largest connected contour, and reduced that contour to 4 corners with `cv2.minAreaRect` — the smallest *rotated* rectangle enclosing it. The contour itself is jagged at the pixel level (ink-thickness variation, anti-aliasing, small notches where ruled lines meet), and `minAreaRect` must expand to cover every point in it, including any single point that happens to jut out slightly further than the table's true edge. One such outlier pixel could pull one corner of the fitted rectangle noticeably past the real border while the other three corners — unaffected by that particular outlier — stayed accurate. This was confirmed, not just suspected: the true top and left border lines were independently fit directly from the underlying pixel data (a least-squares line through pixels well away from any corner, avoiding the same jaggedness), giving a precise, code-independent ground-truth corner position to measure against. The `minAreaRect` corner was off by roughly 70-160px on a ~10,000px-wide scan, depending on which real scan was tested.

**How it affected the product.** `calibrate_template.py` records every row's Qty/Return cell position as a proportion of whichever border `detect_border_corners()` returns for the calibration reference scan, and the (not-yet-built) per-invoice extraction step is designed to apply those same proportions to every new scan's own freshly-detected border. If the reference border itself is off at one corner, every proportion computed relative to it inherits a share of that error — cells near the good corners would be affected only slightly, but cells further from them would drift further from their true position. Downstream, that risks clipped or wrong-cell digit crops during extraction, silently, since the error is corner-specific rather than a uniform shift of the whole border — a quick glance at the overlay could plausibly miss it unless the affected corner specifically is scrutinized closely, which is exactly what caught it here.

**How the fix was designed.** Three candidate corner-extraction methods were tried in turn, each tested against real scans and checked against the same independent, precisely-measured ground truth (never just eyeballed) before being accepted or rejected:

1. **`minAreaRect`** (original) — rejected, per the overshoot confirmed above.
2. **`approxPolyDP` on the contour's convex hull** — the standard technique for reducing a document/table's outline to exactly 4 corners, tried next on the assumption it would be more robust than a rotated-rectangle fit. Rejected: it was actually *worse* in testing, occasionally locking onto a small ink-blob protrusion on an otherwise-straight edge as one of only 4 allowed polygon vertices — producing a corner over 2,000px off, far worse than `minAreaRect`'s own error. Both methods share the same underlying weakness: reducing one noisy, pixel-jagged contour down to a small handful of points, where a single outlier pixel can dominate the result.
3. **Per-edge line fitting restricted to a single connected component** — tried isolating each of the table's 4 border lines by requiring it to survive as one connected component in the horizontal/vertical line masks (filtered by component length, keeping only components spanning most of the table's width/height). Rejected after debugging showed the outer border specifically — being crossed by *every* internal row/column divider along its full length — fragments into many small disconnected pieces at those crossings, while internal single dividers (crossed far less often) stay fully connected. Requiring single-component connectivity meant the length filter was, in practice, only ever finding an internal divider's own fragment, never the true (but fragmented) outer border.
4. **Band-anchored robust line fitting** (shipped) — uses the original combined-mask bounding box only as a rough anchor (empirically already close, just not corner-precise on its own), then, within a narrow search band around each of that box's 4 edges, gathers *every* surviving line-mask pixel inside the band regardless of which disconnected fragment it belongs to, and fits one line per edge with `cv2.fitLine` using a Huber (outlier-robust) loss rather than plain least-squares. The 4 corners are then just the pairwise intersections of adjacent fitted lines (top∩left, top∩right, bottom∩left, bottom∩right). This avoids both prior failure modes at once: it doesn't require single-component connectivity (fixing the fragmentation problem from approach 3), and a robust loss keeps the rare stray pixel inside a band from skewing that edge's fit (fixing the outlier-sensitivity problem from approaches 1 and 2).

**Verification.** The same previously-wrong corner was re-measured against the same independent line-fit ground truth (error dropped from ~70-160px to a few pixels), all 4 corners were re-checked visually via zoomed, pixel-level crops with the detected corner marked, and full detection was re-run across all 96 real scanned invoice pages in `invoices/` — 100% successful detection, aspect ratios tightly clustered with no change in distribution from before the fix, and zero regressions on any of the 4 distinct invoices represented in that set.

## Known issues (production inference pipeline)

**Fixed: one row was skipped during calibration.** The actual `calibrate_template.py` session that produced the checked-in `template_calibration.json` skipped one product row (`Dempster Bread Brown (675g)`, the 3rd row on the printed form) — confirmed two ways: visually (its calibrated row 2 landed on "Dempster Texas Sandwich White Bread" instead) and by measuring row-to-row spacing (every other row is ~250-280px apart on the reference scan; the gap between calibrated rows 1 and 2 was ~577px, almost exactly double). Fixed by inserting an interpolated row between them — box proportions averaged from its two immediate neighbors, which is a low-risk interpolation given how consistent the real row spacing is everywhere else — then re-verified visually (the inserted row lands exactly on "Dempster Bread Brown") both on the calibration reference scan and on a second, different real scan. `template_calibration.json` now has all 24 rows and `product_rows.json` is fully filled in, matching the product list top to bottom.

### Fixed: the reader could not tell the form's printed lines apart from handwriting

**What you would have seen.** The extraction was wrong in both directions at once. Boxes that were actually empty came back with a number in them, and boxes that clearly had "50" written in them came back empty. Two earlier attempts had been made to fix this by teaching the software to recognise the printed box outline and ignore it — first by its shape, then by the fact that it touches the edge of the cut-out picture. Each attempt fixed one of the two problems and made the other one worse.

**Why those attempts couldn't work.** They were both aimed at the wrong thing. The printed outline was a symptom. Underneath it there were four separate problems happening at the same time, and the biggest one had nothing to do with the printed lines at all.

1. **The software was deciding "pencil or paper?" by looking at the whole picture at once.** It picked a single brightness for the cut-out and called everything darker than that pencil. That works when a picture is roughly half dark and half light, but a box on this form is about 95% blank paper with a few faint pencil strokes on it, so the cut-off ended up landing on the paper's own grain instead. On the darker invoices this just made the writing look rough. On a lightly-pencilled invoice it broke the digits into crumbs, and the trailing zeros disappeared completely — a whole page where "30", "40" and "20" were being read as "3", "4" and "2". **This was the largest single cause of wrong numbers**, and it stayed hidden for a long time because it looked like a different problem each time it showed up: sometimes it was blamed on the way digits were being separated, and in at least one case it was blamed on the model being inaccurate.

2. **The scans are about 1.4 degrees crooked.** The step that finds the form's printed lines looks for long *level* lines. On a crooked scan there aren't any — across one box, a "horizontal" line drops about 26 pixels while being only about 7 pixels thick. So the printed lines were never being found cleanly in the first place, which is why *every* attempt to filter them out failed no matter how it was written.

3. **People write over the printed lines.** When a digit crosses a line, the software sees the digit and the line as a single joined-up shape. So erasing the line erased the digit with it. That is what made a box containing "50" come back empty.

4. **The cut-out picture deliberately includes part of the rows above and below**, so that a digit written high or low doesn't get its top or bottom sliced off. But because the neighbouring row's digits were arriving chopped off at the edge of the picture, there was no way to measure how much of them belonged to this row. That is what made empty boxes come back with a number in them.

**What was done.** Each cause was fixed at its own level: the software now judges "pencil or paper?" by comparing each spot against the paper right next to it (1); each box is straightened before anything else happens, which lets the printed lines be found by being long and straight, with a repair step afterwards to rejoin any digit the erasing cut through (2 and 3); and the picture used for reading is made deliberately larger so neighbouring digits appear whole, with each piece of ink then assigned to whichever box holds most of it (4). The old "ignore anything touching the edge" rule was deleted — the new ownership test replaces it and is more precise.

One subtlety in that last part is worth recording. Ink is assigned to a box a whole *character* at a time, not a piece at a time. Erasing the column divider cuts the printed "$" in the next column in half, and the left half of it is then sitting entirely inside the Return box, where a piece-by-piece test can only conclude it belongs there — and it reads as a "1".

**Two more problems found while checking the fix, also fixed.** The boxes in the saved calibration were drawn by hand and overlap each other slightly, so writing that landed in the shared strip was counted twice, once for each row. And the saved calibration reliably identifies *which* box is wanted but not its exact pixels on every scan, so on about a third of pages the Return box reached past the column divider and read the printed price in the next column as a return quantity.

**How it was checked.** Against three real scans from three different invoices, with the correct answers read off the page by eye first, deliberately including the faintest and hardest one. On the faint page every value checked now comes out right except one, where the model read a "9" as a "4". Before the fix, most of that page's second digits were being dropped entirely. Across a 24-page sample, the Return column had been reporting writing in 216 of 576 boxes — implausible on a form where returns are much rarer than orders — and now reports 112, while the Quantity column stayed steady at 169. In other words the fix removed wrong answers rather than just producing fewer answers. About 39% of filled-in fields raise a review flag.

**What is still not perfect.** None of these are silent except the last one:

- A digit written as two strokes that are far apart can still be split up and misread. Automatically joining the pieces was tried and rejected, because the setting that correctly rejoins a split "4" also glues a genuine "50" into a single shape. A leftover stroke raises a review flag instead, which was the deliberate choice.
- One person writing these invoices draws trailing zeros at about half the height of the digit before them. That breaks the otherwise sensible assumption that the digits in one box are all about the same height. It is the reason a piece of ink qualifies as a digit by being either big enough *or* tall enough, rather than by height alone — worth remembering before tightening either setting.
- The model itself is about 94% accurate per digit, so it will occasionally read a digit wrong and be confident about it — for example a "9" read as a "4". Nothing flags a confident mistake. That is a limitation of the model, not of the reading process, and improving it means improving the model.

## Current data state

`invoice_digits/` has a real train/val/test split (several hundred to a thousand+ crops per digit in train, proportionally fewer in val/test) and is now finalized — no further labeled invoice data is expected. Digit classes are imbalanced — `0` has roughly 3-4x more examples than digits like `7`-`9`, handled via class-weighted loss in `finetune.py` rather than by collecting more data for the rare classes.

The shipped `checkpoints/digit_cnn_finetuned.pt` (regenerate with `python finetune.py`) reaches **94.19% test accuracy**. Digits `3` and `9` were historically the weakest (most often confused with `8`/`1` and `4`/`7` respectively); see [FINETUNING_NOTES.md](FINETUNING_NOTES.md) for the diagnosis, including a spot-check suggesting some of the remaining `9`→misclassified-as-something-else cases may be labeling errors rather than model errors — worth a manual look at the flagged crops listed there before assuming the model is at fault.
