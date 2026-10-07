# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Code line counts

Maintained via [line_counts.py](line_counts.py) — after any substantive edit to a tracked `.py` file, re-run `python line_counts.py` and paste its markdown table output back in here. "Code lines" means actual executable syntax only; comments and docstrings are counted separately (not lumped into "code") since this project's convention is full docstrings plus detailed inline comments explaining *why* — so a large share of most files' line counts is documentation, not logic, and this table is meant to make that visible rather than hide it inside one combined number.

| File | Total lines | Code lines | Comment/docstring lines | Blank lines |
|---|---|---|---|---|
| `alignment.py` | 522 | 149 | 323 | 50 |
| `calibrate_template.py` | 414 | 256 | 105 | 53 |
| `calibrate_total_price.py` | 184 | 90 | 71 | 23 |
| `compare_extractions.py` | 78 | 44 | 24 | 10 |
| `data.py` | 106 | 23 | 67 | 16 |
| `digit_reader.py` | 1242 | 360 | 780 | 102 |
| `extract_invoice.py` | 687 | 306 | 318 | 63 |
| `finetune.py` | 499 | 213 | 224 | 62 |
| `label_tool.py` | 342 | 160 | 133 | 49 |
| `line_counts.py` | 122 | 59 | 47 | 16 |
| `match_odoo_products.py` | 60 | 28 | 21 | 11 |
| `model.py` | 122 | 22 | 84 | 16 |
| `odoo_client.py` | 169 | 61 | 96 | 12 |
| `pdf_to_images.py` | 77 | 35 | 25 | 17 |
| `review_screen.py` | 1088 | 531 | 459 | 98 |
| `send_to_odoo.py` | 170 | 80 | 69 | 21 |
| `split_dataset.py` | 107 | 47 | 43 | 17 |
| `train.py` | 157 | 48 | 79 | 30 |
| **Total** | **6146** | **2512** | **2968** | **666** |

*Last updated: 2026-10-06.*

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

A second pipeline, separate from the training pipeline above, for applying the finished fine-tuned model to NEW invoices going forward — the system that automatically extracts quantities from a scanned invoice rather than the system that built the model in the first place. Design: fixed pre-printed invoice template, so product identity comes from row POSITION (not OCR); a one-time calibration records each row's Qty/Return cell positions as proportions of the table border, reused on every future scan via that scan's own freshly-detected border (no full geometric image warp needed); Return is subtracted from Qty per row; low-confidence/ambiguous reads are flagged for human review rather than guessed. Calibration is complete and verified (`template_calibration.json` + `product_rows.json`, both checked into git). Per-invoice extraction (`extract_invoice.py` + `digit_reader.py`) is built and working — the segmentation bug that previously made its output untrustworthy is fixed and verified against hand-read ground truth (see "Known issues" below for what was wrong and what residuals remain). The human review screen (`review_screen.py`) is also built and working, up to a person approving an invoice locally — see "The human review screen: requirements and decisions" below. A "Send to Odoo" button on that screen (built 2026-09-28, see `send_to_odoo.py`) now pushes an approved invoice into Odoo as a draft invoice, verified working against the live test server — see "Next to build: getting this into Odoo" below for what that button does and doesn't cover, and how it fits alongside the still-unstarted full Odoo module.


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
| **calibration** | The one-time recording of where every Qty, Return, and Total Price box sits on the form, saved in `template_calibration.json`. |

1. **[alignment.py](alignment.py)** — shared geometry, used by every other file in this pipeline. `detect_border_corners()` finds the invoice table's outer grid border on a scan (see the detailed fix writeup below for how its precision was hardened). `proportion_to_pixel()`/`pixel_to_proportion()` convert between actual pixel coordinates and proportions of that border (bilinear interpolation across the border's 4 corners), which is the mechanism that lets one calibration be reused on any new scan regardless of exactly where/how big its own border lands. `validate_aspect_ratio()` is the reliability guard — flags a scan whose detected border shape doesn't plausibly match the calibrated template, rather than proceeding with bad coordinates. `ruled_line_positions()` finds a form's printed straight lines in an already-thresholded picture (isolating anything that runs straight for long enough that handwriting never qualifies) — shared by `extract_invoice.py` (snapping a cell box onto its real edges on a new scan) and `calibrate_total_price.py` (finding the Total Price column's own divider once, during calibration).
2. **[calibrate_template.py](calibrate_template.py)** — one-time interactive tkinter tool (same drag-box / keypress-confirm / zoom / pan / undo conventions as `label_tool.py`) for recording each row's Qty and Return cell positions as proportions of the detected border. Run once against any representative scan of the fixed template (a filled-in invoice is fine — only the printed ruled-line geometry matters). Saves `template_calibration.json` (not gitignored, unlike the training pipeline's generated outputs — it holds only template geometry, not business data) plus a `product_rows.json` skeleton for hand-filling in each row's product name afterward, matched by row position — deliberately never overwritten by re-running calibration, so hand-edited product names are never at risk of being clobbered. **A row was skipped during the actual calibration session — see "Known issues" below.**
3. **[calibrate_total_price.py](calibrate_total_price.py)** — a second, smaller calibration tool that adds a `total_price_box` to every row of an already-calibrated `template_calibration.json`. Unlike `calibrate_template.py`, this one isn't interactive: it finds the Total Price column's own left divider automatically (the column's right edge needs no detection at all — it's simply the table's own outer border, since Total Price is the rightmost column with nothing printed after it), then saves a preview image of a few real rows so the result can still be checked by eye before being trusted. Run once, after `calibrate_template.py`; see "Reading Total Price and deriving unit price" below for how this was verified.
4. **[digit_reader.py](digit_reader.py)** — takes the small picture of one Qty, Return, or Total Price box and works out what number is handwritten in it.

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

   A Total Price box goes through the same steps, plus one more: finding the handwritten decimal point, which nothing else on this form has. See "Reading Total Price and deriving unit price" below for how that part works and how solid it currently is (less proven than the rest of this list).
5. **[extract_invoice.py](extract_invoice.py)** — runs one whole invoice from start to finish: find the table's outer border on the new scan, check the border is a sensible shape (if not, the whole invoice is set aside for manual handling), then for every row cut out the Qty box and the Return box, read each one, and subtract Return from Qty. It writes `results.json` plus a picture of every single box — flagged or not — into `extractions/<invoice_name>/`, because the human review screen will need to show a picture of any field a person wants to check. Run with `python extract_invoice.py path/to/scan.png`, or give it several scans at once, e.g. `python extract_invoice.py invoices/*.png`. A batch is much faster than running it once per scan, because the ~20-second startup is only paid once (see "Speed-up" below).

   Three things this file does are worth explaining, because they look odd until you know why:

   - **It straightens each box before cutting it out.** These scans sit about 1.4 degrees crooked. That sounds tiny, but across the width of one box a "horizontal" printed line drifts down by about 26 pixels, while the line itself is only about 7 pixels thick — so the line is nowhere near level, and the step that erases printed lines simply fails on a crooked picture. Straightening costs nothing extra, because the tilt can be worked out from the corners of the box we already calculated.
   - **Each box is cut out twice, at two different sizes.** The copy used for *reading* is deliberately too big, taking in a whole row's height above and below. That seems wrong, but step 4 above needs to see a neighbouring row's digit in full to judge that it belongs to that row and not this one — if it's cut off at the edge of the picture, the visible sliver can look like it belongs here, and that is exactly why blank boxes used to be read as numbers. The copy *saved for a person to look at* is kept tight and tidy.
   - **It nudges each box's edges inwards onto the printed lines it can actually see on this scan.** The saved calibration reliably says *which* box we want, but not its exact position on every scan. On about a third of pages tested, the Return box reached past the column divider and swallowed the printed price in the next column, so "$4.00" was being read as that row's return quantity — quietly, on nearly every row of the page. The nudge only ever makes a box smaller, never bigger: allowing it to grow outwards turned a correctly-read "50" into "501".

   Every row's Total Price box goes through this same crop/straighten/nudge treatment, and its own reading gets divided by that row's line quantity to get a unit price — see "Reading Total Price and deriving unit price" below.
6. **[review_screen.py](review_screen.py)** — the screen a person uses to check, correct, assign a customer to, and approve one invoice's extraction before anything goes to Odoo. Reads `results.json` and `crops/` from `extractions/<invoice_name>/` (both already produced by `extract_invoice.py`, above). Shows every row — not just flagged ones — with the product name, editable boxes for its Qty, Return, and Total Price values, a line quantity (Qty minus Return) and a unit price (Total Price ÷ line quantity), both recalculating live as those are edited. A field the software flagged gets a pink background, a plain-language reason, and its crop image so the reason can be checked against the actual handwriting; an unflagged field skips the image (see "The human review screen" below for why) but stays just as editable. The customer picker at the top is one control doing both jobs the requirements called for: its dropdown offers the saved customer list, and it also accepts typing any name that isn't on it — and a typed name that does match one on the list (ignoring case/whitespace) is folded onto that customer's exact listed spelling rather than kept as separately-typed text. Approving an invoice writes `review.json` next to its `results.json` — the chosen customer, and every row's corrected value kept alongside what the software originally read, so a correction stays visible as a correction rather than overwriting the record of it. Deliberately does not talk to Odoo itself. Run with `python review_screen.py`, or `python review_screen.py "invoice folder name"` to open a specific invoice first.

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
- **Pricing comes from the paper, not Odoo (reversed 2026-09-23 — see
  "Pricing decision reversed" below).** Originally decided as: the
  invoice form has a printed price column, but it is not used — each
  invoice line is created with no price set, so Odoo fills it in from
  whatever price list already applies to that product and customer.
  That depended on a real Odoo price list existing, which has not
  arrived and is not expected soon, so it was reversed in favor of
  computing each line's actual price directly from that invoice's own
  handwriting instead of waiting on one.
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

**Built (2026-09-27): the desktop "Upload PDF" button on
`review_screen.py`.** Not the long-term intake path (see above for
why), but a quick way to feed a new scan through the pipeline locally
without typing commands, while the Odoo module is still being built.
Clicking it opens a file picker, then runs `pdf_to_images.py` and
`extract_invoice.py` as separate SUBPROCESSES (not direct imports) in a
background thread, so this screen's own fast, model-free startup is
never affected — only clicking the button pays their ~20+ second
model-loading cost, and the window stays responsive while it runs. On
success, the invoice list refreshes and jumps straight to the first
newly-processed page. An empty (or missing) `extractions/` folder is
now a normal starting state instead of the app refusing to start,
specifically so this button has something to fill from a blank slate.

**Also built (2026-09-27): the invoice's handwritten date and printed
invoice number, typed by hand for now.** Both are needed on the
eventual Odoo invoice (see "Use the invoice's own handwritten date" and
"the paper invoice number becomes a label" above), but neither has a
calibrated box yet — the date has none at all, and reading the invoice
number automatically would mean reading PRINTED digits, which the model
has never been tested on. Rather than block on building that first, two
plain text fields were added next to Customer on the review screen: a
required date (`YYYY-MM-DD`, validated before approving) and an
optional invoice number (left blank is a normal, allowed state — see
"the number may not always be visible" above). Both are saved into
`review.json` as `invoice_date` and `paper_invoice_number`, loaded back
in if the invoice is re-opened after being approved once.

**Found (2026-09-27): the WSL Odoo test server's actual login.** The
server itself was already found and documented below ("The Odoo server
this will actually run on"), but not a working login — the `admin_passwd`
in `odoo.conf` turned out to be the database MASTER password (for
create/duplicate/drop operations), not a user login, and none of
Odoo's common defaults (`admin`/`admin`, etc.) worked either. The real
login (Jagbir's own email plus a PIN-style password) was supplied
directly and confirmed working over XML-RPC. It's saved in
`odoo_settings.local.json` at the project root (gitignored — see
`.gitignore` — since it's a real credential) along with the server's
current URL and database name (`New_Raja_Bakery`), for whatever
eventually sends approved invoices to Odoo to read. That URL is WSL's
own IP, which can change if WSL restarts; re-check with
`wsl -d Ubuntu -e bash -c "hostname -I"` if the connection ever stops
working.

**Built (2026-09-28): the desktop "Send to Odoo" button — the first
piece that actually pushes an approved invoice into Odoo.** Three new
files, kept as separate layers the same way the rest of this pipeline
is (a raw connection, business rules, then the UI):

- **[odoo_client.py](odoo_client.py)** — a thin wrapper around talking to
  Odoo over XML-RPC (Odoo's own remote-control interface, built into
  Python already, no extra library needed): logging in, looking up a
  product or a customer by name, checking whether a paper invoice
  number has already been used, and creating a draft invoice. Reads
  the server address, database name, and login from
  `odoo_settings.local.json`, same as before.
- **[match_odoo_products.py](match_odoo_products.py)** — a one-time (but
  safe to re-run) setup script that looks up each of this project's 24
  products by name in Odoo and saves the matching Odoo product ID back
  into `product_rows.json`, so the actual per-invoice push never has to
  search by name at all — it just reads the number this script already
  found. Already run once against the live test server: **21 of 24
  matched** — the 3 that didn't are exactly Twinkies/Cupcakes, Lune Moon
  Cake, and Taki Chips, which Jagbir already said aren't being sold
  yet, so that's expected, not a bug. The lookup is
  case-insensitive and reports when Odoo's own spelling differs
  slightly (found twice: "675g" vs "675G", "Whole Grain" vs "Whole
  grain") — worth knowing about, but not something the code needs to
  fix, since matching still succeeds either way.
- **[send_to_odoo.py](send_to_odoo.py)** — the actual business rules for
  turning one approved invoice into a draft Customer Invoice
  (`account.move`) in Odoo, matching every decision already written
  down above: no stock movement, draft not finalized, one line per
  product row with quantity = Qty − Return and price = that row's own
  derived unit price (Total Price ÷ line quantity, never a price
  list), the paper invoice number saved as the invoice's reference for
  the free duplicate check, and a plain warning note (not a negative
  line, not a credit note) attached to the invoice when a row shows
  more returned than ordered.
- **review_screen.py** gained a "Send to Odoo" button next to Approve &
  Save. Clicking it re-runs the same save-and-validate step Approve &
  Save does (so whatever gets sent always matches what's actually on
  screen, even if something was edited since the last explicit Approve
  click) and then pushes to Odoo in a background thread, the same
  pattern Upload PDF already uses, so the window doesn't freeze while
  waiting on the network. Once sent, the invoice's own Odoo draft
  number is saved back into its `review.json` and the button disables
  itself — sending the same invoice a second time is blocked with a
  clear message rather than silently creating a duplicate draft in
  Odoo, unless someone edits `review.json` by hand to clear it.

**Decided (2026-09-28): a row with a quantity but no readable Total
Price blocks the whole send, rather than going through priced at $0.**
This can happen on a rare row where the price genuinely couldn't be
read (see `unreadable_ink` and similar flags). Sending it through
anyway risked a real invoice line reaching Odoo priced at nothing, with
only a flag (not a hard stop) standing between that and an actual
customer being under-billed if nobody caught it before the draft got
posted. Blocking is more friction but safer by default — the price is
already an editable field in the review screen, so fixing it before
sending is one extra step, not a separate tool.

**Verified end-to-end against the live test server**, using one of the
96 real scanned pages, re-extracted fresh so its Total Price fields were
actually populated (its original `results.json` on disk predated Total
Price reading entirely, from before 2026-09-26): a simulated approval
(one obviously-misread row, `$4000.00` for 16 units, corrected to
nothing — the kind of fix a real reviewer would make) produced a
correct draft invoice in Odoo with the right customer, date, reference
number, and line prices, read back and checked field-by-field rather
than just trusted from a success message. The already-sent guard, the
missing-price block, and the over-returned warning note were each
tested separately too — including confirming the warning note actually
lands as real text on the invoice in Odoo, not just returned by the
Python code. All test draft invoices created during this were deleted
afterward, and the one real invoice's test `review.json` (fake
customer/date, used only to exercise the code) was removed rather than
left looking like a genuine approval.

**Found and worked around (2026-09-28) — the WSL test server is
unreliable to develop against unless something stays connected to it.**
The Odoo/Postgres containers kept dropping mid-test with a plain
"connection refused" error, containers' own logs confirming Postgres
was mid-shutdown at that exact moment. First suspected cause (WSL's
overall idle-shutdown timer, `vmIdleTimeout` in `.wslconfig`) was tried
and set to never expire — that did NOT fix it, confirmed by testing:
the containers still reset within seconds of a gap with nothing
attached, even with that setting disabled. Testing narrowed it down
further: the containers stay up fine for as long as SOMETHING stays
continuously connected to the "Ubuntu" WSL distro (confirmed by holding
a connection open for 2 and 3 minutes straight with no drop), but the
moment everything disconnects — even briefly, between two separate
commands — WSL tears the whole distro down and rebuilds it fresh the
next time anything touches it. This is a distro-level behavior, not the
overall-idle-timeout setting. **The confirmed, working fix:** keep one
WSL session attached the whole time — open a terminal, run `wsl -d
Ubuntu`, and just leave that window open while using the Odoo test
server or the "Send to Odoo" button; nothing needs to be typed in it,
it just needs to stay open. Not a bug in any of this project's own
code, and the eventual Mac deployment won't have this problem at all
(Docker there won't be sitting inside a distro that tears itself down
this way).

**Found and fixed (2026-09-28) — one customer's name was corrupted in
Odoo.** "Zaika's Shawarma" was stored there with its apostrophe turned
into an unreadable character (`Zaika�s Shawarma` — a `�` is what a
computer shows when it tried to read text using the wrong encoding and
hit a byte it couldn't make sense of), almost certainly from an
encoding mismatch when the customer list was first typed into Odoo.
Because `send_to_odoo.py` is designed to automatically create a
brand-new contact for any customer name it can't find an exact match
for (per the "one-off customer" decision above), sending an invoice for
this customer would otherwise have created a *second*, duplicate
"Zaika's Shawarma" contact rather than reusing the real one — quietly
splitting that customer's invoice history across two records. Fixed
directly on the Odoo contact record (renamed to match `customers.json`'s
own spelling exactly) and confirmed the lookup `send_to_odoo.py` uses
now resolves it correctly; nothing in this project's own code needed to
change.

**Not yet built:** the fallback when a product genuinely doesn't have
Odoo match yet (currently blocks the whole send with a list of which
products, matching the missing-price decision above, but not yet tried
against a real case since all 21 sellable products already matched
cleanly); and everything from "Next to build: getting this into Odoo"
below that isn't this button — chiefly, the full Odoo module itself.
This button is explicitly the "quick way to start real value flowing
while the module is still being built" step, not the final home for
this feature.

**Built (2026-09-25): banking verified-correct crops as future training
data.** When a reviewer leaves a field unflagged and uncorrected,
that's a high-confidence signal the model's digit-by-digit read was
right, so those digit crops and the model's own labels are now saved
into a growing bank for retraining later — free labeled data, without
anyone hand-labeling anything new.

- `extract_invoice.py` now saves each individual digit picture
  `digit_reader.py` already segments internally (not just the whole
  Qty/Return cell, as before) into each invoice's own
  `extractions/<name>/digit_crops/` folder, in the exact same format
  `label_tool.py`'s own labeled training crops use (28x28, grayscale,
  ink-as-light-on-dark, padded to square before resizing) — so a banked
  crop can later be used for retraining with no reprocessing.
  `results.json` now also records, per field, the model's predicted
  label for each digit alongside the path to its saved picture
  (`quantity_digit_crops` / `return_digit_crops`).
- `review_screen.py`'s Approve step now copies a field's digit crops
  into `digit_bank/<digit>/` at the project root (gitignored, like
  `invoice_digits/`) whenever a reviewer leaves that field BOTH
  unflagged and unchanged from what the software originally read — the
  signal that the read was actually right. A corrected field is
  skipped, per the reasoning above: if the software had mis-split the
  digits in the first place, the corrected number can't always be
  cleanly matched back onto which individual digit picture was wrong.
  The destination filename is built from the invoice name and the
  crop's own filename, so re-approving the same invoice just
  overwrites the same files rather than piling up duplicates.
- **Found and fixed while building this:** classifying each digit and
  then joining them into one number (e.g. digits "0" and "6" becoming
  the number "06", read as 6) was silently swallowing a spurious extra
  mark read as a leading "0" — with no visible effect on the number
  shown and, in 8 of 32 cases found across the 96-page sample, no
  review flag either. Blank cells already read as 0 through a
  different, correct path (no digits found at all), so a leading "0"
  in front of another digit is never a real quantity on this form —
  it's always some other mark. Left alone, this would have quietly
  banked a wrongly-labeled "0" crop as confirmed-correct training data,
  since the field looked right and a reviewer would have no reason to
  touch it. Fixed with a new flag, `leading_zero_digit`, raised
  whenever this happens, so the field goes in front of a reviewer
  instead of straight into the bank.

## Pricing decision reversed: derive price from the invoice itself, not from Odoo (decided 2026-09-23)

The no-price-list plan above assumed a real Odoo price list per customer
would eventually exist. It has not arrived and does not look like it is
coming soon, and shipping is more urgent than waiting for it. The form
itself turns out to already have what's needed instead.

**What the form actually has.** Every printed row on the invoice has a
"Total Price" column — the rightmost column on the table, currently
unused by any code, filled in by hand whenever that row has a quantity.
`template_calibration.json` has never recorded a box for it (only
`quantity_box` and `return_box` exist per row today).

**The plan: read that column, and divide.** `unit price = Total Price ÷
(Qty − Return)`. That computed price gets set explicitly on each draft
invoice line in Odoo, instead of leaving the price blank for a price
list to fill in.

**What this is for (confirmed by Jagbir 2026-09-24).** The point is to
capture, on the spot, any discount or price change given to that
particular customer on that particular invoice — and that handwritten
price, not the printed catalog price, is the one that must appear on the
Odoo invoice line.

**The price applies to that one invoice only (decided 2026-09-24).** It
is not saved as that customer's standing price in Odoo. Jagbir doesn't
know yet whether a customer's discount stays the same from one invoice
to the next, so this stays per-invoice until that's known.

**Checked against 7 real scans (different invoices, customers, dates)
by hand before committing to this, not just assumed:**

- Every row that had a Qty filled in also had a Total Price filled in —
  no gaps found.
- On most rows, Total Price ÷ (Qty − Return) exactly matches the form's
  own printed catalog price (e.g. 9 × $3.00 = $27.00, 20 × $3.50 =
  $70.00) — so most of the time this just confirms the catalog price
  rather than overriding it.
- On bulk rows — Dempster Bread White/Brown especially, ordered 50-190
  at a time — someone writes a discounted price in small handwriting
  squeezed right next to the printed catalog price (e.g. $2.60 instead
  of $3.00), and the Total Price confirms the discounted price, not the
  printed one. This is the actual payoff: a real negotiated price no
  static price list would otherwise capture.
- Totals are reliably written with an actual decimal point ("130.0",
  "231.90", "265.20") rather than an ambiguous mix of formats — good
  news for teaching the digit reader to find it.
- Deliberately **not** attempting to read that small squeezed-in
  discount number directly — it overlaps printed text in a tiny space,
  a harder read than anything the pipeline does today. Reading the
  clean Total Price box and dividing gets the same answer from an
  easier target.
- Total Price is the rightmost column with nothing after it, so unlike
  the Return-into-Unit-Price problem described below, its box has
  nothing to spill into except the table's own edge.

**Built (2026-09-26): reading Total Price and deriving unit price.**
Every piece described above as "not yet built" now exists — a
`total_price_box` per row in `template_calibration.json`, decimal-point
handling in `digit_reader.py`, `total_price`/`unit_price` in
`results.json`, sanity flags, and matching editable fields in
`review_screen.py`. Some of it needs more real-world checking before
being trusted the way Qty/Return now is — see below for exactly what's
solid and what isn't.

- **Calibrating the Total Price box didn't need a new interactive
  tool.** [calibrate_total_price.py](calibrate_total_price.py) finds
  the column's own left divider automatically (reusing the same
  "how long is a straight printed line" logic `extract_invoice.py`
  already uses to snap cell boxes onto real lines, now shared as
  `alignment.ruled_line_positions`), rather than requiring someone to
  drag a box by hand for all 24 rows the way `calibrate_template.py`
  does for Qty/Return. The column's right edge needs no detection at
  all — it's simply the table's own outer border, since Total Price is
  the rightmost column with nothing printed after it. Run once, after
  `calibrate_template.py` has already recorded Qty/Return; safe to
  re-run, since it only ever touches `total_price_box`. It saves a
  preview image stacking the computed box against real handwriting on
  the first, middle, and last row, specifically so the result gets
  checked against a real scan rather than trusted blind — the same
  discipline `detect_border_corners()`'s own fix (below) was built and
  verified with. Already run once against the shipped reference scan;
  `template_calibration.json` now has `total_price_box` on all 24 rows,
  verified this way.
- **Reading a price means finding a decimal point, which nothing else
  on this form needs.** `digit_reader.segment_price_blobs()` /
  `classify_price()` are new, separate functions alongside the existing
  `segment_digit_blobs()`/`classify_blobs()` — kept separate rather than
  adding a "price mode" flag to the existing ones, so Qty/Return reading
  (already checked against real invoices and trusted) can't be affected
  by a change made for this newer, less-proven field. A decimal point is
  told apart from a leftover digit fragment by being much smaller and
  shorter — but getting the actual size right took checking a real
  example: a first-pass guess at the threshold turned out to be *below*
  where genuine decimal points actually measured (checked against a
  clean real example — a handwritten "29.20" — whose point measured
  0.0025 of the cell's area and 0.18 of its height), meaning real
  decimal points were being discarded as dust before ever being
  considered, not misclassified. The dust-vs-ink floor `segment_digit_blobs()`
  uses for Qty/Return (`FRAGMENT_AREA_FRACTION`) turned out to be too
  high a floor for a price cell's decimal point specifically, so
  `segment_price_blobs()` uses its own lower one
  (`PRICE_DUST_AREA_FRACTION`) instead.
- **Where this stands, honestly — and this is the important part.**
  Fixing that one bug took Total Price from flagging essentially every
  filled-in row (`no_decimal_point` on nearly all of them) to correctly
  reading plain cases with no flag at all — e.g. a handwritten "130.0"
  now reads as `130.0` cleanly, and several rows checked against this
  project's own printed catalog prices came back exactly right (20 ×
  $3.50 = $70.00, 16 × $4.00 = $64.00). But a full run across all 96
  real scanned pages in `invoices/` (not just the 2-page spot check
  that first suggested this was working reasonably) told a worse story:
  **893 of 903 filled-in Total Price fields — 99% — get flagged.**
  `ambiguous_decimal_point` alone accounts for 466 of those (more than
  one small mark in the box looks like it could be the point, now that
  the lowered dust floor lets more small ink through), with
  `no_decimal_point` (109), `possible_merged_digits`, and
  `possible_split_digit` making up most of the rest. Nothing crashed
  and no wrong number reached Odoo silently — every flagged field still
  gets a best-effort value and stays fully editable in `review_screen.py`
  with its crop shown, same safety net as everywhere else in this
  pipeline — but as a REVIEW SIGNAL, the Total Price flag is currently
  close to meaningless: at a 99% flag rate it no longer distinguishes
  "check this one" from "this one's fine" the way Qty/Return's ~39%
  rate does. **Treat every filled-in Total Price field as needing a
  look, full stop, until this gets the same kind of real-measurement
  tuning pass every Qty/Return threshold already went through** (see
  `FRAGMENT_AREA_FRACTION`, `MIN_LONE_DIGIT_HEIGHT_FRACTION`, etc. in
  `digit_reader.py` for what that process looks like — each was tuned
  against hundreds of real measured blobs, not one or two examples).
  Total Price has only been checked against a handful of real examples
  so far.
- **What `results.json` and `review_screen.py` now carry.** Each row
  gets `total_price` (float or `null` — a blank Total Price is never
  assumed to mean $0.00 the way a blank Qty/Return is assumed to mean
  0, since every filled-in Qty row's Total Price box had something
  written in it in every real scan checked so far, so a genuinely blank
  one is unexpected) and a derived `unit_price` (`total_price ÷ line
  quantity`, `null` if that division isn't sensible — flagged instead
  as `total_price_without_quantity` or `quantity_without_total_price`).
  `review_screen.py` shows Total Price as a fourth editable field per
  row (same flagged/pink-background/crop-image treatment as Qty/Return)
  plus a live-recalculating Unit $ display, and Approve banks its
  digit crops the same way Qty/Return's are banked — never the decimal
  point itself, since it isn't a 0-9 class.
- **Not built yet:** a sanity flag comparing the derived unit price
  against each product's own printed catalog price (the plan named this
  when it was first written up, above) — doing that needs the catalog
  prices themselves recorded somewhere (`product_rows.json` only has
  product names today), which hasn't been gathered from Jagbir yet.

**Fixed (2026-09-27): the decimal-point detection was re-tuned against
real measurements, exactly as planned above — but the real finding was
that "tune the thresholds" was the wrong framing entirely.**

The plan above assumed a real decimal point and an ordinary stray mark
would separate cleanly if the area/height cutoffs were measured
properly instead of guessed. They don't. Measured directly across all
96 real scans: 1,214 Total Price cells with no handwriting in them at
all still produce small surviving specks of paper grain in 95% of
them, and those specks measure **the same size** as the one real
decimal point originally used to set the thresholds — same area, same
height, same how-solid-the-mark-is. A handwritten pencil dot and a
fleck of paper texture are simply the same physical size on this form.
No area or height cutoff, however carefully measured, was ever going
to tell them apart, which is why 2026-09-26's session found the
90%+ flag rate no matter how the two constants were nudged.

**What actually separates them is position, not size.** A real decimal
point has to sit somewhere close to the number it belongs to — between
two of its digits (`"27.00"`), or just past the last digit when no
cents were written at all (`"130."`, confirmed on a real scan: the dot
sat 15% of the cell's own width past the last digit). A fleck of paper
grain has no such preference and lands anywhere in the cell, digits or
not. `segment_price_blobs()` now scores every small-mark candidate by
how close it sits to the nearest digit and whether that position is
plausibly part of the number (`DECIMAL_POINT_SPAN_MARGIN_FRACTION`),
and picks the closest one with confidence unless a runner-up is nearly
as close (`CONTENDER_GAP_MARGIN_FRACTION`), in which case it's a
genuine tie and stays flagged — but even then, best-effort now uses
that closest guess instead of dropping the decimal point altogether
(previously, an ambiguous field read as a whole integer with no point
at all, e.g. `1300` instead of `130.0`).

One related bug was found and deliberately left unfixed: the same
step that reconnects a digit's strokes after a printed line is erased
through it can also weld a genuine decimal point onto whichever digit
sits close beside it (confirmed on a real `"130.0"` that read as
`1300`/`no_decimal_point` — the point had merged into the final `0`).
Excluding small, decimal-shaped marks from that reconnect step fixes
this specific case, but broke a DIFFERENT, already-correctly-reading
cell in testing (paper-grain specks that used to get silently absorbed
into a nearby digit — harmless — stayed separate instead and produced
a false tie). That change was reverted rather than shipped half-safe;
the swallow bug remains a known, real, but smaller residual case.

**Measured effect, before/after, same 96 scans, same code path (not
just a spot check):**

| | Before | After |
|---|---|---|
| `ambiguous_decimal_point` | 466 | **256** |
| `no_decimal_point` | 109 | 180 |
| Either decimal flag (a field can only get one) | 575 | **436** |
| **All Total Price fields flagged, any reason** | 885/903 (98.0%) | 878/903 (97.2%) |

The decimal-point-specific problem really is smaller now — 139 fewer
fields have a decimal-related flag, a genuine 24% cut, and the ones
still flagged are flagged more honestly (a real tie, not "more than
one speck of dust exists somewhere in this cell"). The rise in
`no_decimal_point` is mostly that same honesty: fields that used to
guess a decimal position from whatever noise happened to be nearby
now correctly say "no point found" when nothing near the digits
actually qualifies.

**But the overall flag rate barely moved (98.0% → 97.2%), because a
separate, much bigger problem was hiding underneath it the whole
time.** `possible_merged_digits` fired on **726 of 903 filled
fields — 80%** — completely unchanged by the decimal-point fix above
(confirmed: identical count before and after, since none of that logic
touches digit classification). Checked by eye against the saved
digit-crop pictures: most of these were not actually two touching
digits.

**Fixed (2026-09-27, same session): the real cause of
`possible_merged_digits` — a leftover fragment of a row's own printed
line, not touching handwriting at all.** Tracing an actual flagged
field's digit crop back through every stage of cleanup (a purpose-built
debug script that dumps the thresholded, line-isolated, subtracted, and
reconnected picture side by side) showed the true cause: a row's own
top or bottom printed ruled line does NOT always get fully erased by
`_remove_printed_lines` — a fragment of it, still visibly a long, thin
line shape (one confirmed example measured 541px wide but only 31px
tall), survives as leftover "ink." That fragment then does one of two
things, both traced to real examples:

- **It gets misread as a digit all on its own.** `is_digit()` accepts
  a blob by AREA alone with no check on its shape, and a long thin line
  has enough raw pixel area (541 × 31px) to clear that floor even
  though it looks nothing like a numeral — this is what
  `possible_merged_digits` was actually catching most of the time.
- **It gets welded onto a real decimal point candidate.** The same
  stroke-repair step that reconnects a digit legitimately split by an
  erased line doesn't check how far apart two pieces are vertically,
  only whether they overlap sideways — so a decimal point sitting well
  above a stray line fragment near the bottom of the same wide cell
  (over 100px apart vertically, confirmed on a real "130.0" that read
  as "1300"/`no_decimal_point`) gets welded into one blob tall and wide
  enough to look like a garbled digit, swallowing the real point in the
  process.

This turned out to be why the bug is so much worse for Total Price than
for Qty/Return: it's not about being next to the outer border as first
suspected — it's simply that Total Price's cells are much WIDER (dollar
amounts need more horizontal room), so an incompletely-erased line
fragment has more room to accumulate real ink area, and a decimal point
elsewhere in that same wide cell has more room to coincidentally line
up sideways with it.

**The fix, in `segment_price_blobs()` only** (Qty/Return's
`segment_digit_blobs()` is untouched and was verified byte-for-byte
identical across all 2,304 Qty/Return fields before and after):

- `is_digit()` now also rejects anything whose width is more than
  `LINE_RESIDUE_MAX_ASPECT_RATIO` (5) times its height, regardless of
  area — measured across all 96 scans, a blob that only qualifies by
  area (not by being tall enough on its own) has an aspect ratio under
  3 in 90% of real cases, while a line-residue fragment routinely
  measures 8 to 70+.
- `_merge_stroke_fragments()` gained an optional vertical-distance cap
  (`max_vertical_gap`, unbounded by default so Qty/Return's own call
  site is completely unaffected). `segment_price_blobs()` passes
  `STROKE_MAX_VERTICAL_GAP_FRACTION` (0.25 of the cell's height) — well
  above a genuine split-digit repair's own gap, well below the 0.44
  measured on the real bad merge above.

**Measured effect, before/after, same 96 scans, same code path:**

| | Decimal-position fix only | + this fix |
|---|---|---|
| All 2,304 Total Price cells flagged, any reason | 1,012 (43.9%) | **694 (30.1%)** |
| Of filled-in cells, `possible_merged_digits` | 726 | **435** |
| Of filled-in cells, flagged for any reason | 878/903 (97.2%) | 577/600 (96.2%) |
| Filled-in cells (has_digit ink at all) | 903 | 600 |

The drop from 903 to 600 "filled" cells is itself part of the fix, not
a new problem: 303 cells that used to show a bogus number (a stray line
fragment or noise speck misread as a lone digit) now correctly show
blank, because there was no real digit ink there at all. Checked
directly: 192 of those 303 former values were under $1.00 — not a
realistic total for any product on this form — and zero cells went the
other way (a cell that was genuinely blank before never gained a fake
value from this fix). Across every cell where a value changed but both
before and after still show a number, only one case came out unflagged
with a different value than before, and it checks out independently:
`50 units × $2.60/unit = $130.00`, exactly the bulk-discount example
already documented above under "Checked against 7 real scans."

**Also fixed (2026-09-27, same session): the same shape-blindness bug,
found again in the decimal-point check.** While investigating
`possible_split_digit` next (below), the same kind of leftover
printed-line sliver turned up again -- this time passing
`is_decimal_point()`'s area-and-height check (small enough in both) with
no check on its WIDTH, so a 178px-wide, 19px-tall residue fragment could
still masquerade as a decimal point candidate even after the digit-side
fix above. Measured across a 30-page sample: 251 of 2,321 decimal
candidates (11%) were one of these wide slivers, not a real dot or dash.
`is_decimal_point()` now also requires the mark be reasonably
compact -- no wider than `DECIMAL_POINT_MAX_ASPECT_RATIO` (3) times its
own height, comfortably above the widest real dash-shaped point measured
so far (2.6). Verified across all 96 scans: Qty/Return untouched (0
differences across 2,304 fields, as always), and `ambiguous_decimal_point`
drops from 254 to 212 as ties that were never real (one candidate a
genuine point, the other a residue sliver) resolve cleanly instead of
being flagged as a tie.

**Fixed (2026-09-27, same session): `possible_split_digit` was a false-
positive machine, not a real signal, 86% of the time.** It was the
largest remaining flag (320 of 600 filled cells) after the two fixes
above, and pulling real examples showed most of them had a perfectly
correct, catalog-matching read (e.g. `20 × $3.00 = $60.00`) flagged
anyway. The cause: `classify_price()` copied `classify_blobs()`'s
(Qty/Return's) check that a digit far smaller than the TALLEST digit
in the same cell is suspicious -- reasonable for a 1-3 digit quantity
where all digits are naturally similar in size, but not for a 3-6
digit dollar amount, where one genuinely tall, thin digit (a "1" is the
common case) routinely reaches well over double the height of an
entirely normal neighbour. Measured directly across all 96 scans: of
every case this check could fire on, 86% (274 of 320) had no leftover
fragment of ink at all -- the flag was firing purely from that size
comparison, never from genuine leftover ink. `classify_price()` now
flags `possible_split_digit` only on an actual leftover fragment
(`fragment_count`), the same real signal `no_decimal_point` and
`ambiguous_decimal_point` already rely on elsewhere. Verified: zero
Qty/Return impact (`classify_blobs()` untouched), and -- unlike the two
fixes above -- this one changes NO Total Price value at all, only
which of them get flagged, since it purely removes a flag condition
rather than touching segmentation. `possible_split_digit` drops from
320 to 46 fields.

**Fixed (2026-09-27, same session): `possible_merged_digits` was using
a Qty/Return threshold that never fit Total Price's own digit shapes.**
It was the largest remaining flag (435 of 600 filled fields) after the
three fixes above.

Pulling real flagged crops showed the same "0" digit, at 100% model
confidence, tripping the flag purely because this handwriting draws a
whole-dollar amount's trailing "00" cents as one connected cursive loop
rather than two separate zeros -- naturally much wider than any single
digit Qty/Return ever produces. `MAX_SINGLE_DIGIT_ASPECT_RATIO` (1.5)
was tuned specifically from Qty/Return's own narrower digits and never
re-measured for Total Price.

Rather than guess a new number, this was measured the same way as
everything else in this session: for every possible_merged_digits field,
whether its unit price matched that same product's usual price
elsewhere in the 96-scan corpus (a proxy for "this read is probably
actually correct" -- a bakery's catalog price should be stable across
invoices, discounts aside) was checked against its widest digit's real
aspect ratio. The two groups' aspect-ratio distributions were nearly
identical up to about 3 -- meaning the ratio carries almost no signal
below that -- and only genuine outliers separated past it. At a
threshold of 5 (`MAX_SINGLE_PRICE_DIGIT_ASPECT_RATIO`, Total-Price-only,
Qty/Return's own constant and flag untouched): **0 of 129
"probably correct" reads still trip the flag**, while 97% of the
"probably not" group stops tripping it too -- what's left is the
genuinely extreme tail (aspect up to 11.8), not ordinary correct reads.

Verified across all 96 scans: zero Qty/Return impact, zero Total Price
VALUES changed (flags only, like the split-digit fix). 435 → **12**
fields. All 12 remaining cases already carry other flags too
(`low_confidence`, `no_decimal_point`, `total_price_without_quantity`)
-- mostly noise read on rows with no real quantity ordered at all --
so nothing was relying on this flag alone to get caught.

**Fixed (2026-09-27, same session): most `ambiguous_decimal_point`
ties didn't actually change the answer, but were flagged anyway.**
Pulling real flagged cells split into two different pictures: some
genuinely had two well-separated marks both plausibly the point (real
ambiguity), but a meaningful share had two candidates sitting in the
SAME gap between the SAME two digits -- meaning either choice produces
the identical number, so there was nothing to actually be unsure about.

Measured directly: for every genuinely-tied cell across all 96 scans,
whether choosing the first vs. second tied candidate changed
`decimal_before_count` (how many digits land before the point) was
checked. **37% of ties (78 of 212) made no difference at all** -- both
candidates placed the point in the same spot. `segment_price_blobs()`
now only keeps the `ambiguous_decimal_point` flag when the two tied
candidates would actually split the digits differently; when they
agree, the shared answer is used with no flag, exactly as if there'd
been no tie to begin with.

(A related, narrower idea was also checked and mostly ruled out: many
early examples looked like a single decimal point that fragmented into
two tiny touching pieces during thresholding. Measured across all
tied cells, though, only 17% of the pairs were actually that close
together -- the median tied pair sat 21% of the cell's width apart, too
far to be one fragmented mark. The tie-doesn't-matter fix above covers
far more cases than chasing that narrower one would have.)

Verified across all 96 scans: zero Qty/Return impact, zero Total Price
values changed (by construction -- this only resolves ties where both
choices already agreed on the value). `ambiguous_decimal_point` drops
from 212 to **134** fields.

**Fixed (2026-09-27, same session): `unreadable_ink` was firing almost
entirely on rows where nothing was ordered at all.** It was the second-
largest remaining flag (85 fields). Checked directly: every field
sampled by hand had `line_quantity == 0` -- a row where no product was
recorded as ordered at all. Measured across all 96 scans, that held for
**93% of them (79 of 85)**.

This one isn't a segmentation bug like the others -- the ink genuinely
couldn't be classified as a digit, and flagging that is technically
correct. The fix is a row-level business-logic point instead, made in
`extract_invoice.py` rather than `digit_reader.py`: a row with nothing
ordered (`line_quantity == 0`) never gets a unit price regardless of
what's in its Total Price box (the division is already gated on
`line_quantity > 0`), so ambiguous leftover ink there can never actually
affect anything downstream. `total_price_without_quantity` (kept,
untouched) already covers the genuinely meaningful sibling case -- a
CONCRETE price was read despite no recorded order, which is worth a
look; `unreadable_ink` on a `line_quantity == 0` row is just paper
grain with nothing at stake.

Verified across all 96 scans: zero Qty/Return impact, zero Total Price
or unit-price values changed (pure flag suppression, gated only on a
field already computed for other reasons). `unreadable_ink` drops from
85 to **6** -- exactly the cases where a real order (`line_quantity > 0`)
had a genuinely unreadable price, which stay flagged as they should.

**Where a future session should pick this up (left off 2026-09-27).**
Six real bugs fixed this session take Total Price's overall flagged
count — across ALL 2,304 cells, not just filled ones — from effectively
100% at the start down to **14.5%**, and flagged-of-filled from 97.2%
down to **49.2%** (unchanged by this particular fix, since it only
touched blank cells). Within reach of Qty/Return's own ~39%. Priority,
in order:

1. **`ambiguous_decimal_point` (134 fields) is still the largest flag
   left**, now representing genuine ambiguity (two well-separated
   candidates that really would give different answers) rather than
   ties that didn't matter. Worth checking a real sample to see whether
   these are truly unresolvable from the image alone (in which case the
   flag is doing its job correctly) or whether another positional signal
   -- e.g. preferring whichever candidate sits closer to the cell's
   vertical middle, where a decimal point usually sits relative to a
   digit's baseline -- could resolve more of them.
2. Of the remaining `total_price_without_quantity` (77), `low_confidence`
   (73), `no_decimal_point` (60), `possible_split_digit` (46), and
   `possible_merged_digits` (12): **`low_confidence` is now checked and
   confirmed to be a legitimate signal, not a bug** (see "Checked
   (2026-09-28)" below), and **`total_price_without_quantity` has a
   real, confirmed root cause but no safe fix yet** (a printed line
   surviving removal and getting misread as a digit — see "Investigated
   (2026-09-28)" below for what was tried and why straightness, not
   size or density, is the next avenue worth trying). `no_decimal_point`,
   `possible_split_digit`, and `possible_merged_digits` are still
   unchecked. **Already checked, ruled out**: the `unreadable_ink` fix's
   own trick -- checking whether a flag correlates with
   `line_quantity == 0` -- was tried against all five. `possible_merged_digits`
   does correlate (83%, 10 of 12), but every one of those 10 already
   carries `total_price_without_quantity` too, so the row stays flagged
   either way -- suppressing it wouldn't reduce how many rows need
   review, only shorten their flag lists. The other four don't
   correlate strongly enough with `line_quantity == 0` (2%-41%) for this
   particular trick to apply at all. A real fix for the three still-
   unchecked flags would need actual per-flag investigation (real crops,
   real measurement), the same as every fix earlier in this session --
   not assumed from this shortcut.
3. Once Total Price's flag rate is actually informative, the remaining
   build order from before still holds: a quick "send to Odoo" button
   on `review_screen.py`, then the full Odoo module (see "Decisions"
   above for why that order).
4. The catalog-price sanity flag noted earlier is worth adding
   alongside step 1, if/when Jagbir provides catalog prices per
   product — it would independently help spot a bad Total Price read
   too, not just serve as its own feature. (This session's aspect-ratio
   measurement already leaned on a rough version of this idea --
   product-modal pricing across the corpus -- worth formalizing.)

**Fixed (2026-09-28): `ambiguous_decimal_point` cut from 134 to 86 by
using real 2D distance instead of horizontal-only distance to judge
which candidate is genuinely close to a digit.** Picking up item 1 from
the list above. The idea first proposed there -- preferring whichever
candidate sits closer to the cell's vertical middle -- was tried first
and measured directly against all 134 real ties: it carried essentially
no signal at all (the chosen candidate and the discarded one had nearly
identical vertical positions relative to the digits, 0.518 vs 0.543 of
the digit span on average, well within noise). That idea was dropped in
favor of a different one, found while digging into why the first one
failed.

The real cause: `gap_to_digits` (the function that measures how close a
candidate mark sits to the nearest digit) only ever measured HORIZONTAL
distance. A mark sitting directly above or below a digit -- with no real
relationship to it at all, like a fleck of paper grain, or bleed-through
from the row above -- registers as "0 gap" purely by sharing that
digit's left-right position, even though it may be dozens of pixels away
vertically. That let a genuinely close, genuinely correct decimal point
get treated as merely "tied" with an unrelated speck that only looked
close by this flawed, one-dimensional measure.

Measured with a true 2D (both horizontal and vertical) distance instead,
across all 134 real ties: in 48 of them (36%), the candidate already
being picked (by the existing horizontal-only logic) turns out to be
genuinely touching a digit in every direction, while the tied "rival"
sits nowhere close by any reasonable margin (a median 0.162 of the
cell's own height away, far outside where a real point could plausibly
be). Checked by eye on real crops, not just the numbers: a handwritten
"48.00" whose real decimal point sits flush against the "8", tied
against a stray fleck of paper grain sitting well below the row
entirely with nothing to do with the number at all. Those 48 are a
confident, correct answer, not a genuine tie. Only 13 of 134 had BOTH
candidates genuinely close to a digit -- those are real ambiguity and
correctly stay flagged.

The fix, in `segment_price_blobs()` only (new constant
`DECIMAL_POINT_TOUCH_DISTANCE_FRACTION`): the existing tie check is
unchanged, but no longer flags a tie when the closer candidate is truly
(2D) touching a digit and the further one plainly isn't. It never
changes which candidate is used or what value gets read -- only whether
it gets flagged -- matching how every other flag-only fix this project
has made was done. Verified across the full 96-scan corpus, comparing
every single field before and after: **zero Qty/Return values changed,
zero Total Price or unit-price values changed, every other flag's count
identical** (`total_price_without_quantity`, `low_confidence`,
`no_decimal_point`, `possible_split_digit`, `possible_merged_digits`,
`quantity_without_total_price`, `unreadable_ink` all unchanged) --
`ambiguous_decimal_point` alone drops from 134 to **86**, and zero new
false flags appeared anywhere.

**What's left, updated:** `ambiguous_decimal_point` (86) is still the
largest single Total Price flag, but the remaining cases are the ones
already checked and found to be genuine two-candidate ambiguity, not
solvable by position alone -- a further improvement here would need a
different kind of signal (ink darkness/solidity, maybe, or accepting
that some of these are genuinely unresolvable from the image and simply
need a person to look). Priorities 2-4 from the list above are
unchanged and still open. Total Price's overall flagged count (all
2,304 cells) is now **12.6%** (291 cells, down from 14.5%), and
flagged-of-filled is down to **42.2%** (253 of 600, down from 49.2%) --
getting closer to Qty/Return's own ~39%, though the remaining flags are
a harder, more genuinely-ambiguous residue than the ones already
resolved.

**Checked (2026-09-28), picking up priority 2 from the list above:
`low_confidence` (73) is a legitimate signal, not a bug — no fix
made.** Unlike every flag fixed so far this session, this one doesn't
show the "correct read wrongly flagged" pattern. Checked against each
product's own most common (modal) unit price elsewhere in the corpus,
the same technique that confirmed the `possible_merged_digits` fix: of
the 30 fields flagged for `low_confidence` alone, 24 (80%) have a unit
price wildly different from that product's normal price — some over
$1,500/unit, plainly wrong reads — while the model's own confidence
score correctly tracks that (these are exactly the cells it read as
low-confidence). This is the opposite result from the earlier fixes:
the flag is correctly catching real misreads, so nothing was changed.

**Investigated (2026-09-28), picking up the other half of priority 2:
`total_price_without_quantity` (77) — a real, confirmed bug found, but
not yet safely fixable.** 62 of 77 cases (81%) are rows where NOTHING
was ordered at all (Qty and Return both 0) yet a small dollar value got
read anyway. Traced one all the way through: the cell itself has no
handwriting in it whatsoever, but the software still read "$8.00" from
it, at 99.9% model confidence. The actual cause, confirmed visually by
dumping the image at every processing stage: a printed vertical column
divider line, slightly crooked (these scans sit about 1.4 degrees off
square, same tilt documented elsewhere in this file), survived the
line-removal step and happened to run mostly inside this cell's own
row band. `is_digit()` accepts anything tall enough to plausibly be a
full-height digit, with no UPPER limit — so a printed line spanning
more than two rows' worth of height (measured: 2.128x the cell's own
height in this example) still qualifies, and the model confidently
(if nonsensically) read its thin, mostly-empty shape as a digit once
squeezed down to the small square image it classifies from.

**Why this isn't fixed yet: neither an obvious height limit nor an ink-
density limit safely tells this apart from a real, unusually large
digit.** Measured across all 96 scans: 58 of 2,176 digit-classified
blobs in Total Price cells are "tall" (over 1.3x the cell's own
height), but plenty of those are genuinely large, correctly-read digits
sitting inside a wide, sloppily-written cell -- not residue. Checked
whether how much of its own bounding box a blob actually fills (a
straight line drawn inside a bounding box sized to its diagonal reach
should fill much less of that box than a solid digit stroke) separates
the two: it helps (tall blobs' median fill is 0.153 vs 0.231 for normal
ones) but the populations still overlap too much to draw a safe line
(normal digits' own 10th percentile, 0.138, already sits inside the
tall group's typical range). A blanket cutoff on either measurement
risks turning a real, correct digit read into a false `unreadable_ink`
-- exactly the kind of regression this session has been careful to
avoid everywhere else. **Worth trying next, not yet attempted:**
checking the blob's actual straightness (fitting a line through it and
measuring how far the real pixels stray from that fit) rather than its
size — the same "long and straight, which handwriting never is"
reasoning `ruled_line_positions()` already uses to find printed lines
in the first place, just applied after the fact to a blob that slipped
through. Measurements from this session (all 2,176 blobs' height,
width, and box-fill-density, tagged by scan and row) haven't been kept
in the repo, per this project's usual practice for scratch analysis
data — a future session picking this up would need to re-run the same
check, not dig through this repository for it.

**Tried and rejected (2026-09-29): rejecting tall, very straight pieces
of ink as leftover printed line.** This was the "check straightness"
idea listed just above. Straightness was measured for every piece of
ink taller than 1.3 times its box (58 of them across the 96 scans), and
about 10 were much straighter than the rest, with a clean gap before
the next one, so a rule using that gap looked promising. A full 96-scan
before/after run showed it is not safe. It fixed the one row it was
built for (a spurious $8.00 in an empty box on
`NEW RAJA BAKERY LTD. (1)_page002`, row 5), but it also broke 9 rows
that were being read correctly: those tall, straight pieces of ink were
real handwritten "1"s, so for example "145" became "45" and "1125.4"
became "125.4", each gaining a flag but the wrong number too. A
handwritten "1" is simply as straight as a printed line. The rule was
reverted and no code changed. Only 1 of the 77
`total_price_without_quantity` cases would have been fixed anyway, so
it would not have moved that flag much even if it had been safe.
Shape alone cannot separate the two.

**Also tried and rejected (2026-09-30): using where the piece of ink
sits in the box.** The idea was that a printed column divider would sit
right on the box's left or right edge, while a real "1" sits inside.
Measured across all 58 tall pieces of ink in the 96 scans, it does not
work: the one confirmed bad case sat 5.6% of the box's width in from its
left edge, and the real "1"s sat 3-9% in, because a number's first digit
naturally starts near the left edge. There is no gap between them. Both
shape and position have now been ruled out, so the remaining
`total_price_without_quantity` cases are best left to the review screen,
where they are already flagged and editable.

**Fixed (2026-09-30): the decimal point was often picked wrongly because
the software chose whichever small mark sat nearest a digit.** After the
two ideas above were ruled out, every remaining Total Price flag was
checked against each product's usual price elsewhere in the 96 scans
(a stand-in for "this read is probably right"). Every flag except
`ambiguous_decimal_point` had almost no reads matching the usual price
(0-14%), so those flags are doing their job. For
`ambiguous_decimal_point`, looking at real crops showed why it was
wrong so often: clear, well-written prices like "36.00", "27.00" and
"80.00" were coming out as 0.36, 2100 and 800.0. The real decimal
point is a bigger mark with space around it, sitting between the
dollars and the cents, but a speck of paper grain, or a piece of one of
the small handwritten cents zeros, often sat closer to a digit, and
"closest to a digit" was the rule. What separates them is the layout
of the number: among reads that matched the usual price, 82% had
exactly two digits after the point and 14% had one. `segment_price_blobs()`
now ranks candidate points by that first (a point ahead of every digit
ranks last, since no total on this form is under $1) and by distance
second, and only raises `ambiguous_decimal_point` when the layout
cannot break the tie.

Measured across all 96 scans: Qty/Return unchanged (0 differences);
91 Total Price values changed; 27 of them now match the product's usual
price and 0 that used to match stopped matching; fields matching the
usual price rose from 159 to 186; `ambiguous_decimal_point` fell from
86 to 1; every other flag count identical. Bulk bread rows now read
$2.50-2.70 a unit, in line with the documented bulk discount. **The
cost:** 75 of the changed values now have no flag, and a rough check
found about 15 more wrong reads that used to be flagged (only because
of the tie) and now are not. Most of those are digit errors the model
makes anyway (a slanted "7" read as "1", an open-top "9" as "4").
This makes the catalog-price sanity flag (see "Not built yet" above)
more valuable, since it would catch exactly these. It still needs
each product's printed price recorded from Jagbir.

**How to check a future change the same way:** run
`python extract_invoice.py invoices/*.png --output-dir <new folder>`
(about 10 minutes for all 96 scans), then
`python compare_extractions.py <old folder> <new folder>`
([compare_extractions.py](compare_extractions.py)). It prints how many
Qty/Return values changed (should be 0 for a Total Price change), how
many Total Price values changed to or from matching the product's usual
price, and each flag's count before and after. The `extractions_full/`
folder in the project (gitignored, real invoice data) holds the results
from before the 2026-09-30 decimal-point fix, if a baseline is needed.

**Built (2026-10-06): the catalog-price check.** Each product's printed
"Unit Price" from the form is now saved as `catalog_price` in
`product_rows.json` (read off the reference scan by eye; the three
unsold products have none). `extract_invoice.py` flags a row
`unit_price_far_from_catalog` when its derived unit price is under 0.7
or over 1.1 times that printed price. The limits come from the 96 scans:
real discounts sit at 0.87-0.99 times the catalog price and almost
nothing legitimate lands above it, while wrong reads (a misplaced
decimal point, a misread digit) usually land far outside, like $0.36 or
$21.00 on a $3.00 product. The printed prices were read by eye, then
cross-checked against Odoo's own list prices the same day: all 21
match exactly.

Measured on all 96 scans against the 2026-09-30 baseline: Qty/Return
unchanged (0 differences), no Total Price or unit-price values changed
by this check (it only adds a flag), and every other flag's count
identical. The new flag fires on 243 of 492 priced rows; for 171 of
those it is the only flag, so those are rows that used to reach review
with nothing marked. The cost: about 55% of priced rows are now flagged
(269 of 492), up from a rate that was too low to trust. A discount below
70% of catalog price will also be flagged, which is a cheap extra look
for a real person to confirm.

**Still open after the 2026-09-30 session:** a leading digit that is cut off at
the box's left edge (seen on one real crop, a "7" partly missing);
`no_decimal_point` (60), `possible_split_digit` (46) and
`total_price_without_quantity` (77) still have no safe fix; the
catalog-price flag above; and the Odoo module.

## The Odoo server this will actually run on (checked 2026-09-23)

Found by checking the machine directly rather than asking Jagbir to
look, since it turned out to be reachable from here (WSL on the same
Windows machine):

- **Odoo 19 Community Edition**, official `odoo:19.0` Docker image, not
  18 as first assumed — running via `docker-compose` in
  `~/newcompany-odoo` inside the "Ubuntu" WSL distro, alongside
  `postgres:15`. Custom modules mount at `/mnt/extra-addons` (already
  wired into `addons_path` in `config/odoo.conf`).
- **Python 3.12.3** inside that container, Ubuntu 24.04, x86_64.
  Torch, OpenCV, numpy, and PyMuPDF are all missing from it — only
  Pillow is present — so a custom Docker image (this project's
  dependencies added on top of `odoo:19.0`) is needed, not the stock
  image.
- **One page took about 73 seconds to read** when first timed inside
  this container (CPU-only). Since sped up — see "Speed-up (2026-09-24)"
  below: on the Windows machine a page now takes about 3 seconds once
  the software is loaded. It hasn't been re-timed inside the Docker
  container yet. Reading should still happen in the background (a cron
  job or queue) rather than inside the upload request, so a large PDF
  can never hit Odoo's roughly 2-minute web request timeout.
- **Final deployment target is a Mac** (onsite), chip unknown (Apple
  silicon vs. Intel) as of this writing — the Docker build needs to
  work on both until that's confirmed, and it isn't yet confirmed the
  official Odoo image even ships an Apple-silicon build.
- The test server's admin and database passwords are still whatever
  ships as the stock `odoo:19.0` image's out-of-the-box default —
  fine for local testing, but must be changed to real secrets before
  anything real goes on this server.
- **Decided:** the Odoo module becomes its own project, `bakery_odoo`,
  containing a Dockerfile (`odoo:19.0` plus this project's
  dependencies), a `docker-compose.yml`, and the module itself — this
  repository (`bakery_ml`) gets installed into that image as a package
  rather than copied in, per the shared-engine decision above. This
  repo may need light packaging changes (e.g. a `pyproject.toml`) to be
  installable that way, since it's currently loose top-level scripts
  that import each other directly.

**Status as of 2026-09-28:** the desktop "Send to Odoo" button described
above is built and verified end-to-end against the live test server —
real value can now flow from an approved invoice into an Odoo draft
invoice, ahead of schedule relative to the plan below (which had it
waiting on Total Price's flag rate coming down first). Two things found
while building it are now resolved: the WSL test server's own
instability (worked around with a confirmed fix — keep one WSL session
open while working, see "Found and worked around" just above) and one
customer's corrupted name in Odoo (fixed directly on the Odoo record).
Separately, the same session picked up priority 1 from the "Where a
future session should pick this up" list below:
`ambiguous_decimal_point` cut from 134 to 86 (see "Fixed (2026-09-28)"
above that list) by measuring true 2D distance to the nearest digit
instead of horizontal-only distance — verified against all 96 scans
with zero value changes anywhere, only flags. What follows below is
slightly out of date as a result — written 2026-09-27, before either of
2026-09-28's fixes:

saving digit pictures for retraining
(2026-09-25) is done. Total Price reading / unit price derivation
(2026-09-26) is built; seven real bugs behind its flag rate were found
and fixed the same session (2026-09-27) — decimal-point detection
re-tuned to use position instead of size, a printed-line-residue bug
(misread as a stray digit, or welded onto a real digit or decimal
point, fixed in two parts), a flag (`possible_split_digit`) firing on
correct reads 86% of the time, a second flag (`possible_merged_digits`)
using a Qty/Return-only threshold that never fit Total Price's own
wider digit shapes, a third flag (`ambiguous_decimal_point`) firing on
ties that didn't actually change the answer 37% of the time, and a
fourth flag (`unreadable_ink`) firing 93% of the time on rows where
nothing was ordered at all (a row-level business-logic fix in
`extract_invoice.py`, not a pixel-level one). Together these cut Total
Price's overall flagged count, across every cell not just filled ones,
from effectively 100% at the start of the session to **14.5%**, and
flagged-of-filled from 97.2% down to **49.2%** — real, verified progress
at every step (each fix checked against a full 96-scan before/after
run, with zero Qty/Return impact throughout, and the last four fixes
changing zero actual Total Price or unit-price values, only which ones
get flagged). Within reach of Qty/Return's own ~39%.
`ambiguous_decimal_point` (now representing genuine ambiguity rather
than moot ties) is still the largest remaining flag and the next thing
worth a look (see "Where a future session should pick this up," above).
The desktop "Send to Odoo" button mentioned as still-ahead here was
actually built the next session (2026-09-28) without waiting on this —
see the status note just above.

## Information needed from Jagbir to finish the build (listed 2026-09-24, answered 2026-09-25)

Everything the remaining build (Total Price, saving digit pictures for
retraining, the Odoo module, setting it up on the Mac) needed that
couldn't be worked out from the code or the scans alone. Kept here,
answers and all, so a future session knows what's settled. Already on
hand before this list was even written: the 24 product names in form
order (`product_rows.json`), the 39 regular customers
(`customers.json`), and the Odoo test server's setup (see "The Odoo
server this will actually run on" above).

**Passwords and logins never go in this file or anywhere in git.**
Where one is needed, it should be given in the chat or put in a local
settings file that git ignores.

### 0. Decided (2026-09-25): the paper invoice number becomes a label, not Odoo's own invoice number

Each paper invoice has its own number printed in the top right corner
(e.g. "Invoice: 21157"), always in the same spot on every invoice book
(confirmed by Jagbir). **Decision: option (a)** — attach it to the Odoo
invoice as a label (e.g. Odoo's built-in reference field, or a tag), so
the Odoo invoice can be searched by the paper number and matched back
to the paper copy. Odoo keeps its own separate numbering
(`INV/2026/00001`-style), which its accounting features expect — the
touchier option (b), replacing Odoo's own numbering with the paper
number, was not chosen.

This still gives the free duplicate check discussed: if an invoice with
the same paper number has already been approved, warn before creating a
second one.

**Reading the number:** it's printed type, not handwriting. The digit
model was trained on handwriting, so how well it reads printed digits
hasn't been tested. It would need a new box added to the calibration
for the number's position, and it must be measured before being
trusted. The review screen should always show the number, with a
picture of it, for the reviewer to confirm or type in, the same as the
other fields.

**New open concern, raised by Jagbir 2026-09-25: the number may not
always be visible.** Even though it's always printed in the same spot,
Jagbir suspects it might sometimes be cut off, obscured, or missing on
a given scan (e.g. a crooked scan cropping it out, or a faded/damaged
copy). Since the duplicate check and the searchable label both depend
on this number being read, the pipeline needs a defined fallback for
when it can't be read: at minimum, flag the invoice for the reviewer to
type the number in by hand (same "always shown, reviewer confirms or
types in" treatment already planned above), and the duplicate check
simply can't run for that invoice until a number is entered. Not
designed in detail yet — worth revisiting once the box is actually
added to calibration and real scans can be checked for how often this
happens.

Whether the numbers are unique across all invoice books in use is still
unconfirmed — needed before the duplicate-check logic can be trusted.

**Checked (2026-09-28): Jagbir's "may not always be visible" concern,
confirmed and measured against all 96 real scans — and a decision made
as a result.** Before building a calibrated box and testing the model
on printed digits (both real work, and both previously left as "not
designed in detail yet" above), the actual scans were checked by eye
first, the same discipline every other real decision in this project
has followed. The top-right corner was cropped out of every one of the
96 real pages and looked at directly:

| Invoice batch | Clean & readable | Cut off, partly visible | Missing entirely |
|---|---|---|---|
| NEW RAJA BAKERY LTD. (1) | 24/24 | 0 | 0 |
| NEW RAJA BAKERY LTD. | 24/24 | 0 | 0 |
| PH 416-727-0623... | 4/24 | 9/24 | 11/24 |
| subzi Mandi chard. | 0/24 | 1/24 | 23/24 |
| **Total** | **52/96 (54%)** | **10/96 (10%)** | **34/96 (35%)** |

Two of the four invoice batches are entirely fine. The other two are
badly affected — in the "subzi Mandi chard." batch, 23 of 24 pages have
no trace of the printed number at all: the scan itself starts partway
down the page, so the letterhead and the "Invoice:" line are simply not
part of the digital file, not just faint or blurry.

**Ruled out first: this is not a bug in `pdf_to_images.py`.** If the
rendering step were cropping pages wrong, every page from the same
source PDF would come out the same size. Checked directly: pages from
the very same PDF come out different pixel sizes (e.g. one
"PH 416-727-0623..." page rendered at 8257×10043, another from the same
file at 7154×8589). That only makes sense if the ORIGINAL scanned pages
already differ in size before this project ever touches them — a real,
physical scanning inconsistency (the paper positioned differently in
the scanner from page to page), not something fixable by changing how
this project renders PDF pages into images.

**Decision: build the auto-increment assist (below) instead of an
automated printed-digit reader, for now.** Even a perfectly accurate
printed-digit reader could only ever help on the 54% of pages that have
the number at all — the other 46% have nothing to read regardless of
how good the reader is. That changes the earlier plan's priority
entirely: the real bottleneck is the scanning process itself (worth
raising with whoever does the scanning, if the top margin can be made
more generous), not the absence of automated reading. Manual entry
already exists in `review_screen.py` and stays the primary path; see
"Built (2026-09-28): invoice-number auto-increment assist" below for
the cheap improvement that was built on top of it instead.

**Built (2026-09-28): invoice-number auto-increment assist.** Real
invoice numbers climb by exactly 1 per page within one day's scanned
batch almost all the time (confirmed against the real numbers visible
in the check above), so a blank invoice-number field on the review
screen is now pre-filled with a guess instead of starting empty:
`_suggest_invoice_number()` walks backward through the SAME batch's
earlier pages (recovering "batch" and "page number" from the
`..._pageNNN` filename `pdf_to_images.py` already gives every page) and
uses the nearest earlier page that has a saved, purely-numeric invoice
number, adding back however many pages separate them. Still just an
editable starting point, never trusted outright -- the same check
above found two real gaps (a skipped number) in a single 24-page batch,
so a reviewer always sees an ordinary, correctable text field, exactly
like every other guess on this screen, never something presented as
already confirmed.

Walking backward past more than just the immediately-previous page
matters for real cases already seen: if that one page was itself left
blank (genuinely unreadable) or hasn't been approved yet at all, the
guess still reaches back to the nearest one that DOES have a usable
number, rather than breaking for every page after a single gap.
Verified directly (not just read through): a synthetic batch with a
blank page and a missing review.json in a row still produced the
correct running guess for every following page.

### Answers from Jagbir (2026-09-25)

Every question below has now been answered, so the build can proceed.
Kept under the same headings and numbers as before, for reference.

**Needed before starting:**

1. & 2. **The 24 products and 39 customers are already set up in Odoo,
   spelled exactly as in `product_rows.json` and `customers.json`.**
   Every product also already has its barcode filled in (the same one
   printed on the form, e.g. `#068721002512`) — so matching a form row
   to its Odoo product should go by barcode, not by name, since names
   are more likely to be spelled slightly differently between the form
   and Odoo.
3. **If a reviewer types a one-off customer who isn't already in
   Odoo, Approve should create them as a new Odoo contact
   automatically** — not refuse and wait for someone to add them by
   hand.
4. **Do not create any new products or contacts on the test server**
   while building — use only the ones already there. Test invoices and
   drafts using those existing products/contacts are fine and expected.
5. **Odoo's Invoicing/Accounting app is already installed and set up**
   on the test server.

**Money and tax:**

6. **Total Price = (Qty − Return) × price**, confirmed. That's the
   formula the pricing plan above already assumed, so `unit price =
   Total Price ÷ (Qty − Return)` stands as designed.
7. **A row with more returned than delivered should never actually
   happen** — if it does, it means something is wrong further back
   (a misread, or a real paperwork problem) that needs fixing right
   away, not a normal case to quietly compute an answer for. **Decided
   2026-09-25:** when it does happen, add a plain warning comment on
   that draft invoice — not a negative invoice line, and not a formal
   Odoo credit note (a separate document Odoo has for genuine refunds)
   either. Just a visible note flagging that this row's numbers don't
   add up and need checking, since it's meant to be treated as urgent.
   (The `return_exceeds_quantity` flag added 2026-09-24 already catches most
   of these before Approve; this decision covers the rare one that
   still gets approved anyway.)
8. **Ignore HST/tax entirely for now.** Invoice lines go in exactly as
   the paper copy shows, with no tax added.
9. **The three rows with no printed price (Twinkies/Cupcakes, Lune
   Moon Cake, Taki Chips) aren't being sold yet** — nothing needs to be
   built for them for now.
10. **Nobody handwrites extra products into the blank rows at the
    bottom of the table** — not a case the software needs to handle.
11. **The software doesn't need to read the Subtotal/HST/TOTAL boxes
    at the bottom of the form.** The Odoo invoice's total should just
    be the sum of the individual product lines it creates from the 24
    printed rows — no separate cross-check against a handwritten total.

**The paper invoice's other details:**

12. **Use the invoice's own handwritten date** for the Odoo invoice,
    not the date it happens to be approved — these two dates won't
    always be the same day.
13. **Every invoice starts as unpaid in Odoo**, until someone changes
    that by hand later — nothing needs to be read from the paper's
    Paid/NOT PAID box.

**How scans arrive:**

14. **Invoices are scanned and brought in as PDFs**, not phone photos
    — good, since a phone photo is what caused the one bad
    border-detection case described above.
15. **One invoice per page, but a day's batch is a single PDF holding
    several invoices side by side (one per page)** — usually around
    8-14 invoices a day. `pdf_to_images.py` and `extract_invoice.py`
    already handle a multi-page/multi-file batch this way.
16. **Scans are always right-side up** — no sideways/upside-down
    handling needs building.

**Who uses it, and from where:**

17. **One person actively handles invoice review.**
18. **Reachable only on the bakery's own Wi-Fi is acceptable if that
    turns out to be the only option, but reachable from outside too
    (home, on the road) is preferred** if it's not too much extra
    work.

**The onsite Mac:**

19. **The exact model isn't known yet** — only that it's a MacBook.
    Chip, macOS version, RAM, and free disk space still need checking
    on setup day (Apple menu → About This Mac).
20. **The Mac won't stay switched on all the time**, so Odoo won't
    always be reachable — but there should be a clickable shortcut/icon
    so the person using it never has to open the Terminal to start it.
21. **This MacBook will hold the real, day-to-day Odoo** once handed
    over — the current WSL test server on the Windows machine is only
    for building and testing, and nothing from it needs to be migrated
    across.
22. **No firm preference on where backups go** — Claude should pick
    whatever's sensible, with the constraint that it must not slow the
    Mac down or use excessive disk space for daily use.
23. **Someone will be onsite with the Mac's admin password on setup
    day.**

**Decisions:**

24. **Build order: left to Claude's judgment.** Going with the order
    already suggested — saving digit pictures for retraining, then
    Total Price, then a quick "send to Odoo" button on the existing
    desktop review screen (so real draft invoices can start within
    days), then the full Odoo module — since it gets real value
    flowing earliest and proves out each piece before the bigger
    module build.
25. **Yes, keep the small digit crops from approved invoices for
    retraining**, as planned.

## Speed-up: reading a page went from ~11 seconds to ~3 (2026-09-24)

**The result.** On this Windows machine (12 processor cores), reading
one invoice page went from an average of **10.8 seconds to 2.9
seconds** once the software is loaded, measured on the same 24 real
pages covering all four invoice PDFs. **The answers did not change at
all.** Every one of the 1,176 output files (24 `results.json` files
plus 1,152 box pictures) was compared byte for byte against the old
code's output, and all were identical. So nothing about accuracy or
flagging needs re-checking because of this.

**What changed, in plain terms:**

1. **Checking which box each piece of ink belongs to**
   (`_owned_ink` in `digit_reader.py`). To decide whether a piece of
   ink belongs to this box or the row above or below, the old code
   looked at each piece separately, and each time it re-scanned the
   whole picture from the start. A box with 30 pieces meant 30 full
   scans. It now counts all the pieces in one scan. This was the single
   biggest waste, about 7 seconds per page.
2. **Finding the table border** (`_fit_line_in_band` in
   `alignment.py`). To find each of the table's four outer edges, the
   old code searched the entire 85-megapixel page for printed-line
   pixels, then threw away everything outside a narrow strip near that
   edge. It now searches only that strip. The four edges are also
   worked out at the same time instead of one after another. This saved
   about 4.5 seconds per page.
3. **Reading all 48 boxes at the same time** (`extract_invoice.py`).
   The image work for each box (cutting it out, cleaning it up, finding
   the separate digits) used to happen one box at a time, using one of
   the 12 cores. It now runs across all cores at once. The model's
   actual reading of the digits still happens one box at a time, in the
   same order as before. That keeps the results from depending on
   which box happens to finish first. To make this possible,
   `digit_reader.read_number()` was split in two:
   `segment_digit_blobs()` does the image work and the new
   `classify_blobs()` does the model reading and the flag checks.
   `read_number()` still exists and does both, for reading a single
   box.
4. **Several scans in one run.** Just starting the program (loading
   PyTorch and the model) takes about 20 seconds on this machine, which
   is now much longer than reading a page. `extract_invoice.py` now
   accepts many scans at once (`python extract_invoice.py
   invoices/*.png`), so that startup cost is paid once per batch
   instead of once per page. The Odoo version won't have this problem
   at all, since it will be a program that stays running. Note that
   `--output-dir` now means the parent folder that each scan's own
   results folder goes in, not the results folder itself.

**What 10 invoices take now:** about 20 seconds to start, plus about 3
seconds per page, so roughly 50 seconds in total, instead of 12
minutes or more.

**What's left, and why it was left alone.** Of the remaining ~3
seconds, about 1.5 is simply opening the scan: the PNG file has to be
decompressed into 85 million pixels, and OpenCV's loader turned out to
be no faster than the current one. About 1.1 is the border search. Both
could be made faster only by working on a shrunken copy of the page,
which would change the answers slightly (the border's corners would
land a pixel or two differently). That wasn't worth it, because the
border's accuracy is what everything else is measured from (see the
next section). One idea for later, in the Odoo version: render each PDF
page straight into memory instead of saving it as a PNG and reading it
back, which would skip most of that 1.5 seconds.

**Not yet re-timed on the Docker/Odoo test server.** The original
73-second figure came from there, and the speed there depends on how
many cores Docker is allowed to use.

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

### Fixed (2026-09-24): the Return box was still reading the printed price next to it

**What you would have seen.** On pages from all four invoice PDFs, the
Return column was full of numbers like 831, 800, 8000 and 80000, when
the box on the paper was actually empty. The software was reading the
printed price in the next column: "$4.2…" came out as "831", "$3.00" as
"800". Across all 96 sample pages, 182 rows came out with more returned
than ordered, which is almost always this misread, and about 90 of those
had no review flag. They would have gone into Odoo as negative
quantities. The same fault was also quietly cutting the first digit off
Qty values: "100" read as 0, "63" as 3, "36" as 6, "130" as 30.

**Why it happened.** The step that nudges each box onto the printed
lines (see "It nudges each box's edges inwards" above) has to recognise
a column divider. It was accepting any straight up-and-down stroke at
least half a box tall. But the printed "$" and "4" in the Unit Price
column are that tall, and so is a handwritten "1". So the Return box's
right edge would stop on the "$" or the "4", in the middle of the price,
instead of on the real divider just before it, and the price then
counted as being inside the Return box. On the Qty side, the left edge
would stop on the stroke of a handwritten "1", cutting that digit and
everything left of it out of the box.

**What was changed:**

1. **A column divider now has to be longer than a whole box is tall**
   (`DIVIDER_MIN_LENGTH_CELL_FRACTION` in `extract_invoice.py`). A real
   divider runs unbroken through the rows above and below too, which the
   picture used for reading includes. No printed character or
   handwritten digit is that long. This alone fixed nearly all of the
   price misreads and the cut-off Qty digits.
2. **A new review flag: more returned than ordered**
   (`return_exceeds_quantity`, added in `extract_invoice.py`, with a
   plain-language explanation on the review screen). A safety net,
   whatever the cause: any row whose Return is bigger than its Qty now
   gets flagged.
3. **A mark that sits alone in a box and is under half the box's height
   is no longer read as a digit** (`MIN_LONE_DIGIT_HEIGHT_FRACTION` in
   `digit_reader.py`). Fix 1 made boxes tighter, and a side effect showed
   up: a speck of paper texture that used to be too small compared to the
   old, oversized box now counted as digit-sized in the correctly-sized
   one, so some empty Return boxes read "3" or "2". A digit written on
   its own is full height, so a short mark alone in a box is treated like
   the other leftover strokes: left out of the number, and flagged if
   it's pen-stroke-sized. The half-height trailing zeros one writer uses
   aren't affected, because they always sit next to a full-size digit.

**How it was checked.** Every page in `invoices/` (96 pages, all four
PDFs) was read before and after, and every value that changed was looked
at against its box picture by eye, about 80 in all, rather than trusting
the totals:

| Across all 96 pages | Before | After |
|---|---|---|
| Return values of 80 or more (prices read as returns) | 137 | 6 |
| Rows with more returned than ordered | 182 | 28, **all now flagged** |
| Fields flagged for a person to check | 416 | 256 |

Of the Qty and Return values that changed and weren't simply a price
turning back into a blank, the new reading was right in the large
majority, and the old reading in almost none. Both the extra digits
recovered ("100", "63", "130") and the blanks restored were confirmed
against the pictures. The few new readings that are still wrong are
mostly ones that were already wrong a different way (a tick mark next
to a "24" read as extra digits, for example), and are flagged.

### Accuracy, measured against correct answers (2026-09-24)

The first real accuracy measurement of the whole reading process, not
just the digit model. 8 pages were picked at random, 2 from each
invoice PDF: 384 boxes. Every box was read by eye from its picture
*before* looking at what the software said, so the software's answer
couldn't sway the reading. 10 boxes whose handwriting is genuinely
ambiguous (a "4" that could be a "9", a crossed-out number) were left
out, leaving 374 scored.

| | Before the 2026-09-24 fix | After |
|---|---|---|
| All boxes read correctly | 88.5% | **95.5%** |
| Boxes with something written in them, read correctly | 75.4% (52 of 69) | **79.7%** (55 of 69) |
| Empty boxes correctly read as empty | 279 of 305 | **302 of 305** |
| Wrong readings | 43 | **17** |
| ...of which flagged for review | 29 | 4 |
| ...of which **not** flagged (would reach Odoo unless the reviewer spots it) | 14 | **13** |

(The "before" column already includes the new "more returned than
ordered" flag, so the real number of unflagged mistakes before the fix
was higher than 14.)

**In plain terms:** empty boxes are now almost always right. Of the
boxes that actually have a number written in them, about 4 in 5 are
read correctly. On an average page with about 9 filled-in boxes, that's
roughly 2 wrong, and most of those won't be flagged. That's why every
field stays on the review screen, not just the flagged ones.

**What the 13 unflagged mistakes are:**

- **The model misreading a digit (9 of the 13).** Mostly "9" read as "4"
  (5 times: this handwriting's 9s have an open top that looks like a 4),
  and "7" read as "1" ("27" read as "21", 3 times), plus "48" read as
  "46". The model was 92-100% sure of every one of these, the same as
  for its correct answers, so no confidence cut-off can catch them. The
  fix is improving the model: retraining it on more of these writers'
  own 9s and 7s, which is exactly what the planned saving of reviewed
  digit pictures would provide (see "Banking verified-correct crops"
  above).
- **An extra digit picked up from nearby ink (2).** "20" read as "220",
  "12" as "120".
- **A stray stroke in an empty box read as "1" (2).**

The by-eye readings (the correct answers) and the scoring script were
kept only in that session's scratch folder, not in this repository,
since they are real business data. Redoing this measurement means
reading the pages again.

### Open problem: one phone-photo page has its table border found in the wrong place

Found while checking the fix above. `NEW RAJA BAKERY LTD. (1)_page006`
is a phone photo, not a flat scan, so the table is tilted and
keystoned (narrower at one end). On that page the table-border finder
put the right-hand edge on the edge of the paper instead of the table's
own right border. Every box on the page is therefore shifted: about
half a row too high and too far right. Its misreads happen to be
flagged, but that's luck, not design. The shape check
(`validate_aspect_ratio`) can't catch it: this page's border is 5% off
the normal shape, but correctly-read pages range almost as far (up to
5.5%). It's the only page out of 96 where this was seen. It needs a
different kind of check, for example confirming that the table's
printed row lines really do end at the detected right edge. Not fixed
yet.

**What is still not perfect.** None of these are silent except the last one:

- A digit written as two strokes that are far apart can still be split up and misread. Automatically joining the pieces was tried and rejected, because the setting that correctly rejoins a split "4" also glues a genuine "50" into a single shape. A leftover stroke raises a review flag instead, which was the deliberate choice.
- One person writing these invoices draws trailing zeros at about half the height of the digit before them. That breaks the otherwise sensible assumption that the digits in one box are all about the same height. It is the reason a piece of ink qualifies as a digit by being either big enough *or* tall enough, rather than by height alone — worth remembering before tightening either setting.
- The model itself is about 94% accurate per digit, so it will occasionally read a digit wrong and be confident about it — for example a "9" read as a "4". Nothing flags a confident mistake. That is a limitation of the model, not of the reading process, and improving it means improving the model.

## Current data state

`invoice_digits/` has a real train/val/test split (several hundred to a thousand+ crops per digit in train, proportionally fewer in val/test) and is now finalized — no further hand-labeled invoice data is expected. Digit classes are imbalanced — `0` has roughly 3-4x more examples than digits like `7`-`9`, handled via class-weighted loss in `finetune.py` rather than by collecting more data for the rare classes.

Separately, `digit_bank/` (new as of 2026-09-25, gitignored, empty until real invoices start getting approved) grows on its own as `review_screen.py` approves invoices — see "Banking verified-correct crops as future training data" above. It is not yet merged into `invoice_digits/`'s train/val/test split or used by `finetune.py`; that merge is future work, once there's enough banked data to be worth a retraining run.

The shipped `checkpoints/digit_cnn_finetuned.pt` (regenerate with `python finetune.py`) reaches **94.19% test accuracy**. Digits `3` and `9` were historically the weakest (most often confused with `8`/`1` and `4`/`7` respectively); see [FINETUNING_NOTES.md](FINETUNING_NOTES.md) for the diagnosis, including a spot-check suggesting some of the remaining `9`→misclassified-as-something-else cases may be labeling errors rather than model errors — worth a manual look at the flagged crops listed there before assuming the model is at fault.
