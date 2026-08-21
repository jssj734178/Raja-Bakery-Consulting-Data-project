# Stage 2 fine-tuning: methods, diagnostics, and results

A record of how `checkpoints/digit_cnn_finetuned.pt` was arrived at — what
was tried, what moved the needle, and what didn't. Kept for reference since
this project is being handed off and `invoice_digits/` is finalized (no more
labeled data is coming in), so this checkpoint isn't expected to be
retrained from scratch again soon.

## 1. Baseline: Stage 1 (MNIST pretraining)

`train.py` trains `DigitCNN` on MNIST for 5 epochs, no changes needed —
reached **99.09% MNIST test accuracy** on the first run. This is the
starting point Stage 2 builds on.

## 2. Zero-shot baseline: does MNIST transfer to real invoice digits at all?

Before writing any fine-tuning code, the key open question was how different
real, hand-drawn invoice digits actually are from MNIST's cleanly rendered
ones. Running the MNIST-only model directly against `invoice_digits/val`
(no fine-tuning) answered that:

**Zero-shot accuracy: 33.04%** — barely better than random for some digits
(`8`: 2.4%, `9`: 0.0%). This confirmed a large domain gap: real pen strokes,
scanner noise, and inconsistent proportions don't look like MNIST's digits
even though both are 28x28 grayscale after `label_tool.py`'s preprocessing.
This number is now printed automatically at the start of every
`finetune.py` run, as the baseline fine-tuning needs to beat.

## 3. First fine-tuning pass: frozen conv layers, weighted loss, light augmentation

Initial design decisions (all still in `finetune.py`, later revised in
step 6):

- **Both conv blocks frozen**, only the FC head fine-tuned — the reasoning
  at the time was that with only ~2,770 real training crops (vs. MNIST's
  60k), letting the conv layers keep updating risked overfitting/forgetting.
- **Class-weighted `CrossEntropyLoss`** (inverse frequency) to counter
  `invoice_digits`'s ~4x imbalance between `0` and `7`-`9`.
- **Augmentation**: small random rotation (±10°) and translation (±2px),
  applied only to the training split, implemented directly with `PIL`
  (no `torchvision` dependency added just for this).
- **Best-checkpoint selection**: save only when validation accuracy
  improves, rather than trusting the final epoch.

Result: **~88% test accuracy** (single run: 88.38%).

## 4. Diagnosing run-to-run variance

A single run's accuracy turned out not to be very trustworthy on its own —
rerunning the identical script gave noticeably different results (86.75% to
89.11% across 5 runs), since `finetune.py` had several unseeded sources of
randomness (augmentation's random rotation/scale/translation, training-batch
shuffling, dropout).

**Fix**: added a `--seed` CLI argument, calling `random.seed()` (covers
augmentation) and `torch.manual_seed()` (covers batch shuffling and dropout)
at the start of `main()`. This makes any specific run's result exactly
reproducible on demand — confirmed by rerunning the same seed and diffing
the resulting checkpoint file byte-for-byte.

**Then swept 10 seeds** to characterize the real spread rather than trusting
one run: 86.2%-89.3%, mean ≈ 87.7%. The best (seed 3, 89.29%) was locked in
as the checkpoint at that point in the process.

## 5. Width/height scaling augmentation

Rotation and translation alone don't capture that different people write
digits with different proportions (taller/narrower vs. shorter/wider).
Added independent random x/y scaling (±10%, not a single shared factor) to
`_augment()`, applied between the rotation and translation steps: resize by
independently sampled `scale_x`/`scale_y`, then paste back onto a centered
28x28 black canvas (PIL's `paste()` clips automatically if the resized image
is larger than the canvas, so no extra cropping logic was needed).

This didn't clearly move the needle on its own (86.75%-89.11% across a few
runs, same range as before) — its main value ended up being combined with
the bigger change in step 6, and as one of three complementary sources of
synthetic variation given that no more real training data is coming.

## 6. Confusion matrix + spot-checking mislabeled data

Per-digit accuracy showed `3` and `9` as consistently weak, but not *why*.
Added a `confusion_matrix()` function (10x10 grid, rows = true digit,
columns = predicted) plus a "most common mistake per digit" summary, printed
after every run's final test evaluation.

This showed two consistent patterns across seeds:
- `3` most often predicted as `8` (7.7%-9.6% of `3`s), secondarily `1`.
- `9` most often predicted as `4` (21.7%-26.1% of `9`s) — the single largest
  confusion in the whole matrix.

**Visually spot-checking the actual misclassified crops** (not just the
matrix) turned up something the matrix alone couldn't: several of the `9`s
the model called `4` visually looked more like **`7`s** on inspection — open
top stroke, diagonal descender, no closed loop. That points to likely
**labeling errors** from the manual labeling pass, not model failure: no
model can correctly predict a digit whose ground-truth label is wrong.

Flagged for manual review (not yet confirmed, since this requires knowing
the source invoices/handwriting, which the model can't):
- `invoice_digits/test/9/NEW RAJA BAKERY LTD._page008_1276.png`
- `invoice_digits/test/9/PH 416-727-0623 or 416-474-9679_page002_0378.png`,
  `_0391.png`, `_0438.png`, `page009_0664.png`, `_0678.png`
- `invoice_digits/test/3/NEW RAJA BAKERY LTD. (1)_page004_0042.png`

If genuinely mislabeled, relabeling these (or removing them) is free
accuracy no amount of further model tuning can otherwise recover.

## 7. Partial unfreezing of conv2 — the biggest single win

Given the large zero-shot domain gap found in step 2, the fully-frozen
conv assumption from step 3 was revisited: maybe restricting fine-tuning to
just the FC head was too conservative.

**Change**: keep `conv1` frozen (its low-level edge/stroke detectors are
generic enough to transfer regardless of domain), but unfreeze `conv2` (the
higher-level, more MNIST-shape-specific block) with its own low learning
rate (`1e-4`, vs. `1e-3` for the FC head) via two Adam parameter groups —
letting it adapt without the large, disruptive updates the FC head needs.

Swept the same 5 seeds with this change:

| seed | frozen conv2 (before) | unfrozen conv2 (after) |
|---|---|---|
| 0 | 86.75% | 92.38% |
| 1 | 89.29% | 92.20% |
| 2 | 88.93% | 93.28% |
| 3 | 89.29% | **94.19%** |
| 4 | 86.75% | 92.20% |

**+3 to +7 points on every single seed tested** — by far the largest
improvement of any change tried. Digit `3` went from 73-83% to 88.5%; digit
`9` went from 65-96% (highly seed-dependent before) to a consistent 95.7%.

This became the new default (`finetune.py`'s `--freeze-conv2` flag now
opts *into* the old, more conservative behavior, for comparison).

## 8. Final locked-in configuration

- Seed `3`, `conv2` unfrozen (both now `finetune.py`'s defaults) —
  reproducible with a plain `python finetune.py`, verified byte-identical
  across reruns.
- **94.19% test accuracy**, no digit below 88.5%.
- Checkpoint: `checkpoints/digit_cnn_finetuned.pt` (gitignored — see
  `.gitignore` — regenerate rather than expecting it in the git history).

## 9. Tried / considered but not pursued (diminishing returns)

- **Ensembling across seeds** — averaging predictions from a few of the
  swept seeds could plausibly buy another point, at the cost of multiple
  forward passes at inference time. Not done, since the labeling spot-check
  (step 6) and conv2 unfreezing (step 7) were higher-value for the effort.
- **Elastic/pen-thickness augmentation, LR scheduling, more epochs** —
  plausible small further gains, but with a fixed, finalized ~2,770-example
  training set and a 551-example test set, the practical ceiling is
  increasingly set by data size/quality rather than training technique.
- **Architecture changes** (e.g. batch norm) — would require redoing Stage 1
  MNIST pretraining too; disproportionate effort for a model already at
  ~99% on MNIST and 94%+ on the real target task.
