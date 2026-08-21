# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

## Current data state

`invoice_digits/` has a real train/val/test split (several hundred to a thousand+ crops per digit in train, proportionally fewer in val/test) and is now finalized — no further labeled invoice data is expected. Digit classes are imbalanced — `0` has roughly 3-4x more examples than digits like `7`-`9`, handled via class-weighted loss in `finetune.py` rather than by collecting more data for the rare classes.

The shipped `checkpoints/digit_cnn_finetuned.pt` (regenerate with `python finetune.py`) reaches **94.19% test accuracy**. Digits `3` and `9` were historically the weakest (most often confused with `8`/`1` and `4`/`7` respectively); see [FINETUNING_NOTES.md](FINETUNING_NOTES.md) for the diagnosis, including a spot-check suggesting some of the remaining `9`→misclassified-as-something-else cases may be labeling errors rather than model errors — worth a manual look at the flagged crops listed there before assuming the model is at fault.
