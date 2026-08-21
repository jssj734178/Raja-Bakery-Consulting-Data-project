# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A pipeline for training a digit-recognition model on handwritten digits found on scanned bakery invoices. The model architecture (`DigitCNN`) is pretrained on MNIST first (Stage 1, implemented), with the intent to fine-tune on real invoice digits afterward (Stage 2, not yet implemented — no fine-tuning script exists yet).

## Environment

- Python 3.14, virtualenv at `venv/` (Windows: `venv\Scripts\python.exe`).
- No `requirements.txt` exists yet — dependencies were installed ad hoc. Currently installed: `torch`, `numpy`, `pillow`, `pymupdf`. If you add a dependency, install it into `venv/` and consider creating a `requirements.txt`.
- Not a git repository — there is no version control on this project currently.
- No test suite, linter, or formatter is configured.

## Pipeline / architecture

The scripts form a sequential pipeline, each a standalone CLI entry point (`python <script>.py`), run roughly in this order:

1. **[pdf_to_images.py](pdf_to_images.py)** — renders each page of one or more input PDFs (scanned invoices) to a PNG at 300 DPI, saved into `invoices/`. Supports `--rotate 90|180|270` for scans with orientation issues.
2. **[label_tool.py](label_tool.py)** — a tkinter GUI for manually drawing boxes around handwritten digits in `invoices/*` images and labeling them 0-9 via keypress. Saves each crop into `invoice_digits/<digit>/`, preprocessed to match MNIST's format (grayscale, inverted so ink is light-on-dark, padded to square before resizing to 28x28 — padding before resize matters so a digit's proportions aren't distorted). Crop filenames encode their source invoice as `{invoice_name}_{4-digit counter}.png`, which `split_dataset.py` depends on.
3. **[split_dataset.py](split_dataset.py)** — moves crops from `invoice_digits/<digit>/` into `invoice_digits/<train|val|test>/<digit>/`. Splits are assigned **by invoice, not by individual digit** (via a deterministic MD5 hash of the invoice name), so digits from the same invoice never end up split across train/val/test — otherwise the model could partially memorize an invoice's handwriting and inflate test accuracy. Safe to re-run repeatedly as more invoices get labeled over time; already-split invoices don't move between splits.
4. **[data.py](data.py)** — `load_mnist()` loads `data/mnist.pkl.gz` (three pickled `(images, labels)` splits) and wraps each in a PyTorch `DataLoader`, reshaping flat 784-vectors to `(1, 28, 28)` and standardizing with MNIST's known mean/std (0.1307 / 0.3081).
5. **[model.py](model.py)** — `DigitCNN`: two conv+pool blocks (1→32→64 channels, 28x28→7x7) feeding two fully-connected layers (3136→128→10). Returns raw logits (no softmax — `CrossEntropyLoss` applies it).
6. **[train.py](train.py)** — Stage 1: trains `DigitCNN` on MNIST for 5 epochs with Adam (`lr=1e-3`), reports val accuracy per epoch and final test accuracy, saves weights (`state_dict`) to `checkpoints/digit_cnn_mnist.pt`. Run with `python train.py`. `checkpoints/` doesn't exist until this has been run at least once.

A Stage 2 script that fine-tunes the MNIST-pretrained weights on the labeled `invoice_digits/` data (loading `checkpoints/digit_cnn_mnist.pt`) is the natural next step but does not exist yet.

## Current data state

`invoice_digits/` is actively being built via manual labeling and already has a real train/val/test split (several hundred to a thousand+ crops per digit in train, proportionally fewer in val/test). Digit classes are imbalanced — `0` has roughly 3-4x more examples than digits like `7`-`9`.
