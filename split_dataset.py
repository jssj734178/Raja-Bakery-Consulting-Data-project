"""
Splits labeled digit crops into train/val/test sets, grouped by which
invoice each digit came from -- never splitting a single invoice's
digits across more than one set. Mixing digits from the same invoice
across train and test would let the model partly "memorize" that
invoice's handwriting during training, making test accuracy look
better than it honestly is.

Run this any time after labeling some digits with label_tool.py --
safe to re-run as you label more over time; already-split invoices
stay in their original split. Moves files out of
invoice_digits/<digit>/ into invoice_digits/<split>/<digit>/, where
<split> is train, val, or test.

Run with:  python3 split_dataset.py
"""

import glob
import hashlib
import os
import re
import shutil

DATASET_DIR = "invoice_digits"
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# test gets whatever fraction is left over: 1 - TRAIN_FRAC - VAL_FRAC

# Matches the naming label_tool.py produces: "{invoice_name}_{4-digit counter}.png"
CROP_FILENAME_PATTERN = re.compile(r"^(.*)_\d{4}\.png$")


def get_invoice_name(filename: str) -> str:
    """
    Recover which invoice a digit crop came from, based on the
    filename label_tool.py saved it with.

    Args:
        filename: a crop's filename, e.g. "your_scans_page047_0012.png".

    Returns:
        The invoice name portion, e.g. "your_scans_page047".
    """
    match = CROP_FILENAME_PATTERN.match(filename)
    if not match:
        raise ValueError(f"Unexpected filename format: {filename}")
    return match.group(1)


def get_split(invoice_name: str) -> str:
    """
    Deterministically assign an invoice to train, val, or test.

    Uses a hash of the invoice's name rather than random shuffling, so
    the same invoice always lands in the same split -- even if you
    label more invoices later and re-run this script, invoices already
    assigned won't move to a different split out from under you.

    Args:
        invoice_name: the invoice identifier (e.g. "your_scans_page047").

    Returns:
        One of "train", "val", "test".
    """
    digest = hashlib.md5(invoice_name.encode()).hexdigest()
    # First 8 hex chars -> an integer -> normalized into [0, 1)
    fraction = int(digest[:8], 16) / 0xFFFFFFFF

    if fraction < TRAIN_FRAC:
        return "train"
    elif fraction < TRAIN_FRAC + VAL_FRAC:
        return "val"
    else:
        return "test"


def main():
    moved_counts = {"train": 0, "val": 0, "test": 0}
    invoices_seen = {"train": set(), "val": set(), "test": set()}

    for digit in range(10):
        source_dir = os.path.join(DATASET_DIR, str(digit))
        if not os.path.isdir(source_dir):
            continue

        for path in glob.glob(os.path.join(source_dir, "*.png")):
            filename = os.path.basename(path)
            invoice_name = get_invoice_name(filename)
            split = get_split(invoice_name)

            dest_dir = os.path.join(DATASET_DIR, split, str(digit))
            os.makedirs(dest_dir, exist_ok=True)

            shutil.move(path, os.path.join(dest_dir, filename))
            moved_counts[split] += 1
            invoices_seen[split].add(invoice_name)

    print("Split complete.")
    for split in ["train", "val", "test"]:
        print(
            f"  {split}: {moved_counts[split]} digit crops "
            f"from {len(invoices_seen[split])} invoices"
        )


if __name__ == "__main__":
    main()
