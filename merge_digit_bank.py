"""
Merges the digit pictures banked from approved invoices (digit_bank/ on
the desktop, or Odoo's bakery_digit_bank/ copied over) into
invoice_digits/<train|val|test>/<digit>/, ready for finetune.py.

Each banked picture is labeled with the number the reviewer approved
(see digit_bank.py). Like split_dataset.py, the split is decided by
invoice, using the same deterministic hash, so every digit from one
invoice lands in the same split and test accuracy isn't inflated by
the model having seen that writer's invoice in training. The banked
invoices are new ones the model has never trained on, so they can't
leak into the existing test set.

Pictures are COPIED, not moved, and already-merged ones are skipped, so
this is safe to re-run as the bank grows. Banked names look like
"<invoice name>_row05_quantity_0.png"; they're stored here as-is.

Run with:  python merge_digit_bank.py [bank_folder]   (default: digit_bank)
Then retrain with:  python finetune.py
"""
import glob
import os
import re
import shutil
import sys

from split_dataset import DATASET_DIR, get_split

BANK_FILENAME_PATTERN = re.compile(r"^(.*)_row\d{2}_.*\.png$")


def main():
    bank_dir = sys.argv[1] if len(sys.argv) > 1 else "digit_bank"
    copied = {"train": 0, "val": 0, "test": 0}
    skipped = 0

    for digit in range(10):
        for path in glob.glob(os.path.join(bank_dir, str(digit), "*.png")):
            filename = os.path.basename(path)
            match = BANK_FILENAME_PATTERN.match(filename)
            if not match:
                print(f"Skipping unrecognized filename: {filename}")
                continue
            split = get_split(match.group(1))
            dest = os.path.join(DATASET_DIR, split, str(digit), filename)
            if os.path.exists(dest):
                skipped += 1
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copyfile(path, dest)
            copied[split] += 1

    print(f"Merged {sum(copied.values())} new digit crops "
          f"(train {copied['train']}, val {copied['val']}, test {copied['test']}); "
          f"{skipped} were already merged.")


if __name__ == "__main__":
    main()
