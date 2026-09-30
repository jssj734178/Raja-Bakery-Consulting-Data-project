"""
Compares two runs of extract_invoice.py over the same scans -- the
before/after check used for every change to digit_reading code, so a fix
is judged by what it did across all real invoices and not by a few
examples.

Usage:
    python compare_extractions.py OLD_FOLDER NEW_FOLDER

Each folder is the parent that holds one sub-folder per scan (what
`extract_invoice.py --output-dir` writes to). Prints:
  - how many Qty/Return values or flags differ (should normally be 0 for
    a change aimed at Total Price),
  - how many Total Price values changed, and how many of those changed
    to or from matching the product's usual price,
  - the count of each Total Price flag before and after.

"Matches the product's usual price" is a stand-in for "read correctly":
a bakery's prices stay stable across invoices, so a unit price equal to
the product's most common unit price is very likely a right read. It
does not count real discounts as right, so treat it as a signal, not
as proof.
"""
import collections
import json
import os
import sys


def load_rows(folder: str) -> dict:
    rows = {}
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name, "results.json")
        if not os.path.exists(path):
            continue
        for row in json.load(open(path, encoding="utf-8"))["rows"]:
            rows[(name, row["row_index"])] = row
    return rows


def main():
    old, new = load_rows(sys.argv[1]), load_rows(sys.argv[2])

    # Each product's most common unit price among reads nothing flagged.
    prices = collections.defaultdict(collections.Counter)
    for row in old.values():
        if row["unit_price"] is not None and not row["total_price_flags"]:
            prices[row["product_name"]][round(row["unit_price"], 2)] += 1
    usual = {p: c.most_common(1)[0][0] for p, c in prices.items()}

    def matches_usual(row):
        unit, target = row["unit_price"], usual.get(row["product_name"])
        return bool(unit is not None and target and abs(unit - target) <= 0.015 * target + 0.005)

    common = old.keys() & new.keys()
    qty_diffs = sum(
        1 for k in common for f in ("quantity", "return", "quantity_flags", "return_flags")
        if old[k][f] != new[k][f]
    )
    changed = [k for k in common if old[k]["total_price"] != new[k]["total_price"]]
    now_right = sum(1 for k in changed if matches_usual(new[k]) and not matches_usual(old[k]))
    now_wrong = sum(1 for k in changed if matches_usual(old[k]) and not matches_usual(new[k]))

    print(f"rows compared: {len(common)}")
    print(f"Qty/Return differences: {qty_diffs}")
    print(f"Total Price values changed: {len(changed)}")
    print(f"  now match usual price: {now_right}   stopped matching: {now_wrong}")
    print(f"  fields matching usual price, old -> new: "
          f"{sum(map(matches_usual, old.values()))} -> {sum(map(matches_usual, new.values()))}")
    flags_old = collections.Counter(f for r in old.values() for f in r["total_price_flags"])
    flags_new = collections.Counter(f for r in new.values() for f in r["total_price_flags"])
    for flag in sorted(set(flags_old) | set(flags_new)):
        print(f"  {flag}: {flags_old[flag]} -> {flags_new[flag]}")
    print(f"  changed AND now unflagged: {sum(1 for k in changed if not new[k]['total_price_flags'])}")


if __name__ == "__main__":
    main()
