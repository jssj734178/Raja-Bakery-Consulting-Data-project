"""
One-time (but safe to re-run) setup step: look up each of this project's
24 products by name in Odoo, and save the matching Odoo product ID back
into product_rows.json as "odoo_product_id".

Why this only needs to run once, and why matching by name is safe to do
here even though CLAUDE.md's build notes say the real per-invoice match
should go by barcode, not name: Jagbir confirmed (2026-09-25) that all 24
products are already set up in Odoo "spelled exactly as in
product_rows.json." Barcode matching was suggested as the safer option
for matching a SCANNED FORM ROW to a product, since OCR/handwriting could
introduce spelling drift -- but product_rows.json's names were typed by
hand to match Odoo already, so an exact-name lookup done once here, and
checked by eye against the printed results below, is reliable. Caching
the resulting ID means the actual per-invoice "send to Odoo" push never
has to search by name at all -- it just reads the number this script
already wrote down, which is both faster and removes any chance of a
future product rename in Odoo silently breaking the match.

Run with: python match_odoo_products.py
"""

import json

from odoo_client import OdooClient

PRODUCT_ROWS_PATH = "product_rows.json"


def main():
    with open(PRODUCT_ROWS_PATH) as f:
        data = json.load(f)

    client = OdooClient()

    matched, unmatched = 0, []
    for row in data["rows"]:
        product_id, odoo_name = client.find_product_id_by_name(row["product_name"])
        row["odoo_product_id"] = product_id
        if product_id is not None:
            matched += 1
            note = "" if odoo_name == row["product_name"] else f"  (Odoo spells it: {odoo_name!r})"
            print(f"  [{row['row_index']:2d}] OK   id={product_id:<6} {row['product_name']}{note}")
        else:
            unmatched.append(row["product_name"])
            print(f"  [{row['row_index']:2d}] MISS              {row['product_name']}")

    with open(PRODUCT_ROWS_PATH, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")

    print(f"\n{matched}/{len(data['rows'])} products matched by name in Odoo.")
    if unmatched:
        print("Not found in Odoo (left as odoo_product_id: null):")
        for name in unmatched:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
