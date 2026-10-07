"""
Runs once when the module is first installed on a fresh Odoo: loads the
bakery's 24 products (with their printed catalog prices) and regular
customer list, so a newly set-up Mac is usable straight away instead of
needing them typed in by hand. Anything already in Odoo under the same
name is left alone, never duplicated or overwritten.
"""

import json
import os


def _bakery_ml_dir():
    # Locate the shared reading engine's folder (see Dockerfile) without
    # importing it, which would load all of PyTorch just to find a path.
    import importlib.util

    return os.path.dirname(importlib.util.find_spec("extract_invoice").origin)


def post_init_hook(env):
    base = _bakery_ml_dir()
    with open(os.path.join(base, "product_rows.json")) as f:
        products = json.load(f)["rows"]
    with open(os.path.join(base, "customers.json")) as f:
        customers = json.load(f)["customers"]

    Product = env["product.product"]
    for row in products:
        name = row["product_name"]
        if Product.search([("name", "=ilike", name)], limit=1):
            continue
        # No tax on anything for now (CLAUDE.md, "Money and tax" item 8).
        Product.create({
            "name": name,
            "list_price": row.get("catalog_price") or 0.0,
            "type": "consu",
            "sale_ok": True,
            "taxes_id": [(5, 0, 0)],
        })

    Partner = env["res.partner"]
    for name in customers:
        if not Partner.search([("name", "=ilike", name)], limit=1):
            Partner.create({"name": name, "customer_rank": 1})
