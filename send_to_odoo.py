"""
Turns one already-approved invoice's review.json into a draft Customer
Invoice in Odoo. Kept separate from odoo_client.py (the raw XML-RPC
connection) and from review_screen.py (the tkinter UI), so the actual
business rules here -- which rows become invoice lines, what gets
blocked, what gets a warning note -- can be read and changed on their
own, the same way extract_invoice.py orchestrates digit_reader.py and
alignment.py without being either of them.

Only ever called for an invoice review_screen.py has already approved
(i.e. one with a review.json) -- by the time that file writes one, it
has already checked that the customer field isn't blank, the date is a
real YYYY-MM-DD date, and every Qty/Return is a whole number of 0 or
more, so none of that needs re-checking here. What DOES get checked
here, freshly, is everything that can only go wrong against Odoo's own
live data: a product this project doesn't have an Odoo match for yet, a
row with a quantity but no price (see CLAUDE.md's "Missing price"
decision -- sending goes blocked, not through at $0), and a paper
invoice number that's already been sent before.
"""

import json
import os

from odoo_client import OdooClient

EXTRACTIONS_DIR = "extractions"
PRODUCT_ROWS_PATH = "product_rows.json"


class SendToOdooError(Exception):
    """
    Raised for anything that should stop a push and be shown directly to
    the reviewer as the reason -- as opposed to a connection failure or
    other unexpected error, which the caller should show as-is rather
    than assume is one of these deliberate, expected stopping points.
    """


def _load_product_odoo_ids() -> dict:
    """product_name -> odoo_product_id (or None), from product_rows.json -- see match_odoo_products.py, which is what fills this in."""
    with open(PRODUCT_ROWS_PATH) as f:
        data = json.load(f)
    return {row["product_name"]: row.get("odoo_product_id") for row in data["rows"]}


def send_invoice(invoice_name: str, client: OdooClient = None) -> dict:
    """
    Push one already-approved invoice to Odoo as a draft Customer
    Invoice, and record the resulting Odoo invoice ID back into that
    invoice's own review.json so it isn't accidentally sent twice.

    Args:
        invoice_name: an extractions/ subfolder name that has already
            been through Approve & Save (has a review.json).
        client: an already-open OdooClient to reuse, or None to open a
            fresh connection.

    Returns:
        {"odoo_invoice_id": int, "warning": str or None}. warning is
        set when the push succeeded but something about it still needs
        a look -- specifically, a row where more was returned than was
        ordered, which CLAUDE.md's "Money and tax" decision says should
        go through with a plain warning note attached in Odoo rather
        than block the whole invoice, since it happens rarely and
        usually means a real paperwork problem worth flagging urgently,
        not silently discarding a real customer's real order.

    Raises:
        SendToOdooError, with a message meant to be read directly by
        whoever clicked the button, for: a row with a quantity but no
        price, a product not yet matched to Odoo, an invoice already
        sent before (by this invoice, or by paper number), or a missing
        customer/date (shouldn't happen if review.json came from
        review_screen.py's own Approve & Save, but checked again here
        rather than trusted blindly, since review.json is a plain file
        someone could hand-edit).
    """
    invoice_dir = os.path.join(EXTRACTIONS_DIR, invoice_name)
    review_path = os.path.join(invoice_dir, "review.json")
    if not os.path.exists(review_path):
        raise SendToOdooError("This invoice hasn't been approved yet -- nothing to send.")
    with open(review_path) as f:
        review = json.load(f)

    if review.get("odoo_invoice_id"):
        raise SendToOdooError(
            f"This invoice was already sent to Odoo, as draft invoice "
            f"#{review['odoo_invoice_id']}. If it genuinely needs to be sent "
            f"again, make the correction directly on that draft in Odoo instead."
        )
    if not review.get("customer"):
        raise SendToOdooError("No customer is set on this invoice.")
    if not review.get("invoice_date"):
        raise SendToOdooError("No invoice date is set on this invoice.")

    product_odoo_ids = _load_product_odoo_ids()
    ordered_rows = [r for r in review["rows"] if r["line_quantity"] > 0]

    missing_price = [r["product_name"] for r in ordered_rows if not r["total_price"]]
    if missing_price:
        raise SendToOdooError(
            "Can't send -- these rows have a quantity but no Total Price. "
            "Fill in the price above before sending to Odoo: " + ", ".join(missing_price)
        )

    unknown_products = [r["product_name"] for r in ordered_rows if product_odoo_ids.get(r["product_name"]) is None]
    if unknown_products:
        raise SendToOdooError(
            "Can't send -- these products aren't set up in Odoo yet: " + ", ".join(unknown_products)
        )

    client = client or OdooClient()

    paper_number = (review.get("paper_invoice_number") or "").strip()
    if paper_number:
        # See CLAUDE.md, "the paper invoice number becomes a label" --
        # this is the free duplicate check that decision earns. Skipped
        # entirely for a blank number, since a blank can't be told
        # apart from any other blank invoice.
        existing_id = client.find_invoice_id_by_reference(paper_number)
        if existing_id:
            raise SendToOdooError(
                f"An invoice with paper number '{paper_number}' has already been "
                f"sent to Odoo (draft invoice #{existing_id}). Check this isn't a "
                f"duplicate submission before sending it again."
            )

    partner_id = client.find_partner_id_by_name(review["customer"])
    if partner_id is None:
        # See CLAUDE.md, decision 3: a typed one-off customer who isn't
        # already in Odoo gets created automatically, rather than
        # blocking the invoice on someone adding them by hand first.
        partner_id = client.create_partner(review["customer"])

    lines = [
        {
            "product_id": product_odoo_ids[r["product_name"]],
            "quantity": r["line_quantity"],
            "price_unit": r["unit_price"],
        }
        for r in ordered_rows
    ]

    # See CLAUDE.md, "Money and tax" item 7: this shouldn't normally
    # happen, and when it does the invoice still goes through (it's
    # usually a real order, not nothing) but with a visible note on it
    # for someone to check against the paper copy.
    over_returned = [r["product_name"] for r in review["rows"] if r["return"] > r["quantity"]]
    warning = None
    if over_returned:
        warning = (
            "Warning: more was returned than ordered on these rows, which "
            "shouldn't normally happen -- check against the paper copy: "
            + ", ".join(over_returned)
        )

    odoo_invoice_id = client.create_draft_invoice(
        partner_id=partner_id,
        invoice_date=review["invoice_date"],
        ref=paper_number,
        lines=lines,
        note=warning,
    )

    review["odoo_invoice_id"] = odoo_invoice_id
    with open(review_path, "w") as f:
        json.dump(review, f, indent=2)

    return {"odoo_invoice_id": odoo_invoice_id, "warning": warning}
