"""
A thin connector to the Odoo server that approved invoices eventually get
pushed into. Kept as its own file, separate from review_screen.py, so the
connection logic can be reused by a one-time setup script (matching this
project's own product list to Odoo's product records, see
match_odoo_products.py) without dragging tkinter into it, and so it can be
tested on its own before it's wired into a button anyone clicks.

Talks to Odoo over XML-RPC, the same remote-control interface Odoo itself
recommends for external scripts -- no extra library needed, since it's
built into Python. Connection details (server address, database name,
login) live in odoo_settings.local.json at the project root, which is
never committed to git because it holds a real password -- see
CLAUDE.md, "Found: the WSL Odoo test server's actual login".
"""

import json
import xmlrpc.client

SETTINGS_PATH = "odoo_settings.local.json"


class OdooClient:
    """
    One authenticated connection to Odoo, plus the handful of lookups and
    writes this project's pipeline needs (finding a product by name,
    finding or creating a customer, checking for a duplicate paper
    invoice number, and creating a draft invoice).
    """

    def __init__(self, settings_path: str = SETTINGS_PATH):
        with open(settings_path) as f:
            settings = json.load(f)
        self.url = settings["url"]
        self.database = settings["database"]
        self.username = settings["username"]
        self.password = settings["password"]

        # Odoo's XML-RPC interface is split into two separate endpoints:
        # "common" just checks a login and hands back a numeric user ID,
        # "object" is the one actually used afterwards to read or write
        # any record, and needs that user ID on every call.
        common = xmlrpc.client.ServerProxy(f"{self.url}/xmlrpc/2/common")
        self.uid = common.authenticate(self.database, self.username, self.password, {})
        if not self.uid:
            raise RuntimeError(
                f"Could not log in to Odoo as '{self.username}' on database "
                f"'{self.database}' at {self.url}. Check odoo_settings.local.json."
            )
        self.models = xmlrpc.client.ServerProxy(f"{self.url}/xmlrpc/2/object")

    def _call(self, model: str, method: str, *args, **kwargs):
        """Run one Odoo API call (e.g. search, read, create) against a given model (e.g. 'product.product')."""
        return self.models.execute_kw(
            self.database, self.uid, self.password, model, method, list(args), kwargs,
        )

    def find_product_id_by_name(self, name: str):
        """
        Look up one Odoo product by name, ignoring case (Odoo's real
        product names turned out to have small capitalization
        differences from product_rows.json's own spelling, e.g.
        "675G" vs "675g" -- see match_odoo_products.py). Returns
        (product_id, odoo_name) for the first match, or (None, None) if
        nothing matches -- the caller gets Odoo's own exact spelling
        back too, so a case difference is still visible rather than
        silently papered over.

        Used only by the one-time reconciliation script
        (match_odoo_products.py), not on every invoice push -- see that
        script's own docstring for why matching by name is safe to do
        just once rather than every time.
        """
        ids = self._call("product.product", "search", [["name", "=ilike", name]])
        if not ids:
            return None, None
        record = self._call("product.product", "read", ids[:1], fields=["name"])[0]
        return record["id"], record["name"]

    def find_partner_id_by_name(self, name: str):
        """
        Look up one Odoo contact by exact name (case-insensitive, since
        Odoo's own '=' domain operator on a char field is
        case-sensitive in Postgres but this project's own customer-name
        folding, in review_screen.py, is not). Returns the contact's
        Odoo ID, or None if nothing matches.
        """
        ids = self._call("res.partner", "search", [["name", "=ilike", name]])
        return ids[0] if ids else None

    def create_partner(self, name: str) -> int:
        """
        Create a new Odoo contact with just a name -- for a one-off
        customer typed into the review screen who isn't on the regular
        list (see CLAUDE.md, decision 3: 'If a reviewer types a one-off
        customer who isn't already in Odoo, Approve should create them
        as a new Odoo contact automatically').
        """
        return self._call("res.partner", "create", {"name": name})

    def find_invoice_id_by_reference(self, paper_invoice_number: str):
        """
        Look up an existing draft/posted customer invoice already
        carrying this paper invoice number as its reference -- the
        free duplicate check described in CLAUDE.md, 'the paper invoice
        number becomes a label'. Returns the invoice's Odoo ID, or None
        if nothing matches. Callers should skip this check entirely for
        a blank paper_invoice_number, since it can't tell duplicates
        apart from every other invoice with no number recorded.
        """
        ids = self._call(
            "account.move", "search",
            [["ref", "=", paper_invoice_number], ["move_type", "=", "out_invoice"]],
        )
        return ids[0] if ids else None

    def create_draft_invoice(
        self, partner_id: int, invoice_date: str, ref: str, lines: list, note: str = None,
    ) -> int:
        """
        Create a draft Customer Invoice (account.move) with one line per
        product row. Left in draft status deliberately -- see CLAUDE.md,
        'Creates a draft invoice, not a finalized one' -- so a person
        still gives it a final look and posts/sends it from inside Odoo.

        Args:
            partner_id: the Odoo contact ID this invoice bills.
            invoice_date: the invoice's own handwritten date, "YYYY-MM-DD"
                -- not the date it's being approved/sent (see CLAUDE.md,
                "Use the invoice's own handwritten date").
            ref: the paper invoice number, stored as this invoice's
                reference so it can be searched later and checked for
                duplicates (see find_invoice_id_by_reference above). May
                be blank.
            lines: a list of {"product_id", "quantity", "price_unit"}
                dicts, one per product row actually ordered. price_unit
                may be None for a row whose Total Price couldn't be
                read -- Odoo then leaves that line's price at its own
                default (0), which is why every such line should be
                flagged to the reviewer as needing a manual price before
                the invoice gets posted.
            note: an optional plain-text warning to attach to the
                invoice (e.g. for a return_exceeds_quantity row that got
                approved anyway -- see CLAUDE.md, item 7 under "Money
                and tax").
        """
        invoice_lines = [
            (0, 0, {
                "product_id": line["product_id"],
                "quantity": line["quantity"],
                "price_unit": line["price_unit"] if line["price_unit"] is not None else 0.0,
            })
            for line in lines
        ]
        move_id = self._call("account.move", "create", {
            "move_type": "out_invoice",
            "partner_id": partner_id,
            "invoice_date": invoice_date,
            "ref": ref or False,
            "invoice_line_ids": invoice_lines,
        })
        if note:
            # A plain internal note on the invoice's chatter (Odoo's
            # activity/message log), not a formal credit note or a
            # negative line -- exactly what CLAUDE.md's decision 7
            # calls for: "a visible note flagging that this row's
            # numbers don't add up and need checking."
            self._call("account.move", "message_post", move_id, body=note)
        return move_id
