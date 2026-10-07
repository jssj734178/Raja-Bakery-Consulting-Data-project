"""
One scanned paper invoice (one PDF page) going through: waiting to be
read -> read by the shared reading engine (extract_invoice.py) -> checked
and corrected by a person -> turned into a draft Customer Invoice.

The business rules here deliberately mirror send_to_odoo.py (the desktop
"Send to Odoo" button), which was verified against a live server, so both
routes into Odoo behave the same way: no stock movement, draft not posted,
price from the invoice's own handwriting (Total Price / (Qty - Return)),
no tax, paper number saved as the reference with a duplicate check, and a
plain warning note when more was returned than ordered.
"""

import base64
import json
import logging
import os
import shutil
import time

from odoo import api, fields, models, tools
from odoo.exceptions import UserError

_logger = logging.getLogger(__name__)

# Every reason the reader can flag a field, in plain words -- same wording
# as the desktop review screen so a reviewer sees the same explanations.
FLAG_EXPLANATIONS = {
    "too_many_blobs": "more separate marks than this box normally has",
    "possible_merged_digits": "a mark much wider than tall - possibly two digits touching",
    "possible_split_digit": "a leftover stroke - possibly a digit written in pieces",
    "unreadable_ink": "a faint mark too small to read, treated as blank",
    "low_confidence": "the software wasn't confident about a digit",
    "leading_zero_digit": "an extra mark was read as a 0 in front of the number",
    "return_exceeds_quantity": "return is bigger than quantity ordered - usually a misread",
    "no_decimal_point": "no decimal point found in the Total Price",
    "ambiguous_decimal_point": "more than one mark could be the decimal point",
    "total_price_without_quantity": "a Total Price was read but nothing was ordered on this row",
    "unit_price_far_from_catalog": "unit price is far from the printed catalog price - misread digit, misplaced decimal point, or a real discount",
    "quantity_without_total_price": "a quantity but no Total Price was read",
}

# Seconds one cron run may keep reading queued scans before stopping and
# letting the next run continue (the cron itself is allowed far longer by
# the limit-time-real-cron setting in docker-compose.yml).
CRON_TIME_BUDGET = 25 * 60

_MODEL_CACHE = {}


def _scans_dir():
    path = os.path.join(tools.config["data_dir"], "bakery_scans")
    os.makedirs(path, exist_ok=True)
    return path


def _digit_bank_dir():
    return os.path.join(tools.config["data_dir"], "bakery_digit_bank")


def _load_engine():
    """Import the reading engine and load the digit model once per server process."""
    import torch
    import extract_invoice

    if "model" not in _MODEL_CACHE:
        device = torch.device("cpu")
        _MODEL_CACHE["device"] = device
        _MODEL_CACHE["model"] = extract_invoice.load_model(extract_invoice.CHECKPOINT_PATH, device)
    return extract_invoice, _MODEL_CACHE["model"], _MODEL_CACHE["device"]


def _file_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read())


class BakeryInvoiceScan(models.Model):
    _name = "bakery.invoice.scan"
    _description = "Scanned invoice"
    _inherit = ["mail.thread"]
    _order = "id desc"

    name = fields.Char(required=True)
    state = fields.Selection(
        [
            ("queued", "Waiting to be read"),
            ("reading", "Being read"),
            ("review", "Ready to review"),
            ("sent", "Draft invoice created"),
            ("failed", "Needs manual handling"),
        ],
        default="queued", required=True, tracking=True,
    )
    failure_reason = fields.Char(readonly=True)
    pdf_page = fields.Binary("Page PDF", attachment=True, readonly=True)
    page_preview = fields.Image("Page", max_width=1600, max_height=1600, attachment=True, readonly=True)

    partner_id = fields.Many2one("res.partner", "Customer")
    invoice_date = fields.Date("Invoice date (handwritten)")
    paper_number = fields.Char("Paper invoice number")
    line_ids = fields.One2many("bakery.invoice.scan.line", "scan_id", "Rows")
    move_id = fields.Many2one("account.move", "Draft invoice", readonly=True, copy=False)
    flagged_count = fields.Integer(compute="_compute_flagged_count")

    @api.depends("line_ids.flag_text")
    def _compute_flagged_count(self):
        for scan in self:
            scan.flagged_count = len(scan.line_ids.filtered("flag_text"))

    # ------------------------------------------------------------------
    # Reading (runs in the background, from the cron)
    # ------------------------------------------------------------------

    @api.model
    def _cron_read_queued(self):
        started = time.time()
        while time.time() - started < CRON_TIME_BUDGET:
            scan = self.search([("state", "=", "queued")], order="id", limit=1)
            if not scan:
                return
            scan.state = "reading"
            self.env.cr.commit()  # so a second cron run doesn't pick the same scan
            try:
                scan._read_page()
            except Exception as exc:  # one bad page must never block the queue
                _logger.exception("Reading scan %s failed", scan.id)
                self.env.cr.rollback()
                scan = self.browse(scan.id)
                scan.write({"state": "failed", "failure_reason": f"Unexpected error: {exc}"})
            self.env.cr.commit()

    def _read_page(self):
        import pymupdf
        from PIL import Image

        self.ensure_one()
        extract_invoice, model, device = _load_engine()
        work = os.path.join(_scans_dir(), str(self.id))
        shutil.rmtree(work, ignore_errors=True)
        os.makedirs(work)

        doc = pymupdf.open(stream=base64.b64decode(self.pdf_page), filetype="pdf")
        zoom = 300 / 72  # same 300 DPI the whole pipeline was tuned on
        pix = doc[0].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
        image_path = os.path.join(work, "page.png")
        pix.save(image_path)
        doc.close()

        result = extract_invoice.extract_invoice(image_path, work, model, device)

        Image.MAX_IMAGE_PIXELS = None
        preview = Image.open(image_path).convert("RGB")
        preview.thumbnail((1600, 1600))
        preview_path = os.path.join(work, "preview.jpg")
        preview.save(preview_path, quality=80)
        self.page_preview = _file_b64(preview_path)
        os.remove(image_path)  # ~30 MB each; the preview and crops are all that's needed

        if result["invoice_flagged"]:
            self.write({
                "state": "failed",
                "failure_reason": "Couldn't find the printed table on this scan "
                                  f"({result['reason']}). Check the scan, or create the invoice by hand.",
            })
            return

        products = {
            p.name.lower(): p for p in self.env["product.product"].search([("sale_ok", "=", True)])
        }
        lines = []
        for row in result["rows"]:
            digit_crops = {
                key: row.get(f"{key}_digit_crops") or []
                for key in ("quantity", "return", "total_price")
            }
            qty_flags = row["quantity_flags"]
            ret_flags = row["return_flags"]
            price_flags = row["total_price_flags"]
            product = products.get(row["product_name"].lower())
            line = {
                "scan_id": self.id,
                "sequence": row["row_index"],
                "product_name": row["product_name"],
                "product_id": product.id if product else False,
                "quantity": row["quantity"],
                "return_qty": row["return"],
                "total_price": row["total_price"] or 0.0,
                "orig_quantity": row["quantity"],
                "orig_return": row["return"],
                "orig_total_price": row["total_price"] or 0.0,
                "qty_flags": ",".join(qty_flags),
                "return_flags": ",".join(ret_flags),
                "price_flags": ",".join(price_flags),
                "digit_crops_json": json.dumps(digit_crops),
            }
            # A picture is only kept for a flagged field -- the same
            # trade-off the desktop screen made: it's what a reviewer
            # actually needs open, and keeps the list short.
            for key, flags, image_field in (
                ("qty", qty_flags, "qty_image"),
                ("return", ret_flags, "return_image"),
                ("total_price", price_flags, "price_image"),
            ):
                path = os.path.join(work, "crops", f"row{row['row_index']:02d}_{key}.png")
                if flags and os.path.exists(path):
                    line[image_field] = _file_b64(path)
            lines.append(line)
        self.env["bakery.invoice.scan.line"].create(lines)
        self.state = "review"

    # ------------------------------------------------------------------
    # Approving: create the draft Customer Invoice
    # ------------------------------------------------------------------

    def action_create_draft_invoice(self):
        self.ensure_one()
        if self.move_id:
            raise UserError("A draft invoice was already created for this scan.")
        if not self.partner_id:
            raise UserError("Choose or type a customer first.")
        if not self.invoice_date:
            raise UserError("Enter the invoice date written on the paper invoice.")

        ordered = self.line_ids.filtered(lambda l: l.line_quantity > 0)
        if not ordered:
            raise UserError("Nothing is ordered on any row - there is nothing to invoice.")
        negative = self.line_ids.filtered(lambda l: l.quantity < 0 or l.return_qty < 0)
        if negative:
            raise UserError("Quantities can't be negative: " + ", ".join(negative.mapped("product_name")))
        no_price = ordered.filtered(lambda l: not l.total_price)
        if no_price:
            raise UserError(
                "These rows have a quantity but no Total Price - fill the price in "
                "first, so nothing reaches the invoice priced at $0:\n" + ", ".join(no_price.mapped("product_name"))
            )
        no_product = ordered.filtered(lambda l: not l.product_id)
        if no_product:
            raise UserError(
                "These products don't exist in Odoo yet - create or pick them first:\n"
                + ", ".join(no_product.mapped("product_name"))
            )

        paper = (self.paper_number or "").strip()
        if paper:
            existing = self.env["account.move"].search(
                [("ref", "=", paper), ("move_type", "=", "out_invoice")], limit=1
            )
            if existing:
                raise UserError(
                    f"An invoice with paper number '{paper}' already exists ({existing.display_name}). "
                    "Check this isn't a duplicate before creating another."
                )

        move = self.env["account.move"].create({
            "move_type": "out_invoice",
            "partner_id": self.partner_id.id,
            "invoice_date": self.invoice_date,
            "ref": paper or False,
            "invoice_line_ids": [
                (0, 0, {
                    "product_id": line.product_id.id,
                    "name": line.product_id.display_name,
                    "quantity": line.line_quantity,
                    "price_unit": line.unit_price,
                    "tax_ids": [(6, 0, [])],  # ignore tax for now, as the paper copy does
                })
                for line in ordered
            ],
        })
        over = self.line_ids.filtered(lambda l: l.return_qty > l.quantity)
        if over:
            move.message_post(body=(
                "Warning: more was returned than ordered on these rows, which shouldn't "
                "normally happen - check against the paper copy: " + ", ".join(over.mapped("product_name"))
            ))

        self._bank_verified_digits()
        shutil.rmtree(os.path.join(_scans_dir(), str(self.id)), ignore_errors=True)
        self.write({"move_id": move.id, "state": "sent"})
        return {
            "type": "ir.actions.act_window", "res_model": "account.move",
            "res_id": move.id, "view_mode": "form",
        }

    def _bank_verified_digits(self):
        """
        Keep the small digit pictures of every field a reviewer left both
        unflagged and unchanged: that's a good sign the software read it
        right, so they become free labeled examples for retraining the digit
        model later (see CLAUDE.md, "Banking verified-correct crops").
        """
        work = os.path.join(_scans_dir(), str(self.id))
        for line in self.line_ids:
            crops = json.loads(line.digit_crops_json or "{}")
            fields_ok = {
                "quantity": not line.qty_flags and line.quantity == line.orig_quantity,
                "return": not line.return_flags and line.return_qty == line.orig_return,
                "total_price": not line.price_flags and abs(line.total_price - line.orig_total_price) < 0.005,
            }
            for key, ok in fields_ok.items():
                if not ok:
                    continue
                for crop in crops.get(key, []):
                    digit = str(crop.get("predicted_digit", ""))
                    source = os.path.join(work, crop["path"])
                    if not digit.isdigit() or not os.path.exists(source):
                        continue  # a decimal point isn't a 0-9 class
                    dest_dir = os.path.join(_digit_bank_dir(), digit)
                    os.makedirs(dest_dir, exist_ok=True)
                    shutil.copyfile(source, os.path.join(dest_dir, f"scan{self.id}_{os.path.basename(crop['path'])}"))


class BakeryInvoiceScanLine(models.Model):
    _name = "bakery.invoice.scan.line"
    _description = "Scanned invoice row"
    _order = "sequence, id"

    scan_id = fields.Many2one("bakery.invoice.scan", required=True, ondelete="cascade")
    sequence = fields.Integer()
    product_name = fields.Char("Product (as printed)", readonly=True)
    product_id = fields.Many2one("product.product", "Odoo product")
    quantity = fields.Integer("Qty")
    return_qty = fields.Integer("Return")
    total_price = fields.Float("Total price", digits=(12, 2))
    line_quantity = fields.Integer("Qty - Return", compute="_compute_amounts")
    unit_price = fields.Float("Unit $", digits=(12, 2), compute="_compute_amounts")

    orig_quantity = fields.Integer(readonly=True)
    orig_return = fields.Integer(readonly=True)
    orig_total_price = fields.Float(readonly=True)
    qty_flags = fields.Char(readonly=True)
    return_flags = fields.Char(readonly=True)
    price_flags = fields.Char(readonly=True)
    flag_text = fields.Char("Check", compute="_compute_flag_text")
    digit_crops_json = fields.Text(readonly=True)
    qty_image = fields.Image("Qty picture", max_width=300, max_height=120, attachment=True, readonly=True)
    return_image = fields.Image("Return picture", max_width=300, max_height=120, attachment=True, readonly=True)
    price_image = fields.Image("Price picture", max_width=300, max_height=120, attachment=True, readonly=True)

    @api.depends("quantity", "return_qty", "total_price")
    def _compute_amounts(self):
        for line in self:
            line.line_quantity = line.quantity - line.return_qty
            line.unit_price = round(line.total_price / line.line_quantity, 2) if line.line_quantity > 0 else 0.0

    @api.depends("qty_flags", "return_flags", "price_flags")
    def _compute_flag_text(self):
        for line in self:
            parts = []
            for label, flags in (("Qty", line.qty_flags), ("Return", line.return_flags), ("Price", line.price_flags)):
                for flag in filter(None, (flags or "").split(",")):
                    parts.append(f"{label}: {FLAG_EXPLANATIONS.get(flag, flag)}")
            line.flag_text = "; ".join(parts)
