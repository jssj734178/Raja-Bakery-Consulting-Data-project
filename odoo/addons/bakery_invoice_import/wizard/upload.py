import base64
import uuid

from odoo import fields, models
from odoo.exceptions import UserError


class BakeryInvoiceUpload(models.TransientModel):
    _name = "bakery.invoice.upload"
    _description = "Upload scanned invoices"

    pdf_file = fields.Binary("Scanned PDF", required=True)
    pdf_filename = fields.Char()

    def action_upload(self):
        import pymupdf

        self.ensure_one()
        try:
            source = pymupdf.open(stream=base64.b64decode(self.pdf_file), filetype="pdf")
        except Exception:
            raise UserError("That file couldn't be opened as a PDF.")
        base = (self.pdf_filename or "scan").rsplit(".", 1)[0]
        # One invoice per page (a day's batch is a single multi-page PDF),
        # so every page becomes its own scan to read and check separately.
        scans = self.env["bakery.invoice.scan"]
        batch_ref = uuid.uuid4().hex
        for number in range(len(source)):
            single = pymupdf.open()
            single.insert_pdf(source, from_page=number, to_page=number)
            scans |= scans.create({
                "name": f"{base} - page {number + 1}",
                "batch_ref": batch_ref,
                "page_number": number + 1,
                "pdf_page": base64.b64encode(single.tobytes()),
            })
            single.close()
        source.close()
        # Ask the background reader to start straight away instead of waiting its minute.
        self.env.ref("bakery_invoice_import.cron_read_queued_scans")._trigger()
        return {
            "type": "ir.actions.act_window",
            "name": "Scanned invoices",
            "res_model": "bakery.invoice.scan",
            "view_mode": "list,form",
            "domain": [("id", "in", scans.ids)],
        }
