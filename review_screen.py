"""
Human review screen: where a person checks an invoice the software has
already read, fixes anything wrong, says which customer it belongs to,
and approves it.

Shows one invoice at a time, reading extractions/<invoice_name>/ --
results.json plus its crops/ folder -- both already produced by
extract_invoice.py. EVERY quantity, return, and Total Price field is
shown and is editable, not just the ones the software flagged, because a
flagged-only screen would miss a confident misread: the model is only
about 94% accurate per digit, and when it's wrong it's often wrong
confidently, so nothing flags it (see CLAUDE.md, "How much gets
reviewed"). A field the software did flag is shown with a pink
background and a plain-language note of why, so the reviewer knows to
look at those first -- but it's still just as editable as any other
field.

Total Price is the handwritten dollar amount for that row; a live Unit
Price (Total Price divided by line quantity) is shown alongside it,
recalculating as either is edited -- see CLAUDE.md, "Pricing decision
reversed" for why the price that ends up on the Odoo invoice line comes
from the invoice's own handwriting rather than a price list. Reading
Total Price is newer than Qty/Return and its decimal point in particular
is genuinely harder to find automatically, so it still flags somewhat
more often -- but a round of real fixes (2026-09-27) brought its real
review burden down to roughly the same ballpark as Qty/Return's, not
the "expect to check nearly every filled-in field" state it started in
(see CLAUDE.md for the full history of what was found and fixed).

The customer field is one editable dropdown covering both required ways
of setting it: pick from customers.json's regular list, or type a name
that isn't on it, for a genuine one-off customer. A typed name that
matches a listed customer (ignoring case and surrounding whitespace) is
folded onto that customer's exact listed spelling rather than kept as
separately-typed text, so a small typing difference can't quietly create
what looks like a second, different customer.

Deliberately does NOT import digit_reader, alignment, or torch directly:
this screen only displays numbers and images extract_invoice.py already
computed and saved to disk, so it opens instantly with no model/GPU
dependency of its own. The "Upload PDF..." button (top bar) is the one
exception that touches that pipeline at all -- it runs pdf_to_images.py
and extract_invoice.py as SEPARATE SUBPROCESSES in a background thread,
not as direct imports, specifically so this screen's own fast startup
is never affected by them; only clicking that button pays their ~20+
second model-loading cost, and the window stays responsive while it
runs. This is the quick, desktop-only way to feed a new scan through the
pipeline described in CLAUDE.md's "Next to build: getting this into
Odoo" -- useful now, while the real long-term intake path (an Odoo
module) is still being built.

The invoice's own handwritten date and its printed invoice number
(top bar, next to Customer) are both typed in by hand for now rather
than read automatically -- there's no calibrated box yet for either
one, and reading the invoice number specifically means reading PRINTED
(not handwritten) digits, which the model has never been tested on. See
CLAUDE.md, "the paper invoice number becomes a label" for why both are
needed on the eventual Odoo invoice regardless. The date is required
before approving; the invoice number is allowed to stay blank, since
CLAUDE.md notes it may sometimes be cut off, obscured, or missing on a
given scan.

Approving writes a review.json (customer, invoice date, invoice number,
corrected values, which fields were originally flagged) next to that
invoice's results.json. Talking to Odoo itself is kept in a separate
file, send_to_odoo.py, imported here only for the "Send to Odoo" button
-- that keeps this screen's own imports light (see above) and keeps the
actual Odoo business rules (which rows become invoice lines, what gets
blocked, what gets a warning note) readable on their own rather than
tangled into this file's widget code. Clicking that button re-runs the
same save as Approve & Save (so what's sent always matches what's on
screen) and then pushes to Odoo in a background thread, the same
pattern Upload PDF uses, so the window doesn't freeze while waiting on
the network. Once sent, the invoice's Odoo draft number is recorded in
its review.json and the button disables itself, so the same invoice
can't be sent to Odoo a second time by accident.

Approving also banks digit pictures for future retraining: for every
Qty/Return/Total Price field the reviewer leaves BOTH unflagged and
unchanged from what the software originally read, its individual digit
crops (already saved by extract_invoice.py into digit_crops/, one small
picture per digit, in the model's own predicted label) are copied into
digit_bank/<digit>/ at the project root -- for Total Price, only the
actual digit crops, never the decimal point itself, since it isn't a
0-9 class. Leaving a field alone is a strong signal the model's read of
it was actually right, so this builds a growing pile of free labeled
training data with nobody having to hand-label anything new -- see
CLAUDE.md, "Banking verified-correct crops as future training data". A
field the reviewer corrected is skipped: if the software had mis-split
the digits in the first place, the corrected number can't always be
cleanly matched back onto which individual digit picture was wrong.

Controls are mouse-driven (unlike label_tool.py / calibrate_template.py's
keyboard shortcuts) since this screen is meant for day-to-day use by
whoever reviews invoices, not just the person building the pipeline.

Run with:  python review_screen.py
       or: python review_screen.py "some invoice folder name"
           (opens that invoice first instead of the first one found)
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from send_to_odoo import SendToOdooError, send_invoice

EXTRACTIONS_DIR = "extractions"
CUSTOMERS_PATH = "customers.json"
INVOICES_DIR = "invoices"

# Where a field's individual digit crops get copied once a reviewer has
# left it both unflagged and unchanged -- see module docstring and
# CLAUDE.md, "Banking verified-correct crops as future training data".
# One subfolder per digit (0-9), the same layout label_tool.py's own
# invoice_digits/ uses, so these can later be folded straight into a
# retraining run.
DIGIT_BANK_DIR = "digit_bank"

# (width, height) bounding box a crop is scaled down to fit inside,
# keeping its own aspect ratio -- see Image.thumbnail below. Saved crops
# run close to 1000x300px (a wide strip, not a square), so width is
# almost always the binding constraint.
CROP_DISPLAY_MAX_SIZE = (260, 100)

# Light red: marks a field the software itself wants a human to
# double-check. Chosen to be obvious at a glance down a long list of
# rows, but not so dark it fights with the black handwriting in the
# crop sitting on top of it.
FLAGGED_BG = "#ffe3e3"

# Every reason digit_reader.py / extract_invoice.py can flag something
# for, translated into what actually happened on the page -- these are
# shown directly to whoever is reviewing, so they need to make sense
# without knowing how the software works. Kept in one place so the
# wording only has to be decided once. Mirrors the explanations already
# written out in CLAUDE.md's pipeline notes.
FLAG_EXPLANATIONS = {
    "too_many_blobs": "more separate marks than this box normally has -- possibly stray marks, or digits that shouldn't be grouped together",
    "possible_merged_digits": "a mark much wider than it is tall -- possibly two digits touching and read as one",
    "possible_split_digit": "a leftover stroke -- possibly a digit that was written in disconnected pieces",
    "unreadable_ink": "there was a faint mark here, too small to read -- treated as blank, but worth a look",
    "low_confidence": "the software wasn't confident about a digit here",
    "leading_zero_digit": "an extra mark was read as a \"0\" in front of the real number -- it didn't change the number shown, but the mark itself should be checked",
    "return_exceeds_quantity": "the return read here is bigger than the quantity ordered -- usually a misread, e.g. the printed price next door read as a number",
    "no_decimal_point": "no decimal point was found in this box -- a Total Price should always have one (e.g. \"27.00\")",
    "ambiguous_decimal_point": "more than one mark in this box looked like it could be the decimal point, so the software isn't sure which one is real",
    "total_price_without_quantity": "a Total Price was read here, but this row's Qty minus Return isn't a positive number, so a unit price couldn't be worked out",
    "quantity_without_total_price": "this row has a quantity but no Total Price was read here -- every filled-in row should have one",
    "border_not_detected": "the software couldn't find the printed table on this scan",
    "aspect_ratio_mismatch": "the detected table doesn't look like the usual form -- possibly a bad or crooked scan",
}


class ReviewScreen:
    """
    A tkinter app showing one invoice extraction at a time for a human
    to correct, assign a customer to, and approve -- see module
    docstring for the full picture.
    """

    def __init__(self, root: tk.Tk, start_invoice: str = None):
        """
        Load the list of extracted invoices and the customer list, build
        the (mostly empty) window, then load the first invoice.

        Args:
            root: the tkinter root window this app runs inside.
            start_invoice: an extractions/ subfolder name to open first,
                or None to start with the first invoice found.
        """
        self.root = root
        self.root.title("Invoice Review")
        self.root.geometry("1150x800")

        # An empty (or missing) extractions/ folder is a normal starting
        # state now that Upload PDF exists to fill it from inside the
        # app, not just an error condition to refuse to start over --
        # see load_invoice() for how the row area handles having
        # nothing to show yet.
        os.makedirs(EXTRACTIONS_DIR, exist_ok=True)
        self.invoice_names = sorted(
            name for name in os.listdir(EXTRACTIONS_DIR)
            if os.path.isfile(os.path.join(EXTRACTIONS_DIR, name, "results.json"))
        )

        with open(CUSTOMERS_PATH) as f:
            self.customers = json.load(f)["customers"]

        self.index = 0
        if start_invoice in self.invoice_names:
            self.index = self.invoice_names.index(start_invoice)

        # Keeps every PhotoImage currently on screen alive -- tkinter
        # doesn't keep its own reference, so without this each image
        # would be garbage-collected the moment this list's previous
        # contents were replaced, leaving blank boxes on screen.
        self._crop_images = []
        # One dict per row currently on screen, holding what
        # approve_and_save() needs to read back out of it (the Vars,
        # plus the row's own identity and original flag state).
        self.row_widgets = []

        self._build_widgets()
        self.load_invoice()

    def _build_widgets(self):
        """Build the (per-invoice-content-free) window chrome: the top bar, customer bar, scrollable row area, and bottom bar."""
        top = tk.Frame(self.root)
        top.pack(fill=tk.X, padx=8, pady=(8, 2))

        tk.Label(top, text="Invoice:").pack(side=tk.LEFT)
        self.invoice_var = tk.StringVar()
        self.invoice_picker = ttk.Combobox(
            top, textvariable=self.invoice_var, values=self.invoice_names,
            state="readonly", width=40,
        )
        self.invoice_picker.pack(side=tk.LEFT, padx=(4, 12))
        self.invoice_picker.bind("<<ComboboxSelected>>", self._on_invoice_picked)

        tk.Button(top, text="< Prev", command=self.prev_invoice).pack(side=tk.LEFT)
        tk.Button(top, text="Next >", command=self.next_invoice).pack(side=tk.LEFT, padx=(4, 12))

        # A quick way to feed a new scan through the pipeline without
        # typing commands, while the real long-term intake path (the
        # Odoo module) is being built -- see CLAUDE.md, "Also planned,
        # not yet started". Runs pdf_to_images.py then extract_invoice.py
        # as SUBPROCESSES, not direct imports, for the same reason this
        # whole file avoids importing torch/digit_reader (see module
        # docstring): those two scripts pull in the model and take
        # ~20+ seconds just to start, which would slow down every single
        # launch of this screen if paid up front instead of only when
        # a PDF is actually uploaded.
        self.upload_button = tk.Button(top, text="Upload PDF...", command=self._upload_pdf)
        self.upload_button.pack(side=tk.LEFT)

        self.status_label = tk.Label(top, text="", fg="#444444")
        self.status_label.pack(side=tk.LEFT, padx=(12, 0))

        cust = tk.Frame(self.root)
        cust.pack(fill=tk.X, padx=8, pady=(0, 2))
        tk.Label(cust, text="Customer:").pack(side=tk.LEFT)
        self.customer_var = tk.StringVar()
        # Deliberately NOT state="readonly", unlike invoice_picker above:
        # picking from the list is the normal path, but typing a name
        # that isn't on the list has to work too, for one-off customers
        # who don't belong on it (see CLAUDE.md, "Customer selection").
        # An editable combobox gives both in one control -- the dropdown
        # for picking, free typing for anything else -- rather than two
        # separate widgets that could disagree about which one "wins".
        self.customer_picker = ttk.Combobox(
            cust, textvariable=self.customer_var, values=self.customers, width=45,
        )
        self.customer_picker.pack(side=tk.LEFT, padx=4)
        # Fold a typed name onto the list as soon as the field is left,
        # not just at save time, so the correction is visible before
        # approving rather than as a surprise afterwards.
        self.customer_picker.bind("<FocusOut>", self._on_customer_focus_out)

        # Both typed by hand for now, not read automatically: the invoice
        # date has no calibrated box at all yet, and the printed invoice
        # number's box exists on the form but reading printed (not
        # handwritten) digits automatically hasn't been built or measured
        # -- see CLAUDE.md, "the paper invoice number becomes a label".
        # Kept here rather than skipped entirely because both are needed
        # on the eventual Odoo invoice (its own date, and a searchable
        # reference for matching back to the paper copy / catching a
        # duplicate submission).
        tk.Label(cust, text="   Date (YYYY-MM-DD):").pack(side=tk.LEFT)
        self.invoice_date_var = tk.StringVar()
        tk.Entry(cust, textvariable=self.invoice_date_var, width=12).pack(side=tk.LEFT, padx=4)

        tk.Label(cust, text="Invoice #:").pack(side=tk.LEFT)
        self.invoice_number_var = tk.StringVar()
        # No validation on this one -- CLAUDE.md flags that the printed
        # number may sometimes be cut off, obscured, or missing on a
        # given scan, so leaving it blank has to be a normal, allowed
        # state, not an error to fix before approving.
        tk.Entry(cust, textvariable=self.invoice_number_var, width=12).pack(side=tk.LEFT, padx=4)

        tk.Label(
            cust,
            text="   Pink = the software flagged this field for you to double-check. Every field can be edited either way.",
            fg="#7a0000",
        ).pack(side=tk.LEFT)

        # Only "Product" gets a column header here -- Qty and Return
        # are labeled on every row instead (see _build_field), since
        # this header is packed at fixed character widths while the
        # rows below are gridded and sized by their images, and the two
        # don't line up column-for-column closely enough to trust for
        # anything narrower than "the far left" and "the far right".
        header = tk.Frame(self.root)
        header.pack(fill=tk.X, padx=8)
        bold = ("Helvetica", 10, "bold")
        tk.Label(header, text="Product", font=bold, anchor="w").pack(side=tk.LEFT)

        # Standard tkinter "scrollable frame" idiom: a Canvas is the
        # only widget that can scroll, so the actual row widgets live in
        # a plain Frame placed *inside* the canvas, and the canvas's
        # scrollregion is kept matched to that frame's real size every
        # time it changes (rows added/removed on switching invoices).
        container = tk.Frame(self.root)
        container.pack(fill=tk.BOTH, expand=True, padx=8, pady=(2, 0))
        self.canvas = tk.Canvas(container, highlightthickness=0)
        scrollbar = tk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.rows_frame = tk.Frame(self.canvas)
        self.rows_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.create_window((0, 0), window=self.rows_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        # Bound globally (bind_all), not just on the canvas, so the
        # wheel scrolls the list no matter which child widget (an
        # Entry, say) the mouse happens to be over at the time.
        self.canvas.bind_all("<MouseWheel>", self._on_mouse_wheel)

        bottom = tk.Frame(self.root)
        bottom.pack(fill=tk.X, padx=8, pady=8)
        self.approve_button = tk.Button(
            bottom, text="Approve & Save", command=self.approve_and_save,
            bg="#2e7d32", fg="white", font=("Helvetica", 10, "bold"),
        )
        self.approve_button.pack(side=tk.RIGHT)
        # Saves (same as Approve & Save, so what's sent always matches
        # what's on screen -- see _send_to_odoo) and then pushes to the
        # live Odoo server, so this is deliberately a separate, later
        # step a reviewer takes on purpose, not something that happens
        # automatically just from approving locally.
        self.send_to_odoo_button = tk.Button(
            bottom, text="Send to Odoo", command=self._send_to_odoo,
            bg="#1565c0", fg="white", font=("Helvetica", 10, "bold"),
        )
        self.send_to_odoo_button.pack(side=tk.RIGHT, padx=(0, 8))

    def _on_mouse_wheel(self, event):
        """Mouse wheel scrolled over the row list: scroll it vertically."""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def load_invoice(self):
        """
        Load the invoice at self.index: read its results.json (and its
        review.json, if it's already been approved once before) and
        rebuild the row list from scratch.
        """
        for child in self.rows_frame.winfo_children():
            child.destroy()
        self._crop_images = []
        self.row_widgets = []

        if not self.invoice_names:
            tk.Label(
                self.rows_frame,
                text="No invoices yet -- click \"Upload PDF...\" above to add one.",
                fg="#444444", font=("Helvetica", 11),
            ).pack(anchor="w", pady=30, padx=10)
            self.approve_button.config(state=tk.DISABLED)
            self.send_to_odoo_button.config(state=tk.DISABLED)
            self.status_label.config(text="No invoices loaded")
            return

        name = self.invoice_names[self.index]
        self.invoice_var.set(name)
        invoice_dir = os.path.join(EXTRACTIONS_DIR, name)

        with open(os.path.join(invoice_dir, "results.json")) as f:
            self.results = json.load(f)

        self.review_path = os.path.join(invoice_dir, "review.json")
        existing_review = None
        if os.path.exists(self.review_path):
            with open(self.review_path) as f:
                existing_review = json.load(f)
        # Re-opening an already-approved invoice starts from the
        # reviewer's own corrected values, not the original software
        # reading -- otherwise every re-open would silently discard a
        # previous review's corrections.
        self.customer_var.set(existing_review["customer"] if existing_review else "")
        self.invoice_date_var.set(existing_review.get("invoice_date", "") if existing_review else "")
        self.invoice_number_var.set(existing_review.get("paper_invoice_number", "") if existing_review else "")
        # Carried forward on every re-save (see _build_review_dict) so
        # that re-approving an invoice after it's been sent to Odoo
        # can't accidentally clear the record of that -- see
        # send_to_odoo.send_invoice, which refuses to send an invoice
        # a second time while this is set.
        self.odoo_invoice_id = existing_review.get("odoo_invoice_id") if existing_review else None

        if self.results.get("invoice_flagged"):
            reason = self.results.get("reason")
            tk.Label(
                self.rows_frame,
                text=(
                    "This invoice could not be read automatically "
                    f"({FLAG_EXPLANATIONS.get(reason, reason)}).\n"
                    "It needs to be entered by hand -- there is nothing here to review."
                ),
                fg="#b71c1c", justify=tk.LEFT, wraplength=900, font=("Helvetica", 11),
            ).pack(anchor="w", pady=30, padx=10)
            self.approve_button.config(state=tk.DISABLED)
            self.send_to_odoo_button.config(state=tk.DISABLED)
            self._update_status(flagged_invoice=True)
            return

        self.approve_button.config(state=tk.NORMAL)
        # Sending is blocked once an invoice has already been sent
        # once (see send_to_odoo.send_invoice's own guard) -- the
        # button itself reflects that immediately, rather than letting
        # a reviewer click it and only find out from an error popup.
        self.send_to_odoo_button.config(state=tk.DISABLED if self.odoo_invoice_id else tk.NORMAL)
        existing_rows_by_index = (
            {r["row_index"]: r for r in existing_review["rows"]} if existing_review else {}
        )
        for row in self.results["rows"]:
            self._build_row(invoice_dir, row, existing_rows_by_index.get(row["row_index"]))

        self._update_status(flagged_invoice=False)

    def _build_row(self, invoice_dir: str, row: dict, existing: dict):
        """
        Add one product row to the row list: its name, its Qty and
        Return crops with editable values next to each, and a live
        line-quantity (Qty minus Return) that updates as those values
        are edited.

        Args:
            invoice_dir: extractions/<invoice_name>, where this row's
                crop images live.
            row: this row's entry from results.json.
            existing: this row's entry from a previous review.json, if
                this invoice was already reviewed once before, else None.
        """
        row_index = row["row_index"]
        frame = tk.Frame(self.rows_frame, bd=1, relief=tk.SOLID)
        frame.pack(fill=tk.X, pady=2, ipady=4)

        tk.Label(
            frame, text=row["product_name"], anchor="w", justify=tk.LEFT, wraplength=200,
        ).grid(row=0, column=0, rowspan=2, sticky="nw", padx=(6, 10), pady=4)

        qty_var = tk.StringVar(value=str(existing["quantity"] if existing else row["quantity"]))
        ret_var = tk.StringVar(value=str(existing["return"] if existing else row["return"]))
        line_var = tk.StringVar()
        unit_price_var = tk.StringVar()

        # .get(...), not row[...], so a results.json written before
        # Total Price reading existed still loads (as a blank, editable
        # field) instead of raising a KeyError.
        original_total_price = row.get("total_price")
        existing_total_price = existing.get("total_price") if existing else None
        total_price_value = existing_total_price if existing is not None else original_total_price
        total_price_var = tk.StringVar(
            value=f"{total_price_value:.2f}" if total_price_value is not None else ""
        )

        def recompute(*_args):
            """
            Keep the displayed line quantity and unit price matched to
            whatever is currently typed, even mid-edit. Unit price is
            derived (Total Price / line quantity, see CLAUDE.md's
            "Pricing decision reversed") rather than typed directly --
            editing either Qty/Return or Total Price updates it live.
            """
            try:
                qty, ret = int(qty_var.get()), int(ret_var.get())
                line_var.set(str(qty - ret))
            except ValueError:
                line_var.set("?")  # mid-edit / not a whole number yet -- resolved at save time
                unit_price_var.set("-")
                return

            typed_total_price = total_price_var.get().strip()
            try:
                total_price = float(typed_total_price) if typed_total_price else None
            except ValueError:
                unit_price_var.set("?")
                return
            if total_price is not None and qty - ret > 0:
                unit_price_var.set(f"{total_price / (qty - ret):.2f}")
            else:
                unit_price_var.set("-")

        qty_var.trace_add("write", recompute)
        ret_var.trace_add("write", recompute)
        total_price_var.trace_add("write", recompute)
        recompute()

        self._build_field(
            frame, col=1, label="Qty",
            crop_path=os.path.join(invoice_dir, "crops", f"row{row_index:02d}_qty.png"),
            var=qty_var, flags=row["quantity_flags"],
        )
        self._build_field(
            frame, col=2, label="Return",
            crop_path=os.path.join(invoice_dir, "crops", f"row{row_index:02d}_return.png"),
            var=ret_var, flags=row["return_flags"],
        )
        self._build_field(
            frame, col=3, label="Total $",
            crop_path=os.path.join(invoice_dir, "crops", f"row{row_index:02d}_total_price.png"),
            var=total_price_var, flags=row.get("total_price_flags", []),
        )

        line_frame = tk.Frame(frame)
        line_frame.grid(row=0, column=4, rowspan=2, sticky="n", padx=(12, 6), pady=4)
        tk.Label(line_frame, text="Line qty", font=("Helvetica", 9, "bold")).pack(anchor="w")
        tk.Label(line_frame, textvariable=line_var, font=("Helvetica", 11, "bold")).pack(anchor="w")
        tk.Label(line_frame, text="Unit $", font=("Helvetica", 9, "bold")).pack(anchor="w", pady=(6, 0))
        tk.Label(line_frame, textvariable=unit_price_var, font=("Helvetica", 11, "bold")).pack(anchor="w")

        self.row_widgets.append({
            "row_index": row_index,
            "product_name": row["product_name"],
            "qty_var": qty_var,
            "ret_var": ret_var,
            "total_price_var": total_price_var,
            "original_quantity": row["quantity"],
            "original_return": row["return"],
            "original_total_price": original_total_price,
            "quantity_was_flagged": bool(row["quantity_flags"]),
            "return_was_flagged": bool(row["return_flags"]),
            "total_price_was_flagged": bool(row.get("total_price_flags")),
            # .get(..., []), not [row[...]], so a results.json written
            # before digit-crop saving existed still loads instead of
            # raising a KeyError -- it just has nothing to bank.
            "quantity_digit_crops": row.get("quantity_digit_crops", []),
            "return_digit_crops": row.get("return_digit_crops", []),
            "total_price_digit_crops": row.get("total_price_digit_crops", []),
        })

    def _build_field(self, parent: tk.Frame, col: int, label: str, crop_path: str, var: tk.StringVar, flags: list):
        """
        Build one Qty or Return field: a small heading, its crop image,
        an editable value entry below it, and -- only if the software
        flagged this specific field -- a pink background plus a
        plain-language note of why.

        The heading is repeated on every single row, rather than relying
        only on the one-time column header above the scrollable list,
        because that header is laid out by a different, simpler mechanism
        (packed, fixed character widths) than each row (gridded, sized by
        its images) -- the two don't line up column-for-column, so a
        reader scanning down the list needs a label attached to the field
        itself to be sure which one they're looking at.

        Args:
            parent: this row's frame; the field is grid-placed into it.
            col: which grid column to place the field in.
            label: "Qty" or "Return", shown above the image.
            crop_path: path to the saved review crop for this field.
            var: the StringVar backing the editable value, already
                wired to the row's live line-quantity recompute.
            flags: this field's flag_reasons list from results.json
                (empty if the software didn't flag it).
        """
        flagged = bool(flags)
        bg = FLAGGED_BG if flagged else parent.cget("bg")
        field_frame = tk.Frame(parent, bg=bg, bd=2 if flagged else 0, relief=tk.GROOVE if flagged else tk.FLAT)
        field_frame.grid(row=0, column=col, rowspan=2, padx=6, pady=2, sticky="n")

        tk.Label(field_frame, text=label, bg=bg, font=("Helvetica", 9, "bold")).pack(anchor="w")

        # Only a flagged field's photo is worth its screen space. With
        # up to 24 rows on screen, showing all ~48 images regardless of
        # flag state made the list too tall to work through comfortably
        # -- an unflagged field is still fully editable, just without
        # its source photo displayed alongside it.
        if flagged:
            photo = ImageTk.PhotoImage(_load_thumbnail(crop_path))
            self._crop_images.append(photo)
            tk.Label(field_frame, image=photo, bg=bg).pack()

        tk.Entry(field_frame, textvariable=var, width=6, justify=tk.CENTER, font=("Helvetica", 11)).pack(pady=3)

        if flagged:
            reason_text = "; ".join(FLAG_EXPLANATIONS.get(reason, reason) for reason in flags)
            tk.Label(
                field_frame, text=reason_text, bg=bg, fg="#7a0000",
                wraplength=230, justify=tk.LEFT, font=("Helvetica", 8),
            ).pack(padx=4, pady=(0, 4))

    def _update_status(self, flagged_invoice: bool):
        """Refresh the status label: position in the list, review state, and (if applicable) how many rows are flagged."""
        name = self.invoice_names[self.index]
        if self.odoo_invoice_id:
            already_approved = f" (sent to Odoo as draft invoice #{self.odoo_invoice_id})"
        elif os.path.exists(self.review_path):
            already_approved = " (already approved)"
        else:
            already_approved = ""
        position = f"[{self.index + 1}/{len(self.invoice_names)}] {name}{already_approved}"
        if flagged_invoice:
            self.status_label.config(text=f"{position} -- needs manual handling")
        else:
            flagged_rows = sum(1 for r in self.results["rows"] if r["flagged"])
            total = len(self.results["rows"])
            self.status_label.config(text=f"{position} -- {flagged_rows}/{total} rows flagged")

    def prev_invoice(self):
        """Move to the previous invoice in the list, if there is one."""
        if self.index > 0:
            self.index -= 1
            self.load_invoice()

    def next_invoice(self):
        """Move to the next invoice in the list, if there is one."""
        if self.index < len(self.invoice_names) - 1:
            self.index += 1
            self.load_invoice()

    def _on_invoice_picked(self, _event):
        """Invoice picker dropdown: jump straight to whichever invoice was picked."""
        self.index = self.invoice_names.index(self.invoice_var.get())
        self.load_invoice()

    def _upload_pdf(self):
        """
        Pick a PDF, render it to page images, and run the full
        extraction pipeline on just those new pages -- run in a
        background thread so the window stays responsive during the
        ~20+ second model startup plus a few seconds per page (see
        CLAUDE.md, "Speed-up"), rather than freezing while it works.
        """
        pdf_path = filedialog.askopenfilename(
            title="Select an invoice PDF", filetypes=[("PDF files", "*.pdf")]
        )
        if not pdf_path:
            return

        self.upload_button.config(state=tk.DISABLED)
        self.approve_button.config(state=tk.DISABLED)
        self.status_label.config(text=f"Processing {os.path.basename(pdf_path)}... this can take a minute.")

        thread = threading.Thread(target=self._run_upload_pipeline, args=(pdf_path,), daemon=True)
        thread.start()

    def _run_upload_pipeline(self, pdf_path: str):
        """
        The actual rendering + extraction work, run off the main thread.
        Both steps are separate SUBPROCESSES (not direct calls) for the
        same reason this file avoids importing torch/digit_reader at
        all -- see module docstring and the Upload PDF button's own
        comment. Schedules _on_upload_finished back onto the main thread
        with the outcome, since tkinter widgets may only be touched from
        the thread that created them.
        """
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]

        render = subprocess.run(
            [sys.executable, "pdf_to_images.py", pdf_path],
            capture_output=True, text=True,
        )
        if render.returncode != 0:
            self.root.after(0, self._on_upload_finished, False, render.stderr or render.stdout, [])
            return

        new_pages = sorted(glob.glob(os.path.join(INVOICES_DIR, f"{base_name}_page*.png")))
        if not new_pages:
            self.root.after(0, self._on_upload_finished, False, "No pages were rendered from that PDF.", [])
            return

        extract = subprocess.run(
            [sys.executable, "extract_invoice.py", *new_pages, "--output-dir", EXTRACTIONS_DIR],
            capture_output=True, text=True,
        )
        if extract.returncode != 0:
            self.root.after(0, self._on_upload_finished, False, extract.stderr or extract.stdout, [])
            return

        new_invoice_names = [os.path.splitext(os.path.basename(p))[0] for p in new_pages]
        self.root.after(0, self._on_upload_finished, True, extract.stdout, new_invoice_names)

    def _on_upload_finished(self, success: bool, message: str, new_invoice_names: list):
        """
        Back on the main thread: report the outcome, and on success,
        refresh the invoice list and jump straight to the first
        newly-added page so the reviewer doesn't have to hunt for it.
        """
        self.upload_button.config(state=tk.NORMAL)
        self.approve_button.config(state=tk.NORMAL)

        if not success:
            self.status_label.config(text="")
            messagebox.showerror(
                "Upload failed",
                f"Could not process that PDF:\n\n{message[-2000:]}",
            )
            return

        self.invoice_names = sorted(
            name for name in os.listdir(EXTRACTIONS_DIR)
            if os.path.isfile(os.path.join(EXTRACTIONS_DIR, name, "results.json"))
        )
        self.invoice_picker.config(values=self.invoice_names)
        if new_invoice_names and new_invoice_names[0] in self.invoice_names:
            self.index = self.invoice_names.index(new_invoice_names[0])
        self.load_invoice()

        page_word = "page" if len(new_invoice_names) == 1 else "pages"
        messagebox.showinfo(
            "Upload complete",
            f"Processed {len(new_invoice_names)} {page_word}. Showing the first one now.",
        )

    def _resolve_customer_name(self, typed: str) -> str:
        """
        Fold a typed customer name onto the list it if matches one
        already there, ignoring case and surrounding whitespace, so a
        small typing difference (capitalization, a stray space) doesn't
        create what looks like a second, separate customer alongside
        the real one. Returns the list's own exact spelling on a match.

        Anything that doesn't match any listed customer is returned
        unchanged (just whitespace-trimmed) -- that's the free-text
        option working as intended, for a genuine one-off customer who
        isn't on the list at all.

        Args:
            typed: whatever is currently in the customer field.
        """
        typed = typed.strip()
        for customer in self.customers:
            if customer.strip().lower() == typed.lower():
                return customer
        return typed

    def _on_customer_focus_out(self, _event):
        """Customer field lost focus: fold it onto the list now, so a correction is visible before Approve rather than a surprise after."""
        self.customer_var.set(self._resolve_customer_name(self.customer_var.get()))

    def approve_and_save(self):
        """Validate every field on screen and write review.json, then confirm with a popup."""
        invoice_name = self._save_review()
        if invoice_name:
            messagebox.showinfo("Saved", f"Saved review for '{invoice_name}'.")

    def _send_to_odoo(self):
        """
        Save the review first -- exactly what Approve & Save does, run
        again here so whatever gets pushed to Odoo always matches
        what's currently on screen, even if something was edited since
        the last explicit Approve & Save click -- then push it to the
        live Odoo server. The actual network call runs in a background
        thread (see _run_send_to_odoo), the same way Upload PDF does,
        so the window stays responsive rather than freezing for however
        long the connection takes.
        """
        invoice_name = self._save_review()
        if not invoice_name:
            return
        self.send_to_odoo_button.config(state=tk.DISABLED)
        self.approve_button.config(state=tk.DISABLED)
        self.status_label.config(text=f"Sending '{invoice_name}' to Odoo...")
        thread = threading.Thread(target=self._run_send_to_odoo, args=(invoice_name,), daemon=True)
        thread.start()

    def _run_send_to_odoo(self, invoice_name: str):
        """The actual Odoo push, run off the main thread. Schedules _on_send_to_odoo_finished back onto the main thread with the outcome, since tkinter widgets may only be touched from the thread that created them."""
        try:
            result = send_invoice(invoice_name)
        except SendToOdooError as e:
            # A deliberate, expected stopping point (see that class's
            # own docstring) -- e.g. a missing price, an unmatched
            # product, an already-sent invoice -- so its message is
            # written to be read directly by the reviewer as-is.
            self.root.after(0, self._on_send_to_odoo_finished, invoice_name, None, str(e))
            return
        except Exception as e:
            # Anything else -- most likely the Odoo server being
            # unreachable -- isn't a message meant for a reviewer to
            # read as-is, so it's wrapped with enough context to know
            # what was being attempted.
            self.root.after(0, self._on_send_to_odoo_finished, invoice_name, None, f"Could not reach Odoo: {e}")
            return
        self.root.after(0, self._on_send_to_odoo_finished, invoice_name, result, None)

    def _on_send_to_odoo_finished(self, invoice_name: str, result: dict, error: str):
        """Back on the main thread: report the outcome, and on success, record the new Odoo invoice ID so this invoice can't be sent a second time by accident."""
        self.approve_button.config(state=tk.NORMAL)
        if error:
            self.send_to_odoo_button.config(state=tk.NORMAL)
            messagebox.showerror("Could not send to Odoo", error)
            return

        # If the reviewer navigated to a different invoice while the
        # send was in flight, don't touch this screen's own state --
        # the result already got written to the right invoice's
        # review.json inside send_invoice() regardless of what's on
        # screen right now.
        if self.invoice_names[self.index] == invoice_name:
            self.odoo_invoice_id = result["odoo_invoice_id"]
            self._update_status(flagged_invoice=False)
        else:
            self.send_to_odoo_button.config(state=tk.NORMAL)

        message = f"Sent '{invoice_name}' to Odoo as draft invoice #{result['odoo_invoice_id']}."
        if result["warning"]:
            message += "\n\n" + result["warning"]
        messagebox.showinfo("Sent to Odoo", message)

    def _save_review(self):
        """
        Validate every field on screen, then write review.json: the
        chosen customer, and each row's corrected Qty/Return/Total
        Price alongside what the software originally read and whether
        it had flagged that field -- so send_to_odoo.send_invoice (and
        anyone re-opening this invoice later) has both the
        human-approved numbers and a record of what got corrected.

        Returns this invoice's name on success, or None if something
        on screen was invalid (an error popup has already been shown
        in that case, and nothing was written).
        """
        if self.results.get("invoice_flagged"):
            # Belt and suspenders: the Approve button is already disabled
            # for a flagged invoice (see load_invoice), but that's a UI
            # state, not a hard stop -- without this check here too, any
            # future path that calls this method directly would silently
            # write out an "approved" review.json with zero rows for an
            # invoice that was never actually read.
            messagebox.showerror(
                "Cannot approve",
                "This invoice needs manual handling -- there is nothing here to approve.",
            )
            return

        # Resolved again here, not just on focus-out above, in case
        # Approve is somehow reached without the field ever losing
        # focus -- and set back onto the field so what's on screen
        # always matches what actually gets saved.
        customer = self._resolve_customer_name(self.customer_var.get())
        self.customer_var.set(customer)
        if not customer:
            messagebox.showerror("Missing customer", "Pick or type a customer before approving.")
            return

        # Required (the eventual Odoo invoice needs its own date -- see
        # CLAUDE.md, "Use the invoice's own handwritten date"), unlike
        # the invoice number just below, which is allowed to be blank.
        invoice_date = self.invoice_date_var.get().strip()
        try:
            datetime.strptime(invoice_date, "%Y-%m-%d")
        except ValueError:
            messagebox.showerror(
                "Missing or invalid date",
                "Enter the invoice's own handwritten date as YYYY-MM-DD before approving.",
            )
            return

        # Freely allowed to be blank -- see CLAUDE.md, "the paper invoice
        # number may not always be visible" -- a missing number just
        # means the (not yet built) duplicate check can't run for this
        # invoice, not that approval should be blocked on it.
        paper_invoice_number = self.invoice_number_var.get().strip()

        invoice_name = self.invoice_names[self.index]
        invoice_dir = os.path.join(EXTRACTIONS_DIR, invoice_name)

        rows_out = []
        for w in self.row_widgets:
            try:
                qty = int(w["qty_var"].get())
                ret = int(w["ret_var"].get())
                if qty < 0 or ret < 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror(
                    "Invalid value",
                    f"'{w['product_name']}' has a Qty or Return that isn't a whole "
                    f"number of 0 or more. Fix it before approving.",
                )
                return

            # Total Price is left blank for a genuinely blank row (see
            # digit_reader.classify_price -- unlike Qty/Return, a blank
            # Total Price is never assumed to mean $0), so an empty
            # field here is valid and means "no total price."
            typed_total_price = w["total_price_var"].get().strip()
            try:
                total_price = float(typed_total_price) if typed_total_price else None
                if total_price is not None and total_price < 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror(
                    "Invalid value",
                    f"'{w['product_name']}' has a Total Price that isn't a number of "
                    f"0 or more (or blank). Fix it before approving.",
                )
                return

            line_quantity = qty - ret
            unit_price = round(total_price / line_quantity, 2) if total_price is not None and line_quantity > 0 else None

            # Bank a field's digits only when it's both unflagged and
            # left exactly as the software read it -- see module
            # docstring for why a corrected field is skipped. Total
            # Price is compared rounded to cents, since that's the
            # precision the field is displayed and edited at.
            if not w["quantity_was_flagged"] and qty == w["original_quantity"]:
                self._bank_digit_crops(invoice_dir, invoice_name, w["quantity_digit_crops"])
            if not w["return_was_flagged"] and ret == w["original_return"]:
                self._bank_digit_crops(invoice_dir, invoice_name, w["return_digit_crops"])
            original_total_price = w["original_total_price"]
            total_price_unchanged = (
                total_price == original_total_price if original_total_price is None or total_price is None
                else round(total_price, 2) == round(original_total_price, 2)
            )
            if not w["total_price_was_flagged"] and total_price_unchanged:
                self._bank_digit_crops(invoice_dir, invoice_name, w["total_price_digit_crops"])

            rows_out.append({
                "row_index": w["row_index"],
                "product_name": w["product_name"],
                "quantity": qty,
                "original_quantity": w["original_quantity"],
                "quantity_was_flagged": w["quantity_was_flagged"],
                "return": ret,
                "original_return": w["original_return"],
                "return_was_flagged": w["return_was_flagged"],
                "line_quantity": line_quantity,
                "total_price": total_price,
                "original_total_price": original_total_price,
                "total_price_was_flagged": w["total_price_was_flagged"],
                "unit_price": unit_price,
            })

        review = {
            "invoice_name": invoice_name,
            "customer": customer,
            "invoice_date": invoice_date,
            "paper_invoice_number": paper_invoice_number,
            "approved": True,
            "reviewed_at": datetime.now().isoformat(timespec="seconds"),
            "rows": rows_out,
            # Carried forward as-is, not recomputed -- this method never
            # sets it itself, only _on_send_to_odoo_finished does, once
            # an actual Odoo push has succeeded. Without carrying it
            # forward here, a plain re-approve (with no Odoo push
            # involved at all) would silently erase the record of an
            # earlier successful send.
            "odoo_invoice_id": self.odoo_invoice_id,
        }
        with open(self.review_path, "w") as f:
            json.dump(review, f, indent=2)

        self._update_status(flagged_invoice=False)
        return invoice_name

    def _bank_digit_crops(self, invoice_dir: str, invoice_name: str, digit_crops: list):
        """
        Copy an already-verified field's individual digit crops into
        digit_bank/<digit>/, for later retraining -- see module
        docstring and CLAUDE.md, "Banking verified-correct crops as
        future training data". Only called for a field the reviewer
        left both unflagged and unchanged, since that's the signal the
        model's own reading of it was actually right.

        The destination filename is built from the invoice name plus
        the crop's own saved filename, which is already unique within
        that invoice (row, field, and digit position) -- so approving
        the same invoice a second time just overwrites the same files
        rather than piling up duplicates.

        Args:
            invoice_dir: extractions/<invoice_name>, where digit_crops/
                lives (written by extract_invoice.py).
            invoice_name: this invoice's own name, for the destination
                filename.
            digit_crops: a field's "quantity_digit_crops" or
                "return_digit_crops" list from results.json, each
                {"path": ..., "predicted_digit": ...}. Empty for a
                blank field -- nothing to bank.
        """
        for crop in digit_crops:
            source_path = os.path.join(invoice_dir, crop["path"])
            if not os.path.exists(source_path):
                continue  # an older results.json predating this feature
            dest_dir = os.path.join(DIGIT_BANK_DIR, crop["predicted_digit"])
            os.makedirs(dest_dir, exist_ok=True)
            dest_name = f"{invoice_name}_{os.path.basename(crop['path'])}"
            shutil.copyfile(source_path, os.path.join(dest_dir, dest_name))


def _load_thumbnail(path: str) -> Image.Image:
    """Load a saved review crop and scale it to fit CROP_DISPLAY_MAX_SIZE, keeping its own aspect ratio."""
    image = Image.open(path)
    image.thumbnail(CROP_DISPLAY_MAX_SIZE)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "invoice_name", nargs="?", default=None,
        help="An extractions/ subfolder name to open first (default: the first one found).",
    )
    args = parser.parse_args()

    root = tk.Tk()
    ReviewScreen(root, start_invoice=args.invoice_name)
    root.mainloop()


if __name__ == "__main__":
    main()
