"""
Human review screen: where a person checks an invoice the software has
already read, fixes anything wrong, says which customer it belongs to,
and approves it.

Shows one invoice at a time, reading extractions/<invoice_name>/ --
results.json plus its crops/ folder -- both already produced by
extract_invoice.py. EVERY quantity and return field is shown and is
editable, not just the ones the software flagged, because a flagged-only
screen would miss a confident misread: the model is only about 94%
accurate per digit, and when it's wrong it's often wrong confidently, so
nothing flags it (see CLAUDE.md, "How much gets reviewed"). A field the
software did flag is shown with a pink background and a plain-language
note of why, so the reviewer knows to look at those first -- but it's
still just as editable as any other field.

The customer field is one editable dropdown covering both required ways
of setting it: pick from customers.json's regular list, or type a name
that isn't on it, for a genuine one-off customer. A typed name that
matches a listed customer (ignoring case and surrounding whitespace) is
folded onto that customer's exact listed spelling rather than kept as
separately-typed text, so a small typing difference can't quietly create
what looks like a second, different customer.

Deliberately does NOT import digit_reader, alignment, or torch: this
screen only displays numbers and images extract_invoice.py already
computed and saved to disk, so it opens instantly with no model/GPU
dependency of its own.

Deliberately does NOT talk to Odoo. Approving an invoice here writes a
review.json (customer, corrected values, which fields were originally
flagged) next to that invoice's results.json -- reading that file and
pushing it into Odoo is a separate piece, built separately.

Approving also banks digit pictures for future retraining: for every
Qty/Return field the reviewer leaves BOTH unflagged and unchanged from
what the software originally read, its individual digit crops (already
saved by extract_invoice.py into digit_crops/, one small picture per
digit, in the model's own predicted label) are copied into
digit_bank/<digit>/ at the project root. Leaving a field alone is a
strong signal the model's read of it was actually right, so this
builds a growing pile of free labeled training data with nobody having
to hand-label anything new -- see CLAUDE.md, "Banking verified-correct
crops as future training data". A field the reviewer corrected is
skipped: if the software had mis-split the digits in the first place,
the corrected number can't always be cleanly matched back onto which
individual digit picture was wrong.

Controls are mouse-driven (unlike label_tool.py / calibrate_template.py's
keyboard shortcuts) since this screen is meant for day-to-day use by
whoever reviews invoices, not just the person building the pipeline.

Run with:  python review_screen.py
       or: python review_screen.py "some invoice folder name"
           (opens that invoice first instead of the first one found)
"""

import argparse
import json
import os
import shutil
import tkinter as tk
from datetime import datetime
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

EXTRACTIONS_DIR = "extractions"
CUSTOMERS_PATH = "customers.json"

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

        self.invoice_names = sorted(
            name for name in os.listdir(EXTRACTIONS_DIR)
            if os.path.isfile(os.path.join(EXTRACTIONS_DIR, name, "results.json"))
        )
        if not self.invoice_names:
            raise FileNotFoundError(
                f"No extracted invoices found in '{EXTRACTIONS_DIR}/'. "
                f"Run extract_invoice.py on a scan first."
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

    def _on_mouse_wheel(self, event):
        """Mouse wheel scrolled over the row list: scroll it vertically."""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def load_invoice(self):
        """
        Load the invoice at self.index: read its results.json (and its
        review.json, if it's already been approved once before) and
        rebuild the row list from scratch.
        """
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

        for child in self.rows_frame.winfo_children():
            child.destroy()
        self._crop_images = []
        self.row_widgets = []

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
            self._update_status(flagged_invoice=True)
            return

        self.approve_button.config(state=tk.NORMAL)
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

        def recompute(*_args):
            """Keep the displayed line quantity matched to whatever is currently typed, even mid-edit."""
            try:
                line_var.set(str(int(qty_var.get()) - int(ret_var.get())))
            except ValueError:
                line_var.set("?")  # mid-edit / not a whole number yet -- resolved at save time

        qty_var.trace_add("write", recompute)
        ret_var.trace_add("write", recompute)
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

        line_frame = tk.Frame(frame)
        line_frame.grid(row=0, column=3, rowspan=2, sticky="n", padx=(12, 6), pady=4)
        tk.Label(line_frame, text="Line qty", font=("Helvetica", 9, "bold")).pack(anchor="w")
        tk.Label(line_frame, textvariable=line_var, font=("Helvetica", 11, "bold")).pack(anchor="w")

        self.row_widgets.append({
            "row_index": row_index,
            "product_name": row["product_name"],
            "qty_var": qty_var,
            "ret_var": ret_var,
            "original_quantity": row["quantity"],
            "original_return": row["return"],
            "quantity_was_flagged": bool(row["quantity_flags"]),
            "return_was_flagged": bool(row["return_flags"]),
            # .get(..., []), not [row[...]], so a results.json written
            # before digit-crop saving existed still loads instead of
            # raising a KeyError -- it just has nothing to bank.
            "quantity_digit_crops": row.get("quantity_digit_crops", []),
            "return_digit_crops": row.get("return_digit_crops", []),
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
        already_approved = " (already approved)" if os.path.exists(self.review_path) else ""
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
        """
        Validate every field on screen, then write review.json:
        the chosen customer, and each row's corrected Qty/Return/line
        quantity alongside what the software originally read and
        whether it had flagged that field -- so the piece that
        eventually pushes this into Odoo (not built here) has both the
        human-approved numbers and a record of what got corrected.
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

            # Bank a field's digits only when it's both unflagged and
            # left exactly as the software read it -- see module
            # docstring for why a corrected field is skipped.
            if not w["quantity_was_flagged"] and qty == w["original_quantity"]:
                self._bank_digit_crops(invoice_dir, invoice_name, w["quantity_digit_crops"])
            if not w["return_was_flagged"] and ret == w["original_return"]:
                self._bank_digit_crops(invoice_dir, invoice_name, w["return_digit_crops"])

            rows_out.append({
                "row_index": w["row_index"],
                "product_name": w["product_name"],
                "quantity": qty,
                "original_quantity": w["original_quantity"],
                "quantity_was_flagged": w["quantity_was_flagged"],
                "return": ret,
                "original_return": w["original_return"],
                "return_was_flagged": w["return_was_flagged"],
                "line_quantity": qty - ret,
            })

        review = {
            "invoice_name": invoice_name,
            "customer": customer,
            "approved": True,
            "reviewed_at": datetime.now().isoformat(timespec="seconds"),
            "rows": rows_out,
        }
        with open(self.review_path, "w") as f:
            json.dump(review, f, indent=2)

        messagebox.showinfo("Saved", f"Saved review for '{invoice_name}'.")
        self._update_status(flagged_invoice=False)

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
