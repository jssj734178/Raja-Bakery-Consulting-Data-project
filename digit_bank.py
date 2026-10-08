"""
Decides what label each saved digit picture gets when an invoice is
approved, for the growing bank of future training data (digit_bank/ on
the desktop, bakery_digit_bank/ inside Odoo).

The idea: once a person has looked at every field on the review screen
and approved the invoice, the final number in each field is the correct
answer, whether the software got it right or the person fixed it. So
every field's digit pictures can be banked, labeled with the FINAL
number's digits, not the software's guess. Fixed fields matter most --
they are exactly the digits the model gets wrong ("9" read as "4").

The one catch: a picture can only be labeled if each picture lines up
with one digit of the final number. If the person's fix changed how
MANY digits there are (the software split one digit in two, or read a
stray mark as a digit), there is no way to tell which picture is which,
so that field is skipped rather than guessed at.
"""


def _final_digit_strings(kind: str, final_value) -> list:
    """
    Every way the final value could have been written down, as plain
    digit strings (no decimal point), so one can be matched to the
    number of pictures actually saved.

    Quantity and Return are whole numbers: one possibility. A Total
    Price could have been written "130", "130.0" or "130.00", which
    leave 3, 4 and 5 digit pictures, so each is a possibility.
    """
    if final_value is None:
        return []
    if kind in ("quantity", "return"):
        return [str(int(final_value))]
    candidates = [f"{final_value:.2f}".replace(".", ""), f"{final_value:.1f}".replace(".", "")]
    if float(final_value).is_integer():
        candidates.append(str(int(final_value)))
    return candidates


def labels_for_field(kind: str, crops: list, original_value, final_value) -> list:
    """
    Work out the label for each digit picture of one field.

    Args:
        kind: "quantity", "return" or "total_price".
        crops: that field's list from results.json, each
            {"path": ..., "predicted_digit": ...}. A decimal point is
            not a 0-9 class and is ignored.
        original_value: what the software read (kept for callers; the
            final value alone decides the labels).
        final_value: what the reviewer approved.

    Returns:
        A list of (crop, label) pairs, or an empty list if the field
        can't be safely labeled.
    """
    digit_crops = [c for c in crops if str(c.get("predicted_digit", "")).isdigit()]
    if not digit_crops:
        return []

    # Unchanged or fixed alike, the pictures must line up with the final
    # number's digits. This also keeps a stray mark read as a leading "0"
    # (the leading_zero_digit flag) from being banked as a real zero.
    for digits in _final_digit_strings(kind, final_value):
        if len(digits) == len(digit_crops):
            return list(zip(digit_crops, digits))
    return []
