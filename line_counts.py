"""
Counts total/code/comment-or-docstring/blank lines for every Python
file in this project, to keep the file-size table at the top of
CLAUDE.md current. Not part of the training/inference pipeline itself
-- a documentation-maintenance utility, re-run after any substantive
edit to a tracked .py file.

"Code" here means actual executable syntax -- it deliberately excludes
both `#` comments AND docstrings/standalone triple-quoted strings
(this project's convention per CLAUDE.md is full docstrings plus
detailed inline comments, so those make up a large share of most
files' line counts, and lumping them in with "code" would misrepresent
how much of each file is actually logic vs. documentation).

Run with:  python line_counts.py
Prints a markdown table -- paste it into CLAUDE.md's line-count
section at the top of the file.
"""

import glob
import io
import tokenize

# Tokens that never end a logical statement on their own, so seeing
# one doesn't tell us whether the NEXT token starts a new statement --
# skipped when tracking "what significant token came before this one."
_IGNORABLE_TOKENS = {tokenize.NL, tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING}

# A STRING token is a docstring/standalone comment-string (not a
# string used as part of a larger expression, like the right side of
# an assignment) only if the token immediately before it was itself
# the end of a statement/block -- i.e. it's the first thing on its own
# logical line. `None` covers the very start of the file (the module
# docstring has nothing before it at all).
_STATEMENT_START_TOKENS = {None, tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING}


def classify_lines(path: str) -> tuple[int, int, int, int]:
    """
    Classify every physical line in a Python file as code, comment/
    docstring, or blank.

    Uses the `tokenize` module (not a plain text heuristic) specifically
    to tell a real docstring apart from an ordinary string that happens
    to also span multiple lines -- e.g. a variable assigned a
    triple-quoted multi-line string as its value is real code, not
    documentation, even though it looks superficially similar to a
    docstring in the raw text.

    Args:
        path: path to a .py source file.

    Returns:
        (total_lines, code_lines, comment_or_doc_lines, blank_lines).
    """
    with open(path, "rb") as f:
        source_bytes = f.read()
    lines = source_bytes.decode("utf-8").splitlines()
    total = len(lines)

    is_doc_line = [False] * (total + 1)  # 1-indexed, ignore index 0

    tokens = list(tokenize.tokenize(io.BytesIO(source_bytes).readline))
    prev_significant = None
    for tok in tokens:
        if tok.type == tokenize.STRING and prev_significant in _STATEMENT_START_TOKENS:
            for line_no in range(tok.start[0], min(tok.end[0], total) + 1):
                is_doc_line[line_no] = True
        if tok.type not in _IGNORABLE_TOKENS:
            prev_significant = tok.type

    blank = comment_or_doc = code = 0
    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if is_doc_line[i]:
            comment_or_doc += 1
        elif not stripped:
            blank += 1
        elif stripped.startswith("#"):
            comment_or_doc += 1
        else:
            # Covers real code, including a code line with a trailing
            # `# comment` after it -- such a line has real syntax on
            # it, so it counts as code, not comment.
            code += 1

    return total, code, comment_or_doc, blank


def main():
    rows = []
    for path in sorted(glob.glob("*.py")):
        rows.append((path, *classify_lines(path)))

    header = f"{'File':<26}{'Total':>8}{'Code':>8}{'Comments/Docs':>16}{'Blank':>8}"
    print(header)
    print("-" * len(header))
    for path, total, code, comment_or_doc, blank in rows:
        print(f"{path:<26}{total:>8}{code:>8}{comment_or_doc:>16}{blank:>8}")

    print("-" * len(header))
    print(
        f"{'TOTAL':<26}"
        f"{sum(r[1] for r in rows):>8}"
        f"{sum(r[2] for r in rows):>8}"
        f"{sum(r[3] for r in rows):>16}"
        f"{sum(r[4] for r in rows):>8}"
    )

    print("\nmarkdown table:\n")
    print("| File | Total lines | Code lines | Comment/docstring lines | Blank lines |")
    print("|---|---|---|---|---|")
    for path, total, code, comment_or_doc, blank in rows:
        print(f"| `{path}` | {total} | {code} | {comment_or_doc} | {blank} |")
    print(
        f"| **Total** | **{sum(r[1] for r in rows)}** | **{sum(r[2] for r in rows)}** "
        f"| **{sum(r[3] for r in rows)}** | **{sum(r[4] for r in rows)}** |"
    )


if __name__ == "__main__":
    main()
