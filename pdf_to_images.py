"""
Converts one or more PDF files (e.g. exported from a phone scanning
app) into individual page images, saved into the invoices/ folder that
label_tool.py reads from. Run this once, before label_tool.py, any
time you have new scans to add.

Run with:  python3 pdf_to_images.py your_scans.pdf
       or: python3 pdf_to_images.py scan1.pdf scan2.pdf scan3.pdf
"""

import os
import sys

import pymupdf

OUTPUT_DIR = "invoices"
DPI = 300  # render resolution -- 300 is print-quality, plenty of
           # detail for cropping small handwritten digits out later


def pdf_to_images(pdf_path: str, rotate: int = 0):
    """
    Render every page of one PDF to its own PNG image file.

    Args:
        pdf_path: path to the PDF file to convert.
        rotate: degrees to rotate each page's output by (0, 90, 180,
            or 270). Use this if pages come out sideways or upside
            down despite the source PDF itself looking correct --
            some phone scanning apps embed an internal orientation
            flip that this rendering step doesn't always inherit.
    """
    doc = pymupdf.open(pdf_path)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]

    # A PDF page is a vector description, not pixels -- this matrix
    # controls how many pixels we render it at. PDF's native unit is
    # 72 points per inch, so dividing our target DPI by 72 gives the
    # right zoom factor to hit that resolution.
    zoom = DPI / 72
    matrix = pymupdf.Matrix(zoom, zoom)
    if rotate:
        matrix = matrix.prerotate(rotate)

    for page_number in range(len(doc)):
        page = doc[page_number]
        pix = page.get_pixmap(matrix=matrix)

        out_name = f"{base_name}_page{page_number + 1:03d}.png"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        pix.save(out_path)
        print(f"saved {out_path}")

    doc.close()


def main():
    args = sys.argv[1:]
    rotate = 0

    if "--rotate" in args:
        idx = args.index("--rotate")
        rotate = int(args[idx + 1])
        del args[idx:idx + 2]  # remove the flag and its value from the file list

    if not args:
        print("Usage: python3 pdf_to_images.py [--rotate 90|180|270] file1.pdf [file2.pdf ...]")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for pdf_path in args:
        pdf_to_images(pdf_path, rotate=rotate)


if __name__ == "__main__":
    main()
