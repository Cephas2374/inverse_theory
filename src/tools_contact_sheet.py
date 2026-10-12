"""Tile rendered PDF pages into one contact-sheet PNG for a quick layout check.
Usage: python src/tools_contact_sheet.py <pdf> <out.png> [cols] [dpi]
"""
import sys

import fitz
from PIL import Image

pdf, out = sys.argv[1], sys.argv[2]
cols = int(sys.argv[3]) if len(sys.argv) > 3 else 5
dpi = int(sys.argv[4]) if len(sys.argv) > 4 else 40
doc = fitz.open(pdf)
pages = []
for p in doc:
    pix = p.get_pixmap(dpi=dpi)
    pages.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
w, h = pages[0].size
rows = (len(pages) + cols - 1) // cols
sheet = Image.new("RGB", (cols * w, rows * h), "white")
for i, im in enumerate(pages):
    sheet.paste(im, ((i % cols) * w, (i // cols) * h))
sheet.save(out)
print(out, sheet.size)
