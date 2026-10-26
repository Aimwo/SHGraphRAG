"""Generate the documentation diagrams locally; no external assets or services.

SVG output needs the standard library. Optional --preview uses Pillow to also
write PNG previews. SVG files are the editable, published source diagrams.
"""
from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

DEST = Path(__file__).resolve().parent
NAVY, CARD, LINE = "#0c1424", "#152238", "#39506d"
WHITE, MUTED, BLUE, TEAL, GOLD = "#f4f7fc", "#b9c9dd", "#80b5ff", "#65d8c0", "#ffd18a"


class Canvas:
    def __init__(self, width, height, preview=False):
        self.width, self.height = width, height
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
                    f'viewBox="0 0 {width} {height}" role="img">',
                    '<style>text{font-family:Segoe UI,Arial,sans-serif}</style>']
        self.image = self.draw = None
        if preview:
            from PIL import Image, ImageDraw
            self.image = Image.new("RGB", (width, height), NAVY)
            self.draw = ImageDraw.Draw(self.image)
        self.rect(0, 0, width, height, NAVY, radius=0)

    def rect(self, x, y, w, h, fill=CARD, stroke=None, radius=18):
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
                        f'fill="{fill}"' + (f' stroke="{stroke}"' if stroke else "") + '/>')
        if self.draw:
            self.draw.rounded_rectangle((x, y, x+w, y+h), radius=radius,
                                        fill=fill, outline=stroke)

    def text(self, x, y, value, size=20, color=WHITE, weight=400):
        self.svg.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
                        f'fill="{color}">{escape(value)}</text>')
        if self.draw:
            from PIL import ImageFont
            font = Path("C:/Windows/Fonts") / ("segoeuib.ttf" if weight >= 600 else "segoeui.ttf")
            try:
                loaded = ImageFont.truetype(str(font), size)
            except OSError:
                loaded = ImageFont.load_default()
            self.draw.text((x, y), value, font=loaded, fill=color, anchor="ls")

    def line(self, x1, y1, x2, y2, color=LINE):
        self.svg.append(f'<path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="{color}" stroke-width="3"/>')
        if self.draw:
            self.draw.line((x1, y1, x2, y2), fill=color, width=3)

    def arrow(self, x1, y1, x2, y2, color=LINE):
        self.line(x1, y1, x2, y2, color)
        if x1 == x2:
            self.line(x2-6, y2-8, x2, y2, color)
            self.line(x2+6, y2-8, x2, y2, color)
        else:
            self.line(x2-8, y2-6, x2, y2, color)
            self.line(x2-8, y2+6, x2, y2, color)

    def save(self, name):
        self.svg.append("</svg>")
        (DEST / (name + ".svg")).write_text("\n".join(self.svg) + "\n", encoding="utf-8")
        if self.image:
            self.image.save(DEST / (name + ".png"))


def overview(preview):
    c = Canvas(1400, 930, preview)
    c.text(56, 60, "SHGraphRAG  /  Supplementary Reproduction", 34, weight=700)
    c.text(56, 96, "Four completed profiles. Two explicit protocols. Traceable source and evaluation.", 20, MUTED)
    c.rect(56, 130, 1288, 96, stroke=LINE)
    c.text(84, 166, "OFFICIAL DEVELOPMENT RELEASES", 15, BLUE, 700)
    c.text(84, 202, "HotpotQA distractor  +  2WikiMultiHopQA  +  MuSiQue-Ans", 25, weight=600)
    for x, seed, accent, model in ((56, "20261020", BLUE, "GPT-4o mini / OpenRouter"),
                                    (720, "42", TEAL, "DeepSeek / historical relay")):
        c.arrow(x+312, 226, x+312, 255)
        c.rect(x, 262, 624, 62, fill="#1a2e48", stroke=accent)
        c.text(x+24, 302, f"SEED {seed}   |   {model}", 22, accent, 600)
        c.rect(x, 350, 624, 144, stroke=LINE)
        c.text(x+24, 388, "SHGraphRAG", 25, weight=700)
        if seed == "20261020":
            lines = ["Global identity merge and sparse graph",
                     "Offline two-hop path-question repository",
                     "Global candidate-union retrieval"]
        else:
            lines = ["Upstream build and supplied tuple QA",
                     "Online two-hop lookup; 0 offline path questions",
                     "Question-scoped retrieval"]
        for i, line in enumerate(lines):
            c.text(x+24, 420+i*25, line, 19, MUTED)
        c.rect(x, 518, 624, 144, stroke=LINE)
        c.text(x+24, 556, "HyperGraphRAG", 25, weight=700)
        lines = (["Native hybrid retrieval; top-k 60",
                  "Audited overflow-only embedding fallback",
                  "32 text workers; local BGE on port 11436"]
                 if seed == "20261020" else
                 ["Native hybrid retrieval; top-k 60",
                  "Visible summaries 500 / generation budget 8192",
                  "Shared physical gate 100; embeddings at most 4"])
        for i, line in enumerate(lines):
            c.text(x+24, 588+i*25, line, 19, MUTED)
        c.arrow(x+312, 662, x+312, 700)
    c.rect(56, 710, 1288, 116, fill="#14352f", stroke=TEAL)
    c.text(84, 749, "INDEPENDENT EVALUATION  +  PROVENANCE", 17, TEAL, 700)
    c.text(84, 785, "1,000 frozen IDs per dataset  /  native and canonical scores kept separate", 24, weight=600)
    c.text(56, 867, "Published: implementation, prompts, locks, ordered IDs, aggregate receipts, reproducible procedures.", 19, MUTED)
    c.text(56, 899, "External inputs: official dataset files, local BGE model, and reader-authorized provider access.", 19, MUTED)
    c.save("reproduction-overview")


def boundaries(preview):
    c = Canvas(1400, 690, preview)
    c.text(56, 60, "Keep inference and evaluation inputs explicit", 34, weight=700)
    c.text(56, 96, "Question IDs align the two streams; answer labels do not enter model prompts.", 20, MUTED)
    c.rect(56, 132, 390, 108, stroke=LINE)
    c.text(80, 174, "Official local development files", 23, weight=600)
    c.text(80, 211, "Pinned release hashes and native fields", 18, MUTED)
    c.arrow(446, 187, 548, 187)
    c.rect(556, 132, 788, 108, fill="#1a2e48", stroke=BLUE)
    c.text(580, 174, "Offline protocol reconstruction", 25, BLUE, 600)
    c.text(580, 211, "Seed, 1,000 ordered IDs, candidate identities, and text hashes verified", 20, MUTED)
    c.line(950, 240, 950, 265)
    c.line(346, 265, 1054, 265)
    c.arrow(346, 265, 346, 294, BLUE)
    c.arrow(1054, 265, 1054, 294, GOLD)
    c.rect(56, 304, 580, 144, stroke=BLUE)
    c.text(80, 345, "INFERENCE", 16, BLUE, 700)
    c.text(80, 382, "ID + question + candidate passage text", 24, weight=600)
    c.text(80, 419, "Documented memberships / graph scope", 20, MUTED)
    c.rect(764, 304, 580, 144, stroke=GOLD)
    c.text(788, 345, "EVALUATION ONLY", 16, GOLD, 700)
    c.text(788, 382, "Native answers, aliases, supporting facts", 23, weight=600)
    c.text(788, 419, "Opened by scorers after predictions exist", 20, MUTED)
    c.arrow(346, 448, 346, 496, BLUE)
    c.arrow(1054, 448, 1054, 496, GOLD)
    c.rect(56, 506, 580, 104, stroke=BLUE)
    c.text(80, 549, "Build → retrieve → generate", 24, BLUE, 600)
    c.text(80, 584, "Saved predictions and immutable attempt history", 19, MUTED)
    c.arrow(636, 558, 754, 558, TEAL)
    c.rect(764, 506, 580, 104, fill="#14352f", stroke=TEAL)
    c.text(788, 549, "Full-denominator EM / F1 + verification", 23, TEAL, 600)
    c.text(788, 584, "No dropping empty, failed, or missing IDs", 20, MUTED)
    c.text(56, 657, "Retrieval scope remains profile-specific; shared input IDs do not imply identical retrieval conditions.", 20, MUTED)
    c.save("evaluation-boundaries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    overview(args.preview)
    boundaries(args.preview)
