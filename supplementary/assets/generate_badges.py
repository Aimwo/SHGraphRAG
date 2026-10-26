"""Regenerate the README badges with the standard library, entirely offline.

Run from any directory: python supplementary/assets/generate_badges.py
Labels describe the published experiment records, not CI or platform support.
Widths are fixed for the small, explicit set of labels; no font download needed.
"""
from __future__ import annotations

from html import escape
from pathlib import Path

DESTINATION = Path(__file__).resolve().parent / "badges"
BADGES = (
    ("license", "license", "MIT", "#238636", 48, 32),
    ("python", "Python", "3.11 | 3.12", "#1f6feb", 49, 82),
    ("profiles", "profiles", "4", "#8250df", 52, 24),
    ("benchmarks", "benchmarks", "3", "#0e7490", 78, 24),
    ("embedding", "embedding", "BGE-M3", "#0f766e", 71, 59),
)


def badge_svg(label, value, color, label_width, value_width):
    width = label_width + value_width
    description = escape(f"{label}: {value}", quote=True)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="20" viewBox="0 0 {width} 20" role="img" aria-label="{description}">
  <title>{description}</title>
  <clipPath id="rounded"><rect width="{width}" height="20" rx="3"/></clipPath>
  <g clip-path="url(#rounded)">
    <rect width="{label_width}" height="20" fill="#464f5f"/>
    <rect x="{label_width}" width="{value_width}" height="20" fill="{color}"/>
    <rect x=".5" y=".5" width="{width - 1}" height="19" rx="2.5" fill="none" stroke="#fff" stroke-opacity=".08"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="Verdana,DejaVu Sans,sans-serif" font-size="11">
    <text x="{label_width / 2:g}" y="14">{escape(label)}</text>
    <text x="{label_width + value_width / 2:g}" y="14">{escape(value)}</text>
  </g>
</svg>
'''


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for name, *settings in BADGES:
        (DESTINATION / f"{name}.svg").write_bytes(badge_svg(*settings).encode("utf-8"))
    print(f"Generated {len(BADGES)} local SVG badges.")


if __name__ == "__main__":
    main()
