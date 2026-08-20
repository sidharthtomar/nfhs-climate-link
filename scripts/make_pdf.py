#!/usr/bin/env python
"""Render REPORT.md to a polished PDF with figures embedded.

Turns the *Figure N -- path* reference lines in REPORT.md into embedded images,
then converts to PDF via pandoc. Produces outputs/REPORT.pdf.

Usage:
    python scripts/make_pdf.py

Requires: pandoc, and a LaTeX engine (tectonic or xelatex) OR libreoffice.
Falls back to libreoffice (soffice) if no LaTeX engine is present.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "REPORT.md"
BUILD = ROOT / "outputs" / "_report_build.md"
PDF = ROOT / "outputs" / "REPORT.pdf"

# pypandoc_binary 1.17's bundled pandoc (and pypandoc.download_pandoc()'s
# auto-picked asset) is x86_64 even on this arm64 Mac, which has no Rosetta 2
# to run it. Fetched the correct arm64 build directly from pandoc's official
# GitHub releases instead (same upstream binary, right architecture) and
# staged it under .venv/, which is already gitignored.
PANDOC = str(ROOT / ".venv" / "pandoc-arm64" / "pandoc")
SOFFICE = "/Applications/LibreOffice.app/Contents/MacOS/soffice"

# Map the figure-caption lines to real image paths + captions
FIG = re.compile(r"\*Figure (\d+) [—-] `?(outputs/figures/[\w./]+\.png)`?:?\s*(.*?)\.?\*")


def embed_figures(md: str) -> str:
    def repl(m):
        num, path, cap = m.group(1), m.group(2), m.group(3).strip()
        abspath = (ROOT / path)
        if not abspath.exists():
            return m.group(0)  # leave caption if image missing
        return f"![**Figure {num}.** {cap}]({path}){{width=85%}}"
    return FIG.sub(repl, md)


def main() -> int:
    if not SRC.exists():
        print(f"Missing {SRC}"); return 1
    md = embed_figures(SRC.read_text())
    BUILD.write_text(md)

    # Prefer a LaTeX engine for quality; fall back to libreoffice.
    engines = ["tectonic", "xelatex", "pdflatex", "wkhtmltopdf"]
    engine = next((e for e in engines if shutil.which(e)), None)

    common = [
        PANDOC, str(BUILD), "-o", str(PDF),
        "-V", "geometry:margin=1in",
        "-V", "fontsize=11pt",
        "-V", "linkcolor:blue",
        "--toc", "--toc-depth=2",
    ]
    try:
        if engine and engine != "wkhtmltopdf":
            cmd = common + [f"--pdf-engine={engine}"]
        elif engine == "wkhtmltopdf":
            cmd = common + ["--pdf-engine=wkhtmltopdf"]
        else:
            # last resort: pandoc -> docx -> soffice -> pdf
            docx = ROOT / "outputs" / "REPORT.docx"
            subprocess.run([PANDOC, str(BUILD), "-o", str(docx), "--toc"], check=True)
            subprocess.run([SOFFICE, "--headless", "--convert-to", "pdf",
                            "--outdir", str(PDF.parent), str(docx)], check=True)
            print(f"Wrote {PDF} (via libreoffice)"); return 0
        subprocess.run(cmd, check=True)
        print(f"Wrote {PDF} (via {engine})")
    except subprocess.CalledProcessError as e:
        print(f"PDF build failed: {e}\nTry: pip install nothing; brew install tectonic  (or)  "
              "pandoc must find a LaTeX engine. Fallback: soffice.")
        return 1
    finally:
        BUILD.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
