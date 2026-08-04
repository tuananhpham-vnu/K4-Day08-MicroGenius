"""Task 1: Collect visible text from legal reference HTML pages.

The four legal documents below are collected from their public HTML pages. The
script never downloads the PDF attachments published by the source sites.
"""

from __future__ import annotations

import html
import json
import os
import re
import textwrap
import unicodedata
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import requests
from fpdf import FPDF
from fpdf.enums import XPos, YPos


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
REQUEST_TIMEOUT = 30
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://thuvienphapluat.vn/",
    "Connection": "keep-alive",
}

# These are public full-text HTML pages, not direct links to PDF attachments.
LAW_DOCUMENTS: list[dict[str, str]] = [
    {
        "title": "Nghi dinh 168/2025/ND-CP ve dang ky doanh nghiep",
        "source_page": (
            "https://thuvienphapluat.vn/phap-luat-doanh-nghiep/bai-viet/"
            "toan-van-nghi-dinh-168-2025-nd-cp-pdf-dang-ky-doanh-nghiep-13181.html"
        ),
        "filename": "nghi-dinh-168-2025-dang-ky-doanh-nghiep.txt",
        "customer_role": "both",
        "legal_area": "business_registration",
    },
    {
        "title": "Thong tu 68/2025/TT-BTC ve bieu mau dang ky doanh nghiep va ho kinh doanh",
        "source_page": (
            "https://thuvienphapluat.vn/phap-luat-doanh-nghiep/bai-viet/"
            "toan-van-thong-tu-68-2025-tt-btc-pdf-bieu-mau-su-dung-trong-dang-ky-"
            "doanh-nghiep-dang-ky-ho-kinh-doanh-13182.html"
        ),
        "filename": "thong-tu-68-2025-bieu-mau-dang-ky-doanh-nghiep-ho-kinh-doanh.txt",
        "customer_role": "both",
        "legal_area": "business_registration_forms",
    },
    {
        "title": "Nghi dinh 01/2021/ND-CP ve dang ky doanh nghiep het hieu luc tu 01/7/2025",
        "source_page": (
            "https://thuvienphapluat.vn/phap-luat-doanh-nghiep/bai-viet/"
            "nghi-dinh-01-2021-nd-cp-ve-dang-ky-doanh-nghiep-se-het-hieu-luc-tu-"
            "01-7-2025-13186.html"
        ),
        "filename": "nghi-dinh-01-2021-dang-ky-doanh-nghiep-het-hieu-luc.txt",
        "customer_role": "both",
        "legal_area": "business_registration_superseded",
    },
    {
        "title": "Luat 76/2025/QH15 sua doi Luat Doanh nghiep 2020",
        "source_page": (
            "https://thuvienphapluat.vn/phap-luat-doanh-nghiep/bai-viet/"
            "08-diem-noi-bat-cua-luat-doanh-nghiep-sua-doi-2025-ma-doanh-nghiep-"
            "can-luu-y-14706.html"
        ),
        "filename": "luat-76-2025-sua-doi-luat-doanh-nghiep-2020.txt",
        "customer_role": "both",
        "legal_area": "enterprise_law_amendment",
    },
]


class VisibleTextParser(HTMLParser):
    """Collect text that a reader can see while ignoring page chrome scripts."""

    IGNORED_TAGS = {"script", "style", "noscript", "svg", "canvas"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self.IGNORED_TAGS:
            self.ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self.IGNORED_TAGS and self.ignored_depth:
            self.ignored_depth -= 1
        elif tag.lower() in {"p", "div", "li", "br", "tr", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.ignored_depth:
            self.parts.append(data)


def setup_directory() -> None:
    """Create the Task 1 landing directory when it is missing."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def extract_visible_text(page_html: str) -> str:
    """Return normalized text from a public HTML page."""
    parser = VisibleTextParser()
    parser.feed(page_html)
    parser.close()
    text = html.unescape("".join(parser.parts))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def fetch_page_text(url: str) -> str:
    """Fetch an HTML source page; direct PDF downloads are intentionally avoided."""
    response = requests.get(
        url,
        headers=REQUEST_HEADERS,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()
    if "html" not in content_type and not response.text.lstrip().startswith("<"):
        raise ValueError(f"Expected an HTML page, received {content_type or 'unknown content'}")

    text = extract_visible_text(response.text)
    if len(text) < 300:
        raise ValueError("The source page did not provide enough visible text to create a legal document.")
    return text


def write_text_document(output_path: Path, content: str) -> None:
    """Save only the visible text extracted from the HTML page."""
    output_path.write_text(content + "\n", encoding="utf-8")


def pdf_lines(value: str, width: int = 40) -> list[str]:
    """Break long words so FPDF can render the extracted text safely."""
    lines: list[str] = []
    for paragraph in value.splitlines():
        if paragraph.strip():
            lines.extend(
                textwrap.wrap(
                    paragraph.strip(),
                    width=width,
                    break_long_words=True,
                    break_on_hyphens=False,
                )
                or [""]
            )
        else:
            lines.append("")
    return lines


def configure_pdf_font(pdf: FPDF) -> bool:
    """Use an installed Unicode font when possible; otherwise use Helvetica."""
    windows_dir = Path(os.environ.get("WINDIR", r"C:\\Windows"))
    arial_path = windows_dir / "Fonts" / "arial.ttf"
    if arial_path.exists():
        pdf.add_font("ArialUnicode", "", str(arial_path))
        pdf.set_font("ArialUnicode", size=10)
        return True

    pdf.set_font("Helvetica", size=10)
    return False


def write_text_pdf(output_path: Path, content: str) -> None:
    """Convert the extracted HTML text to a local selectable-text PDF."""
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)
    pdf.add_page()

    has_unicode_font = configure_pdf_font(pdf)
    text = content if has_unicode_font else unicodedata.normalize("NFKD", content).encode("ascii", "ignore").decode("ascii")
    for line in pdf_lines(text):
        if line:
            pdf.multi_cell(0, 5, line, align="L", new_x=XPos.LEFT, new_y=YPos.NEXT)
        else:
            pdf.ln(2)

    pdf.output(str(output_path))


def write_metadata(
    document: dict[str, str], output_path: Path, content_type: str
) -> None:
    """Save source and retrieval information for later filtering and citations."""
    metadata: dict[str, Any] = {
        "title": document["title"],
        "source_url": document["source_page"],
        "source_page": document["source_page"],
        "attachment_url": None,
        "customer_role": document["customer_role"],
        "legal_area": document["legal_area"],
        "content_type": content_type,
        "output_file": output_path.name,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }
    sidecar_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    sidecar_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def collect_legal_documents() -> list[Path]:
    """Save the four required documents as text extracted from HTML pages."""
    setup_directory()
    outputs: list[Path] = []

    for document in LAW_DOCUMENTS:
        output_path = DATA_DIR / document["filename"]
        content = fetch_page_text(document["source_page"])
        write_text_document(output_path, content)
        write_metadata(document, output_path, "text_from_html")
        outputs.append(output_path)
        print(f"[OK] Saved HTML text: {output_path.name}")

        text_for_pdf = output_path.read_text(encoding="utf-8")
        pdf_path = output_path.with_suffix(".pdf")
        write_text_pdf(pdf_path, text_for_pdf)
        write_metadata(document, pdf_path, "text_pdf_from_html")
        outputs.append(pdf_path)
        print(f"[OK] Created local text PDF: {pdf_path.name}")

    return outputs


if __name__ == "__main__":
    collect_legal_documents()
