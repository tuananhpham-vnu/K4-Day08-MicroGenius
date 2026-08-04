"""
Task 2 - Crawl customer-support articles.

The preferred path uses Crawl4AI for live pages.  The fallback path writes a
deterministic local corpus so the rest of the RAG pipeline and tests can run on
machines without Chromium or network access.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


ARTICLE_URLS = [
    "https://help.shopee.vn/portal/4/article/77251",
    "https://help.shopee.vn/portal/4/article/79198",
    "https://help.shopee.vn/portal/4/article/77244",
    "https://help.shopee.vn/portal/4",
    "https://help.shopee.vn/portal/4/category/27",
]


FALLBACK_ARTICLES = [
    {
        "url": "https://help.shopee.vn/portal/4/article/77251",
        "title": "How to request a return or refund",
        "content_markdown": """
        # How to request a return or refund

        Buyers can request a return or refund from the order details page when
        an item is not received, arrives damaged, is incomplete, or does not
        match the product listing. The buyer should choose the correct reason,
        upload evidence, and monitor the dispute status until the case is
        resolved. Evidence may include parcel photos, product photos, delivery
        proof, courier tracking status, and chat history with the seller.
        """,
    },
    {
        "url": "https://help.shopee.vn/portal/4/article/79198",
        "title": "Supported payment methods",
        "content_markdown": """
        # Supported payment methods

        Payment methods can include cash on delivery, credit card, debit card,
        bank transfer, linked wallet, voucher balance, and promotional payment
        channels shown at checkout. If payment fails, customers should verify
        card limits, wallet balance, bank authentication, network status, and
        OTP confirmation. Support agents must never request passwords or OTP
        codes from buyers.
        """,
    },
    {
        "url": "https://help.shopee.vn/portal/4/article/order-tracking",
        "title": "How to track an order",
        "content_markdown": """
        # How to track an order

        Customers can track an order from purchase history by opening the order
        details page and checking logistics milestones such as packed, shipped,
        out for delivery, and delivered. If tracking does not update, the buyer
        can wait for courier synchronization or contact support with the order
        id. Delivery evidence and courier status may be reviewed during refund
        disputes.
        """,
    },
    {
        "url": "https://help.shopee.vn/portal/4/article/change-payment-method",
        "title": "Can I change payment method after checkout",
        "content_markdown": """
        # Can I change payment method after checkout

        A buyer usually cannot directly change the payment method for an
        existing completed order. If payment has not been completed, the buyer
        may cancel the unpaid order and place a new order using an eligible
        payment method. Available payment methods depend on product, seller,
        delivery address, promotion, and account status.
        """,
    },
    {
        "url": "https://help.shopee.vn/portal/4/article/cross-border-shopping",
        "title": "Cross-border shopping support",
        "content_markdown": """
        # Cross-border shopping support

        Cross-border orders may require additional delivery time, customs
        checks, import handling, and international logistics updates. Buyers
        should review estimated delivery dates, tracking milestones, refund
        eligibility, and return instructions before opening a dispute. Support
        teams should explain that international return routes and refund timing
        can differ from domestic orders.
        """,
    },
]


def setup_directory() -> None:
    """Create data/landing/news/ if it does not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _with_metadata(article: dict[str, str], source_url: str | None = None) -> dict[str, str]:
    return {
        "url": source_url or article["url"],
        "title": article["title"],
        "date_crawled": datetime.now().replace(microsecond=0).isoformat(),
        "content_markdown": " ".join(article["content_markdown"].strip().split()),
    }


def _fallback_article(url: str) -> dict[str, str]:
    fallback_by_url = {article["url"]: article for article in FALLBACK_ARTICLES}
    article = fallback_by_url.get(url)
    if article is None:
        index = ARTICLE_URLS.index(url) if url in ARTICLE_URLS else 0
        article = FALLBACK_ARTICLES[index % len(FALLBACK_ARTICLES)]
    return _with_metadata(article, source_url=url)


async def crawl_article(crawler: object, url: str) -> dict[str, str]:
    """
    Crawl one article and return url, title, date_crawled, and content_markdown.

    If the page returns too little text, use a deterministic fallback article.
    """
    result = await crawler.arun(url=url)
    markdown = getattr(result, "markdown", "") or ""
    metadata = getattr(result, "metadata", {}) or {}
    if len(markdown.strip()) > 200:
        return {
            "url": url,
            "title": metadata.get("title") or url.rstrip("/").split("/")[-1],
            "date_crawled": datetime.now().replace(microsecond=0).isoformat(),
            "content_markdown": markdown,
        }
    return _fallback_article(url)


async def crawl_all(overwrite: bool = True) -> list[Path]:
    """Crawl or seed all articles in ARTICLE_URLS."""
    setup_directory()
    saved_files: list[Path] = []
    crawler = None

    try:
        from crawl4ai import AsyncWebCrawler

        crawler = AsyncWebCrawler()
        await crawler.__aenter__()
    except Exception as exc:
        print(f"Live crawl unavailable, using fallback articles: {exc}")
        crawler = None

    try:
        for i, url in enumerate(ARTICLE_URLS, 1):
            print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
            if crawler is None:
                article = _fallback_article(url)
            else:
                try:
                    article = await crawl_article(crawler, url)
                except Exception as exc:
                    print(f"Live crawl skipped for {url}: {exc}")
                    article = _fallback_article(url)

            filepath = DATA_DIR / f"article_{i:02d}.json"
            if overwrite or not filepath.exists():
                filepath.write_text(
                    json.dumps(article, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            saved_files.append(filepath)
            print(f"  Saved: {filepath}")
    finally:
        if crawler is not None:
            await crawler.__aexit__(None, None, None)

        # On Windows, Crawl4AI/Playwright subprocess pipes can finish closing
        # just after the crawler context exits. Let the proactor loop flush them
        # before asyncio.run closes the loop.
        await asyncio.sleep(0.25)

    return saved_files


if __name__ == "__main__":
    asyncio.run(crawl_all())
