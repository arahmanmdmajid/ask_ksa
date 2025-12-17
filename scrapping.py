import sys
import asyncio
import re
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict

from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from markdownify import markdownify as md  # pip install markdownify


LIST_LINK_SELECTOR = ".td-module-title a"
CONTENT_SELECTOR = "div.td-post-content"
TITLE_SELECTOR = "h1.td-post-title, h1.entry-title"


def slugify_filename(text: str, max_len: int = 80) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[\s_]+", "-", text).strip("-")
    return (text[:max_len] or "article")


def build_full_article_markdown(title: str, url: str, body_md: str) -> str:
    scraped_at = datetime.now(timezone.utc).isoformat()
    safe_title = (title or "Untitled").replace('"', '\\"')

    return "\n".join([
        "---",
        f'title: "{safe_title}"',
        f'source_url: "{url}"',
        f'scraped_at: "{scraped_at}"',
        "---",
        "",
        f"# {title or 'Untitled'}",
        "",
        f"_Source: {url}_",
        "",
        body_md.strip(),
        ""
    ])


async def scrape_article_as_markdown(page, url: str) -> Optional[Dict]:
    print(f"📝 Scraping article: {url}")
    await page.goto(url, wait_until="networkidle", timeout=30000)
    soup = BeautifulSoup(await page.content(), "html.parser")

    content_div = soup.select_one(CONTENT_SELECTOR)
    if not content_div:
        print(f"⚠️ Skipping {url} — content div not found.")
        return None

    title_el = soup.select_one(TITLE_SELECTOR)
    article_title = (title_el.get_text(strip=True) if title_el else "Untitled").strip()

    # Remove junk inside the post body before converting
    for bad in content_div.select("script, style, noscript"):
        bad.decompose()

    # Inner HTML -> Markdown :contentReference[oaicite:3]{index=3}
    html_fragment = content_div.decode_contents()

    # Convert HTML -> Markdown (ATX gives # headings) :contentReference[oaicite:4]{index=4}
    body_md = md(html_fragment, heading_style="ATX", bullets="*+-").strip()

    return {
        "url": url,
        "title": article_title,
        "body_md": body_md
    }


async def collect_article_links_from_page(page, url: str, seen_links: set) -> List[str]:
    print(f"\n📄 Fetching list page: {url}")
    await page.goto(url, wait_until="networkidle", timeout=30000)

    for _ in range(6):
        await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
        await page.wait_for_timeout(1000)

    soup = BeautifulSoup(await page.content(), "html.parser")

    new_links = []
    for a in soup.select(LIST_LINK_SELECTOR):
        href = a.get("href")
        if not href:
            continue
        full = urljoin(url, href)
        if full not in seen_links:
            seen_links.add(full)
            new_links.append(full)

    print(f"🔗 Found {len(new_links)} new article links on this page.")
    return new_links


async def scrape_all_pages_to_markdown(base_url: str, out_dir: str = "data", max_pages: Optional[int] = None):
    seen_links: set = set()
    page_number = 1

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)  # create output folder :contentReference[oaicite:5]{index=5}

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = await browser.new_page()

        while True:
            if max_pages is not None and page_number > max_pages:
                break

            if page_number == 1:
                list_url = base_url
            else:
                base = base_url if base_url.endswith("/") else base_url + "/"
                list_url = urljoin(base, f"page/{page_number}/")

            try:
                page_links = await collect_article_links_from_page(page, list_url, seen_links)
            except Exception as e:
                print(f"❌ Error loading {list_url}: {e}")
                break

            if not page_links:
                print("ℹ️ No new links on this page; assuming end of listing.")
                break

            for article_url in page_links:
                try:
                    result = await scrape_article_as_markdown(page, article_url)
                    if not result:
                        continue

                    title = result["title"]
                    body_md = result["body_md"]

                    slug = slugify_filename(title)
                    url_hash = hashlib.md5(article_url.encode("utf-8")).hexdigest()[:8]
                    md_file = out_path / f"{slug}-{url_hash}.md"

                    full_md = build_full_article_markdown(title, article_url, body_md)

                    # Write UTF-8 text file :contentReference[oaicite:6]{index=6}
                    md_file.write_text(full_md, encoding="utf-8")
                    print(f"✅ Saved: {md_file}")

                except Exception as e:
                    print(f"❌ Error scraping {article_url}: {e}")

            page_number += 1

        await browser.close()

    print(f"\n✅ Done. Markdown files are in: {Path(out_dir).resolve()}")


async def main():
    base_url = "https://lifeinsaudiarabia.net/category/jawazat-and-moi/iqama/"
    await scrape_all_pages_to_markdown(base_url, out_dir="data", max_pages=None)


if __name__ == "__main__":
    asyncio.run(main())