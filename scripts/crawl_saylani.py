"""
scripts/crawl_saylani.py

Crawls the trusted URLs listed in config/sources.yaml, staying within the
allowed domains, and saves cleaned page text + metadata as JSON files under
knowledge_base/raw/ for later ingestion by scripts/ingest_documents.py.

Usage:
    python scripts/crawl_saylani.py
    python scripts/crawl_saylani.py --max-pages 20 --max-depth 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
import urllib.robotparser as robotparser
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse, parse_qsl, urlencode, urlunparse

import requests
import yaml
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import settings  # noqa: E402
from database import db  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sahulatai.crawler")

TRACKING_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "fbclid", "gclid"}


def normalize_url(url: str) -> str:
    url, _ = urldefrag(url)  # drop #fragment
    parsed = urlparse(url)
    query = [(k, v) for k, v in parse_qsl(parsed.query) if k not in TRACKING_PARAMS]
    normalized = parsed._replace(query=urlencode(query), path=parsed.path.rstrip("/") or "/")
    return urlunparse(normalized)


def load_sources_config() -> dict:
    with open(settings.sources_config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_allowed_domain(url: str, allowed_domains: list[str]) -> bool:
    host = urlparse(url).netloc.lower()
    return any(host == d.lower() or host.endswith("." + d.lower()) for d in allowed_domains)


def get_robots_parser(base_url: str, user_agent: str) -> robotparser.RobotFileParser:
    parsed = urlparse(base_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = robotparser.RobotFileParser()
    try:
        resp = requests.get(robots_url, headers={"User-Agent": user_agent}, timeout=settings.crawl_timeout_seconds)
        rp.parse(resp.text.splitlines())
    except Exception as exc:
        logger.warning("Could not fetch robots.txt for %s (%s); assuming allowed.", base_url, exc)
        rp.parse([])  # empty ruleset = allow
    return rp


def extract_clean_text(html: str) -> tuple[str, str]:
    """Return (title, cleaned_visible_text), stripping nav/footer/script noise."""
    soup = BeautifulSoup(html, "lxml")
    title = (soup.title.string.strip() if soup.title and soup.title.string else "").strip()

    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "form", "svg"]):
        tag.decompose()

    # Prefer <main> or <article> if present, else the whole body
    main = soup.find("main") or soup.find("article") or soup.body or soup

    lines = []
    for el in main.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = el.get_text(" ", strip=True)
        if text:
            prefix = "## " if el.name in ("h1", "h2", "h3", "h4") else ""
            lines.append(prefix + text)

    cleaned = "\n".join(lines).strip()
    return title, cleaned


def crawl(max_depth: int | None = None, max_pages: int | None = None) -> dict:
    config = load_sources_config()
    allowed_domains = config.get("allowed_domains", [])
    seed_urls = config.get("seed_urls", [])
    crawl_cfg = config.get("crawl", {})

    max_depth = max_depth if max_depth is not None else crawl_cfg.get("max_depth", settings.crawl_max_depth)
    max_pages = max_pages if max_pages is not None else crawl_cfg.get("max_pages", settings.crawl_max_pages)
    user_agent = crawl_cfg.get("user_agent", settings.crawl_user_agent)
    respect_robots = crawl_cfg.get("respect_robots_txt", True)

    db.init_db()
    run_id = db.start_ingestion_run(source_type="official_website")

    output_dir = settings.knowledge_base_dir / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    visited: set[str] = set()
    queue: list[tuple[str, int, str]] = [(normalize_url(s["url"]), 0, s.get("source_type", "official_website"))
                                          for s in seed_urls]
    robots_cache: dict[str, robotparser.RobotFileParser] = {}
    pages_saved = 0

    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})

    while queue and pages_saved < max_pages:
        url, depth, source_type = queue.pop(0)
        if url in visited or depth > max_depth:
            continue
        visited.add(url)

        if not is_allowed_domain(url, allowed_domains):
            logger.info("Skipping out-of-scope domain: %s", url)
            continue

        domain = urlparse(url).netloc
        if respect_robots:
            if domain not in robots_cache:
                robots_cache[domain] = get_robots_parser(url, user_agent)
            if not robots_cache[domain].can_fetch(user_agent, url):
                logger.info("robots.txt disallows: %s", url)
                continue

        try:
            resp = session.get(url, timeout=settings.crawl_timeout_seconds, allow_redirects=True)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Failed to fetch %s: %s", url, exc)
            continue

        content_type = resp.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            continue

        title, text = extract_clean_text(resp.text)
        if not text or len(text) < 40:
            logger.info("Skipping near-empty page: %s", url)
        else:
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            record = {
                "source_url": url,
                "title": title,
                "text": text,
                "source_type": source_type,
                "crawl_date": datetime.now(timezone.utc).isoformat(),
                "content_hash": content_hash,
            }
            filename = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16] + ".json"
            out_path = output_dir / filename
            existing_hash = None
            if out_path.exists():
                try:
                    existing_hash = json.loads(out_path.read_text(encoding="utf-8")).get("content_hash")
                except Exception:
                    existing_hash = None
            if existing_hash != content_hash:
                out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                pages_saved += 1
                logger.info("Saved (%d/%d): %s", pages_saved, max_pages, url)
            else:
                logger.info("Unchanged, skipping re-save: %s", url)

        if depth < max_depth:
            soup = BeautifulSoup(resp.text, "lxml")
            for a in soup.find_all("a", href=True):
                next_url = normalize_url(urljoin(url, a["href"]))
                if next_url not in visited and is_allowed_domain(next_url, allowed_domains):
                    queue.append((next_url, depth + 1, source_type))

        time.sleep(0.5)  # be polite

    db.finish_ingestion_run(
        run_id, pages_crawled=len(visited), documents_ingested=pages_saved,
        chunks_created=0, status="completed",
    )
    logger.info("Crawl complete. Visited %d URLs, saved %d pages.", len(visited), pages_saved)
    return {"visited": len(visited), "saved": pages_saved}


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl official Saylani sources for SahulatAI's knowledge base.")
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--max-pages", type=int, default=None)
    args = parser.parse_args()
    crawl(max_depth=args.max_depth, max_pages=args.max_pages)


if __name__ == "__main__":
    main()
