#!/usr/bin/env python3
from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import json
import re
import time
from pathlib import Path
from typing import Any, Iterable

import feedparser
import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARCHIVE_DIR = DATA_DIR / "archive"
REPORTS_DIR = ROOT / "reports"

DATA_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

ARXIV_BASE = "https://export.arxiv.org/api/query"
GITHUB_SEARCH_REPOS = "https://api.github.com/search/repositories"
WAYMO_RESEARCH_URL = "https://waymo.com/research/"

ARXIV_SLEEP_SECONDS = 3
REQUEST_TIMEOUT = 30
USER_AGENT = "wosac-research-tracker/0.1 (contact: bob020416@gmail.com)"
MAX_DAYS = 7

TOPIC_KEYWORDS = {
    "wosac": 8,
    "waymo open sim agents challenge": 8,
    "waymo sim agents": 7,
    "sim agents": 6,
    "traffic simulation": 6,
    "driving simulation": 6,
    "closed-loop": 5,
    "closed loop": 5,
    "multi-agent planning": 5,
    "multi-agent simulation": 5,
    "scenario generation": 4,
    "behavior generation": 4,
    "trajectory simulation": 4,
    "interactive simulation": 4,
    "diffusion": 2,
    "offroad": 2,
    "collision": 2,
    "waymo": 2,
    "autonomous driving": 2,
}

QUALITY_HINTS = {
    "benchmark": 2,
    "leaderboard": 2,
    "evaluation": 2,
    "experiment": 2,
    "ablation": 2,
    "code": 1,
    "github": 1,
    "dataset": 1,
    "simulator": 1,
    "offroad": 1,
    "collision": 1,
}

NEGATIVE_HINTS = {
    "survey": -1,
    "tutorial": -2,
    "position paper": -2,
    "opinion": -2,
}

DEFAULT_ARXIV_QUERIES = [
    'all:"WOSAC"',
    'all:"Waymo Open Sim Agents Challenge"',
    'all:"Waymo sim agents"',
    'all:"traffic simulation" AND all:"autonomous driving"',
    'all:"closed-loop" AND all:"autonomous driving"',
    'all:"multi-agent planning" AND all:"driving"',
    'all:"scenario generation" AND all:"autonomous driving"',
    'all:"diffusion" AND all:"traffic simulation"',
]

DEFAULT_GITHUB_QUERIES = [
    "WOSAC",
    '"Waymo sim agents"',
    '"traffic simulation" autonomous driving',
    '"closed-loop" autonomous driving simulation',
    '"multi-agent planning" autonomous driving',
    "diffusion traffic simulation autonomous driving",
]


@dataclasses.dataclass
class Item:
    title: str
    source: str
    url: str
    published: str | None
    authors: list[str]
    abstract: str
    raw_topics: list[str]
    relevance_score: int
    quality_score: int
    keep: bool
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def http_get(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
) -> requests.Response:
    merged_headers = {"User-Agent": USER_AGENT}
    if headers:
        merged_headers.update(headers)
    resp = requests.get(url, headers=merged_headers, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "")
    return text.strip()


def to_iso_date(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue
    return value[:10]


def is_recent(date_str: str | None, *, today: dt.date | None = None) -> bool:
    if not date_str:
        return False
    try:
        published = dt.date.fromisoformat(date_str)
    except ValueError:
        return False
    reference = today or dt.date.today()
    age_days = (reference - published).days
    return 0 <= age_days <= MAX_DAYS


def collect_topics(text: str) -> tuple[int, list[str], list[str]]:
    haystack = normalize_text(text).lower()
    score = 0
    found: list[str] = []
    notes: list[str] = []

    for keyword, weight in TOPIC_KEYWORDS.items():
        if keyword in haystack:
            score += weight
            found.append(keyword)

    if any(k in haystack for k in ["traffic simulation", "driving simulation", "sim agents"]):
        notes.append("direct simulation keyword match")
    if "waymo" in haystack:
        notes.append("mentions Waymo")
    if "closed-loop" in haystack or "closed loop" in haystack:
        notes.append("mentions closed-loop evaluation")
    if "collision" in haystack or "offroad" in haystack:
        notes.append("mentions simulation safety metrics")

    return score, found, notes


def quality_heuristic(text: str, *, source: str, github_stars: int | None = None) -> tuple[int, list[str]]:
    haystack = normalize_text(text).lower()
    score = 0
    notes: list[str] = []

    for keyword, weight in QUALITY_HINTS.items():
        if keyword in haystack:
            score += weight
            notes.append(f"quality hint: {keyword}")

    for keyword, weight in NEGATIVE_HINTS.items():
        if keyword in haystack:
            score += weight
            notes.append(f"negative hint: {keyword}")

    if source == "waymo":
        score += 2
        notes.append("official Waymo research source")
    elif source == "arxiv":
        score += 1
        notes.append("arXiv preprint accepted by preference")
    elif source == "github" and github_stars is not None:
        if github_stars >= 200:
            score += 3
            notes.append("GitHub repo has 200+ stars")
        elif github_stars >= 50:
            score += 2
            notes.append("GitHub repo has 50+ stars")
        elif github_stars >= 10:
            score += 1
            notes.append("GitHub repo has 10+ stars")

    return score, notes


def should_keep(relevance_score: int, quality_score: int) -> bool:
    return relevance_score >= 6 and (relevance_score + quality_score) >= 8


def fetch_arxiv(max_results_per_query: int = 20) -> list[Item]:
    items: list[Item] = []

    for query in DEFAULT_ARXIV_QUERIES:
        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results_per_query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        resp = http_get(ARXIV_BASE, params=params)
        feed = feedparser.parse(resp.text)

        for entry in feed.entries:
            title = normalize_text(entry.get("title", ""))
            abstract = normalize_text(entry.get("summary", ""))
            authors = [a.get("name", "") for a in entry.get("authors", [])]
            url = entry.get("link", "")
            published = to_iso_date(entry.get("published"))

            if published and not is_recent(published):
                continue

            relevance_score, topics, notes = collect_topics(f"{title}\n{abstract}")
            quality_score, q_notes = quality_heuristic(f"{title}\n{abstract}", source="arxiv")
            keep = should_keep(relevance_score, quality_score)

            items.append(
                Item(
                    title=title,
                    source="arxiv",
                    url=url,
                    published=published,
                    authors=authors,
                    abstract=abstract,
                    raw_topics=topics,
                    relevance_score=relevance_score,
                    quality_score=quality_score,
                    keep=keep,
                    notes=notes + q_notes + [f"matched query: {query}"],
                )
            )

        time.sleep(ARXIV_SLEEP_SECONDS)

    return items


def fetch_github(max_results_per_query: int = 10) -> list[Item]:
    items: list[Item] = []
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    for query in DEFAULT_GITHUB_QUERIES:
        params = {
            "q": query,
            "sort": "updated",
            "order": "desc",
            "per_page": max_results_per_query,
        }
        resp = http_get(GITHUB_SEARCH_REPOS, headers=headers, params=params)
        payload = resp.json()

        for repo in payload.get("items", []):
            title = repo.get("full_name", "")
            description = normalize_text(repo.get("description", ""))
            authors = [repo.get("owner", {}).get("login", "")]
            url = repo.get("html_url", "")
            published = to_iso_date(repo.get("updated_at"))
            stars = int(repo.get("stargazers_count", 0))

            if published and not is_recent(published):
                continue

            text = f"{title}\n{description}"
            relevance_score, topics, notes = collect_topics(text)
            quality_score, q_notes = quality_heuristic(text, source="github", github_stars=stars)
            keep = should_keep(relevance_score, quality_score)

            items.append(
                Item(
                    title=title,
                    source="github",
                    url=url,
                    published=published,
                    authors=authors,
                    abstract=description,
                    raw_topics=topics,
                    relevance_score=relevance_score,
                    quality_score=quality_score,
                    keep=keep,
                    notes=notes + q_notes + [f"github stars: {stars}", f"matched query: {query}"],
                )
            )

    return items


def fetch_waymo_research(limit: int = 40) -> list[Item]:
    items: list[Item] = []
    resp = http_get(WAYMO_RESEARCH_URL)
    soup = BeautifulSoup(resp.text, "html.parser")

    seen_urls: set[str] = set()
    anchors = soup.select("a[href]")

    for anchor in anchors:
        href = anchor.get("href", "")
        if not href:
            continue
        if href.startswith("/"):
            href = "https://waymo.com" + href
        if not href.startswith("https://waymo.com/research/"):
            continue
        if href.rstrip("/") == WAYMO_RESEARCH_URL.rstrip("/"):
            continue
        if href in seen_urls:
            continue
        seen_urls.add(href)

        title = normalize_text(anchor.get_text(" ", strip=True))
        if not title:
            continue

        text = title
        relevance_score, topics, notes = collect_topics(text)
        quality_score, q_notes = quality_heuristic(text, source="waymo")
        keep = should_keep(relevance_score, quality_score)

        items.append(
            Item(
                title=title,
                source="waymo",
                url=href,
                published=None,
                authors=[],
                abstract="",
                raw_topics=topics,
                relevance_score=relevance_score,
                quality_score=quality_score,
                keep=keep,
                notes=notes + q_notes,
            )
        )

        if len(items) >= limit:
            break

    return items


def dedupe_items(items: Iterable[Item]) -> list[Item]:
    best_by_key: dict[str, Item] = {}

    for item in items:
        key = normalize_text(item.title).lower() or item.url
        prev = best_by_key.get(key)
        if prev is None:
            best_by_key[key] = item
            continue

        prev_score = prev.relevance_score + prev.quality_score
        curr_score = item.relevance_score + item.quality_score
        if curr_score > prev_score:
            best_by_key[key] = item

    out = list(best_by_key.values())
    out.sort(key=lambda x: ((x.relevance_score + x.quality_score), x.published or ""), reverse=True)
    return out


def save_json(items: list[Item], path: Path) -> None:
    path.write_text(
        json.dumps([item.to_dict() for item in items], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_csv(items: list[Item], path: Path) -> None:
    fieldnames = [
        "title",
        "source",
        "url",
        "published",
        "authors",
        "abstract",
        "raw_topics",
        "relevance_score",
        "quality_score",
        "keep",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            row = item.to_dict()
            row["authors"] = "; ".join(item.authors)
            row["raw_topics"] = "; ".join(item.raw_topics)
            row["notes"] = " | ".join(item.notes)
            writer.writerow(row)


def save_weekly_report(items: list[Item], path: Path, report_date: str) -> None:
    lines = [f"# Weekly Research Update — {report_date}", "", "## New papers", ""]
    if not items:
        lines.append("No new relevant papers this week.")
    else:
        for index, item in enumerate(items, 1):
            lines.extend(
                [
                    f"### {index}. {item.title}",
                    f"- Source: {item.source}",
                    f"- Date: {item.published or 'unknown'}",
                    f"- Relevance score: {item.relevance_score}",
                    f"- Quality score: {item.quality_score}",
                    f"- Topics: {', '.join(item.raw_topics) if item.raw_topics else 'n/a'}",
                    f"- Link: {item.url}",
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    today = dt.date.today().isoformat()

    print("[1/4] Fetching arXiv...")
    arxiv_items = fetch_arxiv(max_results_per_query=15)

    print("[2/4] Fetching GitHub...")
    github_items = fetch_github(max_results_per_query=8)

    print("[3/4] Fetching Waymo research...")
    waymo_items = fetch_waymo_research(limit=50)

    print("[4/4] Merging and filtering...")
    all_items = dedupe_items(arxiv_items + github_items + waymo_items)
    kept_items = [item for item in all_items if item.keep]

    latest_json = DATA_DIR / "latest_kept.json"
    latest_csv = DATA_DIR / "latest_kept.csv"
    archive_json = ARCHIVE_DIR / f"{today}_kept.json"
    report_path = REPORTS_DIR / f"{today}.md"

    save_json(kept_items, latest_json)
    save_json(kept_items, archive_json)
    save_csv(kept_items, latest_csv)
    save_weekly_report(kept_items, report_path, today)

    print(f"Kept after filtering: {len(kept_items)}")
    print("Top kept items:")
    for item in kept_items[:10]:
        total = item.relevance_score + item.quality_score
        print(f"- [{item.source}] score={total:02d} | {item.title} | {item.url}")

    print("Saved files:")
    print(f"- {latest_json}")
    print(f"- {latest_csv}")
    print(f"- {archive_json}")
    print(f"- {report_path}")


if __name__ == "__main__":
    main()
