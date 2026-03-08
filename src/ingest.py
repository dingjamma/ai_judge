"""
Phase 1: Data Ingestion
Fetches SCOTUS cases from CourtListener V4 Search API, then fetches opinion text.

The V4 search result embeds the opinion list directly, so we need only one
extra API call per case (to fetch the full opinion text). No cluster endpoint.

Usage:
    python src/ingest.py --limit 200
    python src/ingest.py --limit 200 --end-year 2010
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

BASE_URL = os.getenv("COURTLISTENER_BASE_URL", "https://www.courtlistener.com/api/rest/v4/")
COURTLISTENER_TOKEN = os.getenv("COURTLISTENER_TOKEN", "")
DATA_RAW = Path(__file__).parent.parent / "data" / "raw"
DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"

REQUEST_DELAY = 1.0   # 1 req/sec — comfortably within rate limits
RETRY_BACKOFF = 10
MAX_RETRIES = 3


def _headers() -> dict:
    h = {"User-Agent": "ai-judge-backtester/1.0 (research project)"}
    if COURTLISTENER_TOKEN:
        h["Authorization"] = f"Token {COURTLISTENER_TOKEN}"
    return h


def get_with_retry(url: str, params: dict | None = None, timeout: int = 45) -> requests.Response:
    delay = RETRY_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, headers=_headers(), timeout=timeout)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", delay))
                print(f"\n[RATE LIMIT] Waiting {wait}s...")
                time.sleep(wait)
                continue
            return resp
        except (requests.Timeout, requests.ConnectionError) as e:
            if attempt == MAX_RETRIES:
                raise
            print(f"\n[WARN] {type(e).__name__} attempt {attempt}, retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2
    return resp


def fetch_opinion_text(opinion_id: int | str) -> str:
    url = f"{BASE_URL}opinions/{opinion_id}/"
    resp = get_with_retry(url)
    if not resp.ok:
        return ""
    data = resp.json()
    return (
        data.get("plain_text")
        or data.get("html_with_citations")
        or data.get("html_columbia")
        or data.get("html")
        or ""
    )


# ── verdict extraction ──────────────────────────────────────────────────────

_REVERSE_RE = re.compile(
    r"\b(revers(ed|es|ing)|we reverse|judgment.*reversed|is reversed)\b", re.I
)
_VACATE_RE = re.compile(
    r"\b(vacat(ed|es|ing)|we vacate|remand(ed)?)\b", re.I
)
_AFFIRM_RE = re.compile(
    r"\b(affirm(ed)?|we affirm|judgment.*affirmed|is affirmed)\b", re.I
)


def extract_verdict(text: str) -> str | None:
    """Scan the tail of the opinion for the judgment language."""
    tail = text[-2000:] if len(text) > 2000 else text
    if _REVERSE_RE.search(tail):
        return "reversed"
    if _VACATE_RE.search(tail):
        return "vacated"
    if _AFFIRM_RE.search(tail):
        return "affirmed"
    return None


def classify_case(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ["criminal", "murder", "robbery", "drug", "felony", "homicide", "assault", "prosecution"]):
        return "criminal"
    if any(w in t for w in ["first amendment", "equal protection", "due process", "fourth amendment", "constitutional"]):
        return "constitutional"
    if any(w in t for w in ["tax", "revenue", "irs", "internal revenue"]):
        return "tax"
    if any(w in t for w in ["antitrust", "patent", "copyright", "trademark", "intellectual property"]):
        return "intellectual_property"
    if any(w in t for w in ["civil", "contract", "tort", "damages", "negligence", "liability"]):
        return "civil"
    return "unknown"


def parse_opinions_field(raw) -> list[dict]:
    """The V4 search `opinions` field is a list (sometimes JSON-encoded string)."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw.replace("'", '"'))
        except Exception:
            return []
    return []


def process_hit(hit: dict, opinion_text: str) -> dict:
    date_filed = hit.get("dateFiled") or ""
    year = int(date_filed[:4]) if date_filed and len(date_filed) >= 4 else None

    # Use syllabus first, fall back to opinion text snippet
    facts = (
        hit.get("syllabus")
        or hit.get("procedural_history")
        or opinion_text[:3000]
    )
    facts = re.sub(r"<[^>]+>", " ", facts or "").strip()

    classify_text = " ".join([
        hit.get("suitNature") or "",
        hit.get("posture") or "",
        facts[:1000],
    ])

    return {
        "id": str(hit["cluster_id"]),
        "name": hit.get("caseNameFull") or hit.get("caseName") or "Unknown",
        "date_filed": date_filed[:10],
        "year": year,
        "case_type": classify_case(classify_text),
        "nature_of_suit": hit.get("suitNature"),
        "scdb_id": hit.get("scdb_id"),
        "actual_verdict": extract_verdict(opinion_text),
        "facts": facts,
        "opinion_text": opinion_text[:10000],
        "source_url": hit.get("absolute_url"),
        "cluster_id": hit["cluster_id"],
        "docket_number": hit.get("docketNumber"),
        "judge": hit.get("judge"),
        "citation": hit.get("citation"),
    }


def ingest(limit: int = 200, end_year: int = 2020) -> int:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    search_url = f"{BASE_URL}search/"
    params = {
        "type": "o",
        "court": "scotus",
        "stat_Precedential": "on",
        "filed_before": f"{end_year}-01-01",
        "order_by": "citeCount desc",  # most-cited = landmark merits decisions
        "page_size": 20,
    }

    saved = 0
    skipped = 0
    pbar = tqdm(total=limit, desc="Ingesting cases")

    resp = get_with_retry(search_url, params=params)
    if not resp.ok:
        print(f"\n[ERROR] Search API {resp.status_code}: {resp.text[:300]}")
        return 0
    page_data = resp.json()

    while page_data and saved < limit:
        results = page_data.get("results", [])
        if not results:
            break

        for hit in results:
            if saved >= limit:
                break

            cluster_id = str(hit.get("cluster_id", ""))
            if not cluster_id:
                skipped += 1
                continue

            processed_path = DATA_PROCESSED / f"{cluster_id}.json"
            raw_path = DATA_RAW / f"{cluster_id}.json"

            if processed_path.exists():
                saved += 1
                pbar.update(1)
                continue

            # Extract opinion ID from embedded opinions list
            opinions = parse_opinions_field(hit.get("opinions", []))
            opinion_id = opinions[0].get("id") if opinions else None

            # Save raw hit
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(hit, f, indent=2, default=str)

            # Fetch opinion text
            opinion_text = ""
            if opinion_id:
                time.sleep(REQUEST_DELAY)
                opinion_text = fetch_opinion_text(opinion_id)

            processed = process_hit(hit, opinion_text)

            # Skip cert denials and orders — real opinions are >500 chars
            if len(opinion_text) < 500:
                skipped += 1
                continue

            with open(processed_path, "w", encoding="utf-8") as f:
                json.dump(processed, f, indent=2, default=str)

            saved += 1
            pbar.update(1)
            pbar.set_postfix({
                "case": processed["name"][:28],
                "verdict": processed["actual_verdict"] or "?",
                "skip": skipped,
            })

        next_url = page_data.get("next")
        if not next_url or saved >= limit:
            break

        time.sleep(REQUEST_DELAY * 2)
        resp = get_with_retry(next_url)
        if not resp.ok:
            print(f"\n[ERROR] Pagination {resp.status_code} — stopping.")
            break
        page_data = resp.json()

    pbar.close()
    print(f"\nDone. Saved {saved} cases. Skipped {skipped}.")
    return saved


def main():
    parser = argparse.ArgumentParser(description="Ingest SCOTUS cases from CourtListener")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--end-year", type=int, default=2020,
                        help="Only fetch cases filed before this year (default: 2020)")
    args = parser.parse_args()

    print(f"Fetching up to {args.limit} SCOTUS cases (filed before {args.end_year})...")
    count = ingest(limit=args.limit, end_year=args.end_year)
    print(f"Ingested {count} cases.")


if __name__ == "__main__":
    main()
