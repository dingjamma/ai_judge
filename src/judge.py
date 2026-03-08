"""
Phase 3: The AI Judge
RAG pipeline — retrieve similar precedents via FAISS, call Claude on Bedrock,
parse and store verdicts.

Usage:
    python src/judge.py --case-id 12345
    python src/judge.py --all
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import boto3
import faiss
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from db import init_db, upsert_case
from embed import embed_text, get_bedrock_client

load_dotenv()

DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"
FAISS_INDEX_PATH = Path(__file__).parent.parent / "data" / "faiss.index"
CASE_MAP_PATH = Path(__file__).parent.parent / "data" / "case_map.json"

JUDGE_MODEL = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
TOP_K = 3

PROMPT_TEMPLATE = """\
You are an impartial judge. Based on the following case facts and similar precedents,
render a verdict and explain your reasoning.

CASE FACTS:
{facts}

SIMILAR PRECEDENTS:
{retrieved_cases}

Respond in this exact JSON format (no markdown, no extra text):
{{
  "verdict": "affirmed" | "reversed" | "vacated",
  "confidence": 0.0-1.0,
  "reasoning": "2-3 sentence explanation",
  "fairness_score": 0.0-1.0,
  "fairness_notes": "any concerns about bias or fairness"
}}"""


def load_index_and_map():
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(CASE_MAP_PATH) as f:
        case_map = json.load(f)
    return index, case_map


def load_case(case_id: str) -> dict | None:
    path = DATA_PROCESSED / f"{case_id}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def retrieve_similar(client, index, case_map, facts: str, current_id: str) -> list[dict]:
    """Return TOP_K most similar cases (excluding the current one)."""
    vec = np.array([embed_text(client, facts)], dtype="float32")
    distances, indices = index.search(vec, TOP_K + 1)

    similar = []
    for idx in indices[0]:
        if idx < 0 or idx >= len(case_map):
            continue
        cid = case_map[idx]
        if cid == current_id:
            continue
        case = load_case(cid)
        if case:
            similar.append(case)
        if len(similar) >= TOP_K:
            break
    return similar


def format_precedents(cases: list[dict]) -> str:
    lines = []
    for i, c in enumerate(cases, 1):
        lines.append(
            f"[{i}] {c.get('name', 'Unknown')} ({c.get('year', '?')}): "
            f"{c.get('facts', '')[:500]} "
            f"[Actual verdict: {c.get('actual_verdict', 'unknown')}]"
        )
    return "\n\n".join(lines)


def call_claude(bedrock_client, prompt: str) -> dict:
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "messages": [{"role": "user", "content": prompt}],
    })
    resp = bedrock_client.invoke_model(
        modelId=JUDGE_MODEL,
        body=body,
        accept="application/json",
        contentType="application/json",
    )
    result = json.loads(resp["body"].read())
    text = result["content"][0]["text"].strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


def judge_case(case: dict, embed_client, bedrock_client, index, case_map) -> dict:
    facts = case.get("facts") or ""
    similar = retrieve_similar(embed_client, index, case_map, facts, case["id"])
    precedents_str = format_precedents(similar)

    prompt = PROMPT_TEMPLATE.format(facts=facts[:3000], retrieved_cases=precedents_str)
    ai_response = call_claude(bedrock_client, prompt)

    return {
        "id": case["id"],
        "name": case.get("name"),
        "year": case.get("year"),
        "case_type": case.get("case_type"),
        "facts": facts[:2000],
        "actual_verdict": case.get("actual_verdict"),
        "ai_verdict": ai_response.get("verdict"),
        "ai_confidence": ai_response.get("confidence"),
        "ai_reasoning": ai_response.get("reasoning"),
        "fairness_score": ai_response.get("fairness_score"),
        "fairness_notes": ai_response.get("fairness_notes"),
        "match": (
            ai_response.get("verdict") == case.get("actual_verdict")
            if case.get("actual_verdict") else None
        ),
        "judged_at": datetime.now(timezone.utc).isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description="AI Judge — render verdicts on SCOTUS cases")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--case-id", type=str, help="Judge a single case by ID")
    group.add_argument("--all", action="store_true", help="Judge all processed cases")
    args = parser.parse_args()

    init_db()
    embed_client = get_bedrock_client()
    bedrock_client = get_bedrock_client()
    index, case_map = load_index_and_map()

    if args.case_id:
        case = load_case(args.case_id)
        if not case:
            print(f"Case {args.case_id} not found.")
            return
        result = judge_case(case, embed_client, bedrock_client, index, case_map)
        upsert_case(result)
        print(json.dumps(result, indent=2))
    else:
        case_files = sorted(DATA_PROCESSED.glob("*.json"))
        for path in tqdm(case_files, desc="Judging cases"):
            with open(path, encoding="utf-8") as f:
                case = json.load(f)
            if not case.get("actual_verdict"):
                continue  # skip cases with no ground-truth verdict
            try:
                result = judge_case(case, embed_client, bedrock_client, index, case_map)
                upsert_case(result)
            except Exception as e:
                print(f"\n[WARN] Failed to judge {case['id']}: {e}")


if __name__ == "__main__":
    main()
