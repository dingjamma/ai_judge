"""
Phase 2: Embeddings + FAISS Index
Embeds processed cases via AWS Bedrock Titan and builds a FAISS index.

Usage:
    python src/embed.py
"""

import json
import os
from pathlib import Path

import boto3
import faiss
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

DATA_PROCESSED = Path(__file__).parent.parent / "data" / "processed"
FAISS_INDEX_PATH = Path(__file__).parent.parent / "data" / "faiss.index"
CASE_MAP_PATH = Path(__file__).parent.parent / "data" / "case_map.json"

EMBEDDING_MODEL = "amazon.titan-embed-text-v1"
EMBED_DIM = 1536  # Titan text embedding dimension


def get_bedrock_client():
    return boto3.client(
        "bedrock-runtime",
        region_name=os.getenv("AWS_REGION", "us-east-1"),
    )


def embed_text(client, text: str) -> list[float]:
    """Call Bedrock Titan to embed a single text string."""
    body = json.dumps({"inputText": text[:8000]})  # Titan max ~8k chars
    resp = client.invoke_model(
        modelId=EMBEDDING_MODEL,
        body=body,
        accept="application/json",
        contentType="application/json",
    )
    result = json.loads(resp["body"].read())
    return result["embedding"]


def build_index():
    client = get_bedrock_client()

    case_files = sorted(DATA_PROCESSED.glob("*.json"))
    if not case_files:
        print("No processed cases found. Run ingest.py first.")
        return

    index = faiss.IndexFlatL2(EMBED_DIM)
    case_map = []  # list of case IDs in index order

    for path in tqdm(case_files, desc="Embedding cases"):
        with open(path, encoding="utf-8") as f:
            case = json.load(f)

        facts = case.get("facts") or ""
        if not facts:
            continue

        embedding = embed_text(client, facts)
        vec = np.array([embedding], dtype="float32")
        index.add(vec)
        case_map.append(case["id"])

    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(CASE_MAP_PATH, "w") as f:
        json.dump(case_map, f)

    print(f"Indexed {index.ntotal} cases -> {FAISS_INDEX_PATH}")


if __name__ == "__main__":
    build_index()
