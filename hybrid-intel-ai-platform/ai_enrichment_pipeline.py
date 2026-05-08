"""
AI Enrichment Pipeline
----------------------
Reads entities.csv, reports.csv, edges.csv from ./data/ and produces
AI-enriched outputs under ./output/:
  - reports_enriched.csv       (per-report summaries + extracted entities)
  - entities_enriched.csv      (AI-adjusted threat scores + anomaly flags)
  - unified_graph_ai.json      (graph artifact consumed by the dashboard)

Provider selection (see llm_providers.py for full details):
  - Auto: ANTHROPIC_API_KEY -> Claude, else OPENAI_API_KEY -> OpenAI,
    else heuristic.
  - Override with LLM_PROVIDER=anthropic|openai|heuristic.

Provider influences:
  - Per-report summarization (single-sentence analyst summaries)
  - Per-report named-entity extraction

Always heuristic (deterministic, no API cost):
  - TF cosine similarity between reports
  - Threat-score adjustment (recurrence + centrality)
  - Anomaly flag (z-score)
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

from llm_providers import Provider, get_provider

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


# ---------------------------------------------------------------------------
# Provider-routed enrichment helpers
# ---------------------------------------------------------------------------

def summarize_report(excerpt: str, provider: Provider | None = None,
                     max_words: int = 18) -> str:
    """Single-sentence summary of a report excerpt via the active provider."""
    p = provider or get_provider()
    return p.summarize(excerpt, max_words=max_words)


def extract_entities_llm(excerpt: str,
                         provider: Provider | None = None) -> list[dict]:
    """Named-entity extraction via the active provider."""
    p = provider or get_provider()
    return p.extract_entities(excerpt)


# ---------------------------------------------------------------------------
# Deterministic enrichment (always heuristic)
# ---------------------------------------------------------------------------

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "for",
    "with", "by", "near", "from", "into", "as", "is", "was", "were", "be",
    "been", "being", "that", "this", "these", "those", "it", "its", "their",
    "they", "his", "her", "he", "she", "we", "our", "have", "has", "had",
    "but", "not", "no", "so", "if", "then", "than", "while", "during",
    "over", "between", "through",
}


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[A-Za-z]+", (text or "").lower())
            if t not in STOPWORDS and len(t) > 2]


def extract_keywords(excerpt: str, top_k: int = 5) -> list[str]:
    counts = Counter(_tokens(excerpt))
    return [w for w, _ in counts.most_common(top_k)]


def cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    num = sum(a[k] * b[k] for k in common)
    da = math.sqrt(sum(v * v for v in a.values()))
    db = math.sqrt(sum(v * v for v in b.values()))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def tf_vector(text: str) -> dict[str, float]:
    counts = Counter(_tokens(text))
    total = sum(counts.values()) or 1
    return {w: c / total for w, c in counts.items()}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    entities = pd.read_csv(DATA_DIR / "entities.csv")
    reports = pd.read_csv(DATA_DIR / "reports.csv")
    edges = pd.read_csv(DATA_DIR / "edges.csv")
    return entities, reports, edges


def enrich_reports(reports: pd.DataFrame,
                   provider: Provider) -> pd.DataFrame:
    reports = reports.copy()
    reports["ai_summary"] = reports["excerpt"].apply(
        lambda t: provider.summarize(t, max_words=18)
    )
    reports["ai_keywords"] = reports["excerpt"].apply(
        lambda t: ", ".join(extract_keywords(t))
    )
    reports["ai_extracted_entities"] = reports["excerpt"].apply(
        lambda t: ", ".join(
            f"{e.get('text','')}({e.get('type','?')})"
            for e in provider.extract_entities(t)
        )
    )
    return reports


def compute_similarity_links(reports: pd.DataFrame,
                             threshold: float = 0.25) -> list[dict]:
    vectors = {row["report_id"]: tf_vector(row["excerpt"])
               for _, row in reports.iterrows()}
    ids = list(vectors.keys())
    links: list[dict] = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            sim = cosine_similarity(vectors[a], vectors[b])
            if sim >= threshold:
                links.append({
                    "source_report": a,
                    "target_report": b,
                    "similarity": round(sim, 3),
                })
    return links


def adjust_threat_scores(entities: pd.DataFrame,
                         reports: pd.DataFrame,
                         edges: pd.DataFrame) -> pd.DataFrame:
    entities = entities.copy()

    mention_counts: Counter = Counter()
    for _, r in reports.iterrows():
        for ent in str(r.get("entities", "")).split("|"):
            ent = ent.strip()
            if ent:
                mention_counts[ent] += 1

    degree: Counter = Counter()
    for _, e in edges.iterrows():
        degree[e["source"]] += 1
        degree[e["target"]] += 1

    base = entities["threat_score"].astype(float)
    mean_score = base.mean()
    std_score = base.std() or 1e-6

    ai_scores, anomaly_flags = [], []
    for _, row in entities.iterrows():
        name = row["name"]
        score = float(row["threat_score"])
        recurrence_boost = min(0.10, 0.02 * mention_counts.get(name, 0))
        centrality_boost = min(0.08, 0.02 * degree.get(name, 0))
        adjusted = min(1.0, score + recurrence_boost + centrality_boost)
        ai_scores.append(round(adjusted, 3))
        z = (score - mean_score) / std_score
        anomaly_flags.append(bool(abs(z) >= 1.0))

    entities["ai_threat_score"] = ai_scores
    entities["anomaly_flag"] = anomaly_flags
    return entities


def build_unified_graph(entities: pd.DataFrame,
                        reports: pd.DataFrame,
                        edges: pd.DataFrame,
                        similarity_links: list[dict],
                        provider_name: str) -> dict:
    nodes = []
    for _, row in entities.iterrows():
        nodes.append({
            "id": row["name"],
            "type": row["type"],
            "threat_score": float(row.get("ai_threat_score",
                                          row["threat_score"])),
            "base_threat_score": float(row["threat_score"]),
            "confidence": row["confidence"],
            "mentions": int(row["mentions"]),
            "last_seen": row["last_seen"],
            "primary_location": row["primary_location"],
            "linked_actor": bool(row["linked_actor"]),
            "status": row["status"],
            "anomaly_flag": bool(row.get("anomaly_flag", False)),
            "latitude": float(row["latitude"])
            if pd.notna(row.get("latitude")) else None,
            "longitude": float(row["longitude"])
            if pd.notna(row.get("longitude")) else None,
        })

    graph_edges = [
        {
            "source": r["source"],
            "target": r["target"],
            "relationship": r["relationship"],
            "weight": float(r["weight"]),
        }
        for _, r in edges.iterrows()
    ]

    return {
        "metadata": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "pipeline_version": "ai-enrichment-2.0",
            "llm_provider": provider_name,
            "node_count": len(nodes),
            "edge_count": len(graph_edges),
            "report_count": len(reports),
        },
        "nodes": nodes,
        "edges": graph_edges,
        "reports": reports.to_dict(orient="records"),
        "ai_artifacts": {
            "report_similarity_links": similarity_links,
        },
    }


def main() -> None:
    provider = get_provider()
    print(f"[0/5] Active LLM provider: {provider.name}")

    print("[1/5] Loading inputs from", DATA_DIR.resolve())
    entities, reports, edges = load_inputs()

    print("[2/5] Enriching reports (summarize + entity extraction)")
    reports_enriched = enrich_reports(reports, provider)
    reports_enriched.to_csv(OUTPUT_DIR / "reports_enriched.csv", index=False)

    print("[3/5] Adjusting threat scores + flagging anomalies")
    entities_enriched = adjust_threat_scores(entities, reports, edges)
    entities_enriched.to_csv(OUTPUT_DIR / "entities_enriched.csv", index=False)

    print("[4/5] Computing report-to-report similarity links")
    similarity_links = compute_similarity_links(reports_enriched)

    print("[5/5] Writing unified_graph_ai.json")
    graph = build_unified_graph(
        entities_enriched, reports_enriched, edges,
        similarity_links, provider.name,
    )
    out_path = OUTPUT_DIR / "unified_graph_ai.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)

    print("Done. Artifacts written to", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
