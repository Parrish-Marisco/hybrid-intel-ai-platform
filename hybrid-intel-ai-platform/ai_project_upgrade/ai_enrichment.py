
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

KNOWN_LOCATIONS = {
    "bamako", "gao", "mopti", "niger border", "niger corridor node",
    "warehouse 17", "transit yard east", "northern supply route"
}

KNOWN_GROUP_HINTS = {"network", "cell", "team", "node", "route", "group", "support"}

RISK_TERMS = {
    "movement": 0.10,
    "coordination": 0.12,
    "facilitation": 0.12,
    "late-night": 0.08,
    "cross-border": 0.18,
    "leadership": 0.16,
    "warehouse": 0.10,
    "transport": 0.10,
    "route": 0.12,
    "border": 0.14,
    "procurement": 0.08,
    "transfer": 0.10,
    "meeting": 0.08,
}

def load_csvs(data_dir: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_path = Path(data_dir)
    entities = pd.read_csv(data_path / "entities.csv")
    reports = pd.read_csv(data_path / "reports.csv")
    edges = pd.read_csv(data_path / "edges.csv")
    return entities, reports, edges

def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9\-]+", str(text).lower())

def sentence_summary(text: str, max_sentences: int = 2) -> str:
    parts = re.split(r"(?<=[.!?])\s+", str(text).strip())
    parts = [p.strip() for p in parts if p.strip()]
    return " ".join(parts[:max_sentences]) if parts else ""

def heuristic_entity_extraction(text: str) -> List[str]:
    extracted = set()
    title_case = re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-zA-Z0-9\-]+){0,3})\b", str(text))
    for item in title_case:
        if len(item) > 2:
            extracted.add(item.strip())
    lowered = str(text).lower()
    for loc in KNOWN_LOCATIONS:
        if loc in lowered:
            extracted.add(loc.title())
    return sorted(extracted)

def jaccard_similarity(a: str, b: str) -> float:
    ta = set(tokenize(a))
    tb = set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)

def compute_similarity_matrix(reports: pd.DataFrame) -> List[Dict]:
    rows = []
    for i in range(len(reports)):
        for j in range(i + 1, len(reports)):
            s = jaccard_similarity(reports.iloc[i]["excerpt"], reports.iloc[j]["excerpt"])
            rows.append({
                "report_id_a": reports.iloc[i]["report_id"],
                "report_id_b": reports.iloc[j]["report_id"],
                "similarity": round(s, 4),
            })
    return rows

def score_report_anomaly(row: pd.Series) -> float:
    score = 0.0
    text = str(row.get("excerpt", "")).lower()
    for term, weight in RISK_TERMS.items():
        if term in text:
            score += weight
    confidence = str(row.get("confidence", "")).lower()
    if confidence == "high":
        score += 0.15
    elif confidence == "medium":
        score += 0.08
    source_type = str(row.get("source_type", "")).upper()
    if source_type == "TECHINT":
        score += 0.10
    elif source_type == "HUMINT":
        score += 0.07
    entity_count = len([x for x in str(row.get("entities", "")).split("|") if x.strip()])
    score += min(entity_count * 0.03, 0.15)
    return min(round(score, 4), 1.0)

def enrich_reports(reports: pd.DataFrame) -> pd.DataFrame:
    df = reports.copy()
    df["ai_summary"] = df["excerpt"].apply(sentence_summary)
    df["ai_entities"] = df["excerpt"].apply(heuristic_entity_extraction)
    df["ai_anomaly_score"] = df.apply(score_report_anomaly, axis=1)
    df["ai_priority"] = df["ai_anomaly_score"].apply(
        lambda x: "High" if x >= 0.75 else "Elevated" if x >= 0.50 else "Normal"
    )
    return df

def build_unified_graph(entities: pd.DataFrame, reports: pd.DataFrame, edges: pd.DataFrame) -> Dict:
    nodes = []
    for _, row in entities.iterrows():
        nodes.append({
            "id": row["name"],
            "type": row["type"],
            "threat_score": float(row["threat_score"]),
            "confidence": row["confidence"],
            "mentions": int(row["mentions"]),
            "last_seen": row["last_seen"],
            "primary_location": row["primary_location"],
            "status": row["status"],
        })
    edge_records = []
    for _, row in edges.iterrows():
        edge_records.append({
            "source": row["source"],
            "target": row["target"],
            "relationship": row["relationship"],
            "weight": float(row["weight"]),
        })
    report_records = []
    for _, row in reports.iterrows():
        report_records.append({
            "report_id": row["report_id"],
            "date": row["date"],
            "source_type": row["source_type"],
            "confidence": row["confidence"],
            "location": row["location"],
            "excerpt": row["excerpt"],
            "entities": [x.strip() for x in str(row["entities"]).split("|") if x.strip()],
            "relevance": row["relevance"],
        })
    return {
        "metadata": {
            "generated_by": "ai_enrichment_pipeline",
            "node_count": len(nodes),
            "edge_count": len(edge_records),
            "report_count": len(report_records),
        },
        "nodes": nodes,
        "edges": edge_records,
        "reports": report_records,
    }

def update_graph_with_ai(graph: Dict, enriched_reports: pd.DataFrame, similarity_links: List[Dict]) -> Dict:
    graph = dict(graph)
    graph["ai"] = {
        "similarity_links": similarity_links,
        "report_enrichment": enriched_reports[[
            "report_id", "ai_summary", "ai_entities", "ai_anomaly_score", "ai_priority"
        ]].to_dict(orient="records"),
    }
    return graph

def save_outputs(enriched_reports: pd.DataFrame, similarity_links: List[Dict], graph: Dict, out_dir: str = "output") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    reports_save = enriched_reports.copy()
    reports_save["ai_entities"] = reports_save["ai_entities"].apply(json.dumps)
    reports_save.to_csv(out / "reports_ai.csv", index=False)
    with open(out / "report_similarity.json", "w", encoding="utf-8") as f:
        json.dump(similarity_links, f, indent=2)
    with open(out / "unified_graph_ai.json", "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)

def main(data_dir: str = "data", out_dir: str = "output") -> None:
    entities, reports, edges = load_csvs(data_dir)
    enriched_reports = enrich_reports(reports)
    similarity_links = compute_similarity_matrix(enriched_reports)
    graph = build_unified_graph(entities, reports, edges)
    graph = update_graph_with_ai(graph, enriched_reports, similarity_links)
    save_outputs(enriched_reports, similarity_links, graph, out_dir)
    print(f"AI outputs written to {out_dir}/")

if __name__ == "__main__":
    main()
