"""
AWS Lambda: unified_graph generator
-----------------------------------
Reads CSVs from S3, builds unified_graph.json, writes it back.

Trigger: S3 PUT events on the input/ prefix, or manual invocation
with the following event payload:

{
  "bucket": "YOUR_BUCKET_NAME",
  "entities_key": "input/entities.csv",
  "reports_key": "input/reports.csv",
  "edges_key": "input/edges.csv",
  "output_key": "output/unified_graph.json"
}

IAM permissions required on the Lambda execution role:
  - s3:GetObject on the input prefix
  - s3:PutObject on the output prefix
"""

from __future__ import annotations

import csv
import io
import json
import os
from datetime import datetime
from typing import Any

import boto3

s3 = boto3.client("s3")


def _read_csv_from_s3(bucket: str, key: str) -> list[dict]:
    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(body)))


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def build_graph(entities: list[dict],
                reports: list[dict],
                edges: list[dict]) -> dict:
    nodes = []
    for row in entities:
        nodes.append({
            "id": row.get("name"),
            "type": row.get("type"),
            "threat_score": _to_float(row.get("threat_score")),
            "confidence": row.get("confidence"),
            "mentions": _to_int(row.get("mentions")),
            "last_seen": row.get("last_seen"),
            "primary_location": row.get("primary_location"),
            "linked_actor": _to_bool(row.get("linked_actor")),
            "status": row.get("status"),
            "latitude": _to_float(row.get("latitude"))
            if row.get("latitude") else None,
            "longitude": _to_float(row.get("longitude"))
            if row.get("longitude") else None,
        })

    graph_edges = [
        {
            "source": e.get("source"),
            "target": e.get("target"),
            "relationship": e.get("relationship"),
            "weight": _to_float(e.get("weight")),
        }
        for e in edges
    ]

    return {
        "metadata": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "pipeline_version": "lambda-1.0",
            "node_count": len(nodes),
            "edge_count": len(graph_edges),
            "report_count": len(reports),
        },
        "nodes": nodes,
        "edges": graph_edges,
        "reports": reports,
    }


def lambda_handler(event: dict, context: Any) -> dict:
    bucket = event.get("bucket") or os.environ.get("BUCKET_NAME")
    entities_key = event.get("entities_key", "input/entities.csv")
    reports_key = event.get("reports_key", "input/reports.csv")
    edges_key = event.get("edges_key", "input/edges.csv")
    output_key = event.get("output_key", "output/unified_graph.json")

    if not bucket:
        return {"statusCode": 400,
                "body": "Missing bucket (event.bucket or BUCKET_NAME env)"}

    entities = _read_csv_from_s3(bucket, entities_key)
    reports = _read_csv_from_s3(bucket, reports_key)
    edges = _read_csv_from_s3(bucket, edges_key)

    graph = build_graph(entities, reports, edges)

    s3.put_object(
        Bucket=bucket,
        Key=output_key,
        Body=json.dumps(graph, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

    return {
        "statusCode": 200,
        "body": json.dumps({
            "bucket": bucket,
            "output_key": output_key,
            "nodes": graph["metadata"]["node_count"],
            "edges": graph["metadata"]["edge_count"],
            "reports": graph["metadata"]["report_count"],
        }),
    }
