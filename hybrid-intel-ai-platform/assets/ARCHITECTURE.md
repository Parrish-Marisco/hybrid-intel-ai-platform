# Architecture Reference

## End-to-end data flow

```
[ HUMINT / OSINT / TECHINT CSVs ]
              │
              ▼
[ AI Enrichment Pipeline ]
   - Report summarization
   - Keyword extraction
   - Semantic similarity (TF cosine)
   - Threat score adjustment
   - Anomaly detection (z-score)
              │
              ▼
[ unified_graph_ai.json ]
              │
              ▼
[ Streamlit Dashboard ]
   - Threat prioritization
   - Link analysis (graph)
   - Geospatial map
   - Report explorer
```

## Cloud variant

```
[ S3 input/ prefix ]  ───►  [ Lambda lambda_function ]  ───►  [ S3 output/ prefix ]
       (CSVs)                        (handler)                   (unified_graph.json)
```

Trigger Lambda on S3 PUT events to keep the artifact fresh.
