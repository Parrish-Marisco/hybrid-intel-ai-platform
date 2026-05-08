
# AI upgrade for your intelligence analysis project

This add-on upgrades your existing project with four AI-style capabilities:

1. Report summarization
2. Entity extraction
3. Similarity search across reports
4. AI-style anomaly scoring for triage

## Inputs
- data/entities.csv
- data/reports.csv
- data/edges.csv

## Outputs
- output/reports_ai.csv
- output/report_similarity.json
- output/unified_graph_ai.json

## Run
```bash
pip install -r requirements.txt
python ai_enrichment.py
```

## Why this matters
This turns your demo from a static dashboard into an AI-assisted analyst workflow:
- summarize reports quickly
- surface recurring patterns
- prioritize risky reporting
- push AI enrichment back into the graph

## Strong next upgrade
Later, you can replace the heuristics with:
- OpenAI summarization
- spaCy NER
- embeddings-based similarity
- model-based anomaly scoring
