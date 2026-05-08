# Hybrid Intelligence Analysis Platform

AI-enriched intelligence analysis platform with graph generation, anomaly scoring, and a Streamlit decision-support dashboard. Pluggable LLM backend (Claude / OpenAI / heuristic) and force-directed link analysis.

> **Modeling Decision-Making Under Uncertainty Using Hybrid Intelligence Signals**

## Problem

Analysts working across HUMINT, OSINT, and TECHINT face fragmented signal streams that rarely arrive in a form ready for prioritization. This platform fuses those signals into a single graph, applies LLM-driven enrichment to surface high-risk entities, and presents both prioritization and evidence traceability through an analyst dashboard.

## Architecture

```
  ┌──────────────┐     ┌────────────────────┐     ┌──────────────────────┐
  │  CSV Inputs  │ ──▶ │  AI Enrichment     │ ──▶ │ unified_graph_ai.json│
  │  (data/)     │     │  Pipeline          │     │ (output/)            │
  └──────────────┘     │   ├─ LLM provider  │     └──────────┬───────────┘
                       │   │  (Claude /      │                │
                       │   │   OpenAI /      │                ▼
                       │   │   heuristic)    │     ┌────────────────────┐
                       │   └─ deterministic  │     │ Streamlit Dashboard│
                       │      scoring        │     │ - Threat priority  │
                       └────────────────────┘     │ - Link analysis    │
                                                   │   (force-directed) │
                                                   │ - Geospatial map   │
                                                   │ - Report explorer  │
                                                   └────────────────────┘

  ┌──────────────┐     ┌────────────────────┐     ┌──────────────────────┐
  │  S3: input/  │ ──▶ │  AWS Lambda        │ ──▶ │ S3: output/          │
  │  *.csv       │     │  lambda_function   │     │ unified_graph.json   │
  └──────────────┘     └────────────────────┘     └──────────────────────┘
```

## Repository Layout

```
hybrid-intel-ai-platform/
├─ app/
│  └─ hybrid_intel_dashboard_from_json.py
├─ data/
│  ├─ entities.csv
│  ├─ reports.csv
│  ├─ edges.csv
│  └─ unified_graph.json        # baseline fallback
├─ output/
│  └─ unified_graph_ai.json     # AI-enriched (committed for demo)
├─ lambda/
│  └─ lambda_function.py
├─ ai_enrichment_pipeline.py
├─ llm_providers.py             # Claude / OpenAI / heuristic dispatch
├─ requirements.txt
├─ .env.example
├─ README.md
└─ .gitignore
```

## Run Locally

1. **Install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **(Optional) Configure an LLM provider** — copy `.env.example` and export keys:
   ```bash
   cp .env.example .env
   # Edit .env and set ANTHROPIC_API_KEY (or OPENAI_API_KEY)
   export $(grep -v '^#' .env | xargs)
   ```
   Without a key the pipeline runs in heuristic mode — still produces all artifacts, just with deterministic summaries instead of LLM ones.

3. **Generate the AI-enriched graph**
   ```bash
   python ai_enrichment_pipeline.py
   ```
   First line of output prints the active provider, e.g. `[0/5] Active LLM provider: anthropic`. Writes `output/reports_enriched.csv`, `output/entities_enriched.csv`, and `output/unified_graph_ai.json`.

4. **Launch the dashboard**
   ```bash
   streamlit run app/hybrid_intel_dashboard_from_json.py
   ```
   The dashboard reads `output/unified_graph_ai.json` first, falls back to `data/unified_graph.json`, then to bundled samples.

## LLM Provider Configuration

The pipeline routes summarization and entity extraction through a pluggable provider (`llm_providers.py`):

| Provider | Activation | Models |
|---|---|---|
| **Anthropic Claude** | `ANTHROPIC_API_KEY` set, or `LLM_PROVIDER=anthropic` | `ANTHROPIC_MODEL` (default `claude-haiku-4-5`) |
| **OpenAI** | `OPENAI_API_KEY` set, or `LLM_PROVIDER=openai` | `OPENAI_MODEL` (default `gpt-4o-mini`) |
| **Heuristic** | Default fallback; or `LLM_PROVIDER=heuristic` | None — first-sentence summaries, capitalized-token extraction |

Selection rules:
- Explicit `LLM_PROVIDER` env var wins.
- Otherwise auto-detect: Anthropic key first, then OpenAI key, then heuristic.
- If the chosen SDK or key is missing at call time, the pipeline emits a one-time warning and falls back to heuristic for that call. The pipeline never crashes on a missing key.

The `metadata.llm_provider` field in `unified_graph_ai.json` records which provider produced the artifact.

## AI Enrichment Steps

| Step | Routes through provider? | Notes |
|---|---|---|
| Per-report summarization | Yes | Single-sentence neutral summaries |
| Per-report entity extraction | Yes | Returns `{text, type}` pairs |
| Keyword extraction | No | Deterministic top-k tokens |
| TF cosine similarity between reports | No | Cheap, stable, no API cost |
| Threat-score adjustment (recurrence + centrality) | No | Pure math, reproducible |
| Anomaly flag (z-score) | No | Pure math, reproducible |

The mix is intentional: LLMs for the steps where language understanding matters, deterministic math for the steps where reproducibility and zero cost matter.

## Link Analysis View

Tab 2 uses [`streamlit-agraph`](https://github.com/ChrisChross/streamlit-agraph) for an interactive force-directed graph: physics simulation, draggable nodes, hover tooltips, edge labels. Visual encoding:

- **Color** — threat band: red (≥ 0.75), amber (0.50–0.75), blue (< 0.50)
- **Size** — mention count
- **Edge width** — relationship weight
- **Shape** — entity type (person / group / location / organization)

If `streamlit-agraph` is not installed, the tab degrades gracefully to a Plotly polar layout.

## AWS Pipeline

`lambda/lambda_function.py` is the same logic as a Lambda handler:

- Trigger on S3 PUT events under `input/`
- Reads `input/entities.csv`, `input/reports.csv`, `input/edges.csv`
- Writes `output/unified_graph.json`

Test event:
```json
{
  "bucket": "YOUR_BUCKET_NAME",
  "entities_key": "input/entities.csv",
  "reports_key": "input/reports.csv",
  "edges_key": "input/edges.csv",
  "output_key": "output/unified_graph.json"
}
```

Required IAM: `s3:GetObject` on input prefix, `s3:PutObject` on output prefix.

## Tradeoffs

- **LLM vs heuristic by default**: ships running heuristic-only so a clone-and-run reviewer gets a working demo without keys. Setting one env var promotes it to a real LLM-driven system.
- **Selective LLM usage**: only the language-understanding steps (summary, NER) call the model. Similarity, scoring, and anomaly detection stay deterministic to keep cost predictable and results reproducible.
- **Static graph artifact**: the dashboard reads a JSON snapshot. For live use, schedule the Lambda on S3 events.

## Why This Matters

The fusion problem is a real-world bottleneck: analysts drown in fragmented reporting and rarely have time to weigh recurrence, association, and confidence systematically. A graph-first platform with explicit scoring rationale and pluggable LLM enrichment lets analysts move from "what's in the inbox" to "what should I look at next, and why."
