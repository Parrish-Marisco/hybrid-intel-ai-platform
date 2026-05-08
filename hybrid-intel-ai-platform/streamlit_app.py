"""
Hybrid Intelligence Analysis Platform - Streamlit Cloud Entry Point
-------------------------------------------------------------------
Self-contained dashboard. No external file dependencies; if the JSON
artifacts aren't present in the repo, this falls back to bundled sample
data so the app always renders.

Reads (in order):
  1. ./output/unified_graph_ai.json    (AI-enriched, from pipeline)
  2. ./data/unified_graph.json         (baseline)
  3. SAMPLE_GRAPH below                (in-file fallback)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Optional: streamlit-agraph for force-directed link analysis. Falls back
# to a Plotly polar layout if the package isn't installed.
try:
    from streamlit_agraph import Config, Edge, Node, agraph
    AGRAPH_AVAILABLE = True
except ImportError:
    AGRAPH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Page config + dark operational theme
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Hybrid Intelligence Analysis Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; color: #e6e6e6; }
    div[data-testid="stMetric"] { background-color: #161b22;
                                  padding: 10px; border-radius: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Loader (with full fallback chain to in-file sample)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
PRIMARY_PATH = ROOT / "output" / "unified_graph_ai.json"
FALLBACK_PATH = ROOT / "data" / "unified_graph.json"

SAMPLE_GRAPH = {
    "metadata": {
        "pipeline_version": "in-file-sample",
        "llm_provider": "none",
        "node_count": 4, "edge_count": 3, "report_count": 2,
    },
    "nodes": [
        {"id": "Sample Person A", "type": "Person", "threat_score": 0.82,
         "confidence": "High", "mentions": 5, "last_seen": "2026-03-31",
         "primary_location": "Bamako", "linked_actor": True,
         "status": "Escalating", "anomaly_flag": True,
         "latitude": 12.6392, "longitude": -8.0029},
        {"id": "Sample Group B", "type": "Group", "threat_score": 0.68,
         "confidence": "Medium", "mentions": 3, "last_seen": "2026-03-30",
         "primary_location": "Gao", "linked_actor": True,
         "status": "Watch", "anomaly_flag": False,
         "latitude": 16.2667, "longitude": -0.0500},
        {"id": "Sample Location C", "type": "Location", "threat_score": 0.45,
         "confidence": "Medium", "mentions": 2, "last_seen": "2026-03-28",
         "primary_location": "Niger Border", "linked_actor": False,
         "status": "Monitor", "anomaly_flag": False,
         "latitude": 15.0, "longitude": 2.0},
        {"id": "Sample Person D", "type": "Person", "threat_score": 0.55,
         "confidence": "Medium", "mentions": 2, "last_seen": "2026-03-27",
         "primary_location": "Mopti", "linked_actor": False,
         "status": "Monitor", "anomaly_flag": False,
         "latitude": 14.4843, "longitude": -4.1820},
    ],
    "edges": [
        {"source": "Sample Person A", "target": "Sample Group B",
         "relationship": "linked_to", "weight": 0.9},
        {"source": "Sample Group B", "target": "Sample Location C",
         "relationship": "operates_near", "weight": 0.7},
        {"source": "Sample Person D", "target": "Sample Person A",
         "relationship": "met_with", "weight": 0.6},
    ],
    "reports": [
        {"report_id": "SAMPLE-001", "date": "2026-03-30",
         "source_type": "HUMINT", "confidence": "Medium",
         "location": "Bamako",
         "excerpt": "Sample Person A observed meeting Sample Group B "
                    "operatives near Bamako transit area.",
         "entities": "Sample Person A|Sample Group B|Bamako",
         "ai_summary": "Sample Person A met Group B operatives in Bamako."},
        {"report_id": "SAMPLE-002", "date": "2026-03-28",
         "source_type": "OSINT", "confidence": "Low",
         "location": "Niger Border",
         "excerpt": "Open source reporting noted activity at Sample "
                    "Location C consistent with Group B movement.",
         "entities": "Sample Location C|Sample Group B|Niger Border",
         "ai_summary": "Activity at Location C aligned with Group B."},
    ],
    "ai_artifacts": {"report_similarity_links": []},
}


def load_graph() -> tuple[dict, str]:
    for path, label in [
        (PRIMARY_PATH, f"AI-enriched: {PRIMARY_PATH.name}"),
        (FALLBACK_PATH, f"Baseline: {FALLBACK_PATH.name}"),
    ]:
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f), label
            except Exception as e:
                st.warning(f"Failed to load {label}: {e}")
    return SAMPLE_GRAPH, "In-file sample (no data files found)"


graph, source_label = load_graph()

with st.sidebar:
    st.header("Data Source")
    st.caption(f"Loaded: **{source_label}**")
    uploaded = st.file_uploader("Upload unified_graph JSON", type=["json"])
    if uploaded is not None:
        graph = json.load(uploaded)
        source_label = f"Uploaded: {uploaded.name}"
        st.caption(f"Now using: **{source_label}**")

nodes_df = pd.DataFrame(graph.get("nodes", []))
edges_df = pd.DataFrame(graph.get("edges", []))
reports_df = pd.DataFrame(graph.get("reports", []))


# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Filters")
    if not nodes_df.empty:
        type_options = sorted(nodes_df["type"].dropna().unique().tolist())
        entity_types = st.multiselect(
            "Entity Type", options=type_options, default=type_options
        )
        min_score = st.slider("Minimum Threat Score", 0.0, 1.0, 0.0, 0.05)
        location_options = sorted(
            nodes_df["primary_location"].dropna().unique().tolist()
        )
        selected_locations = st.multiselect(
            "Location", options=location_options
        )
    else:
        entity_types, min_score, selected_locations = [], 0.0, []

    if not reports_df.empty and "source_type" in reports_df.columns:
        source_options = sorted(
            reports_df["source_type"].dropna().unique().tolist()
        )
        selected_sources = st.multiselect(
            "Source Type", options=source_options, default=source_options
        )
    else:
        selected_sources = []


# ---------------------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------------------

filtered_nodes = nodes_df.copy()
if not filtered_nodes.empty:
    if entity_types:
        filtered_nodes = filtered_nodes[
            filtered_nodes["type"].isin(entity_types)
        ]
    filtered_nodes = filtered_nodes[
        filtered_nodes["threat_score"] >= min_score
    ]
    if selected_locations:
        filtered_nodes = filtered_nodes[
            filtered_nodes["primary_location"].isin(selected_locations)
        ]

filtered_reports = reports_df.copy()
if (not filtered_reports.empty
        and selected_sources
        and "source_type" in filtered_reports.columns):
    filtered_reports = filtered_reports[
        filtered_reports["source_type"].isin(selected_sources)
    ]


# ---------------------------------------------------------------------------
# Header + KPIs
# ---------------------------------------------------------------------------

st.title("Hybrid Intelligence Analysis Platform")
st.caption(
    "AI-enriched fusion of HUMINT/OSINT/TECHINT signals "
    "for analyst decision support."
)

c1, c2, c3, c4 = st.columns(4)
if not filtered_nodes.empty:
    c1.metric(
        "High-Risk Entities",
        int((filtered_nodes["threat_score"] >= 0.75).sum()),
    )
    c2.metric("Total Entities", len(filtered_nodes))
    c3.metric("Locations", filtered_nodes["primary_location"].nunique())
    c4.metric(
        "Avg Threat Score",
        round(filtered_nodes["threat_score"].mean(), 2),
    )
else:
    for col in (c1, c2, c3, c4):
        col.metric("--", 0)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    ["Threat Prioritization", "Link Analysis", "Geospatial", "Reports"]
)


# ---- Tab 1: Threat Prioritization -----------------------------------------
with tab1:
    st.subheader("Threat Prioritization")
    if filtered_nodes.empty:
        st.info("No entities match the current filters.")
    else:
        display_cols = [
            "id", "type", "threat_score", "confidence", "mentions",
            "last_seen", "primary_location", "linked_actor", "status",
        ]
        if "anomaly_flag" in filtered_nodes.columns:
            display_cols.append("anomaly_flag")
        display_cols = [c for c in display_cols
                        if c in filtered_nodes.columns]
        sorted_df = filtered_nodes[display_cols].sort_values(
            "threat_score", ascending=False
        )
        st.dataframe(
            sorted_df.rename(columns={"id": "Entity"}),
            use_container_width=True,
            height=420,
        )

        st.subheader("Threat Score Distribution")
        fig = px.histogram(
            filtered_nodes, x="threat_score", nbins=20,
            color="type", template="plotly_dark",
        )
        fig.update_layout(bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)


# ---- Tab 2: Link Analysis (force-directed) --------------------------------
with tab2:
    st.subheader("Network / Link Analysis")

    if edges_df.empty or filtered_nodes.empty:
        st.info("No relationship data available.")
    else:
        valid_names = set(filtered_nodes["id"])
        sub_edges = edges_df[
            edges_df["source"].isin(valid_names)
            & edges_df["target"].isin(valid_names)
        ]

        def _node_color(score: float) -> str:
            if score >= 0.75:
                return "#ef4444"
            if score >= 0.50:
                return "#f59e0b"
            return "#3b82f6"

        if AGRAPH_AVAILABLE:
            type_shape = {
                "Person": "dot",
                "Group": "diamond",
                "Location": "square",
                "Organization": "triangle",
            }
            node_lookup = filtered_nodes.set_index("id").to_dict("index")

            ag_nodes = []
            for name, info in node_lookup.items():
                score = float(info.get("threat_score", 0))
                ag_nodes.append(Node(
                    id=name,
                    label=name,
                    size=15 + 3 * int(info.get("mentions", 1)),
                    color=_node_color(score),
                    shape=type_shape.get(info.get("type"), "dot"),
                    title=(
                        f"{name}\n"
                        f"Type: {info.get('type')}\n"
                        f"Threat: {score:.2f}\n"
                        f"Status: {info.get('status')}\n"
                        f"Location: {info.get('primary_location')}"
                    ),
                ))

            ag_edges = []
            for _, e in sub_edges.iterrows():
                ag_edges.append(Edge(
                    source=e["source"],
                    target=e["target"],
                    label=e["relationship"],
                    color="#6b7280",
                    width=1 + 3 * float(e.get("weight", 0.5)),
                ))

            cfg = Config(
                width=1100, height=600, directed=True, physics=True,
                hierarchical=False, nodeHighlightBehavior=True,
                highlightColor="#fbbf24", collapsible=False,
            )

            agraph(nodes=ag_nodes, edges=ag_edges, config=cfg)

            with st.expander("Graph legend"):
                st.markdown(
                    "- **Red** = severe (threat ≥ 0.75)\n"
                    "- **Amber** = elevated (0.50–0.75)\n"
                    "- **Blue** = moderate (< 0.50)\n"
                    "- Node size scales with mention count\n"
                    "- Edge width scales with relationship weight"
                )
        else:
            st.warning(
                "`streamlit-agraph` not installed - showing fallback "
                "layout. Add to requirements.txt for force-directed view."
            )
            node_index = {name: i for i, name in
                          enumerate(filtered_nodes["id"].tolist())}
            n = len(node_index)
            positions = {
                name: (math.cos(2 * math.pi * i / max(n, 1)),
                       math.sin(2 * math.pi * i / max(n, 1)))
                for name, i in node_index.items()
            }
            edge_x, edge_y = [], []
            for _, e in sub_edges.iterrows():
                if e["source"] in positions and e["target"] in positions:
                    x0, y0 = positions[e["source"]]
                    x1, y1 = positions[e["target"]]
                    edge_x += [x0, x1, None]
                    edge_y += [y0, y1, None]
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y, mode="lines",
                line=dict(width=1, color="#444"), hoverinfo="none",
            )
            node_trace = go.Scatter(
                x=[positions[n][0] for n in node_index],
                y=[positions[n][1] for n in node_index],
                mode="markers+text",
                text=list(node_index.keys()),
                textposition="top center",
                textfont=dict(color="#e6e6e6", size=10),
                marker=dict(
                    size=[10 + 3 * int(filtered_nodes.set_index("id")
                          .loc[n, "mentions"]) for n in node_index],
                    color=[float(filtered_nodes.set_index("id")
                           .loc[n, "threat_score"]) for n in node_index],
                    colorscale="Reds", cmin=0, cmax=1, showscale=True,
                    colorbar=dict(title="Threat Score"),
                    line=dict(color="#222", width=1),
                ),
                hoverinfo="text",
            )
            fig = go.Figure(data=[edge_trace, node_trace])
            fig.update_layout(
                template="plotly_dark", showlegend=False,
                xaxis=dict(visible=False), yaxis=dict(visible=False),
                height=600,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.caption(f"Showing {len(sub_edges)} edges between "
                   f"{len(filtered_nodes)} entities.")


# ---- Tab 3: Geospatial ----------------------------------------------------
with tab3:
    st.subheader("Geospatial Activity Map")
    if "latitude" in filtered_nodes.columns:
        geo_df = filtered_nodes.dropna(subset=["latitude", "longitude"])
    else:
        geo_df = pd.DataFrame()
    if geo_df.empty:
        st.info("No geocoded entities available for mapping.")
    else:
        fig = px.scatter_mapbox(
            geo_df, lat="latitude", lon="longitude",
            size="mentions", color="threat_score",
            color_continuous_scale="Reds",
            hover_name="id",
            hover_data={"type": True, "status": True,
                        "primary_location": True},
            zoom=4, height=550,
            template="plotly_dark",
        )
        fig.update_layout(mapbox_style="carto-darkmatter",
                          margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)


# ---- Tab 4: Reports -------------------------------------------------------
with tab4:
    st.subheader("Report Explorer")
    if filtered_reports.empty:
        st.info("No reports match the current filters.")
    else:
        sort_col = "date" if "date" in filtered_reports.columns \
            else filtered_reports.columns[0]
        for _, r in filtered_reports.sort_values(
                sort_col, ascending=False).iterrows():
            with st.expander(
                f"[{r.get('report_id', '')}] {r.get('source_type', '')} - "
                f"{r.get('date', '')} - {r.get('location', '')}"
            ):
                st.write(f"**Confidence:** {r.get('confidence', 'N/A')}")
                if "ai_summary" in r and pd.notna(r.get("ai_summary")):
                    st.write(f"**AI Summary:** {r['ai_summary']}")
                st.write(f"**Excerpt:** {r.get('excerpt', '')}")
                if "ai_keywords" in r and pd.notna(r.get("ai_keywords")):
                    st.write(f"**Keywords:** {r['ai_keywords']}")
                if ("ai_extracted_entities" in r
                        and pd.notna(r.get("ai_extracted_entities"))):
                    st.write(
                        f"**LLM-extracted entities:** "
                        f"{r['ai_extracted_entities']}"
                    )
                st.write(f"**Tagged entities:** {r.get('entities', '')}")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
md = graph.get("metadata", {})
st.caption(
    f"Pipeline: {md.get('pipeline_version', 'n/a')} | "
    f"Provider: {md.get('llm_provider', 'n/a')} | "
    f"Generated: {md.get('generated_at', 'n/a')} | "
    f"Nodes: {md.get('node_count', 0)} | "
    f"Edges: {md.get('edge_count', 0)} | "
    f"Reports: {md.get('report_count', 0)} | "
    f"Source: {source_label}"
)
