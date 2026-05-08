"""
Hybrid Intelligence Analysis Platform - Streamlit Dashboard
-----------------------------------------------------------
Reads `output/unified_graph_ai.json` (AI-enriched), falling back to
`data/unified_graph.json`, then to bundled sample data.

Link analysis uses streamlit-agraph for a force-directed Palantir-style
view; if streamlit-agraph isn't installed, the view degrades gracefully
to a Plotly polar layout.

Run:
    streamlit run app/hybrid_intel_dashboard_from_json.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Optional dependency: streamlit-agraph for force-directed link analysis.
try:
    from streamlit_agraph import agraph, Config, Edge, Node
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
# Loader (with fallback chain)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRIMARY_PATH = PROJECT_ROOT / "output" / "unified_graph_ai.json"
FALLBACK_PATH = PROJECT_ROOT / "data" / "unified_graph.json"

SAMPLE_GRAPH = {
    "metadata": {"node_count": 0, "edge_count": 0, "report_count": 0},
    "nodes": [], "edges": [], "reports": [], "ai_artifacts": {},
}


def load_graph() -> tuple[dict, str]:
    for path, label in [
        (PRIMARY_PATH, f"AI-enriched: {PRIMARY_PATH.name}"),
        (FALLBACK_PATH, f"Baseline: {FALLBACK_PATH.name}"),
    ]:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f), label
    return SAMPLE_GRAPH, "Empty sample (no data file found)"


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

    if not reports_df.empty:
        source_options = sorted(reports_df["source_type"].dropna().unique().tolist())
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
        filtered_nodes = filtered_nodes[filtered_nodes["type"].isin(entity_types)]
    filtered_nodes = filtered_nodes[filtered_nodes["threat_score"] >= min_score]
    if selected_locations:
        filtered_nodes = filtered_nodes[
            filtered_nodes["primary_location"].isin(selected_locations)
        ]

filtered_reports = reports_df.copy()
if not filtered_reports.empty and selected_sources:
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

        # Color band by threat score for node coloring
        def _node_color(score: float) -> str:
            if score >= 0.75:
                return "#ef4444"  # severe
            if score >= 0.50:
                return "#f59e0b"  # elevated
            return "#3b82f6"      # moderate

        if AGRAPH_AVAILABLE:
            # Force-directed Palantir-style graph via streamlit-agraph
            type_shape = {
                "Person": "circularImage",
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
                    shape="dot" if info.get("type") == "Person"
                          else type_shape.get(info.get("type"), "dot"),
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
                width=1100,
                height=600,
                directed=True,
                physics=True,
                hierarchical=False,
                nodeHighlightBehavior=True,
                highlightColor="#fbbf24",
                collapsible=False,
                node={"labelProperty": "label", "renderLabel": True},
                link={"renderLabel": True, "labelProperty": "label",
                      "fontColor": "#9ca3af"},
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
            # Plotly polar fallback (no streamlit-agraph installed)
            st.warning(
                "`streamlit-agraph` not installed — showing fallback layout. "
                "Run `pip install streamlit-agraph` for the force-directed view."
            )
            import math as _m
            node_index = {name: i for i, name in
                          enumerate(filtered_nodes["id"].tolist())}
            n = len(node_index)
            positions = {
                name: (_m.cos(2 * _m.pi * i / max(n, 1)),
                       _m.sin(2 * _m.pi * i / max(n, 1)))
                for name, i in node_index.items()
            }
            edge_x, edge_y = [], []
            for _, e in sub_edges.iterrows():
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
                    size=[10 + 3 * filtered_nodes.set_index("id")
                          .loc[n, "mentions"] for n in node_index],
                    color=[filtered_nodes.set_index("id")
                           .loc[n, "threat_score"] for n in node_index],
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
    geo_df = filtered_nodes.dropna(subset=["latitude", "longitude"]) \
        if "latitude" in filtered_nodes.columns else pd.DataFrame()
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
        for _, r in filtered_reports.sort_values(
                "date", ascending=False).iterrows():
            with st.expander(
                f"[{r['report_id']}] {r.get('source_type', '')} - "
                f"{r.get('date', '')} - {r.get('location', '')}"
            ):
                st.write(f"**Confidence:** {r.get('confidence', 'N/A')}")
                if "ai_summary" in r and pd.notna(r["ai_summary"]):
                    st.write(f"**AI Summary:** {r['ai_summary']}")
                st.write(f"**Excerpt:** {r.get('excerpt', '')}")
                if "ai_keywords" in r and pd.notna(r["ai_keywords"]):
                    st.write(f"**Keywords:** {r['ai_keywords']}")
                if ("ai_extracted_entities" in r
                        and pd.notna(r["ai_extracted_entities"])):
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
    f"Reports: {md.get('report_count', 0)}"
)
