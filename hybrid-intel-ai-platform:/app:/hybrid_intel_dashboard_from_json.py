import json
from pathlib import Path
import math

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Hybrid Intelligence Analysis Platform", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 1.2rem;}
.stMetric {background:#111827; border:1px solid #1f2937; padding:.75rem; border-radius:.75rem;}
div[data-testid="stDataFrame"] {border:1px solid #1f2937; border-radius:.75rem; overflow:hidden;}
.intel-card {background:#111827; border:1px solid #1f2937; border-radius:.9rem; padding:1rem; margin-bottom:.75rem;}
.small-muted {color:#9ca3af; font-size:.9rem;}
</style>
""", unsafe_allow_html=True)

SAMPLE_GRAPH = {
    "metadata": {"generated_by": "local_sample", "node_count": 4, "edge_count": 3, "report_count": 3},
    "nodes": [
        {"id": "Amadou Diallo", "type": "Person", "threat_score": 0.83, "confidence": "High", "mentions": 8, "last_seen": "2026-03-31", "primary_location": "Bamako", "status": "Escalating"},
        {"id": "Sahel Transit Network", "type": "Group", "threat_score": 0.79, "confidence": "High", "mentions": 7, "last_seen": "2026-03-31", "primary_location": "Gao", "status": "Escalating"},
        {"id": "Warehouse 17", "type": "Location", "threat_score": 0.67, "confidence": "Medium", "mentions": 5, "last_seen": "2026-03-29", "primary_location": "Bamako", "status": "Watch"},
        {"id": "Niger Corridor Node", "type": "Location", "threat_score": 0.76, "confidence": "High", "mentions": 6, "last_seen": "2026-03-31", "primary_location": "Niger Border", "status": "Escalating"}
    ],
    "edges": [
        {"source": "Amadou Diallo", "target": "Sahel Transit Network", "relationship": "coordinates_with", "weight": 0.92},
        {"source": "Amadou Diallo", "target": "Warehouse 17", "relationship": "uses_site", "weight": 0.86},
        {"source": "Sahel Transit Network", "target": "Niger Corridor Node", "relationship": "uses_route", "weight": 0.90}
    ],
    "reports": [
        {"report_id": "HUMINT-116", "date": "2026-03-31", "source_type": "HUMINT", "confidence": "High", "location": "Niger Border", "excerpt": "Source reported a short-duration leadership meeting focused on corridor timing and border movement.", "entities": ["Amadou Diallo", "Sahel Transit Network", "Niger Corridor Node"], "relevance": "High"},
        {"report_id": "TECHINT-114", "date": "2026-03-30", "source_type": "TECHINT", "confidence": "High", "location": "Bamako", "excerpt": "Cross-device correlations linked warehouse activity and a principal coordinator.", "entities": ["Amadou Diallo", "Warehouse 17"], "relevance": "High"},
        {"report_id": "HUMINT-113", "date": "2026-03-29", "source_type": "HUMINT", "confidence": "High", "location": "Gao", "excerpt": "A source placed a coordinator in a meeting with members of a transit network.", "entities": ["Sahel Transit Network"], "relevance": "High"}
    ]
}

LOCATION_COORDS = {
    "Bamako": {"lat": 12.6392, "lon": -8.0029},
    "Gao": {"lat": 16.2667, "lon": -0.0500},
    "Mopti": {"lat": 14.4843, "lon": -4.1829},
    "Niger Border": {"lat": 15.0500, "lon": 1.1000},
    "Warehouse 17": {"lat": 12.6300, "lon": -8.0150},
    "Niger Corridor Node": {"lat": 15.0500, "lon": 1.1000},
}

PRIMARY_GRAPH_PATH = Path("output/unified_graph_ai.json")
FALLBACK_GRAPH_PATH = Path("data/unified_graph.json")

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_graph():
    if PRIMARY_GRAPH_PATH.exists():
        return load_json(PRIMARY_GRAPH_PATH)
    if FALLBACK_GRAPH_PATH.exists():
        return load_json(FALLBACK_GRAPH_PATH)
    return SAMPLE_GRAPH

def graph_to_frames(graph):
    nodes = pd.DataFrame(graph.get("nodes", []))
    edges = pd.DataFrame(graph.get("edges", []))
    reports = pd.DataFrame(graph.get("reports", []))

    if not nodes.empty and "last_seen" in nodes.columns:
        nodes["last_seen"] = pd.to_datetime(nodes["last_seen"], errors="coerce")
    if not reports.empty and "date" in reports.columns:
        reports["date"] = pd.to_datetime(reports["date"], errors="coerce")
    if not nodes.empty and "mentions" in nodes.columns:
        nodes["mentions"] = pd.to_numeric(nodes["mentions"], errors="coerce").fillna(0).astype(int)
    if not nodes.empty and "threat_score" in nodes.columns:
        nodes["threat_score"] = pd.to_numeric(nodes["threat_score"], errors="coerce").fillna(0.0)

    return nodes, edges, reports

def score_band(score):
    if score >= 0.75:
        return "Severe"
    if score >= 0.50:
        return "Elevated"
    return "Moderate"

def enrich_map_data(nodes_df):
    if nodes_df.empty:
        return nodes_df
    df = nodes_df.copy()
    lat_vals, lon_vals = [], []
    for _, row in df.iterrows():
        key = row.get("primary_location") or row.get("id")
        coords = LOCATION_COORDS.get(key, {})
        lat_vals.append(coords.get("lat"))
        lon_vals.append(coords.get("lon"))
    df["lat"] = lat_vals
    df["lon"] = lon_vals
    return df.dropna(subset=["lat", "lon"])

def build_network_figure(edges_df, nodes_df):
    if edges_df.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", title="No network edges available", height=560)
        return fig

    unique_nodes = pd.unique(edges_df[["source", "target"]].values.ravel("K")).tolist()
    positions = {
        node: (
            math.cos(i * 2 * math.pi / max(len(unique_nodes), 1)),
            math.sin(i * 2 * math.pi / max(len(unique_nodes), 1)),
        )
        for i, node in enumerate(unique_nodes)
    }

    traces = []
    for _, row in edges_df.iterrows():
        x0, y0 = positions[row["source"]]
        x1, y1 = positions[row["target"]]
        traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=max(1.0, float(row.get("weight", 1)) * 4), color="#64748b"),
                hoverinfo="text",
                text=[f'{row["source"]} → {row["target"]}<br>{row.get("relationship", "related_to")}'] * 3,
                showlegend=False,
            )
        )

    x_nodes, y_nodes, labels, sizes, colors = [], [], [], [], []
    for node in unique_nodes:
        x, y = positions[node]
        x_nodes.append(x)
        y_nodes.append(y)
        labels.append(node)
        match = nodes_df[nodes_df["id"] == node] if not nodes_df.empty and "id" in nodes_df.columns else pd.DataFrame()
        if not match.empty:
            score = float(match.iloc[0].get("threat_score", 0))
            sizes.append(18 + score * 22)
            colors.append("#ef4444" if score >= 0.75 else "#f59e0b" if score >= 0.50 else "#10b981")
        else:
            sizes.append(16)
            colors.append("#60a5fa")

    traces.append(
        go.Scatter(
            x=x_nodes,
            y=y_nodes,
            mode="markers+text",
            text=labels,
            textposition="top center",
            marker=dict(size=sizes, color=colors, line=dict(width=1, color="#e5e7eb")),
            hoverinfo="text",
            hovertext=[f"Node: {x}" for x in labels],
            showlegend=False,
        )
    )

    fig = go.Figure(data=traces)
    fig.update_layout(
        template="plotly_dark",
        title="Link Analysis",
        height=580,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig

graph = load_graph()
nodes_df, edges_df, reports_df = graph_to_frames(graph)

all_types = sorted(nodes_df["type"].dropna().unique().tolist()) if not nodes_df.empty and "type" in nodes_df.columns else []
all_conf = sorted(nodes_df["confidence"].dropna().unique().tolist()) if not nodes_df.empty and "confidence" in nodes_df.columns else []
all_locations = sorted(nodes_df["primary_location"].dropna().unique().tolist()) if not nodes_df.empty and "primary_location" in nodes_df.columns else []
all_sources = sorted(reports_df["source_type"].dropna().unique().tolist()) if not reports_df.empty and "source_type" in reports_df.columns else []

date_min = nodes_df["last_seen"].dropna().min().date() if not nodes_df.empty and nodes_df["last_seen"].notna().any() else pd.Timestamp("2026-03-01").date()
date_max = nodes_df["last_seen"].dropna().max().date() if not nodes_df.empty and nodes_df["last_seen"].notna().any() else pd.Timestamp("2026-03-31").date()

st.sidebar.title("Controls")
selected_range = st.sidebar.date_input("Date range", value=(date_min, date_max), min_value=date_min, max_value=date_max)
if isinstance(selected_range, tuple) and len(selected_range) == 2:
    start_date, end_date = selected_range
else:
    start_date, end_date = date_min, date_max

selected_types = st.sidebar.multiselect("Entity type", all_types, default=all_types)
selected_conf = st.sidebar.multiselect("Confidence", all_conf, default=all_conf)
selected_locs = st.sidebar.multiselect("Location", all_locations, default=all_locations)
selected_sources = st.sidebar.multiselect("Report source type", all_sources, default=all_sources)
min_score = st.sidebar.slider("Threat score threshold", 0.0, 1.0, 0.50, 0.01)

filtered_nodes = nodes_df.copy()
if not filtered_nodes.empty:
    if selected_types:
        filtered_nodes = filtered_nodes[filtered_nodes["type"].isin(selected_types)]
    if selected_conf:
        filtered_nodes = filtered_nodes[filtered_nodes["confidence"].isin(selected_conf)]
    if selected_locs:
        filtered_nodes = filtered_nodes[filtered_nodes["primary_location"].isin(selected_locs)]
    filtered_nodes = filtered_nodes[filtered_nodes["threat_score"] >= min_score]
    filtered_nodes = filtered_nodes[
        (filtered_nodes["last_seen"].dt.date >= start_date) &
        (filtered_nodes["last_seen"].dt.date <= end_date)
    ]

node_ids = set(filtered_nodes["id"].tolist()) if not filtered_nodes.empty else set()
filtered_edges = edges_df[
    (edges_df["source"].isin(node_ids)) | (edges_df["target"].isin(node_ids))
].copy() if not edges_df.empty else edges_df.copy()

filtered_reports = reports_df.copy()
if not filtered_reports.empty:
    if selected_sources:
        filtered_reports = filtered_reports[filtered_reports["source_type"].isin(selected_sources)]
    filtered_reports = filtered_reports[
        (filtered_reports["date"].dt.date >= start_date) &
        (filtered_reports["date"].dt.date <= end_date)
    ]

st.title("Hybrid Intelligence Analysis Platform")
st.caption("Dashboard driven by AI-enriched graph output generated by the project pipeline.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Nodes", int(len(filtered_nodes)))
m2.metric("Edges", int(len(filtered_edges)))
m3.metric("Reports", int(len(filtered_reports)))
m4.metric("High-Risk Nodes", int((filtered_nodes["threat_score"] >= 0.75).sum()) if not filtered_nodes.empty else 0)

left, right = st.columns([1.5, 1.0], gap="large")

with left:
    st.subheader("Threat Prioritization")
    if not filtered_nodes.empty:
        table = filtered_nodes.copy()
        table["Band"] = table["threat_score"].apply(score_band)
        table = table.rename(columns={
            "id": "Entity",
            "type": "Type",
            "threat_score": "Threat Score",
            "confidence": "Confidence",
            "mentions": "Mentions",
            "last_seen": "Last Seen",
            "primary_location": "Location",
            "status": "Status",
        })
        st.dataframe(
            table[["Entity", "Type", "Threat Score", "Band", "Confidence", "Mentions", "Last Seen", "Location", "Status"]].sort_values("Threat Score", ascending=False),
            use_container_width=True,
            height=320,
        )
    else:
        st.info("No nodes match the current filters.")

    st.subheader("Threat Score Distribution")
    if not filtered_nodes.empty:
        fig = px.histogram(filtered_nodes, x="threat_score", nbins=12, template="plotly_dark")
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No threat score data available.")

with right:
    st.subheader("Geographic Activity")
    map_df = enrich_map_data(filtered_nodes)
    if not map_df.empty:
        st.map(map_df.rename(columns={"lat": "LAT", "lon": "LON"}), latitude="LAT", longitude="LON", size="threat_score", zoom=4)
    else:
        st.info("No mapped locations available.")

    st.subheader("Graph Metadata")
    st.json(graph.get("metadata", {}))

st.divider()

col1, col2 = st.columns([1.2, 0.8], gap="large")

with col1:
    st.subheader("Network / Link Analysis")
    st.plotly_chart(build_network_figure(filtered_edges, filtered_nodes), use_container_width=True)

with col2:
    st.subheader("Node Detail")
    options = filtered_nodes.sort_values("threat_score", ascending=False)["id"].tolist() if not filtered_nodes.empty else []
    selected = st.selectbox("Select node", options if options else ["No nodes available"])

    if options and selected in set(filtered_nodes["id"]):
        row = filtered_nodes[filtered_nodes["id"] == selected].iloc[0]
        linked_edges = filtered_edges[(filtered_edges["source"] == selected) | (filtered_edges["target"] == selected)]
        related_reports = filtered_reports[
            filtered_reports["entities"].apply(lambda x: selected in x if isinstance(x, list) else False)
        ] if not filtered_reports.empty else pd.DataFrame()

        with st.container(border=True):
            st.markdown(f"### {row['id']}")
            st.markdown(f"**Type:** {row.get('type')}")
            st.markdown(f"**Threat Score:** {row.get('threat_score', 0):.2f}")
            st.markdown(f"**Confidence:** {row.get('confidence')}")
            st.markdown(f"**Mentions:** {int(row.get('mentions', 0))}")
            st.markdown(f"**Last Seen:** {row.get('last_seen').date() if pd.notna(row.get('last_seen')) else 'N/A'}")
            st.markdown(f"**Location:** {row.get('primary_location')}")
            st.markdown(f"**Status:** {row.get('status')}")

        st.markdown("**Connections**")
        if not linked_edges.empty:
            st.dataframe(linked_edges, use_container_width=True, height=180)
        else:
            st.info("No connections for this node under current filters.")

        st.markdown("**Supporting Reports**")
        if not related_reports.empty:
            st.dataframe(
                related_reports[["report_id", "date", "source_type", "confidence", "location", "relevance"]],
                use_container_width=True,
                height=180,
            )
        else:
            st.info("No supporting reports for this node under current filters.")

st.divider()

st.subheader("Report Explorer")
if filtered_reports.empty:
    st.info("No reports available for current filters.")
else:
    search = st.text_input("Search reports")
    display_reports = filtered_reports.copy()
    if search:
        display_reports = display_reports[
            display_reports["excerpt"].fillna("").str.contains(search, case=False, regex=False) |
            display_reports["location"].fillna("").str.contains(search, case=False, regex=False) |
            display_reports["entities"].apply(lambda x: any(search.lower() in str(item).lower() for item in x) if isinstance(x, list) else False)
        ]
    display_reports = display_reports.sort_values("date", ascending=False)
    for _, row in display_reports.head(8).iterrows():
        st.markdown('<div class="intel-card">', unsafe_allow_html=True)
        st.markdown(f"**Report:** {row['report_id']}")
        st.markdown(f"<span class='small-muted'>Date: {row['date'].date()} | Source: {row['source_type']} | Confidence: {row['confidence']} | Location: {row['location']}</span>", unsafe_allow_html=True)
        st.markdown(f"**Relevance:** {row.get('relevance')}")
        st.markdown(f"**Excerpt:** {row.get('excerpt')}")
        st.markdown(f"**Entities:** {', '.join(row.get('entities', []))}")
        st.markdown("</div>", unsafe_allow_html=True)

st.divider()
st.caption("The app prefers output/unified_graph_ai.json, then falls back to data/unified_graph.json, then bundled sample data.")
