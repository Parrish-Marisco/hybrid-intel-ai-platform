
import math
from pathlib import Path
from io import StringIO
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

SAMPLE_ENTITIES = """entity_id,name,type,threat_score,confidence,mentions,last_seen,primary_location,linked_actor,status,lat,lon,first_seen
1,Ibrahim Traore,Person,0.82,High,7,2026-03-31,Bamako,True,Escalating,12.6392,-8.0029,2026-03-05
2,Unknown Logistics Cell,Group,0.76,Medium,5,2026-03-30,Gao,True,Watch,16.2667,-0.0500,2026-03-10
3,Route Node K-12,Location,0.64,Medium,4,2026-03-29,Niger Border,False,Monitor,15.0000,1.0000,2026-03-07
4,Facilitator A,Person,0.71,Medium,6,2026-03-28,Mopti,True,Watch,14.4843,-4.1829,2026-03-01
5,Transit Corridor North,Location,0.58,Low,3,2026-03-27,Gao,False,Monitor,16.3500,-0.2000,2026-03-12
6,Unknown Financier,Person,0.79,High,5,2026-03-31,Bamako,True,Escalating,12.6500,-8.0100,2026-03-09
7,Support Network Red,Group,0.68,Medium,4,2026-03-26,Mopti,True,Watch,14.4900,-4.1900,2026-03-11
8,Warehouse 17,Location,0.61,Medium,4,2026-03-30,Bamako,False,Monitor,12.6300,-8.0150,2026-03-14
9,Courier B,Person,0.55,Low,2,2026-03-24,Gao,False,Monitor,16.2700,-0.0600,2026-03-19
10,Border Transfer Team,Group,0.74,High,6,2026-03-31,Niger Border,True,Escalating,15.0500,1.1000,2026-03-08
"""
SAMPLE_REPORTS = """report_id,date,source_type,confidence,location,excerpt,entities,relevance
HUMINT-001,2026-03-30,HUMINT,Medium,Bamako,"Subject met with unidentified facilitators to discuss movement of supplies near the Niger corridor.","Ibrahim Traore|Unknown Logistics Cell|Bamako",High
OSINT-004,2026-03-31,OSINT,Medium,Gao,"Regional reporting noted movement near a transit corridor linked to a previously flagged logistics network.","Unknown Logistics Cell|Gao|Transit Corridor North",Medium
TECHINT-007,2026-03-29,TECHINT,High,Mopti,"Device metadata suggested repeated coordination between a local facilitator and an external support node.","Facilitator A|Support Network Red|Mopti",High
HUMINT-009,2026-03-28,HUMINT,High,Bamako,"Source reported financial facilitation tied to warehouse activity and unusual late-night transport.","Unknown Financier|Warehouse 17|Bamako",High
OSINT-011,2026-03-27,OSINT,Low,Niger Border,"Open reporting referenced increased movement along an informal crossing point.","Route Node K-12|Border Transfer Team|Niger Border",Medium
TECHINT-013,2026-03-31,TECHINT,High,Niger Border,"Communications pattern analysis indicated synchronized cross-border movements involving a small transport team.","Border Transfer Team|Route Node K-12|Niger Border",High
HUMINT-014,2026-03-24,HUMINT,Low,Gao,"Single-source reporting placed a courier near a staging point with unknown associates.","Courier B|Unknown Logistics Cell|Gao",Low
OSINT-016,2026-03-26,OSINT,Medium,Mopti,"Local reporting referenced warehouse-to-corridor handoffs connected to a red support network.","Support Network Red|Transit Corridor North|Mopti",Medium
"""
SAMPLE_EDGES = """source,target,relationship,weight
Ibrahim Traore,Unknown Logistics Cell,linked_to,0.90
Unknown Logistics Cell,Gao,operates_near,0.70
Ibrahim Traore,Bamako,reported_in,0.80
Facilitator A,Support Network Red,coordinates_with,0.85
Unknown Financier,Warehouse 17,finances,0.92
Border Transfer Team,Route Node K-12,uses_route,0.88
Support Network Red,Transit Corridor North,supports,0.72
Courier B,Unknown Logistics Cell,associated_with,0.65
Warehouse 17,Bamako,located_in,0.95
"""

CONF_ORDER = {"Low": 1, "Medium": 2, "High": 3}
STATUS_ORDER = {"Monitor": 1, "Watch": 2, "Escalating": 3}

def normalize_data(entities, reports, edges):
    entities = entities.copy()
    reports = reports.copy()
    edges = edges.copy()
    entities["last_seen"] = pd.to_datetime(entities["last_seen"])
    entities["first_seen"] = pd.to_datetime(entities["first_seen"])
    reports["date"] = pd.to_datetime(reports["date"])
    entities["linked_actor"] = entities["linked_actor"].astype(str).str.lower().isin(["true", "1", "yes"])
    entities["confidence_rank"] = entities["confidence"].map(CONF_ORDER).fillna(0)
    entities["status_rank"] = entities["status"].map(STATUS_ORDER).fillna(0)
    return entities, reports, edges

def load_default_data():
    return normalize_data(
        pd.read_csv(StringIO(SAMPLE_ENTITIES)),
        pd.read_csv(StringIO(SAMPLE_REPORTS)),
        pd.read_csv(StringIO(SAMPLE_EDGES)),
    )

def try_load_local_csvs():
    data_dir = Path("data")
    ef, rf, xf = data_dir / "entities.csv", data_dir / "reports.csv", data_dir / "edges.csv"
    if ef.exists() and rf.exists() and xf.exists():
        return normalize_data(pd.read_csv(ef), pd.read_csv(rf), pd.read_csv(xf))
    return load_default_data()

@st.cache_data
def get_data(mode, ue=None, ur=None, ux=None):
    if mode == "Use bundled sample data":
        return load_default_data()
    if mode == "Load local CSVs from ./data":
        return try_load_local_csvs()
    if ue is not None and ur is not None and ux is not None:
        return normalize_data(pd.read_csv(ue), pd.read_csv(ur), pd.read_csv(ux))
    return load_default_data()

def score_band(score):
    if score >= 0.75: return "Severe"
    if score >= 0.50: return "Elevated"
    return "Moderate"

def status_badge(status):
    colors = {"Escalating": "#ef4444", "Watch": "#f59e0b", "Monitor": "#10b981"}
    c = colors.get(status, "#94a3b8")
    return f"<span style='background:{c};padding:0.2rem 0.55rem;border-radius:999px;color:white;font-size:0.8rem'>{status}</span>"

def filter_entities(entities, start_date, end_date, locations, entity_types, confidences, min_score):
    df = entities[(entities["last_seen"].dt.date >= start_date) & (entities["last_seen"].dt.date <= end_date)]
    if locations: df = df[df["primary_location"].isin(locations)]
    if entity_types: df = df[df["type"].isin(entity_types)]
    if confidences: df = df[df["confidence"].isin(confidences)]
    return df[df["threat_score"] >= min_score]

def filter_reports(reports, start_date, end_date, source_types, locations, confidences):
    df = reports[(reports["date"].dt.date >= start_date) & (reports["date"].dt.date <= end_date)]
    if source_types: df = df[df["source_type"].isin(source_types)]
    if locations: df = df[df["location"].isin(locations)]
    if confidences: df = df[df["confidence"].isin(confidences)]
    return df

def compute_rising_threats(entities):
    return int(len(entities[(entities["mentions"] >= 5) & (entities["threat_score"] >= 0.70)])) if not entities.empty else 0

def build_network_figure(edges, filtered_entities):
    allowed_nodes = set(filtered_entities["name"].tolist()) | set(filtered_entities["primary_location"].dropna().tolist())
    df = edges[(edges["source"].isin(allowed_nodes)) | (edges["target"].isin(allowed_nodes))].copy()
    if df.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_dark", title="No network edges match the current filters", height=550)
        return fig
    nodes = pd.unique(df[["source", "target"]].values.ravel("K")).tolist()
    positions = {node: (math.cos(i * 2 * math.pi / max(len(nodes), 1)), math.sin(i * 2 * math.pi / max(len(nodes), 1))) for i, node in enumerate(nodes)}
    edge_traces = []
    for _, row in df.iterrows():
        x0, y0 = positions[row["source"]]
        x1, y1 = positions[row["target"]]
        edge_traces.append(go.Scatter(x=[x0, x1, None], y=[y0, y1, None], mode="lines",
                                      line=dict(width=max(1.0, row["weight"] * 4), color="#64748b"),
                                      hoverinfo="text",
                                      text=[f'{row["source"]} → {row["target"]}<br>Relationship: {row["relationship"]}<br>Weight: {row["weight"]}'] * 3,
                                      showlegend=False))
    x_nodes, y_nodes, labels, sizes, colors = [], [], [], [], []
    for node in nodes:
        x, y = positions[node]
        x_nodes.append(x); y_nodes.append(y); labels.append(node)
        match = filtered_entities[filtered_entities["name"] == node]
        if not match.empty:
            score = float(match.iloc[0]["threat_score"])
            sizes.append(18 + score * 22)
            colors.append("#ef4444" if score >= 0.75 else "#f59e0b" if score >= 0.50 else "#10b981")
        else:
            sizes.append(16); colors.append("#60a5fa")
    node_trace = go.Scatter(x=x_nodes, y=y_nodes, mode="markers+text", text=labels, textposition="top center",
                            hoverinfo="text", hovertext=[f"Node: {label}" for label in labels],
                            marker=dict(size=sizes, color=colors, line=dict(width=1, color="#e5e7eb")), showlegend=False)
    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(template="plotly_dark", title="Link Analysis", height=580,
                      margin=dict(l=10, r=10, t=50, b=10),
                      xaxis=dict(showgrid=False, zeroline=False, visible=False),
                      yaxis=dict(showgrid=False, zeroline=False, visible=False))
    return fig

st.sidebar.title("Controls")
data_mode = st.sidebar.radio("Data source", ["Use bundled sample data", "Load local CSVs from ./data", "Upload CSV files"], index=0)
ue = ur = ux = None
if data_mode == "Upload CSV files":
    ue = st.sidebar.file_uploader("Upload entities.csv", type=["csv"])
    ur = st.sidebar.file_uploader("Upload reports.csv", type=["csv"])
    ux = st.sidebar.file_uploader("Upload edges.csv", type=["csv"])

entities, reports, edges = get_data(data_mode, ue, ur, ux)

all_locations = sorted(set(entities["primary_location"].dropna().unique()).union(set(reports["location"].dropna().unique())))
all_types = sorted(entities["type"].dropna().unique())
all_conf = ["High", "Medium", "Low"]
all_source_types = sorted(reports["source_type"].dropna().unique())

date_min = min(entities["last_seen"].min().date(), reports["date"].min().date())
date_max = max(entities["last_seen"].max().date(), reports["date"].max().date())
selected_range = st.sidebar.date_input("Date range", value=(date_min, date_max), min_value=date_min, max_value=date_max)
start_date, end_date = selected_range if isinstance(selected_range, tuple) and len(selected_range) == 2 else (date_min, date_max)

selected_source_types = st.sidebar.multiselect("Source type", all_source_types, default=all_source_types)
selected_locations = st.sidebar.multiselect("Region / location", all_locations, default=all_locations)
selected_entity_types = st.sidebar.multiselect("Entity type", all_types, default=all_types)
selected_confidences = st.sidebar.multiselect("Confidence level", all_conf, default=all_conf)
min_threat_score = st.sidebar.slider("Threat score threshold", 0.0, 1.0, 0.50, 0.01)

filtered_entities = filter_entities(entities, start_date, end_date, selected_locations, selected_entity_types, selected_confidences, min_threat_score)
filtered_reports = filter_reports(reports, start_date, end_date, selected_source_types, selected_locations, selected_confidences)

st.title("Hybrid Intelligence Analysis Platform")
st.caption("Mission-style dashboard for entity prioritization, reporting traceability, geospatial activity, and link analysis.")

k1, k2, k3, k4 = st.columns(4)
k1.metric("High-Risk Entities", int((filtered_entities["threat_score"] >= 0.75).sum()))
k2.metric("New Reports", int(len(filtered_reports)))
k3.metric("Locations of Interest", int(filtered_entities["primary_location"].nunique()) if not filtered_entities.empty else 0)
k4.metric("Rising Threats", compute_rising_threats(filtered_entities))

left, right = st.columns([1.55, 1.0], gap="large")
with left:
    st.subheader("Threat Prioritization")
    if not filtered_entities.empty:
        display_df = filtered_entities.assign(score_band=filtered_entities["threat_score"].apply(score_band))[[
            "name", "type", "threat_score", "score_band", "confidence", "mentions", "last_seen", "primary_location", "linked_actor", "status"
        ]].sort_values(by=["threat_score", "mentions"], ascending=[False, False]).rename(columns={
            "name": "Entity", "type": "Type", "threat_score": "Threat Score", "score_band": "Band",
            "confidence": "Confidence", "mentions": "Mentions", "last_seen": "Last Seen",
            "primary_location": "Location", "linked_actor": "Linked Actor", "status": "Status"
        })
        st.dataframe(display_df, use_container_width=True, height=320)
    else:
        st.info("No entities match the current filters.")
    st.subheader("Threat Trend")
    trend_rows = []
    for _, row in filtered_reports.iterrows():
        for entity in [e.strip() for e in str(row["entities"]).split("|")]:
            match = filtered_entities[filtered_entities["name"] == entity]
            if not match.empty:
                trend_rows.append({"date": row["date"], "entity": entity, "threat_score": float(match.iloc[0]["threat_score"])})
    trend_df = pd.DataFrame(trend_rows)
    if not trend_df.empty:
        selectable = trend_df["entity"].dropna().unique().tolist()
        chosen = st.multiselect("Select up to 3 entities", selectable, default=selectable[:min(3, len(selectable))], max_selections=3, key="trend_entities")
        if chosen:
            fig_trend = px.line(trend_df[trend_df["entity"].isin(chosen)].sort_values("date"), x="date", y="threat_score", color="entity", markers=True, template="plotly_dark")
            fig_trend.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("No report-linked entities available for trend view.")
with right:
    st.subheader("Geographic Activity")
    map_df = filtered_entities[["name", "primary_location", "lat", "lon", "threat_score", "mentions", "status"]].dropna()
    if not map_df.empty:
        st.map(map_df.rename(columns={"lat": "LAT", "lon": "LON"}), latitude="LAT", longitude="LON", size="threat_score", zoom=4)
        st.caption("Map points represent entities and operational locations linked to filtered reporting.")
    else:
        st.info("No mappable entity data is available for the current selection.")
    st.subheader("Score Distribution")
    if not filtered_entities.empty:
        fig_hist = px.histogram(filtered_entities, x="threat_score", nbins=15, template="plotly_dark")
        fig_hist.update_layout(height=260, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("No entity scores available.")

st.divider()
col_network, col_detail = st.columns([1.25, 0.75], gap="large")
with col_network:
    st.subheader("Network / Link Analysis")
    st.plotly_chart(build_network_figure(edges, filtered_entities), use_container_width=True)
with col_detail:
    st.subheader("Entity Detail Panel")
    entity_options = filtered_entities.sort_values("threat_score", ascending=False)["name"].tolist()
    selected_entity = st.selectbox("Select entity", entity_options if entity_options else ["No entities available"])
    if entity_options and selected_entity in filtered_entities["name"].values:
        row = filtered_entities[filtered_entities["name"] == selected_entity].iloc[0]
        linked_edges = edges[(edges["source"] == selected_entity) | (edges["target"] == selected_entity)].copy()
        linked_reports = filtered_reports[filtered_reports["entities"].fillna("").str.contains(selected_entity, regex=False)]
        with st.container(border=True):
            st.markdown(f"### {row['name']}")
            st.markdown(f"**Type:** {row['type']}")
            st.markdown(f"**Threat Score:** {row['threat_score']:.2f}")
            st.markdown(f"**Confidence:** {row['confidence']}")
            st.markdown(status_badge(row["status"]), unsafe_allow_html=True)
            st.markdown(f"**Primary Location:** {row['primary_location']}")
            st.markdown(f"**First Seen:** {row['first_seen'].date()}")
            st.markdown(f"**Last Seen:** {row['last_seen'].date()}")
            st.markdown(f"**Mentions:** {int(row['mentions'])}")
            st.markdown(f"**Linked Threat Actor:** {'Yes' if bool(row['linked_actor']) else 'No'}")
        score_expl = []
        if bool(row["linked_actor"]): score_expl.append("+0.40 known association with flagged actor")
        if int(row["mentions"]) >= 5: score_expl.append("+0.20 repeated reporting volume")
        elif int(row["mentions"]) >= 3: score_expl.append("+0.10 multiple mentions")
        if row["confidence"] == "High": score_expl.append("+0.15 high-confidence reporting")
        elif row["confidence"] == "Medium": score_expl.append("+0.10 medium-confidence reporting")
        if row["status"] == "Escalating": score_expl.append("+0.07 recent escalation pattern")
        st.markdown("**Scoring rationale**")
        st.code(f"Threat Score: {row['threat_score']:.2f}\n- " + "\n- ".join(score_expl if score_expl else ["Baseline risk only"]), language="text")
        st.markdown("**Known connections**")
        st.dataframe(linked_edges.rename(columns={"source":"Source","target":"Target","relationship":"Relationship","weight":"Weight"}), use_container_width=True, height=180) if not linked_edges.empty else st.info("No linked connections found.")
        st.markdown("**Linked reports**")
        st.dataframe(linked_reports[["report_id","date","source_type","confidence","location","relevance"]].sort_values("date", ascending=False), use_container_width=True, height=180) if not linked_reports.empty else st.info("No filtered reports reference this entity.")
    else:
        st.info("No entity detail is available.")

st.divider()
st.subheader("Report Explorer")
if filtered_reports.empty:
    st.info("No reports match the current filters.")
else:
    sort_choice = st.selectbox("Sort reports by", ["Most recent", "Highest confidence", "Source type"])
    report_df = filtered_reports.copy()
    if sort_choice == "Most recent":
        report_df = report_df.sort_values("date", ascending=False)
    elif sort_choice == "Highest confidence":
        report_df["confidence_rank"] = report_df["confidence"].map(CONF_ORDER).fillna(0)
        report_df = report_df.sort_values(["confidence_rank", "date"], ascending=[False, False])
    else:
        report_df = report_df.sort_values(["source_type", "date"], ascending=[True, False])
    report_search = st.text_input("Search excerpts or entity names")
    if report_search:
        report_df = report_df[
            report_df["excerpt"].fillna("").str.contains(report_search, case=False, regex=False) |
            report_df["entities"].fillna("").str.contains(report_search, case=False, regex=False) |
            report_df["location"].fillna("").str.contains(report_search, case=False, regex=False)
        ]
    for _, row in report_df.head(8).iterrows():
        st.markdown('<div class="intel-card">', unsafe_allow_html=True)
        c1, c2 = st.columns([0.7, 0.3])
        with c1:
            st.markdown(f"**Report:** {row['report_id']}")
            st.markdown(f"<span class='small-muted'>Date: {row['date'].date()} | Source: {row['source_type']} | Confidence: {row['confidence']} | Location: {row['location']}</span>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"**Relevance:** {row['relevance']}")
        st.markdown(f"**Excerpt:** {row['excerpt']}")
        st.markdown(f"**Extracted Entities:** {row['entities']}")
        st.markdown('</div>', unsafe_allow_html=True)

st.divider()
st.caption("Place real project CSVs in ./data as entities.csv, reports.csv, and edges.csv, or upload them from the sidebar. This app is intentionally packaged in one file for quick local demos.")
