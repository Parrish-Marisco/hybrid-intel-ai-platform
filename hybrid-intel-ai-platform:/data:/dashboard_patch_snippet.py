
# Add this after loading unified_graph_ai.json into `graph`

ai = graph.get("ai", {})
ai_reports = pd.DataFrame(ai.get("report_enrichment", []))
ai_similarity = pd.DataFrame(ai.get("similarity_links", []))

st.subheader("AI Triage")
if not ai_reports.empty:
    st.dataframe(
        ai_reports.sort_values("ai_anomaly_score", ascending=False),
        use_container_width=True,
        height=240,
    )
else:
    st.info("No AI enrichment records found.")

st.subheader("AI Similarity Links")
if not ai_similarity.empty:
    st.dataframe(
        ai_similarity.sort_values("similarity", ascending=False).head(20),
        use_container_width=True,
        height=240,
    )
else:
    st.info("No similarity links found.")
