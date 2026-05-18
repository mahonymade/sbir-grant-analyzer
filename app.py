"""
SBIR Grant Analyzer — Streamlit web app.

Run locally:
    streamlit run app.py

Two tabs:
    1. Project Similarity Search  — find grants similar to your project
    2. Phase Conversion Analysis  — Phase I → II conversion rates
"""

import io
import os

import pandas as pd
import streamlit as st

from src.data_loader import load_data, DEFAULT_DISPLAY_COLS
from src.conversion import find_conversions
from src.similarity import (
    filter_by_keywords,
    filter_by_embeddings,
    filter_by_llm,
    estimate_llm_cost,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SBIR Grant Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar — data source
# ---------------------------------------------------------------------------

st.sidebar.title("SBIR Grant Analyzer")
st.sidebar.markdown("---")
st.sidebar.subheader("Data Source")

data_source = st.sidebar.radio(
    "Load data from:",
    ["CSV file", "API (coming soon)"],
    index=0,
)

if data_source == "CSV file":
    default_path = os.path.join(os.path.dirname(__file__), "award_data.csv")
    csv_path = st.sidebar.text_input("CSV file path", value=default_path)
    try:
        df_full = load_data(source="csv", path=csv_path)
        st.sidebar.success(f"Loaded {len(df_full):,} grants")
    except FileNotFoundError:
        st.sidebar.error(f"File not found: `{csv_path}`\nSee `data/README.md` for setup instructions.")
        st.stop()
    except Exception as e:
        st.sidebar.error(f"Error loading data: {e}")
        st.stop()
else:
    st.sidebar.info("API integration is not yet available. Please use a CSV file.")
    st.stop()

st.sidebar.markdown("---")

# ---------------------------------------------------------------------------
# Sidebar — filters (shared across tabs)
# ---------------------------------------------------------------------------

st.sidebar.subheader("Filters")

agencies = sorted(df_full["agency"].dropna().unique().tolist())
selected_agencies = st.sidebar.multiselect(
    "Agency",
    options=agencies,
    default=[],
    placeholder="All agencies",
)

programs = sorted(df_full["program"].dropna().unique().tolist())
selected_programs = st.sidebar.multiselect(
    "Program",
    options=programs,
    default=[],
    placeholder="All programs (SBIR / STTR)",
)

phases = sorted(df_full["phase"].dropna().unique().tolist())
selected_phases = st.sidebar.multiselect(
    "Phase",
    options=phases,
    default=[],
    placeholder="All phases",
)

year_min = int(df_full["award_year"].min())
year_max = int(df_full["award_year"].max())
year_range = st.sidebar.slider(
    "Award year range",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max),
)

# Apply filters
df_filtered = df_full.copy()
if selected_agencies:
    df_filtered = df_filtered[df_filtered["agency"].isin(selected_agencies)]
if selected_programs:
    df_filtered = df_filtered[df_filtered["program"].isin(selected_programs)]
if selected_phases:
    df_filtered = df_filtered[df_filtered["phase"].isin(selected_phases)]
df_filtered = df_filtered[
    df_filtered["award_year"].between(year_range[0], year_range[1])
]

st.sidebar.markdown(f"**{len(df_filtered):,}** grants after filters")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_search, tab_conversion = st.tabs(
    ["🔍 Project Similarity Search", "📊 Phase Conversion Analysis"]
)

# ===========================================================================
# TAB 1 — Project Similarity Search
# ===========================================================================

with tab_search:
    st.header("Project Similarity Search")
    st.markdown(
        "Find SBIR grants that are most similar to your research project. "
        "Choose a search mode below."
    )

    project_description = st.text_area(
        "Describe your project",
        value=(
            "Development of an inline chlorination sensor for real-time monitoring "
            "of residual chlorine levels in drinking water distribution systems. "
            "The sensor must be low-cost, accurate, and suitable for continuous deployment."
        ),
        height=110,
    )

    mode = st.radio(
        "Search mode",
        ["Keyword", "Embeddings (semantic)", "LLM Scoring (Admin)"],
        horizontal=True,
        index=0,
    )

    # ---- Keyword mode ----
    if mode == "Keyword":
        col1, col2 = st.columns([3, 1])
        with col1:
            keyword_input = st.text_input(
                "Keywords (comma-separated)",
                value="chlorine, water quality, sensor, drinking water",
            )
        with col2:
            match_mode = st.radio("Match mode", ["Any keyword", "All keywords"], index=0)

        keywords = [k.strip() for k in keyword_input.split(",") if k.strip()]

        if st.button("Search", type="primary", key="keyword_search"):
            if not keywords:
                st.warning("Enter at least one keyword.")
            else:
                results = filter_by_keywords(
                    df_filtered,
                    keywords,
                    match_mode="any" if match_mode == "Any keyword" else "all",
                )
                st.session_state["search_results"] = results
                st.session_state["search_mode"] = "keyword"

    # ---- Embeddings mode ----
    elif mode == "Embeddings (semantic)":
        st.info(
            "This mode uses a local sentence embedding model (~80 MB, downloaded once). "
            "Grants are ranked by semantic similarity to your project description."
        )
        col1, col2 = st.columns(2)
        with col1:
            top_n = st.slider("Maximum results", min_value=10, max_value=200, value=50, step=10)
        with col2:
            threshold = st.slider("Minimum similarity score", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

        if st.button("Run similarity search", type="primary", key="emb_search"):
            if not project_description.strip():
                st.warning("Enter a project description above.")
            else:
                with st.spinner("Computing similarities…"):
                    results = filter_by_embeddings(
                        df_filtered,
                        project_description,
                        top_n=top_n,
                        threshold=threshold,
                    )
                st.session_state["search_results"] = results
                st.session_state["search_mode"] = "embeddings"

    # ---- LLM mode ----
    elif mode == "LLM Scoring (Admin)":
        st.warning("LLM scoring uses the Claude API and incurs cost. Admin access required.")

        admin_unlocked = False
        try:
            stored_password = st.secrets.get("ADMIN_PASSWORD", "")
        except Exception:
            stored_password = ""

        if stored_password:
            password_input = st.text_input("Admin password", type="password")
            admin_unlocked = password_input == stored_password
            if password_input and not admin_unlocked:
                st.error("Incorrect password.")
        else:
            st.info(
                "No admin password configured. "
                "Add `ADMIN_PASSWORD` to `.streamlit/secrets.toml` to enable this mode."
            )

        if admin_unlocked:
            st.success("Admin access granted.")
            try:
                api_key = st.secrets["ANTHROPIC_API_KEY"]
            except Exception:
                api_key = st.text_input("Anthropic API key", type="password")

            col1, col2 = st.columns(2)
            with col1:
                min_score = st.slider("Minimum relevance score (0–10)", 0, 10, 5)
            with col2:
                n_to_score = min(len(df_filtered), 500)
                est_cost = estimate_llm_cost(n_to_score)
                st.metric(
                    "Estimated API cost",
                    f"~${est_cost:.2f}",
                    help=f"Scoring up to {n_to_score} grants (first 500 after filters).",
                )

            if st.button("Run LLM scoring", type="primary", key="llm_search"):
                if not api_key:
                    st.error("API key required.")
                elif not project_description.strip():
                    st.warning("Enter a project description above.")
                else:
                    subset = df_filtered.head(n_to_score)
                    results = filter_by_llm(
                        subset,
                        project_description,
                        api_key=api_key,
                        min_score=min_score,
                    )
                    st.session_state["search_results"] = results
                    st.session_state["search_mode"] = "llm"

    # ---- Results display ----
    st.markdown("---")

    if "search_results" in st.session_state:
        results: pd.DataFrame = st.session_state["search_results"]
        search_mode_used = st.session_state.get("search_mode", "keyword")

        st.metric("Matching grants found", f"{len(results):,}")

        if len(results) == 0:
            st.info("No grants matched your search. Try different keywords or a lower threshold.")
        else:
            show_all_cols = st.toggle("Show all columns", value=False)

            # Choose columns to display
            score_col = None
            if search_mode_used == "embeddings" and "similarity_score" in results.columns:
                score_col = "similarity_score"
            elif search_mode_used == "llm" and "llm_score" in results.columns:
                score_col = "llm_score"

            if show_all_cols:
                display_cols = [c for c in results.columns if not c.startswith("_") and not c.endswith("_lc")]
            else:
                base_cols = [c for c in DEFAULT_DISPLAY_COLS if c in results.columns]
                display_cols = ([score_col] + base_cols) if score_col else base_cols

            display_df = results[display_cols].reset_index(drop=True)

            # Format award_amount nicely
            if "award_amount" in display_df.columns:
                display_df = display_df.copy()
                display_df["award_amount"] = display_df["award_amount"].apply(
                    lambda x: f"${x:,.0f}" if pd.notna(x) else ""
                )
            if score_col in display_df.columns:
                display_df = display_df.copy()
                display_df[score_col] = display_df[score_col].round(3)

            st.dataframe(
                display_df,
                use_container_width=True,
                height=500,
                column_config={
                    "abstract": st.column_config.TextColumn("Abstract", width="large"),
                    "award_title": st.column_config.TextColumn("Award Title", width="medium"),
                },
            )

            # Download button — generates CSV only when clicked
            csv_buffer = io.StringIO()
            results[[c for c in results.columns if not c.endswith("_lc") and not c.startswith("_")]].to_csv(
                csv_buffer, index=False
            )
            st.download_button(
                label="Download results as CSV",
                data=csv_buffer.getvalue(),
                file_name="sbir_search_results.csv",
                mime="text/csv",
            )

# ===========================================================================
# TAB 2 — Phase Conversion Analysis
# ===========================================================================

with tab_conversion:
    st.header("Phase I → Phase II Conversion Analysis")
    st.markdown(
        "Estimate the rate at which companies that received Phase I awards "
        "went on to receive a Phase II award for the same project. "
        "Phase II matching always searches the **full dataset** regardless of year filters."
    )

    # ---- Controls ----
    ctrl_col1, ctrl_col2 = st.columns([2, 1])

    with ctrl_col1:
        p1_year_min = int(df_full["award_year"].min())
        p1_year_max = int(df_full["award_year"].max())
        p1_year_range = st.slider(
            "Phase I award year range",
            min_value=p1_year_min,
            max_value=p1_year_max,
            value=(p1_year_min, p1_year_max),
            help="Filter which Phase I grants are included. Phase II matching always uses all years.",
        )

    with ctrl_col2:
        fuzzy_threshold = st.slider(
            "Title match threshold",
            min_value=60,
            max_value=100,
            value=85,
            step=5,
            help="Higher = stricter title matching. 85 is a good default.",
        )

    # Phase II pool is always the full dataset, all years
    phase2_pool = df_full[df_full["phase"].str.strip() == "Phase II"]

    # Overall Phase I pool: full dataset filtered to selected year range only
    phase1_overall = df_full[
        (df_full["phase"].str.strip() == "Phase I") &
        (df_full["award_year"].between(p1_year_range[0], p1_year_range[1]))
    ]

    # Similarity-filtered Phase I pool (if a search has been run)
    has_search = "search_results" in st.session_state
    if has_search:
        search_res = st.session_state["search_results"]
        phase1_similar = search_res[
            (search_res["phase"].str.strip() == "Phase I") &
            (search_res["award_year"].between(p1_year_range[0], p1_year_range[1]))
        ]
        similar_label = f"Similar grants only ({len(phase1_similar):,} Phase I)"
        st.info(
            f"A similarity search is active with **{len(search_res):,}** results "
            f"(**{len(phase1_similar):,}** are Phase I within the selected year range). "
            "Run the analysis to compare conversion rates."
        )

    if st.button("Run conversion analysis", type="primary"):
        overall = find_conversions(phase1_overall, phase2_pool, fuzzy_threshold=fuzzy_threshold)
        st.session_state["conv_overall"] = overall

        if has_search and len(phase1_similar) > 0:
            filtered = find_conversions(phase1_similar, phase2_pool, fuzzy_threshold=fuzzy_threshold)
            st.session_state["conv_filtered"] = filtered
        else:
            st.session_state.pop("conv_filtered", None)

    # ---- Results ----
    if "conv_overall" in st.session_state:
        r_overall = st.session_state["conv_overall"]
        r_filtered = st.session_state.get("conv_filtered")

        def _fmt(n: int) -> str:
            """Compact number format: 144660 → '144.7K', 63071 → '63.1K'."""
            if n >= 1_000_000:
                return f"{n/1_000_000:.1f}M"
            if n >= 1_000:
                return f"{n/1_000:.1f}K"
            return str(n)

        # Top-level metrics: overall always shown; filtered shown alongside if available
        if r_filtered:
            st.markdown("### Conversion Rate Comparison")
            left, right = st.columns(2)
            with left:
                st.markdown("**Overall (all grants)**")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Phase I", _fmt(r_overall["phase1_count"]))
                m2.metric("Phase II pool", _fmt(r_overall["phase2_count"]))
                m3.metric("Converted", _fmt(r_overall["matched_count"]))
                m4.metric("Rate", f"{r_overall['conversion_rate']:.1%}")
            with right:
                st.markdown("**Similar grants only**")
                delta = r_filtered["conversion_rate"] - r_overall["conversion_rate"]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Phase I", _fmt(r_filtered["phase1_count"]))
                m2.metric("Phase II pool", _fmt(r_filtered["phase2_count"]))
                m3.metric("Converted", _fmt(r_filtered["matched_count"]))
                m4.metric(
                    "Rate",
                    f"{r_filtered['conversion_rate']:.1%}",
                    delta=f"{delta:+.1%} vs overall",
                )
            # Toggle which result set drives the charts/table below
            view = st.radio(
                "Show charts and pairs table for:",
                ["Overall", "Similar grants only"],
                horizontal=True,
            )
            r = r_filtered if view == "Similar grants only" else r_overall
        else:
            st.markdown("### Overall Results")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Phase I grants", _fmt(r_overall["phase1_count"]))
            m2.metric("Phase II pool", _fmt(r_overall["phase2_count"]))
            m3.metric("Matched conversions", _fmt(r_overall["matched_count"]))
            m4.metric("Conversion rate", f"{r_overall['conversion_rate']:.1%}")
            r = r_overall

        st.markdown("---")

        # ---- Charts ----
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.subheader("Conversion rate by agency (top 10)")
            agency_data = r["by_agency"].reset_index().rename(
                columns={"rate": "Conversion Rate"}
            ).head(10)
            agency_data["Conversion Rate %"] = (agency_data["Conversion Rate"] * 100).round(1)
            st.bar_chart(agency_data.set_index("agency")[["Conversion Rate %"]], height=350)

        with chart_col2:
            st.subheader("Conversion rate by Phase I award year")
            year_data = r["by_year"].reset_index()
            year_data["Conversion Rate %"] = (year_data["rate"] * 100).round(1)
            # Cast year to string so Streamlit doesn't add comma thousand-separators
            year_data["Year"] = year_data["award_year"].astype(int).astype(str)
            st.line_chart(year_data.set_index("Year")[["Conversion Rate %"]], height=350)

        st.markdown("---")

        # ---- Matched pairs table ----
        st.subheader(f"Matched grant pairs ({r['matched_count']:,})")
        st.markdown("Each row shows a Phase I grant and its matched Phase II counterpart.")

        show_all_conv = st.toggle("Show all columns", value=False, key="conv_toggle")
        pairs = r["matched_pairs"]
        if not show_all_conv:
            keep = [c for c in pairs.columns if any(
                c.endswith(s) for s in ("company", "award_title", "agency", "award_year", "award_amount")
            )]
            pairs = pairs[keep]

        st.dataframe(pairs.reset_index(drop=True), use_container_width=True, height=400)

        csv_buffer_conv = io.StringIO()
        r["matched_pairs"].to_csv(csv_buffer_conv, index=False)
        st.download_button(
            label="Download matched pairs as CSV",
            data=csv_buffer_conv.getvalue(),
            file_name="sbir_conversion_pairs.csv",
            mime="text/csv",
            key="conv_download",
        )
