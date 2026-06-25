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
    ["Hosted dataset", "Local CSV"],
    index=0,
    help=(
        "Hosted dataset downloads slim, precomputed artifacts once "
        "(no large file for you to manage). Local CSV reads a raw award_data.csv."
    ),
)

try:
    if data_source == "Hosted dataset":
        df_full = load_data(source="parquet")
    else:
        default_path = os.path.join(os.path.dirname(__file__), "award_data.csv")
        csv_path = st.sidebar.text_input("CSV file path", value=default_path)
        st.sidebar.caption(
            "Don't have the data file? Download it from "
            "[SBIR.gov Data Resources](https://www.sbir.gov/data-resources)."
        )
        df_full = load_data(source="csv", path=csv_path)
    st.sidebar.success(f"Loaded {len(df_full):,} grants")
except FileNotFoundError as e:
    st.sidebar.error(f"Data file not found: {e}\nSee `data/README.md` for setup instructions.")
    st.stop()
except Exception as e:
    st.sidebar.error(f"Error loading data: {e}")
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
    "Award year range (pre-filter for all tabs)",
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
# Guide tab content
# ---------------------------------------------------------------------------

_GUIDE_HTML = """
<article class="sbir-writeup">
  <header>
    <h1>From SBIR Phase 1 to Phase 2: How the Transition Works</h1>
    <p class="lede">
      The SBIR (Small Business Innovation Research) program is structured as a deliberate, staged
      funding pipeline designed to de-risk early-stage R&amp;D for both the federal government and
      the small business. Moving from Phase 1 to Phase 2 is the most important inflection point in
      the program — and it is far from automatic.
    </p>
  </header>

  <section>
    <h2>The basic structure</h2>
    <p>
      Phase 1 is a small feasibility award intended to test whether a technical concept can work.
      At most agencies, Phase 1 awards run roughly $150,000 to $295,000 over 6 to 12 months, with
      NIH being a notable outlier that allows Phase 1 grants to extend up to two years.<sup><a href="#ref-1">1</a></sup>
      Deliverables are deliberately modest: a technical feasibility report, an initial
      commercialization plan, and a roadmap for Phase 2.<sup><a href="#ref-1">1</a></sup>
    </p>
    <p>
      Phase 2 is a much larger commitment. Award sizes typically range from $750,000 to roughly
      $2 million over approximately 24 months, funding the prototype development and engineering
      work needed to turn a validated concept into a working technology.<sup><a href="#ref-2">2</a></sup>
      In Technology Readiness Level terms, Phase 1 is generally expected to move a technology from
      TRL 2–3 to TRL 4–5, while Phase 2 targets TRL 5–6.<sup><a href="#ref-3">3</a></sup>
    </p>
    <p>
      Phase 3, by contrast, is not an SBIR-funded award at all — it is the commercialization stage,
      pursued with non-SBIR funds (private capital, government procurement contracts, or licensing
      revenue).<sup><a href="#ref-2">2</a></sup>
    </p>
  </section>

  <section>
    <h2>Eligibility and pathways into Phase 2</h2>
    <p>
      With limited exceptions, Phase 2 is restricted to companies that have already completed a
      Phase 1 award at the same agency.<sup><a href="#ref-4">4</a></sup> A Phase 1 award does not
      guarantee Phase 2 funding; it is essentially a license to compete in a separate, more
      rigorous Phase 2 review against other Phase 1 graduates.
    </p>
    <p>
      Several agencies — notably DoD, NIH, and DOE — offer a <em>Direct to Phase II</em> (D2P2)
      pathway for companies that have already completed equivalent feasibility work using non-SBIR
      funding such as private investment or corporate R&amp;D dollars. NSF does not currently offer
      a Direct to Phase II pathway.<sup><a href="#ref-4">4</a></sup> The D2P2 bar is high:
      reviewers evaluate the submitted feasibility data as though it were a Phase 1 final report,
      and weak feasibility evidence is the leading reason D2P2 proposals fail.<sup><a href="#ref-4">4</a></sup>
    </p>
  </section>

  <section>
    <h2>Conversion rates from Phase 1 to Phase 2</h2>
    <p>
      Because the Phase 2 applicant pool is restricted to Phase 1 graduates, conversion rates are
      meaningfully higher than initial Phase 1 win rates, which generally run 15–25% across major
      agencies.<sup><a href="#ref-1">1</a></sup> Reported Phase 1-to-Phase 2 conversion rates vary
      by source and agency, but typically fall in the range of 30–55%.<sup><a href="#ref-2">2</a></sup><sup>,</sup><sup><a href="#ref-4">4</a></sup>
    </p>
  </section>

  <section>
    <h2>Timeline and submission mechanics</h2>
    <p>
      The submission window and review cadence vary considerably by agency:
    </p>
    <ul>
      <li>
        <strong>NIH:</strong> Phase 2 applications must be submitted within two years of the
        Phase 1 end date.<sup><a href="#ref-4">4</a></sup>
      </li>
      <li>
        <strong>NSF:</strong> Phase 2 invitations are issued after evaluation of Phase 1 results,
        typically 4 to 6 months after Phase 1 ends.<sup><a href="#ref-4">4</a></sup>
      </li>
      <li>
        <strong>DoD components:</strong> Practices vary widely — some components accept Phase 2
        proposals on a rolling basis while others run annual solicitations.<sup><a href="#ref-4">4</a></sup>
      </li>
      <li>
        <strong>Across agencies:</strong> Expect 4 to 9 months from Phase 2 submission to award
        notification.<sup><a href="#ref-4">4</a></sup>
      </li>
    </ul>
  </section>

  <section>
    <h2>What drives Phase 2 progression</h2>
    <p>
      Five factors do most of the work in determining which Phase 1 awardees successfully move on.
    </p>

    <h3>1. Phase 1 technical results</h3>
    <p>
      Reviewers expect Phase 1 outcomes to be described in enough detail to allow independent
      judgment of whether the feasibility hypothesis was genuinely tested and the technical bar
      was met.<sup><a href="#ref-5">5</a></sup> Vague, incomplete, or unconvincing Phase 1 results
      are the most common technical reason for rejection at the Phase 2 stage.
    </p>

    <h3>2. The commercialization plan</h3>
    <p>
      This is where most applicants stumble. Phase 1 tolerates a thin commercialization section;
      Phase 2 does not. At NIH the commercialization plan is weighted as heavily as the research
      plan, and at NSF commercial potential is even more prominent — a dedicated, solicitation-specific
      merit review criterion in addition to the standard Intellectual Merit and Broader Impacts
      criteria.<sup><a href="#ref-6">6</a></sup> Phase 2 commercialization plans are expected to
      cover the market opportunity, customer characteristics, competition, marketing and sales
      strategy, intellectual property strategy, financing plan, and a credible timeline from
      end-of-Phase-2 to market entry.<sup><a href="#ref-7">7</a></sup>
    </p>

    <h3>3. Team and company capability</h3>
    <p>
      Reviewers assess whether the company has the business network, expertise, and structural
      readiness to actually commercialize the technology — not just do more research. Prior
      commercialization success from earlier SBIR awards is a positive signal but is not strictly
      required for first-time Phase 2 applicants.<sup><a href="#ref-8">8</a></sup>
    </p>

    <h3>4. Third-party validation and matching funds</h3>
    <p>
      Letters of support from prospective customers, evidence of investor interest, and — for DoD
      especially — matching commitments from a government end-user materially strengthen
      applications. DoD Phase 2 Enhancements can increase the SBIR award by up to $1 million for
      every $1 of customer matching funds, up to defined caps.<sup><a href="#ref-2">2</a></sup>
      NASA explicitly evaluates the offeror's record in technology commercialization, co-funding
      commitments, and existing or projected Phase 3 funding sources.<sup><a href="#ref-9">9</a></sup>
    </p>

    <h3>5. Budget-to-workplan coherence</h3>
    <p>
      A budget that does not match the scope of work described — or vice versa — is a common
      and avoidable killer. The technical narrative and the budget must tell a consistent story
      about the scale and intensity of the development effort.<sup><a href="#ref-2">2</a></sup>
    </p>
  </section>

  <section>
    <h2>Agency-specific differences</h2>
    <p>
      Each participating agency emphasizes slightly different criteria:
    </p>
    <ul>
      <li>
        <strong>NSF and NIH</strong> treat commercialization potential as a co-equal or even
        primary merit criterion alongside technical merit.<sup><a href="#ref-6">6</a></sup>
      </li>
      <li>
        <strong>DoD</strong> weights potential government and military application heavily and
        rewards demonstrated engagement with end-user program offices.<sup><a href="#ref-9">9</a></sup>
      </li>
      <li>
        <strong>DOE</strong> evaluates "impact" — the likelihood that the work leads to a
        marketable product and the likelihood of attracting follow-on funding after the SBIR
        project ends.<sup><a href="#ref-9">9</a></sup> DOE also now requires a cybersecurity
        self-assessment and five-year cash-flow pro-forma worksheets as part of the Phase 2
        application.<sup><a href="#ref-10">10</a></sup>
      </li>
      <li>
        <strong>NASA</strong> explicitly evaluates the offeror's commercialization track record
        and any co-funding commitments from private or non-SBIR sources.<sup><a href="#ref-9">9</a></sup>
      </li>
    </ul>
  </section>

  <section>
    <h2>Beyond a single Phase 2 award</h2>
    <p>
      Several agencies offer follow-on funding mechanisms for Phase 2 awardees who need
      additional runway before commercial readiness:
    </p>
    <ul>
      <li>
        <strong>NIH Phase IIB</strong> (competing renewals) can provide an additional $2 million
        for continued development.<sup><a href="#ref-2">2</a></sup>
      </li>
      <li>
        <strong>DoD Phase 2 Enhancements</strong> add matching SBIR funds when a customer commits
        non-SBIR dollars.<sup><a href="#ref-2">2</a></sup>
      </li>
      <li>
        <strong>NSF and NIH supplements</strong> provide additional support for commercialization
        activities, I-Corps participation, and other specific purposes.<sup><a href="#ref-2">2</a></sup>
      </li>
      <li>
        <strong>DOE Phase IIA, IIB, and IIC</strong> structures provide layered follow-on funding,
        some issued as cooperative agreements rather than grants.<sup><a href="#ref-10">10</a></sup>
      </li>
    </ul>
  </section>

  <section>
    <h2>The bottom line</h2>
    <p>
      Companies that successfully transition from Phase 1 to Phase 2 tend to treat Phase 1 not as
      a standalone research project but as the first act of a multi-year commercialization story.
      They use Phase 1 to gather customer letters, validate the market, identify regulatory
      pathways, and build the relationships that will populate the Phase 2 commercialization
      plan.<sup><a href="#ref-2">2</a></sup> Phase 2 reviewers are looking for evidence, not
      promises — both technically and commercially.
    </p>
  </section>

  <footer>
    <h2 id="references">References</h2>
    <ol class="references">
      <li id="ref-1">
        SLED.AI. "SBIR Phase 1 vs Phase 2: Funding, Timeline, and How to Move Up (2026)."
        <a href="https://www.sledai.com/blog/sbir-phase-1-vs-phase-2/" target="_blank" rel="noopener">
          sledai.com/blog/sbir-phase-1-vs-phase-2
        </a>
      </li>
      <li id="ref-2">
        Granted AI. "SBIR Phase I vs Phase II: Requirements, Timelines, and Strategy."
        <a href="https://grantedai.com/blog/sbir-phase-1-vs-phase-2-requirements-strategy" target="_blank" rel="noopener">
          grantedai.com/blog/sbir-phase-1-vs-phase-2-requirements-strategy
        </a>
      </li>
      <li id="ref-3">
        Granted AI. "SBIR Grant Guide for First-Time Applicants (2026)."
        <a href="https://grantedai.com/blog/sbir-grant-guide-2026" target="_blank" rel="noopener">
          grantedai.com/blog/sbir-grant-guide-2026
        </a>
      </li>
      <li id="ref-4">
        Grantsights. "SBIR Phase 2 Guide: Awards &amp; How to Apply (2026)."
        <a href="https://grantsights.com/blog/sbir-phase-2-guide" target="_blank" rel="noopener">
          grantsights.com/blog/sbir-phase-2-guide
        </a>
      </li>
      <li id="ref-5">
        USDA NIFA. "Instructions for Reviewing SBIR Phase II Applications — Technical Proposal."
        <a href="https://www.nifa.usda.gov/sites/default/files/resources/SBIR_phase2_technical_proposal_review_instructions.pdf" target="_blank" rel="noopener">
          nifa.usda.gov (PDF)
        </a>
      </li>
      <li id="ref-6">
        U.S. National Science Foundation. "NSF 24-580: SBIR/STTR Phase II Program Solicitation."
        <a href="https://www.nsf.gov/funding/opportunities/sbirsttr-phase-ii-nsf-small-business-innovation-research-small-business/nsf24-580/solicitation" target="_blank" rel="noopener">
          nsf.gov/funding/opportunities/nsf24-580
        </a>
      </li>
      <li id="ref-7">
        USDA NIFA. "Commercialization Plan Guidance for Phase II Applications."
        <a href="https://www.nifa.usda.gov/commercialization-plan-guidance-phase-ii-applications" target="_blank" rel="noopener">
          nifa.usda.gov/commercialization-plan-guidance-phase-ii-applications
        </a>
      </li>
      <li id="ref-8">
        USDA NIFA. "Instructions for Reviewing SBIR Phase II Applications — Commercialization Plan."
        <a href="https://www.nifa.usda.gov/sites/default/files/resources/SBIR_phase2_commercialization_review_instructions.pdf" target="_blank" rel="noopener">
          nifa.usda.gov (PDF)
        </a>
      </li>
      <li id="ref-9">
        SBIR.gov. "Understand the Proposal Evaluation Criteria."
        <a href="https://www.sbir.gov/tutorials/preparing-proposal/tutorial-1" target="_blank" rel="noopener">
          sbir.gov/tutorials/preparing-proposal/tutorial-1
        </a>
      </li>
      <li id="ref-10">
        U.S. Department of Energy Office of Science. "Preparing DOE SBIR/STTR Phase 2 Applications."
        <a href="https://science.osti.gov/sbir/Applicant-Resources/Grant-Application-Phase-II" target="_blank" rel="noopener">
          science.osti.gov/sbir/Applicant-Resources/Grant-Application-Phase-II
        </a>
      </li>
    </ol>
    <p class="meta">
      <em>Last updated: May 2026. SBIR program details, award caps, and agency procedures change
      frequently — verify against the current solicitation or agency program page before relying
      on specific figures.</em>
    </p>
  </footer>
</article>

<style>
  .sbir-writeup {
    max-width: 760px;
    margin: 0 auto;
    padding: 1.5rem;
    line-height: 1.65;
    color: inherit;
  }
  .sbir-writeup h1 {
    font-size: 1.875rem;
    line-height: 1.25;
    margin: 0 0 0.75rem;
  }
  .sbir-writeup h2 {
    font-size: 1.375rem;
    margin: 2rem 0 0.5rem;
    line-height: 1.3;
  }
  .sbir-writeup h3 {
    font-size: 1.05rem;
    margin: 1.25rem 0 0.4rem;
  }
  .sbir-writeup .lede {
    font-size: 1.05rem;
    opacity: 0.85;
  }
  .sbir-writeup ul,
  .sbir-writeup ol {
    padding-left: 1.5rem;
  }
  .sbir-writeup li {
    margin: 0.35rem 0;
  }
  .sbir-writeup sup a {
    text-decoration: none;
    font-weight: 600;
    padding: 0 1px;
  }
  .sbir-writeup sup a:hover {
    text-decoration: underline;
  }
  .sbir-writeup .references {
    font-size: 0.92rem;
  }
  .sbir-writeup .references li {
    margin: 0.6rem 0;
  }
  .sbir-writeup .meta {
    font-size: 0.85rem;
    opacity: 0.7;
    margin-top: 1.5rem;
  }
</style>
"""

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_search, tab_conversion, tab_guide = st.tabs(
    ["🔍 Project Similarity Search", "📊 Phase Conversion Analysis", "📖 Phase 1 → 2 Guide"]
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
        st.warning("LLM scoring uses the Groq API and incurs cost. Admin access required.")

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
                api_key = st.secrets["GROQ_API_KEY"]
            except Exception:
                api_key = st.text_input("Groq API key", type="password")

            # Gate: require embeddings search results as input
            emb_results = (
                st.session_state.get("search_results")
                if st.session_state.get("search_mode") == "embeddings"
                else None
            )

            if emb_results is None:
                st.info(
                    "First run an **Embeddings** search above, then return here to "
                    "re-score those results with Groq for higher precision. "
                    "This keeps API usage within free tier limits."
                )
            else:
                # Free tier cap: 6000 TPM — limit to 30 grants to stay safe
                LLM_MAX_GRANTS = 30
                llm_input = emb_results.head(LLM_MAX_GRANTS)
                n_to_score = len(llm_input)
                if len(emb_results) > LLM_MAX_GRANTS:
                    st.caption(
                        f"ℹ️ Groq free tier is capped at 6 000 tokens/min. "
                        f"Scoring top {LLM_MAX_GRANTS} of {len(emb_results)} embeddings results."
                    )
                est_cost = estimate_llm_cost(n_to_score)
                col1, col2 = st.columns(2)
                with col1:
                    min_score = st.slider("Minimum relevance score (0–10)", 0, 10, 5)
                with col2:
                    st.metric(
                        "Estimated API cost",
                        f"~${est_cost:.3f}",
                        help=f"Scoring {n_to_score} grants from your embeddings search.",
                    )

                if st.button("Run LLM scoring", type="primary", key="llm_search"):
                    if not api_key:
                        st.error("API key required.")
                    elif not project_description.strip():
                        st.warning("Enter a project description above.")
                    else:
                        with st.spinner("Scoring with Groq…"):
                            results = filter_by_llm(
                                llm_input,
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
        "Use the **Award year range** slider in the sidebar to control which years are analysed."
    )

    # ---- Controls ----
    fuzzy_threshold = st.slider(
        "Title match threshold",
        min_value=60,
        max_value=100,
        value=85,
        step=5,
        help="Higher = stricter title matching. 85 is a good default.",
    )

    # Phase II pool: bounded from sidebar start year — Phase II before that year can't match
    phase2_pool = df_full[
        (df_full["phase"].str.strip() == "Phase II") &
        (df_full["award_year"] >= year_range[0])
    ]

    # Overall Phase I pool: full dataset filtered to sidebar year range
    phase1_overall = df_full[
        (df_full["phase"].str.strip() == "Phase I") &
        (df_full["award_year"].between(year_range[0], year_range[1]))
    ]

    # Similarity-filtered Phase I pool (if a search has been run)
    has_search = "search_results" in st.session_state
    if has_search:
        search_res = st.session_state["search_results"]
        phase1_similar = search_res[
            (search_res["phase"].str.strip() == "Phase I") &
            (search_res["award_year"].between(year_range[0], year_range[1]))
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

# ===========================================================================
# TAB 3 — Phase 1 → 2 Guide
# ===========================================================================

with tab_guide:
    st.html(_GUIDE_HTML)
