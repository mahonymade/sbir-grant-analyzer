# SBIR Grant Analyzer — Session Context

## Core Objective & Project State
We are building a **Streamlit web app** for analyzing SBIR grant data, focused on a chlorine sensing / drinking water project (generalizable to other projects). The app lives at `/Users/carolinemahony/Documents/MSAProject/SBIRAnalysis/` and is run locally via `.venv/bin/streamlit run app.py` at `http://localhost:8501`.

The app has **three integrated tabs**:
1. **Project Similarity Search** — keyword / semantic embeddings / LLM (admin-only, Groq) search over grant titles + abstracts
2. **Phase Conversion Analysis** — Phase I → II conversion rates, overall vs similarity-filtered
3. **Phase 1 → 2 Guide** — static HTML reference article rendered via `st.html()`

The codebase is pushed to GitHub at **https://github.com/mahonymade/sbir-grant-analyzer** (public). The data file is gitignored.

---

## Established Constraints & Rules
- **Data file**: `award_data.csv` (351 MB, 207,731 rows, 41 columns) in project root. Gitignored. Source: https://www.sbir.gov/data-resources
- **Column naming**: All normalized to `lowercase_with_underscores` at load time.
- **Phase encoding**: `"Phase I"` / `"Phase II"` (string with space + Roman numeral). Always `.str.strip()` before comparing.
- **Phase II pool**: Bounded from sidebar `year_range[0]` upward — Phase II awards before the earliest sidebar year are excluded (efficiency). Upper bound is NOT applied to Phase II (a Phase I from 2023 may convert in 2025+).
- **Sidebar year range is the single date pre-filter** for both tabs. The redundant per-tab Phase I year slider was removed. `year_range` from the sidebar is used everywhere.
- **`find_conversions()` has NO `@st.cache_data`** — results manually cached in `st.session_state` on button click (Streamlit's cache ignores `_`-prefixed DataFrame params).
- **LLM mode is admin-only**, gated by `ADMIN_PASSWORD` in `.streamlit/secrets.toml` (gitignored).
- **LLM mode requires embeddings pre-filter** — user must run Embeddings search first; LLM scores only those results. A hard cap of `LLM_MAX_GRANTS = 30` is applied (top 30 of embeddings results) to stay under Groq free tier TPM limit.
- **Batch size**: 50 abstracts per API call. 2s sleep between batches (Groq: 30 RPM limit).
- **`award_amount`** stored as comma-formatted strings in CSV — loader strips commas before `pd.to_numeric()`.
- **Year axis labels** cast to `str` to prevent Streamlit adding comma thousand-separators (e.g. "2,026").
- **Large counts** use compact K/M format via `_fmt()`.
- **No git commits made** across the last two sessions — user prefers to commit manually.

---

## Key Decisions Made

| Decision | Rationale |
|---|---|
| Groq (`llama-3.1-8b-instant`) for LLM scoring | `llama3-8b-8192` was decommissioned by Groq ~May 2025. Official replacement is `llama-3.1-8b-instant`. Model string stored in `GROQ_MODEL` constant in `similarity.py`. |
| `response_format={"type": "json_object"}` in Groq call | Forces clean JSON output — eliminates markdown code-fence wrapping bug that plagued Gemini. |
| `LLM_MAX_GRANTS = 30` cap in app.py | Groq free tier limit is 6000 TPM. 30 grants × 600-char abstracts ≈ ~4500 tokens — safely under limit. A UI caption informs the user when results are being capped. |
| Embeddings gate for LLM mode | Limits LLM scoring to top embeddings results rather than full dataset. Prevents rate limit exhaustion. |
| Sidebar year range as single pre-filter | Removed redundant "Phase I award year range" slider from conversion tab. Sidebar `year_range` drives both tabs. |
| Phase II pool bounded by `year_range[0]` | Efficiency: Phase II awards before earliest Phase I year cannot be matches. Upper bound not applied. |
| Replaced `iterrows()` in `find_conversions()` | Phase II dict built via `groupby().apply()` (C-level). Phase I loop uses `.values` arrays. Significant speedup on 144K row datasets. |
| Groq JSON dict unwrapping | Groq JSON mode may return `{"results": [...]}` or bare array — code handles both via `isinstance(scores, dict)` check. |

---

## Open Issues

### 1. LLM Mode Free Tier TPM — Partially Resolved
**Problem**: Groq free tier = 6000 TPM. 50 grants × 600-char abstracts ≈ 6991 tokens → 413 error.
**Current mitigation**: Hard cap at `LLM_MAX_GRANTS = 30` (≈4500 tokens). Not yet confirmed working end-to-end.
**Unresolved discussion**: Better options under consideration (user rejected truncating abstracts):
- **Option A (recommended)**: Add a "grants to LLM score" slider defaulting to 15, giving user explicit control. 15 grants × ~150 tokens + overhead ≈ 2650 tokens — well under limit.
- **Option B**: Smaller batches (15/call) with ~35s sleep between calls. All grants scored but takes 2–3 min.
- **Option C**: Upgrade Groq to Dev tier ($9/mo), no code changes needed.
- **Option D**: Filter by higher embeddings similarity threshold before LLM step.

### 2. Groq API Key Needs Rotation
Key was shared in plaintext in a chat transcript. Rotate at https://console.groq.com/keys and update `.streamlit/secrets.toml`.

### 3. GitHub Not Yet Updated
Many changes across two sessions — no commits made yet. User prefers to commit manually.

---

## Source of Truth Code

### `src/similarity.py` — LLM constants
```python
GROQ_MODEL = "llama-3.1-8b-instant"  # replacement for decommissioned llama3-8b-8192
BATCH_SIZE = 50
COST_PER_1K_INPUT_TOKENS = 0.00005
COST_PER_1K_OUTPUT_TOKENS = 0.00008
AVG_ABSTRACT_TOKENS = 250

# In filter_by_llm():
response = client.chat.completions.create(
    model=GROQ_MODEL,
    messages=[{"role": "user", "content": prompt}],
    max_tokens=512,
    response_format={"type": "json_object"},
)
```

### `app.py` — LLM section
```python
# Gate: require embeddings results first
emb_results = (
    st.session_state.get("search_results")
    if st.session_state.get("search_mode") == "embeddings"
    else None
)

if emb_results is None:
    st.info("First run an Embeddings search above...")
else:
    LLM_MAX_GRANTS = 30
    llm_input = emb_results.head(LLM_MAX_GRANTS)
    n_to_score = len(llm_input)
    if len(emb_results) > LLM_MAX_GRANTS:
        st.caption(
            f"ℹ️ Groq free tier is capped at 6 000 tokens/min. "
            f"Scoring top {LLM_MAX_GRANTS} of {len(emb_results)} embeddings results."
        )
    # ... sliders, cost estimate, button ...
    results = filter_by_llm(llm_input, project_description, api_key=api_key, min_score=min_score)
```

### `app.py` — other key patterns
```python
# Sidebar year slider (single pre-filter for all tabs):
year_range = st.sidebar.slider(
    "Award year range (pre-filter for all tabs)",
    min_value=year_min, max_value=year_max, value=(year_min, year_max),
)

# Conversion tab pools:
phase2_pool = df_full[
    (df_full["phase"].str.strip() == "Phase II") &
    (df_full["award_year"] >= year_range[0])  # lower-bounded only
]
phase1_overall = df_full[
    (df_full["phase"].str.strip() == "Phase I") &
    (df_full["award_year"].between(year_range[0], year_range[1]))
]

# session_state keys in use:
# "search_results"  → pd.DataFrame from last search
# "search_mode"     → "keyword" | "embeddings" | "llm"
# "conv_overall"    → dict from find_conversions(phase1_overall, phase2_pool)
# "conv_filtered"   → dict from find_conversions(phase1_similar, phase2_pool)

# Compact number formatter:
def _fmt(n):
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >= 1_000: return f"{n/1_000:.1f}K"
    return str(n)
```

---

## File Structure
```
SBIRAnalysis/
├── app.py                          # Main Streamlit UI — three tabs
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # CSV loader + column normalization
│   ├── similarity.py               # keyword / embeddings / Groq LLM engines
│   └── conversion.py               # Phase I→II matching (optimized, no iterrows)
├── requirements.txt                # groq (not anthropic or google-generativeai)
├── README.md                       # Project overview + setup instructions
├── CLAUDE.md                       # This file — auto-loaded by Claude Code each session
├── .streamlit/
│   ├── config.toml                 # Blue theme (#2563EB)
│   ├── secrets.toml                # ADMIN_PASSWORD + GROQ_API_KEY (gitignored) ⚠️ KEY NEEDS ROTATION
│   └── secrets.toml.example        # Template
├── .gitignore
├── data/
│   └── README.md                   # Instructions for obtaining award_data.csv
└── award_data.csv                  # 351MB, gitignored
```

## Environment
- **Python env**: `.venv/` in project root
- **Run app**: `.venv/bin/streamlit run app.py`
- **GitHub**: https://github.com/mahonymade/sbir-grant-analyzer (mahonymade account)
- **Packages**: streamlit 1.57, pandas, rapidfuzz, numpy, sentence-transformers 5.4.1, torch, groq
- **secrets.toml**: `ADMIN_PASSWORD = "SBIRAdmin"` and `GROQ_API_KEY` (needs rotation)
