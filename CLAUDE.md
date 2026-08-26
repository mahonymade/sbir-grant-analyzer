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
- **Web artifacts (primary load path)**: app loads slim precomputed artifacts, NOT the raw CSV at runtime.
  - `data/grants.parquet` (~81 MB, zstd, 201,204 rows after dropping empty title+abstract, 8 slim cols + `_row_id`)
  - `data/embeddings.npy` (~147 MB, float16, shape (201204, 384), `all-MiniLM-L6-v2`, **normalized at build** → cosine = dot product)
  - `data/meta.json` (model/n_rows/dim/dtype/built — runtime validates `n_rows` vs embeddings shape)
  - Built by `scripts/build_artifacts.py` (MPS/CUDA auto). Hosted on **HF Hub dataset repo** (default `mahonymade/sbir-grant-analyzer-data`, override via `SBIR_DATA_REPO` env/secret). Downloaded once onto SERVER via `hf_hub_download`; browser visitor downloads nothing. Both big artifacts gitignored.
  - **Row alignment**: `_row_id` (0..N-1) in parquet maps to row i of embeddings. Runtime slices `embeddings[df["_row_id"]]`, robust to any filter/sort. Column starts with `_` so it's hidden from display/download (which strip `_`-prefixed cols).
  - **Artifact resolution is local-first**: `_resolve_artifact()` uses `data/<file>` if present (post-build, no download), else `hf_hub_download`. Lets a local build be tested before upload.
- **Raw CSV (dev/fallback)**: `award_data.csv` (351 MB, 207,731 rows, 41 columns) in project root. Gitignored. Source: https://www.sbir.gov/data-resources. Selected via sidebar "Local CSV". In CSV mode there's no precomputed embeddings → `filter_by_embeddings` falls back to encoding the filtered corpus on the fly (slow path).
- **Column naming**: All normalized to `lowercase_with_underscores` at load time.
- **Keyword search ordering**: `filter_by_keywords` sorts **newest award first** explicitly (`sort_values("award_year", ascending=False, kind="stable")`). Previously it was an unsorted boolean mask, so the order was the incidental order of the source parquet (which happens to be monotonic decreasing by year) — a re-ordered upstream export would have silently changed the display. It also adds two sortable relevance columns: `relevance` (distinct keywords matched; the signal in "any" mode) and `keyword_hits` (total occurrences; the differentiator in "all" mode, where `relevance` is constant). Scoring runs on the matched subset only (~0.6s for 31K rows). A UI caption tells users the default sort and points them at the two columns.
- **Phase encoding**: `"Phase I"` / `"Phase II"` (string with space + Roman numeral). Always `.str.strip()` before comparing.
- **Phase II pool**: Bounded from sidebar `year_range[0]` upward — Phase II awards before the earliest sidebar year are excluded (efficiency). Upper bound is NOT applied to Phase II (a Phase I from 2023 may convert in 2025+).
- **Conversion matching invariants** (enforced in `find_conversions`, verified on the full corpus):
  - **Chronology** — a Phase II award never counts for a Phase I awarded after it. (Was 5.6% of matches before the fix.)
  - **One-to-one** — each Phase II row is consumed by at most one Phase I. Within a company, Phase I awards are processed oldest-first; each takes the highest-scoring *available* Phase II, ties broken toward the earliest. Verified: 45,829 matches, 45,829 distinct Phase II rows.
- **Right-censoring is the biggest methodology trap in this dataset.** Recent Phase I cohorts have not had time to convert, so including them deflates the rate badly. Measured cohort rates: 2016 **33.9%**, 2019 **31.4%**, 2021 **23.3%**, 2023 **2.4%** — the plateau is ~33%, everything after ~2019 is censoring, not signal. `find_conversions(maturity_years=5)` therefore reports `conversion_rate` over mature cohorts only (Phase I year ≤ newest Phase II year − 5) and `conversion_rate_raw` for the uncensored comparison. Full-corpus headline: **34.7% mature vs 32.6% raw**. The by-year chart splits into "Mature cohorts" / "Too recent to judge" so the tail is not misread as a real decline.
- **Sidebar year range is the single date pre-filter** for both tabs. The redundant per-tab Phase I year slider was removed. `year_range` from the sidebar is used everywhere.
- **`find_conversions()` has NO `@st.cache_data`** — results manually cached in `st.session_state` on button click (Streamlit's cache ignores `_`-prefixed DataFrame params).
- **LLM mode is admin-only**, gated by `ADMIN_PASSWORD` in `.streamlit/secrets.toml` (gitignored).
- **LLM mode requires embeddings pre-filter** — user must run Embeddings search first; LLM scores only those results. A hard cap of `LLM_MAX_GRANTS = 30` is applied (top 30 of embeddings results) to stay under Groq free tier TPM limit.
- **Batch size**: 50 abstracts per API call. 2s sleep between batches (Groq: 30 RPM limit).
- **`award_amount`** stored as comma-formatted strings in CSV — loader strips commas before `pd.to_numeric()`.
- **Year axis labels** cast to `str` to prevent Streamlit adding comma thousand-separators (e.g. "2,026").
- **Large counts** use compact K/M format via `_fmt()`.
- **Git**: user commits and pushes manually. Latest push: commit `84a97ef` (2026-05-18).

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
| `find_conversions` works on positional indices + `rapidfuzz.process.cdist` | The old version built `p2_rows_by_company` — every Phase II row as a Python dict including its abstract — which cost **21s and 406 MB** on its own, then called `token_sort_ratio` per pair from Python. Now it groups positional indices per company and lets rapidfuzz do each company block in C. Full corpus: **4.4 min → 8.6s, 406 MB → 70 MB peak.** |
| `_add_search_columns` keeps only `combined_text_lc` | `abstract_lc`/`title_lc` were never read again after being concatenated, but retained **297 MB** on the full corpus (`abstract_lc` duplicates every abstract). Load+search RAM: 913 MB → 616 MB. The concatenation is byte-identical, so existing embeddings artifacts stay valid. |
| `_to_csv` is `@st.cache_data`-wrapped | `st.download_button` needs its payload up front, so the old inline `to_csv` rebuilt a 37 MB string on **every rerun** for a 20k-row result — the "generates CSV only when clicked" comment was false. |
| Groq JSON dict unwrapping | Groq JSON mode returns `{"scores": [...]}` (confirmed) — code unwraps via `isinstance(scores, dict)` → `next(iter(scores.values()))`. |
| Precompute embeddings → ship as artifacts (not runtime encoding) | Old path re-encoded the whole filtered corpus on every search and re-hashed a 200K-string cache key on every filter change. Now corpus is encoded once at build; runtime only encodes the 1-sentence query and slices precomputed embeddings by `_row_id`. Multi-minute → sub-second. |
| zstd (level 10) for grants.parquet | Long abstract text compresses ~47% better than snappy (159 MB → 81 MB) for negligible read cost. |
| float16 embeddings | Halves the embeddings artifact (320 MB → ~147 MB). Materialized to float32 per-slice at query time for the dot product. |
| CPU-only torch wheel in requirements.txt | `--extra-index-url .../whl/cpu` → ~200 MB torch on the Linux Space instead of ~2 GB CUDA. Runtime only embeds one query sentence; no GPU needed. |
| Deploy target = HF Spaces + HF Hub dataset repo | Free tier 16 GB RAM fits model + mmap'd embeddings + parquet. Artifacts hosted on HF Hub dataset repo, fetched server-side. API path not used (SBIR API unavailable as of 2026-06). |

---

## Open Issues

### 1. LLM Mode Free Tier TPM — Resolved ✓
Hard cap at `LLM_MAX_GRANTS = 30` (≈4500 tokens) confirmed working end-to-end. If hitting limits in future: add a user-facing slider (Option A), use smaller batches (Option B), or upgrade to Groq Dev tier (Option C).

### 2. Groq key rotated + model migrated — RESOLVED ✓ (2026-08-26)
Key rotated by the user; verified working. The old key had **expired** (`401 expired_api_key`), which is why LLM mode broke. It was never exposed via git — `.streamlit/secrets.toml` has never been tracked in any commit and no `gsk_` literal appears in any blob in full history, so the public GitHub repo never leaked it.

Three things were wrong at once, all fixed:
1. **Expired key** → rotated (local `secrets.toml` + HF Space secret; the Space injects it as an env var, not into `st.secrets`).
2. **`llama-3.1-8b-instant` decommissioned** — the entire Llama family is gone from Groq's catalogue as of Aug 2026. Now `openai/gpt-oss-20b`.
3. **`max_tokens=512` truncated the new model** → `400 json_validate_failed`. gpt-oss is a *reasoning* model that spends tokens thinking before answering; 512 was sized for a non-reasoning 8B. Now `MAX_COMPLETION_TOKENS = 4096`.

A fourth latent bug surfaced during migration: `_build_scoring_prompt` asked for a bare JSON **array**, but Groq's `json_object` mode requires a top-level **object**. The old Llama wrapped it in `{"scores": [...]}` anyway — which is why the parser has the `isinstance(scores, dict)` unwrap — but newer models obey the prompt literally and get rejected. The prompt now requests the object form explicitly.

**Measured** (`openai/gpt-oss-20b`, 30-grant batch): 1.8s, in=3,675 out=1,130, total 4,805 tokens = 60% of the 8,000 TPM free-tier budget. `filter_by_llm` end-to-end: 30 grants → 12 scoring ≥5 in 1.9s. Rate limits: **8,000 TPM**, 1,000 req/day.

To re-check the model list when this breaks again:
```
.venv/bin/python -c "import tomllib;from groq import Groq;k=tomllib.load(open('.streamlit/secrets.toml','rb'))['GROQ_API_KEY'];print(sorted(m.id for m in Groq(api_key=k).models.list().data))"
```
Candidates tested: `groq/compound-mini` also works (30/30 parsed) but is ~3.5x slower (6.3s) and is an agentic system rather than a plain LLM. `qwen/qwen3.6-27b` failed JSON validation even at higher token ceilings.

### 3. Artifacts omit two useful columns (optional, needs rebuild + re-upload)
`SLIM_COLS` keeps 8 of 41 raw columns. Two dropped ones would strengthen the conversion analysis:
- **`Contract End Date`** (46.4% populated) — the correct anchor for conversion timing; NIH's rule is "within two years of the Phase 1 *end* date", not award year.
- **`Duns`** (60.6% populated; 22,219 distinct vs 32,401 distinct company strings) — confirms company identity independently of name spelling, catching renames and normalization misses.

Neither is populated well enough to *replace* the current name+title matching, but both are cheap supplementary signals (~2–3 MB added to the artifact). Requires re-running `scripts/build_artifacts.py --upload` (~30 min encode) — embeddings would be unchanged, so in principle only the parquet needs rebuilding.

### 4. Float16 embedding precision — measured, not a concern
Re-encoded a 6,000-row sample at fp32 and compared: max per-grant similarity error **4.0e-05** against a score range of −0.232…0.531; **top-10/50/200 set overlap 100%**. Only the ordering of near-ties shifts. This holds because MiniLM output is L2-normalized *before* the cast (components land in ~[−0.26, 0.26], fp16's sweet spot) and the runtime re-divides by actual norms, which partly corrects the rounding. **If the embedding model ever changes, or pre-normalization is dropped, re-verify.**

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
├── scripts/
│   └── build_artifacts.py          # Build (MPS/CUDA) + --upload slim parquet/embeddings/meta to HF Hub
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # parquet+CSV loaders, HF Hub artifact fetch, mmap embeddings loader
│   ├── similarity.py               # keyword / embeddings (precomputed) / Groq LLM engines
│   └── conversion.py               # Phase I→II matching (optimized, no iterrows)
├── requirements.txt                # CPU-torch index; +pyarrow, +huggingface_hub; groq (not anthropic/google)
├── README.md                       # Project overview + setup + HF Spaces deploy
├── CLAUDE.md                       # This file — auto-loaded by Claude Code each session
├── .streamlit/
│   ├── config.toml                 # Blue theme (#2563EB)
│   ├── secrets.toml                # ADMIN_PASSWORD + GROQ_API_KEY (+ optional SBIR_DATA_REPO), gitignored ⚠️ KEY NEEDS ROTATION
│   └── secrets.toml.example        # Template
├── .gitignore                      # also ignores data/grants.parquet + data/embeddings.npy
├── data/
│   ├── README.md                   # Artifact spec + build instructions
│   ├── grants.parquet              # ~81MB slim table, gitignored (hosted on HF Hub)
│   ├── embeddings.npy              # ~147MB float16, gitignored (hosted on HF Hub)
│   └── meta.json                   # tiny — committed-safe build metadata
└── award_data.csv                  # 351MB raw, gitignored (build input only)
```

## Environment
- **Python env**: `.venv/` in project root
- **Run app**: `.venv/bin/streamlit run app.py`
- **Build artifacts**: `.venv/bin/python scripts/build_artifacts.py` (add `--upload --repo <user>/<repo>` to push to HF Hub)
- **GitHub**: https://github.com/mahonymade/sbir-grant-analyzer (mahonymade account)
- **HF Hub dataset repo**: `mahonymade/sbir-grant-analyzer-data` (public; default, override via `SBIR_DATA_REPO`) — artifacts UPLOADED 2026-06-25 (grants.parquet 84.7 MB, embeddings.npy 154.5 MB, meta.json).
- **HF Space (LIVE)**: https://huggingface.co/spaces/mahonymade/sbir-grant-analyzer (Streamlit SDK, free CPU basic). `git remote space` → the Space repo. Deployed via `git push space main`. README YAML frontmatter (sdk: streamlit, app_file: app.py) must be the FIRST line of README.md or HF warns "empty/missing yaml metadata".
- **HF Spaces secrets are ENV VARS, not `st.secrets`** — Space "Variables and secrets" are injected as `os.environ`, NOT written to `.streamlit/secrets.toml`. `app.py:_get_secret()` reads `st.secrets` first (local) then `os.environ` (Space) for `ADMIN_PASSWORD` / `GROQ_API_KEY`. Set both in Space settings for LLM mode to work live.
- **zsh gotcha**: interactive zsh does NOT treat `#` as a comment by default — don't paste commands with trailing `# comments` into the terminal (they become git refspecs and error).
- **Packages**: streamlit 1.57, pandas, rapidfuzz, numpy, pyarrow, huggingface_hub, sentence-transformers 5.4.1, torch (CPU wheel on deploy), groq
- **secrets.toml**: `ADMIN_PASSWORD = "SBIRAdmin"` and `GROQ_API_KEY` (needs rotation)
