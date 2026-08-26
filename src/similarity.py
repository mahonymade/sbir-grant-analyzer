"""
Three similarity engines for matching SBIR grants to a project description.

Modes
-----
keyword   : Boolean keyword search — fast, no external dependencies.
embeddings: Cosine similarity via sentence-transformers (local model, ~80 MB).
llm       : Groq API scoring — admin-only, incurs API cost.
"""

from __future__ import annotations

import json
import re
import time

import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Keyword mode
# ---------------------------------------------------------------------------

def filter_by_keywords(
    df: pd.DataFrame,
    keywords: list[str],
    match_mode: str = "any",  # "any" | "all"
) -> pd.DataFrame:
    """Return rows whose combined title+abstract contain the keywords.

    Adds two sortable relevance columns, since a boolean mask has no inherent
    ranking — without them the most relevant grants are scattered arbitrarily
    through the result set:

    ``relevance``     how many of the distinct keywords the grant matched. The
                      primary signal in "any" mode; constant in "all" mode,
                      where every row matches every keyword by definition.
    ``keyword_hits``  total occurrences of all keywords. The differentiator in
                      "all" mode, and a tiebreaker in "any" mode.

    Results are sorted **newest award first**. That was previously the incidental
    order of the source file rather than a guarantee; sorting explicitly means a
    re-ordered upstream export cannot silently change what users see.
    """
    if not keywords:
        return df

    clean = [kw.strip().lower() for kw in keywords if kw.strip()]
    if not clean:
        return df

    text = df["combined_text_lc"]
    patterns = [re.escape(k) for k in clean]

    if match_mode == "any":
        mask = text.str.contains("|".join(patterns), regex=True)
    else:
        mask = pd.Series(True, index=df.index)
        for pattern in patterns:
            mask &= text.str.contains(pattern, regex=True)

    result = df[mask].copy()

    if result.empty:
        result["relevance"] = pd.Series(dtype="int64")
        result["keyword_hits"] = pd.Series(dtype="int64")
        return result

    # Score only the matched subset — counting over the full corpus would be
    # far more expensive for no benefit.
    matched_text = result["combined_text_lc"]
    counts = pd.DataFrame(
        {kw: matched_text.str.count(pattern) for kw, pattern in zip(clean, patterns)},
        index=result.index,
    )
    result["relevance"] = (counts > 0).sum(axis=1).astype("int64")
    result["keyword_hits"] = counts.sum(axis=1).astype("int64")

    if "award_year" in result.columns:
        # stable: ties within a year keep source order rather than reshuffling
        result = result.sort_values("award_year", ascending=False, kind="stable")
    return result


# ---------------------------------------------------------------------------
# Embeddings mode
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model (first run only)…")
def _load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_data(show_spinner="Computing embeddings for all abstracts…")
def _compute_corpus_embeddings(texts: tuple[str, ...]) -> np.ndarray:
    model = _load_embedding_model()
    return model.encode(list(texts), show_progress_bar=False, batch_size=128)


def _corpus_embeddings_for(df: pd.DataFrame) -> np.ndarray:
    """Return embeddings aligned to ``df``'s rows.

    Fast path: slice the precomputed, mmap'd corpus embeddings by ``_row_id``
    (no encoding at runtime). Fallback: encode ``df`` on the fly — used in local
    CSV-only dev where no precomputed artifact exists.
    """
    from src.data_loader import load_corpus_embeddings

    corpus = load_corpus_embeddings()
    if "_row_id" in df.columns:
        # Hosted-artifacts path: the grant table came from grants.parquet, so the
        # matching embeddings artifact MUST be present. Refuse the on-the-fly
        # fallback here — encoding the corpus on a CPU-only server would appear
        # to hang for hours and could exhaust memory.
        if corpus is None:
            raise RuntimeError(
                f"Precomputed embeddings could not be loaded (Hugging Face Hub "
                f"download may have failed). Refusing to encode {len(df):,} "
                f"abstracts on the fly. Restart the app to retry the download, "
                f"or rebuild artifacts with `python scripts/build_artifacts.py`."
            )
        ids = df["_row_id"].to_numpy()
        if len(ids) and ids.max() >= corpus.shape[0]:
            raise RuntimeError(
                f"Artifacts out of sync: grants.parquet references _row_id "
                f"{ids.max()} but embeddings.npy has only {corpus.shape[0]} rows. "
                f"Rebuild both with `python scripts/build_artifacts.py`."
            )
        # mmap'd float16 → materialize the slice as float32 for the dot product
        return np.asarray(corpus[ids], dtype=np.float32)

    # Local CSV/dev mode (no _row_id): compute embeddings for this corpus on the
    # fly — the documented slow path.
    corpus_texts = tuple(df["combined_text_lc"].tolist())
    return _compute_corpus_embeddings(corpus_texts)


def filter_by_embeddings(
    df: pd.DataFrame,
    project_description: str,
    top_n: int = 50,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """Rank grants by cosine similarity to project_description."""
    model = _load_embedding_model()

    corpus_embeddings = _corpus_embeddings_for(df)

    # Normalize the query so cosine == dot product (corpus is pre-normalized;
    # the fallback path's vectors are normalized here too).
    query_embedding = model.encode(
        [project_description], normalize_embeddings=True
    ).astype(np.float32)

    norms_corpus = np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
    similarities = (corpus_embeddings @ query_embedding.T).squeeze() / (
        norms_corpus.squeeze() + 1e-9
    )

    result = df.copy()
    result["similarity_score"] = similarities

    result = result[result["similarity_score"] >= threshold]
    result = result.nlargest(top_n, "similarity_score")
    return result


# ---------------------------------------------------------------------------
# LLM mode (admin-only)
# ---------------------------------------------------------------------------

# Groq retires models on a rolling basis. llama3-8b-8192 went ~May 2025, and its
# replacement llama-3.1-8b-instant was gone by Aug 2026 (the whole Llama family
# disappeared from the catalogue). Verify against client.models.list() before
# assuming this constant is still live.
GROQ_MODEL = "openai/gpt-oss-20b"
BATCH_SIZE = 50  # abstracts per API call

# gpt-oss is a *reasoning* model: it spends tokens thinking before emitting the
# answer, so the old 512 ceiling (sized for a non-reasoning 8B) truncated the
# JSON mid-object and Groq rejected it with json_validate_failed. A 30-grant
# batch uses ~1,130 completion tokens in practice.
#
# But max_tokens cannot simply be set high: Groq charges TPM on the *reservation*
# (prompt + max_tokens), not on actual usage. A 30-grant prompt is ~3,930 tokens,
# so max_tokens=4096 reserved 8,023 against an 8,000 limit and 413'd by 23 tokens
# despite the model only needing 1,130. The ceiling is therefore derived from the
# leftover budget at call time — see _completion_budget().
GROQ_TPM_LIMIT = 8000  # free tier, per minute, input + output combined
TPM_SAFETY_MARGIN = 250  # absorbs error in the prompt-token estimate
MAX_COMPLETION_TOKENS = 3000
MIN_COMPLETION_TOKENS = 1200  # below this, a 30-entry JSON risks truncation

# Measured on a real 30-grant batch (gpt-oss-20b): in=3,675 out=1,130.
AVG_ABSTRACT_TOKENS = 122
AVG_OUTPUT_TOKENS_PER_GRANT = 38
# Re-check against https://groq.com/pricing — these are carried over from the
# previous model and are not verified for gpt-oss-20b. Free tier bills nothing;
# the figure is shown only to signal relative cost.
COST_PER_1K_INPUT_TOKENS = 0.00005
COST_PER_1K_OUTPUT_TOKENS = 0.00008


def estimate_llm_cost(n_rows: int) -> float:
    """Rough USD cost estimate for scoring n_rows abstracts."""
    n_batches = max(1, -(-n_rows // BATCH_SIZE))  # ceiling division
    input_tokens = n_rows * AVG_ABSTRACT_TOKENS + n_batches * 300  # prompt overhead
    output_tokens = n_rows * AVG_OUTPUT_TOKENS_PER_GRANT
    return (input_tokens / 1000 * COST_PER_1K_INPUT_TOKENS +
            output_tokens / 1000 * COST_PER_1K_OUTPUT_TOKENS)


def _estimate_prompt_tokens(prompt: str) -> int:
    """Rough token count for budgeting.

    chars/4 lands ~23% *above* the true count for these prompts (measured: 4,843
    estimated vs 3,927 actual). Over-estimating is the safe direction — it makes
    the reservation smaller than it needs to be rather than tipping over the TPM
    limit.
    """
    return len(prompt) // 4


def _completion_budget(prompt: str, n_items: int) -> int:
    """max_tokens that keeps prompt + max_tokens inside the TPM limit.

    Scaled to the batch: reserving a flat ceiling for a small batch wastes TPM
    that the pacing loop then has to wait out. ``MIN_COMPLETION_TOKENS`` covers
    the model's reasoning preamble, on top of the measured per-grant output.
    """
    needed = MIN_COMPLETION_TOKENS + n_items * AVG_OUTPUT_TOKENS_PER_GRANT
    available = GROQ_TPM_LIMIT - TPM_SAFETY_MARGIN - _estimate_prompt_tokens(prompt)
    return min(MAX_COMPLETION_TOKENS, needed, available)


def _build_scoring_prompt(project_description: str, batch: list[dict]) -> str:
    items = "\n".join(
        f'{i+1}. TITLE: {r["title"]}\nABSTRACT: {r["abstract"][:600]}'
        for i, r in enumerate(batch)
    )
    return f"""You are evaluating SBIR grant relevance for a research project.

PROJECT DESCRIPTION:
{project_description}

Rate each grant below on a scale of 0–10 for relevance to the project description.
- 0 = completely unrelated
- 5 = tangentially related (same general field)
- 10 = highly relevant (directly addresses the same problem/technology)

Return a JSON object with a single key "scores", whose value is an array of objects
with keys "index" (1-based) and "score" (integer 0-10). Score every grant listed.
Do not wrap the JSON in markdown or code fences.
Example: {{"scores": [{{"index": 1, "score": 7}}, {{"index": 2, "score": 2}}]}}

GRANTS TO SCORE:
{items}"""


def filter_by_llm(
    df: pd.DataFrame,
    project_description: str,
    api_key: str,
    min_score: int = 5,
) -> pd.DataFrame:
    """Score grants using Groq and return those above min_score."""
    from groq import Groq

    client = Groq(api_key=api_key)

    rows = df[["award_title", "abstract"]].fillna("").to_dict("records")
    all_scores: dict[int, int] = {}

    progress = st.progress(0, text="Scoring with Groq…")
    n_batches = max(1, (len(rows) + BATCH_SIZE - 1) // BATCH_SIZE)

    # TPM is a rolling per-minute budget spent on prompt + max_tokens. Track what
    # this run has reserved so a multi-batch job waits for the window to roll
    # instead of 413-ing partway through.
    window_start = time.monotonic()
    reserved_this_window = 0

    for batch_idx in range(0, len(rows), BATCH_SIZE):
        batch_rows = rows[batch_idx : batch_idx + BATCH_SIZE]
        batch_data = [
            {"title": r["award_title"], "abstract": r["abstract"]}
            for r in batch_rows
        ]
        prompt = _build_scoring_prompt(project_description, batch_data)

        max_completion = _completion_budget(prompt, len(batch_data))
        if max_completion < MIN_COMPLETION_TOKENS:
            st.error(
                f"This batch needs about {_estimate_prompt_tokens(prompt):,} prompt "
                f"tokens, leaving too little of the {GROQ_TPM_LIMIT:,}/min budget for "
                f"the reply. Shorten the project description, or score fewer grants."
            )
            break

        reservation = _estimate_prompt_tokens(prompt) + max_completion
        elapsed = time.monotonic() - window_start
        if elapsed >= 60:
            window_start, reserved_this_window = time.monotonic(), 0
        elif reserved_this_window + reservation > GROQ_TPM_LIMIT - TPM_SAFETY_MARGIN:
            wait = 60 - elapsed
            progress.progress(
                min(1.0, batch_idx / max(1, len(rows))),
                text=f"Token budget reached — waiting {wait:.0f}s for the limit to reset…",
            )
            time.sleep(wait)
            window_start, reserved_this_window = time.monotonic(), 0
        reserved_this_window += reservation

        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_completion,
                response_format={"type": "json_object"},
            )
        except Exception as api_err:
            code = getattr(api_err, "status_code", None)
            if code in (413, 429):
                st.error(
                    f"Groq rate limit hit ({code}). The free tier allows "
                    f"{GROQ_TPM_LIMIT:,} tokens/min. Wait a minute and retry, score "
                    f"fewer grants, or upgrade at console.groq.com/settings/billing.\n\n{api_err}"
                )
            else:
                st.error(f"Groq API error: {api_err}")
            break

        try:
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            scores = json.loads(raw)
            # Groq JSON mode may return {"results": [...]} or a bare array
            if isinstance(scores, dict):
                scores = next(iter(scores.values()))
            for entry in scores:
                # Validate each entry independently so one malformed item
                # (e.g. "score": "high") doesn't crash the app or sink the batch.
                try:
                    local_idx = int(entry["index"])
                    score = int(entry["score"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not 1 <= local_idx <= len(batch_rows):
                    continue  # hallucinated index — ignore
                all_scores[batch_idx + local_idx - 1] = max(0, min(10, score))
        except (json.JSONDecodeError, KeyError, IndexError, StopIteration,
                TypeError, ValueError):
            pass

        pct = min(1.0, (batch_idx + BATCH_SIZE) / len(rows))
        progress.progress(pct, text=f"Scoring batch {batch_idx // BATCH_SIZE + 1}/{n_batches}…")

        # Pace requests to stay under Groq free tier (30 RPM)
        if batch_idx + BATCH_SIZE < len(rows):
            time.sleep(2)

    progress.empty()

    result = df.copy()
    # Unscored rows (API failure mid-run or unparseable model output) become NaN
    # and are excluded below — NOT treated as a legitimate score of 0.
    result["llm_score"] = [all_scores.get(i) for i in range(len(df))]
    n_scored = int(result["llm_score"].notna().sum())
    if n_scored < len(result):
        st.warning(
            f"Scored {n_scored} of {len(result)} grants — results are partial. "
            f"Unscored grants are excluded."
        )
    result = result[result["llm_score"] >= min_score]
    result = result.sort_values("llm_score", ascending=False)
    return result
