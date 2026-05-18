"""
Three similarity engines for matching SBIR grants to a project description.

Modes
-----
keyword   : Boolean keyword search — fast, no external dependencies.
embeddings: Cosine similarity via sentence-transformers (local model, ~80 MB).
llm       : Claude API scoring — admin-only, incurs API cost.
"""

from __future__ import annotations

import json
import re

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
    """Return rows whose combined title+abstract contain the keywords."""
    if not keywords:
        return df

    clean = [kw.strip().lower() for kw in keywords if kw.strip()]
    if not clean:
        return df

    text = df["combined_text_lc"]

    if match_mode == "any":
        mask = text.str.contains("|".join(re.escape(k) for k in clean), regex=True)
    else:
        mask = pd.Series(True, index=df.index)
        for kw in clean:
            mask &= text.str.contains(re.escape(kw), regex=True)

    return df[mask].copy()


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


def filter_by_embeddings(
    df: pd.DataFrame,
    project_description: str,
    top_n: int = 50,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """Rank grants by cosine similarity to project_description."""
    model = _load_embedding_model()

    # Use combined title+abstract as corpus; cache keyed on the actual strings
    corpus_texts = tuple(df["combined_text_lc"].tolist())
    corpus_embeddings = _compute_corpus_embeddings(corpus_texts)

    query_embedding = model.encode([project_description])

    # Cosine similarity
    norms_corpus = np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
    norms_query = np.linalg.norm(query_embedding)
    similarities = (corpus_embeddings @ query_embedding.T).squeeze() / (
        norms_corpus.squeeze() * norms_query + 1e-9
    )

    result = df.copy()
    result["similarity_score"] = similarities

    result = result[result["similarity_score"] >= threshold]
    result = result.nlargest(top_n, "similarity_score")
    return result


# ---------------------------------------------------------------------------
# LLM mode (admin-only)
# ---------------------------------------------------------------------------

BATCH_SIZE = 20  # abstracts per API call
COST_PER_1K_INPUT_TOKENS = 0.003   # claude-sonnet-4-6 approximate
COST_PER_1K_OUTPUT_TOKENS = 0.015
AVG_ABSTRACT_TOKENS = 250


def estimate_llm_cost(n_rows: int) -> float:
    """Rough USD cost estimate for scoring n_rows abstracts."""
    n_batches = max(1, n_rows // BATCH_SIZE)
    input_tokens = n_rows * AVG_ABSTRACT_TOKENS + n_batches * 300  # prompt overhead
    output_tokens = n_rows * 10  # ~10 tokens per JSON score entry
    return (input_tokens / 1000 * COST_PER_1K_INPUT_TOKENS +
            output_tokens / 1000 * COST_PER_1K_OUTPUT_TOKENS)


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

Return ONLY a JSON array of objects with keys "index" (1-based) and "score" (integer 0-10).
Example: [{{"index": 1, "score": 7}}, {{"index": 2, "score": 2}}]

GRANTS TO SCORE:
{items}"""


def filter_by_llm(
    df: pd.DataFrame,
    project_description: str,
    api_key: str,
    min_score: int = 5,
) -> pd.DataFrame:
    """Score grants using Claude and return those above min_score."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    rows = df[["award_title", "abstract"]].fillna("").to_dict("records")
    all_scores: dict[int, int] = {}

    progress = st.progress(0, text="Scoring with Claude…")
    n_batches = max(1, (len(rows) + BATCH_SIZE - 1) // BATCH_SIZE)

    for batch_idx in range(0, len(rows), BATCH_SIZE):
        batch_rows = rows[batch_idx : batch_idx + BATCH_SIZE]
        batch_data = [
            {"title": r["award_title"], "abstract": r["abstract"]}
            for r in batch_rows
        ]
        prompt = _build_scoring_prompt(project_description, batch_data)

        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )

        try:
            scores = json.loads(message.content[0].text)
            for entry in scores:
                global_idx = batch_idx + entry["index"] - 1
                all_scores[global_idx] = int(entry["score"])
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

        pct = min(1.0, (batch_idx + BATCH_SIZE) / len(rows))
        progress.progress(pct, text=f"Scoring batch {batch_idx // BATCH_SIZE + 1}/{n_batches}…")

    progress.empty()

    result = df.copy()
    result["llm_score"] = [all_scores.get(i, 0) for i in range(len(df))]
    result = result[result["llm_score"] >= min_score]
    result = result.sort_values("llm_score", ascending=False)
    return result
