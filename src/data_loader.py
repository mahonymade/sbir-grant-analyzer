"""
Data loading abstraction for SBIR grant data.

Primary path is the slim, web-ready artifacts built by ``scripts/build_artifacts.py``
(a compressed Parquet table + precomputed float16 embeddings). They are hosted on a
Hugging Face Hub dataset repo and downloaded once onto the server at startup; the
browser visitor never downloads them. A local raw CSV is still supported for dev/build.

The API stub is ready to fill in once the SBIR API is restored (https://www.sbir.gov/api).
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Default Hugging Face Hub *dataset* repo holding the artifacts. Override with the
# SBIR_DATA_REPO env var or a `SBIR_DATA_REPO` entry in secrets.toml.
DEFAULT_DATA_REPO = "mahonymade/sbir-grant-analyzer-data"
ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "data"

DATE_COLS = [
    "proposal_award_date",
    "contract_end_date",
    "solicitation_close_date",
    "proposal_receipt_date",
    "date_of_notification",
]

DEFAULT_DISPLAY_COLS = [
    "company",
    "award_title",
    "phase",
    "agency",
    "award_year",
    "award_amount",
    "abstract",
]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.str.strip().str.replace(" ", "_").str.replace("-", "_").str.lower()
    )
    return df


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # award_amount is stored as comma-formatted strings (e.g. "171,433") — strip
    # commas before numeric conversion, otherwise pd.to_numeric coerces to NaN.
    for col in ("award_amount", "award_year", "number_employees"):
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].astype(str).str.replace(",", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _add_search_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["abstract_lc"] = df["abstract"].fillna("").str.lower()
    df["title_lc"] = df["award_title"].fillna("").str.lower()
    df["combined_text_lc"] = df["title_lc"] + " " + df["abstract_lc"]
    return df


@st.cache_data(show_spinner="Loading grant data…")
def load_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = _normalize_columns(df)
    df = _coerce_types(df)
    df = _add_search_columns(df)
    return df


# ---------------------------------------------------------------------------
# Web artifacts (slim Parquet + precomputed embeddings)
# ---------------------------------------------------------------------------

def _data_repo_id() -> str:
    repo = os.environ.get("SBIR_DATA_REPO")
    if repo:
        return repo
    try:
        return st.secrets["SBIR_DATA_REPO"]
    except Exception:
        return DEFAULT_DATA_REPO


def _resolve_artifact(filename: str) -> str:
    """Return a local path to an artifact, preferring a local build and falling
    back to downloading (and caching) it from the Hugging Face Hub dataset repo."""
    local = ARTIFACT_DIR / filename
    if local.exists():
        return str(local)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=_data_repo_id(), filename=filename, repo_type="dataset"
    )


@st.cache_data(show_spinner="Loading grant data…")
def load_from_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = _normalize_columns(df)  # parquet is pre-normalized; harmless + defensive
    df = _coerce_types(df)
    df = _add_search_columns(df)  # rebuild *_lc text columns (not stored in parquet)
    return df


def load_grants() -> pd.DataFrame:
    """Load the slim grant table from the web artifact (local or HF Hub)."""
    return load_from_parquet(_resolve_artifact("grants.parquet"))


@st.cache_resource(show_spinner="Loading precomputed embeddings…")
def load_corpus_embeddings():
    """Memory-mapped float16 corpus embeddings, indexed by ``_row_id``.

    Returns None if the artifact is unavailable (e.g. local CSV-only dev), in which
    case the embeddings search falls back to computing them on the fly.
    """
    try:
        emb = np.load(_resolve_artifact("embeddings.npy"), mmap_mode="r")
        try:
            meta = json.loads(Path(_resolve_artifact("meta.json")).read_text())
            if meta.get("n_rows") not in (None, emb.shape[0]):
                st.warning(
                    f"Embeddings row count ({emb.shape[0]}) does not match "
                    f"meta.json ({meta.get('n_rows')}). Artifacts may be out of sync."
                )
        except Exception:
            pass
        return emb
    except Exception:
        return None


def load_from_api(api_params: dict) -> pd.DataFrame:
    """
    Stub for future SBIR API integration.

    Expected API endpoint: https://api.sbir.gov/public/awards
    Typical query params:
        keyword (str): search term
        agency (str): agency abbreviation, e.g. "DOD", "NSF"
        phase (str): "1" or "2"
        rows (int): page size (max 5000)
        start (int): offset for pagination

    Example:
        import requests
        resp = requests.get(
            "https://api.sbir.gov/public/awards",
            params={**api_params, "output": "json"},
            timeout=30,
        )
        resp.raise_for_status()
        records = resp.json()["response"]["docs"]
        df = pd.DataFrame(records)
        df = _normalize_columns(df)
        df = _coerce_types(df)
        df = _add_search_columns(df)
        return df
    """
    raise NotImplementedError(
        "SBIR API integration is not yet implemented. "
        "See the docstring in load_from_api() for the expected endpoint and params."
    )


def load_data(source: str = "parquet", path: str = "award_data.csv", api_params: dict | None = None) -> pd.DataFrame:
    if source == "parquet":
        return load_grants()
    elif source == "csv":
        return load_from_csv(path)
    elif source == "api":
        return load_from_api(api_params or {})
    else:
        raise ValueError(f"Unknown source: {source!r}. Use 'parquet', 'csv', or 'api'.")
