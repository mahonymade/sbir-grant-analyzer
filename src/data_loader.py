"""
Data loading abstraction for SBIR grant data.

Currently supports CSV; the API stub is ready to fill in once the
SBIR API is restored (https://www.sbir.gov/api).
"""

import pandas as pd
import streamlit as st

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


def load_data(source: str = "csv", path: str = "award_data.csv", api_params: dict | None = None) -> pd.DataFrame:
    if source == "csv":
        return load_from_csv(path)
    elif source == "api":
        return load_from_api(api_params or {})
    else:
        raise ValueError(f"Unknown source: {source!r}. Use 'csv' or 'api'.")
