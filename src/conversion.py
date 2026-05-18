"""
Phase I → Phase II conversion analysis.

Matching strategy: same company name + fuzzy title match via rapidfuzz.
This handles the common case where Phase II titles differ slightly from Phase I.
"""

from __future__ import annotations

import pandas as pd
from rapidfuzz import fuzz


def _normalize_company(name: str) -> str:
    """Lowercase and strip common legal suffixes for better matching."""
    name = str(name).lower().strip()
    for suffix in (", inc.", ", inc", ", llc.", ", llc", ", corp.", ", corp",
                   " inc.", " inc", " llc", " corp", ", ltd.", " ltd"):
        name = name.replace(suffix, "")
    return name.strip(" ,.")


def find_conversions(
    _phase1_df: pd.DataFrame,
    _phase2_df: pd.DataFrame,
    fuzzy_threshold: int = 85,
) -> dict:
    """
    Match Phase I grants to Phase II grants by company + title similarity.

    Parameters
    ----------
    _phase1_df : Phase I grants to check for conversion (caller controls year range / similarity filter).
    _phase2_df : Phase II grant pool to match against — typically the full dataset, all years.
    fuzzy_threshold : minimum rapidfuzz token_sort_ratio score (0–100) to count as a match.

    Returns a dict with:
        matched_pairs   : DataFrame of matched Phase I / II rows
        conversion_rate : float (0–1)
        by_agency       : Series — conversion rate per agency
        by_year         : Series — Phase I award year → conversion rate
        phase1_count    : int
        phase2_count    : int
        matched_count   : int
    """
    phase1 = _phase1_df.copy()
    phase2 = _phase2_df.copy()

    phase1["_company_norm"] = phase1["company"].fillna("").apply(_normalize_company)
    phase1["_title_lc"] = phase1["award_title"].fillna("").str.lower()
    phase2["_company_norm"] = phase2["company"].fillna("").apply(_normalize_company)
    phase2["_title_lc"] = phase2["award_title"].fillna("").str.lower()

    # Build lookup dicts via groupby (much faster than iterrows)
    p2_by_company: dict[str, list[str]] = (
        phase2.groupby("_company_norm")["_title_lc"].apply(list).to_dict()
    )
    p2_rows_by_company: dict[str, list[dict]] = (
        phase2.groupby("_company_norm", group_keys=False)
        .apply(lambda g: g.to_dict("records"))
        .to_dict()
    )

    matched_p1_indices = []
    matched_p2_rows = []

    # Iterate using .values arrays — avoids per-row pandas overhead of iterrows()
    p1_companies = phase1["_company_norm"].values
    p1_titles = phase1["_title_lc"].values
    p1_indices = phase1.index.to_numpy()

    for co, title, idx in zip(p1_companies, p1_titles, p1_indices):
        if co not in p2_by_company:
            continue
        p2_titles = p2_by_company[co]
        scores = [fuzz.token_sort_ratio(title, pt) for pt in p2_titles]
        best = max(range(len(scores)), key=scores.__getitem__)
        if scores[best] >= fuzzy_threshold:
            matched_p1_indices.append(idx)
            matched_p2_rows.append(p2_rows_by_company[co][best])

    # Build matched pairs table
    p1_matched = phase1.loc[matched_p1_indices].reset_index(drop=True)
    p2_matched = pd.DataFrame(matched_p2_rows).reset_index(drop=True)

    keep_cols = ["company", "award_title", "agency", "award_year", "award_amount", "abstract"]
    p1_display = p1_matched[[c for c in keep_cols if c in p1_matched.columns]].copy()
    p2_display = (
        p2_matched[[c for c in keep_cols if c in p2_matched.columns]].copy()
        if not p2_matched.empty else pd.DataFrame(columns=keep_cols)
    )

    p1_display.columns = [f"phase1_{c}" for c in p1_display.columns]
    p2_display.columns = [f"phase2_{c}" for c in p2_display.columns]

    matched_pairs = pd.concat([p1_display, p2_display], axis=1)

    conversion_rate = len(matched_p1_indices) / max(len(phase1), 1)

    # Per-agency conversion rate
    agency_p1 = phase1.groupby("agency").size().rename("p1_count")
    agency_converted = (
        phase1.loc[matched_p1_indices].groupby("agency").size().rename("converted")
        if matched_p1_indices else pd.Series(dtype=int, name="converted")
    )
    agency_stats = pd.concat([agency_p1, agency_converted], axis=1).fillna(0)
    agency_stats["rate"] = agency_stats["converted"] / agency_stats["p1_count"]
    by_agency = agency_stats["rate"].sort_values(ascending=False)

    # Per-year conversion rate
    year_p1 = phase1.groupby("award_year").size().rename("p1_count")
    year_converted = (
        phase1.loc[matched_p1_indices].groupby("award_year").size().rename("converted")
        if matched_p1_indices else pd.Series(dtype=int, name="converted")
    )
    year_stats = pd.concat([year_p1, year_converted], axis=1).fillna(0)
    year_stats["rate"] = year_stats["converted"] / year_stats["p1_count"]
    by_year = year_stats["rate"].sort_values(ascending=True)  # chronological

    return {
        "matched_pairs": matched_pairs,
        "conversion_rate": conversion_rate,
        "by_agency": by_agency,
        "by_year": by_year,
        "phase1_count": len(phase1),
        "phase2_count": len(phase2),
        "matched_count": len(matched_p1_indices),
    }
