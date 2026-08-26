"""
Phase I → Phase II conversion analysis.

Matching strategy: same company name + fuzzy title match via rapidfuzz, subject to
two constraints that keep the resulting rate honest:

* **Chronology** — a Phase II award may not predate the Phase I award it converts.
* **One-to-one** — each Phase II award is consumed by at most one Phase I award, so
  a single follow-on cannot inflate the rate across several sibling Phase I grants.

Right-censoring
---------------
A Phase I award granted in the last few years of the data has not had time to
convert, so counting it in the denominator deflates the rate. Measured on the SBIR
corpus, cohorts plateau around 33% but fall off a cliff at the recent end
(2019: 32%, 2021: 22%, 2023: 3%) purely because the follow-on has not happened yet.
``conversion_rate`` is therefore computed over *mature* cohorts only — Phase I years
at least ``maturity_years`` before the last observable Phase II award. The raw,
uncensored rate is still returned as ``conversion_rate_raw`` for comparison.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

# Legal suffixes stripped from the END of company names only. An end-anchored
# regex avoids the substring corruption of naive replace() (e.g. "Acme Corporate
# Solutions" must NOT become "acmeorate solutions"). Long forms included so
# "Acme Corp" and "Acme Corporation" normalize identically.
_COMPANY_SUFFIX_RE = re.compile(
    r"[,\s]+(incorporated|inc|llc|corporation|corp|limited|ltd)\.?$"
)

# Years a Phase I cohort needs before its conversion rate is trustworthy. Chosen
# from the observed cohort curve: rates are flat through ~5 years back from the
# data horizon and collapse after that as censoring takes over.
DEFAULT_MATURITY_YEARS = 5

_PAIR_COLS = ["company", "award_title", "agency", "award_year", "award_amount", "abstract"]


def _normalize_company(name: str) -> str:
    """Lowercase and strip trailing legal suffixes for better matching."""
    name = str(name).lower().strip()
    # Loop to handle stacked suffixes, e.g. "Acme Corp., Ltd."
    while True:
        stripped = _COMPANY_SUFFIX_RE.sub("", name)
        if stripped == name:
            break
        name = stripped
    return name.strip(" ,.")


def _positions_by_company(companies: np.ndarray) -> dict[str, list[int]]:
    """Map normalized company name → positional row indices."""
    out: dict[str, list[int]] = {}
    for pos, co in enumerate(companies):
        out.setdefault(co, []).append(pos)
    return out


def find_conversions(
    _phase1_df: pd.DataFrame,
    _phase2_df: pd.DataFrame,
    fuzzy_threshold: int = 85,
    maturity_years: int = DEFAULT_MATURITY_YEARS,
) -> dict:
    """
    Match Phase I grants to Phase II grants by company + title similarity.

    Parameters
    ----------
    _phase1_df : Phase I grants to check for conversion (caller controls year range / similarity filter).
    _phase2_df : Phase II grant pool to match against — typically the full dataset, all years.
    fuzzy_threshold : minimum rapidfuzz token_sort_ratio score (0–100) to count as a match.
    maturity_years : a Phase I cohort is excluded from the headline rate unless it is at
        least this many years older than the newest Phase II award in the pool.

    Returns a dict with:
        matched_pairs        : DataFrame of matched Phase I / II rows
        conversion_rate      : float (0–1) over mature cohorts only; NaN if none are mature
        conversion_rate_raw  : float (0–1) over every Phase I row, censoring included
        by_agency            : Series — conversion rate per agency (mature cohorts only)
        by_year              : DataFrame indexed by Phase I year, cols rate/p1_count/converted/mature
        phase1_count         : int — all Phase I rows considered
        phase2_count         : int — size of the Phase II pool
        matched_count        : int — matches found across all cohorts
        mature_phase1_count  : int — Phase I rows in mature cohorts
        mature_matched_count : int — matches within mature cohorts
        max_mature_year      : last Phase I year considered mature (NaN if undeterminable)
        censored_count       : int — Phase I rows excluded from the headline rate
    """
    phase1, phase2 = _phase1_df, _phase2_df
    n1, n2 = len(phase1), len(phase2)

    # astype(str) guards the empty-frame case, where pandas infers a float dtype
    # for the column and the .str accessor would raise.
    def _titles(df: pd.DataFrame) -> np.ndarray:
        return df["award_title"].fillna("").astype(str).str.lower().to_numpy()

    def _companies(df: pd.DataFrame) -> np.ndarray:
        return df["company"].fillna("").astype(str).map(_normalize_company).to_numpy()

    p1_company, p1_title = _companies(phase1), _titles(phase1)
    p1_year = pd.to_numeric(phase1["award_year"], errors="coerce").to_numpy(dtype=float)

    p2_company, p2_title = _companies(phase2), _titles(phase2)
    p2_year = pd.to_numeric(phase2["award_year"], errors="coerce").to_numpy(dtype=float)

    # ---- Matching ---------------------------------------------------------
    # Work company-by-company on positional indices only. Storing positions (not
    # row dicts) keeps this O(rows) in memory rather than materializing every
    # Phase II row — including its abstract — as a Python dict.
    p2_by_company = _positions_by_company(p2_company)
    p1_by_company: dict[str, list[int]] = {}
    for pos, co in enumerate(p1_company):
        if co in p2_by_company:
            p1_by_company.setdefault(co, []).append(pos)

    matched_p1_pos: list[int] = []
    matched_p2_pos: list[int] = []

    for co, p1_pos in p1_by_company.items():
        p2_pos = p2_by_company[co]
        # Earlier Phase I awards get first claim on a shared follow-on.
        p1_pos = sorted(p1_pos, key=lambda i: (np.isnan(p1_year[i]), p1_year[i]))

        # rapidfuzz computes the whole company block in C rather than one
        # token_sort_ratio call per pair from Python.
        scores = process.cdist(
            [p1_title[i] for i in p1_pos],
            [p2_title[j] for j in p2_pos],
            scorer=fuzz.token_sort_ratio,
            dtype=np.uint8,
        )
        cand_years = p2_year[p2_pos]
        available = np.ones(len(p2_pos), dtype=bool)

        for row, i in enumerate(p1_pos):
            row_scores = scores[row]
            # `~(cand_years < yi)` keeps NaN years (unknown → don't exclude) and
            # drops any Phase II that predates this Phase I.
            eligible = available & (row_scores >= fuzzy_threshold) & ~(cand_years < p1_year[i])
            cand = np.flatnonzero(eligible)
            if cand.size == 0:
                continue
            # Best title score wins (that is what identifies the same project);
            # ties break toward the earliest follow-on.
            order = np.lexsort((cand_years[cand], -row_scores[cand].astype(np.int16)))
            pick = int(cand[order[0]])
            available[pick] = False
            matched_p1_pos.append(i)
            matched_p2_pos.append(p2_pos[pick])

    matched_mask = np.zeros(n1, dtype=bool)
    matched_mask[matched_p1_pos] = True

    # ---- Maturity (right-censoring) ---------------------------------------
    # The newest Phase II award bounds what conversions are observable at all.
    observable_max = float(np.nanmax(p2_year)) if n2 and not np.isnan(p2_year).all() else np.nan
    max_mature_year = observable_max - maturity_years if not np.isnan(observable_max) else np.nan
    # NaN years compare False, so undated Phase I rows are treated as immature.
    mature_mask = (
        p1_year <= max_mature_year
        if not np.isnan(max_mature_year)
        else np.zeros(n1, dtype=bool)
    )

    n_mature = int(mature_mask.sum())
    n_mature_matched = int((mature_mask & matched_mask).sum())
    conversion_rate = n_mature_matched / n_mature if n_mature else float("nan")
    conversion_rate_raw = int(matched_mask.sum()) / n1 if n1 else 0.0

    # ---- Matched pairs table ----------------------------------------------
    p1_matched = phase1.iloc[matched_p1_pos]
    p2_matched = phase2.iloc[matched_p2_pos]
    p1_display = p1_matched[[c for c in _PAIR_COLS if c in p1_matched.columns]].reset_index(drop=True)
    p2_display = p2_matched[[c for c in _PAIR_COLS if c in p2_matched.columns]].reset_index(drop=True)
    p1_display.columns = [f"phase1_{c}" for c in p1_display.columns]
    p2_display.columns = [f"phase2_{c}" for c in p2_display.columns]
    matched_pairs = pd.concat([p1_display, p2_display], axis=1)

    # ---- Per-agency rate (mature cohorts only, to match the headline) ------
    # Index names are assigned explicitly: a groupby over an empty frame yields an
    # unnamed index, which previously turned into a KeyError downstream.
    agency_stats = pd.DataFrame(
        {
            "agency": phase1["agency"].to_numpy(),
            "converted": matched_mask,
            "mature": mature_mask,
        }
    )
    mature_agency = agency_stats[agency_stats["mature"]]
    grouped = mature_agency.groupby("agency", dropna=True)
    by_agency_df = pd.DataFrame(
        {"p1_count": grouped.size(), "converted": grouped["converted"].sum()}
    )
    by_agency_df["rate"] = by_agency_df["converted"] / by_agency_df["p1_count"]
    by_agency = by_agency_df["rate"].sort_values(ascending=False)
    by_agency.index.name = "agency"

    # ---- Per-year rate (all cohorts, flagged) -----------------------------
    year_stats = pd.DataFrame(
        {"award_year": p1_year, "converted": matched_mask, "mature": mature_mask}
    ).dropna(subset=["award_year"])
    ygrouped = year_stats.groupby("award_year", dropna=True)
    by_year = pd.DataFrame(
        {
            "p1_count": ygrouped.size(),
            "converted": ygrouped["converted"].sum(),
            "mature": ygrouped["mature"].max(),
        }
    )
    by_year["rate"] = by_year["converted"] / by_year["p1_count"]
    by_year = by_year.sort_index()  # chronological
    by_year.index = by_year.index.astype(int)
    by_year.index.name = "award_year"

    return {
        "matched_pairs": matched_pairs,
        "conversion_rate": conversion_rate,
        "conversion_rate_raw": conversion_rate_raw,
        "by_agency": by_agency,
        "by_year": by_year,
        "phase1_count": n1,
        "phase2_count": n2,
        "matched_count": int(matched_mask.sum()),
        "mature_phase1_count": n_mature,
        "mature_matched_count": n_mature_matched,
        "max_mature_year": max_mature_year,
        "censored_count": n1 - n_mature,
    }
