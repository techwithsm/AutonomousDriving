"""Part 2 - Task 1: preliminary data inspection and cleaning of ``Tesla - Deaths.csv``.

The raw file has several quirks (verified during data inspection):

* column names are padded with spaces (``' Deaths '``)
* ``-`` means "none / unknown" in the count and categorical columns
* 13 trailing rows are not cases (column totals, stray numbers, a footnote)
* case 132 has Year ``202`` (typo) and 4 cases have a Year that disagrees with Date
* the same country is spelled two ways (``Holland`` / ``Netherlands``)
* ``Model`` contains two non-model values (``1``, ``2``)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# raw (stripped) column name -> analysis name
COLUMN_MAP = {
    "Case #": "case_id",
    "Year": "year_reported",
    "Date": "date",
    "Country": "country",
    "State": "state",
    "Description": "description",
    "Deaths": "deaths",
    "Tesla driver": "tesla_driver",
    "Tesla occupant": "tesla_occupant",
    "Other vehicle": "other_vehicle",
    "Cyclists/ Peds": "cyclists_peds",
    "TSLA+cycl / peds": "tsla_cycl_peds",
    "Model": "model",
    "Autopilot claimed": "autopilot_claimed",
    "Verified Tesla Autopilot Deaths": "verified_ap_deaths",
    "Verified Tesla Autopilot Deaths + All Deaths Reported to NHTSA SGO": "verified_ap_plus_nhtsa",
}

COUNT_COLUMNS = [
    "deaths", "tesla_driver", "tesla_occupant", "other_vehicle", "cyclists_peds",
    "tsla_cycl_peds", "autopilot_claimed", "verified_ap_deaths", "verified_ap_plus_nhtsa",
]

MISSING_TOKENS = {"-", "", "nan", "NaN", "None"}


@dataclass
class CleaningLog:
    """Human-readable record of every change the cleaner makes."""

    steps: list[str] = field(default_factory=list)

    def add(self, msg: str) -> None:
        self.steps.append(msg)

    def to_markdown(self) -> str:
        return "\n".join(f"{i}. {s}" for i, s in enumerate(self.steps, 1))


# --------------------------------------------------------------------------- load / inspect
def load_raw(path) -> pd.DataFrame:
    """Read the CSV exactly as delivered (column names stripped of padding)."""
    df = pd.read_csv(path, dtype=str, keep_default_na=True)
    df.columns = df.columns.str.strip()
    for c in df.columns:
        df[c] = df[c].str.strip()
    return df


def inspect(df: pd.DataFrame) -> pd.DataFrame:
    """Task 1a - one row per column: dtype, NaN count, '-' placeholder count, unique values."""
    inferred = df.apply(lambda s: pd.to_numeric(s, errors="coerce"))
    rows = []
    for c in df.columns:
        s = df[c]
        rows.append({
            "column": c,
            "raw_dtype": str(s.dtype),
            "numeric_share": round(inferred[c].notna().sum() / max(s.notna().sum(), 1), 2),
            "n_nan": int(s.isna().sum()),
            "n_dash": int((s == "-").sum()),
            "pct_missing": round(100 * (s.isna() | (s == "-")).mean(), 1),
            "n_unique": int(s.nunique()),
            "example": next((v for v in s.dropna() if v != "-"), None),
        })
    return pd.DataFrame(rows)


def duplicates_report(df: pd.DataFrame) -> dict:
    """Task 1a - exact duplicate rows and duplicate case ids."""
    has_case = df["Case #"].notna() if "Case #" in df else pd.Series(True, index=df.index)
    return {
        "rows": len(df),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "exact_duplicate_rows_excluding_blank": int(df[has_case].duplicated().sum()),
        "duplicate_case_ids": int(df.loc[has_case, "Case #"].duplicated().sum()) if "Case #" in df else None,
        "rows_without_case_id": int((~has_case).sum()),
    }


# --------------------------------------------------------------------------- clean
def _to_count(s: pd.Series) -> pd.Series:
    s = s.where(~s.isin(MISSING_TOKENS), None)
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(int)


def clean(raw: pd.DataFrame, cfg: dict | None = None) -> tuple[pd.DataFrame, CleaningLog]:
    """Task 1 - return an analysis-ready frame plus a log of what was changed."""
    cfg = cfg or {}
    log = CleaningLog()
    unknown = cfg.get("unknown_label", "Unknown")
    df = raw.copy()

    # 1. keep only real accident cases
    n0 = len(df)
    df = df[pd.to_numeric(df["Case #"], errors="coerce").notna()].copy()
    log.add(f"Removed {n0 - len(df)} non-case rows (totals, blanks, footnote) -> {len(df)} cases.")

    # 2. drop irrelevant columns (task 1b)
    drop = [c for c in cfg.get("drop_columns", []) if c in df.columns]
    df = df.drop(columns=drop)
    if drop:
        log.add(f"Dropped {len(drop)} irrelevant columns: {', '.join(drop)}.")

    # 3. snake_case names
    df = df.rename(columns=COLUMN_MAP)
    df["case_id"] = df["case_id"].astype(float).astype(int)

    # 4. exact duplicates
    dups = int(df.duplicated().sum())
    if dups:
        df = df.drop_duplicates()
    log.add(f"Exact duplicate cases removed: {dups}.")

    # 5. counts: '-' -> 0, numeric
    for c in COUNT_COLUMNS:
        if c in df:
            n_dash = int(df[c].isin(MISSING_TOKENS).sum() + df[c].isna().sum())
            df[c] = _to_count(df[c])
            log.add(f"'{c}': {n_dash} '-'/blank values interpreted as 0, cast to int.")

    # 6. dates & year
    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
    bad_dates = int(df["date"].isna().sum())
    df["year_reported"] = pd.to_numeric(df["year_reported"], errors="coerce").astype("Int64")
    mismatch = df[df["date"].dt.year != df["year_reported"]]
    if len(mismatch):
        detail = "; ".join(f"case {r.case_id}: {r.year_reported}->{r.date.year}" for r in mismatch.itertuples())
        log.add(f"Year disagrees with Date for {len(mismatch)} cases ({detail}); "
                "'year' is derived from Date, original kept as 'year_reported'.")
    df["year"] = df["date"].dt.year.fillna(df["year_reported"]).astype(int)
    log.add(f"Parsed 'date' to datetime ({bad_dates} unparseable).")

    # derived time features
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df["month_name"] = df["date"].dt.month_name().str[:3]
    df["weekday"] = df["date"].dt.day_name()
    df["day_of_month"] = df["date"].dt.day

    # 7. categorical tidy-up
    aliases = cfg.get("country_aliases", {})
    n_alias = int(df["country"].isin(aliases).sum())
    df["country"] = df["country"].replace(aliases)
    if n_alias:
        log.add(f"Unified country spellings {aliases} ({n_alias} rows).")

    non_us = cfg.get("non_us_state_label", "Non-US")
    is_dash = df["state"].isin(MISSING_TOKENS) | df["state"].isna()
    df.loc[is_dash & (df["country"] != "USA"), "state"] = non_us
    df.loc[is_dash & (df["country"] == "USA"), "state"] = unknown
    log.add(f"State '-' -> '{non_us}' for non-US cases, '{unknown}' for US cases ({int(is_dash.sum())} rows).")

    valid = set(map(str, cfg.get("valid_models", ["S", "3", "X", "Y"])))
    bad_model = ~df["model"].isin(valid)
    odd = sorted(set(df.loc[bad_model, "model"].dropna()) - MISSING_TOKENS)
    df.loc[bad_model, "model"] = unknown
    df["model"] = df["model"].map(lambda m: m if m == unknown else f"Model {m}")
    log.add(f"Model: {int(bad_model.sum())} '-'/invalid values (incl. {odd}) -> '{unknown}'.")

    # 8. analysis flags
    df["driver_died"] = df["tesla_driver"] > 0
    df["occupant_died"] = df["tesla_occupant"] > 0
    df["tesla_person_died"] = df["driver_died"] | df["occupant_died"]
    df["cyclist_ped_died"] = df["cyclists_peds"] > 0
    df["other_vehicle_involved"] = df["other_vehicle"] > 0
    df["autopilot_claimed_flag"] = df["autopilot_claimed"] > 0
    df["verified_ap_flag"] = df["verified_ap_deaths"] > 0
    df["victim_group"] = np.select(
        [df["tesla_person_died"] & df["cyclist_ped_died"],
         df["tesla_person_died"] & df["other_vehicle_involved"],
         df["tesla_person_died"],
         df["cyclist_ped_died"],
         df["other_vehicle_involved"]],
        ["Tesla + cyclist/ped", "Tesla + other vehicle", "Tesla only",
         "Cyclist/ped only", "Other vehicle only"],
        default="Unattributed",
    )
    log.add("Added flags: driver_died, occupant_died, tesla_person_died, cyclist_ped_died, "
            "other_vehicle_involved, autopilot_claimed_flag, verified_ap_flag, victim_group.")

    df = df.sort_values("date").reset_index(drop=True)
    return df, log
