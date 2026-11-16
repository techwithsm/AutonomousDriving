"""Part 2 - Task 2: exploratory data analysis on the cleaned Tesla deaths frame.

Every function is pure: cleaned DataFrame in -> DataFrame/Series/dict out, so the
results can be tested, printed in a notebook, plotted, or written to CSV.
"""
from __future__ import annotations

import pandas as pd

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# =========================================================================== 2a - events over time / place
def events_by_year(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby("year").agg(events=("case_id", "size"), deaths=("deaths", "sum"))
    out["cumulative_events"] = out["events"].cumsum()
    out["yoy_growth_pct"] = (out["events"].pct_change() * 100).round(1)
    return out


def events_by_month(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly time series (gaps filled with 0) - the 'events by date' timeline."""
    s = df.groupby("month").agg(events=("case_id", "size"), deaths=("deaths", "sum"))
    full = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    s = s.reindex(full, fill_value=0)
    s.index.name = "month"
    s["rolling_12m_events"] = s["events"].rolling(12, min_periods=1).sum()
    return s


def events_by_date(df: pd.DataFrame) -> pd.DataFrame:
    """Events and deaths per calendar date (only dates with at least one event)."""
    return (df.groupby(df["date"].dt.date)
              .agg(events=("case_id", "size"), deaths=("deaths", "sum"))
              .sort_index())


def busiest_dates(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return events_by_date(df).sort_values(["events", "deaths"], ascending=False).head(n)


def events_by_weekday(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby("weekday").agg(events=("case_id", "size"), deaths=("deaths", "sum"))
    return out.reindex(WEEKDAYS, fill_value=0)


def events_by_calendar_month(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby("month_name").agg(events=("case_id", "size"), deaths=("deaths", "sum"))
    return out.reindex(MONTHS, fill_value=0)


def events_by_country(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby("country").agg(events=("case_id", "size"), deaths=("deaths", "sum"))
    out["share_pct"] = (100 * out["events"] / out["events"].sum()).round(1)
    out["deaths_per_event"] = (out["deaths"] / out["events"]).round(2)
    return out.sort_values(["events", "deaths"], ascending=False)


def events_by_state(df: pd.DataFrame, us_only: bool = True) -> pd.DataFrame:
    d = df[df["country"] == "USA"] if us_only else df
    out = d.groupby("state").agg(events=("case_id", "size"), deaths=("deaths", "sum"))
    out["share_pct"] = (100 * out["events"] / out["events"].sum()).round(1)
    return out.sort_values(["events", "deaths"], ascending=False)


def _pivot_top(df, key, col, top_n, values="case_id"):
    top = df[key].value_counts().head(top_n).index
    d = df.assign(**{key: df[key].where(df[key].isin(top), "Other")})
    out = d.pivot_table(index=key, columns=col, values=values, aggfunc="count", fill_value=0)
    order = list(top) + (["Other"] if "Other" in out.index else [])
    return out.reindex(order)


def country_year_matrix(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Events per year for each country (top-N, rest folded to 'Other')."""
    return _pivot_top(df, "country", "year", top_n)


def state_year_matrix(df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    """Events per year for each US state (top-N, rest folded to 'Other')."""
    return _pivot_top(df[df["country"] == "USA"], "state", "year", top_n)


def country_weekday_matrix(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Events per day-of-week for each country."""
    return _pivot_top(df, "country", "weekday", top_n).reindex(columns=WEEKDAYS, fill_value=0)


def state_weekday_matrix(df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    """Events per day-of-week for each US state."""
    us = df[df["country"] == "USA"]
    return _pivot_top(us, "state", "weekday", top_n).reindex(columns=WEEKDAYS, fill_value=0)


def events_per_day_rate(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Average events per day (and per year) for each country/state over its active period."""
    g = df.groupby(by)
    out = pd.DataFrame({
        "events": g.size(),
        "first_event": g["date"].min().dt.date,
        "last_event": g["date"].max().dt.date,
    })
    span_days = (pd.to_datetime(out["last_event"]) - pd.to_datetime(out["first_event"])).dt.days + 1
    out["active_days"] = span_days
    out["events_per_day"] = (out["events"] / span_days).round(4)
    out["events_per_year"] = (out["events_per_day"] * 365.25).round(2)
    return out.sort_values("events", ascending=False)


# =========================================================================== 2b - aspects of the death events
def deaths_per_event(df: pd.DataFrame) -> pd.DataFrame:
    """Q: number of victims (deaths) in each accident."""
    vc = df["deaths"].value_counts().sort_index()
    return pd.DataFrame({"events": vc, "share_pct": (100 * vc / vc.sum()).round(1)}).rename_axis("deaths_in_event")


def tesla_driver_deaths(df: pd.DataFrame) -> dict:
    """Q: how many times did Tesla drivers die?"""
    n = int(df["driver_died"].sum())
    return {"events_with_driver_death": n,
            "driver_deaths": int(df["tesla_driver"].sum()),
            "share_of_events_pct": round(100 * n / len(df), 1)}


def occupant_death_proportion(df: pd.DataFrame) -> pd.DataFrame:
    """Q: proportion of events in which one or more occupants died."""
    rows = {
        "Tesla occupant (passenger) died": df["occupant_died"],
        "Tesla driver died": df["driver_died"],
        "Driver or occupant died": df["tesla_person_died"],
        "No Tesla driver/occupant died": ~df["tesla_person_died"],
    }
    out = pd.DataFrame({"events": {k: int(v.sum()) for k, v in rows.items()}})
    out["share_pct"] = (100 * out["events"] / len(df)).round(1)
    occ = df.loc[df["occupant_died"], "tesla_occupant"].value_counts().sort_index()
    for k, v in occ.items():
        out.loc[f"  of which {k} occupant(s) died"] = [int(v), round(100 * v / len(df), 1)]
    return out


def cyclist_ped_distribution(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Q: distribution of events in which the vehicle hit a cyclist or pedestrian."""
    hit = df[df["cyclist_ped_died"]]
    by_count = df["cyclists_peds"].value_counts().sort_index().rename_axis("cyclist_ped_deaths").to_frame("events")
    by_count["share_pct"] = (100 * by_count["events"] / len(df)).round(1)
    return {
        "by_count": by_count,
        "by_year": hit.groupby("year").size().reindex(sorted(df["year"].unique()), fill_value=0).to_frame("events"),
        "by_country": hit["country"].value_counts().to_frame("events"),
        "by_model": hit["model"].value_counts().to_frame("events"),
    }


def tesla_and_cyclist_ped(df: pd.DataFrame) -> dict:
    """Q: accidents with the death of a Tesla driver/occupant AND a cyclist/pedestrian."""
    both = df[df["tesla_person_died"] & df["cyclist_ped_died"]]
    return {"events": int(len(both)),
            "share_of_events_pct": round(100 * len(both) / len(df), 1),
            "case_ids": both["case_id"].tolist()}


def other_vehicle_frequency(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Q: frequency of Tesla colliding with other vehicles."""
    vc = df["other_vehicle"].value_counts().sort_index().rename_axis("other_vehicle_count").to_frame("events")
    vc["share_pct"] = (100 * vc["events"] / len(df)).round(1)
    by_year = df.groupby("year").agg(events=("case_id", "size"), with_other_vehicle=("other_vehicle_involved", "sum"))
    by_year["share_pct"] = (100 * by_year["with_other_vehicle"] / by_year["events"]).round(1)
    return {"by_count": vc, "by_year": by_year}


def victim_group_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    vc = df["victim_group"].value_counts()
    return pd.DataFrame({"events": vc, "share_pct": (100 * vc / len(df)).round(1)})


# =========================================================================== 2c - models
def model_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Events, deaths and victim mix per Tesla model."""
    g = df.groupby("model")
    out = pd.DataFrame({
        "events": g.size(),
        "deaths": g["deaths"].sum(),
        "driver_deaths": g["tesla_driver"].sum(),
        "cyclist_ped_deaths": g["cyclists_peds"].sum(),
        "autopilot_claimed_events": g["autopilot_claimed_flag"].sum(),
        "verified_ap_events": g["verified_ap_flag"].sum(),
    })
    out["share_pct"] = (100 * out["events"] / len(df)).round(1)
    known = out.index != "Unknown"
    out["share_of_known_pct"] = (100 * out["events"] / out.loc[known, "events"].sum()).round(1).where(known)
    out["deaths_per_event"] = (out["deaths"] / out["events"]).round(2)
    return out.sort_values("events", ascending=False)


def model_year_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot_table(index="model", columns="year", values="case_id", aggfunc="count", fill_value=0)


# =========================================================================== 2d - verified autopilot deaths
def verified_autopilot(df: pd.DataFrame) -> dict[str, pd.DataFrame | dict]:
    """Distribution of verified Tesla autopilot deaths."""
    v = df[df["verified_ap_flag"]]
    by_count = df["verified_ap_deaths"].value_counts().sort_index().rename_axis("verified_ap_deaths").to_frame("events")
    by_count["share_pct"] = (100 * by_count["events"] / len(df)).round(1)
    funnel = pd.Series({
        "All events": len(df),
        "Autopilot claimed": int(df["autopilot_claimed_flag"].sum()),
        "Verified AP + NHTSA SGO reported": int((df["verified_ap_plus_nhtsa"] > 0).sum()),
        "Verified autopilot": int(df["verified_ap_flag"].sum()),
    }, name="events")
    return {
        "summary": {
            "verified_events": int(len(v)),
            "verified_deaths": int(df["verified_ap_deaths"].sum()),
            "verified_plus_nhtsa_deaths": int(df["verified_ap_plus_nhtsa"].sum()),
            "share_of_events_pct": round(100 * len(v) / len(df), 1),
            "share_of_all_deaths_pct": round(100 * df["verified_ap_deaths"].sum() / df["deaths"].sum(), 1),
            "claimed_but_not_verified": int((df["autopilot_claimed_flag"] & ~df["verified_ap_flag"]).sum()),
        },
        "by_count": by_count,
        "funnel": funnel.to_frame(),
        "by_year": df.groupby("year")["verified_ap_deaths"].sum().to_frame("verified_ap_deaths"),
        "by_model": v.groupby("model")["verified_ap_deaths"].sum().sort_values(ascending=False).to_frame("verified_ap_deaths"),
        "by_country": v.groupby("country")["verified_ap_deaths"].sum().sort_values(ascending=False).to_frame("verified_ap_deaths"),
        "by_state": v[v["country"] == "USA"].groupby("state")["verified_ap_deaths"].sum()
                     .sort_values(ascending=False).to_frame("verified_ap_deaths"),
        "cases": v[["case_id", "date", "country", "state", "model", "deaths", "verified_ap_deaths", "description"]],
    }


# =========================================================================== headline KPIs
def kpis(df: pd.DataFrame) -> dict:
    return {
        "events": int(len(df)),
        "deaths": int(df["deaths"].sum()),
        "first_event": df["date"].min().date().isoformat(),
        "last_event": df["date"].max().date().isoformat(),
        "countries": int(df["country"].nunique()),
        "us_states": int(df.loc[df["country"] == "USA", "state"].nunique()),
        "usa_share_pct": round(100 * (df["country"] == "USA").mean(), 1),
        "multi_fatality_events": int((df["deaths"] > 1).sum()),
    }
