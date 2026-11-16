"""Part 2 - figures. Each function takes the cleaned frame, returns a matplotlib Figure."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from autodrive.common.plotting import OTHER, PALETTE, PRIMARY, SEQUENTIAL_CMAP, TEXT_MUTED, label_bars

from . import analysis as A


def _bar(ax, s: pd.Series, horizontal=False, color=PRIMARY, fmt="{:.0f}"):
    if horizontal:
        s = s.iloc[::-1]
        ax.barh(s.index.astype(str), s.values, color=color, height=0.65)
        ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
    else:
        ax.bar(s.index.astype(str), s.values, color=color, width=0.65)
    label_bars(ax, fmt=fmt, horizontal=horizontal)
    return ax


# --------------------------------------------------------------------------- 2a time & place
def fig_events_by_year(df):
    t = A.events_by_year(df)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    _bar(axes[0], t["events"]); axes[0].set_title("Fatal events per year")
    _bar(axes[1], t["deaths"], color=PALETTE[1]); axes[1].set_title("Deaths per year")
    for ax in axes:
        ax.set_xlabel("Year")
    fig.tight_layout()
    return fig


def fig_monthly_timeline(df):
    m = A.events_by_month(df)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].bar(m.index, m["events"], width=20, color=PRIMARY)
    axes[0].set_title("Events per month"); axes[0].set_ylabel("Events")
    axes[1].plot(m.index, m["events"].cumsum(), color=PRIMARY)
    axes[1].set_title("Cumulative events"); axes[1].set_ylabel("Events")
    axes[1].axvline(pd.Timestamp("2015-10-14"), color=TEXT_MUTED, lw=1, ls="--")
    axes[1].annotate("Autopilot released\n14 Oct 2015", (pd.Timestamp("2015-10-14"), m["events"].sum() * 0.6),
                     xytext=(6, 0), textcoords="offset points", fontsize=8, color=TEXT_MUTED)
    fig.tight_layout()
    return fig


def fig_weekday_month(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    _bar(axes[0], A.events_by_weekday(df)["events"].rename(lambda d: d[:3])); axes[0].set_title("Events by day of week")
    _bar(axes[1], A.events_by_calendar_month(df)["events"]); axes[1].set_title("Events by calendar month")
    fig.tight_layout()
    return fig


def fig_country_state(df, top_c=10, top_s=12):
    c = A.events_by_country(df)["events"].head(top_c)
    s = A.events_by_state(df)["events"].head(top_s)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _bar(axes[0], c, horizontal=True); axes[0].set_title(f"Events by country (top {top_c})")
    _bar(axes[1], s, horizontal=True); axes[1].set_title(f"Events by US state (top {top_s})")
    fig.tight_layout()
    return fig


def fig_heatmap(matrix: pd.DataFrame, title: str, figsize=(11, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(matrix, annot=True, fmt="d", cmap=SEQUENTIAL_CMAP, linewidths=2, linecolor="white",
                cbar_kws={"label": "Events", "shrink": 0.7}, ax=ax, annot_kws={"fontsize": 8})
    ax.set_title(title); ax.set_xlabel(""); ax.set_ylabel(""); ax.grid(False)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- 2b death events
def fig_death_aspects(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    _bar(axes[0, 0], A.deaths_per_event(df)["events"])
    axes[0, 0].set_title("Deaths per accident"); axes[0, 0].set_xlabel("Deaths in event")

    occ = A.occupant_death_proportion(df)["events"].iloc[:4]
    _bar(axes[0, 1], occ.rename(lambda s: s.replace(" died", "")), horizontal=True)
    axes[0, 1].set_title("Who died in the Tesla (events)")

    cp = A.cyclist_ped_distribution(df)["by_year"]["events"]
    _bar(axes[1, 0], cp, color=PALETTE[2]); axes[1, 0].set_title("Events killing a cyclist/pedestrian, by year")

    ov = A.other_vehicle_frequency(df)["by_count"]["events"]
    _bar(axes[1, 1], ov, color=PALETTE[1]); axes[1, 1].set_title("Other vehicles involved per event")
    axes[1, 1].set_xlabel("Number of other vehicles")
    fig.tight_layout()
    return fig


def fig_victim_groups(df):
    v = A.victim_group_breakdown(df)["events"]
    fig, ax = plt.subplots(figsize=(8, 4))
    _bar(ax, v, horizontal=True); ax.set_title("Events by victim group")
    fig.tight_layout()
    return fig


def fig_other_vehicle_trend(df):
    t = A.other_vehicle_frequency(df)["by_year"]
    fig, ax = plt.subplots(figsize=(10, 4))
    x = t.index.astype(str)
    ax.bar(x, t["events"], color=OTHER, width=0.65, label="All events")
    ax.bar(x, t["with_other_vehicle"], color=PALETTE[1], width=0.45, label="With other vehicle")
    ax.set_title("Collisions involving another vehicle, per year"); ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- 2c models
def fig_models(df):
    m = A.model_distribution(df)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = [OTHER if i == "Unknown" else PRIMARY for i in m.index]
    axes[0].bar(m.index, m["events"], color=colors, width=0.65); label_bars(axes[0])
    axes[0].set_title("Events by model")
    known = m.drop(index="Unknown", errors="ignore")
    _bar(axes[1], known["deaths_per_event"], color=PALETTE[1], fmt="{:.2f}")
    axes[1].set_title("Deaths per event (known models)")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- 2d verified autopilot
def fig_verified_autopilot(df):
    v = A.verified_autopilot(df)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    _bar(axes[0, 0], v["funnel"]["events"], horizontal=True); axes[0, 0].set_title("From all events to verified autopilot")
    _bar(axes[0, 1], v["by_year"]["verified_ap_deaths"], color=PALETTE[6]); axes[0, 1].set_title("Verified autopilot deaths by year")
    _bar(axes[1, 0], v["by_model"]["verified_ap_deaths"], color=PALETTE[6]); axes[1, 0].set_title("Verified autopilot deaths by model")
    _bar(axes[1, 1], v["by_country"]["verified_ap_deaths"], horizontal=True, color=PALETTE[6])
    axes[1, 1].set_title("Verified autopilot deaths by country")
    fig.tight_layout()
    return fig


ALL_FIGURES = {
    "01_events_by_year": fig_events_by_year,
    "02_monthly_timeline": fig_monthly_timeline,
    "03_weekday_month": fig_weekday_month,
    "04_country_state": fig_country_state,
    "05_country_year_heatmap": lambda d: fig_heatmap(A.country_year_matrix(d), "Events per year by country"),
    "06_state_year_heatmap": lambda d: fig_heatmap(A.state_year_matrix(d), "Events per year by US state"),
    "07_country_weekday_heatmap": lambda d: fig_heatmap(A.country_weekday_matrix(d), "Events per day of week by country"),
    "08_state_weekday_heatmap": lambda d: fig_heatmap(A.state_weekday_matrix(d), "Events per day of week by US state"),
    "09_death_aspects": fig_death_aspects,
    "10_victim_groups": fig_victim_groups,
    "11_other_vehicle_trend": fig_other_vehicle_trend,
    "12_models": fig_models,
    "13_model_year_heatmap": lambda d: fig_heatmap(A.model_year_matrix(d), "Events per year by model", (11, 3.5)),
    "14_verified_autopilot": fig_verified_autopilot,
}
