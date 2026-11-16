"""Part 2 - end-to-end pipeline: inspect -> clean -> analyse -> tables, figures, report."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from autodrive.common import get_logger, resolve_path
from autodrive.common.plotting import save

from . import analysis as A
from .cleaning import clean, duplicates_report, inspect, load_raw
from .plots import ALL_FIGURES

log = get_logger("part2")


def _md_table(obj, floatfmt=".1f") -> str:
    df = obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)
    return df.to_markdown(floatfmt=floatfmt)


def run(cfg: dict) -> dict:
    out = cfg["output"]
    tables_dir, fig_dir = resolve_path(out["tables"]), resolve_path(out["figures"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- Task 1: inspection & cleaning
    raw = load_raw(resolve_path(cfg["input_csv"]))
    log.info("Loaded raw data: %s rows x %s columns", *raw.shape)
    insp, dups = inspect(raw), duplicates_report(raw)
    insp.to_csv(tables_dir / "00_raw_inspection.csv", index=False)

    df, clog = clean(raw, cfg.get("cleaning", {}))
    cleaned_csv = resolve_path(out["cleaned_csv"])
    df.to_csv(cleaned_csv, index=False)
    log.info("Cleaned data: %s rows x %s columns -> %s", *df.shape, cleaned_csv)

    # ---- Task 2: analysis tables
    top_c, top_s = cfg["analysis"]["top_n_countries"], cfg["analysis"]["top_n_states"]
    tables = {
        "01_events_by_year": A.events_by_year(df),
        "02_events_by_month": A.events_by_month(df),
        "03_events_by_date": A.events_by_date(df),
        "04_busiest_dates": A.busiest_dates(df),
        "05_events_by_weekday": A.events_by_weekday(df),
        "06_events_by_country": A.events_by_country(df),
        "07_events_by_state": A.events_by_state(df),
        "08_country_year": A.country_year_matrix(df, top_c),
        "09_state_year": A.state_year_matrix(df, top_s),
        "10_country_weekday": A.country_weekday_matrix(df, top_c),
        "11_state_weekday": A.state_weekday_matrix(df, top_s),
        "12_country_event_rate": A.events_per_day_rate(df, "country"),
        "13_state_event_rate": A.events_per_day_rate(df[df["country"] == "USA"], "state"),
        "14_deaths_per_event": A.deaths_per_event(df),
        "15_occupant_proportion": A.occupant_death_proportion(df),
        "16_cyclist_ped_by_count": A.cyclist_ped_distribution(df)["by_count"],
        "17_other_vehicle_by_count": A.other_vehicle_frequency(df)["by_count"],
        "18_other_vehicle_by_year": A.other_vehicle_frequency(df)["by_year"],
        "19_victim_groups": A.victim_group_breakdown(df),
        "20_model_distribution": A.model_distribution(df),
        "21_model_year": A.model_year_matrix(df),
        "22_verified_ap_by_count": A.verified_autopilot(df)["by_count"],
        "23_verified_ap_cases": A.verified_autopilot(df)["cases"],
    }
    for name, t in tables.items():
        t.to_csv(tables_dir / f"{name}.csv")
    log.info("Wrote %d tables -> %s", len(tables), tables_dir)

    # ---- figures
    figs = {name: save(fn(df), fig_dir / f"{name}.png") for name, fn in ALL_FIGURES.items()}
    log.info("Wrote %d figures -> %s", len(figs), fig_dir)

    # ---- report
    report_path = resolve_path(out["report"])
    report_path.write_text(build_report(df, raw, insp, dups, clog, fig_dir, report_path.parent), encoding="utf-8")
    log.info("Report -> %s", report_path)
    return {"clean": df, "tables": tables, "figures": figs, "report": report_path, "cleaning_log": clog}


def build_report(df, raw, insp, dups, clog, fig_dir: Path, report_dir: Path) -> str:
    k = A.kpis(df)
    yr = A.events_by_year(df)
    drv = A.tesla_driver_deaths(df)
    occ = A.occupant_death_proportion(df)
    cp = A.cyclist_ped_distribution(df)
    both = A.tesla_and_cyclist_ped(df)
    ov = A.other_vehicle_frequency(df)
    models = A.model_distribution(df)
    ap = A.verified_autopilot(df)
    country = A.events_by_country(df)
    state = A.events_by_state(df)
    wd = A.events_by_weekday(df)

    def fig(name):
        return f"![{name}]({(fig_dir / f'{name}.png').relative_to(report_dir).as_posix()})"

    peak_year = int(yr["events"].idxmax())
    known_models = models.drop(index="Unknown", errors="ignore")
    ov_share = round(100 * df["other_vehicle_involved"].mean(), 1)
    cp_events = int(df["cyclist_ped_died"].sum())

    return f"""# Tesla Autopilot & Road Safety - Exploratory Data Analysis

*Source: `Tesla - Deaths.csv` - {k['events']} fatal events, {k['deaths']} deaths,
{k['first_event']} to {k['last_event']}, {k['countries']} countries.*

## Key findings

- **Events are rising fast:** {int(yr.loc[peak_year, 'events'])} events in {peak_year} (the peak year),
  versus {int(yr['events'].iloc[0])} in {int(yr.index[0])}. This tracks fleet growth; the data has no
  miles-driven denominator, so it shows volume, not risk per mile.
- **Geography:** the USA accounts for {k['usa_share_pct']}% of events; California alone has
  {int(state['events'].iloc[0])} ({state['share_pct'].iloc[0]}% of US events).
- **Victims per event:** {A.deaths_per_event(df).loc[1, 'share_pct']}% of events killed one person;
  {k['multi_fatality_events']} events killed two or more.
- **Tesla drivers died in {drv['events_with_driver_death']} events** ({drv['share_of_events_pct']}% of all events).
- **Occupants (passengers) died in {int(occ.loc['Tesla occupant (passenger) died', 'events'])} events**
  ({occ.loc['Tesla occupant (passenger) died', 'share_pct']}%); a driver *or* occupant died in
  {occ.loc['Driver or occupant died', 'share_pct']}% of events.
- **Cyclists/pedestrians were killed in {cp_events} events** ({round(100 * cp_events / len(df), 1)}%).
- **A Tesla driver/occupant and a cyclist/pedestrian both died in {both['events']} event(s)**
  (case ids {both['case_ids']}) - these victim groups rarely overlap.
- **Another vehicle was involved in {ov_share}% of events.**
- **Models:** model is unknown for {models.loc['Unknown', 'share_pct'] if 'Unknown' in models.index else 0}% of events;
  among known models, {known_models.index[0]} leads with {int(known_models['events'].iloc[0])} events.
- **Verified autopilot deaths:** {ap['summary']['verified_deaths']} deaths in {ap['summary']['verified_events']} events
  ({ap['summary']['share_of_events_pct']}% of events, {ap['summary']['share_of_all_deaths_pct']}% of all deaths).
  Autopilot was claimed but not verified in {ap['summary']['claimed_but_not_verified']} events.

## 1. Preliminary data inspection & cleaning

### 1a. Raw data
Shape: **{raw.shape[0]} rows x {raw.shape[1]} columns**. Every column loads as text because
`-` is used as the "none/unknown" placeholder.

{insp[['column', 'n_nan', 'n_dash', 'pct_missing', 'n_unique']].to_markdown(index=False)}

Duplicates: {dups}

### 1b. Cleaning steps
{clog.to_markdown()}

Final shape: **{df.shape[0]} rows x {df.shape[1]} columns** (incl. derived features).

## 2a. Events by date, year, day, country and state

{fig('01_events_by_year')}

{_md_table(yr)}

{fig('02_monthly_timeline')}

{fig('03_weekday_month')}

Busiest weekday: **{wd['events'].idxmax()}** ({int(wd['events'].max())} events);
quietest: **{wd['events'].idxmin()}** ({int(wd['events'].min())}).

Busiest dates:

{_md_table(A.busiest_dates(df, 5))}

{fig('04_country_state')}

{_md_table(country.head(10))}

{_md_table(state.head(12))}

Average event rate over each location's active period (top 10):

{_md_table(A.events_per_day_rate(df, 'country').head(10), floatfmt='.3f')}

{fig('05_country_year_heatmap')}

{fig('06_state_year_heatmap')}

{fig('07_country_weekday_heatmap')}

{fig('08_state_weekday_heatmap')}

## 2b. Aspects of the death events

{fig('09_death_aspects')}

**Number of victims per accident**

{_md_table(A.deaths_per_event(df))}

**How many times did Tesla drivers die?** {drv['events_with_driver_death']} events
({drv['driver_deaths']} driver deaths, {drv['share_of_events_pct']}% of events).

**Proportion of events in which one or more occupants died**

{_md_table(occ)}

**Distribution of events in which a cyclist or pedestrian was hit**

{_md_table(cp['by_count'])}

{_md_table(cp['by_country'].head(8))}

**Death of a Tesla driver/occupant together with a cyclist/pedestrian:** {both['events']} events
({both['share_of_events_pct']}%), case ids {both['case_ids']}.

**Frequency of Tesla colliding with other vehicles**

{_md_table(ov['by_count'])}

{fig('11_other_vehicle_trend')}

{fig('10_victim_groups')}

## 2c. Event distribution across models

{fig('12_models')}

{_md_table(models, floatfmt='.2f')}

{fig('13_model_year_heatmap')}

## 2d. Verified Tesla autopilot deaths

{fig('14_verified_autopilot')}

{_md_table(pd.Series(ap['summary']).to_frame('value'), floatfmt='.1f')}

{_md_table(ap['by_count'])}

{_md_table(ap['by_model'])}

Verified cases:

{ap['cases'].assign(date=ap['cases']['date'].dt.date).to_markdown(index=False)}

## Limitations
- The dataset counts **fatal events only**; without miles driven (with/without autopilot) no
  claim about autopilot being safer or riskier per mile can be made.
- `-` was read as zero for count columns; some of these are "unknown" rather than "none".
- Model is missing for most cases, so model comparisons rest on a minority of events.
- Year was taken from the event Date where the two disagreed (5 cases).
"""
