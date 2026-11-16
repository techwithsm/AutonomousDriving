import pandas as pd
import pytest

from autodrive.tesla import analysis as A
from autodrive.tesla.cleaning import clean, duplicates_report, inspect

CFG = {
    "drop_columns": ["Source", "Note"],
    "country_aliases": {"Holland": "Netherlands"},
    "valid_models": ["S", "3", "X", "Y"],
}

COLS = ["Case #", "Year", "Date", "Country", "State", "Description", "Deaths", "Tesla driver",
        "Tesla occupant", "Other vehicle", "Cyclists/ Peds", "TSLA+cycl / peds", "Model",
        "Autopilot claimed", "Verified Tesla Autopilot Deaths",
        "Verified Tesla Autopilot Deaths + All Deaths Reported to NHTSA SGO", "Source", "Note"]


@pytest.fixture
def raw():
    rows = [
        ["3", "2022", "1/17/2023", "USA", "CA", "crash", "1", "1", "-", "-", "-", "1", "S", "1", "1", "1", "u", None],
        ["2", "202", "9/17/2020", "Holland", "-", "crash", "2", "-", "2", "-", "-", "2", "1", "-", "-", "-", "u", None],
        ["1", "2019", "3/1/2019", "USA", "-", "ped", "2", "1", "-", "-", "1", "2", "-", "-", "-", "-", "u", None],
        [None, None, None, None, None, None, "353", "117", None, None, None, None, None, None, None, None, None, None],
    ]
    return pd.DataFrame(rows, columns=COLS)


def test_inspect_counts_dashes(raw):
    insp = inspect(raw).set_index("column")
    assert insp.loc["Tesla occupant", "n_dash"] == 2
    assert duplicates_report(raw)["rows_without_case_id"] == 1


def test_clean(raw):
    df, log = clean(raw, CFG)
    assert len(df) == 3                                   # totals row removed
    assert "source" not in df.columns and "Source" not in df.columns
    assert df["deaths"].dtype.kind == "i" and df["tesla_occupant"].tolist() == [0, 2, 0]
    by_id = df.set_index("case_id")
    assert by_id.loc[2, "year"] == 2020                   # 202 typo fixed from date
    assert by_id.loc[3, "year"] == 2023 and by_id.loc[3, "year_reported"] == 2022
    assert by_id.loc[2, "country"] == "Netherlands"
    assert by_id.loc[2, "state"] == "Non-US" and by_id.loc[1, "state"] == "Unknown"
    assert by_id.loc[2, "model"] == "Unknown" and by_id.loc[3, "model"] == "Model S"
    assert log.steps


def test_analysis(raw):
    df, _ = clean(raw, CFG)
    assert A.tesla_driver_deaths(df)["events_with_driver_death"] == 2
    assert A.tesla_and_cyclist_ped(df)["events"] == 1
    assert A.deaths_per_event(df).loc[2, "events"] == 2
    assert A.verified_autopilot(df)["summary"]["verified_deaths"] == 1
    occ = A.occupant_death_proportion(df)
    assert occ.loc["Tesla occupant (passenger) died", "events"] == 1
    assert A.events_by_year(df)["events"].sum() == 3
    assert A.model_distribution(df).loc["Model S", "share_of_known_pct"] == 100.0
