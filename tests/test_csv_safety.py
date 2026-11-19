"""Tests for CSV formula neutralisation (Issue #1061)."""
import csv
import io

import pytest

from app.models.sla import SLATrendPoint
from app.utils.analytics_exporter import export_trends
from app.utils.csv_safety import (
    FORMULA_PREFIXES,
    neutralize_csv_cell,
    neutralize_csv_row,
)


def test_covers_the_four_formula_characters_plus_tab_and_carriage_return():
    assert set(FORMULA_PREFIXES) == {"=", "+", "-", "@", "\t", "\r"}


@pytest.mark.parametrize("prefix", FORMULA_PREFIXES)
def test_string_starting_with_a_formula_character_gets_a_leading_quote(prefix):
    value = f"{prefix}SUM(A1:A2)"

    assert neutralize_csv_cell(value) == "'" + value


@pytest.mark.parametrize(
    "value",
    [
        "plain text",
        "a=b",
        "10% - 20% of traffic",
        "name@example.test",
        "",
        " =leading space is not a formula",
        "2026-11-19T00:00:00Z",
        '["4G"]',
        "null",
    ],
)
def test_ordinary_text_is_returned_unchanged(value):
    assert neutralize_csv_cell(value) == value


@pytest.mark.parametrize("value", [-5, -2.5, 0, 42, 3.14, True, None])
def test_non_string_values_are_returned_untouched(value):
    assert neutralize_csv_cell(value) is value


def test_a_negative_number_stored_as_text_is_treated_as_text():
    # A string cannot be told apart from a formula, so it is quoted; a real
    # numeric column is covered by the test above.
    assert neutralize_csv_cell("-5") == "'-5"


def test_row_helper_neutralises_values_and_keeps_keys_and_order():
    row = {"id": "out-1", "description": "=1+1", "affected_subscribers": -5}

    safe = neutralize_csv_row(row)

    assert list(safe.keys()) == ["id", "description", "affected_subscribers"]
    assert safe == {"id": "out-1", "description": "'=1+1", "affected_subscribers": -5}
    assert row["description"] == "=1+1", "the input row must not be mutated"


def _trend(date: str) -> SLATrendPoint:
    return SLATrendPoint(
        date=date, total_outages=3, violations=1, rewards=12.5, penalties=0.0
    )


def test_trends_csv_neutralises_text_and_keeps_numeric_columns():
    payload = "=cmd|' /C calc'!A0"

    exported = export_trends([_trend(payload)], format="csv")
    row = next(csv.DictReader(io.StringIO(exported, newline="")))

    assert row["date"] == "'" + payload
    assert row["total_outages"] == "3"
    assert row["violations"] == "1"
    assert row["rewards"] == "12.5"
    assert row["penalties"] == "0.0"


def test_trends_json_export_is_not_altered():
    exported = export_trends([_trend("=1+1")], format="json")

    assert exported[0]["date"] == "=1+1"


def test_ordinary_trend_rows_export_exactly_as_before():
    exported = export_trends([_trend("2026-11-19")], format="csv")

    assert exported == (
        "date,total_outages,violations,rewards,penalties\r\n"
        "2026-11-19,3,1,12.5,0.0\r\n"
    )
