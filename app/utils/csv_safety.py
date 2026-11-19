"""Neutralise spreadsheet formula injection in CSV exports.

A CSV cell whose text starts with ``=``, ``+``, ``-`` or ``@`` is evaluated as a
formula when the file is opened in Excel, LibreOffice or Google Sheets. Exported
fields such as an outage description are written by users, so an export could
otherwise carry a formula that runs on the machine of whoever opens it.
"""
from typing import Any, Mapping

# Tab and carriage return are included because spreadsheet importers drop them
# and then evaluate whatever follows.
FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def neutralize_csv_cell(value: Any) -> Any:
    """Return ``value`` in a form a spreadsheet will not evaluate.

    Only strings are changed. A string starting with a formula character gets a
    leading single quote, which spreadsheets read as "this cell is text".
    Numbers, booleans and ``None`` pass through untouched, so a numeric column
    holding ``-5`` still exports as the number ``-5``.
    """
    if isinstance(value, str) and value.startswith(FORMULA_PREFIXES):
        return "'" + value
    return value


def neutralize_csv_row(row: Mapping[str, Any]) -> dict:
    """Apply :func:`neutralize_csv_cell` to every value of a ``DictWriter`` row."""
    return {key: neutralize_csv_cell(value) for key, value in row.items()}
