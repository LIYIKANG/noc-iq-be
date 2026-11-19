"""Tests for streaming CSV outage export (Issue #507).

``GET /api/v1/outages/export?format=csv`` must return a streaming CSV
response with ``Content-Disposition: attachment; filename=outages.csv`` and
include all exported outage rows.
"""
import csv
import io
import uuid
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.security import require_engineer
from app.main import app
from app.models.orm.outage import OutageORM


def _make_outage(db: Session, outage_id: str) -> OutageORM:
    outage = OutageORM(
        id=outage_id,
        site_name=f"Site {uuid.uuid4().hex[:6]}",
        site_id="site_1",
        severity="high",
        status="open",
        detected_at=datetime.now(timezone.utc),
        description="CSV stream test outage",
        affected_services=["4G"],
    )
    db.add(outage)
    db.commit()
    db.refresh(outage)
    return outage


def test_csv_export_streams_with_attachment_header(client: TestClient, db: Session):
    _make_outage(db, "out-csv-1")

    app.dependency_overrides[require_engineer] = lambda: {"user_id": "test", "role": "engineer"}
    try:
        response = client.get("/api/v1/outages/export?format=csv")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/csv")
    assert response.headers["content-disposition"] == "attachment; filename=outages.csv"

    rows = list(csv.DictReader(io.StringIO(response.text)))
    assert rows, "CSV body should contain at least the seeded outage"
    assert any(r["id"] == "out-csv-1" for r in rows)
    assert "site_name" in rows[0]


def test_csv_export_filters_by_status(client: TestClient, db: Session):
    _make_outage(db, "out-csv-open")
    resolved = OutageORM(
        id="out-csv-resolved",
        site_name="Site B",
        site_id="site_2",
        severity="medium",
        status="resolved",
        detected_at=datetime.now(timezone.utc),
        resolved_at=datetime.now(timezone.utc),
        description="Resolved outage",
        affected_services=["4G"],
    )
    db.add(resolved)
    db.commit()

    app.dependency_overrides[require_engineer] = lambda: {"user_id": "test", "role": "engineer"}
    try:
        response = client.get("/api/v1/outages/export?format=csv&status=resolved")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200, response.text
    rows = list(csv.DictReader(io.StringIO(response.text)))
    ids = {r["id"] for r in rows}
    assert "out-csv-resolved" in ids
    assert "out-csv-open" not in ids


def _export_csv_rows(client: TestClient, query: str = "") -> list[dict]:
    app.dependency_overrides[require_engineer] = lambda: {"user_id": "test", "role": "engineer"}
    try:
        response = client.get(f"/api/v1/outages/export?format=csv{query}")
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 200, response.text
    # newline="" keeps a carriage return inside a quoted cell intact.
    return list(csv.DictReader(io.StringIO(response.text, newline="")))


@pytest.mark.parametrize(
    "payload",
    [
        '=HYPERLINK("http://example.test","click")',
        "+1+1",
        "-2+3",
        "@SUM(A1:A2)",
        "\t=1+1",
        "\r=1+1",
    ],
)
def test_csv_export_neutralizes_formula_prefixes(client: TestClient, db: Session, payload: str):
    """Issue #1061: user-supplied text must not open as a spreadsheet formula."""
    outage_id = f"out-csv-formula-{uuid.uuid4().hex[:8]}"
    db.add(
        OutageORM(
            id=outage_id,
            site_name=payload,
            site_id="site_formula",
            severity="high",
            status="open",
            detected_at=datetime.now(timezone.utc),
            description=payload,
            affected_services=["4G"],
            assigned_to=payload,
        )
    )
    db.commit()

    row = next(r for r in _export_csv_rows(client) if r["id"] == outage_id)

    assert row["description"] == "'" + payload
    assert row["site_name"] == "'" + payload
    assert row["assigned_to"] == "'" + payload


def test_csv_export_leaves_ordinary_text_and_numbers_unchanged(client: TestClient, db: Session):
    outage_id = f"out-csv-plain-{uuid.uuid4().hex[:8]}"
    db.add(
        OutageORM(
            id=outage_id,
            site_name="Lagos North - Site 4",
            site_id="site_plain",
            severity="high",
            status="open",
            detected_at=datetime.now(timezone.utc),
            description="Fibre cut: a=b, 10% - 20% of traffic rerouted",
            affected_services=["4G", "5G"],
            affected_subscribers=1250,
        )
    )
    db.commit()

    row = next(r for r in _export_csv_rows(client) if r["id"] == outage_id)

    assert row["site_name"] == "Lagos North - Site 4"
    assert row["description"] == "Fibre cut: a=b, 10% - 20% of traffic rerouted"
    assert row["affected_subscribers"] == "1250"
    assert row["affected_services"] == '["4G", "5G"]'
    assert row["severity"] == "high"


def test_csv_export_neutralizes_cells_when_columns_are_selected(client: TestClient, db: Session):
    outage_id = f"out-csv-cols-{uuid.uuid4().hex[:8]}"
    db.add(
        OutageORM(
            id=outage_id,
            site_name="Site C",
            site_id="site_cols",
            severity="high",
            status="open",
            detected_at=datetime.now(timezone.utc),
            description="=1+1",
            affected_services=["4G"],
        )
    )
    db.commit()

    rows = _export_csv_rows(client, "&columns=id,description")
    row = next(r for r in rows if r["id"] == outage_id)

    assert list(row.keys()) == ["id", "description"]
    assert row["description"] == "'=1+1"


def test_json_export_does_not_alter_formula_like_text(client: TestClient, db: Session):
    """The quote is a spreadsheet convention; JSON consumers get the stored value."""
    outage_id = f"out-json-formula-{uuid.uuid4().hex[:8]}"
    db.add(
        OutageORM(
            id=outage_id,
            site_name="Site J",
            site_id="site_json",
            severity="high",
            status="open",
            detected_at=datetime.now(timezone.utc),
            description="=1+1",
            affected_services=["4G"],
        )
    )
    db.commit()

    app.dependency_overrides[require_engineer] = lambda: {"user_id": "test", "role": "engineer"}
    try:
        response = client.get("/api/v1/outages/export?format=json")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200, response.text
    exported = next(o for o in response.json() if o["id"] == outage_id)
    assert exported["description"] == "=1+1"


def test_json_export_still_works(client: TestClient, db: Session):
    _make_outage(db, "out-json-1")

    app.dependency_overrides[require_engineer] = lambda: {"user_id": "test", "role": "engineer"}
    try:
        response = client.get("/api/v1/outages/export?format=json")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200, response.text
    data = response.json()
    assert isinstance(data, list)
    assert any(o["id"] == "out-json-1" for o in data)
