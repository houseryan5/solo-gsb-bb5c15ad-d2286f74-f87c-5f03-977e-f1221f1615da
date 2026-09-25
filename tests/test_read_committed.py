"""Regression tests for read-committed isolation of the read layer.

The read layer (``db.query`` / ``db.query_one`` and every endpoint built on
them) runs on dedicated read connections, separate from the shared write
connection: an open write transaction's uncommitted rows are never visible
to readers, rolled-back rows never leak, and committed rows become visible
immediately.  ``/api/events`` additionally reads its ``total`` and its page
of rows from a single committed snapshot, so a concurrent commit landing
between the two SELECTs cannot skew them.
"""

import json
import os
import tempfile

import pytest

from app import db, services
from app.engine import to_epoch


def at(minute: int, second: int = 0) -> float:
    return to_epoch(f"2026-09-23T00:{minute:02d}:{second:02d}Z")


def row(device, metric, minute, value):
    return {
        "device": device,
        "metric": metric,
        "ts": f"2026-09-23T00:{minute:02d}:00Z",
        "value": value,
    }


def import_jsonl(rows, **kwargs):
    return services.import_events("\n".join(json.dumps(r) for r in rows), **kwargs)


@pytest.fixture()
def scratch_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db.DB_PATH = tmp.name
    db._conn = None
    db.init_db()
    yield tmp.name
    for suffix in ("", "-wal", "-shm"):
        try:
            os.unlink(tmp.name + suffix)
        except OSError:
            pass


@pytest.fixture()
def client(scratch_db):
    from fastapi.testclient import TestClient
    from app.main import app

    return TestClient(app)


def _count_events() -> int:
    return db.query_one("SELECT COUNT(*) AS c FROM events")["c"]


# --------------------------------------------------------------------------- #
# committed-view semantics of the read layer
# --------------------------------------------------------------------------- #


def test_uncommitted_insert_invisible_and_rollback_leaves_no_trace(scratch_db):
    import_jsonl([row("d", "m", 1, 10)])
    assert _count_events() == 1

    # Simulate an in-flight write transaction: inserted but not committed.
    conn = db.get_conn()
    conn.execute(
        "INSERT INTO events (device, metric, ts, value, raw) VALUES ('d','m',?,99,'{}')",
        (at(5),),
    )
    # The read layer must keep seeing the last committed state only.
    assert _count_events() == 1
    assert [r["value"] for r in services.list_events()[0]] == [10]

    conn.rollback()
    # Rolled-back content is never visible.
    assert _count_events() == 1
    assert db.query_one("SELECT value FROM events WHERE ts = ?", (at(5),)) is None

    # Once committed, the new row becomes visible to the read layer.
    conn.execute(
        "INSERT INTO events (device, metric, ts, value, raw) VALUES ('d','m',?,99,'{}')",
        (at(5),),
    )
    conn.commit()
    assert _count_events() == 2
    assert db.query_one("SELECT value FROM events WHERE ts = ?", (at(5),))["value"] == 99


def test_rolled_back_transaction_rows_never_visible(scratch_db):
    import_jsonl([row("d", "m", 1, 10)])
    with pytest.raises(RuntimeError, match="boom"):
        with db.transaction() as tx:
            tx.execute(
                "INSERT INTO events (device, metric, ts, value, raw)"
                " VALUES ('d','m',?,99,'{}')",
                (at(5),),
            )
            # Mid-transaction (event inserted, commit pending): queries still
            # read the committed state, not the transaction's private rows.
            assert _count_events() == 1
            rows, total = services.list_events()
            assert total == 1 and [r["value"] for r in rows] == [10]
            raise RuntimeError("boom")
    # The rollback erased the insert; the read layer never saw it.
    assert _count_events() == 1


def test_read_snapshot_pins_one_committed_state(scratch_db):
    import_jsonl([row("d", "m", 1, 10)])
    with db.read_snapshot():
        assert _count_events() == 1
        # A concurrent commit lands while the snapshot is open...
        import_jsonl([row("d", "m", 2, 11)])
        # ...but every SELECT inside still reads the pinned snapshot.
        assert _count_events() == 1
        # Nested blocks join the outer snapshot instead of starting a new one.
        with db.read_snapshot():
            assert _count_events() == 1
    # After the snapshot ends, the committed row is visible.
    assert _count_events() == 2


def test_list_events_total_and_rows_share_one_snapshot(scratch_db, monkeypatch):
    import_jsonl([row("d", "m", 1, 10)])
    real_query_one = db.query_one
    injected = {"done": False}

    def spy(sql, params=()):
        result = real_query_one(sql, params)
        if not injected["done"] and "COUNT(*)" in sql:
            injected["done"] = True
            # A concurrent import commits between the COUNT and the page
            # SELECT; a single-snapshot read must not observe it.
            import_jsonl([row("d", "m", 2, 11)])
        return result

    monkeypatch.setattr(db, "query_one", spy)
    rows, total = services.list_events()
    assert injected["done"] is True
    assert total == 1
    assert [r["value"] for r in rows] == [10]
    # Afterwards the committed row shows up normally.
    rows, total = services.list_events()
    assert total == 2
    assert [r["value"] for r in rows] == [10, 11]


# --------------------------------------------------------------------------- #
# HTTP API: /api/events, /api/faults and the timeline read the committed view
# --------------------------------------------------------------------------- #


def test_api_events_total_and_list_read_committed_view(client):
    client.post("/api/import/text", json={"text": json.dumps(row("d", "m", 1, 10))})

    conn = db.get_conn()
    conn.execute(
        "INSERT INTO events (device, metric, ts, value, raw) VALUES ('d','m',?,99,'{}')",
        (at(5),),
    )
    try:
        # While the write transaction is still open, the API must not see it.
        body = client.get("/api/events").json()
        assert body["total"] == 1
        assert [e["value"] for e in body["events"]] == [10]
        assert client.get("/api/meta").json()["event_count"] == 1
    finally:
        conn.rollback()
    assert client.get("/api/events").json()["total"] == 1

    conn.execute(
        "INSERT INTO events (device, metric, ts, value, raw) VALUES ('d','m',?,99,'{}')",
        (at(5),),
    )
    conn.commit()
    body = client.get("/api/events").json()
    assert body["total"] == 2
    assert [e["value"] for e in body["events"]] == [10, 99]


def test_api_faults_and_timeline_read_committed_view(client):
    r = client.post(
        "/api/rules",
        json={
            "name": "高温",
            "logic": "AND",
            "duration": 0,
            "conditions": [{"device": "d", "metric": "m", "op": ">=", "threshold": 80}],
        },
    )
    assert r.status_code == 200, r.text
    client.post(
        "/api/import/text",
        json={"text": "\n".join(json.dumps(x) for x in [row("d", "m", 1, 90), row("d", "m", 2, 90)])},
    )
    assert len(client.get("/api/faults").json()) == 1

    conn = db.get_conn()
    conn.execute("DELETE FROM fault_intervals")  # uncommitted destructive write
    try:
        # /api/faults and the timeline still read the committed fault rows.
        assert len(client.get("/api/faults").json()) == 1
        assert len(client.get("/api/timeline").json()["faults"]) == 1
    finally:
        conn.rollback()
    assert len(client.get("/api/faults").json()) == 1


def test_api_config_reads_use_committed_view(client):
    r = client.put(
        "/api/channel-ttls",
        json={"device": "d", "metric": "m", "ttl_seconds": 45},
    )
    assert r.status_code == 200, r.text

    conn = db.get_conn()
    conn.execute("UPDATE channel_ttls SET ttl_seconds = 999 WHERE device = 'd' AND metric = 'm'")
    try:
        # Uncommitted config change is invisible to the read API.
        ttls = client.get("/api/channel-ttls").json()
        assert ttls[0]["ttl_seconds"] == 45
    finally:
        conn.rollback()
    assert client.get("/api/channel-ttls").json()[0]["ttl_seconds"] == 45
