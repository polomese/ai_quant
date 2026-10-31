"""Offline tests for the Sharadar client (fake HTTP session, no network, no key)."""

from __future__ import annotations

import json

import pytest

from pitlab.data import sharadar_client as sc


class FakeResp:
    def __init__(self, status=200, content=b"", headers=None):
        self.status_code = status
        self.content = content
        self.text = content.decode()
        self.headers = headers or {}


class FakeSession:
    def __init__(self, handler):
        self.handler = handler
        self.headers = {}
        self.calls = []

    def get(self, url, params=None, **kw):
        self.calls.append((url, dict(params or {})))
        return self.handler(url, params or {}, len(self.calls))


def csv_rows(n, start=0):
    return (
        "ticker,date,close\n"
        + "".join(f"T,2020-01-{(i % 28) + 1:02d},{i}\n" for i in range(start, start + n))
    ).encode()


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(sc.time, "sleep", lambda s: None)


def client(handler):
    return sc.SharadarClient("KEY", session=FakeSession(handler), min_interval=0)


def test_key_goes_in_header_not_url():
    c = client(lambda u, p, n: FakeResp(200, csv_rows(1)))
    list(c.fetch_pages("stocks", {"ticker": "AAPL"}))
    url, params = c.session.calls[0]
    assert c.session.headers["x-api-key"] == "KEY"
    assert "KEY" not in url and "api_key" not in params


def test_pagination_with_skip_and_explicit_from():
    def h(u, p, n):
        return FakeResp(200, csv_rows(3 if p["skip"] < 6 else 1))

    c = client(h)
    pages = list(c.fetch_pages("stocks", {"ticker": "AAPL"}, page_size=3))
    assert len(pages) == 3
    skips = [p["skip"] for _, p in c.session.calls]
    assert skips == [0, 3, 6]
    assert all(p["from"] == sc.EARLIEST for _, p in c.session.calls)


def test_retries_on_429_then_succeeds():
    def h(u, p, n):
        return (
            FakeResp(429, b"slow down", {"Retry-After": "1"})
            if n < 3
            else FakeResp(200, csv_rows(1))
        )

    c = client(h)
    assert len(list(c.fetch_pages("stocks", {}))) == 1
    assert len(c.session.calls) == 3


def test_auth_and_free_tier_errors():
    with pytest.raises(sc.AuthError):
        client(lambda u, p, n: FakeResp(401, b"{}")).get("data/stocks")
    with pytest.raises(sc.FreeTierError):
        client(lambda u, p, n: FakeResp(403, b'{"error":"Exceeds free tier"}')).get("data/stocks")
    assert (
        client(lambda u, p, n: FakeResp(403, b'{"error":"Exceeds free tier"}')).detect_tier()
        == "free"
    )
    assert client(lambda u, p, n: FakeResp(200, csv_rows(1))).detect_tier() == "paid"


def test_bulk_unavailable_returns_none(tmp_path):
    c = client(lambda u, p, n: FakeResp(401, b"{}"))
    assert c.bulk_download("stocks", tmp_path / "s.zip") is None


def test_rest_download_is_resumable(tmp_path):
    state = {"fail": True}

    def h(u, p, n):
        if p.get("ticker") == "C,D" and state["fail"]:
            return FakeResp(500, b"boom")
        return FakeResp(200, csv_rows(2))

    c = sc.SharadarClient("KEY", session=FakeSession(h), min_interval=0, max_retries=1)
    snap = tmp_path / "stocks" / "20260101T000000Z"
    with pytest.raises(sc.SharadarError):
        sc.BATCH["stocks"] = 2
        sc.rest_download(c, "stocks", snap, ["A", "B", "C", "D"])
    prog = json.loads((snap / "progress.json").read_text())
    assert prog["done"] == ["t00000"]
    state["fail"] = False
    n_calls_before = len(c.session.calls)
    total = sc.rest_download(c, "stocks", snap, ["A", "B", "C", "D"])
    assert total == 4
    # resumed: only the second slice was fetched again
    assert [p["ticker"] for _, p in c.session.calls[n_calls_before:]] == ["C,D"]
    sc.BATCH["stocks"] = 3


def test_snapshot_tables_never_get_a_date_filter():
    c = client(lambda u, p, n: FakeResp(200, csv_rows(1)))
    list(c.fetch_pages("tickers", {}))
    list(c.fetch_pages("descriptions", {}))
    list(c.fetch_pages("actions", {}))
    params = [p for _, p in c.session.calls]
    assert "from" not in params[0] and "from" not in params[1]
    assert params[2]["from"] == sc.EARLIEST
