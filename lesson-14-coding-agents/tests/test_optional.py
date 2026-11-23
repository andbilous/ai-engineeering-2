"""Acceptance tests for the optional features 5–7 (SPEC.md). Run them once the mandatory ones pass.

    pytest tests/test_optional.py
"""
from datetime import date

from pytest import approx

from conftest import call, h


# ---------------------------------------------------------------- feature 5: monthly report for finance
def test_report_by_model(client):
    call(client, "emily", "support-bot", "nova-mini", 1_000_000, 0)     # $0.15
    call(client, "daniel", "support-bot", "nova-mini", 0, 1_000_000)    # $0.60
    call(client, "emily", "support-bot", "nova-pro", 0, 100_000)        # $1.00
    r = client.get("/api/projects/support-bot/report.csv", params={"month": "2026-11"}, headers=h("sarah"))
    assert r.status_code == 200 and "text/csv" in r.headers["content-type"]
    assert r.text.strip().splitlines() == [
        "model,calls,input_tokens,output_tokens,cost",
        "nova-mini,2,1000000,1000000,0.7500",
        "nova-pro,1,0,100000,1.0000",
    ]


def test_report_other_month_is_empty(client):
    call(client, "emily", "support-bot", "nova-mini", 1_000, 0)
    r = client.get("/api/projects/support-bot/report.csv", params={"month": "2026-10"}, headers=h("sarah"))
    assert r.text.strip().splitlines() == ["model,calls,input_tokens,output_tokens,cost"]


def test_report_only_for_members(client):
    r = client.get("/api/projects/support-bot/report.csv", params={"month": "2026-11"}, headers=h("james"))
    assert r.status_code == 403


# ---------------------------------------------------------------- feature 6: price changes (data/price_changes.json)
def test_old_price_before_change(client):
    body = call(client, "emily", "support-bot", "atlas-large", 0, 1_000_000).json()
    assert body["cost"] == approx(15.0)


def test_new_price_after_change(client, clock):
    clock.day = date(2026, 12, 2)
    body = call(client, "emily", "support-bot", "atlas-large", 0, 1_000_000).json()
    assert body["cost"] == approx(10.0)


def test_old_calls_keep_old_cost(client, clock):
    call(client, "emily", "support-bot", "atlas-large", 0, 1_000_000)
    clock.day = date(2026, 12, 2)
    calls = client.get("/api/projects/support-bot/calls", headers=h("sarah")).json()
    assert calls[0]["cost"] == approx(15.0)


def test_models_for_date(client):
    models = {m["model"]: m for m in client.get("/api/models", params={"date": "2026-12-02"}).json()}
    assert models["atlas-large"]["output"] == approx(10.0)
    assert models["atlas-small"]["output"] == approx(4.0)


# ---------------------------------------------------------------- feature 7: end-of-month forecast
def test_forecast(client):
    call(client, "olivia", "eval-runs", "nova-pro", 0, 400_000)  # $4.00 by 26 November
    f = client.get("/api/projects/eval-runs/forecast", headers=h("olivia")).json()
    assert f["month"] == "2026-11"
    assert f["spent"] == approx(4.0)
    assert f["projected"] == approx(4.0 / 26 * 30)
    assert f["will_exceed"] is False


def test_forecast_warns(client):
    call(client, "james", "rag-search", "nova-pro", 0, 1_800_000)  # $18.00 by 26 November, budget $20
    f = client.get("/api/projects/rag-search/forecast", headers=h("james")).json()
    assert f["will_exceed"] is True


def test_forecast_only_for_members(client):
    assert client.get("/api/projects/rag-search/forecast", headers=h("emily")).status_code == 403
