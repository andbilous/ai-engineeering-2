"""Acceptance tests for the mandatory features 1–4 (SPEC.md). Do not change them to make them pass.

    pytest tests/test_mandatory.py

Prices are per 1M tokens (data/prices.json). Handy numbers: nova-pro output costs $10 per 1M,
so 100_000 output tokens of nova-pro cost exactly $1.00.
"""
from datetime import date

from pytest import approx

from conftest import budget, call, error_code, h


# ---------------------------------------------------------------- login
def test_no_user_is_401(client):
    assert client.get("/api/projects").status_code == 401


def test_unknown_user_is_401(client):
    assert client.get("/api/projects", headers=h("mallory")).status_code == 401


# ---------------------------------------------------------------- feature 1: log calls and their cost
def test_models_list(client):
    models = {m["model"]: m for m in client.get("/api/models").json()}
    assert set(models) == {"atlas-large", "atlas-small", "nova-pro", "nova-mini"}
    assert models["atlas-large"] == {"model": "atlas-large", "input": 3.0, "cached_input": 0.3, "output": 15.0}


def test_my_projects(client):
    mine = client.get("/api/projects", headers=h("olivia")).json()
    assert {p["id"] for p in mine} == {"rag-search", "eval-runs"}
    assert set(mine[0]) >= {"id", "name", "owner", "monthly_budget", "hard_limit"}


def test_owner_sees_own_projects(client):
    assert {p["id"] for p in client.get("/api/projects", headers=h("michael")).json()} == {"rag-search", "eval-runs"}


def test_call_cost(client):
    # 80k fresh input * $3 + 20k cached * $0.30 + 5k output * $15, all per 1M tokens
    r = call(client, "emily", "support-bot", "atlas-large", 100_000, 5_000, cached_input_tokens=20_000)
    assert r.status_code == 201
    body = r.json()
    assert body["cost"] == approx(0.321)
    assert body["user"] == "emily" and body["project"] == "support-bot" and body["model"] == "atlas-large"
    assert body["created_at"] == "2026-11-26"
    assert body["over_budget"] is False


def test_cached_tokens_default_to_zero(client):
    body = call(client, "emily", "support-bot", "nova-mini", 1_000_000, 0).json()
    assert body["cached_input_tokens"] == 0
    assert body["cost"] == approx(0.15)


def test_cached_more_than_input(client):
    r = call(client, "emily", "support-bot", "nova-mini", 1_000, 10, cached_input_tokens=2_000)
    assert r.status_code == 422 and error_code(r) == "BAD_TOKENS"


def test_negative_tokens(client):
    r = call(client, "emily", "support-bot", "nova-mini", -5, 10)
    assert r.status_code == 422 and error_code(r) == "BAD_TOKENS"


def test_unknown_model(client):
    r = call(client, "emily", "support-bot", "gpt-unknown", 100, 10)
    assert r.status_code == 422 and error_code(r) == "UNKNOWN_MODEL"


def test_unknown_project(client):
    assert call(client, "emily", "no-such-project", "nova-mini", 100, 10).status_code == 404


def test_only_members_log_calls(client):
    assert call(client, "james", "support-bot", "nova-mini", 100, 10).status_code == 403


def test_owner_logs_calls(client):
    assert call(client, "sarah", "support-bot", "nova-mini", 100, 10).status_code == 201


def test_error_has_code_and_message(client):
    detail = call(client, "emily", "support-bot", "gpt-unknown", 100, 10).json()["detail"]
    assert detail["code"] == "UNKNOWN_MODEL"
    assert isinstance(detail["message"], str) and detail["message"]


def test_project_calls_list(client):
    call(client, "emily", "support-bot", "nova-mini", 100, 10)
    call(client, "daniel", "support-bot", "atlas-small", 200, 20)
    calls = client.get("/api/projects/support-bot/calls", headers=h("sarah")).json()
    assert [c["user"] for c in calls] == ["emily", "daniel"]
    assert client.get("/api/projects/support-bot/calls", headers=h("james")).status_code == 403


# ---------------------------------------------------------------- feature 2: monthly budget
def test_budget_status(client):
    call(client, "olivia", "eval-runs", "nova-pro", 0, 100_000)  # $1.00
    b = budget(client, "olivia", "eval-runs")
    assert b["month"] == "2026-11"
    assert b["budget"] == approx(5.0) and b["spent"] == approx(1.0) and b["remaining"] == approx(4.0)
    assert b["percent"] == approx(20.0) and b["status"] == "ok"


def test_warning_at_80_percent(client):
    call(client, "olivia", "eval-runs", "nova-pro", 0, 400_000)  # $4.00 of $5.00
    assert budget(client, "olivia", "eval-runs")["status"] == "warning"


def test_hard_limit_blocks_call(client):
    call(client, "olivia", "eval-runs", "nova-pro", 0, 400_000)  # $4.00
    r = call(client, "olivia", "eval-runs", "nova-pro", 0, 150_000)  # +$1.50 would make $5.50
    assert r.status_code == 422 and error_code(r) == "BUDGET_EXCEEDED"
    assert budget(client, "olivia", "eval-runs")["spent"] == approx(4.0)
    assert len(client.get("/api/projects/eval-runs/calls", headers=h("olivia")).json()) == 1


def test_hard_limit_allows_exactly_the_budget(client):
    call(client, "olivia", "eval-runs", "nova-pro", 0, 400_000)  # $4.00
    assert call(client, "olivia", "eval-runs", "nova-pro", 0, 100_000).status_code == 201  # exactly $5.00
    b = budget(client, "olivia", "eval-runs")
    assert b["remaining"] == approx(0.0) and b["status"] == "exceeded"
    assert call(client, "olivia", "eval-runs", "nova-mini", 1_000, 0).status_code == 422


def test_soft_limit_records_and_flags(client):
    r = call(client, "james", "rag-search", "nova-pro", 0, 2_500_000)  # $25.00 of $20.00
    assert r.status_code == 201 and r.json()["over_budget"] is True
    b = budget(client, "james", "rag-search")
    assert b["status"] == "exceeded" and b["remaining"] == approx(-5.0)


def test_budget_is_per_project(client):
    call(client, "olivia", "eval-runs", "nova-pro", 0, 100_000)
    assert budget(client, "olivia", "rag-search")["spent"] == approx(0.0)


def test_new_month_resets_spend(client, clock):
    call(client, "olivia", "eval-runs", "nova-pro", 0, 400_000)
    clock.day = date(2026, 12, 1)
    b = budget(client, "olivia", "eval-runs")
    assert b["month"] == "2026-12" and b["spent"] == approx(0.0) and b["status"] == "ok"
    assert call(client, "olivia", "eval-runs", "nova-pro", 0, 400_000).status_code == 201
    assert len(client.get("/api/projects/eval-runs/calls", headers=h("olivia")).json()) == 2


def test_budget_only_for_members(client):
    assert client.get("/api/projects/eval-runs/budget", headers=h("emily")).status_code == 403


# ---------------------------------------------------------------- feature 3: only the owner changes the budget
def test_owner_changes_budget(client):
    call(client, "olivia", "eval-runs", "nova-pro", 0, 400_000)  # $4.00
    r = client.put("/api/projects/eval-runs/budget", json={"monthly_budget": 10.0, "hard_limit": True}, headers=h("michael"))
    assert r.status_code == 200 and r.json()["monthly_budget"] == approx(10.0)
    assert budget(client, "olivia", "eval-runs")["status"] == "ok"
    assert call(client, "olivia", "eval-runs", "nova-pro", 0, 150_000).status_code == 201  # was blocked at $5


def test_member_cannot_change_budget(client):
    r = client.put("/api/projects/eval-runs/budget", json={"monthly_budget": 100.0, "hard_limit": False}, headers=h("olivia"))
    assert r.status_code == 403
    assert budget(client, "olivia", "eval-runs")["budget"] == approx(5.0)


def test_budget_must_be_positive(client):
    r = client.put("/api/projects/eval-runs/budget", json={"monthly_budget": 0, "hard_limit": True}, headers=h("michael"))
    assert r.status_code == 422 and error_code(r) == "BAD_BUDGET"


def test_switch_to_soft_limit(client):
    call(client, "olivia", "eval-runs", "nova-pro", 0, 400_000)
    client.put("/api/projects/eval-runs/budget", json={"monthly_budget": 5.0, "hard_limit": False}, headers=h("michael"))
    r = call(client, "olivia", "eval-runs", "nova-pro", 0, 150_000)
    assert r.status_code == 201 and r.json()["over_budget"] is True


# ---------------------------------------------------------------- feature 4: a page in the browser
def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
