"""Shared fixtures for the acceptance tests. Do not change: the tests are the contract from SPEC.md.

Every test gets a fresh SQLite file and a clock it can move, so "today" is 26.11.2026 unless a test says otherwise.
"""
from datetime import date

import pytest
from fastapi.testclient import TestClient


class Clock:
    def __init__(self):
        self.day = date(2026, 11, 26)

    def __call__(self) -> date:
        return self.day


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def client(tmp_path, clock):
    from app.main import create_app

    app = create_app(str(tmp_path / "test.db"), today=clock)
    with TestClient(app) as c:
        yield c


def h(user: str) -> dict:
    """Headers of a logged-in employee (login is simulated with X-User)."""
    return {"X-User": user}


def call(client, user, project, model, input_tokens=0, output_tokens=0, cached_input_tokens=None):
    """Log one LLM call. cached_input_tokens is left out of the body when None (it is optional in the API)."""
    body = {"project": project, "model": model, "input_tokens": input_tokens, "output_tokens": output_tokens}
    if cached_input_tokens is not None:
        body["cached_input_tokens"] = cached_input_tokens
    return client.post("/api/calls", json=body, headers=h(user))


def budget(client, user, project):
    return client.get(f"/api/projects/{project}/budget", headers=h(user)).json()


def error_code(resp) -> str:
    return resp.json()["detail"]["code"]
