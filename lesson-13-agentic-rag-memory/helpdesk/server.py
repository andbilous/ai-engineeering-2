"""Backend of the IT helpdesk: FastAPI + the web UI from web/. Ready, do not change.

    python server.py                    # http://127.0.0.1:8013
    python server.py --fresh            # start with an empty database (output/helpdesk.db)

The AI part lives in agent.py and todo.py (students write it). Until a method is written, the server answers with
a "TODO" message instead of failing, so the UI, memory, audit, tickets and approvals work from the first minute.

Login is simulated: the UI sends the chosen employee in the X-User header. Every endpoint takes the employee only from
that header, so one employee cannot read or delete another's memory by changing the request body.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402
from fastapi import FastAPI, Header, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, PlainTextResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from agent import MODEL, Agent, Session, make_llm  # noqa: E402
from core.kb import KB  # noqa: E402
from core.store import EMPLOYEES, Store  # noqa: E402

DB = ROOT / "output" / "helpdesk.db"
DB.parent.mkdir(exist_ok=True)
if os.getenv("HELPDESK_FRESH") == "1" and DB.exists():
    DB.unlink()

kb, store, llm = KB(), Store(DB), make_llm()
agent = Agent(llm, store, kb)

sessions: dict[str, Session] = {}

app = FastAPI(title="Northwind · IT-асистент")
app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")


def who(x_user: str | None) -> str:
    if not x_user or x_user not in EMPLOYEES:
        raise HTTPException(401, "невідомий співробітник: виберіть себе у списку")
    return x_user


def session(user: str) -> Session:
    if user not in sessions:
        sessions[user] = Session(user)
    return sessions[user]


class ChatIn(BaseModel):
    message: str


class FeedbackIn(BaseModel):
    rating: str
    comment: str = ""


class DecisionIn(BaseModel):
    approve: bool


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/api/employees")
def employees():
    return list(EMPLOYEES.values())


@app.get("/api/status")
def status():
    return {"model": MODEL, "key": llm is not None, "budget": agent.budget}


@app.get("/api/sample-log", response_class=PlainTextResponse)
def sample_log():
    return (ROOT / "data" / "logs" / "vpn-client.log").read_text(encoding="utf-8")


@app.post("/api/session/new")
def new_session(x_user: str | None = Header(None)):
    user = who(x_user)
    try:
        episode = agent.end_session(sessions[user]) if user in sessions else None
    except NotImplementedError as e:
        episode = f"({e}: епізод не записано)"
    sessions[user] = Session(user)
    return {"session": sessions[user].id, "episode": episode}


@app.post("/api/chat")
def chat(body: ChatIn, x_user: str | None = Header(None)):
    user = who(x_user)
    if llm is None:
        raise HTTPException(400, "немає ANTHROPIC_API_KEY у .env")
    if not body.message.strip():
        raise HTTPException(400, "порожнє повідомлення")
    s = session(user)
    try:
        tr = agent.turn(s, body.message)
    except NotImplementedError as e:
        tr = {"answer": f"{e}. Сервер і інтерфейс працюють, відповідати навчиться ваш агент: README, крок 1."}
    return {"session": s.id, "turn": len(s.turns), **tr}


@app.post("/api/feedback")
def feedback(body: FeedbackIn, x_user: str | None = Header(None)):
    try:
        return agent.feedback(who(x_user), body.rating, body.comment)
    except NotImplementedError as e:
        return {"accepted": False, "rule": None, "reason": str(e)}
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.get("/api/memory")
def memory(x_user: str | None = Header(None)):
    user = who(x_user)
    s = sessions.get(user)
    return {"employee": EMPLOYEES[user], "facts": store.facts(user), "rules": store.rules(user),
            "episodes": store.episodes(user, n=10), "summary": s.summary if s else "",
            "history_messages": len(s.history) if s else 0}


@app.delete("/api/memory")
def forget(x_user: str | None = Header(None)):
    user = who(x_user)
    sessions.pop(user, None)
    return {"deleted": store.forget(user)}


@app.get("/api/audit")
def audit(x_user: str | None = Header(None)):
    return store.audit(who(x_user))


@app.get("/api/tickets")
def tickets(x_user: str | None = Header(None)):
    user = who(x_user)
    return {"tickets": store.tickets(user), "requests": store.my_requests(user)}


@app.get("/api/approvals")
def approvals(x_user: str | None = Header(None)):
    user = who(x_user)
    return [{**a, "employee": EMPLOYEES[a["user"]]["name"]} for a in store.to_approve(user)]


@app.post("/api/approvals/{approval_id}")
def decide(approval_id: int, body: DecisionIn, x_user: str | None = Header(None)):
    try:
        return store.decide(who(x_user), approval_id, body.approve)
    except PermissionError as e:
        raise HTTPException(403, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--port", type=int, default=8013)
    a = p.parse_args()
    if a.fresh:
        os.environ["HELPDESK_FRESH"] = "1"
    uvicorn.run("server:app", host="127.0.0.1", port=a.port, reload=False)
