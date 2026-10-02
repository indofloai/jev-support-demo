"""FastAPI server: ticket API plus the static UI."""

from __future__ import annotations

import itertools
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typesafe_sdk import AsyncTypeSafeClient, TypeSafeAuthenticationError, TypeSafeError

from .classifier import PERSONAS, TRIAGE, classify

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# In-memory store: tickets reset when the server restarts.
TICKETS: list[dict] = []
_ids = itertools.count(1001)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The API key stays on the server; the browser never sees it.
    app.state.client = AsyncTypeSafeClient() if os.getenv("TYPESAFE_API_KEY") else None
    yield
    if app.state.client:
        await app.state.client.aclose()


app = FastAPI(title="Jev Ticket Manager", lifespan=lifespan)


class NewTicket(BaseModel):
    message: str = Field(min_length=1, max_length=5000)
    subject: str = Field(default="", max_length=200)
    customer: str = Field(default="", max_length=100)


class Reassign(BaseModel):
    route: str


def _find(ticket_id: int) -> dict:
    for ticket in TICKETS:
        if ticket["id"] == ticket_id:
            return ticket
    raise HTTPException(404, "Ticket not found")


@app.get("/api/personas")
def personas():
    return {"personas": PERSONAS, "configured": app.state.client is not None}


@app.get("/api/tickets")
def list_tickets():
    return list(reversed(TICKETS))


@app.post("/api/tickets", status_code=201)
async def create_ticket(body: NewTicket):
    client = app.state.client
    if client is None:
        raise HTTPException(503, "TYPESAFE_API_KEY is not set. Add it to .env and restart the server.")
    try:
        result = await classify(client, body.message.strip(), body.subject.strip())
    except TypeSafeAuthenticationError:
        raise HTTPException(401, "TypeSafe rejected the API key. Check TYPESAFE_API_KEY.")
    except TypeSafeError as exc:
        raise HTTPException(502, f"TypeSafe request failed: {exc}")

    ticket = {
        "id": next(_ids),
        "subject": body.subject.strip(),
        "customer": body.customer.strip() or "Anonymous",
        "message": body.message.strip(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "open",
        "manual": False,
        **result.to_dict(),
    }
    TICKETS.append(ticket)
    return ticket


@app.post("/api/tickets/{ticket_id}/assign")
def assign_ticket(ticket_id: int, body: Reassign):
    if body.route not in PERSONAS and body.route != TRIAGE:
        raise HTTPException(400, "Unknown persona")
    ticket = _find(ticket_id)
    ticket.update(route=body.route, manual=True, reason="Assigned manually.")
    return ticket


@app.post("/api/tickets/{ticket_id}/resolve")
def resolve_ticket(ticket_id: int):
    ticket = _find(ticket_id)
    ticket["status"] = "resolved"
    return ticket


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")
