# Jev Ticket Manager

A small ticket manager that uses TypeSafe's **Jev** model to read an incoming customer
message and escalate it to one of three personas: **Finance**, **Technical** or **Sales**.

## How it works

One Jev request per ticket asks six questions in parallel (`app/classifier.py`):

| Question | Type | Used for |
| --- | --- | --- |
| `persona` | Choice: finance / technical / sales / other | Which queue gets the ticket |
| `urgency` | Score (4 levels) | Priority P4–P1 |
| `tone` | Choice: calm / frustrated / angry | Shown on the ticket |
| `finance_topic`, `technical_topic`, `sales_topic` | Choice | Sub-topic; only the chosen persona's answer is read |

Routing rules are plain code in `route_ticket()`:

- `other`, or confidence below **0.5**, goes to the **Manual triage** queue, with the leading persona CC'd.
- Any other persona with at least **25%** probability is CC'd.
- Agents can reassign or resolve tickets from the board.

The thresholds are starting points. Tune `MIN_CONFIDENCE` and `CC_PROBABILITY` on real tickets.

## Run

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env   # then paste your key from https://console.typesafe.ai/keys
.venv\Scripts\uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000. Tickets are kept in memory and reset when the server restarts.

## Test

```powershell
.venv\Scripts\pip install pytest
.venv\Scripts\python -m pytest tests
```

The tests cover the routing policy with hand-built Jev responses, so they don't need an API key.
