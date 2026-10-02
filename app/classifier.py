"""Ticket classification with TypeSafe's Jev model.

Jev makes the judgments (which persona, how urgent, what tone); this module's
plain code owns the routing policy (thresholds, CCs, manual triage).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typesafe_sdk import AsyncTypeSafeClient, Choice, Score, SystemOneResponse

# --- Personas -----------------------------------------------------------------

PERSONAS: dict[str, dict[str, str]] = {
    "finance": {
        "name": "Finance",
        "handler": "Finance desk",
        "description": "Billing, invoices, refunds, payments and subscriptions.",
    },
    "technical": {
        "name": "Technical",
        "handler": "Technical support",
        "description": "Bugs, outages, errors, integrations, login and setup problems.",
    },
    "sales": {
        "name": "Sales",
        "handler": "Sales team",
        "description": "Pricing, plans, demos, upgrades and new purchases.",
    },
}

TRIAGE = "triage"  # queue for tickets Jev can't confidently place

# --- Routing policy (tune these on your own tickets) --------------------------

MIN_CONFIDENCE = 0.5  # below this, a person decides where the ticket goes
CC_PROBABILITY = 0.25  # a second persona with at least this share gets a copy

PRIORITIES = ["P4", "P3", "P2", "P1"]  # index matches the urgency levels below

# --- Questions sent to Jev ----------------------------------------------------
# All questions go in one request: they run in parallel, and the code only reads
# the topic question for the persona that was actually chosen (speculative fan-out).

QUESTIONS = {
    "persona": Choice(
        instructions={
            "question": "Which support team should handle this customer message?",
            "focus": "Classify by what the customer needs resolved, not by words they happen to mention.",
        },
        criteria={
            "finance": {
                "what": "Money already owed or paid: billing, invoices, charges, refunds, "
                "payment methods, failed payments, subscription renewals or cancellations",
                "not_for": "Questions about prices or buying something new (sales)",
                "examples": [
                    "I was charged twice this month",
                    "Can I get a copy of my last invoice?",
                    "Please refund my annual plan",
                ],
            },
            "technical": {
                "what": "The product is not working or the customer needs help using it: "
                "bugs, errors, outages, login problems, integrations, setup, performance",
                "not_for": "A failed payment or a card being declined (finance)",
                "examples": [
                    "The app crashes when I upload a file",
                    "I can't log in after resetting my password",
                    "Your API returns a 500 error",
                ],
            },
            "sales": {
                "what": "Interest in buying or expanding: pricing, plan comparison, quotes, "
                "demos, trials, upgrades, adding seats, enterprise deals",
                "not_for": "Disputes about an existing charge (finance)",
                "examples": [
                    "How much is the enterprise plan?",
                    "Can we book a demo for our team?",
                    "We want to add 20 more seats",
                ],
            },
            "other": "Not a support request for any of these teams, such as spam, "
            "a greeting with no request, or an unrelated topic",
        },
    ),
    "urgency": Score(
        instructions="How urgently does this message need a response?",
        criteria=[
            "No time pressure: a general question or feedback",
            "Should be handled within a few days",
            "Needs attention today: the customer is blocked or losing money",
            "Critical right now: a full outage, security issue, or severe financial impact",
        ],
    ),
    "tone": Choice(
        instructions="What is the customer's tone?",
        criteria={
            "calm": "Neutral or friendly, just stating facts",
            "frustrated": "Annoyed or impatient but civil",
            "angry": "Very upset, strong language or threats to leave",
        },
    ),
    "finance_topic": Choice(
        instructions="If this message is for the finance team, what is it about?",
        criteria={
            "billing_error": "A wrong, duplicate or unexpected charge",
            "refund": "The customer wants money back",
            "invoice": "Requests for invoices, receipts or tax documents",
            "payment_method": "Updating a card, failed or declined payments",
            "subscription": "Renewal, cancellation or changing the billing cycle",
            "other": "A finance topic that fits none of the above",
        },
    ),
    "technical_topic": Choice(
        instructions="If this message is for the technical team, what is it about?",
        criteria={
            "bug": "Something behaves incorrectly or shows an error",
            "outage": "The service is down or unreachable",
            "access": "Login, password, permissions or account access",
            "integration": "Connecting to another tool, API or webhook",
            "how_to": "Help using or setting up a feature",
            "other": "A technical topic that fits none of the above",
        },
    ),
    "sales_topic": Choice(
        instructions="If this message is for the sales team, what is it about?",
        criteria={
            "pricing": "Prices, quotes or discounts",
            "demo": "Requests for a demo, call or trial",
            "upgrade": "Upgrading a plan or adding seats or features",
            "enterprise": "Large-team, custom contract or procurement questions",
            "other": "A sales topic that fits none of the above",
        },
    ),
}


@dataclass
class Classification:
    route: str  # a persona key, or TRIAGE
    suggested: str  # Jev's top persona choice (may be "other")
    confidence: float
    probabilities: dict[str, float]
    cc: list[str] = field(default_factory=list)
    topic: str | None = None
    urgency: float = 0.0
    priority: str = "P4"
    tone: str = "calm"
    reason: str = ""
    model: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def route_ticket(response: SystemOneResponse) -> Classification:
    """Turn Jev's answers into a routing decision. Pure code, no model calls."""
    persona = response.choices["persona"]
    urgency = response.scores["urgency"]
    tone = response.choices["tone"]
    probs = dict(persona.probabilities)

    if persona.choice == "other":
        route, reason = TRIAGE, "Jev found no matching team, so a person should review it."
    elif persona.confidence < MIN_CONFIDENCE:
        route = TRIAGE
        reason = (
            f"Jev leaned towards {PERSONAS[persona.choice]['name']} but confidence "
            f"{persona.confidence:.2f} is below {MIN_CONFIDENCE}, so a person decides."
        )
    else:
        route = persona.choice
        reason = f"Escalated to {PERSONAS[route]['name']} (confidence {persona.confidence:.2f})."

    cc = [
        key
        for key in PERSONAS
        if key != route and key != persona.choice and probs.get(key, 0.0) >= CC_PROBABILITY
    ]
    # When the ticket goes to triage, still tell the leading persona it may be theirs.
    if route == TRIAGE and persona.choice in PERSONAS:
        cc.insert(0, persona.choice)

    # Only read the topic question that matches the chosen persona.
    topic = None
    if persona.choice in PERSONAS:
        topic = response.choices[f"{persona.choice}_topic"].choice

    level = min(round(urgency.score), len(PRIORITIES) - 1)
    return Classification(
        route=route,
        suggested=persona.choice,
        confidence=persona.confidence,
        probabilities=probs,
        cc=cc,
        topic=topic,
        urgency=urgency.score,
        priority=PRIORITIES[level],
        tone=tone.choice,
        reason=reason,
        model=response.model,
    )


async def classify(client: AsyncTypeSafeClient, message: str, subject: str = "") -> Classification:
    state = {"ticket": {"subject": subject, "message": message}} if subject else {"ticket": {"message": message}}
    response = await client.system_one(state=state, questions=QUESTIONS)
    return route_ticket(response)
