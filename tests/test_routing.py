"""Routing-policy tests. They build Jev responses by hand, so no API key is needed."""

import json

from typesafe_sdk import SystemOneResponse

from app.classifier import TRIAGE, route_ticket


def make_response(persona: dict[str, float], urgency: float = 1.0) -> SystemOneResponse:
    top = max(persona, key=persona.get)
    topic = {"type": "choice", "choice": "other", "confidence": 1.0, "probabilities": {"other": 1.0}}
    return SystemOneResponse.model_validate_json(json.dumps({
        "model": "jev-test",
        "usage": {"input_tokens": 0, "output_tokens": 0},
        "answers": {
            "persona": {"type": "choice", "choice": top, "confidence": persona[top] * 0.9,
                        "probabilities": persona},
            "urgency": {"type": "score", "score": urgency, "confidence": 1.0,
                        "legend": {str(i): str(i) for i in range(4)},
                        "probabilities": {str(i): float(i == round(urgency)) for i in range(4)}},
            "tone": {"type": "choice", "choice": "calm", "confidence": 1.0, "probabilities": {"calm": 1.0}},
            "finance_topic": topic, "technical_topic": topic, "sales_topic": topic,
        },
    }))


def test_confident_ticket_goes_to_persona():
    r = route_ticket(make_response({"finance": 0.95, "technical": 0.05, "sales": 0.0, "other": 0.0}, 2.6))
    assert r.route == "finance" and r.cc == [] and r.priority == "P1"


def test_second_persona_with_real_share_gets_cc():
    r = route_ticket(make_response({"finance": 0.65, "technical": 0.35, "sales": 0.0, "other": 0.0}))
    assert r.route == "finance" and r.cc == ["technical"]


def test_low_confidence_goes_to_triage():
    r = route_ticket(make_response({"finance": 0.4, "technical": 0.35, "sales": 0.25, "other": 0.0}))
    assert r.route == TRIAGE and r.cc[0] == "finance"


def test_other_goes_to_triage():
    r = route_ticket(make_response({"finance": 0.0, "technical": 0.0, "sales": 0.0, "other": 1.0}, 0.0))
    assert r.route == TRIAGE and r.topic is None and r.priority == "P4"
