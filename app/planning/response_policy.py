from __future__ import annotations

import re

from app.planning.turn_schema import ClarifyAction, ConversationAction, ReferenceAction, TurnPlan

_CANNED = {
    "assistant_identity": {"text": "My name is Nia.", "voice": "I'm Nia."},
    "audio_check": {"text": "Yes, I received your message.", "voice": "Yes, I can hear you."},
    "greeting": {"text": "Hi! How can I help?", "voice": "Hi!"},
    "thanks": {"text": "You're welcome.", "voice": "You're welcome."},
    "farewell": {"text": "Goodbye.", "voice": "Goodbye."},
    "acknowledgement": {"text": "Sure.", "voice": "Sure."},
    "capability": {
        "text": (
            "I'm Nia, a research assistant for the authorized records. "
            "I write answers here. Click Connect voice if you want me to speak."
        ),
        "voice": "I'm Nia. Ask me about the authorized records, or keep talking and I'll speak.",
    },
    "voice_capability": {
        "text": (
            "Yes. Click Connect voice and I will speak the answers. "
            "In this text chat I only write."
        ),
        "voice": "Yes. I can speak in this voice session.",
    },
}


def resolve_final_response(turn: TurnPlan) -> str:
    """AI-mode response contract. Incoherent combinations become clarifications."""
    if only_clarification(turn) or only_direct_response(turn):
        if not (turn.respond_actions() or turn.clarify_actions()):
            raise _response_contract("clarification/respond-only plans require compiler_response")
        return "compiler_response"
    if only_reference(turn):
        return "deterministic"
    has_data = bool(turn.query_actions() or turn.modify_actions() or turn.verify_actions())
    if turn.final_response == "compiler_response" and has_data:
        raise _response_contract("compiler_response cannot suppress a query action")
    if turn.final_response == "llm_synthesis":
        if not has_data and not (turn.needs_explanation or turn.needs_inference):
            raise _response_contract("llm_synthesis requires executable data or explanation intent")
        return "llm_synthesis"
    if has_data:
        return "deterministic"
    return turn.final_response


def _response_contract(message: str):
    from app.planning.plan_validator import PlanValidationError

    return PlanValidationError(message, code="response_contract")


def decide_final_response(turn: TurnPlan) -> str:
    """Trusted override of the compiler's response mode."""

    if only_clarification(turn):
        return "compiler_response"
    if only_direct_response(turn):
        return "compiler_response"
    if turn.verify_actions():
        return "deterministic"
    data_goals = {item.goal for item in turn.query_actions()}
    if turn.modify_actions():
        data_goals.update(item.goal or "list" for item in turn.modify_actions())
    if data_goals:
        # Explicit synthesis intent must win over the mechanical query shape.
        # A model may correctly represent "summarize these records" as a list
        # retrieval plus synthesis; treating every list as deterministic loses
        # the user's requested operation and exposes raw fixture-like rows.
        if turn.needs_inference or turn.needs_explanation or turn.final_response == "llm_synthesis":
            return "llm_synthesis"
        if data_goals <= {"list", "distinct"} and not turn.needs_inference:
            return "deterministic"
        if data_goals <= {"count"} and not turn.needs_explanation and not turn.needs_inference:
            return "deterministic"
        return "deterministic"
    if turn.needs_explanation or turn.needs_inference:
        return "llm_synthesis"
    if turn.final_response == "llm_synthesis":
        return "llm_synthesis"
    return turn.final_response


def only_clarification(turn: TurnPlan) -> bool:
    return bool(turn.actions) and all(isinstance(item, ClarifyAction) for item in turn.actions)


def only_general_conversation(turn: TurnPlan | None) -> bool:
    if turn is None or not turn.actions:
        return False
    return all(
        isinstance(item, ConversationAction) and item.intent == "general_conversation"
        for item in turn.actions
    )


def only_direct_response(turn: TurnPlan) -> bool:
    """Whether this turn is answered in words alone.

    Decided by the actions, not by the declared final_response. A plan carrying only
    conversation actions has no data to retrieve, so words are the only possible
    answer; a planner that labelled such a turn "deterministic" mislabelled the mode
    but was not wrong about the intent. Routing on the declared mode instead sent
    these turns into query construction, where they failed as "no executable query
    action" -- a clarification for a question the planner had understood correctly.
    """
    if not turn.actions or not all(isinstance(item, ConversationAction) for item in turn.actions):
        return False
    return True


def only_reference(turn: TurnPlan) -> bool:
    actionable = [item for item in turn.actions if not isinstance(item, ConversationAction)]
    return bool(actionable) and all(isinstance(item, ReferenceAction) for item in actionable)


def acknowledgement(turn: TurnPlan, *, mode: str = "text") -> str | None:
    if not any(item.intent == "acknowledgement" for item in turn.respond_actions()):
        return None
    canned = _CANNED["acknowledgement"]
    return canned["voice" if mode == "voice" else "text"]


def canned_response(
    turn: TurnPlan,
    *,
    mode: str,
    catalog_text: str | None = None,
    user_display_name: str | None = None,
    last_assistant_text: str | None = None,
) -> str | None:
    """Program-owned replies for conversational intents. Research facts never come from here."""
    for action in turn.respond_actions():
        if action.intent == "describe_available_data" or action.response == "__describe_authorized_data__":
            return catalog_text or action.response
        if action.intent == "user_introduction":
            name = (action.response or user_display_name or "").strip()
            if name:
                return f"Hello {name}, nice to meet you!" if mode != "voice" else f"Hello {name}."
            return "Nice to meet you. What should I call you?"
        if action.intent == "user_identity":
            if user_display_name:
                return (
                    f"Your name is {user_display_name}."
                    if mode != "voice"
                    else user_display_name
                )
            return (
                "I only know a name you tell me in this conversation, "
                "or the account you're signed in with. What should I call you?"
            )
        if action.intent == "capability" and action.response == "__voice__":
            canned = _CANNED["voice_capability"]
            return canned["voice" if mode == "voice" else "text"]
        if action.intent == "capability" and action.response == "__sing__":
            return "I can't sing, but I can speak answers in a voice session."
        if action.intent == "correction":
            if action.response == "__acknowledge_correction__":
                return "Sorry about that. Thanks for the correction. I'll use it."
            previous = _first_sentence(last_assistant_text)
            if previous:
                return (
                    f"I was referring to my last answer: {previous} "
                    "If that was not what you meant, ask another way."
                )
            return "Sorry, I misunderstood. What would you like to know?"
        if action.intent == "general_conversation":
            text = (action.response or "").strip()
            if text:
                return _naturalize_assistant_name(text)
            continue
        canned = _CANNED.get(action.intent)
        if canned:
            return canned["voice" if mode == "voice" else "text"]
    return None


def _first_sentence(text: str | None) -> str:
    if not text:
        return ""
    cleaned = " ".join(text.split())
    for sep in (". ", "\n"):
        if sep in cleaned:
            cleaned = cleaned.split(sep, 1)[0].rstrip(".")
            break
    return cleaned[:180]


def _naturalize_assistant_name(text: str) -> str:
    return re.sub(r"\bNIA\b", "Nia", text)
