from __future__ import annotations

from enum import StrEnum


class VoiceState(StrEnum):
    DISCONNECTED = "Disconnected"
    CONNECTING = "Connecting"
    READY = "Ready"
    LISTENING = "Listening"
    USER_SPEAKING = "User speaking"
    PROCESSING = "Processing"
    ASSISTANT_SPEAKING = "Assistant speaking"
    INTERRUPTED = "Interrupted"
    RECONNECTING = "Reconnecting"
    ERROR = "Error"


_TRANSITIONS: dict[VoiceState, set[VoiceState]] = {
    VoiceState.DISCONNECTED: {VoiceState.CONNECTING},
    VoiceState.CONNECTING: {VoiceState.READY, VoiceState.ERROR, VoiceState.DISCONNECTED},
    VoiceState.READY: {VoiceState.LISTENING, VoiceState.RECONNECTING, VoiceState.DISCONNECTED, VoiceState.ERROR},
    VoiceState.LISTENING: {
        VoiceState.USER_SPEAKING,
        VoiceState.PROCESSING,
        VoiceState.RECONNECTING,
        VoiceState.DISCONNECTED,
        VoiceState.ERROR,
    },
    VoiceState.USER_SPEAKING: {
        VoiceState.PROCESSING,
        VoiceState.LISTENING,
        VoiceState.INTERRUPTED,
        VoiceState.RECONNECTING,
        VoiceState.DISCONNECTED,
        VoiceState.ERROR,
    },
    VoiceState.PROCESSING: {
        VoiceState.ASSISTANT_SPEAKING,
        VoiceState.LISTENING,
        VoiceState.INTERRUPTED,
        VoiceState.RECONNECTING,
        VoiceState.DISCONNECTED,
        VoiceState.ERROR,
    },
    VoiceState.ASSISTANT_SPEAKING: {
        VoiceState.LISTENING,
        VoiceState.INTERRUPTED,
        VoiceState.RECONNECTING,
        VoiceState.DISCONNECTED,
        VoiceState.ERROR,
    },
    VoiceState.INTERRUPTED: {VoiceState.LISTENING, VoiceState.USER_SPEAKING, VoiceState.DISCONNECTED},
    VoiceState.RECONNECTING: {VoiceState.READY, VoiceState.LISTENING, VoiceState.ERROR, VoiceState.DISCONNECTED},
    VoiceState.ERROR: {VoiceState.CONNECTING, VoiceState.DISCONNECTED},
}


class VoiceStateMachine:
    def __init__(self) -> None:
        self.state = VoiceState.DISCONNECTED

    def transition(self, target: VoiceState) -> VoiceState:
        if target == self.state:
            return self.state
        allowed = _TRANSITIONS[self.state]
        if target not in allowed:
            raise ValueError(f"illegal voice transition {self.state} -> {target}")
        self.state = target
        return self.state
