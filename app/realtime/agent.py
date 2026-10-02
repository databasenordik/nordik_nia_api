from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from app.concurrency.cancellation import CancellationToken
from app.config import get_settings
from app.execution.turn import AssistantTurnService
from app.realtime.states import VoiceState, VoiceStateMachine
from app.realtime.turn_hook import AssistantTurnHook
from app.security.access_scope import AccessScope
from app.security.livekit_token import (
    AGENT_NAME,
    conversation_id_from_metadata,
    scope_from_metadata,
    selected_file_id_from_metadata,
)
from app.tts.session import BargeInController, SpeechSession
from app.tts.voice_control import VoicePreference, requested_voice_preference

logger = logging.getLogger("nia.realtime")

PREEMPTIVE_GENERATION = False
NotifyFn = Callable[[dict[str, Any]], Awaitable[None]]
SwitchVoiceFn = Callable[[VoicePreference], Awaitable[None]]


class ResearchVoiceAgent:
    """Voice session that answers only after Assistant Core finishes retrieval."""

    def __init__(
        self,
        hook: AssistantTurnHook,
        notify: NotifyFn | None = None,
        speech: SpeechSession | None = None,
        switch_voice: SwitchVoiceFn | None = None,
    ) -> None:
        self.hook = hook
        self.states = VoiceStateMachine()
        self.pending_reply: str | None = None
        self._notify = notify
        self.speech = speech
        self._switch_voice = switch_voice
        self.cancellation = speech.cancellation if speech else CancellationToken()
        self.barge_in = BargeInController(speech, self.cancellation) if speech else None

    def _new_turn_cancel(self) -> CancellationToken:
        self.cancellation = CancellationToken()
        if self.speech is not None:
            self.speech.cancellation = self.cancellation
            self.speech.interrupted = False
            self.barge_in = BargeInController(self.speech, self.cancellation)
        return self.cancellation

    async def handle_final_transcript(
        self,
        transcript: str,
        scope: AccessScope,
        *,
        conversation_id: str | None = None,
        selected_file_id: int | None = None,
        on_spoken_phrase=None,
    ) -> str:
        self._new_turn_cancel()
        if self.states.state in {VoiceState.READY, VoiceState.LISTENING, VoiceState.USER_SPEAKING}:
            if self.states.state != VoiceState.PROCESSING:
                try:
                    self.states.transition(VoiceState.PROCESSING)
                except ValueError:
                    self.states.transition(VoiceState.LISTENING)
                    self.states.transition(VoiceState.PROCESSING)
        await self.publish_transcript(transcript, final=True)
        await self._emit({"type": "voice_state", "state": VoiceState.PROCESSING.value})
        if self.speech is not None:
            self.speech.begin_turn()

        preference = requested_voice_preference(transcript)
        if preference is not None:
            return await self.handle_voice_preference(
                preference,
                conversation_id=conversation_id,
                on_spoken_phrase=on_spoken_phrase,
                greet=True,
                begin_turn=False,
            )

        async def speak(phrase: str) -> None:
            self.mark_speaking()
            await self._emit({"type": "voice_state", "state": VoiceState.ASSISTANT_SPEAKING.value})
            await self._emit({"type": "assistant_delta", "text": phrase})
            if on_spoken_phrase is not None:
                await on_spoken_phrase(phrase)
            if self.speech is not None:
                await self.speech.enqueue_text(phrase)

        result = await self.hook.on_user_turn_completed(
            transcript,
            scope,
            conversation_id=conversation_id,
            selected_file_id=selected_file_id,
            cancellation=self.cancellation,
            on_spoken_phrase=speak if (on_spoken_phrase or self.speech) else None,
        )
        self.pending_reply = result.core.spoken_text or result.reply
        for citation in result.core.citations:
            item = next((row for row in result.core.evidence if row.get("source_id") == citation), None)
            await self._emit(
                {
                    "type": "citation",
                    "source_id": citation,
                    "fields": (item or {}).get("fields") or {},
                }
            )
        await self._emit(
            {
                "type": "assistant_reply",
                "text": result.reply,
                "conversation_id": result.core.conversation_id or conversation_id,
                "citations": result.core.citations,
                "evidence": result.core.evidence,
                "interrupted": result.core.interrupted,
            }
        )
        self.mark_listening()
        await self._emit({"type": "voice_state", "state": VoiceState.LISTENING.value})
        return result.reply

    async def handle_voice_preference(
        self,
        preference: VoicePreference,
        *,
        conversation_id: str | None = None,
        on_spoken_phrase=None,
        greet: bool = True,
        begin_turn: bool = True,
    ) -> str:
        """Apply a session voice choice; optionally greet in the new voice."""
        if self._switch_voice is not None:
            await self._switch_voice(preference)
        await self._emit({"type": "voice_preference", "preference": preference})
        if not greet:
            return ""

        if begin_turn:
            self._new_turn_cancel()
            if self.states.state in {
                VoiceState.READY,
                VoiceState.LISTENING,
                VoiceState.USER_SPEAKING,
            }:
                try:
                    self.states.transition(VoiceState.PROCESSING)
                except ValueError:
                    pass
            await self._emit({"type": "voice_state", "state": VoiceState.PROCESSING.value})
            if self.speech is not None:
                self.speech.begin_turn()

        greeting = "Hi, I'm here."
        self.pending_reply = greeting
        self.mark_speaking()
        await self._emit({"type": "voice_state", "state": VoiceState.ASSISTANT_SPEAKING.value})
        await self._emit({"type": "assistant_delta", "text": greeting})
        if on_spoken_phrase is not None:
            await on_spoken_phrase(greeting)
        if self.speech is not None:
            await self.speech.enqueue_text(greeting)
        await self._emit(
            {
                "type": "assistant_reply",
                "text": greeting,
                "conversation_id": conversation_id,
                "citations": [],
                "evidence": [],
                "interrupted": False,
            }
        )
        self.mark_listening()
        await self._emit({"type": "voice_state", "state": VoiceState.LISTENING.value})
        return greeting

    async def publish_transcript(self, text: str, *, final: bool) -> None:
        if not text.strip():
            return
        await self._emit(
            {
                "type": "transcript",
                "status": "final" if final else "partial",
                "text": text,
            }
        )

    async def interrupt(self, reason: str = "barge-in") -> None:
        self.mark_interrupted()
        if self.barge_in is not None:
            await self.barge_in.interrupt(reason)
        await self._emit({"type": "voice_state", "state": VoiceState.INTERRUPTED.value})
        self.mark_listening()

    def mark_speaking(self) -> None:
        if self.states.state == VoiceState.PROCESSING:
            self.states.transition(VoiceState.ASSISTANT_SPEAKING)

    def mark_listening(self) -> None:
        if self.states.state in {VoiceState.ASSISTANT_SPEAKING, VoiceState.INTERRUPTED, VoiceState.READY}:
            if self.states.state == VoiceState.READY:
                self.states.transition(VoiceState.LISTENING)
                return
            self.states.transition(VoiceState.LISTENING)

    def mark_interrupted(self) -> None:
        if self.states.state in {VoiceState.ASSISTANT_SPEAKING, VoiceState.PROCESSING}:
            self.states.transition(VoiceState.INTERRUPTED)

    async def _emit(self, payload: dict[str, Any]) -> None:
        if self._notify is None:
            return
        await self._notify(payload)


def session_options() -> dict[str, Any]:
    """LiveKit AgentSession options. No Cloud inference, no preemptive generation."""
    settings = get_settings()
    return {
        "preemptive_generation": {"enabled": PREEMPTIVE_GENERATION},
        "stt": "whisper.cpp",
        "vad": "silero",
        "turn_detection": "vad",
        "tts": settings.tts_provider.strip().lower() or "xai",
        "llm": "assistant-core",
        "endpointing": {
            "min_delay": settings.voice_endpoint_min_delay_seconds,
            "max_delay": settings.voice_endpoint_max_delay_seconds,
        },
    }


async def build_turn_service() -> AssistantTurnService:
    from app.data_gateway.memory import MemoryDataGateway
    from app.data_gateway.repository import AssistantDataGateway
    from app.db.pool import init_pool
    from app.llm.factory import get_reasoning_provider
    from app.memory.store import default_memory_store

    settings = get_settings()
    try:
        pool = await init_pool()
        if settings.app_env == "production":
            from app.db.privilege_selftest import assert_runtime_isolation

            await assert_runtime_isolation()
        gateway = AssistantDataGateway(pool)
    except Exception:
        if settings.app_env == "production":
            logger.exception("voice worker database initialization failed")
            raise
        logger.warning("voice worker using in-memory gateway; Postgres unavailable")
        gateway = MemoryDataGateway()
    return AssistantTurnService(
        gateway,
        store=default_memory_store(),
        reasoner=get_reasoning_provider(),
    )


def message_transcript(new_message: Any) -> str:
    text = getattr(new_message, "text_content", None)
    if text:
        return str(text)
    content = getattr(new_message, "content", "") or ""
    if isinstance(content, list):
        return " ".join(str(item) for item in content)
    return str(content)


async def _agent_entry(ctx: Any) -> None:
    """LiveKit job entrypoint. Must stay a module-level function so the worker
    subprocess pool can pickle it by reference."""
    from livekit.agents import Agent, AgentSession, room_io
    from livekit.plugins import silero

    from app.stt.livekit_adapter import create_livekit_stt
    from app.stt.whisper_cpp import WhisperCppSTTProvider

    ctx.log_context_fields = {"room": ctx.room.name}
    await ctx.connect()
    participant = await ctx.wait_for_participant()
    metadata = participant.metadata
    scope = scope_from_metadata(metadata)
    conversation_id = conversation_id_from_metadata(metadata)
    selected_file_id = selected_file_id_from_metadata(metadata)

    async def notify(payload: dict[str, Any]) -> None:
        await ctx.room.local_participant.publish_data(
            json.dumps(payload).encode("utf-8"),
            reliable=True,
        )

    from app.tts.factory import get_tts_provider
    from app.tts.livekit_adapter import create_livekit_tts

    tts_provider = get_tts_provider()
    speech = SpeechSession(tts_provider, cancellation=CancellationToken())
    try:
        await speech.connect()
    except Exception:
        logger.warning(
            "TTS connect failed; voice continues text-only (transcript + streamed answer)",
            exc_info=True,
        )
    hook = AssistantTurnHook(await build_turn_service())
    settings = get_settings()

    async def switch_voice(preference: VoicePreference) -> None:
        setter = getattr(tts_provider, "set_voice", None)
        if not callable(setter):
            logger.warning("The configured TTS provider cannot switch voices")
            return
        if settings.tts_provider.strip().lower() == "kokoro":
            selected = (
                settings.kokoro_female_voice
                if preference == "female"
                else settings.kokoro_male_voice
            )
        else:
            selected = (
                settings.xai_tts_female_voice
                if preference == "female"
                else settings.xai_tts_male_voice
            )
        await setter(selected)

    voice = ResearchVoiceAgent(hook, notify=notify, speech=speech, switch_voice=switch_voice)
    try:
        voice.states.transition(VoiceState.CONNECTING)
        voice.states.transition(VoiceState.READY)
        voice.states.transition(VoiceState.LISTENING)
    except ValueError:
        pass

    class CoreAgent(Agent):
        def __init__(self) -> None:
            super().__init__(
                instructions=(
                    "Speak the Assistant Core answer only. Do not invent database facts. "
                    "Do not read raw citation IDs aloud."
                )
            )

        async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
            transcript = message_transcript(new_message)

            async def speak_phrase(phrase: str) -> None:
                await session.say(phrase, allow_interruptions=True)

            reply = await voice.handle_final_transcript(
                transcript,
                scope,
                conversation_id=conversation_id,
                selected_file_id=selected_file_id,
                on_spoken_phrase=speak_phrase,
            )
            turn_ctx.add_message(role="assistant", content=reply)

    # livekit-agents 1.6.10 reads turn_handling=TurnHandlingOptions. The older
    # top-level endpointing kwargs are deprecated and no longer drive commits.
    stt_provider = WhisperCppSTTProvider()
    settings = get_settings()
    session = AgentSession(
        stt=create_livekit_stt(stt_provider),
        tts=create_livekit_tts(tts_provider),
        vad=silero.VAD.load(),
        turn_handling={
            "turn_detection": "vad",
            "endpointing": {
                "min_delay": settings.voice_endpoint_min_delay_seconds,
                "max_delay": settings.voice_endpoint_max_delay_seconds,
            },
            "interruption": {"enabled": True},
            "preemptive_generation": {"enabled": False},
        },
        llm=None,
    )

    @session.on("user_state_changed")
    def _on_user_state(ev) -> None:
        new_state = getattr(ev, "new_state", None) or getattr(ev, "state", None)
        if str(new_state).lower().endswith("speaking") and voice.states.state is VoiceState.ASSISTANT_SPEAKING:
            asyncio.create_task(voice.interrupt())

    @session.on("user_input_transcribed")
    def _on_transcript(ev) -> None:
        text = getattr(ev, "transcript", None) or getattr(ev, "text", "") or ""
        is_final = bool(getattr(ev, "is_final", False))
        if text and not is_final:
            asyncio.create_task(voice.publish_transcript(str(text), final=False))

    def _on_data(*args: Any) -> None:
        packet = args[0] if args else None
        raw = getattr(packet, "data", packet)
        try:
            if isinstance(raw, bytes | bytearray):
                payload = json.loads(raw.decode("utf-8"))
            elif isinstance(raw, str):
                payload = json.loads(raw)
            else:
                return
        except Exception:
            return
        if payload.get("type") == "interrupt":
            asyncio.create_task(voice.interrupt())
        elif payload.get("type") == "set_voice" and payload.get("preference") in {
            "male",
            "female",
        }:
            preference = payload["preference"]

            async def apply_voice_preference() -> None:
                async def speak_phrase(phrase: str) -> None:
                    await session.say(phrase, allow_interruptions=True)

                await voice.handle_voice_preference(
                    preference,
                    conversation_id=conversation_id,
                    on_spoken_phrase=speak_phrase,
                    greet=bool(payload.get("greet", True)),
                )

            asyncio.create_task(apply_voice_preference())

    ctx.room.on("data_received", _on_data)
    try:
        await session.start(agent=CoreAgent(), room=ctx.room, room_options=room_io.RoomOptions())
    finally:
        try:
            await stt_provider.aclose()
        finally:
            await speech.close()


def run_livekit_worker() -> None:
    """Entrypoint used by `python -m app.realtime.worker`. Imports LiveKit lazily."""
    from livekit.agents import AgentServer, cli

    server = AgentServer()
    server.rtc_session(agent_name=AGENT_NAME)(_agent_entry)
    cli.run_app(server)
