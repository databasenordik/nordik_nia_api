from __future__ import annotations

from app.llm.schemas import MemoryUpdate, SynthesisAnswer
from app.retrieval.types import EvidencePacket
from app.security.access_scope import AccessScope


def allowed_source_ids(packet: EvidencePacket) -> set[str]:
    ids = {item.source_id for item in packet.items}
    for branch in packet.branches:
        ids.update(item.source_id for item in branch.items)
    return ids


def validate_synthesis(
    answer: SynthesisAnswer,
    packet: EvidencePacket,
    scope: AccessScope,
) -> SynthesisAnswer:
    allowed = allowed_source_ids(packet)
    citations = [item for item in answer.citations if item in allowed]
    last_ids = [item for item in answer.memory_update.last_source_ids if item in allowed]
    file_ids = [file_id for file_id in answer.memory_update.active_file_ids if file_id in scope.allowed_file_ids]
    return SynthesisAnswer(
        answer=answer.answer,
        citations=citations,
        inference=answer.inference,
        memory_update=MemoryUpdate(
            active_file_ids=file_ids,
            active_topics=answer.memory_update.active_topics[:8],
            last_source_ids=last_ids,
        ),
    )


def render_evidence_block(packet: EvidencePacket) -> str:
    if packet.branches:
        lines: list[str] = []
        for branch in packet.branches:
            lines.append(f"ACTION {branch.action_id} GOAL {branch.goal}")
            lines.append("COMPUTED FACTS")
            if branch.facts:
                lines.extend(f"- {fact}" for fact in branch.facts)
            else:
                lines.append("- none")
            lines.append("EVIDENCE")
            if not branch.items:
                lines.append("- none")
            for item in branch.items:
                _append_item(lines, item)
        return "\n".join(lines)
    lines = ["COMPUTED FACTS"]
    if packet.facts:
        lines.extend(f"- {fact}" for fact in packet.facts)
    else:
        lines.append("- none")
    lines.append("EVIDENCE")
    if not packet.items:
        lines.append("- none")
    for item in packet.items:
        _append_item(lines, item)
    return "\n".join(lines)


def _append_item(lines: list[str], item) -> None:
    lines.append(f"[{item.source_id}]")
    for key, value in item.fields.items():
        lines.append(f"  {key}: {value}")
    for key, value in item.raw_fields.items():
        lines.append(f"  raw.{key}: {value}")
