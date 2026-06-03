from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal


MemberType = Literal["operation", "machine", "event"]


@dataclass(slots=True)
class MotifMember:
    member_type: MemberType
    member_id: str
    role: str


@dataclass(slots=True)
class MotifCandidate:
    family: str
    anchor_type: str
    anchor_id: str
    operation_ids: list[int]
    machine_ids: list[int]
    members: list[MotifMember]
    urgency_score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    motif_id: str | None = None

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "anchor_type": self.anchor_type,
            "anchor_id": self.anchor_id,
            "operation_ids": sorted(set(int(op_id) for op_id in self.operation_ids)),
            "machine_ids": sorted(set(int(machine_id) for machine_id in self.machine_ids)),
            "members": sorted(
                (
                    member.member_type,
                    str(member.member_id),
                    member.role,
                )
                for member in self.members
            ),
        }

    def canonical_hash(self) -> str:
        payload = self.canonical_payload()
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def finalize_motif_ids(motifs: list[MotifCandidate]) -> list[MotifCandidate]:
    for motif in motifs:
        motif.motif_id = motif.canonical_hash()
    return motifs


def serialize_motif(motif: MotifCandidate) -> dict[str, Any]:
    return {
        "motif_id": motif.motif_id or motif.canonical_hash(),
        "family": motif.family,
        "anchor_type": motif.anchor_type,
        "anchor_id": motif.anchor_id,
        "operation_ids": sorted(set(int(op_id) for op_id in motif.operation_ids)),
        "machine_ids": sorted(set(int(machine_id) for machine_id in motif.machine_ids)),
        "members": [
            {
                "member_type": member.member_type,
                "member_id": member.member_id,
                "role": member.role,
            }
            for member in motif.members
        ],
        "urgency_score": float(motif.urgency_score),
        "metadata": motif.metadata,
    }
