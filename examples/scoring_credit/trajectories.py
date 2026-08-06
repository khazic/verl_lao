# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Trajectory schema and turn segmentation for offline credit scoring."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

Message = dict[str, Any]
Turn = list[Message]


def group_turns(messages: list[Message]) -> tuple[list[Message], list[Turn]]:
    """Split a chat transcript into a task prompt and a list of agent turns.

    The prompt is every message preceding the first assistant message. Each turn
    then begins at an assistant message and absorbs every following message until
    the next assistant message, so tool results and environment observations stay
    attached to the action that produced them regardless of the role they carry.

    Keeping mid-conversation user messages inside their turn matters: several
    agent scaffolds deliver observations with role ``user``, and hoisting those
    into the prompt would silently reorder the transcript.
    """
    first_assistant = next((i for i, m in enumerate(messages) if m.get("role") == "assistant"), None)
    if first_assistant is None:
        return list(messages), []

    prompt_msgs = list(messages[:first_assistant])
    turns: list[Turn] = []
    for msg in messages[first_assistant:]:
        if msg.get("role") == "assistant":
            turns.append([msg])
        else:
            turns[-1].append(msg)
    return prompt_msgs, turns


@dataclass
class Trajectory:
    """One collected multi-turn trajectory plus the labels the analyses need."""

    idx: int
    messages: list[Message]
    y_star: str
    success: bool
    gold_turns: list[int] | None = None
    # Provenance-versus-restatement pairing (Section 4.2 of the paper).
    pair_id: str | None = None
    variant: str | None = None  # "base" | "restated"
    origin_turn: int | None = None
    restatement_turn: int | None = None

    prompt_msgs: list[Message] = field(default_factory=list, repr=False)
    turns: list[Turn] = field(default_factory=list, repr=False)

    def __post_init__(self):
        self.prompt_msgs, self.turns = group_turns(self.messages)

    @property
    def n_turns(self) -> int:
        return len(self.turns)

    @property
    def final_turn(self) -> Turn | None:
        return self.turns[-1] if self.turns else None

    def context(self, keep: list[Turn]) -> list[Message]:
        """Prompt followed by the given turns, in order."""
        msgs = list(self.prompt_msgs)
        for turn in keep:
            msgs.extend(turn)
        return msgs

    def validate(self) -> str | None:
        """Return a human-readable reason this trajectory is unusable, else None."""
        if self.n_turns < 2:
            return "fewer than two turns, nothing to attribute"
        if not str(self.y_star).strip():
            return "empty y_star"
        for name, value in (
            ("origin_turn", self.origin_turn),
            ("restatement_turn", self.restatement_turn),
        ):
            if value is not None and not 0 <= value < self.n_turns:
                return f"{name}={value} out of range for {self.n_turns} turns"
        if self.gold_turns:
            bad = [t for t in self.gold_turns if not 0 <= t < self.n_turns]
            if bad:
                return f"gold_turns {bad} out of range for {self.n_turns} turns"
        return None


def load_trajectories(path: str, limit: int = 0) -> tuple[list[Trajectory], list[str]]:
    """Read the JSONL trajectory file, returning usable trajectories and skip reasons.

    Every line is one trajectory::

        {"messages": [...], "y_star": "...", "success": true,
         "gold_turns": [2],                       # optional, localization ground truth
         "pair_id": "q17", "variant": "restated", # optional, provenance pairing
         "origin_turn": 2, "restatement_turn": 5}

    Malformed lines are skipped rather than aborting the run, because these files
    are usually produced by a rollout dump that may contain truncated trajectories.
    """
    trajectories: list[Trajectory] = []
    skipped: list[str] = []
    with open(path) as handle:
        for line_no, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                skipped.append(f"line {line_no}: bad JSON ({exc.msg})")
                continue
            try:
                traj = Trajectory(
                    idx=len(trajectories),
                    messages=raw["messages"],
                    y_star=raw["y_star"],
                    success=bool(raw["success"]),
                    gold_turns=raw.get("gold_turns"),
                    pair_id=raw.get("pair_id"),
                    variant=raw.get("variant"),
                    origin_turn=raw.get("origin_turn"),
                    restatement_turn=raw.get("restatement_turn"),
                )
            except KeyError as exc:
                skipped.append(f"line {line_no}: missing field {exc}")
                continue
            reason = traj.validate()
            if reason:
                skipped.append(f"line {line_no}: {reason}")
                continue
            trajectories.append(traj)
            if limit and len(trajectories) >= limit:
                break
    return trajectories, skipped
