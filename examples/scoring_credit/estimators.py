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
"""Probe potentials and counterfactual ablation credits over collected trajectories.

All quantities here are teacher-forced readouts of a frozen policy. Nothing is
sampled and no environment is touched, which is the property the whole method
rests on.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from itertools import combinations
from typing import Protocol

from .backends import SpanScore
from .trajectories import Message, Trajectory, Turn

PROBE_ENSEMBLE = (
    "Based on the work so far, output the final answer now. Answer:",
    "Stop and give your best final answer immediately. Answer:",
    "What is the final answer? Reply with the answer only. Answer:",
    "Ignore any remaining steps. State the final answer. Answer:",
    "Given everything above, the final answer is:",
    "Conclude now. Final answer:",
    "Report the final answer without further tool use. Answer:",
    "Provide the answer you would submit right now. Answer:",
)

NULL_TURN_TEXT = "Let me reconsider the task."
FILLER_SENTENCE = "I will continue working on the task as instructed. "

# Interventions that destroy a turn's information, versus controls that perturb
# the context comparably without removing task-relevant content. The difference
# between the two is what separates attribution from discontinuity surprise.
ABLATION_MODES = ("delete", "replace")
CONTROL_MODES = ("control_null", "control_length", "control_foreign")


class Renderer(Protocol):
    """Turns messages into token ids, and text into token ids."""

    def render(self, messages: list[Message], add_generation_prompt: bool) -> list[int]: ...

    def encode(self, text: str) -> list[int]: ...


class ChatRenderer:
    """Renderer backed by a Hugging Face tokenizer's chat template."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def render(self, messages: list[Message], add_generation_prompt: bool) -> list[int]:
        return self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)


class RequestBook:
    """Accumulates (context, target) scoring requests and deduplicates them.

    Deduplication is not cosmetic. Shapley permutations revisit the same
    coalitions constantly, and every repeat would otherwise cost a real forward
    pass over tens of thousands of tokens.
    """

    def __init__(self):
        self._pairs: list[tuple[list[int], list[int]]] = []
        self._index: dict[tuple, int] = {}
        self.keys: dict = {}

    def add(self, key, context: list[int], target: list[int]) -> None:
        signature = (tuple(context), tuple(target))
        slot = self._index.get(signature)
        if slot is None:
            slot = len(self._pairs)
            self._index[signature] = slot
            self._pairs.append((context, target))
        self.keys[key] = slot

    @property
    def pairs(self) -> list[tuple[list[int], list[int]]]:
        return self._pairs

    def __len__(self) -> int:
        return len(self._pairs)

    def resolve(self, scores: list[SpanScore]) -> dict:
        return {key: scores[slot] for key, slot in self.keys.items()}


@dataclass
class TrajectoryScores:
    """Everything the analyses need from one trajectory."""

    idx: int
    success: bool
    n_turns: int
    phi: list[float] = field(default_factory=list)
    phi_std: list[float] = field(default_factory=list)
    credits: dict[str, list[float]] = field(default_factory=dict)
    controls: dict[str, list[float]] = field(default_factory=dict)
    shapley: list[float] | None = None
    turn_lens: list[int] = field(default_factory=list)
    turn_starts: list[int] = field(default_factory=list)
    gold_turns: list[int] | None = None
    pair_id: str | None = None
    variant: str | None = None
    origin_turn: int | None = None
    restatement_turn: int | None = None

    def controlled_credits(self, mode: str, control: str) -> list[float] | None:
        """Raw credit minus the matched control drop at the same position."""
        raw, ctrl = self.credits.get(mode), self.controls.get(control)
        if raw is None or ctrl is None:
            return None
        return [r - c for r, c in zip(raw, ctrl, strict=True)]


def _length_matched_filler(renderer: Renderer, n_tokens: int) -> str:
    """Neutral text whose token count approximates ``n_tokens``.

    Used by the length-matched control: if credit merely tracked how many tokens
    an intervention removes, this control would reproduce it.
    """
    if n_tokens <= 0:
        return NULL_TURN_TEXT
    per_sentence = max(len(renderer.encode(FILLER_SENTENCE)), 1)
    repeats = max(1, round(n_tokens / per_sentence))
    return (FILLER_SENTENCE * repeats).strip()


def _turn_spans(traj: Trajectory, renderer: Renderer) -> tuple[list[int], list[int]]:
    """Token length and start offset of every turn, measured incrementally.

    Measuring by difference of rendered prefixes rather than by rendering a turn
    in isolation keeps template overhead attributed consistently.
    """
    lens, starts = [], []
    previous = len(renderer.render(traj.prompt_msgs, add_generation_prompt=False))
    for t in range(traj.n_turns):
        starts.append(previous)
        current = len(renderer.render(traj.context(traj.turns[: t + 1]), add_generation_prompt=False))
        lens.append(max(current - previous, 0))
        previous = current
    return lens, starts


def _substitute(traj: Trajectory, t: int, replacement: Turn | None) -> list[Turn]:
    """Non-final turns with turn ``t`` removed or replaced."""
    kept = traj.turns[:-1]
    if replacement is None:
        return kept[:t] + kept[t + 1 :]
    return kept[:t] + [replacement] + kept[t + 1 :]


def build_requests(
    traj: Trajectory,
    renderer: Renderer,
    *,
    probes: tuple[str, ...],
    modes: tuple[str, ...],
    controls: tuple[str, ...],
    include_final_probe: bool,
    shapley_permutations: int,
    foreign_pool: list[Turn],
    rng: random.Random,
    book: RequestBook,
) -> dict:
    """Queue every scoring request this trajectory needs. Returns its layout."""
    n = traj.n_turns
    y_star_ids = renderer.encode(" " + str(traj.y_star).strip())
    final_text = traj.final_turn[0].get("content", "")
    final_ids = renderer.encode(final_text)
    if not final_ids or not y_star_ids:
        return {"skip": "empty final action or target"}

    # --- probe potentials -------------------------------------------------
    # Boundaries run 0..n-1 by default, that is up to but NOT including the
    # final action turn. Including it would put the answer itself in the probe
    # context, making Phi trivially high on successes and inflating every
    # separability number computed from it.
    n_boundaries = n + 1 if include_final_probe else n
    for t in range(n_boundaries):
        base = traj.context(traj.turns[:t])
        for p_idx, probe in enumerate(probes):
            ctx = renderer.render(base + [{"role": "user", "content": probe}], True)
            book.add(("phi", traj.idx, t, p_idx), ctx, y_star_ids)

    # --- ablation credits -------------------------------------------------
    base_ctx = renderer.render(traj.context(traj.turns[:-1]), True)
    book.add(("abl_base", traj.idx), base_ctx, final_ids)

    turn_lens, turn_starts = _turn_spans(traj, renderer)
    for t in range(n - 1):
        for mode in modes:
            replacement = None
            if mode == "replace":
                replacement = [{"role": "assistant", "content": NULL_TURN_TEXT}]
            ctx = renderer.render(traj.context(_substitute(traj, t, replacement)), True)
            book.add(("abl", traj.idx, mode, t), ctx, final_ids)

        for control in controls:
            if control == "control_null":
                filler = [{"role": "assistant", "content": NULL_TURN_TEXT}]
            elif control == "control_length":
                filler = [{"role": "assistant", "content": _length_matched_filler(renderer, turn_lens[t])}]
            elif control == "control_foreign":
                if not foreign_pool:
                    continue
                filler = rng.choice(foreign_pool)
            else:
                raise ValueError(f"unknown control mode {control!r}")
            ctx = renderer.render(traj.context(_substitute(traj, t, filler)), True)
            book.add(("ctl", traj.idx, control, t), ctx, final_ids)

    # --- Shapley coalitions ----------------------------------------------
    permutations: list[tuple[int, ...]] = []
    if shapley_permutations > 0 and n >= 2:
        indices = list(range(n - 1))
        for _ in range(shapley_permutations):
            order = indices[:]
            rng.shuffle(order)
            permutations.append(tuple(order))
        for order in permutations:
            for k in range(len(order) + 1):
                subset = tuple(sorted(order[:k]))
                kept = [traj.turns[i] for i in subset]
                ctx = renderer.render(traj.context(kept), True)
                book.add(("coal", traj.idx, subset), ctx, final_ids)

    return {
        "n_turns": n,
        "n_boundaries": n_boundaries,
        "turn_lens": turn_lens,
        "turn_starts": turn_starts,
        "permutations": permutations,
    }


def assemble(traj: Trajectory, layout: dict, resolved: dict, *, modes, controls) -> TrajectoryScores:
    """Turn resolved span scores into potentials, credits and Shapley values."""
    n = layout["n_turns"]
    scores = TrajectoryScores(
        idx=traj.idx,
        success=traj.success,
        n_turns=n,
        turn_lens=layout["turn_lens"],
        turn_starts=layout["turn_starts"],
        gold_turns=traj.gold_turns,
        pair_id=traj.pair_id,
        variant=traj.variant,
        origin_turn=traj.origin_turn,
        restatement_turn=traj.restatement_turn,
    )

    # Potentials are length-normalized so that targets of different lengths are
    # comparable; the ensemble spread is kept as a probe-sensitivity readout.
    for t in range(layout["n_boundaries"]):
        per_probe = [resolved[key].mean for key in resolved if key[0] == "phi" and key[1] == traj.idx and key[2] == t]
        mean = sum(per_probe) / len(per_probe)
        variance = sum((v - mean) ** 2 for v in per_probe) / max(len(per_probe) - 1, 1)
        scores.phi.append(mean)
        scores.phi_std.append(variance**0.5)

    # Credits use the SUMMED log-prob of the final action, not the per-token
    # mean. The target span is identical across a trajectory's variants, so the
    # difference is what matters; dividing by its length would rescale c_t per
    # trajectory and silently change the effective softmax temperature.
    base = resolved[("abl_base", traj.idx)].total
    for mode in modes:
        scores.credits[mode] = [base - resolved[("abl", traj.idx, mode, t)].total for t in range(n - 1)]
    for control in controls:
        key_present = all(("ctl", traj.idx, control, t) in resolved for t in range(n - 1))
        if key_present:
            scores.controls[control] = [base - resolved[("ctl", traj.idx, control, t)].total for t in range(n - 1)]

    permutations = layout["permutations"]
    if permutations:
        totals = [0.0] * (n - 1)
        for order in permutations:
            for k, turn in enumerate(order):
                before = resolved[("coal", traj.idx, tuple(sorted(order[:k])))].total
                after = resolved[("coal", traj.idx, tuple(sorted(order[: k + 1])))].total
                totals[turn] += after - before
        scores.shapley = [total / len(permutations) for total in totals]
    return scores


def softmax_weights(credits: list[float], kappa: float) -> list[float]:
    """Temperature-controlled normalization of credits into turn weights."""
    if not credits:
        return []
    scaled = [c / kappa for c in credits]
    ceiling = max(scaled)
    exps = [pow(2.718281828459045, s - ceiling) for s in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def exact_shapley(values: dict[frozenset, float], n: int) -> list[float]:
    """Exact Shapley values from a full coalition table. Used only in tests."""
    from math import factorial

    result = []
    for t in range(n):
        others = [i for i in range(n) if i != t]
        total = 0.0
        for size in range(len(others) + 1):
            weight = factorial(size) * factorial(n - size - 1) / factorial(n)
            for subset in combinations(others, size):
                s = frozenset(subset)
                total += weight * (values[s | {t}] - values[s])
        result.append(total)
    return result
