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
"""CPU tests for the offline credit estimators.

The scoring backend is faked so that the *true* dependence structure of the
final action on the context is known exactly. That makes it possible to assert
that each metric detects the specific failure mode it exists to catch, which no
amount of running the real pipeline on a real model would establish.
"""

from __future__ import annotations

import random

import pytest

from examples.scoring_credit.analysis import (
    auroc,
    confounds,
    localization,
    partial_spearman,
    provenance,
    spearman,
)
from examples.scoring_credit.backends import SpanScore
from examples.scoring_credit.build_multihop import RESTATEMENT_OBSERVATION, build_pair
from examples.scoring_credit.estimators import (
    RequestBook,
    as_token_ids,
    assemble,
    build_requests,
    exact_shapley,
    softmax_weights,
)
from examples.scoring_credit.trajectories import Trajectory, group_turns

MARKER = "GOLDFACT"


class FakeRenderer:
    """Word-level tokenizer that keeps rendered text recoverable from token ids."""

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.inverse: dict[int, str] = {}

    def _token(self, word: str) -> int:
        if word not in self.vocab:
            token = len(self.vocab) + 1
            self.vocab[word] = token
            self.inverse[token] = word
        return self.vocab[word]

    def encode(self, text: str) -> list[int]:
        return [self._token(w) for w in text.split()]

    def render(self, messages, add_generation_prompt: bool) -> list[int]:
        words: list[str] = []
        for msg in messages:
            words.append(f"<{msg['role']}>")
            words.extend(str(msg.get("content", "")).split())
        if add_generation_prompt:
            words.append("<assistant>")
        return [self._token(w) for w in words]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.inverse[i] for i in ids)


class FakeBackend:
    """Scores a target span from a caller-supplied value function over the context."""

    def __init__(self, renderer: FakeRenderer, value_fn):
        self.renderer = renderer
        self.value_fn = value_fn

    def score(self, pairs):
        return [SpanScore(total=self.value_fn(self.renderer.decode(ctx)), n_tokens=len(tgt)) for ctx, tgt in pairs]


def make_turn(text: str) -> list[dict]:
    return [
        {"role": "assistant", "content": text},
        {"role": "tool", "content": "ok"},
    ]


def build_trajectory(turn_texts: list[str], **kwargs) -> Trajectory:
    messages = [{"role": "user", "content": "solve the task"}]
    for text in turn_texts:
        messages.extend(make_turn(text))
    messages.append({"role": "assistant", "content": "final answer is forty two"})
    return Trajectory(
        idx=kwargs.pop("idx", 0),
        messages=messages,
        y_star=kwargs.pop("y_star", "forty two"),
        success=kwargs.pop("success", True),
        **kwargs,
    )


def run_pipeline(trajectories, value_fn, *, modes=("delete",), controls=(), shapley=0, seed=0):
    renderer = FakeRenderer()
    book = RequestBook()
    rng = random.Random(seed)
    layouts = {}
    for traj in trajectories:
        layouts[traj.idx] = build_requests(
            traj,
            renderer,
            probes=("Answer:",),
            modes=modes,
            controls=controls,
            include_final_probe=False,
            shapley_permutations=shapley,
            foreign_pool=[],
            rng=rng,
            book=book,
        )
    scores = FakeBackend(renderer, value_fn).score(book.pairs)
    resolved = book.resolve(scores)
    return [assemble(traj, layouts[traj.idx], resolved, modes=modes, controls=controls) for traj in trajectories]


# --------------------------------------------------------------------------
# Turn segmentation
# --------------------------------------------------------------------------


def test_observations_delivered_as_user_stay_inside_their_turn():
    """Agent scaffolds that return observations with role 'user' must not be reordered."""
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "act one"},
        {"role": "user", "content": "observation one"},
        {"role": "assistant", "content": "act two"},
    ]
    prompt, turns = group_turns(messages)
    assert [m["content"] for m in prompt] == ["sys", "task"]
    assert len(turns) == 2
    assert [m["content"] for m in turns[0]] == ["act one", "observation one"]


def test_trajectory_without_assistant_messages_has_no_turns():
    prompt, turns = group_turns([{"role": "user", "content": "hi"}])
    assert turns == [] and len(prompt) == 1


def test_validate_rejects_out_of_range_annotations():
    traj = build_trajectory(["a", "b"], gold_turns=[99])
    assert "gold_turns" in traj.validate()
    assert build_trajectory(["a", "b"], origin_turn=-1).validate().startswith("origin_turn")
    assert build_trajectory(["a", "b"]).validate() is None


# --------------------------------------------------------------------------
# Credit recovers a planted dependence
# --------------------------------------------------------------------------


def test_credit_peaks_at_the_turn_carrying_the_decisive_fact():
    traj = build_trajectory(["noise one", f"the {MARKER} is here", "noise two"], gold_turns=[1])
    scored = run_pipeline([traj], lambda ctx: 0.0 if MARKER in ctx else -5.0)
    credits = scored[0].credits["delete"]
    assert max(range(len(credits)), key=credits.__getitem__) == 1
    assert credits[1] > 4.0 and credits[0] == pytest.approx(0.0)


def test_localization_scores_a_perfect_ranking():
    trajs = [build_trajectory(["noise", f"{MARKER} here", "noise"], gold_turns=[1], idx=i) for i in range(4)]
    scored = run_pipeline(trajs, lambda ctx: 0.0 if MARKER in ctx else -5.0)
    result = localization(scored, lambda s: s.credits.get("delete"))
    assert result["p_at_1"] == 1.0 and result["mrr"] == 1.0


# --------------------------------------------------------------------------
# The provenance-versus-restatement gate
# --------------------------------------------------------------------------


def _redundant_marker_value(ctx: str) -> float:
    """Final action is likely as long as the fact appears anywhere in the context."""
    return 0.0 if MARKER in ctx else -5.0


def test_restatement_captures_credit_from_the_originating_turn():
    """The failure mode the gate exists to detect.

    With the fact stated once, deleting that turn destroys it and credit is
    decisive. With the fact restated later, deleting either mention alone leaves
    the other, so single-turn ablation assigns near-zero credit to the turn that
    originated the information.
    """
    base = build_trajectory(
        ["noise", f"the {MARKER} is here", "noise", "wrap up"],
        idx=0,
        pair_id="q1",
        variant="base",
        origin_turn=1,
    )
    restated = build_trajectory(
        ["noise", f"the {MARKER} is here", "noise", f"recall the {MARKER}"],
        idx=1,
        pair_id="q1",
        variant="restated",
        origin_turn=1,
        restatement_turn=3,
    )
    scored = run_pipeline([base, restated], _redundant_marker_value)
    base_credits, restated_credits = scored[0].credits["delete"], scored[1].credits["delete"]

    assert base_credits[1] > 4.0, "single mention must be decisive"
    assert restated_credits[1] == pytest.approx(0.0), "redundant mention must mask the origin"

    result = provenance(scored, lambda s: s.credits.get("delete"), kappa=1.0)
    assert result["n_pairs"] == 1
    assert result["provenance"] == 0.0
    assert result["migration"] > 0.0, "credit share must move away from the origin"


def test_shapley_splits_credit_between_redundant_mentions():
    """The correction that single-turn ablation cannot make on its own."""
    traj = build_trajectory(
        ["noise", f"the {MARKER} is here", f"recall the {MARKER}", "wrap up"],
        origin_turn=1,
        restatement_turn=2,
    )
    scored = run_pipeline([traj], _redundant_marker_value, shapley=400, seed=7)
    shapley = scored[0].shapley
    assert scored[0].credits["delete"][1] == pytest.approx(0.0)
    # An OR over two carriers splits its 5.0 of total value evenly between them.
    assert shapley[1] == pytest.approx(2.5, abs=0.35)
    assert shapley[2] == pytest.approx(2.5, abs=0.35)
    assert shapley[0] == pytest.approx(0.0, abs=0.2)


def test_monte_carlo_shapley_matches_the_exact_value():
    values = {}
    for mask in range(8):
        subset = frozenset(i for i in range(3) if mask >> i & 1)
        values[subset] = float(len(subset) ** 2)
    expected = exact_shapley(values, 3)
    assert sum(expected) == pytest.approx(values[frozenset({0, 1, 2})] - values[frozenset()])


# --------------------------------------------------------------------------
# Confounds and controls
# --------------------------------------------------------------------------


TURNS_OF_VARYING_LENGTH = ["a", "b b b b b b", "c c c", "d d d d d d d d d d", "e e"]


def test_pure_length_confound_is_fully_exposed():
    """A backend that only counts context tokens must be reported as such.

    Longer turns leave a bigger hole when removed, so credit tracks turn length
    perfectly while carrying no information about the task. The gold turn here is
    the shortest one, which makes raw agreement actively misleading.
    """
    trajs = [build_trajectory(TURNS_OF_VARYING_LENGTH, gold_turns=[0], idx=i) for i in range(5)]
    scored = run_pipeline(trajs, lambda ctx: 0.1 * len(ctx.split()))
    result = confounds(scored, lambda s: s.credits.get("delete"))
    assert result["rho_credit_vs_turn_length"] > 0.95
    assert result["r2_explained_by_length_and_position"] > 0.9
    assert result["rho_with_gold_turns"] < 0.0, "raw agreement points the wrong way"
    # Credit is a deterministic function of the controls, so nothing is left to
    # correlate and the partial statistic is correctly undefined.
    assert result["partial_rho_with_gold_turns"] is None


def test_genuine_signal_survives_partialling_out_length():
    """The control must not destroy real attribution along with the confound."""
    trajs = [
        build_trajectory(
            [t if i != 1 else f"{t} {MARKER}" for i, t in enumerate(TURNS_OF_VARYING_LENGTH)],
            gold_turns=[1],
            idx=i,
        )
        for i in range(5)
    ]
    scored = run_pipeline(trajs, lambda ctx: 0.1 * len(ctx.split()) + (6.0 if MARKER in ctx else 0.0))
    result = confounds(scored, lambda s: s.credits.get("delete"))
    assert result["partial_rho_with_gold_turns"] > 0.5


def test_matched_control_cancels_a_pure_discontinuity_effect():
    """If every intervention costs the same, controlled credit must be zero."""
    turn_texts = ["alpha", "beta", "gamma", "delta"]
    traj = build_trajectory(turn_texts)

    def value(ctx: str) -> float:
        # Only the untouched context retains every turn; any intervention at any
        # position, informative or not, pays one fixed penalty.
        intact = all(text in ctx for text in turn_texts)
        return 0.0 if intact else -3.0

    scored = run_pipeline([traj], value, modes=("replace",), controls=("control_length",))
    assert all(r == pytest.approx(3.0) for r in scored[0].credits["replace"])
    controlled = scored[0].controlled_credits("replace", "control_length")
    assert all(c == pytest.approx(0.0) for c in controlled)


# --------------------------------------------------------------------------
# Probe construction
# --------------------------------------------------------------------------


def test_final_answer_turn_is_excluded_from_probe_contexts_by_default():
    """Probing after the final action would put the answer in its own context."""
    traj = build_trajectory(["a", "b", "c"])
    scored = run_pipeline([traj], lambda ctx: -1.0)
    assert len(scored[0].phi) == traj.n_turns


def test_probe_potential_rises_when_the_target_becomes_reachable():
    traj = build_trajectory(["noise", f"found the {MARKER}", "noise"])
    scored = run_pipeline([traj], lambda ctx: -1.0 if MARKER in ctx else -6.0)
    phi = scored[0].phi
    deltas = [b - a for a, b in zip(phi, phi[1:], strict=False)]
    assert max(range(len(deltas)), key=deltas.__getitem__) == 1


# --------------------------------------------------------------------------
# Plumbing
# --------------------------------------------------------------------------


def test_request_book_deduplicates_identical_contexts():
    book = RequestBook()
    book.add("a", [1, 2, 3], [9])
    book.add("b", [1, 2, 3], [9])
    book.add("c", [1, 2], [9])
    assert len(book) == 2
    resolved = book.resolve([SpanScore(-1.0, 1), SpanScore(-2.0, 1)])
    assert resolved["a"].total == resolved["b"].total == -1.0
    assert resolved["c"].total == -2.0


def test_softmax_weights_recover_uniform_attribution_at_high_temperature():
    credits = [1.0, -2.0, 5.0]
    assert sum(softmax_weights(credits, 1.0)) == pytest.approx(1.0)
    hot = softmax_weights(credits, 1e6)
    assert all(w == pytest.approx(1 / 3, abs=1e-4) for w in hot)
    cold = softmax_weights(credits, 0.01)
    assert cold[2] == pytest.approx(1.0, abs=1e-6)


def test_correlation_helpers():
    assert auroc([1.0, 2.0], [0.0]) == 1.0
    assert auroc([0.0], [0.0]) == 0.5
    assert auroc([], [1.0]) is None
    assert spearman([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # y is a pure function of the control, so nothing survives partialling out.
    assert partial_spearman([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [[1, 2, 3, 4, 5]]) is None


# --------------------------------------------------------------------------
# Tokenizer output shapes across transformers versions
# --------------------------------------------------------------------------


class _BatchEncodingLike:
    """Stands in for the object transformers 5 returns from apply_chat_template."""

    def __init__(self, input_ids):
        self.input_ids = input_ids

    def __getitem__(self, key):
        return {"input_ids": self.input_ids}[key]


def test_token_id_normalization_across_tokenizer_return_types():
    assert as_token_ids([1, 2, 3]) == [1, 2, 3]
    assert as_token_ids([[1, 2, 3]]) == [1, 2, 3]
    assert as_token_ids({"input_ids": [1, 2, 3]}) == [1, 2, 3]
    assert as_token_ids(_BatchEncodingLike([1, 2, 3])) == [1, 2, 3]
    assert as_token_ids(_BatchEncodingLike([[1, 2, 3]])) == [1, 2, 3]


def test_non_token_id_sequences_are_rejected_rather_than_hashed():
    """A wrong type must raise, not collapse the deduplication key.

    transformers 5 changed apply_chat_template to return a BatchEncoding. Because
    that object still hashes, an unguarded request book merged every request into
    a handful of entries and the run completed with silently wrong output.
    """
    with pytest.raises(TypeError):
        as_token_ids("not tokens")
    book = RequestBook()
    with pytest.raises(TypeError):
        book.add("k", _BatchEncodingLike([1, 2]), [3])
    with pytest.raises(TypeError):
        book.add("k", [1, 2], "3")


# --------------------------------------------------------------------------
# Multi-hop trajectory construction
# --------------------------------------------------------------------------

DECISIVE = "The Spree flows through the centre of Berlin."


def hotpot_record(answer: str = "Spree") -> dict:
    """A record shaped like HotpotQA's distractor configuration."""
    return {
        "id": "q1",
        "question": "Which river runs through the 1936 Olympic host city?",
        "answer": answer,
        "supporting_facts": {"title": ["Berlin", "Spree"], "sent_id": [0, 0]},
        "context": {
            "title": ["Berlin", "Spree", "Munich", "Danube", "Rhine"],
            "sentences": [
                ["Berlin hosted the 1936 Summer Olympics."],
                ["The Spree flows through the centre of Berlin.", "It is a tributary."],
                ["Munich is in Bavaria."],
                ["The Danube flows through Vienna."],
                ["The Rhine flows through Cologne."],
            ],
        },
    }


def test_pair_differs_only_by_the_restatement_turn():
    """The provenance comparison is only valid if nothing else varies."""
    base, restated = build_pair(hotpot_record(), random.Random(0), n_distractors=2, fail=False)
    b = Trajectory(idx=0, messages=base["messages"], y_star=base["y_star"], success=True)
    r = Trajectory(idx=1, messages=restated["messages"], y_star=restated["y_star"], success=True)
    assert r.n_turns == b.n_turns + 1
    slot = restated["restatement_turn"]
    without = r.turns[:slot] + r.turns[slot + 1 :]
    assert without == b.turns


def test_annotated_indices_match_the_tool_turn_segmentation():
    """Builder indices and group_turns must agree, or every metric is off by one."""
    base, restated = build_pair(hotpot_record(), random.Random(3), n_distractors=2, fail=False)
    for row in (base, restated):
        traj = Trajectory(idx=0, messages=row["messages"], y_star=row["y_star"], success=True)
        assert traj.validate() is None
        origin_turn = traj.turns[row["origin_turn"]]
        assert DECISIVE in " ".join(m["content"] for m in origin_turn)
        for gold in row["gold_turns"]:
            assert traj.turns[gold][0]["role"] == "assistant"
    slot = restated["restatement_turn"]
    r = Trajectory(idx=0, messages=restated["messages"], y_star=restated["y_star"], success=True)
    assert DECISIVE in r.turns[slot][0]["content"]
    assert slot > restated["origin_turn"]


def test_restatement_adds_no_new_information():
    _, restated = build_pair(hotpot_record(), random.Random(5), n_distractors=2, fail=False)
    turn = Trajectory(idx=0, messages=restated["messages"], y_star="Spree", success=True).turns[
        restated["restatement_turn"]
    ]
    assert turn[1]["content"] == RESTATEMENT_OBSERVATION


def test_failed_trajectory_keeps_the_correct_probe_target():
    """y_star is the known-correct outcome even when the trajectory answered wrongly."""
    base, _ = build_pair(hotpot_record(), random.Random(1), n_distractors=2, fail=True)
    assert base["success"] is False
    assert base["y_star"] == "Spree"
    assert base["answer_given"] != "Spree"
    assert base["messages"][-1]["content"] == base["answer_given"]


def test_records_without_a_locatable_answer_are_skipped():
    record = hotpot_record(answer="Thames")  # appears in no supporting sentence
    assert build_pair(record, random.Random(0), n_distractors=2, fail=False) is None


def test_yes_no_questions_get_the_opposite_answer_on_failure():
    base, _ = build_pair(
        {
            **hotpot_record(),
            "answer": "yes",
            "context": {
                "title": ["Berlin", "Spree", "Munich"],
                "sentences": [
                    ["Berlin hosted the 1936 Summer Olympics, yes."],
                    ["The Spree flows through the centre of Berlin, yes."],
                    ["Munich is in Bavaria."],
                ],
            },
        },
        random.Random(0),
        n_distractors=1,
        fail=True,
    )
    assert base["answer_given"] == "no"
