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
"""Build answer-convertible trajectories with credit ground truth from multi-hop QA.

Each question becomes a retrieval-agent transcript: one turn per search, each
returning one paragraph, ending in a turn that states the answer. Because the
source dataset annotates which paragraphs actually support the answer, the turns
that retrieved them are known decisive turns, and the rest are known distractors.
That is the ground truth the localization and confound studies need and that
natural rollouts cannot supply.

The provenance study needs something natural rollouts cannot supply either:
matched pairs. For each question we emit a ``base`` trajectory in which the
decisive fact appears exactly once, and a ``restated`` trajectory identical to it
except for one later turn that repeats that fact without adding anything. Any
difference in where credit lands between the two is caused by the restatement
alone.

Example::

    python -m examples.scoring_credit.build_multihop \\
        --out data/multihop.jsonl --limit 200 --distractors 3 --failure-rate 0.4

Caveat worth stating plainly: these transcripts are templated rather than
sampled from the policy, so they are somewhat off-distribution. They are the
right instrument for the studies that need ground truth, and the wrong one for
measuring how the probe behaves on the policy's own rollouts. Use real rollouts
for that.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter

SYSTEM_PROMPT = (
    "You are a research assistant with access to a search tool. "
    "Work step by step: issue one search at a time, read the result, "
    "and stop as soon as you can answer the question."
)

SEARCH_INTENTS = (
    "I should look up {title}.",
    "Let me search for {title}.",
    "Next I need information about {title}.",
    "I will check {title}.",
    "Searching for {title} now.",
)

RESTATEMENT_INTENTS = (
    "To recap what I already found: {fact}",
    "Restating the key detail from earlier: {fact}",
    "Just to be clear about what the search returned: {fact}",
    "Noting again what I established above: {fact}",
)

RESTATEMENT_OBSERVATION = "noop() -> ok"


def _sentences_of(context: dict, title: str) -> list[str]:
    titles = context["title"]
    return list(context["sentences"][titles.index(title)]) if title in titles else []


def _paragraph(sentences: list[str]) -> str:
    return " ".join(s.strip() for s in sentences).strip()


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def _sentence_containing(sentences: list[str], answer: str) -> str | None:
    """The sentence that states the answer, which is what a restatement repeats."""
    needle = _normalize(answer)
    if not needle:
        return None
    return next((s.strip() for s in sentences if needle in _normalize(s)), None)


def _wrong_answer(record: dict, gold_titles: list[str], rng: random.Random) -> str:
    """A plausible incorrect answer, taken from a distractor paragraph's title.

    Failed trajectories must end in something the policy could actually have
    produced. A distractor title is wrong but topically adjacent, which is the
    shape of a real retrieval failure.
    """
    candidates = [t for t in record["context"]["title"] if t not in gold_titles]
    answer = record["answer"]
    if answer.lower() in ("yes", "no"):
        return "no" if answer.lower() == "yes" else "yes"
    return rng.choice(candidates) if candidates else "unknown"


def _turn(intent: str, observation: str) -> list[dict]:
    return [
        {"role": "assistant", "content": intent},
        {"role": "tool", "content": observation},
    ]


def build_pair(record: dict, rng: random.Random, *, n_distractors: int, fail: bool) -> list[dict] | None:
    """Build the base and restated trajectories for one question.

    Returns None when the record lacks the structure the studies require, which
    is preferable to emitting a trajectory whose ground truth is a guess.
    """
    context = record["context"]
    gold_titles = list(dict.fromkeys(record["supporting_facts"]["title"]))
    gold_titles = [t for t in gold_titles if t in context["title"]]
    if len(gold_titles) < 2:
        return None

    distractor_titles = [t for t in context["title"] if t not in gold_titles]
    rng.shuffle(distractor_titles)
    distractor_titles = distractor_titles[:n_distractors]
    if not distractor_titles:
        return None

    answer = str(record["answer"])
    # The decisive paragraph is the supporting one that actually states the
    # answer; without it there is nothing for a restatement to repeat.
    decisive_title, decisive_sentence = None, None
    for title in gold_titles:
        sentence = _sentence_containing(_sentences_of(context, title), answer)
        if sentence:
            decisive_title, decisive_sentence = title, sentence
            break
    if decisive_title is None:
        return None

    # Order the searches so the decisive one is never last: a restatement has to
    # fit after it, and a decisive final turn would collide with the answer turn.
    others = [t for t in gold_titles if t != decisive_title] + distractor_titles
    rng.shuffle(others)
    cut = rng.randint(0, max(len(others) - 1, 0))
    ordered = others[:cut] + [decisive_title] + others[cut:]

    turns, gold_turns = [], []
    for index, title in enumerate(ordered):
        sentences = _sentences_of(context, title)
        if not sentences:
            return None
        intent = rng.choice(SEARCH_INTENTS).format(title=title)
        turns.append(_turn(intent, f"search({title}) -> {_paragraph(sentences)}"))
        if title in gold_titles:
            gold_turns.append(index)
    origin_turn = ordered.index(decisive_title)

    final_answer = _wrong_answer(record, gold_titles, rng) if fail else answer
    answer_turn = [{"role": "assistant", "content": final_answer}]
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": record["question"]},
    ]

    def assemble(extra_turn=None, at=None):
        body = list(turns)
        if extra_turn is not None:
            body.insert(at, extra_turn)
        messages = list(prompt)
        for turn in body:
            messages.extend(turn)
        messages.extend(answer_turn)
        return messages

    common = {
        "question_id": record.get("id"),
        "y_star": answer,
        "success": not fail,
        "answer_given": final_answer,
    }

    base = {
        **common,
        "messages": assemble(),
        "gold_turns": gold_turns,
        "pair_id": str(record.get("id")),
        "variant": "base",
        "origin_turn": origin_turn,
    }

    # The restatement goes strictly after the decisive turn and strictly before
    # the answer turn, and adds no information the transcript did not contain.
    slot = rng.randint(origin_turn + 1, len(turns))
    restatement = _turn(
        rng.choice(RESTATEMENT_INTENTS).format(fact=decisive_sentence),
        RESTATEMENT_OBSERVATION,
    )
    restated = {
        **common,
        "messages": assemble(restatement, slot),
        "gold_turns": [t if t < slot else t + 1 for t in gold_turns],
        "pair_id": str(record.get("id")),
        "variant": "restated",
        "origin_turn": origin_turn,
        "restatement_turn": slot,
    }
    return [base, restated]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output JSONL path")
    ap.add_argument("--dataset", default="hotpotqa/hotpot_qa")
    ap.add_argument("--config", default="distractor")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=200, help="number of questions to convert")
    ap.add_argument("--distractors", type=int, default=3, help="non-supporting paragraphs retrieved per trajectory")
    ap.add_argument(
        "--failure-rate",
        type=float,
        default=0.4,
        help="share of questions whose trajectory ends in a wrong answer, so the "
        "probe study has both classes and blame assignment has failures to rank",
    )
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    from datasets import load_dataset

    rng = random.Random(args.seed)
    dataset = (
        load_dataset(args.dataset, args.config, split=args.split)
        if args.config
        else load_dataset(args.dataset, split=args.split)
    )

    written, skipped, turn_counts = [], 0, Counter()
    for record in dataset:
        if len({r["pair_id"] for r in written}) >= args.limit:
            break
        pair = build_pair(record, rng, n_distractors=args.distractors, fail=rng.random() < args.failure_rate)
        if pair is None:
            skipped += 1
            continue
        written.extend(pair)
        for row in pair:
            turn_counts[sum(1 for m in row["messages"] if m["role"] == "assistant")] += 1

    with open(args.out, "w") as handle:
        for row in written:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    successes = sum(r["success"] for r in written)
    print(f"[build] wrote {len(written)} trajectories ({len(written) // 2} pairs) to {args.out}")
    print(f"[build] skipped {skipped} records lacking usable ground truth")
    print(f"[build] success {successes} / failure {len(written) - successes}")
    print(f"[build] turns per trajectory: {sorted(turn_counts.items())}")


if __name__ == "__main__":
    main()
