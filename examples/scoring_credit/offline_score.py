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
"""Offline validation of the scoring-based credit estimators.

Runs the studies that gate the method: whether ablation credit lands on the turn
that originated the decisive information rather than on a later restatement,
whether it survives controlling for turn length and position, how well it
localizes annotated decisive turns, and whether the probe potential separates
successful from failed trajectories. No policy updates, no environment.

Example::

    python -m examples.scoring_credit.offline_score \\
        --model /path/to/Qwen3-8B --data trajs.jsonl --out runs/offline \\
        --tp 1 --modes delete replace --controls control_length control_null
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict

from .analysis import summarize
from .backends import HFBackend, VLLMBackend, max_abs_disagreement
from .estimators import (
    ABLATION_MODES,
    CONTROL_MODES,
    PROBE_ENSEMBLE,
    ChatRenderer,
    RequestBook,
    assemble,
    build_requests,
)
from .trajectories import load_trajectories


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--data", required=True, help="trajectory JSONL")
    ap.add_argument("--out", required=True, help="output prefix; writes <out>.jsonl and <out>.summary.json")
    ap.add_argument("--tp", type=int, default=1, help="vLLM tensor parallel size")
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0, help="only score the first N trajectories")
    ap.add_argument(
        "--modes",
        nargs="+",
        default=["delete", "replace"],
        choices=ABLATION_MODES,
        help="ablation variants to score; both share the unablated prefix, so scoring "
        "them together is far cheaper than two separate runs",
    )
    ap.add_argument("--controls", nargs="+", default=["control_length"], choices=CONTROL_MODES)
    ap.add_argument("--probe-ensemble", type=int, default=3, help="number of probe phrasings")
    ap.add_argument(
        "--include-final-probe",
        action="store_true",
        help="also probe after the final action turn. Off by default: that context "
        "contains the answer itself, which makes the potential trivially high on "
        "successes and inflates every separability number derived from it",
    )
    ap.add_argument(
        "--shapley-permutations",
        type=int,
        default=0,
        help="Monte Carlo permutations for Shapley credit; 0 disables. Cost is "
        "roughly this many extra coalition scorings per turn",
    )
    ap.add_argument("--kappa", type=float, default=1.0, help="softmax temperature used for weight-share metrics")
    ap.add_argument("--phi-floor", type=float, default=-6.0, help="potential floor, in mean log-prob per token")
    ap.add_argument(
        "--thinking",
        choices=["auto", "off", "on"],
        default="auto",
        help="how to handle a hybrid-thinking policy. Targets are scored right after "
        "the generation prompt, which on such a model is where it expects to open a "
        "reasoning block, so leaving thinking enabled measures willingness to skip "
        "thinking rather than belief in the target. auto disables it when the chat "
        "template supports it",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--check-backend-agreement",
        type=int,
        default=0,
        help="score the first N requests with an eager HF forward pass as well and "
        "report the largest disagreement. Run this once before trusting a go/no-go",
    )
    ap.add_argument(
        "--reference-device",
        default="cpu",
        help="device for the HF reference backend. CPU by default: it scores only a "
        "small sample, and a float32 copy of the policy does not fit alongside the "
        "vLLM engine that is already holding the GPU",
    )
    ap.add_argument(
        "--reference-dtype",
        default="float32",
        help="dtype of the HF reference backend. float32 by default because bfloat16 "
        "scoring noise is larger than the backend gap being measured",
    )
    ap.add_argument("--dry-run", action="store_true", help="build requests and report cost, then stop")
    return ap.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    trajectories, skipped = load_trajectories(args.data, limit=args.limit)
    if skipped:
        print(f"[data] skipped {len(skipped)} trajectories; first few:")
        for reason in skipped[:5]:
            print(f"  - {reason}")
    if not trajectories:
        raise SystemExit("no usable trajectories")
    print(
        f"[data] {len(trajectories)} trajectories, mean turns "
        f"{sum(t.n_turns for t in trajectories) / len(trajectories):.1f}"
    )

    from transformers import AutoTokenizer

    renderer = ChatRenderer(AutoTokenizer.from_pretrained(args.model), thinking=args.thinking)
    tail = renderer.generation_prompt_tail()
    print(f"[template] thinking={args.thinking} kwargs={renderer.template_kwargs}")
    print(f"[template] targets are scored at: ...{tail!r}")
    probes = PROBE_ENSEMBLE[: max(1, args.probe_ensemble)]
    modes = tuple(args.modes)
    controls = tuple(args.controls)

    # Foreign-turn control pool: real turns from other trajectories, spliced in at
    # the same position, so the control perturbs the conversation as much as a
    # genuine intervention does without removing this trajectory's information.
    foreign_pool = [t.turns[len(t.turns) // 2] for t in trajectories if t.n_turns >= 3]

    book = RequestBook()
    layouts = {}
    for traj in trajectories:
        layout = build_requests(
            traj,
            renderer,
            probes=probes,
            modes=modes,
            controls=controls,
            include_final_probe=args.include_final_probe,
            shapley_permutations=args.shapley_permutations,
            foreign_pool=foreign_pool,
            rng=rng,
            book=book,
        )
        if "skip" not in layout:
            layouts[traj.idx] = layout

    total_tokens = sum(len(ctx) + len(tgt) for ctx, tgt in book.pairs)
    print(
        f"[plan] {len(book)} unique scoring requests, {total_tokens / 1e6:.2f}M tokens, "
        f"{len(book) / max(len(layouts), 1):.1f} per trajectory"
    )
    if args.dry_run:
        return

    backend = VLLMBackend(args.model, tensor_parallel_size=args.tp, max_model_len=args.max_model_len)
    started = time.time()
    scores = backend.score(book.pairs)
    elapsed = time.time() - started
    print(f"[score] {len(scores)} requests in {elapsed:.1f}s ({total_tokens / max(elapsed, 1e-9) / 1e3:.1f}k tok/s)")

    # Persist before anything optional runs. The scoring pass is the expensive
    # part of the job, and an auxiliary diagnostic must never be able to discard
    # it: an earlier version ran the reference check first and lost eighteen
    # minutes of completed work when that check hit an out-of-memory error.
    resolved = book.resolve(scores)
    scored = [
        assemble(traj, layouts[traj.idx], resolved, modes=modes, controls=controls)
        for traj in trajectories
        if traj.idx in layouts
    ]

    with open(f"{args.out}.jsonl", "w") as handle:
        for record in scored:
            handle.write(json.dumps(asdict(record)) + "\n")
    print(f"[write] {len(scored)} scored trajectories -> {args.out}.jsonl")

    backend_gap, backend_error = None, None
    if args.check_backend_agreement:
        n = min(args.check_backend_agreement, len(book.pairs))
        try:
            reference = HFBackend(args.model, device=args.reference_device, dtype=args.reference_dtype).score(
                book.pairs[:n]
            )
            backend_gap = max_abs_disagreement(scores[:n], reference)
            print(f"[check] max |vLLM - HF| mean log-prob over {n} requests: {backend_gap:.5f}")
        except Exception as exc:  # diagnostic only; never fatal
            backend_error = f"{type(exc).__name__}: {exc}"
            print(f"[check] reference backend unavailable, skipping: {backend_error}")

    summary = summarize(scored, modes=modes, controls=controls, kappa=args.kappa, phi_floor=args.phi_floor)
    summary["config"] = {
        "model": args.model,
        "modes": list(modes),
        "controls": list(controls),
        "probe_ensemble": len(probes),
        "include_final_probe": args.include_final_probe,
        "shapley_permutations": args.shapley_permutations,
        "kappa": args.kappa,
        "phi_floor": args.phi_floor,
        "seed": args.seed,
        "thinking": args.thinking,
        "template_kwargs": renderer.template_kwargs,
        "generation_prompt_tail": tail,
        "n_skipped_inputs": len(skipped),
        "backend_max_disagreement": backend_gap,
        "backend_check_error": backend_error,
        "scoring_seconds": elapsed,
        "unique_requests": len(book),
        "scored_tokens": total_tokens,
    }
    with open(f"{args.out}.summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
