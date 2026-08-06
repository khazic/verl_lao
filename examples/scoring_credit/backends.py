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
"""Teacher-forced span scoring backends.

A backend answers exactly one question: given a context token sequence and a
target token sequence, what is the summed log-probability the policy assigns to
the target continuation? Everything else in this recipe is arithmetic on top.

Two backends are provided deliberately. vLLM is the throughput path used for the
real runs; Hugging Face is a slow reference used by ``--check-backend-agreement``
to confirm that the offline numbers match what an eager forward pass produces.
Without that check an offline go/no-go decision could rest on a scoring path that
differs from the one the trainer will later use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SpanScore:
    """Summed and per-token log-probability of a target span."""

    total: float
    n_tokens: int

    @property
    def mean(self) -> float:
        return self.total / max(self.n_tokens, 1)


class ScoringBackend(Protocol):
    """Scores (context, target) token-id pairs in batches."""

    def score(self, pairs: list[tuple[list[int], list[int]]]) -> list[SpanScore]: ...


class VLLMBackend:
    """Batched span scoring through a vLLM engine's prompt-logprob path.

    Prefix caching is what makes this affordable: the ablation variants of one
    trajectory share every token before the intervention point, so the engine
    recomputes only the suffix, which is the ``TL/2`` accounting the paper claims.
    """

    def __init__(self, model: str, tensor_parallel_size: int = 1, max_model_len: int | None = None, **kwargs):
        from vllm import LLM

        self.llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            enable_prefix_caching=True,
            **kwargs,
        )

    def score(self, pairs: list[tuple[list[int], list[int]]]) -> list[SpanScore]:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        if not pairs:
            return []
        prompts = [TokensPrompt(prompt_token_ids=ctx + tgt) for ctx, tgt in pairs]
        # max_tokens=1 because we only need the prompt log-probs; nothing is sampled.
        params = SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0.0)
        outputs = self.llm.generate(prompts, params, use_tqdm=False)

        scores = []
        for (ctx, tgt), out in zip(pairs, outputs, strict=True):
            span = out.prompt_logprobs[len(ctx) : len(ctx) + len(tgt)]
            total = 0.0
            for token_id, position in zip(tgt, span, strict=True):
                if position is None:
                    # Only possible at index 0, which an empty context would cause.
                    raise ValueError("empty context: target token has no conditional log-prob")
                # Index by the actual token id rather than taking an arbitrary
                # dict entry, which would silently read a top-k candidate.
                entry = position.get(token_id)
                if entry is None:
                    raise ValueError(f"vLLM returned no log-prob for token id {token_id}")
                total += entry.logprob
            scores.append(SpanScore(total=total, n_tokens=len(tgt)))
        return scores


class HFBackend:
    """Reference span scoring with a plain Hugging Face forward pass."""

    def __init__(self, model: str, device: str = "cuda", dtype: str = "bfloat16", batch_size: int = 4):
        import torch
        from transformers import AutoModelForCausalLM

        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=getattr(torch, dtype), device_map=device
        ).eval()

    def score(self, pairs: list[tuple[list[int], list[int]]]) -> list[SpanScore]:
        torch = self.torch
        results: list[SpanScore] = []
        for start in range(0, len(pairs), self.batch_size):
            chunk = pairs[start : start + self.batch_size]
            seqs = [ctx + tgt for ctx, tgt in chunk]
            width = max(len(s) for s in seqs)
            # Left-pad so that every sequence ends at the same position; the
            # attention mask keeps the padding out of the context.
            input_ids = torch.full((len(seqs), width), 0, dtype=torch.long)
            attn = torch.zeros((len(seqs), width), dtype=torch.long)
            for row, seq in enumerate(seqs):
                input_ids[row, width - len(seq) :] = torch.tensor(seq, dtype=torch.long)
                attn[row, width - len(seq) :] = 1
            input_ids = input_ids.to(self.device)
            attn = attn.to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids=input_ids, attention_mask=attn).logits
            logprobs = torch.log_softmax(logits.float(), dim=-1)

            for row, (_, tgt) in enumerate(chunk):
                total = 0.0
                for offset, token_id in enumerate(tgt):
                    # Position of this target token in the padded row, and the
                    # logits slot that predicts it (one step earlier).
                    pos = width - len(tgt) + offset
                    total += logprobs[row, pos - 1, token_id].item()
                results.append(SpanScore(total=total, n_tokens=len(tgt)))
        return results


def max_abs_disagreement(a: list[SpanScore], b: list[SpanScore]) -> float:
    """Largest absolute difference in mean log-prob between two backends' scores."""
    return max((abs(x.mean - y.mean) for x, y in zip(a, b, strict=True)), default=math.nan)
