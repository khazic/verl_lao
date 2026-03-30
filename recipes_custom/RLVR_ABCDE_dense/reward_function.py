# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""
Reward function for multiple-choice (ABCDE) tasks.
"""

import re

DEFAULT_CHOICES = ("A", "B", "C", "D", "E")
BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
CHOICE_PATTERN = re.compile(
    r"(?:answer|option|choice|答案|选项)\s*[:：=是为]?\s*\**([A-E])\**",
    re.IGNORECASE,
)


def _extract_boxed_answer(text: str) -> str:
    matches = BOXED_PATTERN.findall(text)
    return matches[-1] if matches else ""


def _scan_first_choice(text: str, valid_choices=DEFAULT_CHOICES) -> str:
    """Scan text character by character and return the first valid choice letter."""
    for char in (text or "").upper():
        if char in valid_choices:
            return char
    return ""


def _normalize_choice(text: str, valid_choices=DEFAULT_CHOICES) -> str:
    """Exact match only — used for CHOICE_PATTERN results and ground truth."""
    text = (text or "").strip().upper()
    if text in valid_choices:
        return text
    return ""


def extract_choice(text: str, valid_choices=DEFAULT_CHOICES) -> str:
    """
    Extract a single-letter choice from model output.
    Priority:
      1. \\boxed{X} — scan within the boxed content (already constrained)
      2. Explicit declaration: "Answer: C" / "答案：B"
    Returns "" if no answer can be reliably extracted.
    """
    text = str(text or "")
    boxed = _extract_boxed_answer(text)
    if boxed:
        candidate = _scan_first_choice(boxed, valid_choices)
        if candidate:
            return candidate
    match = CHOICE_PATTERN.search(text)
    if match:
        candidate = _normalize_choice(match.group(1), valid_choices)
        if candidate:
            return candidate
    return ""


def mcq_reward_function(data_source, solution_str, ground_truth, extra_info=None):
    try:
        model_choice = extract_choice(solution_str)
        gold_choice = _normalize_choice(ground_truth)
        return 1 if model_choice and gold_choice and model_choice == gold_choice else 0
    except Exception:
        print(ground_truth, solution_str)
        return 0
