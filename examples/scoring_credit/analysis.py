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
"""Metrics for the offline validation studies.

Summary keys are named after the columns of the paper's tables so that filling
those tables is transcription rather than interpretation.
"""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import fmean

import numpy as np

from .estimators import TrajectoryScores, softmax_weights


def auroc(positive: list[float], negative: list[float]) -> float | None:
    """Probability that a random positive outranks a random negative."""
    if not positive or not negative:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


def _ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks, so ties do not bias the correlation."""
    order = values.argsort()
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sums = np.zeros(len(counts))
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    """Rank correlation, or None when it is undefined."""
    if len(x) < 3 or len(x) != len(y):
        return None
    rx, ry = _ranks(np.asarray(x, dtype=float)), _ranks(np.asarray(y, dtype=float))
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _residualize(target: np.ndarray, controls: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(target)), controls])
    coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
    return target - design @ coeffs


def _is_degenerate(residual: np.ndarray, original: np.ndarray) -> bool:
    """True when the controls explain the variable up to floating-point noise.

    An exact ``std() == 0`` test is not enough: least squares leaves residuals of
    order 1e-16 on perfectly collinear inputs, and correlating two such noise
    vectors returns a confident and meaningless 1.0.
    """
    return bool(residual.std() <= 1e-8 * (original.std() + 1e-12))


def partial_spearman(x: list[float], y: list[float], controls: list[list[float]]) -> float | None:
    """Rank correlation of x and y after linearly removing the control variables.

    This is the quantity that answers whether ablation credit still tracks the
    ground truth once turn length and position are accounted for.
    """
    if len(x) < 4 or len(x) != len(y):
        return None
    rx = _ranks(np.asarray(x, dtype=float))
    ry = _ranks(np.asarray(y, dtype=float))
    rc = np.column_stack([_ranks(np.asarray(c, dtype=float)) for c in controls])
    ex, ey = _residualize(rx, rc), _residualize(ry, rc)
    if _is_degenerate(ex, rx) or _is_degenerate(ey, ry):
        return None
    return float(np.corrcoef(ex, ey)[0, 1])


def variance_explained(y: list[float], controls: list[list[float]]) -> float | None:
    """R^2 of predicting ranked y from the ranked control variables alone."""
    if len(y) < 4:
        return None
    ry = _ranks(np.asarray(y, dtype=float))
    rc = np.column_stack([_ranks(np.asarray(c, dtype=float)) for c in controls])
    residual = _residualize(ry, rc)
    total = ((ry - ry.mean()) ** 2).sum()
    if total == 0:
        return None
    return float(1.0 - (residual**2).sum() / total)


def _mean(values: list[float | None]) -> float | None:
    kept = [v for v in values if v is not None and not math.isnan(v)]
    return fmean(kept) if kept else None


def localization(scored: list[TrajectoryScores], credits_of) -> dict:
    """Precision@1, precision@3 and MRR of a credit ranking against gold turns."""
    hits1, hits3, reciprocal = [], [], []
    for s in scored:
        credits = credits_of(s)
        if not credits or not s.gold_turns:
            continue
        gold = {t for t in s.gold_turns if t < len(credits)}
        if not gold:
            continue
        order = sorted(range(len(credits)), key=lambda t: credits[t], reverse=True)
        hits1.append(int(order[0] in gold))
        hits3.append(int(any(t in gold for t in order[:3])))
        rank = next((i for i, t in enumerate(order) if t in gold), None)
        reciprocal.append(1.0 / (rank + 1) if rank is not None else 0.0)
    if not hits1:
        return {"n": 0, "p_at_1": None, "p_at_3": None, "mrr": None}
    return {
        "n": len(hits1),
        "p_at_1": fmean(hits1),
        "p_at_3": fmean(hits3),
        "mrr": fmean(reciprocal),
    }


def confounds(scored: list[TrajectoryScores], credits_of) -> dict:
    """How much of the credit ranking is explained by turn length and position."""
    rho_len, rho_pos, r2, partial_gold, raw_gold = [], [], [], [], []
    for s in scored:
        credits = credits_of(s)
        if not credits:
            continue
        lens = s.turn_lens[: len(credits)]
        pos = list(range(len(credits)))
        rho_len.append(spearman(credits, lens))
        rho_pos.append(spearman(credits, pos))
        r2.append(variance_explained(credits, [lens, pos]))
        if s.gold_turns:
            indicator = [1.0 if t in s.gold_turns else 0.0 for t in range(len(credits))]
            if 0 < sum(indicator) < len(indicator):
                raw_gold.append(spearman(credits, indicator))
                partial_gold.append(partial_spearman(credits, indicator, [lens, pos]))
    return {
        "rho_credit_vs_turn_length": _mean(rho_len),
        "rho_credit_vs_position": _mean(rho_pos),
        "r2_explained_by_length_and_position": _mean(r2),
        "rho_with_gold_turns": _mean(raw_gold),
        "partial_rho_with_gold_turns": _mean(partial_gold),
    }


def provenance(scored: list[TrajectoryScores], credits_of, kappa: float) -> dict:
    """Does credit land on the turn that originated the information or on a restatement?

    ``provenance`` is the headline: the fraction of restated variants whose credit
    peaks at the originating turn. ``migration`` compares the originating turn's
    normalized share against its share in the paired base variant, so a positive
    value is credit that the restatement pulled away from the origin.
    """
    by_pair: dict[str, dict[str, TrajectoryScores]] = defaultdict(dict)
    for s in scored:
        if s.pair_id and s.variant:
            by_pair[s.pair_id][s.variant] = s

    peaks_at_origin, origin_beats, shares, migrations = [], [], [], []
    for pair in by_pair.values():
        restated = pair.get("restated")
        if restated is None or restated.origin_turn is None or restated.restatement_turn is None:
            continue
        credits = credits_of(restated)
        origin, restatement = restated.origin_turn, restated.restatement_turn
        if not credits or origin >= len(credits) or restatement >= len(credits):
            continue

        peaks_at_origin.append(int(max(range(len(credits)), key=credits.__getitem__) == origin))
        origin_beats.append(int(credits[origin] > credits[restatement]))

        weights = softmax_weights(credits, kappa)
        pair_mass = weights[origin] + weights[restatement]
        if pair_mass > 0:
            shares.append(weights[restatement] / pair_mass)

        base = pair.get("base")
        if base is not None and base.origin_turn is not None:
            base_credits = credits_of(base)
            if base_credits and base.origin_turn < len(base_credits):
                base_weights = softmax_weights(base_credits, kappa)
                migrations.append(base_weights[base.origin_turn] - weights[origin])

    return {
        "n_pairs": len(peaks_at_origin),
        "provenance": fmean(peaks_at_origin) if peaks_at_origin else None,
        "origin_beats_restatement": fmean(origin_beats) if origin_beats else None,
        "restatement_share_of_pair": fmean(shares) if shares else None,
        "migration": fmean(migrations) if migrations else None,
        "n_with_base_partner": len(migrations),
    }


def probe_separability(scored: list[TrajectoryScores], phi_floor: float) -> dict:
    """Success-versus-failure separation of the probe potential."""
    rise_success, rise_fail, jump_success, jump_fail = [], [], [], []
    below, spreads, all_phi = [], [], []
    for s in scored:
        if len(s.phi) < 2:
            continue
        deltas = [b - a for a, b in zip(s.phi, s.phi[1:], strict=False)]
        rise = s.phi[-1] - s.phi[0]
        peak = max(deltas)
        (rise_success if s.success else rise_fail).append(rise)
        (jump_success if s.success else jump_fail).append(peak)
        below.extend(int(v < phi_floor) for v in s.phi)
        spreads.extend(s.phi_std)
        all_phi.extend(s.phi)

    quantiles = {q: float(np.quantile(all_phi, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)} if all_phi else {}
    return {
        "n_success": len(rise_success),
        "n_fail": len(rise_fail),
        "auroc_total_rise": auroc(rise_success, rise_fail),
        "auroc_max_jump": auroc(jump_success, jump_fail),
        "mean_rise_success": _mean(rise_success),
        "mean_rise_fail": _mean(rise_fail),
        "fraction_below_floor": fmean(below) if below else None,
        "probe_ensemble_sd": _mean(spreads),
        "phi_quantiles": quantiles,
    }


def agreement(scored: list[TrajectoryScores], credits_of) -> dict:
    """Rank correlation between ablation credit and the probe difference.

    Both readouts come from the same policy, so agreement is a health diagnostic
    and not independent validation. Disagreement is the informative direction.
    """
    correlations = []
    for s in scored:
        credits = credits_of(s)
        if not credits or len(s.phi) < 2:
            continue
        deltas = [b - a for a, b in zip(s.phi, s.phi[1:], strict=False)]
        width = min(len(credits), len(deltas))
        if width >= 3:
            correlations.append(spearman(credits[:width], deltas[:width]))
    kept = [c for c in correlations if c is not None]
    return {
        "n": len(kept),
        "mean_rank_correlation": _mean(kept),
        "fraction_negative": fmean([int(c < 0) for c in kept]) if kept else None,
    }


def summarize(scored: list[TrajectoryScores], *, modes, controls, kappa: float, phi_floor: float) -> dict:
    """Full summary, one block per estimator variant."""
    summary: dict = {
        "n_trajectories": len(scored),
        "n_success": sum(s.success for s in scored),
        "mean_turns": _mean([float(s.n_turns) for s in scored]),
        "probe": probe_separability(scored, phi_floor),
        "estimators": {},
    }

    variants: dict[str, callable] = {mode: (lambda s, m=mode: s.credits.get(m)) for mode in modes}
    for mode in modes:
        for control in controls:
            variants[f"{mode}-minus-{control}"] = lambda s, m=mode, c=control: s.controlled_credits(m, c)
    if any(s.shapley for s in scored):
        variants["shapley"] = lambda s: s.shapley

    for name, credits_of in variants.items():
        if not any(credits_of(s) for s in scored):
            continue
        summary["estimators"][name] = {
            "provenance": provenance(scored, credits_of, kappa),
            "localization": localization(scored, credits_of),
            "confounds": confounds(scored, credits_of),
            "agreement_with_probe": agreement(scored, credits_of),
        }
    return summary
