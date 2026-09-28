"""Metrics produced by the RL-ZVP estimator."""

from __future__ import annotations

import torch


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    denom = mask.sum()
    if denom.item() == 0:
        return 0.0
    return ((values * mask).sum() / denom).detach().float().item()


def _selected_token_mean(values: torch.Tensor, response_mask: torch.Tensor, rows: torch.Tensor) -> float:
    token_rows = rows.unsqueeze(-1).expand_as(response_mask)
    return _masked_mean(values, response_mask * token_rows)


def compute_rl_zvp_metrics(
    *,
    scores: torch.Tensor,
    response_mask: torch.Tensor,
    token_entropy: torch.Tensor,
    advantages: torch.Tensor,
    group_types: torch.Tensor,
) -> dict[str, float]:
    """Return scalar metrics; group types are 0=mixed, 1=positive, 2=negative."""
    positive = group_types.eq(1)
    negative = group_types.eq(2)
    mixed = group_types.eq(0)
    lengths = response_mask.sum(dim=-1)
    correct = scores > 0
    wrong = ~correct

    def row_mean(values: torch.Tensor, rows: torch.Tensor) -> float:
        if not rows.any():
            return 0.0
        return values[rows].float().mean().item()

    return {
        "zvp/all_ratio": (positive | negative).float().mean().item(),
        "zvp/positive_ratio": positive.float().mean().item(),
        "zvp/negative_ratio": negative.float().mean().item(),
        "zvp/mixed_ratio": mixed.float().mean().item(),
        "entropy/mean": _masked_mean(token_entropy, response_mask),
        "entropy/positive_zvp_mean": _selected_token_mean(token_entropy, response_mask, positive),
        "entropy/negative_zvp_mean": _selected_token_mean(token_entropy, response_mask, negative),
        "advantage/mean": _masked_mean(advantages, response_mask),
        "advantage/abs_mean": _masked_mean(advantages.abs(), response_mask),
        "advantage/positive_zvp_mean": _selected_token_mean(advantages, response_mask, positive),
        "advantage/negative_zvp_mean": _selected_token_mean(advantages, response_mask, negative),
        "advantage/mixed_mean": _selected_token_mean(advantages, response_mask, mixed),
        "response_length/mean": lengths.float().mean().item(),
        "response_length/correct_mean": row_mean(lengths, correct),
        "response_length/wrong_mean": row_mean(lengths, wrong),
        "reward/mean": scores.float().mean().item(),
    }
