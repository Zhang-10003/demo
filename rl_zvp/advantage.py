"""Readable tensor implementation of the RL-ZVP advantage estimator."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import torch

from .metrics import compute_rl_zvp_metrics

_LAST_METRICS: dict[str, float] = {}


def _cfg_value(config: Any, path: tuple[str, ...]) -> Any:
    value = config
    for key in path:
        if isinstance(value, dict):
            value = value[key]
        else:
            value = getattr(value, key)
    return value


def compute_grpo_like_verl(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: Sequence[Any],
    epsilon: float,
) -> torch.Tensor:
    """Match verl v0.9.1 GRPO exactly.

    Args:
        token_level_rewards: `[B, T_response]`.
        response_mask: `[B, T_response]`.
        index: length-B prompt/group identifiers.
        epsilon: normalization epsilon from experiment config.
    """
    scores = token_level_rewards.sum(dim=-1).clone()
    grouped: dict[Any, list[torch.Tensor]] = defaultdict(list)
    for row, uid in enumerate(index):
        grouped[uid].append(scores[row])

    means: dict[Any, torch.Tensor] = {}
    stds: dict[Any, torch.Tensor] = {}
    for uid, values in grouped.items():
        if len(values) == 1:
            means[uid] = scores.new_tensor(0.0)
            stds[uid] = scores.new_tensor(1.0)
        else:
            group_scores = torch.stack(values)
            means[uid] = group_scores.mean()
            stds[uid] = group_scores.std()  # verl v0.9.1 uses unbiased torch.std.

    for row, uid in enumerate(index):
        scores[row] = (scores[row] - means[uid]) / (stds[uid] + epsilon)
    return scores.unsqueeze(-1) * response_mask


def compute_rl_zvp_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: Sequence[Any],
    token_entropy: torch.Tensor,
    *,
    alpha: float,
    zero_variance_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute RL-ZVP token advantages.

    All tensors have shape `[B, T_response]`. Mixed groups use pinned verl GRPO;
    all-correct groups use `+alpha*H`; all-wrong groups use
    `-alpha*(H_i,max-H_i,t)`. Entropy is detached and padding is always zero.
    """
    if token_level_rewards.shape != response_mask.shape:
        raise ValueError("token_level_rewards and response_mask must have identical shape")
    if token_entropy.shape != response_mask.shape:
        raise ValueError("token_entropy and response_mask must have identical shape")
    if len(index) != token_level_rewards.shape[0]:
        raise ValueError("index length must equal batch size")
    if alpha < 0 or zero_variance_epsilon <= 0:
        raise ValueError("alpha must be nonnegative and zero_variance_epsilon positive")

    with torch.no_grad():
        mask = response_mask.to(dtype=token_level_rewards.dtype)
        entropy = token_entropy.detach().to(dtype=token_level_rewards.dtype) * mask
        scores = token_level_rewards.sum(dim=-1)
        grpo = compute_grpo_like_verl(token_level_rewards, mask, index, zero_variance_epsilon)
        advantages = torch.zeros_like(token_level_rewards)

        groups: dict[Any, list[int]] = defaultdict(list)
        for row, uid in enumerate(index):
            groups[uid].append(row)

        group_types = torch.empty(scores.shape[0], dtype=torch.int8, device=scores.device)
        for rows in groups.values():
            row_tensor = torch.as_tensor(rows, device=scores.device)
            group_scores = scores[row_tensor]
            if (group_scores.max() - group_scores.min()) > zero_variance_epsilon:
                advantages[row_tensor] = grpo[row_tensor]
                group_types[row_tensor] = 0  # mixed
            elif group_scores.mean() > zero_variance_epsilon:
                advantages[row_tensor] = alpha * entropy[row_tensor]
                group_types[row_tensor] = 1  # positive ZVP
            else:
                group_entropy = entropy[row_tensor]
                group_mask = mask[row_tensor].bool()
                masked_entropy = group_entropy.masked_fill(~group_mask, -torch.inf)
                row_max = masked_entropy.max(dim=-1).values
                row_max = torch.where(group_mask.any(dim=-1), row_max, torch.zeros_like(row_max))
                shaped = -alpha * (row_max.unsqueeze(-1) - group_entropy)
                advantages[row_tensor] = shaped * mask[row_tensor]
                group_types[row_tensor] = 2  # negative ZVP

        advantages = advantages * mask
        metrics = compute_rl_zvp_metrics(
            scores=scores,
            response_mask=mask,
            token_entropy=entropy,
            advantages=advantages,
            group_types=group_types,
        )
        _LAST_METRICS.clear()
        _LAST_METRICS.update(metrics)
        return advantages, advantages.clone()


def registered_rl_zvp_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: Sequence[Any],
    token_entropy: torch.Tensor,
    config: Any,
    **_: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """verl registry-compatible adapter; the bridge supplies `token_entropy`."""
    return compute_rl_zvp_advantage(
        token_level_rewards,
        response_mask,
        index,
        token_entropy,
        alpha=float(_cfg_value(config, ("rl_zvp", "alpha"))),
        zero_variance_epsilon=float(_cfg_value(config, ("rl_zvp", "zero_variance_epsilon"))),
    )


def register_with_verl() -> None:
    """Register `rl_zvp` through verl's real string registry API."""
    from verl.trainer.ppo import core_algos

    existing = core_algos.ADV_ESTIMATOR_REGISTRY.get("rl_zvp")
    if existing is None:
        core_algos.register_adv_est("rl_zvp")(registered_rl_zvp_advantage)
    elif existing is not registered_rl_zvp_advantage:
        raise RuntimeError("a different verl estimator is already registered as 'rl_zvp'")


def consume_last_metrics() -> dict[str, float]:
    metrics = dict(_LAST_METRICS)
    _LAST_METRICS.clear()
    return metrics
