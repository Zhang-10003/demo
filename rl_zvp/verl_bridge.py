"""Minimal bridge from verl v0.9.1's stored entropy to RL-ZVP."""

from __future__ import annotations

import os
from typing import Any

import torch

_INSTALLED = False


def align_entropy(token_entropy: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Validate `[B,T_response]`, detach, and clear padding positions."""
    if token_entropy.shape != response_mask.shape:
        raise ValueError(
            f"entropy/mask shape mismatch: {tuple(token_entropy.shape)} != {tuple(response_mask.shape)}"
        )
    return token_entropy.detach() * response_mask.to(token_entropy.dtype)


def install_verl_bridge() -> None:
    """Install RL-ZVP-only wrappers in a Ray worker process."""
    global _INSTALLED
    if _INSTALLED:
        return

    import transfer_queue as tq
    from verl.trainer.ppo import core_algos
    from verl.trainer.ppo.v1 import trainer_base, utils as v1_utils

    from .advantage import consume_last_metrics, register_with_verl

    register_with_verl()
    original_compute = v1_utils.compute_advantage
    original_method = trainer_base.PPOTrainer._compute_advantage

    def compute_with_entropy(data, adv_estimator, *args, **kwargs):
        estimator_name = getattr(adv_estimator, "value", adv_estimator)
        if estimator_name != "rl_zvp":
            return original_compute(data, adv_estimator, *args, **kwargs)
        if "entropy" not in data.batch:
            raise RuntimeError("RL-ZVP requires verl old-log-prob entropy in the advantage batch")
        entropy = align_entropy(data.batch["entropy"], data.batch["response_mask"])
        estimator_fn = core_algos.get_adv_estimator_fn("rl_zvp")
        advantages, returns = estimator_fn(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            token_entropy=entropy,
            config=kwargs.get("config"),
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        return data

    def advantage_method_with_entropy(self, batch, metrics: dict[str, Any]):
        original_get = tq.kv_batch_get

        def get_with_entropy(*args, **kwargs):
            fields = kwargs.get("select_fields")
            if fields and "rm_scores" in fields and "entropy" not in fields:
                kwargs["select_fields"] = [*fields, "entropy"]
            return original_get(*args, **kwargs)

        tq.kv_batch_get = get_with_entropy
        try:
            result = original_method(self, batch, metrics)
            metrics.update(consume_last_metrics())
            return result
        finally:
            tq.kv_batch_get = original_get

    v1_utils.compute_advantage = compute_with_entropy
    trainer_base.PPOTrainer._compute_advantage = advantage_method_with_entropy
    _INSTALLED = True


def worker_setup() -> None:
    """Ray `worker_process_setup_hook` entrypoint."""
    install_verl_bridge()


# A string Ray setup hook imports a module; keep ordinary/test imports lightweight.
if os.environ.get("RL_ZVP_INSTALL_BRIDGE") == "1":
    worker_setup()
