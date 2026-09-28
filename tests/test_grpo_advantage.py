import numpy as np
import pytest
import torch

from rl_zvp.advantage import compute_grpo_like_verl


def test_grpo_group_normalization_matches_pinned_semantics():
    rewards = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    mask = torch.ones_like(rewards)
    uid = np.array(["q", "q", "q", "q"], dtype=object)
    actual = compute_grpo_like_verl(rewards, mask, uid, 1e-6)
    scores = rewards[:, 0]
    expected = ((scores - scores.mean()) / (scores.std() + 1e-6)).unsqueeze(-1)
    torch.testing.assert_close(actual, expected)


def test_against_native_verl_when_installed():
    core_algos = pytest.importorskip("verl.trainer.ppo.core_algos")
    rewards = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    mask = torch.ones_like(rewards)
    uid = np.array([0, 0, 0, 0])
    native, _ = core_algos.compute_grpo_outcome_advantage(rewards, mask, uid, epsilon=1e-6)
    ours = compute_grpo_like_verl(rewards, mask, uid, 1e-6)
    torch.testing.assert_close(ours, native)
