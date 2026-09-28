import numpy as np
import pytest
import torch

from rl_zvp.advantage import compute_grpo_like_verl, compute_rl_zvp_advantage
from rl_zvp.verl_bridge import align_entropy


def test_entropy_shape_and_mask_alignment():
    entropy = torch.tensor([[0.2, 0.7, 5.0]], requires_grad=True)
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    aligned = align_entropy(entropy, mask)
    assert aligned.shape == mask.shape
    assert aligned[0, 2].item() == 0
    assert not aligned.requires_grad


def test_shape_mismatch_fails():
    with pytest.raises(ValueError, match="shape mismatch"):
        align_entropy(torch.ones(2, 3), torch.ones(2, 2))


def test_rl_zvp_receives_aligned_entropy():
    rewards = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    entropy = align_entropy(torch.tensor([[0.2, 0.4], [0.3, 9.0]]), mask)
    advantage, _ = compute_rl_zvp_advantage(
        rewards,
        mask,
        np.array(["q", "q"], dtype=object),
        entropy,
        alpha=0.1,
        zero_variance_epsilon=1e-6,
    )
    torch.testing.assert_close(advantage, 0.1 * entropy)


def test_grpo_has_no_entropy_dependency():
    rewards = torch.tensor([[1.0], [0.0]])
    mask = torch.ones_like(rewards)
    result = compute_grpo_like_verl(rewards, mask, np.array([0, 0]), 1e-6)
    assert result.shape == mask.shape
