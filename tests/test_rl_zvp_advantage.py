import numpy as np
import torch

from rl_zvp.advantage import compute_grpo_like_verl, compute_rl_zvp_advantage


ALPHA = 0.1
EPS = 1e-6


def run(rewards, entropy, mask=None, uid=None):
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    entropy = torch.as_tensor(entropy, dtype=torch.float32)
    if rewards.ndim == 1:
        rewards = rewards[:, None].expand_as(entropy).clone()
        rewards[:, 1:] = 0
    if mask is None:
        mask = torch.ones_like(entropy)
    else:
        mask = torch.as_tensor(mask, dtype=torch.float32)
    if uid is None:
        uid = np.array(["q"] * len(rewards), dtype=object)
    return compute_rl_zvp_advantage(
        rewards, mask, uid, entropy, alpha=ALPHA, zero_variance_epsilon=EPS
    )[0]


def test_mixed_group_equals_grpo():
    rewards = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    mask = torch.ones(4, 2)
    token_rewards = torch.zeros_like(mask)
    token_rewards[:, 0] = rewards[:, 0]
    entropy = torch.rand_like(mask)
    uid = np.array(["q"] * 4, dtype=object)
    actual = run(token_rewards, entropy, mask, uid)
    expected = compute_grpo_like_verl(token_rewards, mask, uid, EPS)
    torch.testing.assert_close(actual, expected)


def test_positive_zvp_is_alpha_entropy():
    entropy = torch.tensor([[0.2, 0.7], [0.4, 0.5], [0.8, 0.1], [0.3, 0.9]])
    actual = run([1, 1, 1, 1], entropy)
    torch.testing.assert_close(actual, ALPHA * entropy)


def test_negative_zvp_is_per_response_max_gap():
    entropy = torch.tensor([[0.2, 0.7], [0.4, 0.5], [0.8, 0.1], [0.3, 0.9]])
    actual = run([0, 0, 0, 0], entropy)
    expected = -ALPHA * (entropy.max(dim=-1).values[:, None] - entropy)
    torch.testing.assert_close(actual, expected)


def test_padding_advantage_is_zero_and_not_used_in_max():
    entropy = [[0.2, 0.7, 99.0], [0.4, 0.5, 99.0]]
    mask = [[1, 1, 0], [1, 1, 0]]
    actual = run([0, 0], entropy, mask)
    assert torch.equal(actual[:, 2], torch.zeros(2))
    torch.testing.assert_close(actual[0, :2], torch.tensor([-0.05, 0.0]))


def test_multiple_groups_are_classified_independently():
    rewards = [1, 0, 1, 1, 0, 0]
    entropy = torch.tensor([[0.2, 0.4]] * 6)
    uid = np.array(["mixed", "mixed", "pos", "pos", "neg", "neg"], dtype=object)
    actual = run(rewards, entropy, uid=uid)
    token_rewards = torch.zeros_like(entropy)
    token_rewards[:, 0] = torch.tensor(rewards, dtype=torch.float32)
    mixed = compute_grpo_like_verl(token_rewards[:2], torch.ones(2, 2), uid[:2], EPS)
    torch.testing.assert_close(actual[:2], mixed)
    torch.testing.assert_close(actual[2:4], ALPHA * entropy[2:4])
    expected_neg = -ALPHA * (entropy[4:].max(dim=-1).values[:, None] - entropy[4:])
    torch.testing.assert_close(actual[4:], expected_neg)


def test_high_entropy_gets_smaller_negative_punishment():
    actual = run([0, 0], [[0.8, 0.2], [0.8, 0.2]])
    assert actual[0, 0] > actual[0, 1]
    assert torch.all(actual <= 0)


def test_entropy_is_detached():
    entropy = torch.tensor([[0.2, 0.7], [0.4, 0.5]], requires_grad=True)
    actual = run([1, 1], entropy)
    assert not actual.requires_grad
    assert actual.grad_fn is None
