"""Exact, non-Monte-Carlo checks of the policy-gradient estimator."""
from itertools import product

import pytest
import torch

from rl_recsys.models import RewardModel, reinforce_loss


DTYPE = torch.float64
TRAJECTORIES = tuple(product(range(2), repeat=2))


def _mdp(kl_scope="both"):
    # Row 0 is the initial state; rows 1 and 2 are reached by action 0/1.
    logits = torch.tensor(
        [[0.35, -0.45], [-0.2, 0.65], [0.8, -0.1]],
        dtype=DTYPE,
        requires_grad=True,
    )
    log_policy = logits.log_softmax(-1)
    policy = log_policy.exp()
    reference = torch.tensor(
        [[0.4, 0.6], [0.85, 0.15], [0.3, 0.7]], dtype=DTYPE
    )
    kl = (policy * (log_policy - reference.log())).sum(-1)
    mask = torch.tensor(
        {"current": [1, 0, 0], "future": [0, 1, 1], "both": [1, 1, 1]}[kl_scope],
        dtype=DTYPE,
    )
    rewards = torch.tensor([[0.2, -0.4], [1.1, -0.3], [-0.6, 0.8]], dtype=DTYPE)
    return logits, log_policy, rewards, kl * mask


def _probability(log_policy, trajectory):
    first, second = trajectory
    return (log_policy[0, first] + log_policy[1 + first, second]).exp()


def _exact_loss(log_policy, rewards, kls, beta):
    # Differentiate the full trajectory distribution, rewards and state KLs.
    # reinforce_loss averages over time, so the exact objective must too.
    return sum(
        _probability(log_policy, (first, second))
        * (beta * (kls[0] + kls[1 + first])
           - rewards[0, first] - rewards[1 + first, second])
        / 2
        for first, second in TRAJECTORIES
    )


def _batch_loss(log_policy, rewards, kls, beta, trajectories):
    first = torch.tensor([trajectory[0] for trajectory in trajectories])
    second = torch.tensor([trajectory[1] for trajectory in trajectories])
    return reinforce_loss(
        [log_policy[0, first], log_policy[1 + first, second]],
        [rewards[0, first], rewards[1 + first, second]],
        [kls[0].expand(len(trajectories)), kls[1 + first]],
        beta,
    )


def _expected_estimator_loss(log_policy, rewards, kls, beta, batch_size):
    # Enumerate independent draws WITH replacement, including identical pairs.
    # Sampling weights are detached: differentiating them here would add a
    # second score-function term to the estimator being tested.
    return sum(
        torch.stack([_probability(log_policy, path) for path in batch]).prod().detach()
        * _batch_loss(log_policy, rewards, kls, beta, batch)
        for batch in product(TRAJECTORIES, repeat=batch_size)
    )


@pytest.mark.parametrize(
    "beta,kl_scope,reward_scale",
    [
        pytest.param(0.0, "both", 1.0, id="reward-to-go"),
        pytest.param(0.7, "current", 0.0, id="current-kl-direct-gradient"),
        pytest.param(0.7, "future", 0.0, id="future-kl-state-credit"),
        pytest.param(0.7, "both", 1.0, id="reward-and-kl"),
    ],
)
def test_enumerated_two_step_gradient_matches_exact_objective(beta, kl_scope, reward_scale):
    logits, log_policy, rewards, kls = _mdp(kl_scope)
    rewards = rewards * reward_scale
    exact_gradient, = torch.autograd.grad(
        _exact_loss(log_policy, rewards, kls, beta), logits, retain_graph=True
    )
    estimated_gradient, = torch.autograd.grad(
        _expected_estimator_loss(log_policy, rewards, kls, beta, batch_size=1), logits
    )
    torch.testing.assert_close(estimated_gradient, exact_gradient, rtol=1e-12, atol=1e-12)
    assert exact_gradient.norm().item() > 1e-3
    if kl_scope == "future":
        # Initial action changes the next state's KL. A direct-KL-only loss
        # misses this nonzero root gradient even if child gradients are correct.
        assert exact_gradient[0].norm().item() > 1e-3


@pytest.mark.parametrize("beta", [0.0, 0.7])
def test_independent_leave_one_out_baseline_has_exact_expected_gradient(beta):
    logits, log_policy, rewards, kls = _mdp()
    exact_gradient, = torch.autograd.grad(
        _exact_loss(log_policy, rewards, kls, beta), logits, retain_graph=True
    )
    # All 4 x 4 ordered trajectory pairs, not the four correlated paths as one batch.
    pair_gradient, = torch.autograd.grad(
        _expected_estimator_loss(log_policy, rewards, kls, beta, batch_size=2), logits
    )
    torch.testing.assert_close(pair_gradient, exact_gradient, rtol=1e-12, atol=1e-12)
    assert exact_gradient.norm().item() > 1e-3


@pytest.mark.parametrize("reward_source", ["leaf", "reward_model"])
def test_rewards_are_detached_without_blocking_policy_or_direct_kl_gradients(reward_source):
    logits, log_policy, _, kls = _mdp()
    model = None
    if reward_source == "reward_model":
        with torch.random.fork_rng():
            torch.manual_seed(17)
            model = RewardModel(
                6, 2, {"hidden": 4}, mean_history=True
            ).to(dtype=DTYPE)
        # Distinct state-action pairs, evaluated by a real trainable RM.
        contexts = torch.tensor([[0, 2], [0, 2], [2, 2], [2, 2], [2, 3], [2, 3]])
        items = torch.tensor([2, 3, 2, 3, 2, 3])
        rewards = model(contexts, items)[:, 0].reshape(3, 2)
        rewards.retain_grad()
    else:
        rewards = torch.tensor(
            [[0.2, -0.4], [1.1, -0.3], [-0.6, 0.8]], dtype=DTYPE, requires_grad=True
        )

    kls.retain_grad()
    paths = ((0, 1), (1, 0))
    loss = _batch_loss(log_policy, rewards, kls, 0.7, paths)
    detached_reward_loss = _batch_loss(log_policy, rewards.detach(), kls, 0.7, paths)
    expected_policy_gradient, = torch.autograd.grad(
        detached_reward_loss, logits, retain_graph=True
    )
    # autograd.grad also populates retained intermediate gradients; isolate
    # the subsequent backward pass when checking the direct KL contribution.
    kls.grad = None
    loss.backward()

    assert rewards.grad is None
    if model is not None:
        assert all(parameter.grad is None for parameter in model.parameters())
    assert logits.grad is not None
    assert logits.grad.norm().item() > 1e-3
    torch.testing.assert_close(logits.grad, expected_policy_gradient, rtol=1e-12, atol=1e-12)
    # beta * mean(batch KL) / horizon: root occurs twice, each child once.
    torch.testing.assert_close(
        kls.grad, torch.tensor([0.35, 0.175, 0.175], dtype=DTYPE), rtol=1e-12, atol=1e-12
    )
