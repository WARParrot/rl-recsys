import torch
from rl_recsys.models import Policy, RewardModel, append_action, reinforce_loss

CFG = {"hidden": 16, "layers": 1, "heads": 2, "dropout": 0.0}

def test_policy_and_reward_shapes():
    p = Policy(12, 4, CFG)
    x = torch.tensor([[0, 0, 0, 2], [0, 0, 2, 3]])
    assert p(x).shape == (2, 10)
    r = RewardModel(12, 4, CFG)
    assert r(x, torch.tensor([2, 3])).shape == (2, 2)
    assert append_action(x, torch.tensor([1, 2])).shape == x.shape

def test_reinforce_finite_and_gradients():
    lp = [torch.randn(4, requires_grad=True), torch.randn(4, requires_grad=True)]
    rw = [torch.rand(4), torch.rand(4)]
    kl = [torch.rand(4), torch.rand(4)]
    loss = reinforce_loss(lp, rw, kl, .1)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(x.grad is not None for x in lp)

def test_nonzero_context_required():
    p = Policy(12, 4, CFG)
    try: p(torch.zeros(1, 4, dtype=torch.long))
    except ValueError: pass
    else: assert False
