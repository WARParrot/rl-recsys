"""GPT-2 item-ID policy, and independent history-conditioned reward model."""
import torch
from torch import nn
from transformers import GPT2Config, GPT2Model


def backbone(vocab_size, length, model_cfg):
    return GPT2Model(GPT2Config(
        vocab_size=vocab_size, n_positions=length, n_ctx=length,
        n_embd=model_cfg["hidden"], n_layer=model_cfg["layers"], n_head=model_cfg["heads"],
        resid_pdrop=model_cfg["dropout"], embd_pdrop=model_cfg["dropout"],
        attn_pdrop=model_cfg["dropout"], use_cache=False,
        pad_token_id=0, bos_token_id=1, eos_token_id=None,
    ))


def encode(network, context):
    if (context[:, -1] == 0).any():
        raise ValueError("Contexts must be left-padded and end with a real or unknown token")
    mask = context.ne(0)
    positions = (mask.long().cumsum(-1) - 1).clamp_min(0)
    return network(input_ids=context, attention_mask=mask, position_ids=positions).last_hidden_state[:, -1]


class Policy(nn.Module):
    def __init__(self, vocab_size, length, model_cfg):
        super().__init__()
        self.encoder = backbone(vocab_size, length, model_cfg)

    def forward(self, context):
        hidden = encode(self.encoder, context)
        # Only last-token logits; never allocate B x L x catalog.
        # Special tokens have no action support (not merely -large logits).
        return nn.functional.linear(hidden, self.encoder.wte.weight[2:])


class RewardModel(nn.Module):
    """Independent from policy: either mean-history or attention-history encoder.

    Predict bounded completion and bounded played-seconds proxy. Mean-history
    auditor differs architecturally from attention reward optimized by policy.
    """
    def __init__(self, vocab_size, length, model_cfg, mean_history=False):
        super().__init__()
        self.mean_history = mean_history
        hidden = model_cfg["hidden"]
        self.encoder = (nn.Embedding(vocab_size, hidden, padding_idx=0) if mean_history
                        else backbone(vocab_size, length, model_cfg))
        self.item = nn.Embedding(vocab_size, hidden)
        self.head = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.Tanh(), nn.Linear(hidden, 2))

    def history(self, context):
        if self.mean_history:
            mask = context.ne(0).unsqueeze(-1)
            return (self.encoder(context) * mask).sum(1) / mask.sum(1).clamp_min(1)
        return encode(self.encoder, context)

    def from_history(self, hidden, items):
        item = self.item(items)
        return self.head(torch.cat([hidden, item, hidden * item], -1)).sigmoid()

    def forward(self, context, items):
        return self.from_history(self.history(context), items)

    @torch.no_grad()
    def catalog(self, context, chunk_size=1024):
        hidden = self.history(context)
        count = self.item.num_embeddings - 2
        chunks = []
        for start in range(2, count + 2, chunk_size):
            ids = torch.arange(start, min(start + chunk_size, count + 2), device=context.device)
            h = hidden[:, None, :].expand(-1, len(ids), -1)
            i = ids[None, :].expand(len(context), -1)
            chunks.append(self.from_history(h, i))
        return torch.cat(chunks, dim=1)


def append_action(context, action):
    """Action indexes exclude special tokens; append hypothetical item token."""
    return torch.cat([context[:, 1:], (action + 2).unsqueeze(-1)], -1)


def reinforce_loss(logprobs, rewards, kls, beta):
    """Undiscounted reward-to-go with leave-one-out (independent) baseline.

    Exact current-state KL contributes directly, AND future KL is included in
    score-function returns. Detaching all KL returns without the direct KL term
    would miss its action-distribution gradient. Context transitions are discrete.
    """
    if len(logprobs) != len(rewards) or len(kls) != len(rewards):
        raise ValueError("Mismatched trajectory")
    losses = []
    future = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        # state KL at t is independent of action_t; future state KL is not.
        score_return = rewards[t].detach() + future
        if len(score_return) > 1:
            baseline = (score_return.sum() - score_return) / (len(score_return) - 1)
        else:
            baseline = torch.zeros_like(score_return)
        losses.append(-(logprobs[t] * (score_return - baseline).detach()).mean() + beta * kls[t].mean())
        future = score_return - beta * kls[t].detach()
    return torch.stack(losses).sum() / len(rewards)
