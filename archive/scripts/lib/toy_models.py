"""Toy model + trainer for the low-rank separability experiments (scripts/toy/experiment8*,
experiment9*, check_exact_vs_sampled). Extracted verbatim from the original Exp2/Exp3 scripts
(now in archive/) so the current toy pipeline carries no dependency on archived code.

  ModelA_MLP : a FREE per-lexeme embedding -> one-hidden-layer MLP head (d -> h_dim -> vocab).
               Because the embedding is free and h_dim >= rank, a fully separable solution
               EXISTS -- so any entanglement the measure reports is learned, not architecturally
               forced (the load-bearing property; see SEPARABILITY_FINDINGS.md sec.0).
  _train     : sampled-token SGD with EARLY STOPPING on the exact expected loss against the true P
               (plateau = converged; the gap to entropy at low d is a capacity signal, not
               undertraining). Used by experiment8; the experiment9 grid uses its own _train_exact.
"""
import numpy as np  # noqa: F401  (kept: callers pass numpy arrays; re-exported convenience)
import torch
import torch.nn as nn
import torch.nn.functional as F


def _identity(x):
    return x


_ACTIVATIONS = {
    "identity": _identity,   # linear hidden layer -- the negative control: no nonlinearity, so it
                             # SHOULD read separable. Whether it does is empirical (gauge-contingent).
    "relu": F.relu,
    "tanh": torch.tanh,      # centered nonlinearity -- rules out ReLU-orthant artifacts.
}


class _MLPHead(nn.Module):
    def __init__(self, d, h_dim, vocab_size, activation):
        super().__init__()
        self.hidden = nn.Linear(d, h_dim)
        self.out = nn.Linear(h_dim, vocab_size, bias=False)
        self.act = _ACTIVATIONS[activation]

    def hid(self, e):
        return self.act(self.hidden(e))

    def forward(self, e):
        return self.out(self.act(self.hidden(e)))


class ModelA_MLP(nn.Module):
    def __init__(self, n_verbs, vocab_size, d, h_dim, activation):
        super().__init__()
        self.n_verbs = n_verbs
        self.embed = nn.Embedding(n_verbs, d)
        nn.init.normal_(self.embed.weight, std=0.1)
        self.head = _MLPHead(d, h_dim, vocab_size, activation)

    def _embedding(self, idx):
        return self.embed(idx)

    def forward(self, idx):
        return self.head(self.embed(idx))

    def get_all_embeddings(self):
        with torch.no_grad():
            dev = self.embed.weight.device
            return self.embed(torch.arange(self.n_verbs, device=dev)).cpu().numpy()

    def get_all_hidden(self):
        with torch.no_grad():
            dev = self.embed.weight.device
            return self.head.hid(self.embed(torch.arange(self.n_verbs, device=dev))).cpu().numpy()


def _train(m, P, max_steps, batch_size, lr, device="cpu",
           eval_every=500, patience=10, min_delta=3e-4, amp=False):
    """Train with EARLY STOPPING on the exact expected loss against the true P (not a noisy
    sampled-token estimate). Stop when that loss fails to improve by >min_delta for `patience`
    consecutive evals (a real PLATEAU = converged), or at max_steps. Returns (steps_run, best_loss).
    NOTE: converged means plateaued, NOT "reached the entropy of P" -- low-d models are
    capacity-limited and plateau above that floor; the gap to entropy is a capacity signal.
    amp=True -> bf16 autocast for the TRAINING forward on cuda; the early-stopping eval stays fp32
    so the stop criterion is exact, and get_all_hidden (the reps we measure) is unaffected."""
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False   # pure fp32 matmuls (keep the regime clean)
        torch.backends.cudnn.allow_tf32 = False
    m.to(device)
    Pt = torch.tensor(P, dtype=torch.float32, device=device)
    try:
        opt = torch.optim.Adam(m.parameters(), lr=lr, fused=(device == "cuda"))  # fused = free speedup
    except (TypeError, RuntimeError):
        opt = torch.optim.Adam(m.parameters(), lr=lr)
    nl = P.shape[0]
    all_idx = torch.arange(nl, device=device)
    use_amp = amp and device == "cuda"

    def expected_loss():                                   # fp32 -> precise stopping signal
        m.eval(); tot = 0.0
        with torch.no_grad():
            for s in range(0, nl, 4096):
                idx = all_idx[s:s + 4096]
                tot += -(Pt[idx] * F.log_softmax(m(idx), dim=-1)).sum(-1).sum().item()
        m.train()
        return tot / nl

    best = float("inf"); bad = 0; steps = 0
    for step in range(max_steps):
        li = torch.randint(0, nl, (batch_size,), device=device)
        tk = torch.multinomial(Pt[li], 1).squeeze(-1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss = F.cross_entropy(m(li), tk)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        steps = step + 1
        if steps % eval_every == 0:
            v = expected_loss()
            if v < best - min_delta:
                best = v; bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break
    return steps, (best if best < float("inf") else expected_loss())
