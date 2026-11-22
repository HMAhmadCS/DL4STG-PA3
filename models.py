"""Lightweight spatio-temporal forecasting models (Q4b, Q4c, Q5)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def row_normalize(adj):
    """Directed (random-walk) normalisation D_out^{-1} (A + I).

    The LargeST road graph is directed, so the symmetric D^{-1/2} A D^{-1/2} form does not
    apply. Row normalisation keeps each row a convex combination of the node's own feature
    and its out-neighbours' features. The SD matrix already has ones on the diagonal, so
    the self-loop is set to 1 rather than added a second time.
    """
    a = adj.astype(np.float64).copy()
    np.fill_diagonal(a, 1.0)
    return a / a.sum(1, keepdims=True)


class TemporalEncoder(nn.Module):
    """Per-sensor 1D convolutions over the 8 input steps: 8 -> 6 -> 2 -> 1."""

    def __init__(self, in_dim=1, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden, kernel_size=3), nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, dilation=2), nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=2),
        )

    def forward(self, x):
        # x: (B, T, N, F) -> (B*N, F, T)
        b, t, n, f = x.shape
        z = self.net(x.permute(0, 2, 3, 1).reshape(b * n, f, t))
        return z.reshape(b, n, -1)  # (B, N, d)


class ForecastHead(nn.Module):
    def __init__(self, hidden=32, out_len=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_len))

    def forward(self, h):
        return self.net(h).permute(0, 2, 1).unsqueeze(-1)  # (B, 4, N, 1)


class AdaptiveAdjacency(nn.Module):
    """A = Softmax(ReLU(E1 E2^T)) row-wise, optionally keeping the top-k logits per row.

    Sparsification is applied *before* the softmax: logits outside each row's top-k are
    set to -inf, so the softmax renormalises over the k kept entries and every row still
    sums to one. Top-k selection itself is a piecewise-constant index choice; gradients
    flow to E1 and E2 through the k retained logits of each row.
    """

    def __init__(self, num_nodes, emb_dim=10, topk=20):
        super().__init__()
        self.e1 = nn.Parameter(torch.randn(num_nodes, emb_dim))
        self.e2 = nn.Parameter(torch.randn(num_nodes, emb_dim))
        self.topk = topk

    def forward(self):
        logits = F.relu(self.e1 @ self.e2.T)
        if self.topk:
            kth = logits.topk(self.topk, dim=1).values[:, -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        return torch.softmax(logits, dim=1)


class GraphConv(nn.Module):
    """H = ReLU(A_hat Z W_g), with a residual connection to Z."""

    def __init__(self, hidden=32):
        super().__init__()
        self.w = nn.Linear(hidden, hidden, bias=False)

    def forward(self, z, adj):
        # adj: (N, N) applied over the node dimension for every batch item
        return F.relu(torch.einsum("ij,bjd->bid", adj, self.w(z))) + z


class STModel(nn.Module):
    """graph in {None, 'fixed', 'adaptive', 'hybrid'}; None gives the temporal-only baseline."""

    def __init__(self, num_nodes, graph=None, fixed_adj=None, hidden=32, emb_dim=10, topk=20,
                 alpha_init=0.5):
        super().__init__()
        self.graph = graph
        self.encoder = TemporalEncoder(hidden=hidden)
        self.head = ForecastHead(hidden)
        if graph is not None:
            self.gconv = GraphConv(hidden)
        if graph in ("fixed", "hybrid"):
            self.register_buffer("a_fixed", torch.tensor(row_normalize(fixed_adj), dtype=torch.float32))
        if graph in ("adaptive", "hybrid"):
            self.adaptive = AdaptiveAdjacency(num_nodes, emb_dim, topk)
        if graph == "hybrid":
            # alpha = sigmoid(theta) stays in [0, 1] and is learned
            self.alpha_logit = nn.Parameter(torch.logit(torch.tensor(alpha_init)))

    def alpha(self):
        return torch.sigmoid(self.alpha_logit)

    def adjacency(self):
        if self.graph == "fixed":
            return self.a_fixed
        if self.graph == "adaptive":
            return self.adaptive()
        if self.graph == "hybrid":
            # Both matrices are row-stochastic, so the convex mix is row-stochastic too.
            a = self.alpha()
            return a * self.a_fixed + (1 - a) * self.adaptive()
        return None

    def forward(self, x):
        h = self.encoder(x)
        if self.graph is not None:
            h = self.gconv(h, self.adjacency())
        return self.head(h)
