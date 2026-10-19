#!/usr/bin/env python3
"""Matching-sampling benchmark: structural estimators against cost-matched sampling.

MAISI's statement of ARC's matching sampling principle asks for an estimator G whose advice does
not depend on the context c and whose error over contexts, E_c[(G - E_x f(M_c(x)))^2], competes
with random sampling at comparable cost. Here a context is one draw of the weights of a random
ReLU network y = a . h_L, h_l = ReLU(W_l h_{l-1} + b_l), with 64 independent inputs (Rademacher
signs or standard Gaussians). The advice is each estimator's fixed rule. The targets are the mean
output (T1) and an upper-tail probability (T2).

Estimators (each receives the weights and nothing else that depends on the context):

  S  plain Monte Carlo with n samples, n matched to another estimator's FLOPs, then 10x and 100x.
  D  diagonal Gaussian closure: per-neuron mean and variance, cross-covariances dropped.
  C  full-covariance Gaussian closure (ARC's covariance propagation adapted to ReLU): mean vector
     and covariance matrix through every layer, with exact moments of ReLU on a bivariate
     Gaussian with nonzero means (Rosenbaum's truncated-moment formulas; the orthant probability
     uses Genz's bivariate normal algorithm with 20-point Gauss-Legendre quadrature).
  A  the essay's conditional averaging (sign inputs only): keep the first k signs, set the rest to
     their mean zero, evaluate the 2^k reference states, and add the boundary-radius bias bound
     of Averaging.mean_error_le, computed by the algorithm of boundary.py in float64.

Ground truth per context is antithetic Monte Carlo on the GPU in float64.

Subcommands: validate, pilot, sweep, tables. RESULTS.md lists the exact commands.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import socket
import sys
import time
from fractions import Fraction
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
DT = torch.float64
SQRT2PI = math.sqrt(2.0 * math.pi)
TWOPI = 2.0 * math.pi
TINY = 1e-300
RMAX = 1.0 - 1e-12

# ----------------------------------------------------------------------------------------------
# The benchmark family and every fixed choice. Written into the results file's "config".

FAMILY = {
    "input_dim": 64,
    "depths": [2, 3],
    "inputs": ["sign", "gauss"],
    "sigma_w2": [1.0, 2.0, 4.0],
    "sigma_b": [0.0, 0.1],
    "widths": [16, 64, 256, 1024],
}
BASE_SEED = 20261018
STREAMS = {"main": 0, "pilot": 1, "validate": 2}
KEPT = [4, 8]
TAIL_K = [3, 4]
MULTIPLIERS = [1, 10, 100]
# Fixed after the pilot measured cost (see the results file's "pilot" entry); never tuned on errors.
CONTEXTS = {16: 200, 64: 200, 256: 200, 1024: 100}
TRUTH_PAIRS = {16: 2**23, 64: 2**23, 256: 2**22, 1024: 2**22}
# Added later, after the width-1024 results showed C's error no longer falling between widths 256
# and 1024: does that floor move with the input dimension? Labeled as added later in the results.
ADDED_LATER = [{"depth": 2, "input": kind, "sigma_w2": 2.0, "sigma_b": 0.1, "width": w, "input_dim": d}
               for d in (16, 256) for w in (256, 1024) for kind in ("sign", "gauss")]
ADDED_LATER_REASON = ("Added after seeing the main grid: C's relative RMSE fell from width 16 to 256 but not from 256 "
                      "to 1024 at input dimension 64. These settings vary the input dimension (16 and 256) at depth 2, "
                      "sigma_w2 = 2, sigma_b = 0.1 to test whether that floor is set by the input dimension.")

# FLOP model. A multiply-add is 2 FLOPs, every other arithmetic operation, comparison, abs, max
# or clamp is 1, and each exp, sin, asin, sqrt or normal-CDF evaluation is charged TAU FLOPs.
# Random-number generation for S and the truth is not counted.
TAU = 20


def univ(tau=TAU):
    """ReLU mean, second moment and variance of one N(mu, v) neuron: sqrt, CDF, exp and 15 others."""
    return 3 * tau + 15


def pair(tau=TAU):
    """One unordered pair (i, j) of the C closure (49 transcendental evaluations, 247 others).

correlation and clamp (4); Genz orthant probability for |r| < 0.925: asin and 4 (TAU + 5),
20 nodes of sin, exp and 9 others (20 * (2 TAU + 9)), the closing product of two CDFs (2 TAU + 5);
Rosenbaum moment: sqrt (TAU + 3), K and H (6), two CDFs (2 TAU + 2), two densities (2 TAU + 6),
E[X], E[Y], E[XY] (23 + TAU), the combination (11) and the covariance (2).
    """
    return 49 * tau + 247


def flops_forward(d, w, L):
    """One sample: 2*in*out per matrix-vector product, +out bias, +out ReLU, 2w for the readout."""
    return 2 * d * w + 2 * w + (L - 1) * (2 * w * w + 2 * w) + 2 * w


def flops_S_per_sample(d, w, L, tails=0):
    """A forward pass, one addition into the running mean, a comparison and an add per threshold."""
    return flops_forward(d, w, L) + 1 + 2 * tails


def flops_D(d, w, L, tail=False, tau=TAU):
    f = 2 * d * w + w                    # layer-1 mean W1 mu_x + b1
    f += 3 * d * w                       # layer-1 variance (W1*W1) var_x
    f += L * w * univ(tau)               # ReLU mean and variance of every neuron
    f += (L - 1) * (2 * w * w + w)       # later means W m + b
    f += (L - 1) * 3 * w * w             # later variances (W*W) v
    f += 2 * w                           # readout mean
    if tail:
        f += 3 * w + tau + 3             # readout variance, sqrt, threshold standardization
    return f


def flops_C(d, w, L, tail=False, tau=TAU):
    """T1 needs full covariances through layer L-1 and only the diagonal at layer L."""
    pairs = w * (w - 1) // 2
    f = 2 * d * w + w                    # layer-1 mean
    f += 2 * d * w * w                   # layer-1 covariance W1 W1^T (Sigma_x = I)
    f += L * w * univ(tau)               # diagonal ReLU moments at every layer
    f += (L if tail else L - 1) * pairs * pair(tau)
    f += (L - 1) * (2 * w * w + w)       # later means
    if tail:
        f += (L - 1) * (4 * w ** 3 + w * w)          # W Cov W^T and symmetrization
        f += 2 * w + 2 * w * w + 2 * w + tau + 2     # readout mean, a^T Cov a, sqrt, threshold
    else:
        f += (L - 2) * (4 * w ** 3 + w * w) + 2 * w ** 3 + 2 * w * w   # last layer: diagonal only
        f += 2 * w
    return f


def flops_A(d, w, L, k):
    once = 2 * (d - k) * w + (L - 1) * w * w     # discarded-column radius, |W_l| for l >= 2
    per = 2 * k * w + 2 * w                       # layer-1 reference preactivation and ReLU
    per += (L - 1) * (4 * w * w + 11 * w)         # later layers: activation radius, W h + b, |W| r
    per += 2 * w + 1                              # readout and running mean
    per += (L - 1) * (2 * w * w + 2 * w)          # backward suffixes through the fixed gates
    per += L * 3 * w                              # ramps max(0, delta - |z|)
    per += L * 9 * w                              # |suffix| . ramp and the signed interval
    return once + 2 ** k * per


def cost_table(d, w, L, kind):
    costs = {
        "forward": flops_forward(d, w, L),
        "D": flops_D(d, w, L),
        "D_tail": flops_D(d, w, L, tail=True),
        "C": flops_C(d, w, L),
        "C2": flops_C(d, w, L, tail=True),
        "C_pairwise_part": (L - 1) * (w * (w - 1) // 2) * pair(),
        "C_tau1": flops_C(d, w, L, tau=1),
        "D_tau1": flops_D(d, w, L, tau=1),
    }
    if kind == "sign":
        for k in KEPT:
            costs[f"A{k}"] = flops_A(d, w, L, k)
    per1 = flops_S_per_sample(d, w, L)
    per2 = flops_S_per_sample(d, w, L, tails=len(TAIL_K))
    budgets = {}
    for name in ["D", "C"] + ([f"A{k}" for k in KEPT] if kind == "sign" else []):
        for m in MULTIPLIERS:
            budgets[f"{name}x{m}"] = max(1, round(m * costs[name] / per1))
    for m in MULTIPLIERS:
        budgets[f"C2x{m}"] = max(1, round(m * costs["C2"] / per2))
    return costs, budgets, per1, per2


# ----------------------------------------------------------------------------------------------
# Contexts.


def draw_context(depth, width, index, sigma_w2, sigma_b, stream="main", d=64):
    """W_l ~ N(0, sigma_w2/fan_in), b_l ~ N(0, sigma_b^2), a ~ N(0, 1/width).

    The seed excludes sigma_w2, sigma_b and the input law, so those settings share the same
    standard-normal draws (common random numbers across the grid).
    """
    rng = np.random.default_rng(np.random.SeedSequence([BASE_SEED, STREAMS[stream], depth, width, d, index]))
    Ws, bs = [], []
    fan_in = d
    for _ in range(depth):
        Ws.append(rng.standard_normal((width, fan_in)) * math.sqrt(sigma_w2 / fan_in))
        bs.append(rng.standard_normal(width) * sigma_b)
        fan_in = width
    a = rng.standard_normal(width) / math.sqrt(width)
    return {"W": Ws, "b": bs, "a": a}


class Net:
    def __init__(self, raw, device):
        self.Ws = [torch.as_tensor(W, dtype=DT, device=device) for W in raw["W"]]
        self.bs = [torch.as_tensor(b, dtype=DT, device=device) for b in raw["b"]]
        self.a = torch.as_tensor(raw["a"], dtype=DT, device=device)
        self.raw = raw

    @property
    def depth(self):
        return len(self.Ws)

    @property
    def d(self):
        return self.Ws[0].shape[1]


def seed_int(*parts):
    digest = hashlib.sha256(repr(("msp", BASE_SEED) + parts).encode()).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def sample_inputs(kind, n, d, gen, device):
    if kind == "sign":
        bits = torch.randint(0, 2, (n, d), generator=gen, device=device, dtype=torch.int8)
        return bits.to(DT).mul_(2.0).sub_(1.0)
    return torch.randn((n, d), generator=gen, device=device, dtype=DT)


def forward(net, X):
    h = X
    for W, b in zip(net.Ws, net.bs):
        h = torch.addmm(b, h, W.T).clamp_min_(0.0)
    return h @ net.a


def chunk_size(w, d):
    return max(2 ** 14, min(2 ** 20, 2 ** 29 // (8 * max(w, d))))


def truth_mc(net, kind, n_pairs, seed, thresholds, device):
    """Antithetic pairs (x, -x); both input laws are symmetric. Float64 throughout."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    w = net.Ws[-1].shape[0]
    chunk = chunk_size(w, net.d)
    zero = torch.zeros((), dtype=DT, device=device)
    s1, s2, sy, syy = zero.clone(), zero.clone(), zero.clone(), zero.clone()
    thr = torch.as_tensor(thresholds, dtype=DT, device=device)
    si = torch.zeros(len(thresholds), dtype=DT, device=device)
    sii = torch.zeros(len(thresholds), dtype=DT, device=device)
    done = 0
    while done < n_pairs:
        m = min(chunk, n_pairs - done)
        X = sample_inputs(kind, m, net.d, gen, device)
        y1 = forward(net, X)
        y2 = forward(net, X.neg_())
        pair = 0.5 * (y1 + y2)
        s1 += pair.sum()
        s2 += (pair * pair).sum()
        sy += y1.sum() + y2.sum()
        syy += (y1 * y1).sum() + (y2 * y2).sum()
        ind = 0.5 * ((y1[:, None] > thr).to(DT) + (y2[:, None] > thr).to(DT))
        si += ind.sum(0)
        sii += (ind * ind).sum(0)
        done += m
    n = float(n_pairs)
    mean = float(s1) / n
    pair_var = max(float(s2) / n - mean * mean, 0.0) * n / (n - 1)
    ey = float(sy) / (2 * n)
    var_y = max(float(syy) / (2 * n) - ey * ey, 0.0)
    p = (si / n).tolist()
    p_var = ((sii / n) - (si / n) ** 2).clamp_min(0).tolist()
    return {
        "mean": mean,
        "se": math.sqrt(pair_var / n),
        "var": var_y,
        "p": p,
        "p_se": [math.sqrt(v * n / (n - 1) / n) for v in p_var],
        "samples": 2 * n_pairs,
        "antithetic_variance_ratio": pair_var / (var_y / 2) if var_y > 0 else float("nan"),
    }


def sampling(net, kind, n, seed, thresholds, device):
    """Plain Monte Carlo with n independent samples (the S estimator)."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    w = net.Ws[-1].shape[0]
    chunk = chunk_size(w, net.d)
    total = torch.zeros((), dtype=DT, device=device)
    thr = torch.as_tensor(thresholds, dtype=DT, device=device)
    above = torch.zeros(len(thresholds), dtype=DT, device=device)
    done = 0
    while done < n:
        m = min(chunk, n - done)
        y = forward(net, sample_inputs(kind, m, net.d, gen, device))
        total += y.sum()
        above += (y[:, None] > thr).to(DT).sum(0)
        done += m
    return float(total) / n, (above / n).tolist()


# ----------------------------------------------------------------------------------------------
# Gaussian closures.


def _phi(x):
    return torch.exp(-0.5 * x * x) / SQRT2PI


def _Phi(x):
    return torch.special.ndtr(x)


def relu_moments(mu, var):
    """E[ReLU(Z)] and E[ReLU(Z)^2] for Z ~ N(mu, var), var > 0 (exact closed forms)."""
    sd = torch.sqrt(var)
    a = mu / sd
    P = _Phi(a)
    p = _phi(a)
    m1 = mu * P + sd * p
    m2 = (mu * mu + var) * P + mu * sd * p
    return m1, m2


_GL_T, _GL_W = np.polynomial.legendre.leggauss(20)
_GL_CACHE = {}


def _gl(device):
    key = str(device)
    if key not in _GL_CACHE:
        _GL_CACHE[key] = (torch.as_tensor(1.0 + _GL_T, dtype=DT, device=device),
                          torch.as_tensor(_GL_W, dtype=DT, device=device))
    return _GL_CACHE[key]


def _bvnu_low(h, k, r):
    """|r| < 0.925: Plackett's formula in theta = asin(rho), 20-point Gauss-Legendre."""
    x, wt = _gl(h.device)
    hk = h * k
    hs = 0.5 * (h * h + k * k)
    asr = 0.5 * torch.asin(r)
    sn = torch.sin(asr[:, None] * x[None, :])
    val = (torch.exp((sn * hk[:, None] - hs[:, None]) / (1.0 - sn * sn)) * wt[None, :]).sum(1)
    return val * asr / TWOPI + _Phi(-h) * _Phi(-k)


def _bvnu_high(h, k, r):
    """|r| >= 0.925: Genz's expansion around |rho| = 1 (Drezner-Wesolowsky with a correction)."""
    x, wt = _gl(h.device)
    neg = r < 0
    k = torch.where(neg, -k, k)
    hk = h * k
    as_ = (1.0 - r) * (1.0 + r)
    a = torch.sqrt(as_)
    bs = (h - k) ** 2
    c = (4.0 - hk) / 8.0
    dd = (12.0 - hk) / 80.0
    asr = -0.5 * (bs / as_ + hk)
    bvn = torch.where(asr > -100.0,
                      a * torch.exp(asr) * (1.0 - c * (bs - as_) * (1.0 - dd * bs) / 3.0 + c * dd * as_ * as_),
                      torch.zeros_like(h))
    b = torch.sqrt(bs)
    sp = math.sqrt(TWOPI) * _Phi(-b / a)
    bvn = torch.where(hk > -100.0,
                      bvn - torch.exp(-0.5 * hk) * sp * b * (1.0 - c * bs * (1.0 - dd * bs) / 3.0),
                      bvn)
    a2 = 0.5 * a
    xs = (a2[:, None] * x[None, :]) ** 2
    asr2 = -0.5 * (bs[:, None] / xs + hk[:, None])
    sp2 = 1.0 + c[:, None] * xs * (1.0 + 5.0 * dd[:, None] * xs)
    rs = torch.sqrt(1.0 - xs)
    ep = torch.exp(-0.5 * hk[:, None] * xs / (1.0 + rs) ** 2) / rs
    term = torch.where(asr2 > -100.0, torch.exp(asr2) * (sp2 - ep), torch.zeros_like(xs))
    bvn = (a2 * (term * wt[None, :]).sum(1) - bvn) / TWOPI
    pos_out = bvn + _Phi(-torch.maximum(h, k))
    band = torch.where(h < 0, _Phi(k) - _Phi(h), _Phi(-h) - _Phi(-k))
    neg_out = torch.where(h >= k, -bvn, band - bvn)
    return torch.where(neg, neg_out, pos_out)


def bvnu(h, k, r):
    """P(X > h, Y > k) for a standard bivariate normal with correlation r (Genz, BVNU)."""
    out = torch.empty_like(h)
    low = r.abs() < 0.925
    if bool(low.all()):
        return _bvnu_low(h, k, r).clamp_(0.0, 1.0)
    if bool(low.any()):
        out[low] = _bvnu_low(h[low], k[low], r[low])
    high = ~low
    out[high] = _bvnu_high(h[high], k[high], r[high])
    return out.clamp_(0.0, 1.0)


def relu_cross_moment(mu1, sd1, mu2, sd2, r):
    """E[ReLU(U) ReLU(V)] for jointly Gaussian (U, V) with means mu, sds sd and correlation r."""
    h = -mu1 / sd1
    k = -mu2 / sd2
    L = bvnu(h, k, r)
    s = torch.sqrt((1.0 - r) * (1.0 + r))
    K = (k - r * h) / s
    H = (h - r * k) / s
    QK = _Phi(-K)
    QH = _Phi(-H)
    ph = _phi(h)
    pk = _phi(k)
    EX = ph * QK + r * pk * QH
    EY = pk * QH + r * ph * QK
    EXY = r * L + r * h * ph * QK + r * k * pk * QH + s * pk * _phi(H)
    return mu1 * mu2 * L + mu1 * sd2 * EY + mu2 * sd1 * EX + sd1 * sd2 * EXY


_TRIU = {}


def _triu(w, device):
    key = (w, str(device))
    if key not in _TRIU:
        _TRIU[key] = torch.triu_indices(w, w, 1, device=device)
    return _TRIU[key]


def relu_cov(mu, Sigma):
    """Mean and covariance of ReLU(Z) for Z ~ N(mu, Sigma), exact pairwise."""
    w = mu.shape[0]
    var = Sigma.diagonal().clamp_min(TINY)
    sd = torch.sqrt(var)
    m1, m2 = relu_moments(mu, var)
    iu, ju = _triu(w, mu.device)
    r = (Sigma[iu, ju] / (sd[iu] * sd[ju])).clamp(-RMAX, RMAX)
    E = relu_cross_moment(mu[iu], sd[iu], mu[ju], sd[ju], r)
    vals = E - m1[iu] * m1[ju]
    cov = torch.zeros((w, w), dtype=DT, device=mu.device)
    cov[iu, ju] = vals
    cov[ju, iu] = vals
    cov.diagonal().copy_((m2 - m1 * m1).clamp_min(0.0))
    return m1, cov, int((r.abs() >= 0.925).sum())


def closure_full(net, tail):
    """C: Gaussian closure with the full covariance. Inputs have mean 0 and covariance I.

    With tail=False only what the mean needs is computed (layer L's covariance diagonal).
    Returns (mean, sd or None, number of pairs that used the |r| >= 0.925 branch).
    """
    Ws, bs, a = net.Ws, net.bs, net.a
    L = len(Ws)
    mu = Ws[0] @ torch.zeros(net.d, dtype=DT, device=a.device) + bs[0]
    Sigma = Ws[0] @ Ws[0].T
    diag = None
    high = 0
    for l in range(L):
        last = l == L - 1
        if last and not tail:
            var = diag if diag is not None else Sigma.diagonal()
            m1, _ = relu_moments(mu, var.clamp_min(TINY))
            return float(a @ m1), None, high
        m1, cov, nh = relu_cov(mu, Sigma)
        high += nh
        if last:
            var_y = float(a @ (cov @ a))
            return float(a @ m1), math.sqrt(max(var_y, 0.0)), high
        W, b = Ws[l + 1], bs[l + 1]
        mu = W @ m1 + b
        if l + 1 == L - 1 and not tail:
            diag = ((W @ cov) * W).sum(1)
            Sigma = None
        else:
            Sigma = W @ cov @ W.T
            Sigma = 0.5 * (Sigma + Sigma.T)
    raise AssertionError("unreachable")


def closure_diag(net, tail=True):
    """D: per-neuron Gaussian closure; cross-covariances are dropped at every layer."""
    Ws, bs, a = net.Ws, net.bs, net.a
    L = len(Ws)
    mu = Ws[0] @ torch.zeros(net.d, dtype=DT, device=a.device) + bs[0]
    var = (Ws[0] * Ws[0]) @ torch.ones(net.d, dtype=DT, device=a.device)
    for l in range(L):
        m1, m2 = relu_moments(mu, var.clamp_min(TINY))
        v = (m2 - m1 * m1).clamp_min(0.0)
        if l + 1 < L:
            W, b = Ws[l + 1], bs[l + 1]
            mu = W @ m1 + b
            var = (W * W) @ v
    mean = float(a @ m1)
    sd = math.sqrt(max(float((a * a) @ v), 0.0)) if tail else None
    return mean, sd


# ----------------------------------------------------------------------------------------------
# A: conditional averaging with the boundary-radius bound (boundary.py's algorithm, float64).

_CUBE = {}


def sign_cube(k, device):
    """All 2^k sign vectors in itertools.product order (first coordinate varies slowest)."""
    key = (k, str(device))
    if key not in _CUBE:
        _CUBE[key] = torch.tensor(list(itertools.product((-1.0, 1.0), repeat=k)), dtype=DT, device=device)
    return _CUBE[key]


def averaging(net, k, keep_trace=False):
    """Keep the first k signs, replace the rest by 0, and bound the change in the mean.

    For each reference state: z_l are the reference preactivations, delta_l bounds |actual - z_l|
    (layer 1: sum of |W1| over discarded columns; later: |W_l| times the activation enclosure
    max(|ReLU(z-delta) - ReLU(z)|, |ReLU(z+delta) - ReLU(z)|)), ramp_l = max(0, delta_l - |z_l|),
    and suffix_l is the reference backward pass through the fixed gates (z_l > 0). The bias bound
    is the mean over states of sum_l |suffix_l| . ramp_l, and the signed interval uses the
    negative and positive parts of suffix_l.
    """
    Ws, bs, a = net.Ws, net.bs, net.a
    L = len(Ws)
    cells = sign_cube(k, a.device)
    W1 = Ws[0]
    z = cells @ W1[:, :k].T + bs[0]
    delta = W1[:, k:].abs().sum(1).expand_as(z)
    pres, deltas = [z], [delta]
    h = z.clamp_min(0.0)
    for W, b in zip(Ws[1:], bs[1:]):
        radius = torch.maximum(((z - delta).clamp_min(0.0) - h).abs(), ((z + delta).clamp_min(0.0) - h).abs())
        z = h @ W.T + b
        delta = radius @ W.abs().T
        pres.append(z)
        deltas.append(delta)
        h = z.clamp_min(0.0)
    outputs = h @ a
    suffix = a.expand(cells.shape[0], -1)
    suffixes = [None] * L
    for l in reversed(range(L)):
        suffixes[l] = suffix
        if l > 0:
            suffix = ((pres[l] > 0).to(DT) * suffix) @ Ws[l]
    ramps = [(dl - zl.abs()).clamp_min(0.0) for zl, dl in zip(pres, deltas)]
    bound = sum((s.abs() * r).sum(1) for s, r in zip(suffixes, ramps)).mean()
    lower = sum((s.clamp_max(0.0) * r).sum(1) for s, r in zip(suffixes, ramps)).mean()
    upper = sum((s.clamp_min(0.0) * r).sum(1) for s, r in zip(suffixes, ramps)).mean()
    result = (float(outputs.mean()), float(bound), float(lower), float(upper))
    if keep_trace:
        return result, {"pres": pres, "deltas": deltas, "ramps": ramps}
    return result


# ----------------------------------------------------------------------------------------------
# Validation.


def _quad(f, lo, hi, points=None):
    from scipy import integrate
    val, err = integrate.quad(f, lo, hi, points=points, epsabs=1e-15, epsrel=1e-13, limit=500)
    return val


def relu_moment_reference(mu, sd, power):
    lo = max(-mu / sd, -40.0)
    if lo >= 40.0:
        return 0.0
    return _quad(lambda t: (mu + sd * t) ** power * math.exp(-0.5 * t * t) / SQRT2PI, lo, 40.0)


def relu_cross_reference(mu1, sd1, mu2, sd2, r):
    """E[ReLU(U)ReLU(V)] by 1D quadrature over U of U * E[ReLU(V) | U] (no orthant formula)."""
    from scipy import special
    c = math.sqrt(max((1 - r) * (1 + r), 0.0)) * sd2
    lo = max(-mu1 / sd1, -40.0)
    if lo >= 40.0:
        return 0.0

    def f(t):
        m = mu2 + r * sd2 * t
        g = m * special.ndtr(m / c) + c * math.exp(-0.5 * (m / c) ** 2) / SQRT2PI if c > 0 else max(m, 0.0)
        return (mu1 + sd1 * t) * g * math.exp(-0.5 * t * t) / SQRT2PI

    # E[ReLU(V) | U] bends over a width c / |r sd2| around t0; give quad breakpoints there.
    points = None
    if r != 0:
        t0 = -mu2 / (r * sd2)
        width = c / abs(r * sd2)
        cand = [t0 + j * width for j in (-50, -10, -3, 0, 3, 10, 50)]
        points = [p for p in cand if lo < p < 40.0] or None
    return _quad(f, lo, 40.0, points)


def bvn_lower_owen(x, y, rho):
    """Phi2(x, y; rho) through Owen's T function (scipy.special.owens_t); x, y nonzero."""
    from scipy import special
    s = np.sqrt((1 - rho) * (1 + rho))
    ax = (y - rho * x) / (x * s)
    ay = (x - rho * y) / (y * s)
    beta = np.where(x * y > 0, 0.0, 0.5)
    return 0.5 * special.ndtr(x) + 0.5 * special.ndtr(y) - special.owens_t(x, ax) - special.owens_t(y, ay) - beta


def validate_univariate(device):
    rows = []
    worst1 = worst2 = 0.0
    for mu in np.linspace(-6.0, 6.0, 25):
        for sd in [0.05, 0.5, 1.0, 2.0, 10.0]:
            m1, m2 = relu_moments(torch.tensor([mu], dtype=DT, device=device), torch.tensor([sd * sd], dtype=DT, device=device))
            r1 = relu_moment_reference(mu, sd, 1)
            r2 = relu_moment_reference(mu, sd, 2)
            e1 = abs(float(m1) - r1) / max(sd, abs(mu))
            e2 = abs(float(m2) - r2) / max(sd, abs(mu)) ** 2
            worst1, worst2 = max(worst1, e1), max(worst2, e2)
            rows.append(1)
    return {"cases": len(rows), "max_scaled_error_mean": worst1, "max_scaled_error_second_moment": worst2,
            "reference": "scipy.integrate.quad over the positive half-line (epsrel 1e-13)",
            "scaling": "errors divided by max(sd, |mu|) and its square", "passed": worst1 < 1e-11 and worst2 < 1e-11}


def validate_bvn(device):
    rng = np.random.default_rng([BASE_SEED, 11])
    n = 20000
    h = rng.uniform(-7, 7, 4 * n)
    k = rng.uniform(-7, 7, 4 * n)
    sign = rng.choice([-1.0, 1.0], 4 * n)
    r = np.concatenate([
        rng.uniform(-0.925, 0.925, n),
        sign[n:2 * n] * rng.uniform(0.925, 0.999, n),
        sign[2 * n:3 * n] * (1 - 10 ** rng.uniform(-12, -3, n)),
        sign[3 * n:] * (1 - 10 ** rng.uniform(-12, -1.1, n)),
    ])
    k[3 * n:] = h[3 * n:] * np.where(sign[3 * n:] > 0, 1, -1) + rng.normal(0, 1e-3, n)  # near-degenerate band
    keep = (np.abs(h) > 1e-9) & (np.abs(k) > 1e-9)
    h, k, r = h[keep], k[keep], r[keep]
    ref = bvn_lower_owen(-h, -k, r)
    got = bvnu(torch.tensor(h, dtype=DT, device=device), torch.tensor(k, dtype=DT, device=device),
               torch.tensor(r, dtype=DT, device=device)).cpu().numpy()
    err = np.abs(got - ref)
    low = np.abs(r) < 0.925
    from scipy import integrate
    spot = []
    for i in range(0, len(h), len(h) // 24):
        if abs(r[i]) > 0.999:
            continue
        cov = 1.0 / (2 * math.pi * math.sqrt(1 - r[i] ** 2))
        dens = lambda y, x, rr=r[i]: cov * math.exp(-(x * x - 2 * rr * x * y + y * y) / (2 * (1 - rr * rr)))
        val, _ = integrate.dblquad(dens, h[i], h[i] + 12, k[i], k[i] + 12, epsabs=1e-13, epsrel=1e-11)
        spot.append(abs(val - got[i]))
    return {
        "cases": int(len(h)), "max_abs_error_low_branch": float(err[low].max()),
        "max_abs_error_high_branch": float(err[~low].max()),
        "reference": "Owen's T formula with scipy.special.owens_t; dblquad spot checks",
        "dblquad_spot_checks": len(spot), "dblquad_max_abs_error": float(max(spot)),
        "passed": float(err.max()) < 1e-12 and max(spot) < 1e-9,
    }


def validate_bivariate(device):
    rng = np.random.default_rng([BASE_SEED, 12])
    rhos = [-(1 - 1e-6), -0.999, -0.95, -0.93, -0.9, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9, 0.93, 0.95, 0.999, 1 - 1e-6]
    worst = 0.0
    worst_cov = 0.0
    count = 0
    for rho in rhos:
        for _ in range(16):
            mu1, mu2 = rng.uniform(-3, 3, 2)
            s1, s2 = rng.uniform(0.2, 3.0, 2)
            t = lambda v: torch.tensor([v], dtype=DT, device=device)
            got = float(relu_cross_moment(t(mu1), t(s1), t(mu2), t(s2), t(rho)))
            ref = relu_cross_reference(mu1, s1, mu2, s2, rho)
            m1 = relu_moment_reference(mu1, s1, 1)
            m2 = relu_moment_reference(mu2, s2, 1)
            worst = max(worst, abs(got - ref) / (s1 * s2 + abs(mu1 * mu2)))
            worst_cov = max(worst_cov, abs((got - m1 * m2) - (ref - m1 * m2)) / (s1 * s2))
            count += 1
    return {"cases": count, "correlations": rhos, "max_scaled_error": worst, "max_cov_error_over_s1s2": worst_cov,
            "reference": "1D quadrature of U * E[ReLU(V) | U] (scipy quad), independent of the orthant formula",
            "passed": worst < 1e-10}


def validate_depth_one(device):
    """A one-hidden-layer network with Gaussian inputs: layer 1 is exactly Gaussian, so the
    closures' means are exact, and C's output variance is exact as well (D's is not)."""
    out = []
    ok = True
    for idx, (sw2, sb, w) in enumerate([(2.0, 0.0, 16), (2.0, 0.1, 16), (4.0, 0.1, 12), (1.0, 0.1, 24)]):
        raw = draw_context(1, w, idx, sw2, sb, stream="validate")
        net = Net(raw, device)
        W, b, a = raw["W"][0], raw["b"][0], raw["a"]
        sd = np.sqrt((W * W).sum(1))
        m = np.array([relu_moment_reference(b[i], sd[i], 1) for i in range(w)])
        exact_mean = float(a @ m)
        cov = np.zeros((w, w))
        G = W @ W.T
        for i in range(w):
            cov[i, i] = relu_moment_reference(b[i], sd[i], 2) - m[i] ** 2
            for j in range(i + 1, w):
                rho = G[i, j] / (sd[i] * sd[j])
                cov[i, j] = cov[j, i] = relu_cross_reference(b[i], sd[i], b[j], sd[j], rho) - m[i] * m[j]
        exact_sd = math.sqrt(float(a @ cov @ a))
        muD, sdD = closure_diag(net, tail=True)
        muC1, _, _ = closure_full(net, tail=False)
        muC, sdC, _ = closure_full(net, tail=True)
        row = {"sigma_w2": sw2, "sigma_b": sb, "width": w, "exact_mean": exact_mean, "exact_sd": exact_sd,
               "D_mean_error": muD - exact_mean, "C_mean_error": muC - exact_mean, "C_T1_mean_error": muC1 - exact_mean,
               "C_sd_error": sdC - exact_sd, "D_sd_error_expected_nonzero": sdD - exact_sd}
        ok &= max(abs(row["D_mean_error"]), abs(row["C_mean_error"]), abs(row["C_T1_mean_error"])) < 1e-12
        ok &= abs(row["C_sd_error"]) < 1e-10
        out.append(row)
    return {"networks": out, "reference": "per-neuron and per-pair scipy quadrature", "passed": bool(ok)}


def validate_averaging_exhaustive(device):
    """Float64 A on every input of small sign networks: |true - est| <= bound, the signed interval,
    and the pointwise radius and remainder certificates for every input and neuron."""
    tol = 1e-10
    instances = viol_bound = viol_interval = viol_point = 0
    ratios = []
    exact_zero_cases = 0
    rows = []
    for d, w, L, sw2, sb, k in itertools.product([8, 12, 16], [16, 64], [2, 3], [1.0, 2.0, 4.0], [0.0, 0.1], KEPT):
        if k > d:
            continue
        X = sign_cube(d, device)
        cell = ((X[:, :k] + 1) / 2) @ torch.tensor([2.0 ** (k - 1 - j) for j in range(k)], dtype=DT, device=device)
        cell = cell.long()
        for ctx in range(4):
            raw = draw_context(L, w, 1000 * d + ctx, sw2, sb, stream="validate", d=d)
            net = Net(raw, device)
            (est, bound, lower, upper), tr = averaging(net, k, keep_trace=True)
            h = X
            point_ok = True
            for l, (W, b) in enumerate(zip(net.Ws, net.bs)):
                z = h @ W.T + b
                ref = tr["pres"][l][cell]
                diff = z - ref
                point_ok &= bool((diff.abs() <= tr["deltas"][l][cell] + tol).all())
                rem = z.clamp_min(0) - ref.clamp_min(0) - (ref > 0).to(DT) * diff
                point_ok &= bool(((rem >= -tol) & (rem <= tr["ramps"][l][cell] + tol)).all())
                h = z.clamp_min(0)
            true = float((h @ net.a).mean())
            e = true - est
            instances += 1
            viol_bound += abs(e) > bound + tol
            viol_interval += not (lower - tol <= e <= upper + tol)
            viol_point += not point_ok
            if bound == 0:
                exact_zero_cases += 1
                assert abs(e) < tol
            else:
                ratios.append(abs(e) / bound)
            rows.append([d, w, L, sw2, sb, k, ctx, true, est, bound, lower, upper])
    ratios = np.array(ratios)
    return {
        "instances": instances, "bound_violations": int(viol_bound), "interval_violations": int(viol_interval),
        "pointwise_certificate_failures": int(viol_point), "zero_bound_instances": exact_zero_cases,
        "error_over_bound": {"median": float(np.median(ratios)), "max": float(ratios.max()),
                             "p10": float(np.quantile(ratios, 0.1)), "p90": float(np.quantile(ratios, 0.9))},
        "grid": "d in {8,12,16}, width in {16,64}, depth in {2,3}, sigma_w2 in {1,2,4}, sigma_b in {0,0.1}, "
                "k in {4,8} (k <= d), 4 contexts each; every one of the 2^d inputs enumerated",
        "columns": ["d", "width", "depth", "sigma_w2", "sigma_b", "k", "context", "true", "estimate", "bound", "lower", "upper"],
        "rows": rows, "tolerance": tol,
        "passed": viol_bound == 0 and viol_interval == 0 and viol_point == 0,
    }


def validate_averaging_rational(device):
    """Exact rationals through boundary.py itself: its prepare/estimate on the float64 weights
    read as exact dyadic rationals, its exhaustive_reference (exact pointwise identities) for
    d = 10, and agreement of the float64 implementation used in the sweep."""
    sys.path.insert(0, str(HERE))
    import boundary  # noqa: PLC0415  (the essay's own exact-rational implementation)

    cases = [(10, 6, 2, 2.0, 0.1, 4), (10, 6, 3, 4.0, 0.0, 4), (10, 8, 2, 1.0, 0.1, 8), (10, 5, 3, 2.0, 0.1, 8),
             (64, 16, 2, 2.0, 0.1, 4), (64, 16, 3, 2.0, 0.0, 4), (64, 16, 2, 4.0, 0.1, 8)]
    out = []
    ok = True
    for i, (d, w, L, sw2, sb, k) in enumerate(cases):
        started = time.perf_counter()
        raw = draw_context(L, w, 5000 + i, sw2, sb, stream="validate", d=d)
        layers = [([[Fraction(float(x)) for x in row] for row in W], [Fraction(float(v)) for v in b])
                  for W, b in zip(raw["W"], raw["b"])]
        readout = [Fraction(float(v)) for v in raw["a"]]
        summary = boundary.prepare(layers, d, k)
        pred, bound, lower, upper, uniform = boundary.estimate(layers, summary, readout, Fraction(0))
        est64, bound64, lower64, upper64 = averaging(Net(raw, device), k)
        row = {"d": d, "width": w, "depth": L, "sigma_w2": sw2, "sigma_b": sb, "k": k,
               "estimate_exact": float(pred), "bound_exact": float(bound),
               "float64_estimate_error": est64 - float(pred), "float64_bound_error": bound64 - float(bound),
               "float64_interval_error": max(abs(lower64 - float(lower)), abs(upper64 - float(upper)))}
        ok &= abs(row["float64_estimate_error"]) < 1e-12 and abs(row["float64_bound_error"]) < 1e-12 * max(1.0, float(bound))
        if d <= 12:
            mean, identities = boundary.exhaustive_reference(layers, summary, [(readout, Fraction(0))])
            true = sum((x * y for x, y in zip(readout, mean)), Fraction(0))
            diff = true - pred
            row.update({"true_exact": float(true), "difference_exact": float(diff),
                        "exact_identities_checked": identities,
                        "bound_holds_exactly": abs(diff) <= bound, "interval_holds_exactly": lower <= diff <= upper})
            ok &= row["bound_holds_exactly"] and row["interval_holds_exactly"]
        row["seconds"] = time.perf_counter() - started
        out.append(row)
    return {"cases": out, "implementation": "boundary.prepare, boundary.estimate, boundary.exhaustive_reference",
            "passed": bool(ok)}


def validate_truth(device):
    """The GPU truth pipeline against exact answers: a depth-one Gaussian network (quadrature)
    and a 16-sign depth-two network (enumeration), plus a CPU NumPy forward-pass comparison."""
    rows = []
    ok = True
    raw = draw_context(1, 16, 0, 2.0, 0.1, stream="validate")
    W, b, a = raw["W"][0], raw["b"][0], raw["a"]
    sd = np.sqrt((W * W).sum(1))
    exact = float(a @ np.array([relu_moment_reference(b[i], sd[i], 1) for i in range(16)]))
    net = Net(raw, device)
    tr = truth_mc(net, "gauss", 2 ** 22, seed_int("validate-truth", 0), [0.0], device)
    z = (tr["mean"] - exact) / tr["se"]
    rows.append({"case": "depth 1, gauss, w 16", "exact": exact, "mc": tr["mean"], "se": tr["se"], "z": z})
    ok &= abs(z) < 5
    raw = draw_context(2, 64, 1, 2.0, 0.1, stream="validate", d=16)
    net = Net(raw, device)
    X = sign_cube(16, device)
    exact = float(forward(net, X).mean())
    tr = truth_mc(net, "sign", 2 ** 22, seed_int("validate-truth", 1), [0.0], device)
    z = (tr["mean"] - exact) / tr["se"]
    rows.append({"case": "depth 2, 16 signs, w 64", "exact": exact, "mc": tr["mean"], "se": tr["se"], "z": z})
    ok &= abs(z) < 5
    Xc = X[:256].cpu().numpy()
    h = Xc
    for Wl, bl in zip(raw["W"], raw["b"]):
        h = np.maximum(h @ Wl.T + bl, 0)
    cpu = h @ raw["a"]
    diff = float(np.abs(cpu - forward(net, X[:256]).cpu().numpy()).max())
    ok &= diff < 1e-12
    return {"checks": rows, "gpu_vs_numpy_forward_max_abs": diff, "passed": bool(ok)}


def run_validation(device):
    started = time.perf_counter()
    out = {}
    for name, fn in [("univariate_relu_moments", validate_univariate), ("bivariate_normal_orthant", validate_bvn),
                     ("bivariate_relu_moment", validate_bivariate), ("depth_one_gaussian_exactness", validate_depth_one),
                     ("averaging_exhaustive_float64", validate_averaging_exhaustive),
                     ("averaging_exact_rational", validate_averaging_rational), ("truth_pipeline", validate_truth)]:
        t0 = time.perf_counter()
        out[name] = fn(device)
        out[name]["seconds"] = time.perf_counter() - t0
        print(f"[validate] {name}: passed={out[name]['passed']} ({out[name]['seconds']:.1f} s)", flush=True)
    out["passed"] = all(v["passed"] for v in out.values() if isinstance(v, dict))
    out["seconds"] = time.perf_counter() - started
    return out


# ----------------------------------------------------------------------------------------------
# Sweep.


def now(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def settings_grid(full_1024=True):
    grid = []
    for w in FAMILY["widths"]:
        for L in FAMILY["depths"]:
            for kind in FAMILY["inputs"]:
                for sw2 in FAMILY["sigma_w2"]:
                    for sb in FAMILY["sigma_b"]:
                        if w == 1024 and not full_1024 and sw2 != 2.0:
                            continue
                        grid.append({"depth": L, "input": kind, "sigma_w2": sw2, "sigma_b": sb, "width": w})
    return grid


def setting_key(s):
    key = f"L{s['depth']}-{s['input']}-sw{s['sigma_w2']:g}-sb{s['sigma_b']:g}-w{s['width']}"
    d = s.get("input_dim", FAMILY["input_dim"])
    return key if d == FAMILY["input_dim"] else f"{key}-d{d}"


def run_context(net, s, ctx, budgets, n_pairs, device):
    rec, tm = {}, {}
    kind = s["input"]
    t = now(device)
    muD, sdD = closure_diag(net, tail=True)
    tm["D"] = now(device) - t
    t = now(device)
    muC1, _, _ = closure_full(net, tail=False)
    tm["C"] = now(device) - t
    t = now(device)
    muC, sdC, high = closure_full(net, tail=True)
    tm["C2"] = now(device) - t
    rec.update(D_mean=muD, D_sd=sdD, C_mean=muC1, C_sd=sdC, C_full_minus_T1_mean=muC - muC1, C_high_r_pairs=high)
    if kind == "sign":
        for k in KEPT:
            t = now(device)
            est, bound, lower, upper = averaging(net, k)
            tm[f"A{k}"] = now(device) - t
            rec.update({f"A{k}_est": est, f"A{k}_bound": bound, f"A{k}_lower": lower, f"A{k}_upper": upper})
    thresholds = [muC + kk * sdC for kk in TAIL_K] + [muD + kk * sdD for kk in TAIL_K]
    key = setting_key(s)
    t = now(device)
    tr = truth_mc(net, kind, n_pairs, seed_int("truth", key, ctx), thresholds, device)
    tm["truth"] = now(device) - t
    rec.update(truth_mean=tr["mean"], truth_se=tr["se"], truth_var=tr["var"],
               truth_antithetic_ratio=tr["antithetic_variance_ratio"])
    for i, kk in enumerate(TAIL_K):
        rec[f"truth_pC{kk}"] = tr["p"][i]
        rec[f"truth_pC{kk}_se"] = tr["p_se"][i]
        rec[f"truth_pD{kk}"] = tr["p"][len(TAIL_K) + i]
        rec[f"truth_pD{kk}_se"] = tr["p_se"][len(TAIL_K) + i]
    for name, n in budgets.items():
        t = now(device)
        mean, tails = sampling(net, kind, n, seed_int("S", name, key, ctx), thresholds[:len(TAIL_K)], device)
        tm[f"S@{name}"] = now(device) - t
        rec[f"S@{name}"] = mean
        if name.startswith("C2"):
            for i, kk in enumerate(TAIL_K):
                rec[f"S@{name}_tail{kk}"] = tails[i]
    return rec, tm


def _q(x):
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def summarize_setting(s, per, costs, budgets, per1, per2):
    kind = s["input"]
    T = np.array(per["truth_mean"])
    se = np.array(per["truth_se"])
    var = np.array(per["truth_var"])
    n_ctx = len(T)
    noise2 = float(np.mean(se ** 2))
    std_raw = float(np.std(T, ddof=1))
    stdT = math.sqrt(max(std_raw ** 2 - noise2, 0.0))

    def stats(G, scale):
        e = np.asarray(G, dtype=float) - T
        mse_raw = float(np.mean(e ** 2))
        rmse = math.sqrt(max(mse_raw - noise2, 0.0))
        return {"rmse": rmse, "rmse_raw": math.sqrt(mse_raw), "rel_rmse": rmse / scale if scale > 0 else None,
                "bias": float(e.mean()), "bias_se": float(e.std(ddof=1) / math.sqrt(n_ctx)),
                "truth_se_over_rmse": math.sqrt(noise2) / rmse if rmse > 0 else None}

    t1 = {}
    for name in ["D", "C"] + ([f"A{k}" for k in KEPT] if kind == "sign" else []):
        col = per[f"{name}_mean"] if name in ("D", "C") else per[f"{name}_est"]
        t1[name] = stats(col, stdT) | {"flops": costs[name]}
    for name, n in budgets.items():
        if name.startswith("C2"):
            continue
        st = stats(per[f"S@{name}"], stdT)
        st.update({"n": n, "flops": n * per1, "rmse_expected": math.sqrt(float(var.mean()) / n)})
        st["rel_rmse_expected"] = st["rmse_expected"] / stdT
        t1[f"S@{name}"] = st
    comparisons = {}
    for name in t1:
        if name.startswith("S@"):
            continue
        ref = t1[f"S@{name}x1"]
        comparisons[name] = {
            "beats_matched_S": t1[name]["rmse"] < ref["rmse_expected"],
            "rmse_over_matched_S_expected": t1[name]["rmse"] / ref["rmse_expected"],
            "beats_10x_S": t1[name]["rmse"] < t1[f"S@{name}x10"]["rmse_expected"],
            "beats_100x_S": t1[name]["rmse"] < t1[f"S@{name}x100"]["rmse_expected"],
            "samples_S_needs_to_match": float(var.mean()) / t1[name]["rmse"] ** 2 if t1[name]["rmse"] > 0 else None,
        }
        if f"{name}_tau1" in costs:
            n1 = max(1, round(costs[f"{name}_tau1"] / per1))
            comparisons[name]["matched_n_tau1"] = n1
            comparisons[name]["rmse_over_matched_S_expected_tau1"] = t1[name]["rmse"] / math.sqrt(float(var.mean()) / n1)

    t2 = {}
    muD, sdD = np.array(per["D_mean"]), np.array(per["D_sd"])
    muC, sdC = np.array(per["C_mean"]), np.array(per["C_sd"])
    for kk in TAIL_K:
        p = np.array(per[f"truth_pC{kk}"])
        psep = np.array(per[f"truth_pC{kk}_se"])
        noise_p = float(np.mean(psep ** 2))
        meanp = float(p.mean())
        stdp = math.sqrt(max(float(np.var(p, ddof=1)) - noise_p, 0.0))

        def pstats(G):
            e = np.asarray(G, dtype=float) - p
            mse_raw = float(np.mean(e ** 2))
            rmse = math.sqrt(max(mse_raw - noise_p, 0.0))
            return {"rmse": rmse, "rel_rmse_vs_mean_p": rmse / meanp if meanp > 0 else None,
                    "rel_rmse_vs_std_p": rmse / stdp if stdp > 0 else None, "bias": float(e.mean())}

        thr = muC + kk * sdC
        entry = {"threshold_rule": f"t = C mean + {kk} * C sd", "gaussian_value": _q(kk),
                 "true_p_mean": meanp, "true_p_std": stdp, "true_p_min": float(p.min()), "true_p_max": float(p.max()),
                 "truth_se_mean": float(psep.mean()),
                 "C": pstats(np.full(n_ctx, _q(kk))) | {"flops": costs["C2"]},
                 "D": pstats([_q((t - m) / sd) for t, m, sd in zip(thr, muD, sdD)]) | {"flops": costs["D_tail"]}}
        for m in MULTIPLIERS:
            n = budgets[f"C2x{m}"]
            st = pstats(per[f"S@C2x{m}_tail{kk}"])
            st.update({"n": n, "flops": n * per2, "rmse_expected": math.sqrt(float(np.mean(p * (1 - p))) / n)})
            entry[f"S@C2x{m}"] = st
        pD = np.array(per[f"truth_pD{kk}"])
        entry["D_own_threshold"] = {"threshold_rule": f"t = D mean + {kk} * D sd", "true_p_mean": float(pD.mean()),
                                    "true_p_std": float(pD.std(ddof=1)),
                                    "rmse_of_gaussian_value": float(np.sqrt(np.mean((pD - _q(kk)) ** 2)))}
        t2[f"k{kk}"] = entry

    a_stats = {}
    if kind == "sign":
        for k in KEPT:
            est = np.array(per[f"A{k}_est"])
            bound = np.array(per[f"A{k}_bound"])
            lo, hi = np.array(per[f"A{k}_lower"]), np.array(per[f"A{k}_upper"])
            e = T - est
            a_stats[f"A{k}"] = {
                "bound_holds_point_truth": int((np.abs(e) <= bound).sum()),
                "significant_violations_4se": int((np.abs(e) > bound + 4 * se).sum()),
                "interval_violations_4se": int(((e < lo - 4 * se) | (e > hi + 4 * se)).sum()),
                "contexts": n_ctx,
                "bound_median": float(np.median(bound)), "bound_mean": float(bound.mean()),
                "bound_over_abs_error_median": float(np.median(bound / np.maximum(np.abs(e), 1e-300))),
                "bound_over_abs_error_min": float(np.min(bound / np.maximum(np.abs(e), 1e-300))),
                "bound_over_truth_std_median": float(np.median(bound)) / stdT,
                "bound_over_rmse": float(np.sqrt(np.mean(bound ** 2))) / t1[f"A{k}"]["rmse"],
            }
    return {
        "contexts": n_ctx, "truth_std_across_contexts": stdT, "truth_std_raw": std_raw,
        "truth_se_rms": math.sqrt(noise2), "mean_output_sd_over_x": float(np.sqrt(var).mean()),
        "C_sd_over_true_sd_median": float(np.median(np.array(per["C_sd"]) / np.sqrt(var))),
        "D_sd_over_true_sd_median": float(np.median(np.array(per["D_sd"]) / np.sqrt(var))),
        "antithetic_variance_ratio_median": float(np.median(per["truth_antithetic_ratio"])),
        "C_high_r_pairs_total": int(np.sum(per["C_high_r_pairs"])),
        "C_full_vs_T1_mean_max_abs": float(np.max(np.abs(per["C_full_minus_T1_mean"]))),
        "T1": t1, "T1_comparisons": comparisons, "T2": t2, "A_bound": a_stats,
    }


def _r(x):
    """Round floats to 12 significant digits; also the JSON default hook for numpy scalars."""
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return float(f"{x:.12g}") if math.isfinite(x) else str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (int, str, list, dict, type(None))):
        return x
    raise TypeError(f"cannot serialize {type(x).__name__}")


def run_setting(s, device, n_ctx, n_pairs, stream="main"):
    d = s.get("input_dim", FAMILY["input_dim"])
    L, w, kind = s["depth"], s["width"], s["input"]
    costs, budgets, per1, per2 = cost_table(d, w, L, kind)
    warm = Net(draw_context(L, w, 0, s["sigma_w2"], s["sigma_b"], stream="pilot", d=d), device)
    run_context(warm, s, -1, {k: v for k, v in budgets.items() if k.endswith("x1")}, min(n_pairs, 2 ** 16), device)
    per, times = {}, {}
    started = time.perf_counter()
    for c in range(n_ctx):
        net = Net(draw_context(L, w, c, s["sigma_w2"], s["sigma_b"], stream=stream, d=d), device)
        rec, tm = run_context(net, s, c, budgets, n_pairs, device)
        for k, v in rec.items():
            per.setdefault(k, []).append(v)
        for k, v in tm.items():
            times.setdefault(k, []).append(v)
    wall = time.perf_counter() - started
    summary = summarize_setting(s, per, costs, budgets, per1, per2)
    timing = {k: {"mean_s": float(np.mean(v)), "median_s": float(np.median(v))} for k, v in times.items()}
    return {
        "setting": s, "key": setting_key(s), "contexts": n_ctx, "truth_samples_per_context": 2 * n_pairs,
        "costs_flops": costs, "S_budgets": budgets, "S_flops_per_sample": {"T1": per1, "T2": per2},
        "summary": summary, "wall_time": timing, "setting_wall_seconds": wall,
        "per_context": {k: [_r(float(x)) for x in v] for k, v in per.items()
                        if k not in ("C_full_minus_T1_mean", "truth_antithetic_ratio")},
    }


def dump(obj):
    """Indented JSON whose per-context arrays stay on one line each."""
    arrays = {}

    def strip(o):
        if isinstance(o, dict):
            return {k: (_hold(v) if k == "per_context" else strip(v)) for k, v in o.items()}
        if isinstance(o, list):
            return [strip(v) for v in o]
        return o

    def _hold(pc):
        held = {}
        for k, v in pc.items():
            token = f"@@{len(arrays)}@@"
            arrays[token] = json.dumps(v, separators=(",", ":"))
            held[k] = token
        return held

    text = json.dumps(strip(obj), indent=1, default=_r)
    for token, value in arrays.items():
        text = text.replace(f'"{token}"', value, 1)
    return text + "\n"


def wall_time_comparison(res):
    """Post hoc: compare each structural estimator with sampling at equal measured GPU wall time.

    For estimator E with median wall time t_E, n_wall is the largest S budget in the same setting
    whose median wall time is at most t_E. Every S budget here is launch-bound, so n_wall is a
    lower bound on what S could afford in t_E; the comparison therefore favors E. RMSE of S at
    n_wall is the exact plain-sampling value sqrt(mean_c Var_x(y) / n_wall).
    """
    sm, wt, budgets = res["summary"], res["wall_time"], res["S_budgets"]
    mean_var = float(np.mean(res["per_context"]["truth_var"]))
    out = {}
    for name, st in sm["T1"].items():
        if name.startswith("S@"):
            continue
        t_E = wt["C" if name == "C" else name]["median_s"]
        fits = [n for b, n in budgets.items() if wt[f"S@{b}"]["median_s"] <= t_E]
        n_wall = max(fits) if fits else None
        out[name] = {"median_wall_s": t_E, "n_wall_lower_bound": n_wall,
                     "largest_S_budget_run": max(budgets.values()),
                     "rmse_over_S_at_n_wall": st["rmse"] / math.sqrt(mean_var / n_wall) if n_wall else None}
    return out


def environment(device):
    env = {
        "python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__,
        "cuda": torch.version.cuda, "platform": platform.platform(), "hostname": socket.gethostname(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "dtype": "float64 everywhere (closures, A, S and the truth)",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "boundary_py_sha256": hashlib.sha256((HERE / "boundary.py").read_bytes()).hexdigest(),
        "argv": sys.argv,
    }
    import scipy
    env["scipy"] = scipy.__version__
    return env


def config_dict(full_1024, contexts, pairs):
    return {
        "family": FAMILY,
        "model": "y = a . h_L, h_l = ReLU(W_l h_{l-1} + b_l), h_0 = x; no output bias",
        "inputs": {"sign": "64 independent Rademacher signs", "gauss": "x ~ N(0, I_64)"},
        "context": "W_l ~ N(0, sigma_w2/fan_in), b_l ~ N(0, sigma_b^2), a ~ N(0, 1/width); one draw per context",
        "advice": "the estimator's fixed rule; nothing depends on the context except the weights",
        "common_random_numbers": "contexts share standard-normal draws across sigma_w2, sigma_b and the input law",
        "contexts_per_width": contexts, "truth_antithetic_pairs_per_width": pairs,
        "width_1024": "full grid" if full_1024 else "sigma_w2 = 2 only (cost cap)",
        "budget_decision": "The pilot (2 pilot-stream contexts per depth, width and input law, timing only) "
                           "estimated 1.25 h for the full grid with 2^21 truth pairs at width 1024. That is under "
                           "the 3 h cap, so the full grid runs, and width 1024 gets 2^22 pairs (about 1.9 h).",
        "targets": {
            "T1": "E_x[y]",
            "T2": "P_x[y > t], t = mu_C + k sd_C, k in {3, 4}: a fixed rule applied to the C closure's own mean and "
                  "standard deviation (no sampling, no truth). The same t is used to score S, D and C, so all three "
                  "estimate the same number. C's estimate is then Q(k) by construction. We also record the true p "
                  "at D's own threshold mu_D + k sd_D.",
        },
        "estimators": {
            "S": "plain Monte Carlo, n = round(m * FLOPs(E) / FLOPs per sample) for E in {D, C, A4, A8} and m in "
                 "{1, 10, 100}; C2 budgets match C's T2 pipeline. Independent seeds per budget and context.",
            "D": "diagonal Gaussian closure (exact univariate ReLU moments)",
            "C": "full-covariance Gaussian closure; exact bivariate ReLU moments with nonzero means (Rosenbaum 1961 "
                 "truncated moments; Genz BVNU orthant probability, 20-point Gauss-Legendre in both branches); the T1 "
                 "estimate uses the reduced pipeline that stops at layer L's covariance diagonal",
            "A": "conditional averaging over the first k in {4, 8} signs with the Averaging.mean_error_le budget and "
                 "boundary radius exactly as boundary.py computes it, in float64; exact rationals for the validation "
                 "instances (boundary.py's own code)",
        },
        "kept_signs": KEPT, "tail_k": TAIL_K, "S_multipliers": MULTIPLIERS,
        "flop_model": {"TAU_per_transcendental": TAU, "UNIV_per_neuron": univ(), "PAIR_per_pair": pair(),
                       "sensitivity": "T1_comparisons also report C and D against S matched to FLOPs with TAU = 1",
                       "formulas": "flops_forward, flops_D, flops_C, flops_A in msp.py"},
        "error_statistics": "RMSE over contexts after subtracting the mean squared standard error of the truth "
                            "(estimator and truth noise are independent); relative RMSE divides by the std of the "
                            "truth across contexts with the same correction; rmse_expected for S is "
                            "sqrt(mean_c Var_x(y) / n), the exact MSE of unbiased plain Monte Carlo",
        "seeds": {"base": BASE_SEED, "weights": "numpy SeedSequence([base, stream, depth, width, d, context])",
                  "monte_carlo": "torch CUDA generator seeded by sha256 of (msp, base, purpose, setting, context)"},
    }


def sweep(args):
    device = torch.device("cuda")
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if args.validation:
        validation = json.loads(Path(args.validation).read_text())
    else:
        validation = run_validation(device)
        (work / "validation.json").write_text(json.dumps(validation, indent=1, default=_r))
    if not validation["passed"]:
        print("validation failed; refusing to run the sweep", flush=True)
        sys.exit(2)
    full_1024 = not args.cap_1024
    grid = settings_grid(full_1024)
    if args.only:
        wanted = set(args.only.split(","))
        grid = [g for g in grid if setting_key(g) in wanted]
        assert len(grid) == len(wanted), "unknown setting key in --only"
    results = []
    for s in grid:
        key = setting_key(s)
        path = work / f"{key}.json"
        if path.exists():
            results.append(json.loads(path.read_text()))
            print(f"[sweep] {key}: loaded", flush=True)
            continue
        res = run_setting(s, device, CONTEXTS[s["width"]], TRUTH_PAIRS[s["width"]])
        path.write_text(json.dumps(res, default=_r))
        results.append(res)
        print(f"[sweep] {key}: {res['setting_wall_seconds']:.1f} s", flush=True)
    added = []
    if args.added_later:
        for s in ADDED_LATER:
            key = setting_key(s)
            path = work / f"added-{key}.json"
            if path.exists():
                added.append(json.loads(path.read_text()))
                print(f"[added later] {key}: loaded", flush=True)
                continue
            res = run_setting(s, device, CONTEXTS[s["width"]], TRUTH_PAIRS[s["width"]])
            path.write_text(json.dumps(res, default=_r))
            added.append(res)
            print(f"[added later] {key}: {res['setting_wall_seconds']:.1f} s", flush=True)
    for res in results + added:
        res["summary"]["T1_wall_time_comparison"] = wall_time_comparison(res)
    pilot = json.loads(Path(args.pilot).read_text()) if args.pilot else None
    out = {
        "description": "Matching-sampling benchmark for the Week 6 essay: structural estimators against "
                       "cost-matched plain Monte Carlo on random ReLU networks.",
        "config": config_dict(full_1024, CONTEXTS, TRUTH_PAIRS),
        "environment": environment(device),
        "validation": validation,
        "pilot": pilot,
        "settings": results,
        "added_later": {"reason": ADDED_LATER_REASON, "settings": added} if added else None,
        "sweep_wall_seconds": time.perf_counter() - started,
    }
    Path(args.output).write_text(dump(out))
    print(f"wrote {args.output}", flush=True)


def pilot(args):
    """Time a few pilot-stream contexts per (depth, width) and extrapolate the grid."""
    device = torch.device("cuda")
    rows = []
    total_full = 0.0
    total_capped = 0.0
    for w in FAMILY["widths"]:
        for L in FAMILY["depths"]:
            for kind in FAMILY["inputs"]:
                s = {"depth": L, "input": kind, "sigma_w2": 2.0, "sigma_b": 0.1, "width": w}
                res = run_setting(s, device, args.contexts, TRUTH_PAIRS[w], stream="pilot")
                per_ctx = res["setting_wall_seconds"] / args.contexts
                n_settings = len(FAMILY["sigma_w2"]) * len(FAMILY["sigma_b"])
                est = per_ctx * CONTEXTS[w] * n_settings
                total_full += est
                total_capped += est if w < 1024 else est / len(FAMILY["sigma_w2"])
                rows.append({"width": w, "depth": L, "input": kind, "seconds_per_context": per_ctx,
                             "truth_seconds_per_context": res["wall_time"]["truth"]["mean_s"],
                             "estimated_seconds_for_this_width_depth_input": est,
                             "truth_se_rms": res["summary"]["truth_se_rms"],
                             "antithetic_variance_ratio_median": res["summary"]["antithetic_variance_ratio_median"],
                             "S_budgets": res["S_budgets"], "costs_flops": res["costs_flops"],
                             "wall_time": res["wall_time"]})
                print(f"[pilot] w={w} L={L} {kind}: {per_ctx:.3f} s/context, grid share {est / 60:.1f} min", flush=True)
    out = {"contexts_per_cell": args.contexts, "rows": rows, "estimated_full_grid_seconds": total_full,
           "estimated_capped_grid_seconds": total_capped, "contexts": CONTEXTS, "truth_pairs": TRUTH_PAIRS,
           "note": "pilot-stream contexts (never used in the sweep); timing only"}
    Path(args.output).write_text(json.dumps(out, indent=1, default=_r) + "\n")
    print(f"full grid ~{total_full / 3600:.2f} h, capped ~{total_capped / 3600:.2f} h", flush=True)


# ----------------------------------------------------------------------------------------------
# Tables.


def _pct(x):
    if x is None:
        return "n/a"
    return f"{100 * x:.3g}%" if x < 10 else f"{100 * x:.0f}%"


def _g(x, digits=3):
    return "n/a" if x is None else f"{x:.{digits}g}"


def _flops(x):
    return f"{x:.2e}".replace("e+0", "e").replace("e+", "e")


def tables(args):
    data = json.loads(Path(args.results).read_text())
    S = data["settings"]
    lines = []

    def emit(*xs):
        lines.extend(xs)

    emit("Costs in FLOPs and the matched sample counts n = FLOPs / FLOPs per sample. Neither the input law nor",
         "the sigmas change them.", "",
         "| Depth | Width | One sample | D | C (T1) | C (T2) | A, k=4 | A, k=8 | n for D | n for C | n for C (T2) | n for A4 | n for A8 |",
         "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    seen = set()
    for r in S:
        s = r["setting"]
        if s["input"] != "sign" or (s["depth"], s["width"]) in seen:
            continue
        seen.add((s["depth"], s["width"]))
        c, b = r["costs_flops"], r["S_budgets"]
        emit(f"| {s['depth']} | {s['width']} | {_flops(c['forward'])} | {_flops(c['D'])} | {_flops(c['C'])} | "
             f"{_flops(c['C2'])} | {_flops(c['A4'])} | {_flops(c['A8'])} | {b['Dx1']} | {b['Cx1']} | {b['C2x1']} | "
             f"{b['A4x1']} | {b['A8x1']} |")
    for L in FAMILY["depths"]:
        for kind in FAMILY["inputs"]:
            emit("", f"T1 (mean output), depth {L}, {kind} inputs.", "",
                 "| sigma_w^2 | sigma_b | Width | Truth std | D RMSE | D rel. | S@D rel. | C RMSE | C rel. | C bias | "
                 "S@C rel. | S@10C rel. | S@100C rel. | C / S@C |",
                 "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            for r in S:
                s = r["setting"]
                if s["depth"] != L or s["input"] != kind:
                    continue
                sm, t1 = r["summary"], r["summary"]["T1"]
                emit(f"| {s['sigma_w2']:g} | {s['sigma_b']:g} | {s['width']} | {_g(sm['truth_std_across_contexts'])} | "
                     f"{_g(t1['D']['rmse'])} | {_pct(t1['D']['rel_rmse'])} | {_pct(t1['S@Dx1']['rel_rmse_expected'])} | "
                     f"{_g(t1['C']['rmse'])} | {_pct(t1['C']['rel_rmse'])} | {_g(t1['C']['bias'], 2)} | "
                     f"{_pct(t1['S@Cx1']['rel_rmse_expected'])} | {_pct(t1['S@Cx10']['rel_rmse_expected'])} | "
                     f"{_pct(t1['S@Cx100']['rel_rmse_expected'])} | "
                     f"{_g(sm['T1_comparisons']['C']['rmse_over_matched_S_expected'], 2)} |")
    for L in FAMILY["depths"]:
        emit("", f"A (conditional averaging), depth {L}, sign inputs. The bound columns are medians over contexts.", "",
             "| sigma_w^2 | sigma_b | Width | A4 RMSE | A4 rel. | S@A4 rel. | A4 bound | A4 bound / truth std | "
             "A8 RMSE | A8 rel. | S@A8 rel. | A8 bound | A8 bound / truth std | Bound holds |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for r in S:
            s = r["setting"]
            if s["depth"] != L or s["input"] != "sign":
                continue
            sm, t1, ab = r["summary"], r["summary"]["T1"], r["summary"]["A_bound"]
            held = sum(ab[f"A{k}"]["bound_holds_point_truth"] for k in KEPT)
            emit(f"| {s['sigma_w2']:g} | {s['sigma_b']:g} | {s['width']} | {_g(t1['A4']['rmse'])} | "
                 f"{_pct(t1['A4']['rel_rmse'])} | {_pct(t1['S@A4x1']['rel_rmse_expected'])} | "
                 f"{_g(ab['A4']['bound_median'])} | {_g(ab['A4']['bound_over_truth_std_median'])} | "
                 f"{_g(t1['A8']['rmse'])} | {_pct(t1['A8']['rel_rmse'])} | {_pct(t1['S@A8x1']['rel_rmse_expected'])} | "
                 f"{_g(ab['A8']['bound_median'])} | {_g(ab['A8']['bound_over_truth_std_median'])} | "
                 f"{held}/{2 * sm['contexts']} |")
    emit("", "Measured GPU wall time (median per context, sign inputs, sigma_w^2 = 2, sigma_b = 0.1). n_wall is the "
         "largest S budget that ran no slower than C.", "",
         "| Depth | Width | D | C (T1) | A, k=8 | S@C | S@100C | Truth | n for C | n_wall | C / S at n_wall |",
         "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in S:
        s = r["setting"]
        if s["input"] != "sign" or s["sigma_w2"] != 2.0 or s["sigma_b"] != 0.1:
            continue
        wt, b = r["wall_time"], r["S_budgets"]
        wc = r["summary"].get("T1_wall_time_comparison", {}).get("C", {})
        ms = lambda k: f"{1e3 * wt[k]['median_s']:.2f} ms"
        emit(f"| {s['depth']} | {s['width']} | {ms('D')} | {ms('C')} | {ms('A8')} | {ms('S@Cx1')} | {ms('S@Cx100')} | "
             f"{1e3 * wt['truth']['median_s']:.0f} ms | {b['Cx1']} | {wc.get('n_wall_lower_bound', 'n/a')} | "
             f"{_g(wc.get('rmse_over_S_at_n_wall'), 2)} |")
    emit("", "T2 (tail probability at t = C mean + k C sd), sigma_w^2 = 2, sigma_b = 0.1. Relative RMSE divides by the "
         "mean true p over contexts.", "",
         "| Depth | Input | Width | k | Mean true p | Q(k) | C rel. | D rel. | S@C2 n | S@C2 rel. | S@10C2 rel. | S@100C2 rel. |",
         "| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in S:
        s = r["setting"]
        if s["sigma_w2"] != 2.0 or s["sigma_b"] != 0.1:
            continue
        for kk in TAIL_K:
            e = r["summary"]["T2"][f"k{kk}"]
            rel = lambda m: _pct(e[f"S@C2x{m}"]["rmse_expected"] / e["true_p_mean"])
            emit(f"| {s['depth']} | {s['input']} | {s['width']} | {kk} | {e['true_p_mean']:.3g} | {e['gaussian_value']:.3g} | "
                 f"{_pct(e['C']['rel_rmse_vs_mean_p'])} | {_pct(e['D']['rel_rmse_vs_mean_p'])} | {e['S@C2x1']['n']} | "
                 f"{rel(1)} | {rel(10)} | {rel(100)} |")
    added = (data.get("added_later") or {}).get("settings") or []
    if added:
        emit("", "Added later: input dimension, depth 2, sigma_w^2 = 2, sigma_b = 0.1 (the d = 64 rows come from the "
             "main grid).", "",
             "| Input | d | Width | Truth std | D rel. | C rel. | S@C n | S@C rel. | C / S@C |",
             "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        main = [r for r in S if r["setting"]["depth"] == 2 and r["setting"]["sigma_w2"] == 2.0
                and r["setting"]["sigma_b"] == 0.1 and r["setting"]["width"] in (256, 1024)]
        rows = sorted(added + main, key=lambda r: (r["setting"]["input"], r["setting"].get("input_dim", 64),
                                                   r["setting"]["width"]))
        for r in rows:
            s, sm = r["setting"], r["summary"]
            t1 = sm["T1"]
            emit(f"| {s['input']} | {s.get('input_dim', 64)} | {s['width']} | {_g(sm['truth_std_across_contexts'])} | "
                 f"{_pct(t1['D']['rel_rmse'])} | {_pct(t1['C']['rel_rmse'])} | {r['S_budgets']['Cx1']} | "
                 f"{_pct(t1['S@Cx1']['rel_rmse_expected'])} | "
                 f"{_g(sm['T1_comparisons']['C']['rmse_over_matched_S_expected'], 2)} |")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("validate")
    p.add_argument("--output", required=True)
    p = sub.add_parser("pilot")
    p.add_argument("--output", required=True)
    p.add_argument("--contexts", type=int, default=2)
    p = sub.add_parser("sweep")
    p.add_argument("--output", required=True)
    p.add_argument("--work-dir", required=True, help="per-setting checkpoints; reused when present")
    p.add_argument("--validation", help="reuse a validation record instead of rerunning it")
    p.add_argument("--pilot", help="pilot record to embed in the results")
    p.add_argument("--cap-1024", action="store_true", help="width 1024 only at sigma_w2 = 2")
    p.add_argument("--added-later", action="store_true", help="also run the ADDED_LATER diagnostic settings")
    p.add_argument("--only", help="comma-separated setting keys, e.g. L2-sign-sw2-sb0.1-w64 (partial rerun)")
    p = sub.add_parser("tables")
    p.add_argument("results")
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.command == "validate":
        device = torch.device("cuda")
        out = run_validation(device)
        out["environment"] = environment(device)
        Path(args.output).write_text(json.dumps(out, indent=1, default=_r) + "\n")
        print(f"validation passed: {out['passed']}", flush=True)
    elif args.command == "pilot":
        pilot(args)
    elif args.command == "sweep":
        sweep(args)
    else:
        tables(args)


if __name__ == "__main__":
    main()
