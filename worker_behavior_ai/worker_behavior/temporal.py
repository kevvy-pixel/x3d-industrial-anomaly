"""Learned temporal model: per-frame classifier + Viterbi state-machine decoding.

Replaces hand-tuned thresholds with a model trained on labelled clips:
  * a small MLP per-frame classifier over the shared feature vector (with
    temporal context), producing calibrated class probabilities
  * a state-transition matrix estimated from the training label sequences
  * Viterbi decoding, so states persist and only plausible transitions occur
    (this is what the old rule engine hard-coded as latch/hysteresis)

Artifacts live in worker_ai/weights/temporal_clf.pkl, written by
industrial/temporal/export_model.py (see training/13_export_temporal_model.py).

Online:  `update(feature_vector)` per sampled frame (rolling buffer, edge-padded).
Offline: `run_sequence(seq)` Viterbi-decodes a whole clip properly.
"""
import os
import pickle

import numpy as np
import torch
import torch.nn as nn

WEIGHTS = os.environ.get("MODEL_DIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))
MODEL_PATH = os.path.join(WEIGHTS, "temporal_clf.pkl")

CLASSES = ("normal", "single_hand", "touch_face", "board_drop")  # must match training
EPS = 1e-6


def _log(x):
    return np.log(np.asarray(x, dtype=np.float64) + EPS)


def viterbi(log_em, log_A):
    """log_em: (T,C) emission log-probs, log_A: (C,C) transition log-probs."""
    T = log_em.shape[0]
    dp = log_em[0].copy()
    bp = np.zeros((T, log_em.shape[1]), int)
    for t in range(1, T):
        cand = dp[:, None] + log_A
        bp[t] = cand.argmax(0)
        dp = cand.max(0) + log_em[t]
    path = np.zeros(T, int)
    path[-1] = int(dp.argmax())
    for t in range(T - 2, -1, -1):
        path[t] = bp[t + 1, path[t + 1]]
    return path


class MLPHead(nn.Module):
    """Small MLP classifier (defined here so pickled artifacts stay loadable)."""

    def __init__(self, d=405, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 4))
        self.register_buffer("mu", torch.zeros(d))
        self.register_buffer("sd", torch.ones(d))

    def set_norm(self, mu, sd):
        self.mu.copy_(torch.as_tensor(mu, dtype=torch.float32))
        self.sd.copy_(torch.as_tensor(sd, dtype=torch.float32))

    def forward(self, x):
        return self.net((x - self.mu) / self.sd)


class TemporalModel:
    """Per-frame classifier + learned transition matrix + Viterbi decoding."""

    def __init__(self, path=MODEL_PATH, ctx=None, self_boost=None):
        with open(path, "rb") as f:
            art = pickle.load(f)
        if art.get("kind") != "mlp":
            raise ValueError(f"unsupported model kind: {art.get('kind')!r}")
        clf = MLPHead(art["dim"], art.get("hidden", 256))
        clf.load_state_dict(art["state_dict"])       # includes mu/sd buffers
        self.clf = clf
        self.A = np.asarray(art["A"], dtype=np.float64)
        self.ctx = int(art.get("ctx", 2) if ctx is None else ctx)
        self.self_boost = float(art.get("self_boost", 3.0) if self_boost is None else self_boost)
        self.classes = list(art.get("classes", CLASSES))
        log_A = _log(self.A)
        for i in range(len(self.classes)):
            log_A[i, i] += np.log(self.self_boost)
        self.log_A = log_A
        self.clf.eval()
        self.reset()

    # ---------- probabilities ----------
    def proba(self, X):
        dev = next(self.clf.parameters()).device
        with torch.no_grad():
            t = torch.as_tensor(X, dtype=torch.float32, device=dev)
            return torch.softmax(self.clf(t), 1).cpu().numpy()

    # ---------- online ----------
    def reset(self):
        self._raw = []
        self._path = []
        self.last_proba = None

    def update(self, feat):
        """Push one 81-dim frame feature; return the current label string."""
        self._raw.append(np.asarray(feat, np.float32))
        keep = self.ctx * 2 + 1
        if len(self._raw) > keep:
            self._raw.pop(0)
        if not self._raw:
            return "normal"
        x = self._context(len(self._raw) - 1)
        p = self.proba(x[None, :])[0]
        self.last_proba = p
        path = viterbi(_log(p)[None, :], self.log_A)
        return self.classes[int(path[0])]

    def _context(self, i):
        from .features import contextualise
        return contextualise(self._raw, i, self.ctx)

    # ---------- offline ----------
    def run_sequence(self, seq):
        """seq: (N,81) -> list of labels (proper Viterbi over the whole clip)."""
        from .features import contextualise
        if len(seq) == 0:
            return []
        X = np.stack([contextualise(seq, i, self.ctx) for i in range(len(seq))])
        p = self.proba(X)
        path = viterbi(_log(p), self.log_A)
        self._path = list(path)
        return [self.classes[int(q)] for q in path]
