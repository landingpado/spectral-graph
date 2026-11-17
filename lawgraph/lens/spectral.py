"""
Spectral lens: regularized Laplacian eigenmaps.

    emb = Spectral(k=16)(G)
    emb.vals            # eigenvalues λ_2..λ_{k+1} of the (regularized) normalized Laplacian, ascending
    emb.vecs            # (n, k) embedding, rows aligned with G.ids
    emb.fiedler()       # boolean split along the sign of the first non-trivial eigenvector
    emb.gap()           # index of the largest relative gap in the spectrum: a community-count hint
    emb.localization()  # inverse participation ratio per eigenvector: ~1/n spread out, ~1 stuck on a few nodes

Why regularize: on degree-heterogeneous graphs (any citation network) the bottom of the normalized spectrum
is occupied by eigenvectors localized on weakly attached nodes -- they say nothing about the body of the
graph and make every body node look identical. Adding τ to every degree before normalizing (Amini et al.
2013; Qin & Rohe 2013) suppresses them. τ = mean degree is the standard default.

Solver: eigenvalues of L = I - D_τ^{-1/2} W D_τ^{-1/2} are 1 - eigenvalues of the normalized adjacency, so
we ask Lanczos for the *largest* algebraic eigenvalues of the adjacency -- well conditioned, no factorization,
seconds on a million-edge graph. Shift-invert on L is the slow, fill-heavy alternative.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from .base import Graph


@dataclass
class SpectralResult:
    vals: np.ndarray
    vecs: np.ndarray
    ids: np.ndarray
    normalized: bool
    tau: float = 0.0

    def fiedler(self) -> np.ndarray:
        return self.vecs[:, 0] >= 0

    def gap(self) -> int:
        v = np.r_[0.0, self.vals]
        rel = np.diff(v)[1:] / np.maximum(v[1:-1], 1e-12)
        return int(np.argmax(rel)) + 1

    def coords(self, dims=(0, 1)) -> np.ndarray:
        return self.vecs[:, list(dims)]

    def localization(self) -> np.ndarray:
        """Inverse participation ratio of each eigenvector (unit-normalized): 1/n means perfectly spread,
        values near 1 mean the vector lives on a handful of nodes and should be distrusted."""
        v = self.vecs / np.linalg.norm(self.vecs, axis=0, keepdims=True)
        return (v ** 4).sum(axis=0)


@dataclass
class Spectral:
    k: int = 16
    normalized: bool = True
    tau: float | str | None = "auto"   # degree regularizer; "auto" = mean degree, 0/None = off
    random_walk: bool = False          # rescale by D_τ^{-1/2} (random-walk eigenvectors); amplifies low-degree nodes
    tol: float = 1e-6
    dense_below: int = 3000

    def __call__(self, G: Graph) -> SpectralResult:
        n, k = G.n, min(self.k + 1, G.n - 1)
        d = np.asarray(G.degree, dtype=float)
        tau = 0.0 if self.tau in (None, 0) else (float(d.mean()) if self.tau == "auto" else float(self.tau))
        if not self.normalized:
            tau = 0.0
        if self.normalized:
            dinv = 1.0 / np.sqrt(d + tau)
            A = sp.diags(dinv) @ G.W @ sp.diags(dinv)            # normalized adjacency; eig(L) = 1 - eig(A)
            if n <= self.dense_below:
                avals, avecs = np.linalg.eigh(A.toarray())
                avals, avecs = avals[::-1][:k], avecs[:, ::-1][:, :k]
            else:
                avals, avecs = eigsh(A.tocsr(), k=k, which="LA", tol=self.tol)
                order = np.argsort(-avals); avals, avecs = avals[order], avecs[:, order]
            vals, vecs = 1.0 - avals, avecs
        else:
            L = sp.diags(d) - G.W
            if n <= self.dense_below:
                vals, vecs = np.linalg.eigh(L.toarray()); vals, vecs = vals[:k], vecs[:, :k]
            else:
                vals, vecs = eigsh(L.tocsr(), k=k, sigma=-1e-3, which="LM", tol=self.tol)
                order = np.argsort(vals); vals, vecs = vals[order], vecs[:, order]
        vals, vecs = vals[1:], vecs[:, 1:]                        # drop the trivial eigenvector
        if self.random_walk and self.normalized:
            vecs = vecs / np.sqrt(d + tau)[:, None]
        return SpectralResult(vals, vecs, G.ids, self.normalized, tau)
