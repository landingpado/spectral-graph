"""
Align spectral embeddings across time: principal angles between subspaces and Procrustes rotation.

    A = align(emb_1980, emb_1990)          # SpectralResult from the same Frame coordinates (or any shared id space)
    A.angles                               # principal angles (radians), ascending; 0 = identical directions
    A.grassmann                            # sqrt(sum angles^2): distance between the two k-dim subspaces
    A.chordal                              # sqrt(sum sin^2): alternative, bounded by sqrt(k)
    A.rotated                              # emb_2's vectors rotated onto emb_1's orientation (for plotting side by side)
    A.common                               # ids present in both

Subspaces are compared on the nodes they share; cumulative graphs grow, so the earlier slice's node set
is the comparison set. Use the same k and the same build settings on both sides.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Alignment:
    angles: np.ndarray
    grassmann: float
    chordal: float
    rotated: np.ndarray       # emb_2 restricted to common nodes, rotated into emb_1's frame
    reference: np.ndarray     # emb_1 restricted to common nodes
    common: np.ndarray
    k: int
    R: np.ndarray = None      # (k, k) rotation: emb_2.vecs[:, :k] @ R is in emb_1's orientation (all nodes, not just common)


def _orth(V: np.ndarray) -> np.ndarray:
    Q, _ = np.linalg.qr(V)
    return Q


def principal_angles(V1: np.ndarray, V2: np.ndarray) -> np.ndarray:
    """Angles between span(V1) and span(V2), both (n, k) on the same rows."""
    s = np.linalg.svd(_orth(V1).T @ _orth(V2), compute_uv=False)
    return np.arccos(np.clip(s, -1.0, 1.0))


def align(emb1, emb2, k: int | None = None) -> Alignment:
    """emb1, emb2: objects with .vecs (n_i, k_i) and .ids (n_i,). Compared on shared ids."""
    k = k or min(emb1.vecs.shape[1], emb2.vecs.shape[1])
    common, i1, i2 = np.intersect1d(emb1.ids, emb2.ids, assume_unique=True, return_indices=True)
    V1, V2 = emb1.vecs[i1, :k], emb2.vecs[i2, :k]
    th = principal_angles(V1, V2)
    # orthogonal Procrustes: rotate V2 onto V1
    U, _, Vt = np.linalg.svd(V2.T @ V1)
    R = U @ Vt
    return Alignment(th, float(np.sqrt((th ** 2).sum())), float(np.sqrt((np.sin(th) ** 2).sum())),
                     V2 @ R, V1, common, k, R)


def align_sequence(embs: list, k: int | None = None, reference: str = "last") -> list[np.ndarray]:
    """Rotate every embedding into one shared orientation by chaining consecutive Procrustes rotations.
    Returns one (n_i, k) array per embedding, rows aligned with embs[i].ids. reference="last" keeps the final
    slice as drawn (the fullest graph); "first" keeps the first."""
    k = k or min(e.vecs.shape[1] for e in embs)
    order = list(range(len(embs)))
    if reference == "last":
        order = order[::-1]
    out = [None] * len(embs)
    Rc = np.eye(k)
    out[order[0]] = embs[order[0]].vecs[:, :k].copy()
    for prev, cur in zip(order, order[1:]):
        Rc = align(embs[prev], embs[cur], k).R @ Rc          # cur -> prev -> ... -> reference
        out[cur] = embs[cur].vecs[:, :k] @ Rc
    return out


def drift(embs: list, k: int | None = None) -> np.ndarray:
    """Grassmann distance between consecutive embeddings: the time series of subspace change."""
    return np.array([align(embs[i], embs[i + 1], k).grassmann for i in range(len(embs) - 1)])
