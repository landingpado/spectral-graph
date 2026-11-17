"""
Community lens: Leiden partitions and their lineage across time.

    part = Leiden(resolution=0.05, objective="CPM")(G)        # labels aligned with G.ids
    part.sizes()                                               # community sizes, largest first
    part.top(k=10, score=G.degree)                             # representative node ids per community

    lin = lineage([part_1950, part_1960, ...], min_overlap=0.3)  # partitions on a shared Frame (or any id space)
    lin.edges                                                  # (t, community_t, t+1, community_t+1, overlap, jaccard)
    lin.chains()                                               # persistent communities: lists of (t, community) following the strongest link
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .base import Graph


@dataclass
class Partition:
    labels: np.ndarray            # int label per row of ids
    ids: np.ndarray
    t: str
    quality: float = float("nan")
    meta: dict = field(default_factory=dict)

    @property
    def k(self) -> int:
        return int(self.labels.max()) + 1 if len(self.labels) else 0

    def sizes(self) -> np.ndarray:
        return np.bincount(self.labels, minlength=self.k)

    def members(self, c: int) -> np.ndarray:
        return self.ids[self.labels == c]

    def top(self, k: int = 10, score: np.ndarray | None = None) -> dict[int, np.ndarray]:
        """Top-k ids per community by `score` (default: arbitrary). Use G.degree or in-citations."""
        out = {}
        for c in range(self.k):
            m = np.where(self.labels == c)[0]
            if score is not None:
                m = m[np.argsort(-score[m])]
            out[c] = self.ids[m[:k]]
        return out

    def onehot(self) -> sp.csr_matrix:
        """(n, k) indicator matrix."""
        n = len(self.labels)
        return sp.csr_matrix((np.ones(n), (np.arange(n), self.labels)), shape=(n, self.k))


@dataclass
class Leiden:
    resolution: float = 1.0
    objective: str = "modularity"     # "modularity" | "CPM"  (CPM: resolution = min internal density, size-controlling)
    weighted: bool = True
    n_iterations: int = -1            # -1: iterate until stable
    seed: int | None = 0
    min_size: int = 1                 # communities smaller than this are pooled into one trailing label

    def __call__(self, G: Graph) -> Partition:
        import igraph as ig
        import leidenalg as la
        W = sp.triu(G.W, k=1).tocoo()
        g = ig.Graph(n=G.n, edges=np.column_stack([W.row, W.col]).tolist(), directed=False)
        kw = {"weights": W.data.tolist()} if self.weighted else {}
        if self.objective == "CPM":
            part = la.find_partition(g, la.CPMVertexPartition, resolution_parameter=self.resolution,
                                     n_iterations=self.n_iterations, seed=self.seed, **kw)
        else:
            part = la.find_partition(g, la.RBConfigurationVertexPartition, resolution_parameter=self.resolution,
                                     n_iterations=self.n_iterations, seed=self.seed, **kw)
        labels = np.asarray(part.membership)
        # relabel by size, largest = 0; optionally pool tiny communities into one trailing label
        order = np.argsort(-np.bincount(labels))
        remap = np.empty_like(order); remap[order] = np.arange(len(order))
        labels = remap[labels]
        if self.min_size > 1:
            sizes = np.bincount(labels)
            small = sizes < self.min_size
            if small.any():
                big_k = int((~small).sum())
                labels = np.where(small[labels], big_k, labels)
        Q = float(g.modularity(part.membership, weights=kw.get("weights")))   # normalized modularity, not leidenalg's raw quality
        return Partition(labels, G.ids, G.t, Q,
                         {"resolution": self.resolution, "objective": self.objective, "n_edges": int(W.nnz)})


# ---------------------------------------------------------------- lineage across time
@dataclass
class Lineage:
    parts: list[Partition]
    edges: list[tuple]            # (i, c_i, i+1, c_j, overlap_frac, jaccard)
    overlaps: list[np.ndarray]    # dense (k_i, k_{i+1}) shared-node counts per consecutive pair

    def chains(self, min_jaccard: float = 0.0) -> list[list[tuple[int, int]]]:
        """Follow each community forward along its strongest Jaccard link. Returns chains of (slice_index, label)."""
        nxt: dict[tuple, tuple] = {}
        for i, ci, j, cj, frac, jac in self.edges:
            if jac >= min_jaccard and jac > nxt.get((i, ci), (None, -1.0))[1]:
                nxt[(i, ci)] = ((j, cj), jac)
        seen, chains = set(), []
        for i, p in enumerate(self.parts):
            for c in range(p.k):
                if (i, c) in seen:
                    continue
                chain, cur = [], (i, c)
                while cur is not None and cur not in seen:
                    seen.add(cur); chain.append(cur)
                    cur = nxt.get(cur, (None, None))[0]
                if len(chain) > 1:
                    chains.append(chain)
        return sorted(chains, key=len, reverse=True)

    def events(self, min_frac: float = 0.3):
        """Splits and merges: communities with >1 significant successor / predecessor."""
        succ, pred = {}, {}
        for i, ci, j, cj, frac, jac in self.edges:
            if frac >= min_frac:
                succ.setdefault((i, ci), []).append((j, cj, frac))
        for i, ci, j, cj, frac, jac in self.edges:
            # fraction of the successor that came from this predecessor
            share = self.overlaps[i][ci, cj] / max(self.parts[j].sizes()[cj], 1)
            if share >= min_frac:
                pred.setdefault((j, cj), []).append((i, ci, share))
        return {"splits": {k: v for k, v in succ.items() if len(v) > 1},
                "merges": {k: v for k, v in pred.items() if len(v) > 1}}


def lineage(parts: list[Partition], min_overlap: float = 0.1) -> Lineage:
    """Link communities in consecutive partitions by shared node ids. Partitions may have different id
    sets (cumulative graphs grow); overlap_frac = shared / size of the earlier community."""
    edges, overlaps = [], []
    for i in range(len(parts) - 1):
        a, b = parts[i], parts[i + 1]
        common, ia, ib = np.intersect1d(a.ids, b.ids, assume_unique=True, return_indices=True)
        Pa = sp.csr_matrix((np.ones(len(ia)), (np.arange(len(ia)), a.labels[ia])), shape=(len(ia), a.k))
        Pb = sp.csr_matrix((np.ones(len(ib)), (np.arange(len(ib)), b.labels[ib])), shape=(len(ib), b.k))
        O = (Pa.T @ Pb).toarray()                       # shared counts
        overlaps.append(O)
        sa, sb = a.sizes(), b.sizes()
        for ci, cj in zip(*np.nonzero(O)):
            shared = O[ci, cj]
            frac = shared / max(sa[ci], 1)
            jac = shared / max(sa[ci] + sb[cj] - shared, 1)
            if frac >= min_overlap:
                edges.append((i, int(ci), i + 1, int(cj), float(frac), float(jac)))
    return Lineage(parts, edges, overlaps)
