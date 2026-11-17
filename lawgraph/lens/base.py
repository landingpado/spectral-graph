"""
Shared layer for all lenses.

    snap = st.at(2000)
    ids  = select(snap, "cluster", where="court = 'scotus'")      # or any SQL on the node table / joins
    G    = build(snap, ("cluster", "cites", "cluster"), ids, mode="cocite", min_degree=2)
    L    = laplacian(G)                                           # normalized by default
    result = SomeLens(...)(G)

A Graph is the one object lenses consume: a symmetric sparse weight matrix W, the node ids in row
order, and what they are. Lenses never touch DuckDB; everything they need is on the Graph.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from ..store import Snapshot

EdgeKey = tuple[str, str, str]


@dataclass
class Graph:
    W: sp.csr_matrix                 # symmetric, non-negative, zero diagonal
    ids: np.ndarray                  # node ids, row order
    node_type: str
    key: EdgeKey                     # relation the graph was built from
    mode: str                        # "sym" | "cocite" | "bibcoup"
    t: str                           # snapshot date literal
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.W.shape[0]

    @property
    def degree(self) -> np.ndarray:
        return np.asarray(self.W.sum(1)).ravel()

    def pos(self, ids) -> np.ndarray:
        """Row positions of the given ids (-1 if absent)."""
        p = np.searchsorted(self.ids, ids)
        ok = (p < self.n)
        ok[ok] &= self.ids[p[ok]] == np.asarray(ids)[ok]
        return np.where(ok, p, -1)

    def subgraph(self, mask: np.ndarray) -> "Graph":
        return Graph(self.W[mask][:, mask].tocsr(), self.ids[mask], self.node_type, self.key, self.mode, self.t,
                     dict(self.meta, parent_n=self.n))


class Lens(Protocol):
    """A lens is any callable from a Graph to a result. Dataclasses with parameters are the convention."""
    def __call__(self, G: Graph): ...


# ---------------------------------------------------------------- selection
def select(snap: Snapshot, node_type: str, where: str | None = None, join: str = "") -> np.ndarray:
    """Ids of `node_type` present in the snapshot satisfying `where`. `join` may add tables;
    the node table is aliased `n`. Example (SCOTUS decisions):
        select(snap, "cluster", where="d.dst_id = 'scotus'",
               join="JOIN edge_cluster_in_docket_docket cd ON cd.src_id = n.id "
                    "JOIN edge_docket_filed_in_court d ON d.src_id = cd.dst_id")
    """
    tbl = snap.schema.nodes[node_type].table
    pred = snap.predicate                      # qualify the snapshot predicate on the node alias
    for c in ("valid_from", "valid_to", "active_from", "active_to"):
        pred = pred.replace(c, f"n.{c}")
    sql = f"SELECT DISTINCT n.id FROM {tbl} n {join} WHERE {pred}" + (f" AND ({where})" if where else "")
    return np.sort(np.asarray(snap.con.execute(sql).fetchnumpy()["id"]))


# ---------------------------------------------------------------- construction
def build(snap: Snapshot, key: EdgeKey, ids: np.ndarray | None = None, mode: str = "sym",
          weight: str | None = None, min_degree: int = 1, giant: bool = True) -> Graph:
    """Symmetric weighted graph on `ids` (default: all nodes of the relation's src type) from one relation.
    mode: "sym"     W = A + A^T           (linked if either cites the other)
          "cocite"  W = A^T A             (linked if cited together by a later node; the usual doctrinal similarity)
          "bibcoup" W = A A^T             (linked if they cite the same things)
    Isolated nodes and nodes with degree < min_degree are dropped; optionally keep only the giant component."""
    src_t, _, dst_t = key
    if src_t != dst_t:
        raise ValueError("build() needs a square relation (same src and dst type); compose rectangular ones first")
    all_ids = snap.index(src_t)
    A = snap.adjacency(key, weight=weight)
    if ids is not None:
        m = np.isin(all_ids, ids)
        A, all_ids = A[m][:, m], all_ids[m]
    W = {"sym": lambda: A + A.T, "cocite": lambda: A.T @ A, "bibcoup": lambda: A @ A.T}[mode]().tocsr()
    W.setdiag(0); W.eliminate_zeros()
    deg = np.asarray(W.sum(1)).ravel()
    keep = deg >= max(min_degree, 1e-12)
    W, all_ids = W[keep][:, keep], all_ids[keep]
    dropped_degree = int((~keep).sum())
    dropped_comp = 0
    if giant and W.shape[0]:
        _, lab = connected_components(W, directed=False)
        big = lab == np.bincount(lab).argmax()
        dropped_comp = int((~big).sum())
        W, all_ids = W[big][:, big], all_ids[big]
    return Graph(W.tocsr(), all_ids, src_t, key, mode, snap.t,
                 {"dropped_low_degree": dropped_degree, "dropped_other_components": dropped_comp})


def laplacian(G: Graph, normalized: bool = True) -> sp.csr_matrix:
    d = G.degree
    if not normalized:
        return (sp.diags(d) - G.W).tocsr()
    Dm12 = sp.diags(1.0 / np.sqrt(np.where(d > 0, d, 1.0)))
    return (sp.eye(G.n) - Dm12 @ G.W @ Dm12).tocsr()
