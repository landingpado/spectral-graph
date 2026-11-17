"""
Frame: a fixed coordinate system over ALL nodes (every type, every time), so that
snapshots taken at different times are expressed in the same (N x N) grid.

    f = Frame(store)                              # rows = all nodes ever, stacked by type
    f.pos("opinion", [1000, 2000])                # -> global positions, constant for all t
    f.lookup([17, 42])                            # -> [(type, id), ...]
    A_1990 = f.adjacency(store.at(1990), ("opinion", "cites", "opinion"))   # (N, N) CSR
    B_1990 = f.block(store.at(1990))              # all relations summed into one (N, N)
    stack  = f.stack(range(1900, 2000, 10), ("opinion", "cites", "opinion"))  # list of (N, N) CSR
    t, i, j, w = f.coo(range(1900, 2000, 10), ("opinion", "cites", "opinion"))  # 3-D sparse (T, N, N)

A node absent at t is an empty row/column (free in sparse form). A pair (i, j) is the
same cell in every slice, so the history of one edge is stack[k][i, j] over k.
"""
from __future__ import annotations

import numpy as np

from .store import Snapshot, Store

EdgeKey = tuple[str, str, str]


class Frame:
    def __init__(self, store: Store, types: list[str] | None = None, at=None):
        """types: which node types to include and in what order (default: schema order).
        at: build the frame from the snapshot at this time instead of from all rows ever."""
        self.store, self.schema = store, store.schema
        self.types = list(types or self.schema.nodes)
        self.ids: dict[str, np.ndarray] = {}
        self.offsets: dict[str, int] = {}
        n = 0
        for t in self.types:
            tbl = self.schema.nodes[t].table
            where = f"WHERE {store.at(at).predicate}" if at is not None else ""
            ids = store.con.execute(f"SELECT DISTINCT id FROM {tbl} {where} ORDER BY id").fetchnumpy()["id"]
            self.ids[t] = np.asarray(ids)
            self.offsets[t] = n
            n += len(ids)
        self.n = n

    # -- identity ---------------------------------------------------------------
    def pos(self, node_type: str, ids) -> np.ndarray:
        """Global positions of CL ids of one type. -1 where the id is not in the frame."""
        ref = self.ids[node_type]
        ids = np.asarray(ids)
        p = np.searchsorted(ref, ids)
        ok = (p < len(ref))
        ok[ok] &= ref[p[ok]] == ids[ok]
        out = np.where(ok, p + self.offsets[node_type], -1)
        return out

    def lookup(self, positions) -> list[tuple[str, object]]:
        """Inverse of pos(): global position -> (type, id)."""
        bounds = [(self.offsets[t], self.offsets[t] + len(self.ids[t]), t) for t in self.types]
        out = []
        for g in np.atleast_1d(positions):
            for lo, hi, t in bounds:
                if lo <= g < hi:
                    out.append((t, self.ids[t][g - lo].item()))
                    break
            else:
                raise IndexError(g)
        return out

    def present(self, snap: Snapshot) -> np.ndarray:
        """Boolean mask of length n: which frame rows exist in this snapshot."""
        m = np.zeros(self.n, dtype=bool)
        for t in self.types:
            p = self.pos(t, snap.index(t))
            m[p[p >= 0]] = True
        return m

    # -- matrices in frame coordinates ------------------------------------------
    def adjacency(self, snap: Snapshot, key: EdgeKey, weight: str | None = None, dtype=np.float64):
        """(n, n) CSR for one relation at this snapshot. Edges are kept only if both endpoints
        are in the snapshot (same rule as Snapshot.adjacency), then mapped to frame positions."""
        import scipy.sparse as sp
        src_t, _, dst_t = key
        tbl = self.schema.edges[key].table
        w = weight if weight else "1"
        e = snap.con.execute(f"SELECT src_id, dst_id, {w} AS w FROM {tbl} WHERE {snap.edge_predicate}").fetchnumpy()
        si, di = snap.index(src_t), snap.index(dst_t)
        ok = np.isin(e["src_id"], si) & np.isin(e["dst_id"], di)
        r = self.pos(src_t, e["src_id"][ok])
        c = self.pos(dst_t, e["dst_id"][ok])
        return sp.csr_matrix((np.asarray(e["w"][ok], dtype=dtype), (r, c)), shape=(self.n, self.n))

    def block(self, snap: Snapshot, keys: list[EdgeKey] | None = None,
              weights: dict[EdgeKey, float] | None = None):
        """All relations (or `keys`) in one (n, n) matrix, each scaled by weights[key] (default 1)."""
        keys = keys or [k for k in self.schema.edges if k[0] in self.types and k[2] in self.types]
        M = None
        for k in keys:
            A = self.adjacency(snap, k) * (weights or {}).get(k, 1.0)
            M = A if M is None else M + A
        return M

    # -- time-stratified objects -------------------------------------------------
    def stack(self, times, key: EdgeKey, weight: str | None = None, active: bool = False):
        """One (n, n) CSR per time, all in the same coordinates."""
        return [self.adjacency(self.store.at(t, active=active), key, weight) for t in times]

    def coo(self, times, key: EdgeKey, weight: str | None = None, active: bool = False):
        """3-D sparse coordinates (t_index, row, col, value) over the stack -- feed to
        sparse.COO((t,i,j), w, shape=(T, n, n)) or group by (i, j) to get each pair's history."""
        ts, rs, cs, ws = [], [], [], []
        for k, A in enumerate(self.stack(times, key, weight, active)):
            A = A.tocoo()
            ts.append(np.full(A.nnz, k)); rs.append(A.row); cs.append(A.col); ws.append(A.data)
        cat = lambda xs: np.concatenate(xs) if xs else np.array([], dtype=int)
        return cat(ts), cat(rs), cat(cs), cat(ws)

    def __repr__(self) -> str:
        return "Frame(" + ", ".join(f"{t}={len(self.ids[t])}@{self.offsets[t]}" for t in self.types) + f"; n={self.n})"


def timeline(start, end, step: str = "1Y") -> list:
    """Dates from start to end (inclusive) at a fixed step: '1D', '1W', '1M', '6M', '1Y', ...
    Accepts dates, 'YYYY-MM-DD' strings, or years. Use for Frame.stack / Frame.coo."""
    import datetime as dt
    import re
    conv = lambda x: dt.date(x, 1, 1) if isinstance(x, int) else dt.date.fromisoformat(x) if isinstance(x, str) else x
    start, end = conv(start), conv(end)
    n, unit = re.fullmatch(r"(\d+)([DWMY])", step.upper()).groups()
    n = int(n)
    out, k = [], 0
    while True:
        if unit == "D":
            d = start + dt.timedelta(days=n * k)
        elif unit == "W":
            d = start + dt.timedelta(weeks=n * k)
        else:
            months = n * k * (12 if unit == "Y" else 1)
            y, m = divmod(start.month - 1 + months, 12)
            d = start.replace(year=start.year + y, month=m + 1)
        if d > end:
            return out
        out.append(d)
        k += 1
