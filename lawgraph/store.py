"""
Store: the DuckDB-backed graph object.  Snapshot: one time slice of it.

    store = Store("lawgraph.duckdb")                 # opens/creates the file
    store.load(dsn)                                  # one-off: attach postgres + build all tables
    snap = store.at("1990-01-01")                    # everything on record as of that date
    snap = store.at("1990-01-01", active=True)       # only what was in force: sitting judges, open dockets...
    snap.nodes("opinion")                            # -> DuckDB relation (lazy; .df() / .arrow() / .fetchall())
    snap.edges("opinion", "cites", "opinion")
    A   = snap.adjacency(("opinion", "cites", "opinion"))        # scipy.sparse CSR, rows=src, cols=dst
    X   = snap.features("opinion", ["citation_count"])           # ndarray aligned with snap.index("opinion")
    snap.views("s1990")                              # optional: schema s1990 with a view per table, for raw SQL

Nothing is materialised by a snapshot; it is a predicate applied lazily to the
base tables, so any number of snapshots can coexist. Lenses (next piece) only
ever see a Snapshot.
"""
from __future__ import annotations

import datetime as dt

import duckdb
import numpy as np

from .schema import Schema

EdgeKey = tuple[str, str, str]


def _date(x) -> str:
    """Validate and render a date literal; accepts date / 'YYYY-MM-DD' / year int."""
    if isinstance(x, int):
        x = dt.date(x, 1, 1)
    elif isinstance(x, str):
        x = dt.date.fromisoformat(x)
    return f"DATE '{x.isoformat()}'"


class Store:
    def __init__(self, path: str = ":memory:", schema: Schema | None = None, read_only: bool = False):
        if schema is None:
            from .cl import SCHEMA
            schema = SCHEMA
        self.schema = schema
        self.con = duckdb.connect(path, read_only=read_only)

    # -- loading --------------------------------------------------------------
    def load(self, dsn: str = "", only: set[str] | None = None, verbose: bool = True,
             remirror: bool = False) -> None:
        """Mirror CL tables into raw.* (only those missing, or all if remirror), then build graph
        tables from the mirror. dsn="" uses libpq PG* env vars. Needs postgres only for mirroring."""
        from .cl import MIRROR, attach, mirror, mirrored
        missing = set(MIRROR) - mirrored(self.con)
        if remirror or missing:
            attach(self.con, dsn)
            mirror(self.con, only=None if remirror else missing, verbose=verbose)
        self.schema.build(self.con, only=only, verbose=verbose)

    @classmethod
    def mock(cls) -> "Store":
        """In-memory store built from the bundled fake CL data; for tests and lens development."""
        from .cl import mirror
        from .mock import make
        st = cls()
        make(st.con)
        mirror(st.con, verbose=False)
        st.schema.build(st.con, verbose=False)
        return st

    # -- introspection --------------------------------------------------------
    def tables(self) -> dict[str, int]:
        return {s.table: self.con.execute(f"SELECT count(*) FROM {s.table}").fetchone()[0]
                for s in self.schema.specs}

    def attrs(self, table: str) -> dict[str, str]:
        return self.schema.attrs(self.con, table)

    def time_range(self) -> tuple[dt.date, dt.date]:
        q = " UNION ALL ".join(f"SELECT min(valid_from) lo, max(valid_from) hi FROM {s.table}"
                               for s in self.schema.specs)
        return self.con.execute(f"SELECT min(lo), max(hi) FROM ({q})").fetchone()

    # -- time slicing ---------------------------------------------------------
    def at(self, t, until=None, active: bool = False) -> "Snapshot":
        """Point snapshot at t, or (with `until`) anything overlapping [t, until).
        active=False uses the existence interval; active=True the activity interval."""
        return Snapshot(self, t, until, active)

    def window(self, t, until) -> "Snapshot":
        """Event window: nodes that exist by `until`, edges FORMED in [t, until) -- e.g. citations made by
        decisions filed in the window, whatever the age of the cited decision. (at(t, until) instead
        returns everything *in force* at any time in the window, which for never-expiring records is cumulative.)"""
        return Snapshot(self, t, until, False, events=True)

    def sql(self, query: str):
        return self.con.sql(query)


class Snapshot:
    def __init__(self, store: Store, t, until=None, active: bool = False, events: bool = False):
        self.store, self.schema, self.con = store, store.schema, store.con
        self.t, self.until = _date(t), (_date(until) if until is not None else None)
        self.active, self.events = active, events
        self._idx: dict[str, np.ndarray] = {}

    # -- the one predicate everything else is built from ----------------------
    @property
    def predicate(self) -> str:
        """Row filter for NODE tables (and for edges unless this is an event window)."""
        lo, hi = ("active_from", "active_to") if self.active else ("valid_from", "valid_to")
        if self.until is None:
            return f"{lo} <= {self.t} AND ({hi} IS NULL OR {hi} > {self.t})"
        if self.events:                                  # nodes: exist by the end of the window
            return f"{lo} < {self.until}"
        return f"{lo} < {self.until} AND ({hi} IS NULL OR {hi} > {self.t})"

    @property
    def edge_predicate(self) -> str:
        """Row filter for EDGE tables: formed within the window if events=True, else same as nodes."""
        if self.events:
            return f"valid_from >= {self.t} AND valid_from < {self.until}"
        return self.predicate

    def sql(self, table: str) -> str:
        pred = self.edge_predicate if table.split(".")[-1].startswith("edge_") else self.predicate
        return f"SELECT * FROM {table} WHERE {pred}"

    def nodes(self, node_type: str):
        return self.con.sql(self.sql(self.schema.nodes[node_type].table))

    def edges(self, src: str, rel: str, dst: str):
        return self.con.sql(self.sql(self.schema.edges[(src, rel, dst)].table))

    def views(self, name: str) -> None:
        """Create schema `name` holding a view per table, for ad-hoc SQL against this slice."""
        self.con.execute(f"CREATE SCHEMA IF NOT EXISTS {name}")
        for s in self.schema.specs:
            self.con.execute(f"CREATE OR REPLACE VIEW {name}.{s.table} AS {self.sql('main.' + s.table)}")

    # -- tensor adapters ------------------------------------------------------
    def index(self, node_type: str) -> np.ndarray:
        """Sorted array of node ids present in this slice; position = dense row index. Cached."""
        if node_type not in self._idx:
            tbl = self.schema.nodes[node_type].table
            self._idx[node_type] = np.asarray(self.con.execute(
                f"SELECT DISTINCT id FROM {tbl} WHERE {self.predicate} ORDER BY id").fetchnumpy()["id"])
        return self._idx[node_type]

    def uids(self, node_type: str) -> np.ndarray:
        """Stable UUIDs (as strings) aligned with index(node_type)."""
        tbl = self.schema.nodes[node_type].table
        r = self.con.execute(f"SELECT id, CAST(uid AS VARCHAR) AS uid FROM {tbl} WHERE {self.predicate} "
                             f"QUALIFY row_number() OVER (PARTITION BY id ORDER BY valid_from DESC) = 1 ORDER BY id").fetchnumpy()
        return np.asarray(r["uid"], dtype=object)

    def global_index(self, types: list[str] | None = None) -> dict[str, int]:
        """Offsets for stacking node types into one coordinate system: global row of
        (type, position) = offsets[type] + position. Order = `types` or schema order."""
        offs, n = {}, 0
        for t in types or list(self.schema.nodes):
            offs[t] = n
            n += len(self.index(t))
        offs["__total__"] = n
        return offs

    def features(self, node_type: str, cols: list[str]) -> np.ndarray:
        """Attribute matrix aligned with index(node_type). Versioned nodes: latest valid_from wins."""
        tbl = self.schema.nodes[node_type].table
        sel = ", ".join(cols)
        rows = self.con.execute(
            f"SELECT id, {sel} FROM {tbl} WHERE {self.predicate} "
            f"QUALIFY row_number() OVER (PARTITION BY id ORDER BY valid_from DESC) = 1 ORDER BY id"
        ).fetchnumpy()
        ids = self.index(node_type)
        out = np.full((len(ids), len(cols)), np.nan)
        pos = np.searchsorted(ids, rows["id"])
        for j, c in enumerate(cols):
            out[pos, j] = np.asarray(rows[c], dtype=float)
        return out

    def adjacency(self, key: EdgeKey, weight: str | None = None, dtype=np.float64):
        """scipy.sparse CSR of shape (n_src, n_dst); multi-edges are summed. Dangling edges dropped."""
        import scipy.sparse as sp
        src_t, _, dst_t = key
        tbl = self.schema.edges[key].table
        w = weight if weight else "1"
        e = self.con.execute(f"SELECT src_id, dst_id, {w} AS w FROM {tbl} WHERE {self.edge_predicate}").fetchnumpy()
        si, di = self.index(src_t), self.index(dst_t)
        r, c = np.searchsorted(si, e["src_id"]), np.searchsorted(di, e["dst_id"])
        ok = (r < len(si)) & (c < len(di))
        ok[ok] &= (si[r[ok]] == e["src_id"][ok]) & (di[c[ok]] == e["dst_id"][ok])
        return sp.csr_matrix((np.asarray(e["w"][ok], dtype=dtype), (r[ok], c[ok])), shape=(len(si), len(di)))

    def adjacencies(self, weight: str | None = None) -> dict[EdgeKey, "sp.csr_matrix"]:
        """Every relation at once -- the 'tensor' view of the slice."""
        return {k: self.adjacency(k, weight) for k in self.schema.edges}

    def __repr__(self) -> str:
        span = f"[{self.t}, {self.until})" if self.until else f"at {self.t}"
        span += " active" if self.active else ""
        span += " events" if self.events else ""
        counts = ", ".join(f"{n}={len(self.index(n))}" for n in self.schema.nodes)
        return f"Snapshot({span}: {counts})"
