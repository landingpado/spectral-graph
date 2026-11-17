"""
Typed, temporal, heterogeneous graph over DuckDB -- schema layer.

A Schema declares *topology only*: for each node/edge type, where the rows come
from and which columns play the structural roles. Every other column on the
source passes through as an attribute with its original name and type, so
nothing from the upstream database is re-declared here.

Table layout produced by build():
    node_<type>                 (id, uid, valid_from, valid_to, active_from, active_to, <attrs...>)
    edge_<src>_<rel>_<dst>      (src_id, dst_id, valid_from, valid_to, active_from, active_to, <attrs...>)

id is the source system's key, unique within a type. uid is a deterministic UUID,
md5("<type>:<id>"), unique across all types and stable across rebuilds.

Two half-open intervals per row, each with NULL upper bound = open-ended:
    [valid_from, valid_to)    existence: the record is part of the graph (default: from its
                              event date, forever)
    [active_from, active_to)  activity: the thing is in force (a judge sits, a docket is open,
                              a person is alive). Defaults to the valid interval.
Snapshot at t  <=>  X_from <= t AND (X_to IS NULL OR t < X_to), X = valid | active
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

TIME_TYPE = "DATE"
_IDENT = re.compile(r"^\w+$")


@dataclass(frozen=True)
class NodeSpec:
    name: str
    source: str              # table name or FROM-clause fragment (joins, subqueries, aliases ok)
    id: str                  # column or expression
    valid_from: str          # column or expression
    valid_to: str = "NULL"
    active_from: str | None = None   # default: valid_from
    active_to: str | None = None     # default: valid_to
    select: str = "*"        # attribute columns to pass through ("" for none)
    exclude: tuple[str, ...] = ()
    where: str = ""

    @property
    def table(self) -> str:
        return f"node_{self.name}"

    def sql(self) -> str:
        uid = f"CAST(md5('{self.name}:' || CAST({self.id} AS VARCHAR)) AS UUID)"
        return _ctas(self.table, {"id": self.id, "uid": uid}, self)


@dataclass(frozen=True)
class EdgeSpec:
    src: str
    rel: str
    dst: str
    source: str
    src_id: str
    dst_id: str
    valid_from: str
    valid_to: str = "NULL"
    active_from: str | None = None
    active_to: str | None = None
    select: str = "*"
    exclude: tuple[str, ...] = ()
    where: str = ""

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.src, self.rel, self.dst)

    @property
    def table(self) -> str:
        return f"edge_{self.src}_{self.rel}_{self.dst}"

    def sql(self) -> str:
        return _ctas(self.table, {"src_id": self.src_id, "dst_id": self.dst_id}, self)


def _ctas(table: str, keys: dict[str, str], spec) -> str:
    cols = [f"{expr} AS {name}" for name, expr in keys.items()]
    a_from = spec.active_from if spec.active_from is not None else spec.valid_from
    a_to = spec.active_to if spec.active_to is not None else spec.valid_to
    cols += [f"CAST({spec.valid_from} AS {TIME_TYPE}) AS valid_from",
             f"CAST({spec.valid_to} AS {TIME_TYPE}) AS valid_to",
             f"CAST({a_from} AS {TIME_TYPE}) AS active_from",
             f"CAST({a_to} AS {TIME_TYPE}) AS active_to"]
    if spec.select:
        # drop structural columns from the pass-through when they are plain identifiers
        drop = [e for e in (*keys.values(), *spec.exclude) if _IDENT.match(e)]
        sel = spec.select
        if sel == "*" and drop:
            sel = f"* EXCLUDE ({', '.join(dict.fromkeys(drop))})"
        cols.append(sel)
    where = f"\nWHERE {spec.where}" if spec.where else ""
    return f"CREATE OR REPLACE TABLE {table} AS\nSELECT {', '.join(cols)}\nFROM {spec.source}{where};"


_MISSING_EXCLUDE = re.compile(r'Column "(\w+)" in EXCLUDE list not found')


def _execute_tolerant(con, sql: str) -> None:
    """Run a CTAS; if an EXCLUDE names a column the source lacks, drop it from the SQL and retry.
    Lets one mapping serve replicas whose column sets differ slightly."""
    import warnings
    for _ in range(20):
        try:
            con.execute(sql)
            return
        except Exception as e:  # duckdb.BinderException
            m = _MISSING_EXCLUDE.search(str(e))
            if not m:
                raise
            col = m.group(1)
            warnings.warn(f"source lacks column {col!r}; dropping it from EXCLUDE")
            sql = re.sub(rf"(EXCLUDE \([^)]*?)\b{col}\b,?\s*", r"\1", sql)
            sql = re.sub(r",\s*\)", ")", sql)
    con.execute(sql)


@dataclass
class Schema:
    nodes: dict[str, NodeSpec] = field(default_factory=dict)
    edges: dict[tuple[str, str, str], EdgeSpec] = field(default_factory=dict)

    def node(self, name: str, **kw) -> NodeSpec:
        self.nodes[name] = n = NodeSpec(name, **kw)
        return n

    def edge(self, src: str, rel: str, dst: str, **kw) -> EdgeSpec:
        for t in (src, dst):
            if t not in self.nodes:
                raise ValueError(f"edge ({src},{rel},{dst}): unknown node type {t!r}")
        e = EdgeSpec(src, rel, dst, **kw)
        self.edges[e.key] = e
        return e

    @property
    def specs(self):
        return [*self.nodes.values(), *self.edges.values()]

    # -- build ----------------------------------------------------------------
    def sql(self) -> list[str]:
        return [s.sql() for s in self.specs]

    def build(self, con, only: set[str] | None = None, verbose: bool = True) -> None:
        """Materialise every node/edge table from the attached source. `only` filters by table name."""
        for s in self.specs:
            if only and s.table not in only:
                continue
            if verbose:
                print(f"building {s.table} ...", end=" ", flush=True)
            _execute_tolerant(con, s.sql())
            if verbose:
                print(con.execute(f"SELECT count(*) FROM {s.table}").fetchone()[0], "rows")

    # -- introspection (attributes are discovered, never declared) -------------
    def attrs(self, con, table: str) -> dict[str, str]:
        """{column: duckdb_type} for the attribute columns of a built table."""
        structural = {"id", "uid", "src_id", "dst_id", "valid_from", "valid_to", "active_from", "active_to"}
        return {c: t for c, t, *_ in con.execute(f"DESCRIBE {table}").fetchall() if c not in structural}

    def describe(self, con) -> str:
        out = []
        for s in self.specs:
            a = self.attrs(con, s.table)
            out.append(f"{s.table:<34} {', '.join(f'{k}:{v}' for k, v in a.items())}")
        return "\n".join(out)
