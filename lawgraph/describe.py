"""
Generate a Markdown reference for a built graph file: topology, per-table coverage, attributes.

    python -m lawgraph.describe graph.duckdb > SCHEMA.md
    python -m lawgraph.describe graph.duckdb --no-attrs      # topology + coverage only (fast)

Everything is read from the file itself (DESCRIBE / SUMMARIZE / counts), so the document
always matches the data rather than the code.
"""
from __future__ import annotations

import argparse
import sys

import duckdb

from .cl import SCHEMA

STRUCTURAL = {"id", "uid", "src_id", "dst_id", "valid_from", "valid_to", "active_from", "active_to"}


def coverage(con, table: str) -> dict:
    r = con.execute(f"""
        SELECT count(*), min(valid_from), max(valid_from),
               count(*) FILTER (WHERE valid_from IS NULL),
               count(*) FILTER (WHERE active_to IS NOT NULL)
        FROM {table}""").fetchall()[0]
    n, lo, hi, nodate, closed = r
    return {"rows": n, "first": lo, "last": hi, "undated_pct": 100 * nodate / max(n, 1), "active_closed": closed}


def attrs(con, table: str) -> list[tuple[str, str, float]]:
    """(column, type, null %) for non-structural columns, via SUMMARIZE."""
    out = []
    for row in con.execute(f"SUMMARIZE {table}").fetchall():
        col, typ, null_pct = row[0], row[1], row[-1]
        if col not in STRUCTURAL:
            out.append((col, typ, float(str(null_pct).rstrip("%") or 0)))
    return out


def render(con, with_attrs: bool = True) -> str:
    L = []
    L.append("# lawgraph schema reference\n")
    L.append("Generated from the DuckDB file. Every table has `valid_from/valid_to` (the record exists) and "
             "`active_from/active_to` (the thing is in force); half-open, NULL upper bound = open-ended.\n")

    L.append("## Topology\n")
    L.append("| relation | src → dst | rows | first | last | undated % |")
    L.append("|---|---|---:|---|---|---:|")
    for e in SCHEMA.edges.values():
        c = coverage(con, e.table)
        L.append(f"| `{e.rel}` | {e.src} → {e.dst} | {c['rows']:,} | {c['first']} | {c['last']} | {c['undated_pct']:.1f} |")

    L.append("\n## Node types\n")
    L.append("| type | rows | first | last | undated % | with active_to |")
    L.append("|---|---:|---|---|---:|---:|")
    for n in SCHEMA.nodes.values():
        c = coverage(con, n.table)
        L.append(f"| `{n.name}` | {c['rows']:,} | {c['first']} | {c['last']} | {c['undated_pct']:.1f} | {c['active_closed']:,} |")

    if with_attrs:
        L.append("\n## Attributes (inherited from CourtListener)\n")
        for spec in SCHEMA.specs:
            L.append(f"### `{spec.table}`\n")
            L.append(f"Source: `{spec.source}`  \nInterval: valid_from = `{spec.valid_from}`, active_to = `{spec.active_to or spec.valid_to}`\n")
            a = attrs(con, spec.table)
            if not a:
                L.append("_no attributes_\n"); continue
            L.append("| column | type | null % |")
            L.append("|---|---|---:|")
            for col, typ, nullp in a:
                L.append(f"| `{col}` | {typ} | {nullp:.1f} |")
            L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="lawgraph.describe", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("db")
    ap.add_argument("--no-attrs", action="store_true", help="skip per-column SUMMARIZE (much faster on 70M-row tables)")
    a = ap.parse_args(argv)
    con = duckdb.connect(a.db, read_only=True)
    print(render(con, with_attrs=not a.no_attrs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
