"""
Build (or rebuild) the graph store from a CourtListener postgres replica.

Credentials are never stored in code. Either:
    export CL_DSN="dbname=courtlistener host=db.example.edu user=me password=..."
or use the standard libpq variables (PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD,
or a ~/.pgpass file) and leave CL_DSN unset.

    python -m lawgraph.build lawgraph.duckdb                       # mirror CL tables into raw.*, then build all
    python -m lawgraph.build lawgraph.duckdb --remirror            # re-pull raw.* from postgres first
    python -m lawgraph.build lawgraph.duckdb --only node_court node_person edge_person_sits_on_court
    python -m lawgraph.build lawgraph.duckdb --dry-run             # print SQL, touch nothing
    python -m lawgraph.build /tmp/test.duckdb --mock               # fake data, no postgres
    python -m lawgraph.build lawgraph.duckdb --describe            # attributes of built tables
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from .store import Store


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lawgraph.build", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("db", help="path of the DuckDB file to create/update")
    ap.add_argument("--dsn", default=os.environ.get("CL_DSN", ""),
                    help="libpq connection string (default: $CL_DSN, else PG* env vars)")
    ap.add_argument("--only", nargs="+", metavar="TABLE", help="build only these tables")
    ap.add_argument("--dry-run", action="store_true", help="print the SQL and exit")
    ap.add_argument("--describe", action="store_true", help="list attributes of existing tables and exit")
    ap.add_argument("--mock", action="store_true", help="build from bundled fake data instead of postgres")
    ap.add_argument("--remirror", action="store_true", help="re-pull all raw.* tables from postgres first")
    ap.add_argument("--threads", type=int, default=None, help="DuckDB threads (default: all cores)")
    ap.add_argument("--memory", default=None, help="DuckDB memory limit, e.g. 200GB")
    a = ap.parse_args(argv)

    st = Store(a.db)
    if a.threads:
        st.con.execute(f"SET threads = {a.threads}")
    if a.memory:
        st.con.execute(f"SET memory_limit = '{a.memory}'")

    if a.dry_run:
        print("\n\n".join(st.schema.sql()))
        return 0
    if a.describe:
        print(st.schema.describe(st.con))
        return 0

    only = set(a.only) if a.only else None
    if only:
        known = {s.table for s in st.schema.specs}
        bad = only - known
        if bad:
            sys.exit(f"unknown table(s): {sorted(bad)}\nknown: {sorted(known)}")

    t0 = time.time()
    if a.mock:
        from .mock import make
        from .cl import mirror
        make(st.con)
        mirror(st.con, verbose=False)
        st.schema.build(st.con, only=only)
    else:
        st.load(a.dsn, only=only, remirror=a.remirror)
    print(f"\ndone in {time.time() - t0:.0f}s -> {a.db}")
    print(st.schema.describe(st.con))
    return 0


if __name__ == "__main__":
    sys.exit(main())
