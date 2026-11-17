"""
Upgrade a graph file built by an earlier version of lawgraph, in place, without postgres.

    python -m lawgraph.migrate graph.duckdb

Adds, where missing:
  * active_from / active_to on every table (activity interval), derived from local data
  * uid on every node table
  * edge_person_succeeds_person, derived from edge_person_sits_on_court
and moves any old lifetime-style valid_to into active_to (records never leave).
Safe to run repeatedly.
"""
from __future__ import annotations

import sys

import duckdb

from .cl import SCHEMA

# active_to expressions, evaluated against the local tables (t = the table being migrated)
ACTIVE_TO = {
    "node_court":   "t.end_date",
    "node_docket":  "t.date_terminated",
    "node_person":  "t.date_dod",
    "node_attorney": """(SELECT CASE WHEN bool_or(d.date_terminated IS NULL) THEN NULL ELSE max(d.date_terminated) END
                        FROM edge_attorney_attorney_in_docket e JOIN node_docket d ON d.id = e.dst_id WHERE e.src_id = t.id)""",
    "node_party":   """(SELECT CASE WHEN bool_or(d.date_terminated IS NULL) THEN NULL ELSE max(d.date_terminated) END
                        FROM edge_party_party_in_docket e JOIN node_docket d ON d.id = e.dst_id WHERE e.src_id = t.id)""",
    "edge_person_presides_over_docket": "(SELECT d.date_terminated FROM node_docket d WHERE d.id = t.dst_id)",
    "edge_attorney_attorney_in_docket": "(SELECT d.date_terminated FROM node_docket d WHERE d.id = t.dst_id)",
    "edge_attorney_represents_party":   "(SELECT d.date_terminated FROM node_docket d WHERE d.id = t.docket_id)",
    "edge_party_party_in_docket":       "coalesce(t.date_terminated, (SELECT d.date_terminated FROM node_docket d WHERE d.id = t.dst_id))",
    "edge_person_sits_on_court":        "t.date_termination",
}


def columns(con, table) -> set[str]:
    return {r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()}


def migrate(con: duckdb.DuckDBPyConnection, verbose=True) -> None:
    log = print if verbose else (lambda *a: None)
    existing = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}

    for spec in SCHEMA.specs:
        tbl = spec.table
        if tbl not in existing:
            continue
        cols = columns(con, tbl)

        if "active_from" not in cols:
            log(f"{tbl}: adding activity interval")
            con.execute(f"ALTER TABLE {tbl} ADD COLUMN active_from DATE")
            con.execute(f"ALTER TABLE {tbl} ADD COLUMN active_to DATE")
            # old lifetime-style valid_to (if any) becomes active_to; records never leave
            con.execute(f"UPDATE {tbl} SET active_from = valid_from, active_to = valid_to")
            con.execute(f"UPDATE {tbl} SET valid_to = NULL")
            if tbl in ACTIVE_TO:
                con.execute(f"UPDATE {tbl} t SET active_to = {ACTIVE_TO[tbl]}")

        if tbl.startswith("node_") and "uid" not in cols:
            log(f"{tbl}: adding uid")
            con.execute(f"ALTER TABLE {tbl} ADD COLUMN uid UUID")
            con.execute(f"UPDATE {tbl} SET uid = CAST(md5('{spec.name}:' || CAST(id AS VARCHAR)) AS UUID)")

    if "edge_person_succeeds_person" not in existing and "edge_person_sits_on_court" in existing \
            and "predecessor_id" in columns(con, "edge_person_sits_on_court"):
        log("edge_person_succeeds_person: deriving from sits_on")
        con.execute("""
            CREATE TABLE edge_person_succeeds_person AS
            SELECT src_id, predecessor_id AS dst_id,
                   CAST(date_start AS DATE) AS valid_from, CAST(NULL AS DATE) AS valid_to,
                   CAST(date_start AS DATE) AS active_from, CAST(NULL AS DATE) AS active_to,
                   id AS position_id, dst_id AS court_id, position_type, date_start, date_termination
            FROM edge_person_sits_on_court WHERE predecessor_id IS NOT NULL""")
    log("migration complete")


if __name__ == "__main__":
    con = duckdb.connect(sys.argv[1])
    migrate(con)
    print(SCHEMA.describe(con))
