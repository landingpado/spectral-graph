"""
Random spot checks against CourtListener, for hand verification. Prints plain text with CL links.

    python -m lawgraph.spotcheck graph.duckdb                 # 5 of each kind
    python -m lawgraph.spotcheck graph.duckdb -n 10 --seed 7  # reproducible sample
    python -m lawgraph.spotcheck graph.duckdb --only cites authors   # subset: cites authors bench dockets succeeds

Each block says what to verify on the CL page.
"""
from __future__ import annotations

import argparse
import sys

import duckdb

CL = "https://www.courtlistener.com"
OP = lambda cid: f"{CL}/opinion/{cid}/x/"
DK = lambda did: f"{CL}/docket/{did}/x/"
PS = lambda pid: f"{CL}/person/{pid}/x/"


def rows(con, sql, params=None):
    cur = con.execute(sql, params or [])
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def head(title, check):
    print("\n" + "=" * 100); print(title); print("verify:", check); print("=" * 100)


def cites(con, n):
    head("CITATIONS  (cluster -> cluster)",
         "open the CITING case; its 'Authorities' tab should list the CITED case. Dates: citing >= cited.")
    for r in rows(con, f"""
        SELECT e.src_id, e.dst_id, e.depth, e.opinion_pairs,
               a.case_name AS citing_name, a.date_filed AS citing_date,
               b.case_name AS cited_name,  b.date_filed AS cited_date
        FROM edge_cluster_cites_cluster e
        JOIN node_cluster a ON a.id = e.src_id JOIN node_cluster b ON b.id = e.dst_id
        WHERE a.case_name <> '' AND b.case_name <> ''
        USING SAMPLE {n} ROWS"""):
        print(f"\n  CITING  {r['citing_name'][:70]}  ({r['citing_date']})\n          {OP(r['src_id'])}")
        print(f"  CITED   {r['cited_name'][:70]}  ({r['cited_date']})\n          {OP(r['dst_id'])}")
        print(f"  depth={r['depth']}  opinion_pairs={r['opinion_pairs']}")


def authors(con, n):
    head("AUTHORSHIP  (person -> cluster)",
         "open the opinion; the author line should name this judge. Open the judge; court/tenure should match the decision's court and date.")
    for r in rows(con, f"""
        SELECT e.src_id AS person_id, e.dst_id AS cluster_id, c.case_name, c.date_filed,
               p.name_first, p.name_middle, p.name_last, d.dst_id AS court
        FROM edge_person_author_cluster e
        JOIN node_cluster c ON c.id = e.dst_id
        JOIN node_person p ON p.id = e.src_id
        LEFT JOIN edge_cluster_in_docket_docket cd ON cd.src_id = c.id
        LEFT JOIN edge_docket_filed_in_court d ON d.src_id = cd.dst_id
        WHERE c.case_name <> ''
        USING SAMPLE {n} ROWS"""):
        who = " ".join(x for x in (r["name_first"], r["name_middle"], r["name_last"]) if x)
        print(f"\n  {r['case_name'][:70]}  ({r['date_filed']}, {r['court']})\n          {OP(r['cluster_id'])}")
        print(f"  author  {who}\n          {PS(r['person_id'])}")


def bench(con, n):
    head("BENCH  (person -> court, active interval)",
         "open the judge; the position for this court should show these start/end dates (end = termination or senior status).")
    for r in rows(con, f"""
        SELECT e.src_id AS person_id, e.dst_id AS court, p.name_first, p.name_last,
               e.active_from, e.active_to, e.position_type, e.termination_reason, e.date_retirement, e.date_termination
        FROM edge_person_sits_on_court e JOIN node_person p ON p.id = e.src_id
        USING SAMPLE {n} ROWS"""):
        end = r["active_to"] or "still active"
        print(f"\n  {r['name_first']} {r['name_last']}  on {r['court']}  {r['active_from']} -> {end}"
              f"   [{r['position_type']}, term={r['termination_reason'] or '-'}, retire={r['date_retirement']}, termination={r['date_termination']}]")
        print(f"          {PS(r['person_id'])}")


def dockets(con, n):
    head("DOCKETS  (judge, parties, attorneys on a PACER docket)",
         "open the docket; the assigned judge, party list and counsel should match.")
    for r in rows(con, f"""
        SELECT d.id, d.court_id, d.case_name, d.docket_number, d.date_filed, d.date_terminated
        FROM node_docket d
        WHERE d.pacer_case_id IS NOT NULL AND d.docket_number LIKE '%:%-cv-%' AND d.date_filed >= DATE '2010-01-01'
          AND EXISTS (SELECT 1 FROM edge_party_party_in_docket p WHERE p.dst_id = d.id)
        USING SAMPLE {n} ROWS"""):
        did = r["id"]
        print(f"\n  {r['case_name'][:70]}  {r['docket_number']}  ({r['court_id']}, filed {r['date_filed']}, terminated {r['date_terminated']})"
              f"\n          {DK(did)}")
        for j in rows(con, """SELECT p.name_first, p.name_last, e.role FROM edge_person_presides_over_docket e
                              JOIN node_person p ON p.id = e.src_id WHERE e.dst_id = ?""", [did]):
            print(f"  judge     {j['name_first']} {j['name_last']} ({j['role']})")
        for p in rows(con, """SELECT pt.name, e.party_type FROM edge_party_party_in_docket e
                              JOIN node_party pt ON pt.id = e.src_id WHERE e.dst_id = ? ORDER BY e.party_type LIMIT 8""", [did]):
            print(f"  party     {p['name'][:50]:<50} {p['party_type']}")
        for a in rows(con, """SELECT DISTINCT a.name FROM edge_attorney_attorney_in_docket e
                              JOIN node_attorney a ON a.id = e.src_id WHERE e.dst_id = ? LIMIT 6""", [did]):
            print(f"  attorney  {a['name'][:50]}")


def succeeds(con, n):
    head("SEAT LINEAGE  (person -> predecessor)",
         "open the successor; their position should list the predecessor.")
    for r in rows(con, f"""
        SELECT e.src_id, e.dst_id, e.court_id, e.date_start, a.name_last AS successor, b.name_last AS predecessor
        FROM edge_person_succeeds_person e JOIN node_person a ON a.id = e.src_id JOIN node_person b ON b.id = e.dst_id
        USING SAMPLE {n} ROWS"""):
        print(f"\n  {r['successor']} succeeded {r['predecessor']} on {r['court_id']} ({r['date_start']})")
        print(f"          {PS(r['src_id'])}   <-   {PS(r['dst_id'])}")


CHECKS = {"cites": cites, "authors": authors, "bench": bench, "dockets": dockets, "succeeds": succeeds}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="lawgraph.spotcheck", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("db")
    ap.add_argument("-n", type=int, default=5, help="rows per check")
    ap.add_argument("--seed", type=int, default=None, help="fix the random sample")
    ap.add_argument("--only", nargs="+", choices=list(CHECKS), default=list(CHECKS))
    a = ap.parse_args(argv)
    con = duckdb.connect(a.db, read_only=True)
    if a.seed is not None:
        con.execute(f"SELECT setseed({(a.seed % 1000) / 1000.0})")
    for name in a.only:
        CHECKS[name](con, a.n)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
