"""
CourtListener mapping: which CL tables feed which node/edge types.

Attributes are NOT declared -- every non-structural column on the source
passes through with CL's own name and type. Only topology and the temporal
interval of each type are decided here.

Two intervals per row:
  valid  = existence: the record enters the graph at its event date and never leaves.
  active = in force:  court [start_date, end_date); docket [date_filed, date_terminated);
           person [date_dob, date_dod); attorney/party [first docket, last docket closed);
           opinion/cites/author/in_docket/filed_in: same as valid (never lapse);
           sits_on [date_start, date_termination); presides_over / party_in / attorney_in /
           represents: the docket's open interval.
  Node entry dates: court start_date; docket date_filed; opinion cluster.date_filed;
  person date_dob; attorney/party first appearance on any docket.
  presides_over has no reassignment history in CL (only the current assigned_to_id).

Usage: see build.py (CLI) or Store.load(dsn) in store.py.
"""
import duckdb

from .schema import Schema

PG = "cl.public"     # attached postgres catalog (read only during mirror)
CL = "raw"           # local verbatim copies; every graph table is built from these
EPOCH = "1600-01-01"                               # dates before this are CL typos (year 19, 217, ...)
NOISE = ("date_created", "date_modified")          # CL record-time columns; not event time
# full-text columns on search_opinion: tens of GB, not graph data. Trim this tuple if a name
# does not exist on your replica (DuckDB errors on EXCLUDE of an unknown column).
# position_type codes that are not active judicial service (CL get_choices: position_type)
NON_JUDICIAL = ", ".join(f"'{c}'" for c in (
    "clerk", "clerk-chief-dep", "staff-atty",                                   # clerkships
    "ret-act-jus", "ret-ass-jud", "ret-c-jud", "ret-jus", "ret-senior-jud",     # retired / senior rows
    "att-gen", "att-gen-ass", "att-gen-ass-spec", "sen-counsel", "dep-sol-gen",
    "pres", "gov", "mayor", "prof", "adj-prof", "prac", "pros", "pub-def", "da", "ada",
    "legis", "sen", "state-sen"))
OPINION_TEXT = ("plain_text", "html", "html_lawbox", "html_columbia", "html_anon_2020",
                "xml_harvard", "html_with_citations")

# CL tables mirrored verbatim into raw.* (name -> columns to leave behind). Mirroring is a pure
# projection through the postgres scanner; all joins/expressions then run locally in DuckDB.
MIRROR = {
    "search_court": (), "search_docket": (), "search_opinioncluster": (),
    "search_opinion": OPINION_TEXT, "search_opinion_joined_by": (), "search_opinionscited": (),
    "people_db_person": (), "people_db_attorney": (), "people_db_party": (),
    "people_db_role": (), "people_db_partytype": (), "people_db_position": (),
}


def attach(con: duckdb.DuckDBPyConnection, dsn: str) -> None:
    con.execute("INSTALL postgres; LOAD postgres;")
    con.execute(f"ATTACH IF NOT EXISTS '{dsn}' AS cl (TYPE postgres, READ_ONLY)")


def mirror(con: duckdb.DuckDBPyConnection, only: set[str] | None = None, verbose: bool = True) -> None:
    """Copy CL tables into raw.* unchanged (minus NOISE / text columns). Requires attach()."""
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {CL}")
    for name, drop in MIRROR.items():
        if only and name not in only:
            continue
        if verbose:
            print(f"mirroring {name} ...", end=" ", flush=True)
        have = {r[0] for r in con.execute(f"DESCRIBE {PG}.{name}").fetchall()}
        excl = [c for c in dict.fromkeys((*NOISE, *drop)) if c in have]
        sel = f"* EXCLUDE ({', '.join(excl)})" if excl else "*"
        con.execute(f"CREATE OR REPLACE TABLE {CL}.{name} AS SELECT {sel} FROM {PG}.{name}")
        if verbose:
            print(con.execute(f"SELECT count(*) FROM {CL}.{name}").fetchone()[0], "rows")


def mirrored(con) -> set[str]:
    return {r[0] for r in con.execute(f"SELECT table_name FROM information_schema.tables WHERE table_schema='{CL}'").fetchall()}


def courtlistener_schema() -> Schema:
    s = Schema()
    T = lambda name: f"{CL}.{name}"

    # ---- nodes ----------------------------------------------------------------
    # Most courts (87%) and most judges (55%) have no start_date / date_dob in CL. A node with no
    # valid_from is invisible to every snapshot, so entry dates fall back to the earliest evidence.
    s.node("court",
           source=f"{T('search_court')} c"
                  f" LEFT JOIN (SELECT court_id, min(date_filed) AS first_seen FROM {T('search_docket')}"
                  f"            WHERE date_filed > DATE '{EPOCH}' GROUP BY 1) d ON d.court_id = c.id"
                  f" LEFT JOIN (SELECT court_id, min(date_start) AS first_seen FROM {T('people_db_position')}"
                  f"            WHERE date_start > DATE '{EPOCH}' GROUP BY 1) q ON q.court_id = c.id"
                  f" LEFT JOIN (SELECT dk.court_id, min(k.date_filed) AS first_seen FROM {T('search_opinioncluster')} k"
                  f"            JOIN {T('search_docket')} dk ON dk.id = k.docket_id"
                  f"            WHERE k.date_filed > DATE '{EPOCH}' GROUP BY 1) k ON k.court_id = c.id",
           id="c.id",
           valid_from="coalesce(c.start_date, least(d.first_seen, q.first_seen, k.first_seen))",
           active_to="c.end_date", select="c.* EXCLUDE (id)")

    s.node("docket", source=T("search_docket"),
           id="id", valid_from="date_filed", active_to="date_terminated")

    s.node("opinion",
           source=f"{T('search_opinion')} o JOIN {T('search_opinioncluster')} c ON c.id = o.cluster_id",
           id="o.id", valid_from="c.date_filed",
           select="o.* EXCLUDE (id), "
                  "c.date_filed, c.case_name, c.precedential_status, c.citation_count, c.docket_id")

    # A cluster is CL's "decision": one per case, grouping its opinions (majority, concurrences,
    # dissents, and CL's legacy/new duplicate versions). Citations live on opinions; authorship often
    # on a different opinion row of the same cluster -- the cluster is where they reconcile.
    s.node("cluster", source=T("search_opinioncluster"),
           id="id", valid_from="date_filed")

    s.node("person",
           source=f"{T('people_db_person')} p"
                  f" LEFT JOIN (SELECT person_id, min(date_start) AS first_seen FROM {T('people_db_position')}"
                  f"            WHERE date_start > DATE '{EPOCH}' GROUP BY 1) q ON q.person_id = p.id"
                  f" LEFT JOIN (SELECT o.author_id, min(c.date_filed) AS first_seen FROM {T('search_opinion')} o"
                  f"            JOIN {T('search_opinioncluster')} c ON c.id = o.cluster_id"
                  f"            WHERE o.author_id IS NOT NULL AND c.date_filed > DATE '{EPOCH}' GROUP BY 1) a ON a.author_id = p.id",
           id="p.id", valid_from="coalesce(p.date_dob, least(q.first_seen, a.first_seen))",
           active_to="p.date_dod", select="p.* EXCLUDE (id)")

    # RECAP entities carry no event dates of their own; first appearance on a docket is used.
    s.node("attorney",
           source=f"{T('people_db_attorney')} a LEFT JOIN ("
                  f"  SELECT r.attorney_id, min(coalesce(r.date_action, d.date_filed)) AS first_seen,"
                  f"         CASE WHEN bool_or(d.date_terminated IS NULL) THEN NULL ELSE max(d.date_terminated) END AS last_seen"
                  f"  FROM {T('people_db_role')} r JOIN {T('search_docket')} d ON d.id = r.docket_id"
                  f"  GROUP BY 1) f ON f.attorney_id = a.id",
           id="a.id", valid_from="f.first_seen", active_to="f.last_seen",
           select="a.* EXCLUDE (id)")

    s.node("party",
           source=f"{T('people_db_party')} p LEFT JOIN ("
                  f"  SELECT pt.party_id, min(d.date_filed) AS first_seen,"
                  f"         CASE WHEN bool_or(d.date_terminated IS NULL) THEN NULL ELSE max(d.date_terminated) END AS last_seen"
                  f"  FROM {T('people_db_partytype')} pt JOIN {T('search_docket')} d ON d.id = pt.docket_id"
                  f"  GROUP BY 1) f ON f.party_id = p.id",
           id="p.id", valid_from="f.first_seen", active_to="f.last_seen",
           select="p.* EXCLUDE (id)")

    # ---- edges ----------------------------------------------------------------
    s.edge("person", "presides_over", "docket",
           source=f"(SELECT assigned_to_id AS person_id, id AS docket_id, date_filed, date_terminated, 'assigned' AS role"
                  f"   FROM {T('search_docket')} WHERE assigned_to_id IS NOT NULL"
                  f" UNION ALL"
                  f" SELECT referred_to_id, id, date_filed, date_terminated, 'referred'"
                  f"   FROM {T('search_docket')} WHERE referred_to_id IS NOT NULL) p",
           src_id="person_id", dst_id="docket_id", valid_from="date_filed", active_to="date_terminated",
           select="role")

    s.edge("person", "author", "opinion",
           source=f"{T('search_opinion')} o JOIN {T('search_opinioncluster')} c ON c.id = o.cluster_id",
           src_id="o.author_id", dst_id="o.id", valid_from="c.date_filed",
           select="o.type", where="o.author_id IS NOT NULL")

    s.edge("person", "joined_by", "opinion",
           source=f"{T('search_opinion_joined_by')} j JOIN {T('search_opinion')} o ON o.id = j.opinion_id"
                  f" JOIN {T('search_opinioncluster')} c ON c.id = o.cluster_id",
           src_id="j.person_id", dst_id="j.opinion_id", valid_from="c.date_filed", select="")

    s.edge("attorney", "attorney_in", "docket",
           source=f"{T('people_db_role')} r JOIN {T('search_docket')} d ON d.id = r.docket_id",
           src_id="r.attorney_id", dst_id="r.docket_id",
           valid_from="coalesce(r.date_action, d.date_filed)", active_to="d.date_terminated",
           select="r.role, r.role_raw, r.party_id, r.date_action")

    s.edge("party", "party_in", "docket",
           source=f"{T('people_db_partytype')} pt JOIN {T('search_docket')} d ON d.id = pt.docket_id",
           src_id="pt.party_id", dst_id="pt.docket_id",
           valid_from="d.date_filed", active_to="coalesce(pt.date_terminated, d.date_terminated)",
           select="pt.name AS party_type, pt.extra_info, pt.date_terminated")

    s.edge("attorney", "represents", "party",
           source=f"{T('people_db_role')} r JOIN {T('search_docket')} d ON d.id = r.docket_id",
           src_id="r.attorney_id", dst_id="r.party_id",
           valid_from="coalesce(r.date_action, d.date_filed)", active_to="d.date_terminated",
           select="r.docket_id, r.role")

    s.edge("opinion", "cites", "opinion",
           source=f"{T('search_opinionscited')} x JOIN {T('search_opinion')} o ON o.id = x.citing_opinion_id"
                  f" JOIN {T('search_opinioncluster')} c ON c.id = o.cluster_id",
           src_id="x.citing_opinion_id", dst_id="x.cited_opinion_id", valid_from="c.date_filed",
           select="x.depth")

    s.edge("opinion", "in_cluster", "cluster",
           source=f"{T('search_opinion')} o JOIN {T('search_opinioncluster')} c ON c.id = o.cluster_id",
           src_id="o.id", dst_id="o.cluster_id", valid_from="c.date_filed",
           select="o.type, o.main_version_id")

    s.edge("cluster", "in_docket", "docket", source=T("search_opinioncluster"),
           src_id="id", dst_id="docket_id", valid_from="date_filed", select="")

    # ---- cluster-level (decision-level) relations: the canonical level for hubs -------------
    # ~2% of clusters carry CL's legacy+new duplicate opinion rows, but they receive ~20% of all
    # citations and hold most authorship links, so citation and authorship are reconciled here.
    s.edge("cluster", "cites", "cluster",
           source=f"(SELECT a.cluster_id AS citing, b.cluster_id AS cited, ca.date_filed,"
                  f"        sum(x.depth) AS depth, count(*) AS opinion_pairs"
                  f" FROM {T('search_opinionscited')} x"
                  f" JOIN {T('search_opinion')} a ON a.id = x.citing_opinion_id"
                  f" JOIN {T('search_opinion')} b ON b.id = x.cited_opinion_id"
                  f" JOIN {T('search_opinioncluster')} ca ON ca.id = a.cluster_id"
                  f" WHERE a.cluster_id <> b.cluster_id GROUP BY 1, 2, 3) q",
           src_id="citing", dst_id="cited", valid_from="date_filed", select="depth, opinion_pairs")

    s.edge("person", "author", "cluster",
           source=f"(SELECT DISTINCT o.author_id, o.cluster_id, c.date_filed"
                  f" FROM {T('search_opinion')} o JOIN {T('search_opinioncluster')} c ON c.id = o.cluster_id"
                  f" WHERE o.author_id IS NOT NULL) q",
           src_id="author_id", dst_id="cluster_id", valid_from="date_filed", select="")

    s.edge("opinion", "in_docket", "docket",
           source=f"{T('search_opinion')} o JOIN {T('search_opinioncluster')} c ON c.id = o.cluster_id",
           src_id="o.id", dst_id="c.docket_id", valid_from="c.date_filed", select="")

    s.edge("docket", "filed_in", "court", source=T("search_docket"),
           src_id="id", dst_id="court_id", valid_from="date_filed", select="")

    # people_db_position holds every position at a court: judgeships, retirements, clerkships,
    # prosecutors, professors. sits_on = active judicial service only; clerkships get their own edge.
    s.edge("person", "sits_on", "court", source=T("people_db_position"),
           src_id="person_id", dst_id="court_id",
           valid_from="date_start",
           # senior status / retirement ends active service. Written out because least() returns
           # NULL on any NULL argument in DuckDB < 1.2.
           active_to="CASE WHEN date_termination IS NULL THEN date_retirement "
                     "WHEN date_retirement IS NULL THEN date_termination "
                     "ELSE least(date_termination, date_retirement) END",
           # blank position_type is allowed only when the free-text job_title is a bench title
           where=f"court_id IS NOT NULL AND coalesce(position_type, '') NOT IN ({NON_JUDICIAL}) "
                 f"AND (coalesce(position_type, '') <> '' OR regexp_matches(lower(coalesce(job_title, '')), "
                 f"'judge|justice|surrogate|chancellor|magistrate|referee|commissioner|jurist'))")

    s.edge("person", "clerked_for", "court", source=T("people_db_position"),
           src_id="person_id", dst_id="court_id",
           valid_from="date_start", active_to="date_termination",
           select="id AS position_id, position_type, supervisor_id, date_start, date_termination",
           where="court_id IS NOT NULL AND position_type IN ('clerk', 'clerk-chief-dep', 'staff-atty')")

    s.edge("person", "succeeds", "person", source=T("people_db_position"),
           src_id="person_id", dst_id="predecessor_id", valid_from="date_start",
           select="id AS position_id, court_id, position_type, date_start, date_termination",
           where="predecessor_id IS NOT NULL")

    return s


SCHEMA = courtlistener_schema()

