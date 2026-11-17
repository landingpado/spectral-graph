"""
Local API + static site for the citation-space explorer. One command, one process, local only:

    python -m lawgraph.serve --db graph.duckdb --port 8080 [--no-open]

    /            viewer/ (index.html, years.html, js/, vendor/, data/)
    /api/...     read-only DuckDB queries over the whole corpus (the browser never loads it)

Guards on every query
  * the database is opened once, read-only; requests share that one connection (a fixed set of worker
    threads each hold a cursor of it, DuckDB's handle for concurrent use of a single connection, created
    at startup), so nothing is opened per request
  * a statement runs in a worker thread and is interrupted after `timeout` seconds (default 5) -> HTTP 504
  * every statement is wrapped in  SELECT * FROM (...) LIMIT <row_cap>  (default 5000) and fetched with
    fetchmany(row_cap): no endpoint can return more rows than that
  * SQL text is fixed at startup; request values only ever travel as bound parameters
  * results are cached in memory (LRU), so a repeated query costs a dictionary lookup

Endpoints
  GET /api/courts                                         [{id, name, n_cases}]
  GET /api/cases?year=1973&court=scotus&q=roe&min_cited=10&limit=500
  GET /api/case/{id}                                      metadata, opinions, authors, CourtListener URL
  GET /api/cites/{id}?direction=in|out&year=2000&limit=200
  GET /api/bench?court=ca9&date=1990-06-01                judges sitting that day
  GET /api/timeline?court=scotus&start=1900&end=2020      cases filed and citations made per year
  GET /api/meta                                           row cap, timeout, cache statistics
"""
# No `from __future__ import annotations` here: FastAPI resolves endpoint annotations at definition time, and
# the Year / Limit aliases below are local to create_app (Limit depends on the app's row cap).
import argparse
import datetime as dt
import os
import queue
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Annotated

from . import hw

ROW_CAP = 5000
TIMEOUT_S = 5.0
CL_URL = "https://www.courtlistener.com/opinion/{id}/x/"
VIEWER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "viewer")


class QueryTimeout(Exception):
    """A statement ran past the time limit and was interrupted."""


# ---------------------------------------------------------------- the guarded connection
class Guard:
    """Runs fixed, parameterised SQL on one shared read-only connection with a time limit, a row cap and an
    LRU result cache. See the module docstring for what each guard means."""

    def __init__(self, con, timeout: float = TIMEOUT_S, row_cap: int = ROW_CAP, cache_size: int = 1024, workers: int = 8):
        self.con, self.timeout, self.row_cap = con, float(timeout), int(row_cap)
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="duck")
        self._local = threading.local()
        # One cursor per worker thread, all made here on the constructing thread, so that afterwards only the
        # owner ever touches `con` itself. DuckDB raises "connection is busy with another call" rather than
        # waiting when a connection object is used from two threads at once (seen at shutdown: close() on the
        # owner racing a worker that was still asking for its cursor).
        self._cursors = [con.cursor() for _ in range(workers)]
        self._free: queue.SimpleQueue = queue.SimpleQueue()
        for cur in self._cursors:
            self._free.put(cur)
        self._cache: OrderedDict = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()
        self.stats = {"queries": 0, "cache_hits": 0, "timeouts": 0, "errors": 0}

    def _cursor(self):
        cur = getattr(self._local, "cur", None)
        if cur is None:
            cur = self._local.cur = self._free.get_nowait()   # this worker thread's own handle on the shared database
        return cur

    def _run(self, sql: str, params: tuple, slot: dict) -> list[dict]:
        cur = slot["cur"] = self._cursor()
        res = cur.execute(f"SELECT * FROM ({sql}) AS q LIMIT {self.row_cap}", list(params))
        cols = [d[0] for d in res.description]
        return [dict(zip(cols, row)) for row in res.fetchmany(self.row_cap)]

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        """Rows as dicts, never more than row_cap. Raises QueryTimeout after `timeout` seconds."""
        key = (sql, params)
        with self._lock:
            self.stats["queries"] += 1
            if key in self._cache:
                self._cache.move_to_end(key); self.stats["cache_hits"] += 1
                return self._cache[key]
        slot: dict = {}
        fut = self._pool.submit(self._run, sql, params, slot)
        try:
            rows = fut.result(timeout=self.timeout)
        except FutureTimeout:
            with self._lock:
                self.stats["timeouts"] += 1
            cur = slot.get("cur")
            if cur is not None:
                cur.interrupt()                              # the worker gets an InterruptException and is free again
            else:
                fut.cancel()                                 # still queued behind other statements
            raise QueryTimeout(f"query exceeded {self.timeout:g} s and was interrupted") from None
        except Exception:
            with self._lock:
                self.stats["errors"] += 1
            raise
        with self._lock:
            self._cache[key] = rows
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return rows

    def columns(self, table: str) -> set[str]:
        return {r[0] for r in self.con.execute(f"DESCRIBE {table}").fetchall()}

    def close(self) -> None:
        """Stop accepting work, interrupt whatever is still running, and release the cursors."""
        for cur in self._cursors:
            try: cur.interrupt()
            except Exception: pass
        self._pool.shutdown(wait=True, cancel_futures=True)
        for cur in self._cursors:
            try: cur.close()
            except Exception: pass


# ---------------------------------------------------------------- SQL (fixed text; values are always bound)
class Sql:
    """Every statement the API can run, built once from the columns the database actually has (the mock
    schema used in tests has fewer attribute columns than a real CourtListener mirror)."""

    def __init__(self, guard: Guard):
        court, person, cluster, sits = (guard.columns(t) for t in ("node_court", "node_person", "node_cluster", "edge_person_sits_on_court"))
        cname = "coalesce(nullif(c.full_name, ''), nullif(c.short_name, ''), c.id)" if "full_name" in court else "coalesce(nullif(c.short_name, ''), c.id)"
        parts = [f"p.{c}" for c in ("name_first", "name_middle", "name_last", "name_suffix") if c in person]
        pname = f"trim(regexp_replace(concat_ws(' ', {', '.join(parts)}), '\\s+', ' ', 'g'))" if parts else "CAST(p.id AS VARCHAR)"
        extra = "".join(f", n.{c}" for c in ("case_name_full", "judges", "scdb_id", "nature_of_suit", "posture", "syllabus") if c in cluster)
        pos = "".join(f", s.{c}" for c in ("position_type", "job_title") if c in sits)
        in_court = ("JOIN edge_cluster_in_docket_docket cd ON cd.src_id = n.id "
                    "JOIN edge_docket_filed_in_court f ON f.src_id = cd.dst_id")

        self.courts = f"""
            SELECT c.id, {cname} AS name, coalesce(k.n, 0) AS n_cases
            FROM node_court c
            LEFT JOIN (SELECT f.dst_id AS court, count(*) AS n
                       FROM edge_cluster_in_docket_docket cd JOIN edge_docket_filed_in_court f ON f.src_id = cd.dst_id
                       GROUP BY 1) k ON k.court = c.id
            ORDER BY n_cases DESC, c.id"""

        # cases filed in one year; in_cites_to_date = citations received up to the end of that year
        def cases(courts: bool, q: bool) -> str:
            return f"""
            WITH y AS (
                SELECT n.id, n.case_name, n.date_filed, n.citation_count, f.dst_id AS court
                FROM node_cluster n {in_court}
                WHERE n.date_filed >= make_date(CAST(? AS INTEGER), 1, 1) AND n.date_filed < make_date(CAST(? AS INTEGER) + 1, 1, 1)
                  {"AND f.dst_id IN (SELECT unnest(CAST(? AS VARCHAR[])))" if courts else ""}
                  {"AND n.case_name ILIKE ? ESCAPE '!'" if q else ""}
            ), cited AS (
                SELECT e.dst_id AS id, count(*) AS n
                FROM edge_cluster_cites_cluster e
                WHERE e.dst_id IN (SELECT id FROM y) AND e.valid_from < make_date(CAST(? AS INTEGER) + 1, 1, 1)
                GROUP BY 1
            )
            SELECT y.id, y.case_name, y.date_filed, y.court, y.citation_count, coalesce(c.n, 0) AS in_cites_to_date
            FROM y LEFT JOIN cited c ON c.id = y.id
            WHERE coalesce(c.n, 0) >= CAST(? AS BIGINT)
            ORDER BY in_cites_to_date DESC, y.citation_count DESC NULLS LAST, y.date_filed, y.id
            LIMIT CAST(? AS INTEGER)"""
        self.cases = {(c, q): cases(c, q) for c in (False, True) for q in (False, True)}

        self.case = f"""
            SELECT n.id, n.case_name, n.date_filed, n.precedential_status, n.citation_count{extra},
                   cd.dst_id AS docket_id, f.dst_id AS court, {cname} AS court_name
            FROM node_cluster n
            LEFT JOIN edge_cluster_in_docket_docket cd ON cd.src_id = n.id
            LEFT JOIN edge_docket_filed_in_court f ON f.src_id = cd.dst_id
            LEFT JOIN node_court c ON c.id = f.dst_id
            WHERE n.id = CAST(? AS BIGINT)"""
        self.opinions = """
            SELECT o.src_id AS id, o.type FROM edge_opinion_in_cluster_cluster o
            WHERE o.dst_id = CAST(? AS BIGINT) ORDER BY o.type, o.src_id"""
        self.authors = f"""
            SELECT DISTINCT p.id, {pname} AS name
            FROM edge_person_author_cluster a JOIN node_person p ON p.id = a.src_id
            WHERE a.dst_id = CAST(? AS BIGINT) ORDER BY name, p.id"""

        # citing (in) or cited (out) decisions; `year` keeps citations made up to the end of that year
        def cites(direction: str, year: bool) -> str:
            me, other = ("dst_id", "src_id") if direction == "in" else ("src_id", "dst_id")
            return f"""
            SELECT n.id, n.case_name, n.date_filed, n.citation_count, e.valid_from AS cited_on, CAST(e.depth AS BIGINT) AS depth
            FROM edge_cluster_cites_cluster e JOIN node_cluster n ON n.id = e.{other}
            WHERE e.{me} = CAST(? AS BIGINT) {"AND e.valid_from < make_date(CAST(? AS INTEGER) + 1, 1, 1)" if year else ""}
            ORDER BY n.citation_count DESC NULLS LAST, n.date_filed, n.id
            LIMIT CAST(? AS INTEGER)"""
        self.cites = {(d, y): cites(d, y) for d in ("in", "out") for y in (False, True)}

        self.bench = f"""
            SELECT p.id, {pname} AS name{pos}, s.active_from AS "from", s.active_to AS "to"
            FROM edge_person_sits_on_court s JOIN node_person p ON p.id = s.src_id
            WHERE s.dst_id = ? AND s.active_from <= CAST(? AS DATE) AND (s.active_to IS NULL OR s.active_to > CAST(? AS DATE))
            QUALIFY row_number() OVER (PARTITION BY p.id ORDER BY s.active_from DESC) = 1
            ORDER BY "from", name"""

        # cases filed and citations made (by those cases) per year, one grouped statement
        def timeline(court: bool) -> str:
            return f"""
            WITH k AS (
                SELECT n.id, year(n.date_filed) AS y
                FROM node_cluster n {in_court if court else ""}
                WHERE n.date_filed >= make_date(CAST(? AS INTEGER), 1, 1) AND n.date_filed < make_date(CAST(? AS INTEGER) + 1, 1, 1)
                  {"AND f.dst_id = ?" if court else ""}
            ), made AS (
                SELECT k.y, count(*) AS n FROM edge_cluster_cites_cluster e JOIN k ON k.id = e.src_id GROUP BY 1
            )
            SELECT g.y AS year, g.cases, coalesce(m.n, 0) AS citations_made
            FROM (SELECT y, count(*) AS cases FROM k GROUP BY 1) g LEFT JOIN made m ON m.y = g.y
            ORDER BY 1"""
        self.timeline = {c: timeline(c) for c in (False, True)}


def like_pattern(q: str) -> str:
    """%q% with ILIKE wildcards in q taken literally (the statement uses ESCAPE '!')."""
    return "%" + q.replace("!", "!!").replace("%", "!%").replace("_", "!_") + "%"


# ---------------------------------------------------------------- app
def create_app(con, viewer_dir: str | None = VIEWER_DIR, timeout: float = TIMEOUT_S, row_cap: int = ROW_CAP):
    """FastAPI app over an open DuckDB connection (read-only for a real database; tests pass Store.mock().con).
    viewer_dir=None serves the API only."""
    from fastapi import FastAPI, HTTPException, Path, Query, Request
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles

    guard = Guard(con, timeout=timeout, row_cap=row_cap)
    sql = Sql(guard)
    app = FastAPI(title="lawgraph explorer", docs_url="/api/docs", openapi_url="/api/openapi.json", redoc_url=None)
    app.state.guard, app.state.sql = guard, sql
    cap = guard.row_cap

    @app.exception_handler(QueryTimeout)
    def _timeout(_: Request, exc: QueryTimeout):
        return JSONResponse(status_code=504, content={"detail": str(exc)})

    @app.middleware("http")
    async def _revalidate(request: Request, call_next):
        # a rebuild replaces viewer/data in place: make the browser revalidate instead of mixing old and new chunks
        response = await call_next(request)
        if not request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    Year = Annotated[int, Query(ge=1600, le=2200)]
    Limit = Annotated[int, Query(ge=1, le=cap, description=f"at most {cap}")]

    @app.get("/api/meta")
    def meta():
        return {"row_cap": cap, "timeout_s": guard.timeout, "stats": dict(guard.stats), "courtlistener": CL_URL}

    @app.get("/api/courts")
    def courts():
        return guard.query(sql.courts)

    @app.get("/api/cases")
    def cases(year: Year, court: Annotated[list[str] | None, Query(description="court id; repeat or comma-separate for several")] = None,
              q: Annotated[str | None, Query(max_length=200)] = None, min_cited: Annotated[int, Query(ge=0)] = 0,
              limit: Limit = min(500, cap)):
        ids = sorted({c.strip() for item in (court or []) for c in item.split(",") if c.strip()})
        if len(ids) > 500:
            raise HTTPException(422, "too many courts")
        q = (q or "").strip()
        params: list = [year, year]
        if ids: params.append(ids)
        if q: params.append(like_pattern(q))
        params += [year, min_cited, limit]
        return guard.query(sql.cases[(bool(ids), bool(q))], tuple(tuple(p) if isinstance(p, list) else p for p in params))

    @app.get("/api/case/{id}")
    def case(id: Annotated[int, Path(ge=0)]):
        rows = guard.query(sql.case, (id,))
        if not rows:
            raise HTTPException(404, f"no cluster {id}")
        return {**rows[0], "opinions": guard.query(sql.opinions, (id,)), "authors": guard.query(sql.authors, (id,)),
                "url": CL_URL.format(id=id)}

    @app.get("/api/cites/{id}")
    def cites(id: Annotated[int, Path(ge=0)], direction: Annotated[str, Query(pattern="^(in|out)$")] = "in",
              year: Annotated[int | None, Query(ge=1600, le=2200)] = None, limit: Limit = min(200, cap)):
        params = (id, year, limit) if year is not None else (id, limit)
        return guard.query(sql.cites[(direction, year is not None)], params)

    @app.get("/api/bench")
    def bench(court: Annotated[str, Query(min_length=1, max_length=64)], date: dt.date):
        return guard.query(sql.bench, (court, date.isoformat(), date.isoformat()))

    @app.get("/api/timeline")
    def timeline(court: Annotated[str | None, Query(max_length=64)] = None, start: Year = 1600, end: Year = 2200):
        if end < start:
            raise HTTPException(422, "end must not be before start")
        return guard.query(sql.timeline[bool(court)], (start, end, court) if court else (start, end))

    if viewer_dir and os.path.isdir(viewer_dir):
        app.mount("/", StaticFiles(directory=viewer_dir, html=True), name="viewer")
    return app


def warm(guard: Guard, sql: Sql) -> None:
    """Run the one statement every page load needs (court list with counts: a 10M x 73M join) ahead of time."""
    try:
        t = time.time(); guard.query(sql.courts)
        print(f"court list ready ({time.time() - t:.1f}s)", flush=True)
    except Exception as e:                                           # a slow disk only delays the first request
        print(f"court list not pre-computed ({type(e).__name__}); the first request will run it", flush=True)


def open_browser(url: str) -> None:
    import webbrowser
    webbrowser.open(url)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="graph.duckdb", help="DuckDB graph file (opened read-only)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--viewer", default=VIEWER_DIR, help="directory served at / (default: the repo's viewer/)")
    ap.add_argument("--no-open", action="store_true", help="do not open a browser")
    ap.add_argument("--threads", type=int, default=None, help="DuckDB threads (default: all cores)")
    ap.add_argument("--memory-gb", type=float, default=None, help="DuckDB memory limit (default: half of RAM)")
    a = ap.parse_args(argv)
    if not os.path.exists(a.db):
        sys.exit(f"database not found: {a.db}")
    if not os.path.exists(os.path.join(a.viewer, "data", "manifest.json")):
        print(f"note: no {os.path.join(a.viewer, 'data', 'manifest.json')} -- the 3-D view needs\n"
              f"  python -m lawgraph.viewer_build {a.db} --out {os.path.join(a.viewer, 'data')}", flush=True)

    import duckdb
    import uvicorn
    con = hw.tune(duckdb.connect(a.db, read_only=True), a.threads, a.memory_gb)
    app = create_app(con, a.viewer)
    threading.Thread(target=warm, args=(app.state.guard, app.state.sql), daemon=True).start()

    url = f"http://127.0.0.1:{a.port}/"
    print(f"explorer at {url}   year view at {url}years.html   API at {url}api/docs   (Ctrl-C to stop)", flush=True)
    if not a.no_open:
        threading.Timer(1.0, open_browser, args=(url,)).start()      # once uvicorn is listening
    try:
        uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")      # loopback only: this is a local tool
    finally:
        app.state.guard.close(); con.close()


if __name__ == "__main__":
    main()
