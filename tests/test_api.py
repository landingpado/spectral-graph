"""
API tests against the bundled mock database (lawgraph/mock.py): every endpoint, the row cap, the timeout.

    python -m pytest tests/test_api.py -q

The mock has one court (scotus), three decisions (Marbury 1803, McCulloch 1819, Dred Scott 1857), their
citations and three justices. The fixture adds a handful of rows to its in-memory copy so that same-year
citations and the row cap have something to bite on; graph.duckdb is never touched.
"""
import time

import pytest
from fastapi.testclient import TestClient

from lawgraph import serve
from lawgraph.store import Store

N_EXTRA = 12          # extra 1857 decisions, each citing Dred Scott (300) and Marbury (100)


@pytest.fixture(scope="module")
def con():
    st = Store.mock()
    con = st.con
    for k in range(N_EXTRA):
        cid, did, day = 400 + k, 40 + k, f"1857-{4 + k % 8:02d}-{1 + k:02d}"
        con.execute("INSERT INTO node_cluster (id, valid_from, active_from, date_filed, case_name, precedential_status, citation_count, docket_id) "
                    "VALUES (?, ?, ?, ?, ?, 'Published', ?, ?)", [cid, day, day, day, f"Ableman {k:02d}% v. Booth_", k, did])
        con.execute("INSERT INTO edge_cluster_in_docket_docket (src_id, dst_id, valid_from, active_from) VALUES (?, ?, ?, ?)", [cid, did, day, day])
        con.execute("INSERT INTO edge_docket_filed_in_court (src_id, dst_id, valid_from, active_from) VALUES (?, 'scotus', ?, ?)", [did, day, day])
        for cited in (300, 100):
            con.execute("INSERT INTO edge_cluster_cites_cluster (src_id, dst_id, valid_from, active_from, depth, opinion_pairs) VALUES (?, ?, ?, ?, 1, 1)",
                        [cid, cited, day, day])
    yield con
    con.close()


@pytest.fixture(scope="module")
def client(con):
    app = serve.create_app(con, viewer_dir=None)
    with TestClient(app) as c:
        yield c
    app.state.guard.close()


def names(rows):
    return [r["case_name"] for r in rows]


# ---------------------------------------------------------------- endpoints
def test_courts(client):
    r = client.get("/api/courts")
    assert r.status_code == 200
    assert r.json() == [{"id": "scotus", "name": "SCOTUS", "n_cases": 3 + N_EXTRA}]


def test_cases_by_year_and_court(client):
    rows = client.get("/api/cases", params={"year": 1803, "court": "scotus"}).json()
    assert names(rows) == ["Marbury"]
    assert set(rows[0]) == {"id", "case_name", "date_filed", "court", "citation_count", "in_cites_to_date"}
    assert rows[0]["date_filed"] == "1803-02-24" and rows[0]["court"] == "scotus"
    assert rows[0]["in_cites_to_date"] == 0                  # first cited in 1819: nothing by the end of 1803
    assert client.get("/api/cases", params={"year": 1803, "court": "nowhere"}).json() == []
    assert client.get("/api/cases", params={"year": 1900}).json() == []


def test_cases_citations_are_counted_up_to_that_year(client):
    rows = client.get("/api/cases", params={"year": 1857, "court": "scotus"}).json()
    by = {r["case_name"]: r for r in rows}
    assert by["Dred Scott"]["in_cites_to_date"] == N_EXTRA   # cited by the twelve later 1857 decisions
    assert by["Dred Scott"]["citation_count"] == 0           # CourtListener's own (present-day) count is passed through as is
    assert names(rows)[0] == "Dred Scott"                    # ordered by citations received so far
    cited = client.get("/api/cases", params={"year": 1857, "min_cited": 1}).json()
    assert names(cited) == ["Dred Scott"]
    assert client.get("/api/cases", params={"year": 1857, "min_cited": N_EXTRA + 1}).json() == []


def test_cases_name_search_is_ilike_with_literal_wildcards(client):
    assert names(client.get("/api/cases", params={"year": 1857, "q": "dred"}).json()) == ["Dred Scott"]
    assert len(client.get("/api/cases", params={"year": 1857, "q": "ABLEMAN"}).json()) == N_EXTRA
    assert len(client.get("/api/cases", params={"year": 1857, "q": "03% v."}).json()) == 1      # % is a character, not a wildcard
    assert client.get("/api/cases", params={"year": 1857, "q": "D_ed"}).json() == []            # so is _
    assert len(client.get("/api/cases", params={"year": 1857, "q": "Booth_"}).json()) == N_EXTRA


def test_cases_court_list_forms(client):
    a = client.get("/api/cases?year=1857&court=scotus&court=ca9").json()
    b = client.get("/api/cases?year=1857&court=scotus,ca9").json()
    assert a == b and len(a) == 1 + N_EXTRA


def test_case_detail(client):
    d = client.get("/api/case/300").json()
    assert d["case_name"] == "Dred Scott" and d["date_filed"] == "1857-03-06" and d["court"] == "scotus"
    assert d["court_name"] == "SCOTUS" and d["docket_id"] == 30 and d["precedential_status"] == "Published"
    assert d["opinions"] == [{"id": 3000, "type": "020lead"}, {"id": 3001, "type": "040dissent"}]
    assert d["authors"] == [{"id": 2, "name": "Joseph Story"}, {"id": 3, "name": "Roger B. Taney"}]
    assert d["url"] == "https://www.courtlistener.com/opinion/300/x/"
    assert client.get("/api/case/999").status_code == 404
    assert client.get("/api/case/abc").status_code == 422


def test_cites_in_and_out(client):
    citing = client.get("/api/cites/100", params={"direction": "in"}).json()
    assert {"McCulloch", "Dred Scott"} <= set(names(citing)) and len(citing) == 2 + N_EXTRA
    assert set(citing[0]) == {"id", "case_name", "date_filed", "citation_count", "cited_on", "depth"}
    early = client.get("/api/cites/100", params={"direction": "in", "year": 1820}).json()
    assert names(early) == ["McCulloch"] and early[0]["cited_on"] == "1819-03-06"
    assert client.get("/api/cites/100", params={"direction": "in", "year": 1818}).json() == []
    cited = client.get("/api/cites/300", params={"direction": "out"}).json()
    assert sorted(names(cited)) == ["Marbury", "McCulloch"]
    assert client.get("/api/cites/100").json() == citing                      # direction defaults to "in"
    assert client.get("/api/cites/100", params={"direction": "sideways"}).status_code == 422


def test_bench(client):
    def sitting(date):
        r = client.get("/api/bench", params={"court": "scotus", "date": date}); assert r.status_code == 200
        return [j["name"] for j in r.json()]
    assert sitting("1820-01-01") == ["John Marshall", "Joseph Story"]
    assert sitting("1840-01-01") == ["Joseph Story", "Roger B. Taney"]         # Marshall died in 1835
    assert sitting("1850-01-01") == ["Roger B. Taney"]                        # Story retired in 1845
    assert sitting("1790-01-01") == []
    assert client.get("/api/bench", params={"court": "ca9", "date": "1820-01-01"}).json() == []
    assert client.get("/api/bench", params={"court": "scotus", "date": "not-a-date"}).status_code == 422
    assert client.get("/api/bench", params={"court": "scotus"}).status_code == 422


def test_timeline(client):
    rows = client.get("/api/timeline", params={"court": "scotus", "start": 1800, "end": 1860}).json()
    assert rows == [{"year": 1803, "cases": 1, "citations_made": 0},
                    {"year": 1819, "cases": 1, "citations_made": 1},
                    {"year": 1857, "cases": 1 + N_EXTRA, "citations_made": 2 + 2 * N_EXTRA}]
    assert client.get("/api/timeline", params={"start": 1800, "end": 1860}).json() == rows      # all courts = the one court
    assert client.get("/api/timeline", params={"court": "scotus", "start": 1810, "end": 1830}).json() == [rows[1]]
    assert client.get("/api/timeline", params={"court": "ca9"}).json() == []
    assert client.get("/api/timeline", params={"start": 1900, "end": 1800}).status_code == 422


def test_meta(client):
    m = client.get("/api/meta").json()
    assert m["row_cap"] == 5000 and m["timeout_s"] == 5.0


# ---------------------------------------------------------------- guards
def test_limit_cannot_exceed_the_row_cap(client):
    assert client.get("/api/cases", params={"year": 1857, "limit": 5000}).status_code == 200
    assert client.get("/api/cases", params={"year": 1857, "limit": 5001}).status_code == 422
    assert client.get("/api/cites/100", params={"limit": 999999}).status_code == 422
    assert client.get("/api/cases", params={"year": 1857, "limit": 0}).status_code == 422
    assert len(client.get("/api/cases", params={"year": 1857, "limit": 4}).json()) == 4


def test_row_cap_is_enforced_on_every_query(con):
    """With the cap lowered to 3, no endpoint returns more than 3 rows, whatever the request asks for."""
    app = serve.create_app(con, viewer_dir=None, row_cap=3)
    with TestClient(app) as c:
        assert len(c.get("/api/cases", params={"year": 1857}).json()) == 3
        assert len(c.get("/api/cites/100").json()) == 3
        assert c.get("/api/cases", params={"year": 1857, "limit": 4}).status_code == 422       # limit is validated against the cap
        assert len(c.get("/api/case/300").json()["opinions"]) <= 3
        assert len(app.state.guard.query("SELECT range AS i FROM range(100000)")) == 3        # the guard itself
    app.state.guard.close()


SLOW = "SELECT count(*) AS n FROM range(100000000) a, range(100000000) b WHERE a.range + b.range < 0"


def test_guard_interrupts_a_slow_query(con):
    guard = serve.Guard(con, timeout=0.3)
    t = time.time()
    with pytest.raises(serve.QueryTimeout):
        guard.query(SLOW)
    assert time.time() - t < 3.0                              # gave up at the limit, did not wait for the query
    assert guard.stats["timeouts"] == 1
    time.sleep(0.5)                                           # the interrupted worker is usable again
    assert guard.query("SELECT 42 AS x") == [{"x": 42}]
    guard.close()


def test_timeout_becomes_http_504(con):
    app = serve.create_app(con, viewer_dir=None, timeout=0.3)
    app.state.sql.courts = SLOW                               # make one endpoint slow
    with TestClient(app) as c:
        t = time.time(); r = c.get("/api/courts")
        assert r.status_code == 504 and "interrupted" in r.json()["detail"]
        assert time.time() - t < 3.0
        time.sleep(0.5)
        assert c.get("/api/bench", params={"court": "scotus", "date": "1820-01-01"}).status_code == 200    # still serving
    app.state.guard.close()


def test_results_are_cached(con):
    guard = serve.Guard(con)
    a = guard.query("SELECT count(*) AS n FROM node_cluster WHERE id > ?", (0,))
    b = guard.query("SELECT count(*) AS n FROM node_cluster WHERE id > ?", (0,))
    assert a is b and guard.stats == {"queries": 2, "cache_hits": 1, "timeouts": 0, "errors": 0}
    assert guard.query("SELECT count(*) AS n FROM node_cluster WHERE id > ?", (100,)) is not a          # other parameters, other entry
    guard.close()


def test_request_values_are_bound_never_interpolated(client, con):
    before = con.execute("SELECT count(*) FROM node_cluster").fetchone()[0]
    for evil in ("'; DROP TABLE node_cluster; --", "' OR '1'='1", "%' UNION SELECT 1,2,3,4,5,6 --"):
        assert client.get("/api/cases", params={"year": 1857, "q": evil}).json() == []
        assert client.get("/api/cases", params={"year": 1857, "court": evil}).json() == []
        assert client.get("/api/bench", params={"court": evil, "date": "1820-01-01"}).json() == []
        assert client.get("/api/timeline", params={"court": evil}).json() == []
    assert client.get("/api/case/300;DROP TABLE node_cluster").status_code == 422
    assert con.execute("SELECT count(*) FROM node_cluster").fetchone()[0] == before


def test_static_site_is_mounted_with_the_api(con, tmp_path):
    (tmp_path / "index.html").write_text("<!doctype html><title>viewer</title>")
    (tmp_path / "data").mkdir(); (tmp_path / "data" / "manifest.json").write_text("{}")
    app = serve.create_app(con, viewer_dir=str(tmp_path))
    with TestClient(app) as c:
        r = c.get("/")
        assert r.status_code == 200 and "viewer" in r.text and r.headers["cache-control"] == "no-cache"
        assert c.get("/data/manifest.json").json() == {}
        assert c.get("/api/courts").status_code == 200                       # the API is not shadowed by the static mount
        assert c.get("/missing.html").status_code == 404
    app.state.guard.close()


# ---------------------------------------------------------------- the command line (T7)
class _Now:
    """threading.Timer stand-in that fires immediately."""
    def __init__(self, delay, fn, args=()): self.fn, self.args = fn, args
    def start(self): self.fn(*self.args)


@pytest.fixture()
def mock_db(tmp_path):
    """A mock database on disk: main() takes a path and opens it read-only."""
    from lawgraph.cl import mirror
    from lawgraph.mock import make
    path = str(tmp_path / "mock.duckdb")
    st = Store(path)
    make(st.con); mirror(st.con, verbose=False); st.schema.build(st.con, verbose=False)
    st.con.close()
    return path


def _run_main(monkeypatch, argv):
    import uvicorn
    seen = {"opened": [], "run": None}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(run=(app, kw)))
    monkeypatch.setattr(serve, "open_browser", lambda url: seen["opened"].append(url))
    monkeypatch.setattr(serve.threading, "Timer", _Now)
    serve.main(argv)
    return seen


def test_main_serves_on_loopback_and_opens_the_browser(monkeypatch, mock_db, tmp_path, capsys):
    (tmp_path / "v" / "data").mkdir(parents=True); (tmp_path / "v" / "index.html").write_text("ok")
    seen = _run_main(monkeypatch, ["--db", mock_db, "--port", "8123", "--viewer", str(tmp_path / "v")])
    app, kw = seen["run"]
    assert kw["host"] == "127.0.0.1" and kw["port"] == 8123                 # local only
    assert seen["opened"] == ["http://127.0.0.1:8123/"]
    assert "http://127.0.0.1:8123/" in capsys.readouterr().out             # prints the URL
    assert {r.path for r in app.routes} >= {"/api/courts", "/api/cases", "/api/case/{id}", "/api/cites/{id}", "/api/bench", "/api/timeline"}


def test_main_no_open(monkeypatch, mock_db, tmp_path):
    seen = _run_main(monkeypatch, ["--db", mock_db, "--no-open", "--viewer", str(tmp_path)])
    assert seen["opened"] == [] and seen["run"][1]["port"] == 8080


def test_main_opens_the_database_read_only(monkeypatch, mock_db, tmp_path):
    import duckdb
    modes = []
    real = duckdb.connect
    monkeypatch.setattr(duckdb, "connect", lambda path, **kw: (modes.append(kw.get("read_only")), real(path, **kw))[1])
    _run_main(monkeypatch, ["--db", mock_db, "--no-open", "--viewer", str(tmp_path)])
    assert modes == [True]


def test_main_refuses_a_missing_database(tmp_path):
    with pytest.raises(SystemExit) as e:
        serve.main(["--db", str(tmp_path / "nope.duckdb"), "--no-open"])
    assert "not found" in str(e.value)
    assert not (tmp_path / "nope.duckdb").exists()                        # and does not create one


def test_concurrent_requests_share_the_connection_safely(con):
    """96 simultaneous uncached requests on 32 client threads all succeed: each API worker thread runs its
    statements on its own cursor of the one shared connection."""
    from concurrent.futures import ThreadPoolExecutor
    app = serve.create_app(con, viewer_dir=None)
    with TestClient(app) as c:
        def hit(k):
            r = c.get("/api/cites/100", params={"direction": "in", "limit": 1 + k % 13})      # distinct parameters: no cache hits
            return r.status_code, len(r.json())
        with ThreadPoolExecutor(32) as pool:
            results = list(pool.map(hit, range(96)))
    assert all(code == 200 for code, _ in results), [r for r in results if r[0] != 200][:3]
    assert all(n == 1 + k % 13 for k, (_, n) in enumerate(results))
    assert app.state.guard.stats["errors"] == 0
    app.state.guard.close()
