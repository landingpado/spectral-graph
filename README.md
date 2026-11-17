# lawgraph

A typed, temporal, heterogeneous graph over CourtListener, stored as plain DuckDB tables.

Node types: court, docket, cluster (a decision), opinion, person (judges), attorney, party.
Relations: cluster/opinion cites, person author (to cluster and to opinion), joined_by, in_cluster,
in_docket (cluster and opinion), filed_in, presides_over, sits_on, clerked_for, succeeds,
attorney_in, party_in, represents. Every table has valid_* (exists) and active_* (in force) intervals.

Use the cluster level by default: ~2% of clusters carry duplicate legacy/new opinion rows but
receive ~20% of all citations, and authorship often sits on a different opinion row than the
citations do. `cluster -[cites]-> cluster` and `person -[author]-> cluster` reconcile this.

```
lawgraph/
  schema.py         generic: NodeSpec / EdgeSpec / Schema -> CREATE TABLE AS SELECT per type
  cl.py             the CourtListener mapping (6 node types, 11 relations); no attributes declared
  store.py          Store (connection + build) and Snapshot (a time slice: predicate + tensor adapters)
  build.py          CLI
  mock.py           tiny fake CL schema for tests
  lens/             Graph, select/build, Spectral, Leiden + lineage, Procrustes alignment
  viewer_build.py   precompute for the explorer: graph.duckdb (read-only) -> viewer/data
  viewer_export.py  the binary/JSON formats the viewer reads
  force_layout.py   ForceAtlas2 layout of the Leiden keyframes with community gravity (GPU or CPU)
  serve.py          FastAPI: viewer/ as a static site + read-only /api over the whole corpus
  hw.py             cores / RAM / GPU probe, DuckDB thread and memory settings
viewer/             the explorer: index.html (3-D citation space), years.html (year view), js/, vendor/, data/
tests/              test_api.py, test_build.py (pytest), test_viewer.mjs (headless Chrome)
```

## Explorer (local)

Two commands; the database is only ever opened read-only.

```bash
python -m lawgraph.viewer_build graph.duckdb --out viewer/data --summaries notebooks/case_summaries.csv --gpu
python -m lawgraph.serve --db graph.duckdb --port 8080       # prints the URL and opens a browser (--no-open)
```

The first precomputes the SCOTUS citation space (about 75 s on a 128-core machine with a GPU, about 7 min
without `--gpu`); the second serves the 3-D view at `/`, the year view at `/years.html` and the API at
`/api`. See `viewer/README.md` for what is computed, the controls, the data formats and the tests.

## Install

```bash
pip install duckdb numpy scipy        # the graph store alone; scipy only needed for adjacency()
pip install -r requirements.txt       # everything: lenses, explorer precompute, API server, tests
```

DuckDB's `postgres` extension is installed on first use (`INSTALL postgres`), which needs
network access once. On an air-gapped node, run it once on a login node with the same DuckDB
version; the extension is cached in `~/.duckdb/extensions/`.

## Credentials

Nothing is stored in code. Either

```bash
export CL_DSN="dbname=courtlistener host=... port=5432 user=... password=..."
```

or set the standard libpq variables (`PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`,
`PGPASSWORD`, or `~/.pgpass`) and leave `CL_DSN` unset. The replica is attached read-only.

## Build

Two phases. First the 12 CourtListener tables are copied verbatim into a `raw` schema inside the
DuckDB file (the only step that touches postgres; ~75 min for the full database). Then every graph
table is built from `raw` locally, so rebuilding any table later needs no DSN.

```bash
python -m lawgraph.build /tmp/t.duckdb --mock                        # smoke test, no postgres
python -m lawgraph.build graph.duckdb --dry-run                      # print the SQL
python -m lawgraph.build graph.duckdb --threads 32 --memory 200GB    # mirror (if raw.* missing) + build all
python -m lawgraph.build graph.duckdb --only edge_person_sits_on_court   # rebuild one table from raw (seconds)
python -m lawgraph.build graph.duckdb --remirror                     # re-pull raw.* from postgres, then build all
```

Running the full build on a file that already has `raw.*` skips the mirror and rebuilds the graph
tables only (`CREATE OR REPLACE`), which takes minutes, not an hour.

If a build errors with an unknown column in an `EXCLUDE`, a column name on your replica
differs from the mapping; trim `NOISE` / `OPINION_TEXT` or the `select=` string in `cl.py`.

## Use

```python
from lawgraph.store import Store
st = Store("graph.duckdb", read_only=True)

s = st.at(1990)                       # everything on record as of 1990-01-01
a = st.at(1990, active=True)          # only what was in force (sitting judges, open dockets...)
w = st.at(1980, until=1990)           # anything overlapping the window

s.nodes("opinion").df()               # lazy DuckDB relation
s.edges("opinion", "cites", "opinion").df()
s.index("opinion")                    # sorted ids; position = dense row index
A = s.adjacency(("opinion", "cites", "opinion"), weight="depth")   # scipy CSR (n_src x n_dst)
T = s.adjacencies()                   # {(src, rel, dst): CSR} for every relation
X = s.features("opinion", ["citation_count"])                       # aligned with index()
s.views("y1990")                      # schema of views for raw SQL: SELECT ... FROM y1990.node_docket
```

Every table has `valid_from/valid_to` (the record exists) and `active_from/active_to`
(the thing is in force). Intervals are half-open; a NULL upper bound is open-ended.
Attribute columns are CourtListener's own, inherited at build time; `st.attrs(table)` lists them.

Beware attributes that are aggregates computed by CL *today* (e.g. `citation_count`):
they leak the future into a historical snapshot. Derive such quantities from the sliced
adjacency instead.
