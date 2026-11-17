# Build spec: local deployment of the citation-space explorer

You are extending an existing, working codebase. Read this whole file, then `README.md` and `viewer/README.md`, before writing anything.

## Non-negotiables

1. **`graph.duckdb` is an input. Never open it writable, never run `python -m lawgraph.build`, never "optimize" or migrate the schema.** Every process opens it with `Store(path, read_only=True)` or `duckdb.connect(path, read_only=True)`. If you think the schema needs changing, stop and say so instead.
2. Do not rewrite `lawgraph/store.py`, `lawgraph/lens/*`, or `lawgraph/viewer_export.py` beyond what the tasks below require. They are tested. Add new modules next to them.
3. Everything must run locally with two commands: `python -m lawgraph.viewer_build ...` (precompute) and `python -m lawgraph.serve` (API + static site). No Docker, no build toolchain, no CDN dependencies at runtime (vendor any JS into `viewer/vendor/`).
4. Keep the existing viewer features working: GPU-tweened yearly animation, play/scrub, rotate/zoom/pan, search → pin → trail, co-citation neighbours, legend filter, hover, two-sentence summaries, CourtListener links.
5. Test with a headless browser (puppeteer-core + the local Chrome, or Playwright) before claiming a task is done. "It renders" means a screenshot and zero console errors.

## What exists

- `lawgraph/`: DuckDB-backed temporal heterogeneous graph over CourtListener. `Store.at(year)` is a snapshot; `select` / `build` make a symmetric co-citation graph; `Spectral`, `Leiden`, `lineage`, `align_sequence` are the lenses. Read `lawgraph/lens/base.py` and `lawgraph/viewer_build.py` to see the pipeline.
- `lawgraph/viewer_build.py`: standalone pipeline → `viewer/data/` (positions per year as Int16 decade chunks, chain ids, labels, neighbours, summaries). Currently exports only the first **3** aligned eigenvectors.
- `viewer/index.html`: three.js point cloud reading `viewer/data/`. Single file, ES module, vendored three.js.
- `notebooks/case_summaries.csv`: two-sentence summaries for ~50 label cases.
- Scale: SCOTUS cumulative co-citation at 2020 is ~27k nodes / 2.5M edges. The whole corpus is 10M clusters / 73M citations — only reachable through DuckDB, never loaded into the browser.

## Target architecture

```
graph.duckdb (read-only)
   │
   ├─ python -m lawgraph.viewer_build  →  viewer/data/      (precomputed: spectral K-dim per year, force layout per keyframe, chains, labels, neighbours)
   │
   └─ python -m lawgraph.serve         →  FastAPI on :8080
         ├─ /                          static viewer/ (index.html + data/)
         └─ /api/...                   DuckDB queries, read-only connection, 5 s statement timeout, 5k row cap
```

## Tasks, in order. Each has acceptance criteria; do not start the next until the current one passes.

### T1 — Export all K eigenvectors, not 3
Change `viewer_build.py` / `viewer_export.py` so positions are `(T, N, K)` with `K = --k-align` (default 8), quantised Int16 per dimension, still chunked by decade. Manifest gains `"K": 8` and per-dimension `centre`/`half`. The viewer's decode must take `[i, j, k]` dimension indices and build the 3-D buffers from them.
- Accept: `manifest.json` has `K`, chunk sizes scale by K/3, viewer still renders with default dims `[0,1,2]`.
- Also export per-year eigenvalues (`vals`, length K) into the manifest, so the UI can show which directions are informative.

### T2 — Eigenvector picker
UI: three dropdowns (x, y, z) listing `ev1 … evK` with the current year's eigenvalue beside each; changing one re-decodes and swaps buffers without reloading. Default `[0,1,2]`. Add a "robust set" shortcut: the three dimensions with the lowest mean principal angle across consecutive years (compute at build time with `lawgraph.lens.align.principal_angles` per dimension pair; store as `manifest["robust_dims"]`).
- Accept: switching to `[0,1,3]` on real data shows the capital-punishment split (Furman/Gregg cluster separates); screenshot both.

### T3 — Force-directed Leiden layout, precomputed
New module `lawgraph/force_layout.py`. For each Leiden keyframe year (every `--keyframe-step`, default 5):
- Build the co-citation graph (already in memory in `viewer_build`), take the Leiden partition.
- Layout in 2-D with ForceAtlas2 (`fa2` package; fall back to igraph `layout_fruchterman_reingold` with `weights`) **warm-started from the previous keyframe's positions** (new nodes start at their community's centroid plus small noise), 300 iterations first keyframe, 100 thereafter. Add a community-gravity term: each node is pulled toward its community centroid with strength `--community-gravity` (default 0.3), and community centroids repel each other (LinLog or simple inverse-square on centroids). This is what "Leiden-based force repulsion" means here.
- Align each keyframe to the previous with Procrustes (rotation + reflection + scale) before saving, so motion is real change, not solver drift.
- Export as a second position track: `force_<y0>_<y1>.bin`, Int16 `(T_keyframes, N, 2)`, listed in the manifest under `"tracks": {"spectral": {...}, "force": {...}}`. The viewer linearly interpolates between keyframes for yearly playback (z = 0).
- Accept: build time < 10 min on SCOTUS; consecutive keyframes have mean node displacement < 10% of layout radius after alignment; screenshot at 1960 and 2020.

### T4 — Projection toggle
A segmented control: **Spectral** (with the T2 pickers) | **Force (Leiden)**. Switching crossfades positions over ~600 ms (tween between the two tracks' current-year coordinates), keeping colours, selection, trail and labels. Trails are recomputed in the active track.
- Accept: toggling at 2000 with Miranda pinned keeps Miranda pinned and redraws its trail in the new track.

### T5 — Community labels by top-k, CourtListener links
Labels are already the top-degree case. Change to top-k (`--label-k`, default 3): label text is `"Miranda · Mapp · Gideon"` (first word of each case name before " v."), full names on hover, with the community's size and lineage chain id. Keep the boilerplate stoplist (`United States v. Detroit Timber`). Every case shown anywhere (pinned panel, neighbour list, year table) links to `https://www.courtlistener.com/opinion/<cluster_id>/x/`.
- Accept: label strings on real data name recognisable doctrines; links resolve.

### T6 — Year view (any court), served by the API
New module `lawgraph/serve.py` (FastAPI + uvicorn). One read-only DuckDB connection, `SET statement_timeout` style guard via `con.execute` in a thread with a 5 s timeout, hard `LIMIT 5000` on every query, parameterised SQL only (no string interpolation of user input).

Endpoints:
- `GET /api/courts` → `[{id, name, n_cases}]` from `node_court` joined to counts.
- `GET /api/cases?year=1973&court=scotus&q=roe&min_cited=10&limit=500` → cases filed that year (`node_cluster` via `edge_cluster_in_docket_docket` → `edge_docket_filed_in_court`), with `id, case_name, date_filed, court, citation_count, in_cites_to_date`. `q` is `ILIKE` on `case_name`. `min_cited` filters on cumulative citations received by `year` (count rows in `edge_cluster_cites_cluster` with `dst_id = id AND valid_from < year+1`).
- `GET /api/case/{id}` → metadata + opinions (`edge_opinion_in_cluster_cluster`) + authors (`edge_person_author_cluster` → `node_person`) + CourtListener URL.
- `GET /api/cites/{id}?direction=in|out&year=2000&limit=200` → citing or cited clusters as of `year`.
- `GET /api/bench?court=ca9&date=1990-06-01` → judges sitting (`edge_person_sits_on_court`, active interval).
- `GET /api/timeline?court=scotus&start=1900&end=2020` → cases filed and citations made per year (one query, grouped).

UI: a second page `viewer/years.html` (or a tab in `index.html`): filters (court multi-select from `/api/courts`, year slider, name search, min-citations), a table of matching cases (sortable by citations), and a small 2-D scatter of the same rows positioned by `(date_filed, log citations)`; clicking a row opens the case panel (`/api/case`). **Animate**: a play button advances the year filter one year per `secPerYear` and refreshes; for SCOTUS rows also highlight them in the main 3-D cloud if the id is present there.
- Accept: `/api/cases?year=1973&court=scotus` returns Roe v. Wade; the year view animates 1950→1980 for `court=ca9` without errors; every endpoint answers < 500 ms on cached queries and never returns > 5000 rows.

### T7 — Single-command serve
`python -m lawgraph.serve --db graph.duckdb --port 8080` mounts `viewer/` as static at `/` and the API at `/api`. Prints the URL and opens a browser unless `--no-open`. Replace `viewer/serve.py` with a stub that calls it.
- Accept: fresh shell, two commands (build, serve), browser opens, both views work, headless test passes.

## Hardware: use it
The build machine has many cores, a lot of RAM, and possibly an NVIDIA GPU. The precompute is independent across years, so:
- **Parallelise the per-year loops** in `viewer_build` (spectral per year, Leiden per keyframe, force layout per keyframe once warm-start order is respected, neighbours per decade) with `concurrent.futures.ProcessPoolExecutor`. Each worker opens its own `Store(db, read_only=True)` — DuckDB connections are not shareable across processes. Default workers = `os.cpu_count() // 2`; flag `--workers`. Force layout keyframes depend on the previous keyframe's positions, so run those in a chain but parallelise the graph builds that feed them.
- **DuckDB threads**: set `PRAGMA threads=<cores>` and `memory_limit` to ~50% of RAM on every connection, including the API server's.
- **Eigensolver**: `Spectral` already uses Lanczos on the normalised adjacency; make sure numpy/scipy are linked against a multithreaded BLAS (MKL in conda is fine) and do not oversubscribe when also running the process pool (`OMP_NUM_THREADS=2` per worker).
- **GPU, if present**: `nvidia-smi` to check. If available, use RAPIDS `cugraph.force_atlas2` for T3 (orders of magnitude faster than `fa2`) behind a `--gpu` flag with the CPU path as fallback; do not make the GPU a requirement. Do not put the force simulation in the browser regardless.
- Target: full SCOTUS build (121 yearly embeddings, 25 keyframes with Leiden + force layout, 13 neighbour decades) under 5 minutes wall-clock on the build machine. Print per-stage timings.

## Data formats (keep these exactly; the viewer depends on them)

- Positions: Int16, `(years_in_chunk, N, K)` row-major, `-32768` = absent; decode `v / 32767 * half[d] + centre[d]`.
- `chain.bin`: Uint8 `(T, N)`, 255 = none.
- `cases.json`: `[{i, id, name, year}]`, `i` = row index.
- `nbr_<decade>.bin`: Int32 `(N, k)` row indices (-1 none) then Float32 `(N, k)` weights.
- Manifest additions: `K`, `vals` (per year), `robust_dims`, `tracks`, `label_k`.

## Performance budgets
- First paint of the 3-D view on real data < 3 s on localhost (load the current decade chunk first, rest in background — already done; keep it).
- Yearly playback 60 fps with 27k points; no per-frame allocations in the render loop.
- API p95 < 500 ms; DuckDB connection is shared and read-only; no connection per request.

## Testing
- `tests/test_api.py`: pytest against a `Store.mock()` database (see `lawgraph/mock.py`) for every endpoint, including the row cap and the timeout.
- `tests/test_viewer.mjs`: puppeteer-core script (Chrome at `/usr/bin/google-chrome` or the user's path) that loads both pages, scrubs, searches, pins, toggles projection, changes eigenvectors, runs the year animation for 3 s, and fails on any console error.
- Run both before reporting done. Include the screenshots in your final message.

## What not to do
- No framework rewrite (no React/Vite). Keep `index.html` as a single ES-module page; split into `viewer/js/*.js` modules if it gets long, still vendored, still no bundler.
- No live force simulation in the browser for the full cloud; precompute (T3). A live simulation is acceptable only for the ≤ 500-row year-view scatter.
- No arbitrary SQL endpoint in this phase.
- No uploads, no cloud config, no auth. Local only. Deployment is a later phase and must not leak into this one.
