# Citation-space explorer

Two pages, served locally by one process:

- **`index.html`** — three.js point cloud of every SCOTUS decision, tweened year by year on the GPU and coloured by
  Leiden lineage chain. Two projections: **Spectral** (any three of the K aligned Laplacian eigenvectors) and
  **Force (Leiden)** (a precomputed force-directed layout of the Leiden communities). Search, pin, trails,
  co-citation neighbours, community labels, legend filter.
- **`years.html`** — cases filed in one year in *any* court, straight from DuckDB through the API: filters, a sortable
  table, a (date filed, log citations) scatter, a decision panel, and a year-by-year animation. SCOTUS rows light
  up in the 3-D cloud.

No build toolchain, no CDN (three.js is vendored in `vendor/`), no Docker. `graph.duckdb` is an input: every process
opens it read-only.

## Quick start

    python -m lawgraph.viewer_build graph.duckdb --out viewer/data --summaries notebooks/case_summaries.csv --gpu
    python -m lawgraph.serve --db graph.duckdb --port 8080

The second command prints the URL and opens a browser (`--no-open` to skip). `python viewer/serve.py` still works
and does the same thing. Drop `--gpu` on a machine without an NVIDIA GPU (see *Hardware* below).

## Controls

3-D view
- drag to rotate, wheel to zoom, right-drag to pan; slider or **space** to play; `faster` / `slower` change seconds per year
- **Spectral | Force (Leiden)** switches projection with a 600 ms crossfade; colours, the pinned case, its trail (redrawn
  in the new projection) and the labels carry over
- **x / y / z** dropdowns pick which eigenvectors are drawn; the number beside each is that direction's eigenvalue at the
  current year. Picking an axis that is in use swaps the two. **robust set** selects the three dimensions that rotate
  least from year to year. Nothing is fetched again: the page keeps all K dimensions in memory
- type in the search box and click a result: the case is pinned, its trail through time is drawn, its neighbours listed
- labels name each large community by its top three cases; hover one for the full names, the community's size and its
  lineage chain; click one to pin its top case
- click a legend entry to isolate a lineage; **Esc** clears the selection
- **Year view** opens `years.html` in a drawer; while it is open the table's SCOTUS rows are highlighted in the cloud
  (the last ten years while animating) and the cloud follows to the year after the table's year
- deep link: `index.html?year=1973&case=108713&track=force&dims=0,1,3` (`case` is a CourtListener cluster id)

Year view
- tick courts (none ticked = every court), move the year slider, filter by name, set a minimum citation count
- click a column to sort, a row or a dot for the decision panel; **Animate** advances one year per `s / yr`
- every case shown anywhere links to `https://www.courtlistener.com/opinion/<cluster_id>/x/`

## What the build computes

`python -m lawgraph.viewer_build graph.duckdb --out viewer/data [flags]`

1. For every year `--start..--end` (default 1900–2020), in a process pool: the cumulative co-citation graph of the
   court's published decisions (`--court scotus`, `--min-degree 5`), its regularised spectral embedding, and at every
   `--keyframe-step` (5) years its Leiden partition (`--resolution 1.0`).
2. Lineage chains across the keyframes (colours); chained Procrustes alignment of the embeddings to the last year; all
   `--k-align` (8) aligned eigenvectors are exported, not three. Per year the eigenvalues (`vals`) and the Rayleigh
   quotient of each *displayed* direction (`vals_aligned`: after alignment an early year's axis is a mixture of that
   year's eigenvectors, so this is the number the picker shows). `robust_dims` are the three dimensions with the
   smallest mean principal angle between consecutive years. Eigenvector signs are fixed so a rebuild draws the same picture.
3. Force layout (`lawgraph/force_layout.py`): each keyframe is laid out in 2-D with ForceAtlas2 on log-weighted
   co-citation, 300 iterations for the first keyframe and 100 for each later one, warm-started from the previous
   keyframe (new cases start at their community's centroid plus noise). Every 5 solver iterations three forces are
   applied: each case moves `--community-gravity` (0.3) of the way to its community centroid; community centroids repel
   each other by an inverse-square law (`--community-repulsion` 0.1); and each case that already existed moves
   `--force-anchor` (0.6) of the way back to where it was. Each keyframe is then Procrustes-aligned (rotation,
   reflection, scale, translation) to the previous one.
   The anchor is an addition to the original recipe: without it, five years of new citations move an early keyframe's
   *converged* layout by 15–30% of its radius (measured; the solver itself drifts about 1%), which plays back as churn.
   `--force-anchor 0` gives the bare recipe. On SCOTUS the mean displacement between consecutive keyframes is 5% of the
   RMS layout radius and the worst keyframe 7%; the build prints these and stores them in the manifest.
4. Community labels: the top `--label-k` (3) cases of each community of at least `--label-min` (300) by co-citation
   degree, as short names (`Miranda v. Arizona` → `Miranda`; a sovereign first party is skipped, `Ex parte` / `In re` /
   `The` dropped, a lone `New` / `Bank` / `National` extended to the next word). `United States v. Detroit Timber`,
   cited in every syllabus, is never a label.
5. Top-10 co-citation neighbours per case per decade, and the two-sentence summaries from `--summaries`.

Other flags: `--pos-step`, `--max-labels`, `--no-neighbors`, `--no-force`, `--workers`, `--blas-threads`. The build
prints per-stage wall-clock timings.

### Hardware

- Years are independent, so step 1 runs in `--workers` processes (default: half the cores, capped at one per 8 GB of RAM;
  a late slice of the full corpus peaks near 7 GB). Each worker opens its own read-only connection, with DuckDB
  threads = cores / workers and 2 BLAS threads (`--blas-threads`). Neighbours run in the same pool while the parent
  aligns and lays out.
- Every DuckDB connection, the API server's included, gets `PRAGMA threads` and a memory limit of half the RAM, and spills
  to the system temp directory, never next to the database.
- `--gpu` runs the force layout on RAPIDS cuGraph (`pylibcugraph.force_atlas2`, the routine behind
  `cugraph.force_atlas2`): about 5 s for the 25 SCOTUS keyframes. Without it the `fa2` package does the same on one
  CPU thread in about 6 min (its Barnes–Hut opening angle is raised from 1.2 to 2.0 for that; at 1.2 it takes 13 min).
  If neither cuGraph nor `fa2` can be imported the build falls back to igraph's Fruchterman–Reingold.
- Measured on 128 cores / 1 TB RAM / RTX PRO 6000: full SCOTUS build 75 s with `--gpu`, about 7 min without.

## Data files (`viewer/data/`)

| file | what |
|---|---|
| `manifest.json` | `years`, `N`, `K`, per-dimension `centre` / `half`, `chunks`, `vals`, `vals_aligned`, `robust_dims`, `dim_angles`, `tracks`, `label_k`, `palette`, `labels`, `summaries`, `neighbors` |
| `pos_<y0>_<y1>.bin` | spectral track: Int16 `(years_in_chunk, N, K)` row-major, one chunk per decade; `-32768` = absent; decode `v / 32767 * half[d] + centre[d]` |
| `force_<y0>_<y1>.bin` | force track: Int16 `(keyframes_in_chunk, N, 2)`, same quantisation, keyframe years only; the viewer interpolates linearly in between, z = 0 |
| `chain.bin` | Uint8 `(T, N)` lineage chain id per (year, case), 255 = none |
| `cases.json` | `[{i, id, name, year}]`, `i` = row index, `id` = CourtListener cluster id |
| `nbr_<decade>.bin` | Int32 `(N, k)` neighbour row indices (-1 none) then Float32 `(N, k)` weights |

`manifest.tracks = {"spectral": {K, years, centre, half, chunks}, "force": {K: 2, years, centre, half, chunks, backend,
displacement, radius, params}}`; the top-level `years / centre / half / chunks` repeat the spectral track. A label is
`{name, full, ids, size, chain, p (K-dim anchor), pf (force anchor)}`. Real SCOTUS data: 27,157 cases × 121 years × 8
dimensions → 53 MB of spectral positions (4.3 MB per decade), 2.7 MB force track, 3 MB chain, 2 MB per neighbour decade.

The page loads the manifest, the case index and the current decade first (first paint about 0.4 s on localhost) and
fetches the remaining decades and the force track in the background.

## API (`python -m lawgraph.serve`)

One read-only DuckDB connection shared by all requests; every statement is fixed SQL with bound parameters, runs
under a 5 s limit (interrupted → HTTP 504), and is wrapped in `LIMIT 5000`. Results are cached in memory. The server
binds to 127.0.0.1 only. Interactive docs at `/api/docs`.

| endpoint | returns |
|---|---|
| `GET /api/courts` | `[{id, name, n_cases}]` |
| `GET /api/cases?year=1973&court=scotus&q=roe&min_cited=10&limit=500` | cases filed that year: `id, case_name, date_filed, court, citation_count, in_cites_to_date`. `court` may repeat or be comma-separated; `q` is `ILIKE`; `in_cites_to_date` / `min_cited` count citations received up to the end of that year |
| `GET /api/case/{id}` | metadata, opinions, authors, CourtListener URL |
| `GET /api/cites/{id}?direction=in\|out&year=2000&limit=200` | citing or cited decisions, as of the end of `year` |
| `GET /api/bench?court=ca9&date=1990-06-01` | judges sitting that day |
| `GET /api/timeline?court=scotus&start=1900&end=2020` | cases filed and citations made per year |

`citation_count` is CourtListener's present-day total; `in_cites_to_date` is derived from the citation edges and does
not leak the future.

## Tests

    python -m pytest tests                    # API on the mock database; force layout, export and label helpers
    cd tests && npm install && cd ..          # once: puppeteer-core
    node tests/test_viewer.mjs                # headless Chrome on the real data; starts the server itself

`test_viewer.mjs` loads both pages, scrubs, plays, searches, pins, toggles the projection, changes eigenvectors, hovers
a label, runs the year animation for 3 s, checks the highlight link between the pages, writes screenshots to
`tests/screenshots/` and fails on any console error, failed request or HTTP error. `CHROME=/path/to/chrome` selects the
browser (it looks for google-chrome, chromium, a Playwright Chromium, Brave), `PYTHON=` the interpreter,
`--url http://127.0.0.1:8080` an already running server, `GL=gpu` hardware rendering.

## Static hosting

`index.html`, `js/`, `vendor/` and `data/` are plain static files, so the 3-D view also works from any static server;
the year view needs the API. Hosting anywhere but localhost is a later phase and has not been set up or tested here.
