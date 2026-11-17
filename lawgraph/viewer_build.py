"""
Build the viewer data straight from graph.duckdb (no notebook needed). The database is an input: every
process here opens it read-only.

    python -m lawgraph.viewer_build graph.duckdb --out viewer/data --summaries notebooks/case_summaries.csv
    python -m lawgraph.viewer_build graph.duckdb --out viewer/data --court scotus --start 1900 --end 2020 \\
        --keyframe-step 5 --pos-step 1 --resolution 1.0 --min-degree 5 --k-align 8 --label-min 300 --workers 64 \\
        --gpu --community-gravity 0.3 --label-k 3

Steps
  1. per year, in a process pool (each worker owns a read-only Store): cumulative co-citation graph,
     spectral embedding every --pos-step years, Leiden partition every --keyframe-step years
  2. lineage chains over the keyframes (colours), chained Procrustes alignment of the embeddings, all
     --k-align aligned eigenvectors kept as positions (T, N, K), per-year eigenvalues, and the three
     dimensions that move least from year to year (robust_dims)
  3. force-directed layout of every Leiden keyframe (lawgraph.force_layout): ForceAtlas2 with community
     gravity and centroid repulsion, warm-started and Procrustes-aligned along the chain of keyframes;
     --gpu runs it on RAPIDS cuGraph, otherwise the fa2 package (or igraph) on one CPU thread
  4. top-10 co-citation neighbours per decade (same pool), labels, then export_viewer()
Per-stage wall-clock timings are printed at the end.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from . import hw

KEY = ("cluster", "cites", "cluster")
BOILERPLATE = {"united states v. detroit timber"}      # cited in every syllabus; never a doctrinal label
COURT_JOIN = ("JOIN edge_cluster_in_docket_docket cd ON cd.src_id = n.id "
              "JOIN edge_docket_filed_in_court d ON d.src_id = cd.dst_id")
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
SPECTRAL_K = 12                                        # eigenvectors solved per year; the first --k-align are kept


def is_boilerplate(name: str) -> bool:
    n = str(name).lower()
    return any(n.startswith(b) for b in BOILERPLATE)


# A label names a community by its top cases: "Miranda v. Arizona" -> "Miranda". The rule is the first word
# of the name before " v."; the lists below only repair the names that rule would make meaningless.
_SOVEREIGN = re.compile(r"^(united states|state|states|commonwealth|people|the people|territory)( of .*| ex rel\.? .*)?$", re.I)
_LEAD = re.compile(r"^(the|ex parte|in re|in the matter of|matter of)\s+", re.I)
_WEAK = {"bank", "board", "city", "county", "town", "state", "new", "national", "first", "second", "third", "standard",
         "general", "american", "united", "union", "great", "northern", "southern", "eastern", "western", "central",
         "pacific", "atlantic", "st", "saint", "san", "los", "des", "fort", "mount", "lake", "north", "south", "east", "west"}
_JOIN = {"of", "the", "and", "&", "for", "de", "la"}


def short_name(case_name: str) -> str:
    """'Miranda v. Arizona' -> 'Miranda'; 'United States v. Darby' -> 'Darby' (a sovereign names nothing);
    'Ex Parte Young' -> 'Young'; 'New York Times Co. v. Sullivan' -> 'New York' (a lone weak first word such
    as New / Bank / National, or an initial, is extended to the next real word)."""
    name = " ".join(str(case_name).split())
    first, sep, second = name.partition(" v. ")
    party = _LEAD.sub("", first).strip()
    if sep and _SOVEREIGN.match(party) and second.strip():
        party = _LEAD.sub("", second).strip()
    words = party.split()
    if not words:
        return name[:20]
    out = [words[0]]
    for w in words[1:]:
        last = out[-1].lower().strip(",.")
        if not (last in _WEAK or last in _JOIN or len(last) == 1):    # a lone initial ("A. L. A. Schechter") is weak too
            break
        out.append(w)
    while len(out) > 1 and out[-1].lower().strip(",.") in _JOIN:
        out.pop()
    return " ".join(out).strip(" ,;:")


def community_label(candidates, names, k: int):
    """Top-k label for one community. candidates: member ids by descending degree; names: id -> case name.
    Boilerplate hubs are skipped, and so is a case whose short name is already in the label (duplicate
    cluster rows, 'Employers' Liability Cases' twice). Returns (text, full names, ids)."""
    shorts, full, ids = [], [], []
    for cid in candidates:
        nm = names.get(int(cid))
        if nm is None or nm != nm or is_boilerplate(nm):
            continue
        sn = short_name(nm)
        if not sn or sn.lower() in (x.lower() for x in shorts):
            continue
        shorts.append(sn); full.append(str(nm)); ids.append(int(cid))
        if len(shorts) == k:
            break
    return " · ".join(shorts), full, ids


# ---------------------------------------------------------------- worker side
_ST = None           # one read-only Store per worker process (DuckDB connections cannot cross processes)
_BLAS = None


def _init_worker(db: str, duck_threads: int, duck_mem_gb: float, blas_threads: int) -> None:
    global _ST, _BLAS
    from .store import Store
    try:                                              # the env vars set by the parent already cap BLAS; this
        from threadpoolctl import threadpool_limits   # also covers a BLAS that ignores them
        _BLAS = threadpool_limits(limits=blas_threads)
    except ImportError:
        pass
    _ST = Store(db, read_only=True)
    hw.tune(_ST.con, duck_threads, duck_mem_gb)


def court_ids(snap, court: str) -> np.ndarray:
    """Published decisions of `court` present in the snapshot."""
    from .lens import select
    return select(snap, "cluster", where=f"d.dst_id = '{court}' AND n.precedential_status = 'Published'",
                  join=COURT_JOIN)


def _year_task(y: int, court: str, min_degree: int, k_align: int, embed: bool, leiden: bool, resolution: float) -> dict:
    """One cumulative slice: co-citation graph, then (optionally) its spectral embedding and Leiden partition.
    The graph itself is only shipped back for keyframes (labels and the force layout need it)."""
    from .lens import Leiden, Spectral, build
    t0 = time.time()
    s = _ST.at(y)
    G = build(s, KEY, court_ids(s, court), mode="cocite", min_degree=min_degree)
    out = {"year": y, "n": G.n, "emb": None, "part": None, "G": None}
    if embed and G.n >= k_align + 2:
        out["emb"] = Spectral(k=max(SPECTRAL_K, k_align))(G)
    if leiden and G.n >= 50:
        out["part"], out["G"] = Leiden(resolution=resolution)(G), G
    out["secs"] = time.time() - t0
    return out


def _nbr_task(decade: int, ids: np.ndarray, k: int):
    from .viewer_export import cocitation_neighbors
    return cocitation_neighbors(lambda y: _ST.at(y), [decade], ids, k=k)


# ---------------------------------------------------------------- alignment diagnostics
def canonical_signs(reference: np.ndarray) -> np.ndarray:
    """One sign per dimension so that a rebuild draws the same picture: eigenvectors are only defined up to
    sign, and the solver's choice is arbitrary. Each column of the reference slice is oriented so its long
    tail (sum of cubes) points to +."""
    s = np.sign((np.asarray(reference, dtype=float) ** 3).sum(axis=0))
    return np.where(s == 0, 1.0, s)


def dimension_stability(embs: list, aligned: list[np.ndarray]) -> np.ndarray:
    """Mean principal angle (radians) between consecutive slices, one value per aligned dimension: the angle
    between dimension d at t and at t+1 as 1-D subspaces over the nodes both slices share. Small = the axis
    means the same thing from year to year."""
    from .lens.align import principal_angles
    K = aligned[0].shape[1]
    ang = np.zeros((max(len(embs) - 1, 1), K))
    for t in range(len(embs) - 1):
        _, ia, ib = np.intersect1d(embs[t].ids, embs[t + 1].ids, assume_unique=True, return_indices=True)
        if len(ia) < 2:
            ang[t] = np.pi / 2; continue
        A, B = aligned[t][ia], aligned[t + 1][ib]
        for d in range(K):
            ang[t, d] = principal_angles(A[:, [d]], B[:, [d]])[0]
    return ang.mean(axis=0)


# ---------------------------------------------------------------- main
def default_workers() -> int:
    """Half the cores, but no more than RAM allows: a late slice of the full corpus peaks near 7 GB per worker."""
    return max(1, min(hw.cores() // 2, int(hw.ram_gb() // 8)))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("db"); ap.add_argument("--out", default="viewer/data")
    ap.add_argument("--court", default="scotus"); ap.add_argument("--start", type=int, default=1900); ap.add_argument("--end", type=int, default=2020)
    ap.add_argument("--keyframe-step", type=int, default=5); ap.add_argument("--pos-step", type=int, default=1)
    ap.add_argument("--resolution", type=float, default=1.0); ap.add_argument("--min-degree", type=int, default=5)
    ap.add_argument("--k-align", type=int, default=8, help="aligned eigenvectors exported per year (K)")
    ap.add_argument("--label-min", type=int, default=300); ap.add_argument("--max-labels", type=int, default=14)
    ap.add_argument("--label-k", type=int, default=3, help="top cases named in each community label")
    ap.add_argument("--summaries", default=None); ap.add_argument("--no-neighbors", action="store_true")
    ap.add_argument("--gpu", action="store_true", help="run the force layout on the GPU (RAPIDS cuGraph); falls back to CPU if unavailable")
    ap.add_argument("--no-force", action="store_true", help="skip the force-layout track")
    ap.add_argument("--community-gravity", type=float, default=0.3, help="pull of each node towards its Leiden community's centroid")
    ap.add_argument("--community-repulsion", type=float, default=0.1, help="inverse-square repulsion between community centroids")
    ap.add_argument("--force-anchor", type=float, default=0.6, help="pull of each node towards its position in the previous keyframe (0 = off)")
    ap.add_argument("--workers", type=int, default=None, help="worker processes (default: cores // 2, capped by RAM)")
    ap.add_argument("--blas-threads", type=int, default=2, help="OMP/BLAS threads per worker")
    a = ap.parse_args(argv)
    if not re.fullmatch(r"[\w.-]+", a.court):
        ap.error(f"--court must be a court id like 'scotus' or 'ca9', got {a.court!r}")
    if a.label_k < 1:
        ap.error("--label-k must be at least 1")
    if a.k_align < 3:
        ap.error("--k-align must be at least 3 (the viewer shows three dimensions at a time)")
    if not os.path.exists(a.db):
        ap.error(f"database not found: {a.db}")
    a.workers = a.workers or default_workers()
    return a


def main(argv=None):
    a = parse_args(argv)
    import pandas as pd
    from .lens import align_sequence, lineage
    from .store import Store
    from .viewer_export import export_viewer

    timings: list[tuple[str, float]] = []
    t_all = time.time()
    cores, ram = hw.cores(), hw.ram_gb()
    print(f"hardware: {cores} cores, {ram:.0f} GB RAM, GPU: {hw.gpu() or 'none'}", flush=True)
    print(f"workers: {a.workers} x {a.blas_threads} BLAS threads, DuckDB {max(1, cores // a.workers)} threads each", flush=True)

    st = Store(a.db, read_only=True); con = hw.tune(st.con)

    def names(ids):
        df = con.execute("SELECT id, case_name, year(date_filed) AS year FROM node_cluster WHERE id IN (SELECT unnest(?))",
                         [list(map(int, ids))]).df().set_index("id")
        return df.reindex(ids)

    key_years = set(range(a.start, a.end + 1, a.keyframe_step))
    pos_years = set(range(a.start, a.end + 1, a.pos_step))

    # workers inherit these at spawn; the parent's BLAS is already initialised and keeps its own setting
    saved_env = {v: os.environ.get(v) for v in THREAD_VARS}
    os.environ.update({v: str(a.blas_threads) for v in THREAD_VARS})
    pool = ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn"), initializer=_init_worker,
                               initargs=(os.path.abspath(a.db), max(1, cores // a.workers),
                                         max(2.0, ram * 0.5 / a.workers), a.blas_threads))
    try:
        # 1. graphs, embeddings and partitions; latest (largest) slices first so the long jobs start early
        t0 = time.time(); res = {}
        todo = sorted(key_years | pos_years, reverse=True)
        futs = [pool.submit(_year_task, y, a.court, a.min_degree, a.k_align, y in pos_years, y in key_years, a.resolution)
                for y in todo]
        for n, f in enumerate(as_completed(futs), 1):
            r = f.result(); res[r["year"]] = r
            if r["part"] is not None:
                print(f"keyframe {r['year']}: {r['n']:,} nodes, {r['part'].k} communities   ({time.time()-t0:.0f}s)", flush=True)
            elif n % 20 == 0:
                print(f"  {n}/{len(todo)} slices   ({time.time()-t0:.0f}s)", flush=True)
        graphs = {y: r["G"] for y, r in res.items() if r["G"] is not None}
        parts = {y: r["part"] for y, r in res.items() if r["part"] is not None}
        embs = {y: r["emb"] for y, r in res.items() if r["emb"] is not None}
        if not embs or not parts:
            sys.exit(f"nothing to export: no slice of court {a.court!r} in {a.start}-{a.end} is large enough")
        timings.append((f"graphs + spectral + leiden ({len(todo)} slices)", time.time() - t0))

        ys = sorted(parts); years_pos = sorted(embs)
        all_ids = np.unique(np.concatenate([embs[y].ids for y in years_pos])); N = len(all_ids); T = len(years_pos)
        K = a.k_align

        # neighbours run in the pool while the parent aligns and labels
        nbr_futs = []; t_nbr = time.time()
        if not a.no_neighbors:
            decades = list(range((a.start // 10) * 10, a.end + 1, 10))
            nbr_futs = [pool.submit(_nbr_task, d, all_ids, 10) for d in decades]

        # 2. lineage chains, alignment, positions (T, N, K), eigenvalues
        t0 = time.time()
        lin = lineage([parts[y] for y in ys], min_overlap=0.1)
        cid = {}
        for n, ch in enumerate(lin.chains(min_jaccard=0.2)):
            for k in ch: cid[k] = n
        lchain = {y: {c: cid.get((i, c), -1) for c in range(parts[y].k)} for i, y in enumerate(ys)}

        emb_seq = [embs[y] for y in years_pos]
        al_seq = align_sequence(emb_seq, k=K, reference="last")
        sgn = canonical_signs(al_seq[-1])                # same reflection for every year, so alignment is kept
        al_seq = [A * sgn for A in al_seq]
        al = dict(zip(years_pos, al_seq))
        dim_angles = dimension_stability(emb_seq, al_seq)
        robust_dims = sorted(int(d) for d in np.argsort(dim_angles, kind="stable")[:3])
        print("mean principal angle between consecutive years, per dimension (deg): "
              + " ".join(f"ev{d+1}={np.degrees(v):.2f}" for d, v in enumerate(dim_angles))
              + f"   robust set: {[f'ev{d+1}' for d in robust_dims]}", flush=True)
        pos = np.full((T, N, K), np.nan, dtype=np.float32)
        vals = np.full((T, K), np.nan); vals_al = np.full((T, K), np.nan)
        for i, y in enumerate(years_pos):
            e = embs[y]
            pos[i, np.searchsorted(all_ids, e.ids)] = al[y][:, :K]
            lam = e.vals[:K]; vals[i, :len(lam)] = lam
            R = e.vecs[:, :K].T @ al[y]                  # the year's cumulative rotation (eigenvectors are orthonormal)
            vals_al[i] = (R ** 2).T @ lam                # Rayleigh quotient of each displayed direction

        def rows(ids):
            """Buffer rows of `ids`, and which of them have a row at all (a keyframe that is not also a
            position year can hold a case no position year has)."""
            r = np.searchsorted(all_ids, ids); ok = r < N
            ok[ok] &= all_ids[r[ok]] == np.asarray(ids)[ok]
            return r, ok

        # chain id per case per year (forward-filled from keyframes) and palette
        keyframe_of = {y: max([k for k in ys if k <= y], default=None) for y in years_pos}
        chain_year = np.full((T, N), -1, dtype=np.int32)
        for i, y in enumerate(years_pos):
            ky = keyframe_of[y]
            if ky is None: continue
            r, ok = rows(parts[ky].ids)
            chain_year[i, r[ok]] = np.array([lchain[ky][c] for c in parts[ky].labels])[ok]
        from matplotlib import colormaps
        magma = colormaps["magma"]
        chains_all = sorted({c for y in ys for c in lchain[y].values() if c >= 0})
        perm = np.random.default_rng(0).permutation(len(chains_all))
        palette = {ch: list(magma(0.25 + 0.7 * perm[i] / max(len(chains_all) - 1, 1))[:3]) for i, ch in enumerate(chains_all)}
        timings.append(("lineage + alignment", time.time() - t0))

        # 3. force layout of the keyframes: a second position track (T_keyframes, N, 2)
        force = None; fpos = {}
        if not a.no_force:
            from .force_layout import ForceParams, layout_sequence, pick_backend
            t0 = time.time()
            if not a.gpu and hw.gpu():
                print("force layout: a GPU is present; add --gpu to run this stage on it (seconds instead of minutes)", flush=True)
            fp = ForceParams(community_gravity=a.community_gravity, community_repulsion=a.community_repulsion, anchor=a.force_anchor)
            backend = pick_backend(a.gpu)
            print(f"force layout: {backend.name}, {len(ys)} keyframes", flush=True)
            fpos, fstats = layout_sequence([(y, graphs[y].W, graphs[y].ids, parts[y].labels) for y in ys], fp, backend,
                                           log=lambda m: print("  " + m, flush=True))
            track = np.full((len(ys), N, 2), np.nan, dtype=np.float32)
            for i, y in enumerate(ys):
                r, ok = rows(graphs[y].ids)
                track[i, r[ok]] = fpos[y][ok]
            disp = fstats["displacement"]; worst = max(disp.values(), default=0.0)
            print(f"force layout: mean displacement between consecutive keyframes {100 * np.mean(list(disp.values()) or [0]):.1f}% "
                  f"of the layout radius, worst {100 * worst:.1f}%" + ("" if worst < 0.10 else "   ** above the 10% target **"), flush=True)
            force = {"years": ys, "pos": track, "backend": fstats["backend"], "displacement": {str(y): round(v, 4) for y, v in disp.items()},
                     "radius": {str(y): float(v) for y, v in fstats["radius"].items()}, "params": fstats["params"]}
            timings.append((f"force layout ({len(ys)} keyframes, {fstats['backend']})", time.time() - t0))

        # 4. labels per keyframe: the community's top-k members by co-citation degree (boilerplate hubs
        #    skipped), anchored at its median position in each track
        t0 = time.time()
        n_cand = a.label_k + 8                           # spare candidates for skipped boilerplate / duplicates
        cand = {}                                        # (keyframe, community) -> member rows by descending degree
        for ky in ys:
            if ky not in embs: continue
            G, p = graphs[ky], parts[ky]; deg = G.degree
            for c in range(p.k):
                m = np.where(p.labels == c)[0]
                if len(m) >= a.label_min:
                    cand[(ky, c)] = m[np.argsort(-deg[m], kind="stable")]
        top_ids = np.unique(np.concatenate([graphs[ky].ids[o[:n_cand]] for (ky, c), o in cand.items()])) if cand else np.array([], dtype=int)
        top_names = {int(i): n for i, n in names(top_ids).case_name.items()} if len(top_ids) else {}
        labels_year = {}
        for ky in ys:
            if ky not in embs: continue
            i = years_pos.index(ky); G, p = graphs[ky], parts[ky]; found = []
            for c in range(p.k):
                order = cand.get((ky, c))
                if order is None: continue
                text, full, lids = community_label(G.ids[order[:n_cand]], top_names, a.label_k)
                pp = pos[i, np.searchsorted(all_ids, G.ids[order])]; pp = pp[~np.isnan(pp[:, 0])]
                if text and len(pp):
                    lab = {"name": text, "full": full, "ids": lids, "size": int(len(order)),
                           "p": np.median(pp, axis=0), "chain": lchain[ky][c]}
                    if ky in fpos:                                  # the same community's anchor in the force track
                        lab["pf"] = np.median(fpos[ky][order], axis=0)
                    found.append((len(order), lab))
            labels_year[ky] = [r[1] for r in sorted(found, key=lambda r: -r[0])[:a.max_labels]]
        timings.append(("labels", time.time() - t0))

        nbrs = None
        if nbr_futs:
            nbrs = pd.concat([f.result() for f in nbr_futs], ignore_index=True)
            print(f"neighbours: {len(nbrs):,} rows", flush=True)
            timings.append((f"neighbours ({len(nbr_futs)} decades, overlaps the stages above)", time.time() - t_nbr))
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        for v, old in saved_env.items():
            if old is None: os.environ.pop(v, None)
            else: os.environ[v] = old

    # 5. export
    t0 = time.time()
    summ = pd.read_csv(a.summaries) if a.summaries else None
    export_viewer(a.out, years_pos, all_ids, pos, chain_year, palette, names(all_ids), labels_year,
                  neighbors=nbrs, summaries=summ, vals=vals, vals_aligned=vals_al,
                  robust_dims=robust_dims, dim_angles=dim_angles, force=force, label_k=a.label_k)
    timings.append(("export", time.time() - t0))

    print("\nstage timings (wall-clock)")
    for name, secs in timings:
        print(f"  {secs:7.1f} s  {name}")
    print(f"  {time.time() - t_all:7.1f} s  total", flush=True)


if __name__ == "__main__":
    main()
