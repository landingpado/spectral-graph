"""
Export the temporal embedding + communities to the static files the viewer reads.

    from lawgraph.viewer_export import export_viewer, cocitation_neighbors
    export_viewer("viewer/data", years_pos, all_ids, pos, chain_year, chain_palette, meta, labels_year,
                  neighbors=neighbors_df, vals=vals, robust_dims=[0, 1, 3], force={"years": ..., "pos": ...})

Writes
    manifest.json          years, N, K, quantisation, chunk list, eigenvalues, tracks, palette, labels, summaries
    pos_<y0>_<y1>.bin      Int16 spectral positions, shape (years_in_chunk, N, K) row-major, one chunk per decade;
                           -32768 = absent; decode  v / 32767 * half[d] + centre[d]
    force_<y0>_<y1>.bin    Int16 force-layout positions at the Leiden keyframes, shape (keyframes_in_chunk, N, 2),
                           same quantisation (optional second track; the viewer interpolates between keyframes)
    chain.bin              Uint8 chain id per (year, case); 255 = no chain
    cases.json             [{i, id, name, year}]  (i = row index into the buffers)
    nbr_<decade>.bin       Int32 (N, k) row indices of top-k co-citation neighbours (-1 = none) followed by
                           Float32 (N, k) weights; one file per decade (optional)

Positions must already be aligned (align_sequence) so that motion between years is meaningful.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

CHUNK_YEARS = 10
_OWNED = re.compile(r"^(?:(?:pos|force)_\d+_\d+|nbr_\d+)\.bin$")      # files this exporter writes and may replace


def _quantise(pos: np.ndarray):
    """pos (T, N, K) float with NaN for absent -> Int16 and the per-dimension scale/offset to invert it.
    NaN -> -32768 sentinel."""
    finite = np.isfinite(pos[..., 0])
    lo = np.nanmin(pos, axis=(0, 1)); hi = np.nanmax(pos, axis=(0, 1))
    centre = (lo + hi) / 2; half = np.maximum((hi - lo) / 2, 1e-12)
    q = np.round((pos - centre) / half * 32767).astype(np.float64)
    q = np.where(finite[..., None], q, -32768).astype(np.int16)
    return q, centre.tolist(), half.tolist()


def write_track(out: str, prefix: str, years, pos: np.ndarray, chunk: int = CHUNK_YEARS) -> dict:
    """Quantise one position track (T, N, K) and write it as Int16 chunks '<prefix>_<y0>_<y1>.bin', `chunk`
    frames per file. Returns the track's manifest entry: K, years, centre, half (per dimension), chunks."""
    years = [int(y) for y in years]; T = len(years)
    pos = np.asarray(pos)
    assert pos.ndim == 3 and pos.shape[0] == T, f"{prefix}: expected ({T}, N, K), got {pos.shape}"
    q, centre, half = _quantise(pos)
    chunks = []
    for a in range(0, T, chunk):
        b = min(a + chunk, T)
        fn = f"{prefix}_{years[a]}_{years[b-1]}.bin"
        q[a:b].tofile(os.path.join(out, fn))
        chunks.append({"file": fn, "years": years[a:b]})
    return {"K": int(pos.shape[2]), "years": years, "centre": centre, "half": half, "chunks": chunks}


def _label(r) -> dict:
    """One label anchor. Tuples (name, x, y, z[, chain_id]) are the original format; dicts pass through
    (name, p = K-dim anchor, chain, and optionally pf = force-track anchor, full, ids, size)."""
    if isinstance(r, dict):
        d = dict(r)
        d["p"] = [float(v) for v in d["p"]]
        if d.get("pf") is not None:
            d["pf"] = [float(v) for v in d["pf"]]
        ch = d.get("chain")
        d["chain"] = int(ch) if ch is not None and ch >= 0 else None
        return d
    return {"name": r[0], "p": [float(r[1]), float(r[2]), float(r[3])],
            "chain": (int(r[4]) if len(r) > 4 and r[4] is not None and r[4] >= 0 else None)}


def _rows(a) -> list | None:
    """(T, K) float array -> JSON rows with None for NaN (a year can have fewer than K eigenvalues)."""
    if a is None:
        return None
    return [[(float(v) if v == v else None) for v in row] for row in np.asarray(a, dtype=float)]


def export_viewer(out: str, years, ids, pos3, chain_year, palette, meta, labels_year=None,
                  neighbors=None, con=None, summaries=None, verbose=True, *,
                  vals=None, vals_aligned=None, robust_dims=None, dim_angles=None, force=None, label_k=1):
    """
    years        list of ints, one per frame (yearly is what the viewer expects)
    ids          (N,) cluster ids, row order of the buffers
    pos3         (T, N, K) aligned coordinates, K >= 3 (all K aligned eigenvectors; the viewer picks three),
                 NaN where the case is absent at that year
    chain_year   (T, N) int chain id per case per year, -1 for none
    palette      dict chain_id -> [r, g, b] in 0..1
    meta         DataFrame indexed by cluster id with columns case_name, year  (names(ids) from the notebook)
    labels_year  dict year -> list of label anchors for that year: (name, x, y, z[, chain_id]) or dicts (optional)
    neighbors    DataFrame (decade, id, nbr_id, w) from cocitation_neighbors (optional)
    summaries    DataFrame with name_key, case_name, year, citation, summary (optional)
    vals         (T, K) eigenvalues of each year's own Laplacian, ascending (optional)
    vals_aligned (T, K) Rayleigh quotient of each *displayed* (aligned) direction at each year (optional)
    robust_dims  three dimension indices that are most stable from year to year (optional)
    dim_angles   (K,) mean principal angle between consecutive years, per dimension, radians (optional)
    force        {"years": keyframe years, "pos": (T_key, N, 2), ...extra manifest fields} second track (optional)
    label_k      how many top cases make up a community label
    """
    os.makedirs(out, exist_ok=True)
    years = [int(y) for y in years]; T, N = len(years), len(ids)
    pos3 = np.asarray(pos3)
    assert pos3.ndim == 3 and pos3.shape[:2] == (T, N) and pos3.shape[2] >= 3 and chain_year.shape == (T, N)
    K = int(pos3.shape[2])

    tracks = {"spectral": write_track(out, "pos", years, pos3)}
    if force is not None:
        extra = {k: v for k, v in force.items() if k not in ("years", "pos")}
        fpos = np.asarray(force["pos"])
        assert fpos.shape[1:] == (N, 2), f"force track must be (T_keyframes, {N}, 2), got {fpos.shape}"
        tracks["force"] = {**write_track(out, "force", force["years"], fpos), **extra}
    sp = tracks["spectral"]

    ch = np.where(chain_year < 0, 255, chain_year).astype(np.uint8)
    ch.tofile(os.path.join(out, "chain.bin"))

    rows = []
    for i, cid in enumerate(ids):
        r = meta.loc[cid] if cid in meta.index else None
        rows.append({"i": i, "id": int(cid), "name": (str(r.case_name) if r is not None else ""),
                     "year": (int(r.year) if r is not None and r.year == r.year else None)})
    json.dump(rows, open(os.path.join(out, "cases.json"), "w"))

    labels = {}
    if labels_year:
        for y, lst in labels_year.items():
            labels[str(int(y))] = [_label(r) for r in lst]

    summ = {}
    if summaries is not None:
        for _, r in summaries.iterrows():
            summ[str(r.name_key).lower()] = {"case_name": r.case_name, "year": int(r.year), "citation": r.citation, "summary": r.summary}

    # top-level years/centre/half/chunks describe the spectral track (same objects as tracks.spectral)
    manifest = {"years": years, "N": N, "K": K, "centre": sp["centre"], "half": sp["half"], "chunks": sp["chunks"],
                "vals": _rows(vals), "vals_aligned": _rows(vals_aligned),
                "robust_dims": ([int(d) for d in robust_dims] if robust_dims is not None else None),
                "dim_angles": ([float(v) for v in dim_angles] if dim_angles is not None else None),
                "tracks": tracks, "label_k": int(label_k),
                "palette": {str(int(k)): [float(c) for c in v] for k, v in palette.items()},
                "labels": labels, "summaries": summ,
                "neighbors": None}
    written = {c["file"] for t in tracks.values() for c in t["chunks"]}
    if neighbors is not None and len(neighbors):
        K_nbr = int(neighbors.groupby(["decade", "id"]).size().max())
        row_of = {int(c): i for i, c in enumerate(ids)}
        decades = sorted(int(d) for d in neighbors.decade.unique())
        for d in decades:
            sub = neighbors[neighbors.decade == d].sort_values(["id", "w"], ascending=[True, False])
            idx = np.full((N, K_nbr), -1, dtype=np.int32); w = np.zeros((N, K_nbr), dtype=np.float32); slot = np.zeros(N, dtype=np.int32)
            for cid, nid, ww in zip(sub.id.values, sub.nbr_id.values, sub.w.values):
                r, c = row_of.get(int(cid)), row_of.get(int(nid))
                if r is None or c is None or slot[r] >= K_nbr: continue
                idx[r, slot[r]] = c; w[r, slot[r]] = ww; slot[r] += 1
            with open(os.path.join(out, f"nbr_{d}.bin"), "wb") as f:
                f.write(idx.tobytes()); f.write(w.tobytes())
            written.add(f"nbr_{d}.bin")
        manifest["neighbors"] = {"decades": decades, "k": K_nbr, "file": "nbr_{decade}.bin"}
    # a previous export with other years/chunking leaves chunk files the new manifest does not list; drop them
    for fn in os.listdir(out):
        if _OWNED.match(fn) and fn not in written:
            os.remove(os.path.join(out, fn))
    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"))

    if verbose:
        size = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out)) / 1e6
        print(f"wrote {out}: {T} years x {N:,} cases x {K} dims, {len(sp['chunks'])} position chunks"
              + (f", force track {len(tracks['force']['years'])} keyframes" if "force" in tracks else "")
              + f", {size:.1f} MB total")


def cocitation_neighbors(snap_fn, years, ids, k=10, key=("cluster", "cites", "cluster")):
    """Top-k co-citation neighbours per case at each year in `years` (use decades). snap_fn(year) -> Snapshot.
    Returns a DataFrame (decade, id, nbr_id, w) restricted to `ids`."""
    import pandas as pd
    import scipy.sparse as sp
    out = []
    ids = np.asarray(ids)
    for y in years:
        s = snap_fn(y)
        all_ids = s.index(key[0]); A = s.adjacency(key)
        m = np.isin(all_ids, ids); A = A[m][:, m]; sub = all_ids[m]
        W = (A.T @ A).tocsr(); W.setdiag(0); W.eliminate_zeros()
        for r in range(W.shape[0]):
            lo, hi = W.indptr[r], W.indptr[r + 1]
            if hi == lo: continue
            cols, w = W.indices[lo:hi], W.data[lo:hi]
            top = np.argsort(-w)[:k]
            for c, ww in zip(cols[top], w[top]):
                out.append((int(y), int(sub[r]), int(sub[c]), float(ww)))
    return pd.DataFrame(out, columns=["decade", "id", "nbr_id", "w"])
