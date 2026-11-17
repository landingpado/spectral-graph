"""
Unit tests for the precompute: force layout (lawgraph/force_layout.py), the K-dimensional export
(lawgraph/viewer_export.py) and the label / alignment helpers in lawgraph/viewer_build.py.
No database needed.

    python -m pytest tests/test_build.py -q
"""
import json
import os

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from lawgraph import force_layout as fl
from lawgraph.viewer_build import canonical_signs, community_label, dimension_stability, is_boilerplate, short_name
from lawgraph.viewer_export import export_viewer


# ---------------------------------------------------------------- synthetic graphs
def blocks(sizes, p_in=0.25, p_out=0.01, seed=0):
    """Planted-partition graph: symmetric weights, zero diagonal, connected in practice."""
    rng = np.random.default_rng(seed); n = sum(sizes); lab = np.repeat(np.arange(len(sizes)), sizes)
    P = np.where(lab[:, None] == lab[None, :], p_in, p_out)
    A = np.triu(rng.random((n, n)) < P, 1).astype(float) * rng.integers(1, 6, (n, n))
    for i in range(n - 1):
        A[i, i + 1] = max(A[i, i + 1], 1.0)                     # a path through everything keeps it connected
    return sp.csr_matrix(A + A.T), lab


def growing_sequence(steps=4, seed=1):
    """The same planted graph revealed node by node: each frame adds nodes (ids sorted, labels kept)."""
    W, lab = blocks([60, 50, 40], seed=seed); n = W.shape[0]; rng = np.random.default_rng(seed)
    order = rng.permutation(n); frames = []
    for k in range(steps):
        keep = np.sort(order[: int(n * (0.6 + 0.4 * k / (steps - 1)))])
        frames.append((1900 + 5 * k, W[keep][:, keep].tocsr(), keep.astype(np.int64) + 1000, lab[keep]))
    return frames


# ---------------------------------------------------------------- geometry
def test_procrustes_recovers_a_similarity_transform():
    rng = np.random.default_rng(0); X = rng.normal(size=(200, 2))
    th = 0.7; R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) @ np.diag([1, -1])     # rotation and a reflection
    Y = 3.5 * X @ R + np.array([10.0, -4.0])
    tf = fl.procrustes(X, Y)
    assert np.allclose(fl.apply_similarity(X, tf), Y, atol=1e-9)
    assert tf[0] == pytest.approx(3.5) and np.allclose(tf[1] @ tf[1].T, np.eye(2))
    extra = rng.normal(size=(5, 2))                           # points not used in the fit map the same way
    assert np.allclose(fl.apply_similarity(extra, tf), 3.5 * extra @ R + np.array([10.0, -4.0]))


def test_rms_radius():
    assert fl.rms_radius(np.array([[1.0, 0], [-1, 0], [0, 1], [0, -1]]) * 3 + 7) == pytest.approx(3.0)


def test_community_gravity_pulls_towards_centroids():
    rng = np.random.default_rng(0); lab = np.repeat([0, 1], 50)
    pos = rng.normal(size=(100, 2)) + np.where(lab[:, None] == 0, -5.0, 5.0)
    cent, sizes = fl.centroids(pos, lab, 2); assert sizes.tolist() == [50, 50]
    out = fl.community_step(pos, lab, 2, gravity=0.3, repulsion=0.0)
    assert np.allclose(out, pos + 0.3 * (cent[lab] - pos))                       # exactly 30% of the way
    assert np.allclose(fl.centroids(out, lab, 2)[0], cent)                       # centroids do not move
    assert np.allclose(fl.community_step(pos, lab, 2, 0.0, 0.0), pos)


def test_centroid_repulsion_separates_communities_rigidly():
    rng = np.random.default_rng(0); lab = np.repeat([0, 1, 2], 40)
    pos = rng.normal(size=(120, 2)) + np.array([[-2.0, 0], [2.0, 0], [0, 3.0]])[lab]
    c0, _ = fl.centroids(pos, lab, 3)
    out = fl.community_step(pos, lab, 3, gravity=0.0, repulsion=0.5)
    c1, _ = fl.centroids(out, lab, 3)
    d0 = np.linalg.norm(c0[:, None] - c0[None], axis=-1); d1 = np.linalg.norm(c1[:, None] - c1[None], axis=-1)
    assert (d1[np.triu_indices(3, 1)] > d0[np.triu_indices(3, 1)]).all()         # every pair moved apart
    for c in range(3):                                                           # each community moved as a rigid body
        assert np.allclose(out[lab == c] - pos[lab == c], (c1 - c0)[c])
    R = fl.rms_radius(pos); assert (np.linalg.norm(c1 - c0, axis=1) <= 0.1 * R + 1e-9).all()   # capped per step
    same = np.zeros((10, 2)); assert np.isfinite(fl.community_step(same, np.repeat([0, 1], 5), 2, 0.3, 1.0)).all()   # coincident centroids


def test_anchor_step_ignores_uniform_motion_and_resists_shape_change():
    rng = np.random.default_rng(0); prev = rng.normal(size=(80, 2)); old = np.arange(80)
    moved = 2.0 * prev @ np.array([[0.0, -1.0], [1.0, 0.0]]) + 5.0               # grown, turned, shifted: nothing to resist
    assert np.allclose(fl.anchor_step(moved, old, prev, 0.6), moved)
    bent = moved.copy(); bent[:10] += 3.0                                        # ten nodes wander off
    out = fl.anchor_step(bent, old, prev, 0.6)
    assert np.linalg.norm(out[:10] - moved[:10]) < np.linalg.norm(bent[:10] - moved[:10])
    assert np.allclose(fl.anchor_step(bent, old, prev, 0.0), bent)
    pos = np.vstack([bent, [[9.0, 9.0]]])                                        # a new node is never anchored
    assert np.allclose(fl.anchor_step(pos, old, prev, 0.6)[-1], [9.0, 9.0])


def test_warm_start_keeps_old_nodes_and_seeds_new_ones_at_their_community():
    rng = np.random.default_rng(0)
    ids_prev = np.array([1, 2, 3, 4, 5, 6]); pos_prev = np.array([[0, 0], [0, 1], [1, 0], [10, 10], [10, 11], [11, 10]], float)
    ids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]); lab = np.array([0, 0, 0, 1, 1, 1, 1, 0, 2])            # 7 -> community 1, 8 -> 0, 9 -> a new community
    init = fl.warm_start(ids, lab, ids_prev, pos_prev, rng, noise=0.02)
    assert np.array_equal(init[:6], pos_prev)
    R = fl.rms_radius(pos_prev)
    assert np.linalg.norm(init[6] - pos_prev[3:].mean(0)) < 0.2 * R and np.linalg.norm(init[7] - pos_prev[:3].mean(0)) < 0.2 * R
    assert np.linalg.norm(init[8] - pos_prev.mean(0)) < 0.2 * R                  # whole community new: layout centre
    assert np.array_equal(fl.warm_start(ids_prev, lab[:6], ids_prev, pos_prev, rng), pos_prev)


def test_separate_makes_coincident_points_distinct():
    pos = np.vstack([np.zeros((50, 2)), np.ones((50, 2)) * 100])
    out = fl.separate(pos, np.random.default_rng(0))
    assert len(np.unique(out, axis=0)) == 100 and np.abs(out - pos).max() < 0.02 * fl.rms_radius(pos)


# ---------------------------------------------------------------- solvers and the keyframe chain
def cpu_backends():
    out = []
    for cls in (fl.CpuFA2, fl.IgraphFR):
        try: out.append(cls())
        except ImportError: pass
    return out


@pytest.mark.parametrize("backend", cpu_backends(), ids=lambda b: b.name)
def test_layout_sequence_is_stable_and_groups_communities(backend):
    frames = growing_sequence()
    pos, stats = fl.layout_sequence(frames, fl.ForceParams(first_iterations=120, iterations=60), backend, log=lambda m: None)
    assert list(pos) == [f[0] for f in frames] and stats["backend"] == backend.name
    for year, W, ids, lab in frames:
        P = pos[year]; assert P.shape == (len(ids), 2) and np.isfinite(P).all()
        cent, _ = fl.centroids(P, lab, 3)
        own = np.linalg.norm(P - cent[lab], axis=1).mean()
        other = np.mean([np.linalg.norm(P - cent[(lab + s) % 3], axis=1).mean() for s in (1, 2)])
        assert own < 0.8 * other, f"{year}: communities are not grouped ({own:.2f} vs {other:.2f})"
    assert len(stats["displacement"]) == len(frames) - 1
    assert max(stats["displacement"].values()) < 0.25          # small graphs growing by 13% a step; SCOTUS is checked in the build
    assert all(r > 0 for r in stats["radius"].values())


def test_anchor_reduces_displacement():
    frames = growing_sequence(); be = fl.CpuFA2() if any(isinstance(b, fl.CpuFA2) for b in cpu_backends()) else fl.IgraphFR()
    run = lambda a: fl.layout_sequence(frames, fl.ForceParams(first_iterations=120, iterations=60, anchor=a), be, log=lambda m: None)[1]["displacement"]
    free, held = run(0.0), run(0.8)
    assert np.mean(list(held.values())) < np.mean(list(free.values()))


def test_reused_fa2_objects_give_the_package_result():
    """CpuFA2 keeps fa2's node/edge objects between calls; the numbers must be what a fresh call returns."""
    try:
        be = fl.CpuFA2()
    except ImportError:
        pytest.skip("fa2 not installed")
    W, _ = blocks([30, 30]); p = fl.ForceParams(); start = np.random.default_rng(0).normal(scale=50, size=(60, 2))
    h = be.prepare(W); a1 = be.run(h, start, 7, p); a2 = be.run(h, a1, 7, p)
    h2 = be.prepare(W); b1 = be.run(h2, start, 7, p)
    h3 = be.prepare(W); b2 = be.run(h3, b1, 7, p)             # fresh objects for the second leg
    assert np.array_equal(a1, b1) and np.array_equal(a2, b2)


def test_pick_backend_falls_back_when_the_gpu_solver_is_unavailable(monkeypatch):
    def broken(self): raise ImportError("no cupy here")
    monkeypatch.setattr(fl.GpuFA2, "__init__", broken)
    said = []
    be = fl.pick_backend(gpu=True, log=said.append)
    assert not isinstance(be, fl.GpuFA2) and any("unavailable" in m for m in said)
    assert not isinstance(fl.pick_backend(gpu=False), fl.GpuFA2)


# ---------------------------------------------------------------- labels
@pytest.mark.parametrize("name, short", [
    ("Miranda v. Arizona", "Miranda"), ("Mapp v. Ohio", "Mapp"), ("Gideon v. Wainwright", "Gideon"),
    ("United States v. Darby", "Darby"), ("State of Missouri v. Holland", "Holland"), ("Ex Parte Young", "Young"),
    ("In Re Ayers", "Ayers"), ("The Lottawanna", "Lottawanna"), ("New York Times Co. v. Sullivan", "New York"),
    ("Bank of Augusta v. Earle", "Bank of Augusta"), ("Erie Railroad v. Tompkins", "Erie"),
    ("A. L. A. Schechter Poultry Corp. v. United States", "A. L. A. Schechter"), ("St. Louis, Iron Mountain & Southern Railway Co. v. Taylor", "St. Louis"),
    ("Chicago, Burlington & Quincy Railroad v. Chicago", "Chicago"), ("M'culloch v. State of Maryland", "M'culloch"),
])
def test_short_name(name, short):
    assert short_name(name) == short


def test_community_label_top_k_skips_boilerplate_and_duplicates():
    names = {1: "United States v. Detroit Timber & Lumber Co.", 2: "Miranda v. Arizona", 3: "Miranda v. Arizona", 4: "Mapp v. Ohio",
             5: "Gideon v. Wainwright", 6: "Boyd v. United States"}
    assert is_boilerplate(names[1]) and not is_boilerplate(names[2])
    text, full, ids = community_label([1, 2, 3, 4, 5, 6], names, 3)
    assert text == "Miranda · Mapp · Gideon" and ids == [2, 4, 5] and full == ["Miranda v. Arizona", "Mapp v. Ohio", "Gideon v. Wainwright"]
    assert community_label([2], names, 3)[0] == "Miranda"                      # fewer than k members is fine
    assert community_label([1], names, 3) == ("", [], [])
    assert community_label([7, 2], {**names, 7: float("nan")}, 1)[2] == [2]     # a missing name is skipped


# ---------------------------------------------------------------- alignment diagnostics
class Emb:
    def __init__(self, ids, vecs): self.ids, self.vecs = np.asarray(ids), vecs


def test_canonical_signs_orients_each_dimension():
    ref = np.array([[3.0, -3.0], [-1.0, 1.0], [-1.0, 1.0]])
    assert canonical_signs(ref).tolist() == [1.0, -1.0]
    assert canonical_signs(-ref).tolist() == [-1.0, 1.0]
    assert canonical_signs(np.zeros((3, 2))).tolist() == [1.0, 1.0]


def test_dimension_stability_ranks_a_rotating_dimension_last():
    rng = np.random.default_rng(0); n = 300; base = np.linalg.qr(rng.normal(size=(n, 3)))[0]
    seq, embs = [], []
    for t in range(6):
        A = base.copy(); A[:, 2] = np.linalg.qr(np.c_[base[:, 2] + 0.6 * t * rng.normal(size=n) / np.sqrt(n)])[0][:, 0]   # dim 2 drifts
        seq.append(A); embs.append(Emb(np.arange(n), A))
    ang = dimension_stability(embs, seq)
    assert ang.shape == (3,) and ang[0] < 1e-6 and ang[1] < 1e-6 and ang[2] > 0.1
    assert sorted(np.argsort(ang)[:2].tolist()) == [0, 1]


# ---------------------------------------------------------------- export
def read_track(out, track, N):
    K = track["K"]; rows = []
    for c in track["chunks"]:
        q = np.fromfile(os.path.join(out, c["file"]), dtype=np.int16).reshape(len(c["years"]), N, K).astype(float)
        x = q / 32767 * np.array(track["half"]) + np.array(track["centre"]); x[q == -32768] = np.nan; rows.append(x)
    return np.concatenate(rows)


def toy_export(tmp_path, K=8, T=23, N=40, **kw):
    rng = np.random.default_rng(0); years = list(range(1900, 1900 + T)); ids = np.arange(100, 100 + N)
    pos = rng.normal(size=(T, N, K)).astype(np.float32); pos[:5, 30:] = np.nan            # ten cases arrive in year 5
    chain = rng.integers(-1, 4, size=(T, N))
    meta = pd.DataFrame({"case_name": [f"Case {i} v. State" for i in ids], "year": 1850.0 + np.arange(N)}, index=ids)
    out = str(tmp_path / "data")
    export_viewer(out, years, ids, pos, chain, {0: [1, 0, 0], 1: [0, 1, 0]}, meta, verbose=False, **kw)
    return out, years, ids, pos, chain, json.load(open(os.path.join(out, "manifest.json")))


def test_export_writes_all_k_dimensions_in_decade_chunks(tmp_path):
    vals = np.linspace(0.1, 0.9, 23 * 8).reshape(23, 8); vals[0, 6:] = np.nan
    out, years, ids, pos, chain, M = toy_export(tmp_path, vals=vals, robust_dims=[0, 2, 4], dim_angles=np.arange(8) / 10, label_k=3)
    assert M["K"] == 8 and M["N"] == 40 and M["years"] == years and len(M["centre"]) == 8 and len(M["half"]) == 8
    assert [c["file"] for c in M["chunks"]] == ["pos_1900_1909.bin", "pos_1910_1919.bin", "pos_1920_1922.bin"]
    assert os.path.getsize(os.path.join(out, "pos_1900_1909.bin")) == 10 * 40 * 8 * 2                 # Int16 (years, N, K)
    assert M["tracks"]["spectral"]["chunks"] == M["chunks"] and "force" not in M["tracks"]
    got = read_track(out, M["tracks"]["spectral"], 40)
    assert np.array_equal(np.isnan(got), np.isnan(pos))                                               # -32768 = absent
    assert np.nanmax(np.abs(got - pos)) <= 1.01 * max(M["half"]) / 32767                              # decode v/32767*half+centre
    assert M["robust_dims"] == [0, 2, 4] and M["label_k"] == 3 and len(M["vals"]) == 23 and M["vals"][0][7] is None and M["vals"][1][7] == pytest.approx(vals[1, 7])
    ch = np.fromfile(os.path.join(out, "chain.bin"), dtype=np.uint8).reshape(23, 40)
    assert np.array_equal(ch, np.where(chain < 0, 255, chain))
    cases = json.load(open(os.path.join(out, "cases.json")))
    assert cases[3] == {"i": 3, "id": 103, "name": "Case 103 v. State", "year": 1853}


def test_export_force_track_labels_and_neighbours(tmp_path):
    rng = np.random.default_rng(1); fy = [1900, 1905, 1910, 1915, 1920]; fpos = rng.normal(scale=500, size=(5, 40, 2)); fpos[0, 30:] = np.nan
    labels = {1900: [{"name": "A · B · C", "full": ["A v. X", "B v. Y", "C v. Z"], "ids": [100, 101, 102], "size": 12, "p": np.arange(8.0), "pf": np.array([1.0, 2.0]), "chain": 1},
                     {"name": "D", "p": np.zeros(8), "chain": -1}],
              1905: [("Old style", 0.1, 0.2, 0.3, 2)]}                                                # the notebook's tuple format still works
    nb = pd.DataFrame({"decade": [1900, 1900, 1910], "id": [100, 100, 101], "nbr_id": [101, 102, 100], "w": [5.0, 9.0, 2.0]})
    out, *_, M = toy_export(tmp_path, labels_year=labels, neighbors=nb, force={"years": fy, "pos": fpos, "backend": "test", "displacement": {"1905": 0.05}})
    f = M["tracks"]["force"]
    assert f["K"] == 2 and f["years"] == fy and f["backend"] == "test" and [c["file"] for c in f["chunks"]] == ["force_1900_1920.bin"]
    got = read_track(out, f, 40); assert np.array_equal(np.isnan(got), np.isnan(fpos)) and np.nanmax(np.abs(got - fpos)) < 0.1
    a, d = M["labels"]["1900"]
    assert a["name"] == "A · B · C" and a["full"][0] == "A v. X" and a["size"] == 12 and a["chain"] == 1 and a["pf"] == [1.0, 2.0] and len(a["p"]) == 8
    assert d["chain"] is None and M["labels"]["1905"] == [{"name": "Old style", "p": [0.1, 0.2, 0.3], "chain": 2}]
    assert M["neighbors"] == {"decades": [1900, 1910], "k": 2, "file": "nbr_{decade}.bin"}
    buf = open(os.path.join(out, "nbr_1900.bin"), "rb").read()
    idx = np.frombuffer(buf, dtype=np.int32, count=80).reshape(40, 2); w = np.frombuffer(buf, dtype=np.float32, offset=320).reshape(40, 2)
    assert idx[0].tolist() == [2, 1] and w[0].tolist() == [9.0, 5.0] and idx[1].tolist() == [-1, -1]   # strongest first; -1 = none


def test_export_accepts_three_dimensions_as_before(tmp_path):
    out, *_, M = toy_export(tmp_path, K=3)
    assert M["K"] == 3 and M["robust_dims"] is None and M["vals"] is None and os.path.getsize(os.path.join(out, "pos_1900_1909.bin")) == 10 * 40 * 3 * 2


def test_export_removes_chunks_a_new_manifest_no_longer_lists(tmp_path):
    out, *_ = toy_export(tmp_path)
    for stale in ("pos_1850_1859.bin", "force_1850_1895.bin", "nbr_1850.bin"):
        open(os.path.join(out, stale), "wb").write(b"old")
    open(os.path.join(out, "notes.txt"), "w").write("mine"); open(os.path.join(out, "pos_backup.bin"), "wb").write(b"mine")
    export_viewer(out, list(range(1900, 1923)), np.arange(100, 140), np.zeros((23, 40, 8), np.float32) + np.arange(8), np.zeros((23, 40), int), {},
                  pd.DataFrame({"case_name": [], "year": []}), verbose=False)
    left = set(os.listdir(out))
    assert not {"pos_1850_1859.bin", "force_1850_1895.bin", "nbr_1850.bin"} & left          # this exporter's own stale chunks: gone
    assert {"notes.txt", "pos_backup.bin", "pos_1900_1909.bin", "manifest.json"} <= left    # anything else: untouched
