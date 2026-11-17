"""
Force-directed layout of the Leiden keyframes, precomputed. The browser never simulates the full cloud.

    from lawgraph.force_layout import ForceParams, layout_sequence, pick_backend
    frames = [(1900, G.W, G.ids, part.labels), (1905, ...), ...]           # keyframes in time order
    pos, stats = layout_sequence(frames, ForceParams(), pick_backend(gpu=True))
    pos[1905]                      # (n, 2) positions aligned with that keyframe's ids
    stats["displacement"][1905]    # mean node displacement from 1900, as a fraction of the layout radius

Each keyframe is laid out in 2-D with ForceAtlas2, warm-started from the previous keyframe (nodes that are
new start at the centroid of their community plus a little noise). Between every `chunk` solver iterations
three extra forces are applied to the positions:

  community gravity    every node moves a fraction `community_gravity` of the way to its community's centroid
  centroid repulsion   community centroids push each other apart with an inverse-square force, and each
                       community is translated rigidly by the push on its centroid
  temporal anchor      every node that existed in the previous keyframe moves a fraction `anchor` of the way
                       back to where it was (after fitting the old layout to the new one, so uniform growth,
                       rotation and translation are free)

The first two are the "Leiden-based force repulsion": ForceAtlas2 arranges cases by co-citation, the extra
terms pull each Leiden community together and keep different communities from sitting on top of each other.
The anchor is there because a warm start alone is not enough on this data: five years of new citations move
the equilibrium of an early (small, soft) keyframe by 15-30% of its radius even when the solver has fully
converged, which reads as churn rather than history. anchor=0 switches it off.
Finally each keyframe is Procrustes-aligned (rotation, reflection, scale, translation) to the previous one
on the nodes they share, so the motion the viewer shows is change in the graph, not solver drift.

Backends, in order of preference:
  GPU      RAPIDS cuGraph ForceAtlas2 (pylibcugraph.force_atlas2, the routine behind cugraph.force_atlas2)
  fa2      the `fa2` package (or its maintained fork `fa2_modified`), single-threaded Barnes-Hut
  igraph   Graph.layout_fruchterman_reingold(weights=...), when neither of the above can be imported
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class ForceParams:
    first_iterations: int = 300        # solver iterations for the first keyframe (cold start)
    iterations: int = 100              # ... for every later keyframe (warm start)
    chunk: int = 5                     # solver iterations between two applications of the extra forces
    community_gravity: float = 0.3     # fraction of the way to the community centroid per application
    community_repulsion: float = 0.1   # strength of the inverse-square repulsion between centroids (0 = off)
    anchor: float = 0.6                # fraction of the way back to its previous-keyframe position per application
                                       # (nodes that already existed; 0 = no temporal anchoring)
    scaling_ratio: float = 2.0         # ForceAtlas2 repulsion
    gravity: float = 1.0               # ForceAtlas2 pull towards the origin
    jitter_tolerance: float = 1.0      # ForceAtlas2 swing tolerance: lower = smaller, more careful steps
    cpu_theta: float = 2.0             # Barnes-Hut opening angle for the fa2 package. Its default, 1.2, takes
                                       # 13 min for the SCOTUS chain; 2.0 is 2.6x faster and moves the result by
                                       # ~2.5% of the layout radius. (The GPU solver keeps its own 0.5.)
    log_weights: bool = True           # lay out on log1p(co-citation count): raw counts glue hubs together
    noise: float = 0.02                # jitter for newly added nodes, as a fraction of the layout radius
    seed: int = 0


# ---------------------------------------------------------------- geometry
def rms_radius(pos: np.ndarray) -> float:
    """Layout radius: root-mean-square distance of the nodes from their centroid."""
    return float(np.sqrt(((pos - pos.mean(axis=0)) ** 2).sum(axis=1).mean()))


def procrustes(X: np.ndarray, Y: np.ndarray):
    """Similarity transform taking X onto Y in the least-squares sense: rotation or reflection R, uniform
    scale s and translation, returned as (s, R, mu_x, mu_y) so that  s * (P - mu_x) @ R + mu_y  maps any
    points P given in X's frame."""
    mx, my = X.mean(axis=0), Y.mean(axis=0)
    Xc, Yc = X - mx, Y - my
    U, S, Vt = np.linalg.svd(Xc.T @ Yc)
    R = U @ Vt                                         # orthogonal; a reflection is allowed
    denom = float((Xc ** 2).sum())
    s = float(S.sum() / denom) if denom > 0 else 1.0
    return s, R, mx, my


def apply_similarity(P: np.ndarray, tf) -> np.ndarray:
    s, R, mx, my = tf
    return s * (P - mx) @ R + my


def centroids(pos: np.ndarray, labels: np.ndarray, k: int):
    sizes = np.bincount(labels, minlength=k).astype(float)
    cent = np.stack([np.bincount(labels, weights=pos[:, d], minlength=k) for d in range(pos.shape[1])], axis=1)
    return cent / np.maximum(sizes, 1.0)[:, None], sizes


def community_step(pos: np.ndarray, labels: np.ndarray, k: int, gravity: float, repulsion: float) -> np.ndarray:
    """One application of the two community forces (see module docstring). Scale-free: distances are measured
    in units of the layout radius, so the same strengths work for any solver."""
    cent, sizes = centroids(pos, labels, k)
    out = pos + gravity * (cent[labels] - pos)
    if repulsion > 0 and k > 1:
        R = rms_radius(pos) or 1.0
        d = cent[:, None, :] - cent[None, :, :]                       # (k, k, 2): from b to a
        dist = np.sqrt((d ** 2).sum(axis=-1))
        dist = np.maximum(dist, 0.05 * R)                             # soften coincident centroids
        np.fill_diagonal(dist, np.inf)
        w = sizes / max(sizes.sum(), 1.0)                             # bigger communities push harder
        push = repulsion * R ** 3 * (d / dist[..., None] ** 3 * w[None, :, None]).sum(axis=1)
        mag = np.linalg.norm(push, axis=1)
        push *= np.minimum(1.0, 0.1 * R / np.maximum(mag, 1e-12))[:, None]   # at most 10% of the radius per step
        push[sizes == 0] = 0.0
        out = out + push[labels]
    return out


def anchor_step(pos: np.ndarray, old: np.ndarray, prev: np.ndarray, strength: float) -> np.ndarray:
    """Temporal anchoring: nodes that existed in the previous keyframe (rows `old`, previous positions `prev`)
    move a fraction `strength` of the way back to where they were. The previous layout is first fitted to
    the current one with a similarity transform, so growing, turning or shifting as a whole costs nothing;
    only a change of shape is resisted."""
    if strength <= 0 or len(old) < 2:
        return pos
    target = apply_similarity(prev, procrustes(prev, pos[old]))
    out = pos.copy()
    out[old] += strength * (target - pos[old])
    return out


def separate(pos: np.ndarray, rng: np.random.Generator, eps: float = 1e-3) -> np.ndarray:
    """Barnes-Hut trees degenerate (and the GPU solver can hang) when points coincide, which pulling nodes
    towards shared centroids invites. A jitter of 0.1% of the layout radius keeps every point distinct."""
    return pos + rng.normal(scale=eps * (rms_radius(pos) or 1.0), size=pos.shape)


def warm_start(ids: np.ndarray, labels: np.ndarray, ids_prev: np.ndarray, pos_prev: np.ndarray,
               rng: np.random.Generator, noise: float = 0.02) -> np.ndarray:
    """Initial positions for a keyframe: nodes kept from the previous keyframe stay where they were; new
    nodes start at the centroid of the already-placed members of their community, plus small noise
    (at the layout centroid if the whole community is new)."""
    n = len(ids); k = int(labels.max()) + 1 if n else 0
    init = np.zeros((n, 2)); placed = np.zeros(n, dtype=bool)
    _, ia, ib = np.intersect1d(ids_prev, ids, assume_unique=True, return_indices=True)
    init[ib] = pos_prev[ia]; placed[ib] = True
    if placed.all():
        return init
    if not placed.any():
        return rng.normal(scale=1.0, size=(n, 2))
    centre = init[placed].mean(axis=0); R = rms_radius(init[placed]) or 1.0
    cent, sizes = centroids(init[placed], labels[placed], k)
    cent[sizes == 0] = centre
    new = ~placed
    init[new] = cent[labels[new]] + rng.normal(scale=noise * R, size=(int(new.sum()), 2))
    return init


# ---------------------------------------------------------------- solvers
class GpuFA2:
    """RAPIDS cuGraph ForceAtlas2 on the GPU (Barnes-Hut). The graph is uploaded once per keyframe."""
    name = "cuGraph ForceAtlas2 (GPU)"

    def __init__(self):
        import cupy
        import pylibcugraph
        self.cp, self.plc = cupy, pylibcugraph
        self.rh = pylibcugraph.ResourceHandle()
        cupy.zeros(1).sum()                                           # fail here, not mid-build, if the GPU is unusable

    def prepare(self, W: sp.csr_matrix):
        cp, plc = self.cp, self.plc
        coo = W.tocoo()
        G = plc.SGGraph(self.rh, plc.GraphProperties(is_symmetric=True, is_multigraph=False),
                        cp.asarray(coo.row.astype(np.int32)), cp.asarray(coo.col.astype(np.int32)),
                        weight_array=cp.asarray(coo.data.astype(np.float32)),
                        store_transposed=False, renumber=False, do_expensive_check=False)
        return G, W.shape[0]

    def run(self, handle, pos: np.ndarray, iterations: int, p: ForceParams) -> np.ndarray:
        cp = self.cp; G, n = handle
        v, x, y = self.plc.force_atlas2(
            self.rh, p.seed, G, int(iterations),
            cp.arange(n, dtype=np.int32), cp.asarray(pos[:, 0], dtype=np.float32), cp.asarray(pos[:, 1], dtype=np.float32),
            True,          # outbound_attraction_distribution: hubs attract less
            False,         # lin_log_mode
            False, None, None, 100.0,                                 # prevent_overlapping, vertex radii, overlap scaling
            1.0, float(p.jitter_tolerance),                           # edge_weight_influence, jitter_tolerance
            True, 0.5,     # barnes_hut_optimize, barnes_hut_theta
            float(p.scaling_ratio), False, float(p.gravity),          # scaling_ratio, strong_gravity_mode, gravity
            None, None, None, None, False, False)                     # mobility, mass, verbose, expensive check
        order = cp.argsort(v)
        return cp.asnumpy(cp.stack([x[order], y[order]], axis=1)).astype(np.float64)


class CpuFA2:
    """The `fa2` package (Cython Barnes-Hut, one thread). Its forceatlas2() rebuilds a Python object per node
    and per edge on every call (seconds on a million edges), and the extra forces need a call every `chunk`
    iterations, so the objects are built once per keyframe and only the positions are reset between calls.
    The iteration loop itself is the package's own."""
    name = "fa2 ForceAtlas2 (CPU)"

    def __init__(self):
        try:
            from fa2 import ForceAtlas2
        except ImportError:
            from fa2_modified import ForceAtlas2                      # same API, builds on current Pythons

        class Reusing(ForceAtlas2):
            _built = None

            def init(self, G, pos=None):
                if self._built is None:
                    self._built = super().init(G, pos)
                    return self._built
                nodes, edges = self._built
                for n, xy in zip(nodes, pos):
                    n.x, n.y = float(xy[0]), float(xy[1])
                    n.dx = n.dy = n.old_dx = n.old_dy = 0.0
                return nodes, edges
        self.cls = Reusing

    def prepare(self, W: sp.csr_matrix):
        return {"G": W.tolil(), "fa": None}                           # lil is what fa2 converts to internally

    def run(self, handle, pos: np.ndarray, iterations: int, p: ForceParams) -> np.ndarray:
        if handle["fa"] is None:
            handle["fa"] = self.cls(outboundAttractionDistribution=True, linLogMode=False, adjustSizes=False,
                                    edgeWeightInfluence=1.0, jitterTolerance=p.jitter_tolerance,
                                    barnesHutOptimize=True, barnesHutTheta=p.cpu_theta, scalingRatio=p.scaling_ratio,
                                    strongGravityMode=False, gravity=p.gravity, verbose=False)
        out = handle["fa"].forceatlas2(handle["G"], pos=np.asarray(pos, dtype=float), iterations=int(iterations))
        return np.asarray(out, dtype=np.float64)


class IgraphFR:
    """Last resort: igraph's Fruchterman-Reingold with edge weights, seeded with the current positions."""
    name = "igraph Fruchterman-Reingold (CPU)"

    def __init__(self):
        import igraph
        self.ig = igraph

    def prepare(self, W: sp.csr_matrix):
        U = sp.triu(W, k=1).tocoo()
        g = self.ig.Graph(n=W.shape[0], edges=np.column_stack([U.row, U.col]).tolist(), directed=False)
        g.es["weight"] = U.data.tolist()
        return g

    def run(self, handle, pos: np.ndarray, iterations: int, p: ForceParams) -> np.ndarray:
        # FR's ideal edge length is ~1 in its own units; hand it a layout of comparable size and a low
        # start temperature so a warm start is refined rather than shaken apart
        n = handle.vcount(); scale = (rms_radius(pos) or 1.0) / max(np.sqrt(n), 1.0)
        lay = handle.layout_fruchterman_reingold(weights="weight", niter=int(iterations), seed=(pos / scale).tolist(),
                                                 start_temp=max(np.sqrt(n) / 100.0, 0.05), grid="auto")
        return np.asarray(lay.coords, dtype=np.float64) * scale


def pick_backend(gpu: bool = False, log=print):
    """--gpu asks for cuGraph; without it, or if cuGraph/the GPU is unusable, fa2, then igraph."""
    order = ([GpuFA2] if gpu else []) + [CpuFA2, IgraphFR]
    for cls in order:
        try:
            return cls()
        except Exception as e:                                        # ImportError, or CUDA failing to initialise
            log(f"force layout: {cls.name} unavailable ({type(e).__name__}: {str(e)[:80]}); trying the next solver")
    raise RuntimeError("no force-layout solver available: install fa2 (or fa2_modified) or python-igraph")


# ---------------------------------------------------------------- one keyframe, then the chain
def layout_keyframe(W: sp.csr_matrix, labels: np.ndarray, init: np.ndarray, iterations: int,
                    p: ForceParams, backend, old: np.ndarray | None = None, prev: np.ndarray | None = None,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """`iterations` solver iterations from `init`. Between chunks of `p.chunk` iterations the community
    forces are applied, and nodes listed in `old` are anchored towards their previous positions `prev`."""
    W = W.tocsr().astype(np.float64)
    if p.log_weights:
        W = W.copy(); W.data = np.log1p(W.data)
    k = int(labels.max()) + 1
    rng = rng or np.random.default_rng(p.seed)
    chunk = max(1, int(p.chunk))
    handle = backend.prepare(W)
    pos, done = separate(np.asarray(init, dtype=np.float64), rng), 0
    while done < iterations:
        step = min(chunk, iterations - done)
        pos = backend.run(handle, pos, step, p)
        done += step
        if not np.isfinite(pos).all():
            raise FloatingPointError(f"{backend.name} returned non-finite positions")
        if done < iterations:                                         # the solver always has the last word
            pos = community_step(pos, labels, k, p.community_gravity, p.community_repulsion)
            if old is not None:
                pos = anchor_step(pos, old, prev, p.anchor)
            pos = separate(pos, rng)
    if old is not None:                                               # ... except for one closing pull on the anchors:
        pos = anchor_step(pos, old, prev, p.anchor)                   # the solver relaxes most of an in-loop pull away
    return pos


def layout_sequence(frames, params: ForceParams | None = None, backend=None, log=print):
    """frames: [(year, W, ids, labels)] in time order; ids sorted, labels aligned with ids.
    Returns ({year: (n, 2) aligned positions}, stats). stats has, per keyframe after the first, the mean
    displacement of the nodes shared with the previous keyframe divided by the layout radius."""
    p = params or ForceParams()
    backend = backend or pick_backend(False, log)
    rng = np.random.default_rng(p.seed)
    out, disp, radius, secs = {}, {}, {}, {}
    raw_prev = aligned_prev = ids_prev = None
    for year, W, ids, labels in frames:
        t0 = time.time()
        ids = np.asarray(ids); labels = np.asarray(labels); n = len(ids)
        if raw_prev is None:
            init, iters, old, prev = rng.normal(scale=10.0 * np.sqrt(n), size=(n, 2)), p.first_iterations, None, None
        else:                                                          # warm start in the solver's own frame
            init, iters = warm_start(ids, labels, ids_prev, raw_prev, rng, p.noise), p.iterations
            _, ia, old = np.intersect1d(ids_prev, ids, assume_unique=True, return_indices=True)
            prev = raw_prev[ia]
        raw = layout_keyframe(W, labels, init, iters, p, backend, old, prev, rng)
        if aligned_prev is None:
            aligned = raw - raw.mean(axis=0)
        else:
            _, ia, ib = np.intersect1d(ids_prev, ids, assume_unique=True, return_indices=True)
            aligned = apply_similarity(raw, procrustes(raw[ib], aligned_prev[ia])) if len(ia) >= 2 else raw - raw.mean(axis=0)
            if len(ia):
                disp[int(year)] = float(np.linalg.norm(aligned[ib] - aligned_prev[ia], axis=1).mean() / (rms_radius(aligned) or 1.0))
        radius[int(year)] = rms_radius(aligned); secs[int(year)] = time.time() - t0
        out[int(year)] = aligned
        raw_prev, aligned_prev, ids_prev = raw, aligned, ids
        log(f"force {year}: {n:,} nodes, {iters} iterations, {secs[int(year)]:.1f}s"
            + (f", mean displacement {100 * disp[int(year)]:.1f}% of radius" if int(year) in disp else ""))
    stats = {"backend": backend.name, "displacement": disp, "radius": radius, "seconds": secs, "params": asdict(p)}
    return out, stats
