"""Generates notebooks/leiden_sanity.ipynb (kept as a script so the notebook is reproducible)."""
import json

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md("""# Leiden sanity check: communities and their lineage over time
Same co-citation slices as `spectral_sanity.ipynb`. Checks, in order:
1. one slice: do the communities have names?
2. is the partition stable across random seeds?
3. does it agree with the spectral picture?
4. timeline of partitions
5. lineage: which communities persist, split, merge
6. a flow picture
7. the export tables the site will consume""")

code('''import sys, time, numpy as np, pandas as pd, matplotlib.pyplot as plt
import scipy.sparse as sp
ROOT = "."                 # <-- directory that contains lawgraph/
DB   = "graph.duckdb"      # <-- your file
sys.path.insert(0, ROOT)
from lawgraph.store import Store
from lawgraph.lens import select, build, Spectral, Leiden, lineage
pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 70)
st = Store(DB, read_only=True); con = st.con

COURT   = "scotus"
YEARS   = [1900, 1920, 1940, 1960, 1980, 2000, 2020]
MODE    = "cocite"
MIN_DEG = 5
RES     = 1.0              # Leiden resolution (modularity). Higher = more, smaller communities.
MIN_SHOW = 50              # only describe communities at least this big

def names(ids):
    df = con.execute("SELECT id, case_name, year(date_filed) AS year FROM node_cluster WHERE id IN (SELECT unnest(?))",
                     [list(map(int, ids))]).df().set_index("id")
    return df.reindex(ids)

def scotus(snap):
    return select(snap, "cluster", where=f"d.dst_id = '{COURT}' AND n.precedential_status = 'Published'",
                  join="JOIN edge_cluster_in_docket_docket cd ON cd.src_id = n.id JOIN edge_docket_filed_in_court d ON d.src_id = cd.dst_id")

def describe(part, G, k=6, min_size=MIN_SHOW):
    """Print each community's size, median year and top-k members by degree."""
    deg = G.degree
    for c in range(part.k):
        m = np.where(part.labels == c)[0]
        if len(m) < min_size: continue
        top = m[np.argsort(-deg[m])[:k]]
        yrs = names(G.ids[m]).year
        print(f"--- community {c}: {len(m):,} cases, median year {int(np.nanmedian(yrs))}")
        for _, r in names(G.ids[top]).iterrows(): print(f"      {str(r.case_name)[:62]}  {r.year}")

def ari(a, b):
    """Adjusted Rand index between two labelings of the same nodes."""
    from scipy.special import comb
    n = len(a); M = sp.csr_matrix((np.ones(n), (a, b))).toarray()
    s = comb(M, 2).sum(); sa = comb(M.sum(1), 2).sum(); sb = comb(M.sum(0), 2).sum()
    e = sa * sb / comb(n, 2); return float((s - e) / ((sa + sb) / 2 - e))''')

md("""## 1. One slice: do the communities have names?
**Sensible:** 8–25 communities above `MIN_SHOW`, each describable as an area of law from its top cases. A handful of tiny leftovers is normal.""")

code('''t0 = time.time()
s = st.at(2000)
G = build(s, ("cluster", "cites", "cluster"), scotus(s), mode=MODE, min_degree=MIN_DEG)
print(f"graph {G.n:,} nodes, {G.W.nnz//2:,} edges   ({time.time()-t0:.1f}s)")
for r in (0.5, 1.0, 2.0):
    t0 = time.time(); p = Leiden(resolution=r)(G)
    sz = p.sizes(); print(f"resolution {r}: {p.k} communities, {int((sz >= MIN_SHOW).sum())} with >= {MIN_SHOW} nodes, "
                          f"largest {sz[:5].tolist()}, quality {p.quality:.3f}   ({time.time()-t0:.1f}s)")
part = Leiden(resolution=RES)(G)
describe(part, G)''')

md("""## 2. Stability across seeds
**Sensible:** adjusted Rand index between runs ≥ 0.7. Below ~0.5 the partition is not reproducible at this resolution — lower `RES`.""")

code('''parts_seed = [Leiden(resolution=RES, seed=sd)(G) for sd in (0, 1, 2)]
for i in range(3):
    for j in range(i + 1, 3):
        print(f"seed {i} vs {j}: ARI = {ari(parts_seed[i].labels, parts_seed[j].labels):.3f}   (k = {parts_seed[i].k}, {parts_seed[j].k})")''')

md("""## 3. Agreement with the spectral picture
**Sensible:** each arm of the spectral star is (mostly) one colour; communities are contiguous regions, not salt-and-pepper.""")

code('''emb = Spectral(k=12)(G)
big = np.where(part.sizes() >= MIN_SHOW)[0]
lab = np.where(np.isin(part.labels, big), part.labels, -1)
fig, ax = plt.subplots(1, 2, figsize=(13, 6))
ax[0].scatter(emb.vecs[:, 0], emb.vecs[:, 1], s=2, alpha=.5, c=lab, cmap="tab20"); ax[0].set_title(f"Leiden communities on spectral coords ({len(big)} shown)")
ax[1].scatter(emb.vecs[:, 2], emb.vecs[:, 3], s=2, alpha=.5, c=lab, cmap="tab20"); ax[1].set_title("same, eigenvectors 3 x 4")
plt.show()
# how pure is each community in spectral terms? mean cosine of members to their community centroid
V = emb.vecs[:, :8] / np.linalg.norm(emb.vecs[:, :8], axis=1, keepdims=True)
rows = []
for c in big:
    m = part.labels == c; cen = V[m].mean(0); cen /= np.linalg.norm(cen)
    rows.append((c, int(m.sum()), round(float((V[m] @ cen).mean()), 3)))
display(pd.DataFrame(rows, columns=["community", "size", "mean cosine to centroid"]).set_index("community"))''')

md("""## 4. Timeline
**Sensible:** community count grows slowly with the graph; quality (modularity) stays in a sensible band (0.3–0.7 for co-citation).""")

code('''graphs, parts = {}, {}
for y in YEARS:
    t0 = time.time(); s = st.at(y)
    Gy = build(s, ("cluster", "cites", "cluster"), scotus(s), mode=MODE, min_degree=MIN_DEG)
    if Gy.n < 50: continue
    py = Leiden(resolution=RES)(Gy)
    graphs[y], parts[y] = Gy, py
    sz = py.sizes()
    print(f"{y}: {Gy.n:>7,} nodes  {py.k:>3} communities ({int((sz >= MIN_SHOW).sum())} >= {MIN_SHOW})  "
          f"largest {sz[:4].tolist()}  Q = {py.quality:.3f}   ({time.time()-t0:.1f}s)")''')

md("""## 5. Lineage
Communities in consecutive slices are linked by shared cases. `overlap` = share of the earlier community that lands in the later one; `jaccard` = symmetric.
**Sensible:** most big communities have one dominant successor (overlap > 0.5); the chains have recognisable names and run for several slices; splits happen where a doctrine actually grew (criminal procedure after 1960, say).""")

code('''ys = sorted(parts); plist = [parts[y] for y in ys]
lin = lineage(plist, min_overlap=0.1)
# strongest links between the last two slices
i = len(ys) - 2
tab = pd.DataFrame([(ys[a], ca, ys[b], cb, round(f, 2), round(j, 2)) for a, ca, b, cb, f, j in lin.edges if a == i],
                   columns=["from", "comm", "to", "comm'", "overlap", "jaccard"])
tab = tab[tab.comm.map(lambda c: plist[i].sizes()[c] >= MIN_SHOW)].sort_values(["comm", "overlap"], ascending=[True, False])
display(tab.groupby("comm").head(3).set_index(["from", "comm"]))''')

code('''def label_of(i, c, k=3):
    """Name a community by its top-k cases (by degree) in that slice."""
    y = ys[i]; G_, p_ = graphs[y], parts[y]
    m = np.where(p_.labels == c)[0]; top = m[np.argsort(-G_.degree[m])[:k]]
    return " / ".join(str(n)[:28] for n in names(G_.ids[top]).case_name)

chains = [ch for ch in lin.chains(min_jaccard=0.2) if parts[ys[ch[0][0]]].sizes()[ch[0][1]] >= MIN_SHOW]
print(f"{len(chains)} chains of length > 1 (jaccard >= 0.2)\\n")
for ch in chains[:25]:
    span = f"{ys[ch[0][0]]}-{ys[ch[-1][0]]}"
    sizes = [int(parts[ys[i]].sizes()[c]) for i, c in ch]
    print(f"{span:>9}  len {len(ch)}  sizes {sizes}")
    print(f"           start: {label_of(*ch[0])}")
    print(f"           end:   {label_of(*ch[-1])}")''')

code('''ev = lin.events(min_frac=0.3)
print("splits (a community with >1 significant successor):")
for (i, c), succ in list(ev["splits"].items())[:10]:
    if parts[ys[i]].sizes()[c] < MIN_SHOW: continue
    print(f"  {ys[i]} [{label_of(i, c, 2)}] ->", ", ".join(f"{ys[j]}:{cj} ({f:.2f})" for j, cj, f in succ))
print("\\nmerges (a community with >1 significant predecessor):")
for (j, cj), pred in list(ev["merges"].items())[:10]:
    if parts[ys[j]].sizes()[cj] < MIN_SHOW: continue
    print(f"  {ys[j]} [{label_of(j, cj, 2)}] <-", ", ".join(f"{ys[i]}:{ci} ({s:.2f})" for i, ci, s in pred))''')

md("""## 6. Flow picture
Bars = communities per slice (height ∝ size), lines = shared cases between consecutive slices (width ∝ count). The site's Sankey is this with labels.""")

code('''fig, ax = plt.subplots(figsize=(14, 7))
pos = {}
for xi, y in enumerate(ys):
    sz = parts[y].sizes(); y0 = 0
    for c, n in enumerate(sz):
        if n < MIN_SHOW: continue
        ax.add_patch(plt.Rectangle((xi - 0.08, y0), 0.16, n, color=plt.cm.tab20(c % 20)))
        pos[(xi, c)] = (y0, y0 + n); y0 += n * 1.08
for a, ca, b, cb, f, j in lin.edges:
    if (a, ca) in pos and (b, cb) in pos and j >= 0.05:
        (l0, l1), (r0, r1) = pos[(a, ca)], pos[(b, cb)]
        ax.plot([a + 0.08, b - 0.08], [(l0 + l1) / 2, (r0 + r1) / 2], lw=max(0.3, 8 * j), alpha=.35, color="gray")
ax.set_xticks(range(len(ys))); ax.set_xticklabels(ys); ax.set_ylabel("cases"); ax.set_xlim(-0.5, len(ys) - 0.5)
ax.set_ylim(0, max(e for s, e in pos.values()) * 1.02); ax.set_title(f"community flow, {COURT} {MODE}, resolution {RES}"); plt.show()''')

md("""## 7. Export tables
What the site needs: one row per (slice, case) with its community and chain, and one row per lineage edge. Set `EXPORT = True` to write Parquet.""")

code('''EXPORT = False
OUT = "out"
chain_id = {}                                        # (slice_index, community) -> chain number
for n, ch in enumerate(lin.chains(min_jaccard=0.2)):
    for key in ch: chain_id[key] = n
nodes = pd.concat([pd.DataFrame({"slice": y, "cluster_id": parts[y].ids, "community": parts[y].labels,
                                 "chain": [chain_id.get((i, c), -1) for c in parts[y].labels]})
                   for i, y in enumerate(ys)], ignore_index=True)
edges = pd.DataFrame([(ys[a], ca, ys[b], cb, int(lin.overlaps[a][ca, cb]), round(f, 4), round(j, 4))
                      for a, ca, b, cb, f, j in lin.edges],
                     columns=["from_slice", "from_comm", "to_slice", "to_comm", "shared", "overlap", "jaccard"])
labels = pd.DataFrame([(ys[i], c, int(parts[ys[i]].sizes()[c]), label_of(i, c, 3))
                       for i in range(len(ys)) for c in range(parts[ys[i]].k) if parts[ys[i]].sizes()[c] >= MIN_SHOW],
                      columns=["slice", "community", "size", "label"])
print(nodes.shape, edges.shape, labels.shape); display(labels.head(12))
if EXPORT:
    import os; os.makedirs(OUT, exist_ok=True)
    nodes.to_parquet(f"{OUT}/leiden_nodes_{COURT}.parquet", index=False)
    edges.to_parquet(f"{OUT}/leiden_lineage_{COURT}.parquet", index=False)
    labels.to_parquet(f"{OUT}/leiden_labels_{COURT}.parquet", index=False)
    print("written to", OUT)''')

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("notebooks/leiden_sanity.ipynb", "w"), indent=1)
print("wrote notebooks/leiden_sanity.ipynb")
