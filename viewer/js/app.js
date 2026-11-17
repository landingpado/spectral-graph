// 3-D citation-space view: three.js point cloud, tweened year by year on the GPU, in two projections
// (spectral eigenvector coordinates, or the precomputed force layout of the Leiden keyframes).
import * as THREE from '../vendor/three.module.js';
import { OrbitControls } from '../vendor/OrbitControls.js';
import { $, esc, clLink } from './common.js';
import { loadStore } from './data.js';

const status = m => { $('#status').textContent = m; };
const t0 = performance.now();
const MIX_MS = 600;                       // crossfade between projections

// ---------------------------------------------------------------- data
let store;
try { store = await loadStore('./data/'); }
catch (err) { status(`could not load viewer/data (${err.message}). Build it: python -m lawgraph.viewer_build graph.duckdb --out viewer/data`); throw err; }
const { M, years, T, N, cases, chain } = store;

const byId = new Map(cases.map(c => [c.id, c.i]));       // CourtListener cluster id -> row
const pal = new Float32Array(256 * 3).fill(0.35);
for (const [k, rgb] of Object.entries(M.palette)) { pal[k * 3] = rgb[0]; pal[k * 3 + 1] = rgb[1]; pal[k * 3 + 2] = rgb[2]; }
const labelYears = Object.keys(M.labels).map(Number).sort((a, b) => a - b);
const chainNames = {};   // chain id -> its most recent label text, for the legend
for (const y of labelYears) for (const l of M.labels[y]) if (l.chain != null) chainNames[l.chain] = l.name;

// ---------------------------------------------------------------- scene
const canvas = $('#c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x000000);
const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100); camera.position.set(0, 0, 2.6);
const controls = new OrbitControls(camera, canvas); controls.enableDamping = true; controls.dampingFactor = 0.08;

// Per track, positions at year i (A) and i+1 (B) and a (present at A, present at B) pair. The vertex shader
// tweens A->B with `t` inside each track and then blends the two tracks with `u` (0 = spectral, 1 = force).
const geo = new THREE.BufferGeometry();
const posA = new Float32Array(N * 3), posB = new Float32Array(N * 3), fposA = new Float32Array(N * 3), fposB = new Float32Array(N * 3);
const colA = new Float32Array(N * 3), colB = new Float32Array(N * 3);
const alive = new Float32Array(N * 2), falive = new Float32Array(N * 2), dim = new Float32Array(N).fill(1);
const hl = new Float32Array(N);                                    // 0..1: rows the year view asked us to highlight
geo.setAttribute('position', new THREE.BufferAttribute(posA, 3));
geo.setAttribute('posB', new THREE.BufferAttribute(posB, 3));
geo.setAttribute('fposA', new THREE.BufferAttribute(fposA, 3));
geo.setAttribute('fposB', new THREE.BufferAttribute(fposB, 3));
geo.setAttribute('colA', new THREE.BufferAttribute(colA, 3));
geo.setAttribute('colB', new THREE.BufferAttribute(colB, 3));
geo.setAttribute('alive', new THREE.BufferAttribute(alive, 2));
geo.setAttribute('falive', new THREE.BufferAttribute(falive, 2));
geo.setAttribute('dim', new THREE.BufferAttribute(dim, 1));
geo.setAttribute('hl', new THREE.BufferAttribute(hl, 1));
geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(), 4);      // positions change on the GPU; never cull
const mat = new THREE.ShaderMaterial({
  uniforms: { t: { value: 0 }, u: { value: 0 }, size: { value: 3.0 * renderer.getPixelRatio() } },
  transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  vertexShader: `
    attribute vec3 posB; attribute vec3 fposA; attribute vec3 fposB; attribute vec3 colA; attribute vec3 colB;
    attribute vec2 alive; attribute vec2 falive; attribute float dim; attribute float hl;
    uniform float t; uniform float u; uniform float size; varying vec3 vc; varying float va;
    void track(vec3 pa, vec3 pb, vec2 al, out vec3 p, out float w){
      bool a = al.x > 0.5, b = al.y > 0.5;
      p = (a && b) ? mix(pa, pb, t) : (b ? pb : pa);
      w = (a && b) ? 1.0 : (b ? t : (a ? 1.0 - t : 0.0));
    }
    void main(){
      vec3 ps; vec3 pf; float ws; float wf;
      track(position, posB, alive, ps, ws); track(fposA, fposB, falive, pf, wf);
      vec3 p; float al;
      if (ws > 0.0 && wf > 0.0) { p = mix(ps, pf, u); al = mix(ws, wf, u); }
      else if (wf > 0.0) { p = pf; al = wf * u; }
      else { p = ps; al = ws * (1.0 - u); }
      float em = max(dim, hl);
      vc = mix(mix(colA, colB, t), vec3(1.0, 0.96, 0.82), 0.75 * hl); va = al * em;
      vec4 mv = modelViewMatrix * vec4(p, 1.0);
      gl_PointSize = size * (0.6 + 0.8 * em + 1.6 * hl) * (2.0 / -mv.z);
      gl_Position = projectionMatrix * mv;
      if (al <= 0.0) gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    }`,
  fragmentShader: `
    varying vec3 vc; varying float va;
    void main(){ vec2 d = gl_PointCoord - 0.5; float r = dot(d, d); if (r > 0.25) discard;
      float a = smoothstep(0.25, 0.08, r) * 0.85 * va; gl_FragColor = vec4(vc * (0.6 + 0.4 * va), a); }`
});
const points = new THREE.Points(geo, mat); points.frustumCulled = false; scene.add(points);

// trail (selected case's path through time)
const trailGeo = new THREE.BufferGeometry(); const trailMat = new THREE.LineBasicMaterial({ color: 0xf6b26b, transparent: true, opacity: 0.9 });
const trail = new THREE.Line(trailGeo, trailMat); trail.frustumCulled = false; scene.add(trail);
let trailCount = 0, trailTrack = null, pendingTrail = null;
const markGeo = new THREE.SphereGeometry(0.012, 12, 12); const mark = new THREE.Mesh(markGeo, new THREE.MeshBasicMaterial({ color: 0xf6b26b })); mark.visible = false; scene.add(mark);

// ---------------------------------------------------------------- time
let ti = -1, tfrac = 0, playing = false, secPerYear = 0.5, lastTs = 0;
let track = 'spectral', u = 0;            // u: 0 = spectral, 1 = force, in between while crossfading
let mixing = false, mixStart = -1, mixFrom = 0, mixTo = 0;
$('#t').max = T - 1;
function fillColours(col, k) {
  for (let i = 0, o = 0; i < N; i++, o += 3) { const c = chain[k * N + i] * 3; col[o] = pal[c]; col[o + 1] = pal[c + 1]; col[o + 2] = pal[c + 2]; }
}
/** Decode years i and i+1 into the A/B buffers (after a year step, a dimension change, a projection switch).
 *  The force buffers are only kept current while that projection is on screen. */
function refill() {
  const i = ti, j = Math.min(T - 1, i + 1);
  store.decode(i, posA, alive, 0); store.decode(j, posB, alive, 1);
  fillColours(colA, i); fillColours(colB, j);
  for (const a of ['position', 'posB', 'colA', 'colB', 'alive']) geo.attributes[a].needsUpdate = true;
  if (track === 'force' || mixing) {
    store.decodeForce(i, fposA, falive, 0); store.decodeForce(j, fposB, falive, 1);
    for (const a of ['fposA', 'fposB', 'falive']) geo.attributes[a].needsUpdate = true;
  }
}
const yearEl = $('#year'); let shownYear = NaN;
function showYear() {       // called every frame while playing: only touch the DOM when the number changes
  const y = Math.round(years[ti] + tfrac * (years[Math.min(T - 1, ti + 1)] - years[ti]));
  if (y !== shownYear) { shownYear = y; yearEl.textContent = y; }
}
async function setYear(i, frac = 0) {
  i = Math.max(0, Math.min(T - 1, i)); const j = Math.min(T - 1, i + 1);
  if (!store.loaded(i) || !store.loaded(j)) await Promise.all([store.ensureYear(i), store.ensureYear(j)]);
  const stepped = i !== ti;
  ti = i; tfrac = frac; mat.uniforms.t.value = frac;
  if (stepped) { refill(); applyFilter(); labelsDirty = true; showEigenvalues(); hideTip(); }
  showYear(); $('#t').value = i;
  updateTrailMark();
}
$('#t').addEventListener('input', e => { playing = false; glideTo = -1; $('#play').textContent = '▶ Play'; setYear(+e.target.value); });
$('#play').onclick = () => { playing = !playing; glideTo = -1; $('#play').textContent = playing ? '❚❚ Pause' : '▶ Play'; lastTs = 0; };
$('#faster').onclick = () => { secPerYear = Math.max(0.1, secPerYear / 1.5); $('#speed').firstChild.textContent = secPerYear.toFixed(2) + ' s / yr '; };
$('#slower').onclick = () => { secPerYear = Math.min(5, secPerYear * 1.5); $('#speed').firstChild.textContent = secPerYear.toFixed(2) + ' s / yr '; };

// ---------------------------------------------------------------- projection: spectral | force
const camFrom = new THREE.Vector3(), camTo = new THREE.Vector3(), tgtFrom = new THREE.Vector3(); let camTween = false;
function showSubtitle() {
  $('#sub').textContent = track === 'force'
    ? 'SCOTUS co-citation · force layout (ForceAtlas2 + Leiden community gravity) · Leiden communities'
    : `SCOTUS co-citation · spectral ${store.dims.map(k => 'ev' + (k + 1)).join(' · ')} · Leiden communities`;
}
/** Switch projection with a ~600 ms crossfade. Colours, selection and labels carry over; the pinned case's
 *  trail is recomputed in the new track and swapped in half-way through. */
async function setTrack(name, animate = true) {
  if (name === track || (name !== 'spectral' && name !== 'force')) return;
  if (name === 'force') { if (!store.hasForce) return; await store.ensureForce(); }
  track = name; mixing = true;                       // refill() now also keeps the force buffers current
  refill();
  mixFrom = u; mixTo = name === 'force' ? 1 : 0; mixStart = animate ? -1 : -Infinity;
  if (name === 'force') {                            // the force layout is flat: turn to face it
    camFrom.copy(camera.position); tgtFrom.copy(controls.target);
    camTo.set(0, 0, camera.position.distanceTo(controls.target)); camTween = true;
  } else camTween = false;
  document.querySelectorAll('#proj button').forEach(b => b.classList.toggle('on', b.dataset.track === name));
  $('#axes').classList.toggle('off', name === 'force');
  showSubtitle(); labelsDirty = true; hideTip();
  if (pinned >= 0) { const i = pinned; const pts = await trailPoints(i, name); if (i === pinned && track === name) pendingTrail = pts; }
}
/** One frame of the crossfade (called from the render loop; allocates nothing). */
function stepMix(ts) {
  if (mixStart === -1) mixStart = ts;
  const p = Math.min(1, (ts - mixStart) / MIX_MS), e = p * p * (3 - 2 * p);
  u = mixFrom + (mixTo - mixFrom) * e; mat.uniforms.u.value = u;
  if (camTween) { camera.position.lerpVectors(camFrom, camTo, e); controls.target.lerpVectors(tgtFrom, scene.position, e); }
  if (pendingTrail && p >= 0.5) { showTrail(pendingTrail, track); pendingTrail = null; }
  trailMat.opacity = 0.9 * Math.abs(2 * p - 1);     // old trail fades out, new one fades in
  if (p >= 1) { mixing = false; camTween = false; trailMat.opacity = 0.9; if (pendingTrail) { showTrail(pendingTrail, track); pendingTrail = null; } }
  updateTrailMark();
}
document.querySelectorAll('#proj button').forEach(b => { b.onclick = () => setTrack(b.dataset.track); });
if (!store.hasForce) { const b = $('#proj button[data-track=force]'); b.disabled = true; b.title = 'viewer/data has no force track (rebuild without --no-force)'; }

// ---------------------------------------------------------------- eigenvector picker
// Three dropdowns choose which of the K aligned eigenvectors are x, y, z. Changing one re-decodes the two
// year buffers from the Int16 chunks already in memory; nothing is fetched again.
const axSel = [0, 1, 2].map(a => $('#ax' + a));
const sameDims = (a, b) => !!b && a.length === b.length && a.every((v, k) => v === b[k]);
for (const sel of axSel) for (let d = 0; d < store.K; d++) sel.appendChild(new Option(`ev${d + 1}`, d));
function showEigenvalues() {
  for (const sel of axSel) for (let d = 0; d < store.K; d++) {
    const lam = store.eigenvalue(ti, d), txt = lam == null ? `ev${d + 1}` : `ev${d + 1} · λ ${lam.toFixed(3)}`;
    if (sel.options[d].text !== txt) sel.options[d].text = txt;
  }
}
function setDims(d) {
  store.setDims(d);
  axSel.forEach((sel, a) => { sel.value = store.dims[a]; });
  $('#robust').classList.toggle('on', sameDims(store.dims, M.robust_dims));
  showSubtitle();
  if (ti < 0) return;
  refill(); labelsDirty = true; updateTrailMark();
  if (track === 'spectral') drawTrail();
}
axSel.forEach((sel, a) => sel.addEventListener('change', () => {
  const d = store.dims.slice(), v = +sel.value, other = d.indexOf(v);
  if (other >= 0 && other !== a) d[other] = d[a];       // picking an axis already in use swaps the two
  d[a] = v; setDims(d);
}));
if (M.robust_dims) $('#robust').onclick = () => setDims(sameDims(store.dims, M.robust_dims) ? [0, 1, 2] : M.robust_dims);
else $('#robust').disabled = true;

// ---------------------------------------------------------------- labels (HTML overlay, projected)
// The set of labels changes only when playback crosses a keyframe; per frame we just re-project anchors
// into preallocated scratch space and touch the DOM when a label actually moved.
const lblRoot = $('#lbl'); const v3 = new THREE.Vector3();
let labelsDirty = true, labK0 = NaN, labK1 = NaN;
const labItems = [];        // { el, a, b: spectral anchors at the two keyframes, fa, fb: force anchors, sx, sy, vis }
const sameLabel = (l, m) => (l.chain != null && m.chain != null) ? l.chain === m.chain : l.name === m.name;
function rebuildLabels() {
  const y = years[ti]; let k0 = NaN, k1 = NaN;
  for (const k of labelYears) { if (k <= y) k0 = k; else { k1 = k; break; } }
  labK0 = k0; labK1 = k1; labelsDirty = false;
  const L0 = Number.isNaN(k0) ? [] : M.labels[k0], L1 = Number.isNaN(k1) ? null : M.labels[k1];
  const old = new Map(labItems.map(it => [it.key, it])); labItems.length = 0;
  for (const l of L0) {
    const key = l.chain != null ? 'c' + l.chain : 'n' + l.name;
    let it = old.get(key); old.delete(key);
    if (!it) { it = { key, el: document.createElement('div'), a: [0, 0, 0], b: [0, 0, 0], fa: [0, 0, 0], fb: [0, 0, 0], moves: false, sx: NaN, sy: NaN, vis: null };
      it.el.className = 'lab'; lblRoot.appendChild(it.el); }
    if (it.el.textContent !== l.name) it.el.textContent = l.name;
    it.label = l; it.w = it.el.offsetWidth; it.h = it.el.offsetHeight;
    const m = L1 && L1.find(x => sameLabel(l, x)); it.moves = !!m;
    store.anchor(l, it.a, 'spectral'); store.anchor(m || l, it.b, 'spectral');
    store.anchor(l, it.fa, 'force'); store.anchor(m || l, it.fb, 'force');
    labItems.push(it);
  }
  for (const it of old.values()) it.el.remove();
}
function updateLabels() {
  if (labelsDirty) rebuildLabels();
  const f = Number.isNaN(labK1) ? 0 : (years[ti] + tfrac * (years[Math.min(T - 1, ti + 1)] - years[ti]) - labK0) / (labK1 - labK0);
  for (let n = 0; n < labItems.length; n++) {
    const it = labItems[n], a = it.a, b = it.b, fa = it.fa, fb = it.fb, g = it.moves ? f : 0;
    const x = a[0] + g * (b[0] - a[0]), y = a[1] + g * (b[1] - a[1]), z = a[2] + g * (b[2] - a[2]);
    const fx = fa[0] + g * (fb[0] - fa[0]), fy = fa[1] + g * (fb[1] - fa[1]), fz = fa[2] + g * (fb[2] - fa[2]);
    v3.set(x + u * (fx - x), y + u * (fy - y), z + u * (fz - z)).project(camera);
    const sx = Math.round((v3.x + 1) * 0.5 * innerWidth), sy = Math.round((1 - v3.y) * 0.5 * innerHeight), vis = v3.z < 1;
    if (sx !== it.sx || sy !== it.sy) { it.el.style.left = sx + 'px'; it.el.style.top = sy + 'px'; it.sx = sx; it.sy = sy; }
    if (vis !== it.vis) { it.el.style.opacity = vis ? 0.9 : 0; it.vis = vis; }
  }
}

/** The label under the mouse, if any. Labels ignore pointer events (so dragging the cloud works through
 *  them); this is a plain rectangle test against where updateLabels() last put them. */
function labelAt(mx, my) {
  for (let n = 0; n < labItems.length; n++) { const it = labItems[n];
    if (it.vis && Math.abs(mx - it.sx) <= it.w / 2 + 2 && Math.abs(my - it.sy) <= it.h / 2 + 2) return it.label; }
  return null;
}
/** Hover text for a community label: the full case names, the community's size and its lineage chain. */
function labelTip(l) {
  const names = (l.full || [l.name]).map(esc).join('<br>');
  const meta = [l.size != null ? `${l.size.toLocaleString()} cases in this community` : null, l.chain != null ? `lineage chain ${l.chain}` : 'no lineage chain'].filter(Boolean).join(' · ');
  return `${names}<br><span class="meta">${meta}</span>`;
}

// ---------------------------------------------------------------- picking, search, trail, neighbours
let hover = -1, hoverLabel = null, pinned = -1;
const tip = $('#tip'); const mvp = new THREE.Matrix4();
/** The scene moved under a resting mouse (year step, projection switch): what the tooltip said is stale. */
function hideTip() { hover = -1; hoverLabel = null; tip.style.display = 'none'; }
/** Where point i is drawn right now: the same year tween and track blend the vertex shader does. */
function pointNow(i, out) {
  const o = i * 3, sa = alive[i * 2] > 0.5, sb = alive[i * 2 + 1] > 0.5, fa = falive[i * 2] > 0.5, fb = falive[i * 2 + 1] > 0.5;
  const hasS = (sa || sb) && u < 1, hasF = (fa || fb) && u > 0;
  if (!hasS && !hasF) return false;
  let x = 0, y = 0, z = 0;
  if (sa || sb) { const f = sa && sb ? tfrac : (sb ? 1 : 0); x = posA[o] + f * (posB[o] - posA[o]); y = posA[o + 1] + f * (posB[o + 1] - posA[o + 1]); z = posA[o + 2] + f * (posB[o + 2] - posA[o + 2]); }
  if (fa || fb) { const f = fa && fb ? tfrac : (fb ? 1 : 0);
    const fx = fposA[o] + f * (fposB[o] - fposA[o]), fy = fposA[o + 1] + f * (fposB[o + 1] - fposA[o + 1]), fz = fposA[o + 2] + f * (fposB[o + 2] - fposA[o + 2]);
    if (sa || sb) { x += u * (fx - x); y += u * (fy - y); z += u * (fz - z); } else { x = fx; y = fy; z = fz; } }
  out.set(x, y, z); return true;
}
function nearest(mx, my) {
  if (ti < 0) return -1;
  mvp.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse); const e = mvp.elements;
  const hw = innerWidth / 2, hh = innerHeight / 2; let best = -1, bd = 12 * 12;
  for (let i = 0; i < N; i++) {
    if (!pointNow(i, v3)) continue;
    const x = v3.x, y = v3.y, z = v3.z, w = e[3] * x + e[7] * y + e[11] * z + e[15]; if (w <= 0) continue;
    const sx = (e[0] * x + e[4] * y + e[8] * z + e[12]) / w * hw + hw, sy = hh - (e[1] * x + e[5] * y + e[9] * z + e[13]) / w * hh;
    const d = (sx - mx) * (sx - mx) + (sy - my) * (sy - my); if (d < bd) { bd = d; best = i; }
  }
  return best;
}
let lastMove = 0;
canvas.addEventListener('mousemove', e => { const now = performance.now(); if (now - lastMove < 40) return; lastMove = now;
  hoverLabel = labelAt(e.clientX, e.clientY);
  hover = hoverLabel ? -1 : nearest(e.clientX, e.clientY);
  if (hoverLabel) tip.innerHTML = labelTip(hoverLabel);
  else if (hover >= 0) { const c = cases[hover]; tip.textContent = `${c.name} (${c.year ?? '?'})`; }
  if (hoverLabel || hover >= 0) { tip.style.display = 'block'; tip.style.left = (e.clientX + 12) + 'px'; tip.style.top = (e.clientY + 12) + 'px'; }
  else tip.style.display = 'none'; });
canvas.addEventListener('mouseleave', hideTip);
canvas.addEventListener('click', () => {
  if (hoverLabel) { const top = (hoverLabel.ids || []).map(id => byId.get(id)).find(i => i !== undefined); if (top !== undefined) select(top); }   // a label pins its top case
  else if (hover >= 0) select(hover);
});

function summaryFor(name) { const n = name.toLowerCase(); for (const [k, s] of Object.entries(M.summaries)) if (n.includes(k)) return s; return null; }
const INFO_HINT = '<span class="meta">Hover a point. Click to pin it, show its trail and co-cited neighbours.</span>';
/** The pinned case's path in one track; needs every year (loads the remaining chunks on first use). */
async function trailPoints(i, name) {
  await store.ensureAll(); if (name === 'force') await store.ensureForce();
  return store.trail(i, name);
}
function showTrail(pts, name) {
  trailCount = pts.length / 3; trailTrack = name;
  trailGeo.setAttribute('position', new THREE.BufferAttribute(pts, 3));
  updateTrailMark();
}
async function drawTrail() {
  const i = pinned, name = track; if (i < 0) return;
  const pts = await trailPoints(i, name);
  if (i === pinned && name === track) { pendingTrail = null; showTrail(pts, name); }
}
async function select(i) {
  pinned = i; const c = cases[i]; const s = summaryFor(c.name);
  $('#info').innerHTML = `<h3>${esc(c.name)}</h3><div class="meta">${c.year ?? ''}${s ? ' · ' + esc(s.citation) : ''} · ${clLink(c.id)}</div>` +
    (s ? `<p>${esc(s.summary)}</p>` : '') + `<div class="meta" style="margin-top:8px">co-cited with (this decade)</div><ul id="nbrs"><li class="meta">loading…</li></ul>`;
  applyFilter();
  loadNeighbors(i);
  await drawTrail();
}
function updateTrailMark() {
  if (pinned < 0 || !pointNow(pinned, v3)) { mark.visible = false; return; }
  mark.visible = true; mark.position.copy(v3);
}
function clearSelection() { pinned = -1; mark.visible = false; trailCount = 0; trailTrack = null; pendingTrail = null;
  trailGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(0), 3));
  applyFilter(); $('#info').innerHTML = INFO_HINT; }
window.addEventListener('keydown', e => { if (e.key === 'Escape') clearSelection(); if (e.key === ' ' && document.activeElement !== $('#q')) { e.preventDefault(); $('#play').click(); } });

const yearIndex = y => Math.max(0, years.findIndex(v => v >= y));
/** A CourtListener link inside a clickable row opens the link only; it does not also pin the row. */
const keepLinkClicks = el => el.querySelectorAll('a.cl').forEach(a => a.addEventListener('click', e => e.stopPropagation()));
$('#q').addEventListener('input', e => {
  const q = e.target.value.trim().toLowerCase(); const ul = $('#results'); ul.innerHTML = ''; if (q.length < 3) return;
  const hits = cases.filter(c => c.name.toLowerCase().includes(q)).slice(0, 30);
  for (const c of hits) { const li = document.createElement('li'); li.innerHTML = `${esc(c.name)}<span>${c.year ?? ''}</span> ${clLink(c.id, '↗')}`;
    li.onclick = () => { select(c.i); if (c.year) setYear(yearIndex(c.year)); }; keepLinkClicks(li); ul.appendChild(li); }
});

// neighbours: one small binary per decade, loaded lazily
const nbrCache = {};
async function loadNeighbors(i) {
  const ul = $('#nbrs'); const nb = M.neighbors; if (!nb) { ul.innerHTML = '<li class="meta">no neighbour table exported</li>'; return; }
  const y = years[ti]; const ds = nb.decades.filter(d => d <= y); const d = ds.length ? Math.max(...ds) : nb.decades[0];
  try {
    if (!nbrCache[d]) nbrCache[d] = fetch(store.base + nb.file.replace('{decade}', d)).then(r => r.arrayBuffer()).then(buf => ({ idx: new Int32Array(buf, 0, N * nb.k), w: new Float32Array(buf, N * nb.k * 4, N * nb.k) }));
    const { idx, w } = await nbrCache[d]; if (i !== pinned || !$('#nbrs')) return; ul.innerHTML = '';
    let any = false;
    for (let s = 0; s < nb.k; s++) { const j = idx[i * nb.k + s]; if (j < 0) continue; any = true;
      const li = document.createElement('li'); li.innerHTML = `${esc(cases[j].name)} (${cases[j].year ?? ''}) · ${w[i * nb.k + s]} ${clLink(cases[j].id, '↗')}`;
      li.onclick = () => select(j); keepLinkClicks(li); ul.appendChild(li); }
    if (!any) ul.innerHTML = '<li class="meta">none in this decade</li>';
    $('#nbrs').previousElementSibling.textContent = `co-cited with (${d}s)`;
  } catch (err) { ul.innerHTML = `<li class="meta">neighbours unavailable (${esc(String(err).slice(0, 60))})</li>`; console.warn(err); }
}

// legend: chains
const leg = $('#legend'); let chainFilter = null;
for (const [k, rgb] of Object.entries(M.palette)) {
  const d = document.createElement('div'); d.className = 'chain'; d.dataset.k = k;
  d.innerHTML = `<span class="sw" style="background:rgb(${rgb.map(v => Math.round(v * 255)).join(',')})"></span>${esc(chainNames[k] || 'chain ' + k)}`;
  d.onclick = () => { chainFilter = chainFilter === +k ? null : +k; document.querySelectorAll('.chain').forEach(e => e.classList.toggle('on', chainFilter !== null && +e.dataset.k === chainFilter)); applyFilter(); };
  leg.appendChild(d);
}
/** Per-point emphasis: legend filter first, then the pinned case stands out against everything else; while
 *  the year view is highlighting rows, everything else steps back. */
function applyFilter() {
  if (ti < 0) return;
  const base = (pinned >= 0 && chainFilter === null ? 0.55 : 1) * (hlCount > 0 ? 0.45 : 1);
  for (let i = 0; i < N; i++) dim[i] = chainFilter === null ? base : (chain[ti * N + i] === chainFilter ? 1 : 0.08);
  if (pinned >= 0) dim[pinned] = 1;
  geo.attributes.dim.needsUpdate = true;
}

// ---------------------------------------------------------------- year view: drawer + highlighted rows
// years.html (in the drawer below, or in another tab) posts the SCOTUS rows of its table on a BroadcastChannel.
// Rows that are in this cloud are drawn larger and brighter; while the year view animates, the last
// HL_WINDOW years stay lit and fade, so decisions can be seen arriving in citation space. The cloud follows
// to the year after the table's year: a snapshot is "as of 1 January", so that is the first one that can
// contain decisions filed during the table's year.
const HL_WINDOW = 10;
const hlYears = new Map();                 // table year -> rows of this cloud
let hlCount = 0, hlLast = null, follow = true, glideTo = -1;
function applyHighlight() {
  hl.fill(0); hlCount = 0; let visible = 0;
  for (const [y, rows] of hlYears) { const w = 1 - (hlLast - y) / HL_WINDOW; for (const r of rows) hl[r] = Math.max(hl[r], w); }
  for (let i = 0; i < N; i++) if (hl[i] > 0) { hlCount++; if (alive[i * 2] > 0.5) visible++; }
  geo.attributes.hl.needsUpdate = true; applyFilter();
  return visible;
}
function showBanner(m, inCloud, visible) {
  const b = $('#hlbar'); if (hlLast === null) { b.style.display = 'none'; return; }
  b.style.display = 'block';
  $('#hltext').textContent = `Year view ${m.year}: ${m.ids.length} SCOTUS row${m.ids.length === 1 ? '' : 's'} of ${m.total} · ${inCloud} in this cloud` +
    (hlYears.size > 1 ? ` · ${hlCount} highlighted (last ${hlYears.size} years)` : '') + ` · ${visible} visible in ${years[ti]}`;
}
function onYearRows(m) {
  if (!Array.isArray(m.ids)) return;
  const rows = []; for (const id of m.ids) { const r = byId.get(id); if (r !== undefined) rows.push(r); }
  if (hlLast !== null && (m.year < hlLast || m.year - hlLast > 1 || (!m.playing && m.year !== hlLast))) hlYears.clear();   // a jump starts afresh
  hlYears.set(m.year, rows); hlLast = m.year;
  for (const y of [...hlYears.keys()]) if (m.year - y >= HL_WINDOW) hlYears.delete(y);
  const target = yearIndex(m.year + 1);
  const done = () => showBanner(m, rows.length, applyHighlight());
  if (!follow || years[T - 1] < m.year || years[0] > m.year + 1) { done(); return; }
  if (playing) { playing = false; $('#play').textContent = '▶ Play'; }
  if (target === ti + 1 && m.playing) { glideTo = target; lastTs = 0; done(); }        // one step ahead: glide there
  else if (target !== ti || tfrac !== 0) { glideTo = -1; setYear(target).then(done); }
  else done();
}
function clearHighlight() { hlYears.clear(); hlLast = null; glideTo = -1; applyHighlight(); $('#hlbar').style.display = 'none'; }
$('#hlclear').onclick = clearHighlight;
const chan = 'BroadcastChannel' in window ? new BroadcastChannel('lawgraph') : null;
if (chan) chan.onmessage = e => { const m = e.data || {};
  if (m.type === 'rows') onYearRows(m);
  else if (m.type === 'pin' && byId.has(m.id)) { select(byId.get(m.id)); if (m.year) setYear(yearIndex(m.year)); } };

let drawer = null;
function toggleDrawer(open = !document.body.classList.contains('drawer')) {
  if (open && !drawer) { drawer = document.createElement('iframe'); drawer.id = 'yearsFrame'; drawer.title = 'Year view'; drawer.src = './years.html?embed=1'; $('#drawer').appendChild(drawer); }
  document.body.classList.toggle('drawer', open); $('#yearsBtn').classList.toggle('on', open);
  if (!open) clearHighlight();
  resize();
}
$('#yearsBtn').onclick = () => toggleDrawer();

// ---------------------------------------------------------------- start: optional deep link, then the loop
// index.html?year=1973&case=108713&track=force&dims=0,1,3   (case = CourtListener cluster id)
const params = new URLSearchParams(location.search);
function resize() {
  renderer.setSize(innerWidth, innerHeight, false); camera.aspect = innerWidth / innerHeight;
  // centre the scene in the area the side panel leaves free (labels and picking use the same projection)
  const panel = $('#panel').getBoundingClientRect(), shift = innerWidth > 900 ? Math.round((innerWidth - panel.left) / 2) : 0;
  const lift = document.body.classList.contains('drawer') ? Math.round($('#drawer').getBoundingClientRect().height / 2) : 0;
  camera.setViewOffset(innerWidth, innerHeight, shift, lift, innerWidth, innerHeight);
  camera.updateProjectionMatrix(); labelsDirty = true;
}
addEventListener('resize', resize); resize();
const startDims = (params.get('dims') || '').split(',').map(Number);
setDims(startDims.length === 3 && startDims.every(d => Number.isInteger(d) && d >= 0 && d < store.K) ? startDims : [0, 1, 2]);
await setYear(params.has('year') ? yearIndex(+params.get('year')) : 0);
status(`${N.toLocaleString()} cases · ${years[0]}–${years[T - 1]} · ${store.K} dimensions`);
let frames = 0;
function loop(ts) {
  requestAnimationFrame(loop);
  if (mixing) stepMix(ts);
  controls.update();
  if (playing || glideTo >= 0) {
    const stopAt = playing ? T - 1 : glideTo, sec = playing ? secPerYear : 0.4;      // following the year view: a quick glide to its year
    if (lastTs) { let f = tfrac + (ts - lastTs) / 1000 / sec; let i = ti;
      while (f >= 1 && i < stopAt) { f -= 1; i += 1; }
      if (i >= stopAt) { f = 0; i = stopAt; if (playing) { playing = false; $('#play').textContent = '▶ Play'; } glideTo = -1; }
      if (i !== ti) setYear(i, f); else { tfrac = f; mat.uniforms.t.value = f; showYear(); updateTrailMark(); } }
    lastTs = ts; }
  updateLabels(); renderer.render(scene, camera); frames++;
}
requestAnimationFrame(loop);
const firstPaintMs = Math.round(performance.now() - t0);
// everything else arrives in the background: remaining spectral decades, then the (small) force track
store.prefetch(ti).then(() => store.ensureForce()).catch(err => console.warn('background chunk load failed', err));
if (params.get('track') === 'force') await setTrack('force', false);
if (params.has('case') && byId.has(+params.get('case'))) select(byId.get(+params.get('case')));
if (params.has('years')) toggleDrawer(true);
chan?.postMessage({ type: 'hello' });                 // a year view that is already open re-sends its rows

// handle for the headless test (tests/test_viewer.mjs) and for poking around in the console
window.__viewer = {
  ready: true, store, firstPaintMs, setYear, select, clearSelection, setDims, setTrack, toggleDrawer, clearHighlight, camera, controls,
  state: () => ({ ti, year: years[ti], tfrac, playing, pinned, pinnedName: pinned >= 0 ? cases[pinned].name : null,
    dims: store.dims.slice(), track, u, mixing, trailPoints: trailCount, trailTrack, frames, labels: labItems.map(it => it.el.textContent),
    labelBoxes: labItems.map(it => ({ name: it.label.name, x: it.sx, y: it.sy, w: it.w, h: it.h, vis: it.vis })),
    alive: alive.reduce((s, v, k) => s + (k % 2 === 0 ? v : 0), 0), aliveForce: falive.reduce((s, v, k) => s + (k % 2 === 0 ? v : 0), 0),
    markVisible: mark.visible, mark: mark.position.toArray(), highlighted: hlCount, highlightYears: [...hlYears.keys()],
    drawer: document.body.classList.contains('drawer'), banner: $('#hlbar').style.display !== 'none' ? $('#hltext').textContent : null }),
};
