// Data layer: manifest, lazily loaded Int16 position chunks, and decoding straight into GPU buffers.
//
// Positions are Int16 (frames_in_chunk, N, K) row-major, -32768 = absent; the true coordinate is
//   v / 32767 * half[d] + centre[d]
// The view subtracts centre[d] again (so the bounding box sits on the origin) and multiplies by one scale
// shared by the three displayed dimensions, so decode() writes  v / 32767 * half[d] * scale.
//
// There are two position tracks. "spectral" has K dimensions for every year. "force" (optional) has 2-D
// positions at the Leiden keyframes only; years in between are interpolated linearly here, z = 0. A case is
// drawn in the force track only in years where the spectral track has it too, so both projections show the
// same set of cases and a crossfade never makes points appear or vanish.
import { getJSON, getBuffer } from './common.js';

const ABSENT = -32768;
const scratch3 = [0, 0, 0];

export async function loadStore(base = './data/') {
  const M = await getJSON(base + 'manifest.json');
  const years = M.years, T = years.length, N = M.N;
  // manifests written before the K-dimensional export have no tracks / K
  const sp = M.tracks?.spectral ?? { K: M.K ?? 3, years, centre: M.centre, half: M.half, chunks: M.chunks };
  const K = sp.K;
  const [cases, chainBuf] = await Promise.all([getJSON(base + 'cases.json'), getBuffer(base + 'chain.bin')]);
  const chain = new Uint8Array(chainBuf);                      // (T, N), 255 = none

  // year index -> { q: Int16Array of its chunk, off: offset of this year's first value }
  const raw = new Array(T).fill(null);
  const chunkOf = new Array(T).fill(null);
  sp.chunks.forEach(c => c.years.forEach((y, j) => { chunkOf[years.indexOf(y)] = { c, j }; }));
  const pending = new Map();
  function loadChunk(c) {
    if (!pending.has(c.file)) pending.set(c.file, getBuffer(base + c.file).then(buf => {
      const q = new Int16Array(buf);
      if (q.length !== c.years.length * N * K) throw new Error(`${c.file}: ${q.length} values, expected ${c.years.length * N * K}`);
      c.years.forEach((y, j) => { raw[years.indexOf(y)] = { q, off: j * N * K }; });
    }));
    return pending.get(c.file);
  }

  // ---- force track: keyframes only
  const ft = M.tracks?.force ?? null;
  const fraw = ft ? new Array(ft.years.length).fill(null) : [];
  // View scale for the force track. The layout is already centred on its centre of mass, and a few far-flung
  // cases stretch its bounding box, so frame it by the RMS radius the build reports (3 radii fill the view)
  // rather than by the box; older exports without a radius fall back to the box, centred like the spectral track.
  const fradius = ft && ft.radius ? Math.max(...Object.values(ft.radius)) : 0;
  const fscale = ft ? (fradius > 0 ? 1 / (3 * fradius) : 1 / Math.max(ft.half[0], ft.half[1])) : 1;
  const fox = ft && fradius > 0 ? ft.centre[0] : 0, foy = ft && fradius > 0 ? ft.centre[1] : 0;
  // for every year index: the keyframes on either side (k1 = -1 past the last one) and the blend between them
  const fk0 = new Int32Array(T).fill(-1), fk1 = new Int32Array(T).fill(-1), fmix = new Float32Array(T);
  if (ft) for (let ti = 0; ti < T; ti++) {
    const y = years[ti]; let a = -1;
    for (let k = 0; k < ft.years.length; k++) if (ft.years[k] <= y) a = k;
    if (a < 0) { fk0[ti] = 0; }                                         // before the first keyframe: hold it
    else if (a === ft.years.length - 1 || ft.years[a] === y) { fk0[ti] = a; }
    else { fk0[ti] = a; fk1[ti] = a + 1; fmix[ti] = (y - ft.years[a]) / (ft.years[a + 1] - ft.years[a]); }
  }
  let forceReady = null;
  function loadForce() {
    if (!ft) return Promise.resolve();
    if (!forceReady) forceReady = Promise.all(ft.chunks.map(c => getBuffer(base + c.file).then(buf => {
      const q = new Int16Array(buf);
      if (q.length !== c.years.length * N * 2) throw new Error(`${c.file}: ${q.length} values, expected ${c.years.length * N * 2}`);
      c.years.forEach((y, j) => { fraw[ft.years.indexOf(y)] = { q, off: j * N * 2 }; });
    })));
    return forceReady;
  }
  /** Force position of case i at year ti into out[0..2]; false if it has none. Assumes the chunks are loaded. */
  function forcePoint(ti, i, out) {
    const r = raw[ti]; if (!r || r.q[r.off + i * K] === ABSENT) return false;        // not in the cloud that year
    const A = fraw[fk0[ti]]; if (!A) return false;
    const pa = A.off + i * 2, hasA = A.q[pa] !== ABSENT;
    const B = fk1[ti] >= 0 ? fraw[fk1[ti]] : null, pb = B ? B.off + i * 2 : 0, hasB = B !== null && B.q[pb] !== ABSENT;
    const hx = ft.half[0] / 32767, hy = ft.half[1] / 32767; let qx, qy;
    if (hasA && hasB) { const f = fmix[ti]; qx = A.q[pa] + f * (B.q[pb] - A.q[pa]); qy = A.q[pa + 1] + f * (B.q[pb + 1] - A.q[pa + 1]); }
    else if (hasB) { qx = B.q[pb]; qy = B.q[pb + 1]; }                               // new since the last keyframe
    else if (hasA) { qx = A.q[pa]; qy = A.q[pa + 1]; }
    else return false;
    out[0] = (qx * hx + fox) * fscale; out[1] = (qy * hy + foy) * fscale; out[2] = 0; return true;
  }

  const store = {
    M, years, T, N, K, cases, chain, base,
    hasForce: ft !== null, force: ft,
    ensureForce: loadForce,
    forceLoaded() { return ft !== null && fraw.every(r => r !== null); },
    /** Like decode(), for the force track (interpolated between keyframes, z = 0). */
    decodeForce(ti, pos, alive, col) {
      if (!ft || !raw[ti] || !fraw[fk0[ti]]) { for (let i = 0; i < N; i++) alive[i * 2 + col] = 0; return false; }
      const o3 = scratch3;
      for (let i = 0, o = 0; i < N; i++, o += 3) {
        if (forcePoint(ti, i, o3)) { pos[o] = o3[0]; pos[o + 1] = o3[1]; pos[o + 2] = 0; alive[i * 2 + col] = 1; }
        else { pos[o] = 0; pos[o + 1] = 0; pos[o + 2] = 0; alive[i * 2 + col] = 0; }
      }
      return true;
    },
    dims: [0, 1, 2],
    scale: 1,

    /** Choose which three of the K dimensions are x, y, z. Callers re-decode their buffers afterwards. */
    setDims(d) {
      this.dims = d.map(v => Math.max(0, Math.min(K - 1, v | 0)));
      this.scale = 1 / Math.max(...this.dims.map(k => sp.half[k]));
    },
    /** Resolves once year index ti can be decoded. */
    ensureYear(ti) { return raw[ti] ? Promise.resolve() : loadChunk(chunkOf[ti].c); },
    ensureAll() { return Promise.all(sp.chunks.map(loadChunk)); },
    loaded(ti) { return raw[ti] !== null; },
    /** Fetch the remaining chunks one at a time, nearest to `ti` first, without blocking the first paint. */
    async prefetch(ti = 0) {
      const order = sp.chunks.map((c, n) => [Math.abs(years.indexOf(c.years[0]) - ti), n]).sort((a, b) => a[0] - b[0]);
      for (const [, n] of order) await loadChunk(sp.chunks[n]);
    },

    /** Decode year ti into pos (N*3) and alive (N*2, column col). Absent points get 0,0,0 and alive 0.
     *  No allocation: this runs on every year step during playback. */
    decode(ti, pos, alive, col) {
      const r = raw[ti]; if (!r) return false;
      const q = r.q, d0 = this.dims[0], d1 = this.dims[1], d2 = this.dims[2];
      const s0 = sp.half[d0] / 32767 * this.scale, s1 = sp.half[d1] / 32767 * this.scale, s2 = sp.half[d2] / 32767 * this.scale;
      for (let i = 0, p = r.off, o = 0; i < N; i++, p += K, o += 3) {
        const v = q[p + d0];
        if (v === ABSENT) { pos[o] = 0; pos[o + 1] = 0; pos[o + 2] = 0; alive[i * 2 + col] = 0; }
        else { pos[o] = v * s0; pos[o + 1] = q[p + d1] * s1; pos[o + 2] = q[p + d2] * s2; alive[i * 2 + col] = 1; }
      }
      return true;
    },
    /** Position of case i at year ti in the given track (spectral: current dims), written to out[0..2];
     *  false if absent/not loaded. */
    point(ti, i, out, track = 'spectral') {
      if (track === 'force') return ft !== null && forcePoint(ti, i, out);
      const r = raw[ti]; if (!r) return false;
      const p = r.off + i * K, v = r.q[p + this.dims[0]];
      if (v === ABSENT) return false;
      out[0] = v * sp.half[this.dims[0]] / 32767 * this.scale;
      out[1] = r.q[p + this.dims[1]] * sp.half[this.dims[1]] / 32767 * this.scale;
      out[2] = r.q[p + this.dims[2]] * sp.half[this.dims[2]] / 32767 * this.scale;
      return true;
    },
    /** The path of case i through every loaded year in one track, as a flat xyz array (call ensureAll(),
     *  and ensureForce() for the force track, first). */
    trail(i, track = 'spectral') {
      const pts = [], o = [0, 0, 0];
      for (let k = 0; k < T; k++) if (this.point(k, i, o, track)) pts.push(o[0], o[1], o[2]);
      return new Float32Array(pts);
    },
    /** A label anchor in view coordinates: the K-dim spectral anchor for the current dims, or the label's
     *  force-track anchor (falls back to the spectral one when the build exported none). */
    anchor(label, out, track = 'spectral') {
      if (track === 'force' && ft && label.pf) {
        out[0] = (label.pf[0] - ft.centre[0] + fox) * fscale; out[1] = (label.pf[1] - ft.centre[1] + foy) * fscale; out[2] = 0; return out;
      }
      for (let a = 0; a < 3; a++) { const d = this.dims[a]; out[a] = ((label.p[d] ?? 0) - sp.centre[d]) * this.scale; }
      return out;
    },
    /** Eigenvalue shown beside dimension d at year ti: the Rayleigh quotient of the displayed (aligned)
     *  direction when the build exported it, else that year's d-th eigenvalue. */
    eigenvalue(ti, d) { return (M.vals_aligned ?? M.vals)?.[ti]?.[d] ?? null; },
  };
  store.setDims([0, 1, 2]);
  return store;
}
