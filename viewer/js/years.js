// Year view: cases filed in one year, in any court, straight from the API (python -m lawgraph.serve).
// Filters -> /api/cases -> a sortable table and a (date filed, log citations) scatter of the same rows.
// A row opens the decision (/api/case, /api/cites, /api/bench). "Animate" steps the year and refreshes.
// SCOTUS rows are broadcast to the 3-D view (same origin, any tab or the embedding page), which highlights them.
import { $, esc, clLink, getJSON } from './common.js';

const params = new URLSearchParams(location.search);
const embed = params.has('embed'); document.body.classList.toggle('embed', embed);
const sleep = ms => new Promise(r => setTimeout(r, ms));
const chan = 'BroadcastChannel' in window ? new BroadcastChannel('lawgraph') : null;

const state = {
  year: +params.get('year') || 1973,
  courts: new Set((params.get('court') ?? 'scotus').split(',').filter(Boolean)),
  q: '', minCited: 0, limit: 500,
  sort: { k: 'in_cites_to_date', dir: -1 }, measure: 'in_cites_to_date',
  rows: [], selected: null, playing: false, secPerYear: 1.0,
  loads: 0, errors: 0, lastMs: 0, yearMin: 1900, yearMax: 2020,
};
let courts = [], courtName = new Map(), timelineRows = [], seq = 0;

function api(path, q = {}) {
  const u = new URLSearchParams();
  for (const [k, v] of Object.entries(q)) { if (v === null || v === undefined || v === '') continue; if (Array.isArray(v)) v.forEach(x => u.append(k, x)); else u.append(k, v); }
  const s = u.toString();
  return getJSON('/api/' + path + (s ? '?' + s : ''));
}
function status(msg, err = false) { const el = $('#status'); el.textContent = msg; el.classList.toggle('err', err); if (err) state.errors++; }
const fmt = n => (n ?? 0).toLocaleString();

// ---------------------------------------------------------------- courts (multi-select)
function drawCourts() {
  const f = $('#courtFilter').value.trim().toLowerCase();
  const hits = courts.filter(c => state.courts.has(c.id) || !f || c.id.includes(f) || c.name.toLowerCase().includes(f)).slice(0, 80);
  $('#courtList').innerHTML = hits.map(c =>
    `<label><input type="checkbox" value="${esc(c.id)}"${state.courts.has(c.id) ? ' checked' : ''}> <span>${esc(c.name)} <span class="meta">${esc(c.id)}</span></span><span class="n">${fmt(c.n_cases)}</span></label>`).join('');
  $('#chips').innerHTML = [...state.courts].map(id => `<span class="chip" data-id="${esc(id)}" title="remove">${esc(id)} ×</span>`).join('');
}
$('#courtFilter').addEventListener('input', drawCourts);
$('#courtList').addEventListener('change', e => { const id = e.target.value; if (e.target.checked) state.courts.add(id); else state.courts.delete(id); drawCourts(); courtsChanged(); });
$('#chips').addEventListener('click', e => { const id = e.target.dataset.id; if (id) { state.courts.delete(id); drawCourts(); courtsChanged(); } });
function courtsChanged() { refresh(); loadTimeline(); }

// ---------------------------------------------------------------- rows: fetch, table, scatter
async function refresh() {
  const mine = ++seq, t0 = performance.now(), y = state.year;
  $('#year').textContent = y; $('#t').value = y;
  try {
    const rows = await api('cases', { year: y, court: [...state.courts], q: state.q, min_cited: state.minCited || null, limit: state.limit });
    if (mine !== seq) return false;                                   // a newer request superseded this one
    state.rows = rows; state.loads++; state.lastMs = Math.round(performance.now() - t0);
    sortRows(); drawTable(); drawScatter(); drawSpark();
    status(`${fmt(rows.length)} case${rows.length === 1 ? '' : 's'} filed in ${y}${rows.length >= state.limit ? ' (row limit reached)' : ''} · ${state.lastMs} ms`);
    broadcast();
    return true;
  } catch (err) {
    if (mine === seq) status(`could not load cases: ${err.message}. Is this page served by  python -m lawgraph.serve ?`, true);
    return false;
  }
}
function sortRows() {
  const { k, dir } = state.sort;
  state.rows.sort((a, b) => { const x = a[k] ?? '', y = b[k] ?? ''; return (x < y ? -1 : x > y ? 1 : 0) * dir || a.id - b.id; });
}
function drawTable() {
  document.querySelectorAll('th').forEach(th => { th.classList.toggle('sorted', th.dataset.k === state.sort.k); });
  $('#rows').innerHTML = state.rows.map(r =>
    `<tr data-id="${r.id}"${r.id === state.selected ? ' class="on"' : ''}><td class="name">${esc(r.case_name) || '<span class="meta">(no name)</span>'} ${clLink(r.id, '↗')}</td>` +
    `<td title="${esc(courtName.get(r.court) ?? '')}">${esc(r.court)}</td><td>${esc(r.date_filed)}</td><td class="num">${fmt(r.in_cites_to_date)}</td><td class="num">${fmt(r.citation_count)}</td></tr>`).join('');
}
document.querySelector('thead').addEventListener('click', e => {
  const k = e.target.closest('th')?.dataset.k; if (!k) return;
  state.sort = { k, dir: state.sort.k === k ? -state.sort.dir : (k === 'case_name' || k === 'court' || k === 'date_filed' ? 1 : -1) };
  sortRows(); drawTable();
});
$('#rows').addEventListener('click', e => { if (e.target.closest('a')) return; const tr = e.target.closest('tr'); if (tr) openCase(+tr.dataset.id); });

// scatter: x = day of the year, y = log10(1 + citations)
const sc = $('#scatter'), ctx = sc.getContext('2d'); let dots = [];
const dayOfYear = iso => { const d = new Date(iso + 'T00:00:00Z'); return (d - Date.UTC(d.getUTCFullYear(), 0, 1)) / 864e5; };
function drawScatter() {
  const dpr = Math.min(devicePixelRatio || 1, 2), W = sc.clientWidth, H = sc.clientHeight;
  if (sc.width !== W * dpr || sc.height !== H * dpr) { sc.width = W * dpr; sc.height = H * dpr; }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
  const L = 34, R = 10, Tp = 8, B = 18, m = state.measure;
  const top = Math.max(1, Math.log10(1 + Math.max(1, ...state.rows.map(r => r[m] || 0))));
  ctx.font = '10px system-ui'; ctx.fillStyle = '#666'; ctx.strokeStyle = '#1c1c1c'; ctx.lineWidth = 1;
  for (let p = 0; p <= Math.ceil(top); p++) { const y = H - B - (Math.log10(1 + (10 ** p - (p === 0 ? 1 : 0))) / top) * (H - Tp - B); if (y < Tp - 1) break;
    ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(W - R, y); ctx.stroke(); ctx.fillText(p === 0 ? '0' : String(10 ** p), 4, y + 3); }
  ['Jan', 'Apr', 'Jul', 'Oct'].forEach((mth, k) => ctx.fillText(mth, L + (k * 91 / 366) * (W - L - R), H - 5));
  dots = state.rows.map(r => ({ r, x: L + (dayOfYear(r.date_filed) / 366) * (W - L - R), y: H - B - (Math.log10(1 + (r[m] || 0)) / top) * (H - Tp - B) }));
  ctx.fillStyle = 'rgba(246,178,107,0.55)';
  for (const d of dots) { ctx.beginPath(); ctx.arc(d.x, d.y, 2.2, 0, 6.2832); ctx.fill(); }
  const sel = dots.find(d => d.r.id === state.selected);
  if (sel) { ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(sel.x, sel.y, 4.5, 0, 6.2832); ctx.fill(); }
}
function dotAt(e) {
  const b = sc.getBoundingClientRect(), mx = e.clientX - b.left, my = e.clientY - b.top; let best = null, bd = 64;
  for (const d of dots) { const q = (d.x - mx) ** 2 + (d.y - my) ** 2; if (q < bd) { bd = q; best = d; } }
  return best;
}
sc.addEventListener('mousemove', e => { const d = dotAt(e), tip = $('#tip');
  if (!d) { tip.style.display = 'none'; return; }
  tip.textContent = `${d.r.case_name} · ${d.r.court} · ${d.r.date_filed} · ${fmt(d.r[state.measure])} citations`;
  tip.style.display = 'block'; tip.style.left = Math.min(e.clientX + 12, innerWidth - tip.offsetWidth - 8) + 'px'; tip.style.top = (e.clientY + 12) + 'px'; });
sc.addEventListener('mouseleave', () => { $('#tip').style.display = 'none'; });
sc.addEventListener('click', e => { const d = dotAt(e); if (d) openCase(d.r.id); });
$('#measure').addEventListener('change', e => { state.measure = e.target.value; drawScatter(); });
addEventListener('resize', () => { drawScatter(); drawSpark(); });

// ---------------------------------------------------------------- timeline sparkline + year range
async function loadTimeline() {
  const court = state.courts.size ? [...state.courts][0] : null;
  try { timelineRows = await api('timeline', { court, start: state.yearMin, end: state.yearMax }); drawSpark(); }
  catch (err) { timelineRows = []; }
}
function drawSpark() {
  const c = $('#spark'), g = c.getContext('2d'), dpr = Math.min(devicePixelRatio || 1, 2), W = c.clientWidth, H = c.clientHeight;
  if (c.width !== W * dpr) { c.width = W * dpr; c.height = H * dpr; } g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
  if (!timelineRows.length) return;
  const span = Math.max(1, state.yearMax - state.yearMin), top = Math.max(...timelineRows.map(r => r.cases)), bw = Math.max(1, W / (span + 1));
  for (const r of timelineRows) { const h = Math.max(1, r.cases / top * (H - 2)); g.fillStyle = r.year === state.year ? '#f6b26b' : '#3a3a3a'; g.fillRect((r.year - state.yearMin) / (span + 1) * W, H - h, bw, h); }
}

// ---------------------------------------------------------------- decision panel
const OPINION = { '010combined': 'combined opinion', '015unamimous': 'unanimous opinion', '020lead': 'lead opinion', '025plurality': 'plurality opinion',
  '030concurrence': 'concurrence', '035concurrenceinpart': 'concurrence in part', '040dissent': 'dissent', '050addendum': 'addendum',
  '060remittitur': 'remittitur', '070rehearing': 'rehearing', '080onthemerits': 'on the merits', '090onmotiontostrike': 'on motion to strike', '100trialcourt': 'trial court document' };
const caseItem = r => `<li class="go" data-id="${r.id}">${esc(r.case_name) || '(no name)'} <span class="meta">${esc((r.date_filed || '').slice(0, 4))} · ${fmt(r.citation_count)}</span> ${clLink(r.id, '↗')}</li>`;
async function openCase(id) {
  state.selected = id; drawTable(); drawScatter();
  const box = $('#case'); box.innerHTML = '<span class="meta">loading…</span>';
  try {
    const c = await api('case/' + id);
    if (state.selected !== id) return;
    const [citedBy, cites, bench] = await Promise.all([
      api(`cites/${id}`, { direction: 'in', limit: 15 }), api(`cites/${id}`, { direction: 'out', limit: 15 }),
      c.court && c.date_filed ? api('bench', { court: c.court, date: c.date_filed }) : Promise.resolve([])]);
    if (state.selected !== id) return;
    const kinds = {}; for (const o of c.opinions) { const k = OPINION[o.type] || o.type || 'opinion'; kinds[k] = (kinds[k] || 0) + 1; }
    const in3d = c.court === 'scotus' ? ` · <a class="cl" id="to3d" href="./index.html?case=${id}&year=${(+c.date_filed.slice(0, 4)) + 1}" target="${embed ? '_self' : '_blank'}">show in 3-D view</a>` : '';
    box.innerHTML = `<h3>${esc(c.case_name)}</h3>` +
      `<div class="meta">${esc(c.court_name || c.court || '')} · ${esc(c.date_filed)} · ${esc(c.precedential_status || '')}</div>` +
      `<p>${clLink(id)}${in3d}</p>` +
      (c.case_name_full && c.case_name_full !== c.case_name ? `<p class="meta">${esc(c.case_name_full.slice(0, 300))}</p>` : '') +
      `<p>Cited ${fmt(c.citation_count)} times to date.</p>` +
      `<h4>Opinions</h4><ul>${Object.entries(kinds).map(([k, n]) => `<li>${n > 1 ? n + ' × ' : ''}${esc(k)}</li>`).join('') || '<li class="meta">none on record</li>'}</ul>` +
      `<h4>Authors</h4><ul>${c.authors.map(a => `<li>${esc(a.name)}</li>`).join('') || `<li class="meta">${esc(c.judges || 'not recorded')}</li>`}</ul>` +
      `<h4>Most-cited decisions citing it</h4><ul>${citedBy.map(caseItem).join('') || '<li class="meta">none</li>'}</ul>` +
      `<h4>Most-cited decisions it cites</h4><ul>${cites.map(caseItem).join('') || '<li class="meta">none</li>'}</ul>` +
      `<h4>Sitting on ${esc(c.court || 'the court')} that day (${bench.length})</h4><ul>${bench.map(j => `<li>${esc(j.name)} <span class="meta">${esc((j.from || '').slice(0, 4))}–${esc((j.to || '').slice(0, 4))}</span></li>`).join('') || '<li class="meta">no judges on record</li>'}</ul>`;
    const to3d = $('#to3d');
    if (to3d && embed) to3d.addEventListener('click', e => { e.preventDefault(); chan?.postMessage({ type: 'pin', id, year: +c.date_filed.slice(0, 4) + 1 }); });
  } catch (err) { if (state.selected === id) { box.innerHTML = `<span class="meta">could not load decision ${id}: ${esc(err.message)}</span>`; state.errors++; } }
}
$('#case').addEventListener('click', e => { if (e.target.closest('a')) return; const li = e.target.closest('li.go'); if (li) openCase(+li.dataset.id); });

// ---------------------------------------------------------------- filters, slider, animation
const debounce = (fn, ms = 250) => { let h; return (...a) => { clearTimeout(h); h = setTimeout(() => fn(...a), ms); }; };
$('#q').addEventListener('input', debounce(e => { state.q = e.target.value.trim(); refresh(); }));
$('#minCited').addEventListener('input', debounce(e => { state.minCited = Math.max(0, +e.target.value || 0); refresh(); }));
$('#limit').addEventListener('change', e => { state.limit = +e.target.value; refresh(); });
function setYear(y) { state.year = Math.max(state.yearMin, Math.min(state.yearMax, Math.round(y))); return refresh(); }
$('#t').addEventListener('input', e => { stop(); state.year = +e.target.value; $('#year').textContent = state.year; });
$('#t').addEventListener('change', e => { setYear(+e.target.value); });
function showSpeed() { $('#spy').textContent = state.secPerYear.toFixed(2); }
$('#faster').onclick = () => { state.secPerYear = Math.max(0.2, state.secPerYear / 1.5); showSpeed(); };
$('#slower').onclick = () => { state.secPerYear = Math.min(5, state.secPerYear * 1.5); showSpeed(); };
let run = 0;
/** One year per secPerYear; a year that takes longer to load simply holds the animation until it has arrived. */
async function play(until = state.yearMax) {
  const mine = ++run; state.playing = true; $('#play').textContent = '❚❚ Pause';
  while (mine === run && state.year < until) {
    const t0 = performance.now();
    state.year += 1; await refresh();
    if (mine !== run) return;
    await sleep(Math.max(0, state.secPerYear * 1000 - (performance.now() - t0)));
  }
  if (mine === run) stop();
}
function stop() { run++; if (!state.playing) return; state.playing = false; $('#play').textContent = '▶ Animate'; broadcast(); }
$('#play').onclick = () => { if (state.playing) stop(); else play(); };

// ---------------------------------------------------------------- link to the 3-D view
function broadcast() {
  chan?.postMessage({ type: 'rows', year: state.year, playing: state.playing, total: state.rows.length,
    ids: state.rows.filter(r => r.court === 'scotus').map(r => r.id) });
}
if (chan) chan.onmessage = e => { if (e.data?.type === 'hello') broadcast(); };      // a 3-D view opened after us

// ---------------------------------------------------------------- start
window.__years = { ready: false, state: () => ({ ...state, courts: [...state.courts], rows: state.rows.length, first: state.rows[0] ?? null, tableRows: $('#rows').children.length, dots: dots.length }),
  refresh, setYear, play, stop, openCase, setCourts: ids => { state.courts = new Set(ids); drawCourts(); loadTimeline(); return refresh(); } };
showSpeed();
try {
  const all = await api('timeline');                                  // every court: gives the range of years on record
  const seen = all.filter(r => r.cases > 0 && r.year >= 1650 && r.year <= new Date().getFullYear());
  if (seen.length) { state.yearMin = seen[0].year; state.yearMax = seen[seen.length - 1].year; }
  $('#t').min = state.yearMin; $('#t').max = state.yearMax; state.year = Math.max(state.yearMin, Math.min(state.yearMax, state.year));
  courts = (await api('courts')).filter(c => c.n_cases > 0); courtName = new Map(courts.map(c => [c.id, c.name]));
  drawCourts();
  await Promise.all([refresh(), loadTimeline()]);
} catch (err) { status(`the API is not reachable (${err.message}). Start the explorer with:  python -m lawgraph.serve --db graph.duckdb`, true); }
window.__years.ready = true;
