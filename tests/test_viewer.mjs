// Headless end-to-end test of the explorer: both pages, real data, zero console errors.
//
//   node tests/test_viewer.mjs                      starts `python -m lawgraph.serve` itself on a free port
//   node tests/test_viewer.mjs --url http://127.0.0.1:8080      uses a server that is already running
//
// Environment: CHROME=/path/to/chrome (default: google-chrome, chromium, a Playwright Chromium, Brave),
//              PYTHON=python  LAWGRAPH_DB=graph.duckdb  GL=gpu (default: software GL, works without a display)
// Needs viewer/data (python -m lawgraph.viewer_build ...) and `npm install` in tests/ (puppeteer-core).
// Screenshots go to tests/screenshots/. Exit code 0 only if every step passed and the console stayed clean.
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import puppeteer from 'puppeteer-core';

const HERE = path.dirname(fileURLToPath(import.meta.url)), ROOT = path.dirname(HERE), SHOTS = path.join(HERE, 'screenshots');
const args = process.argv.slice(2), argOf = k => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : null; };
const sleep = ms => new Promise(r => setTimeout(r, ms));

function findChrome() {
  const pw = path.join(os.homedir(), '.cache', 'ms-playwright');
  const playwright = fs.existsSync(pw) ? fs.readdirSync(pw).filter(d => /^chromium-\d+$/.test(d)).sort().reverse().map(d => path.join(pw, d, 'chrome-linux64', 'chrome')) : [];
  const candidates = [process.env.CHROME, '/usr/bin/google-chrome', '/usr/bin/google-chrome-stable', '/usr/bin/chromium', '/usr/bin/chromium-browser',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', ...playwright, '/usr/bin/brave-browser'].filter(Boolean);
  const hit = candidates.find(p => fs.existsSync(p));
  if (!hit) throw new Error(`no Chrome/Chromium found; set CHROME=/path/to/chrome (looked in: ${candidates.join(', ')})`);
  return hit;
}
const freePort = () => new Promise((res, rej) => { const s = net.createServer(); s.on('error', rej); s.listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => res(p)); }); });

let server = null;
async function startServer() {
  const port = await freePort(), py = process.env.PYTHON || 'python', db = process.env.LAWGRAPH_DB || path.join(ROOT, 'graph.duckdb');
  server = spawn(py, ['-m', 'lawgraph.serve', '--db', db, '--port', String(port), '--no-open'], { cwd: ROOT, stdio: ['ignore', 'pipe', 'pipe'] });
  let log = ''; server.stdout.on('data', d => { log += d; }); server.stderr.on('data', d => { log += d; });
  const url = `http://127.0.0.1:${port}`;
  for (let k = 0; k < 150; k++) {
    if (server.exitCode !== null) throw new Error(`lawgraph.serve exited (${server.exitCode}):\n${log}`);
    try { if ((await fetch(url + '/api/meta')).ok) return url; } catch { /* not up yet */ }
    await sleep(200);
  }
  throw new Error(`lawgraph.serve did not answer on ${url} within 30 s:\n${log}`);
}

// ---------------------------------------------------------------- harness
const problems = [], warnings = [], steps = [];
let failed = 0;
function watch(page, tag) {
  page.on('console', m => { const t = m.type(); if (t === 'error') problems.push(`[${tag}] console.error: ${m.text()}`); else if (t === 'warn' || t === 'warning') warnings.push(`[${tag}] ${m.text()}`); });
  page.on('pageerror', e => problems.push(`[${tag}] uncaught: ${e.message}`));
  page.on('requestfailed', r => problems.push(`[${tag}] request failed: ${r.url()} (${r.failure()?.errorText})`));
  page.on('response', r => { if (r.status() >= 400) problems.push(`[${tag}] HTTP ${r.status()}: ${r.url()}`); });
}
function check(cond, what, detail = '') { if (!cond) throw new Error(`${what}${detail ? ' — ' + detail : ''}`); }
async function step(name, fn) {
  const t0 = Date.now();
  try { const note = await fn(); steps.push(`  ok    ${name}${note ? '  (' + note + ')' : ''}  [${Date.now() - t0} ms]`); }
  catch (e) { failed++; steps.push(`  FAIL  ${name}: ${e.message}`); }
}
const shot = (page, name) => page.screenshot({ path: path.join(SHOTS, name) });
const until = async (page, fn, what, ms = 15000) => { try { await page.waitForFunction(fn, { timeout: ms }); } catch { throw new Error(`timed out waiting for ${what}`); } };

// ---------------------------------------------------------------- run
fs.mkdirSync(SHOTS, { recursive: true });
let browser = null;
try {
  const base = (argOf('--url') || await startServer()).replace(/\/$/, '');
  const gl = process.env.GL === 'gpu' ? ['--use-angle=vulkan', '--enable-features=Vulkan'] : ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'];
  browser = await puppeteer.launch({ executablePath: findChrome(), headless: true, args: ['--no-sandbox', '--window-size=1400,900', '--ignore-gpu-blocklist', ...gl] });
  console.log(`testing ${base} with ${await browser.version()}${server ? ' (server started by this script)' : ''}`);

  // ============================== 3-D view ==============================
  const page = await browser.newPage(); await page.setViewport({ width: 1400, height: 900 }); watch(page, 'index');
  const S = () => page.evaluate(() => window.__viewer.state());
  const yearIdx = y => page.evaluate(y => window.__viewer.store.years.indexOf(y), y);

  await step('3-D view loads and paints', async () => {
    const t0 = Date.now(); await page.goto(base + '/index.html', { waitUntil: 'load' });
    await until(page, 'window.__viewer && window.__viewer.ready', 'the viewer to be ready', 30000);
    const info = await page.evaluate(() => ({ paint: window.__viewer.firstPaintMs, K: window.__viewer.store.K, N: window.__viewer.store.N, T: window.__viewer.store.T,
      M: (({ K, vals, robust_dims, tracks, label_k }) => ({ K, vals: vals?.length, robust_dims, tracks: Object.keys(tracks || {}), label_k }))(window.__viewer.store.M) }));
    check(info.paint < 3000, 'first paint under 3 s', `${info.paint} ms`);
    check(info.M.K === info.K && info.K >= 3, 'manifest has K'); check(info.M.vals === info.T, 'manifest has per-year eigenvalues');
    check(Array.isArray(info.M.robust_dims) && info.M.robust_dims.length === 3, 'manifest has robust_dims');
    check(info.M.tracks.includes('spectral') && info.M.tracks.includes('force'), 'manifest has both tracks'); check(info.M.label_k >= 1, 'manifest has label_k');
    const s = await S(); check(s.alive > 0 && s.dims.join() === '0,1,2' && s.track === 'spectral', 'renders the default dims of the spectral track');
    await sleep(300); await shot(page, '01_default.png');
    return `${info.N} cases x ${info.T} years x ${info.K} dims, first paint ${info.paint} ms, ready ${Date.now() - t0} ms after navigation`;
  });

  await step('scrub to 1960, then play', async () => {
    const i = await yearIdx(1960); check(i >= 0, '1960 is in the data');
    await page.$eval('#t', (el, i) => { el.value = i; el.dispatchEvent(new Event('input')); }, i);
    await until(page, 'window.__viewer.state().year === 1960', 'the year to become 1960');
    const a = await S(); check(a.alive > 1000, 'cases are alive in 1960', String(a.alive));
    await page.click('#play'); await sleep(1500);
    const b = await S(); check(b.playing, 'playing after pressing Play'); check(b.year > 1960 || b.tfrac > 0, 'time advanced');
    await page.click('#play'); const c = await S(); check(!c.playing, 'paused');
    return `1960: ${a.alive} cases; played to ${b.year}`;
  });

  await step('search, pin, trail, neighbours, summary, links', async () => {
    await page.evaluate(i => window.__viewer.setYear(i), await yearIdx(2000));
    await page.type('#q', 'Miranda v. Ariz');
    await until(page, () => document.querySelectorAll('#results li').length > 0, 'search results');
    await page.click('#results li');
    await until(page, () => { const s = window.__viewer.state(); return s.pinnedName === 'Miranda v. Arizona' && s.trailPoints > 10; }, 'Miranda to be pinned with a trail');
    await page.evaluate(i => window.__viewer.setYear(i), await yearIdx(2000));
    await until(page, () => document.querySelectorAll('#nbrs li a.cl').length >= 5, 'co-citation neighbours with links');
    const info = await page.evaluate(() => ({ link: document.querySelector('#info a.cl').href, summary: document.querySelector('#info p')?.textContent ?? '',
      nbrs: document.querySelectorAll('#nbrs li').length, nbrLink: document.querySelector('#nbrs li a.cl').href, searchLink: document.querySelector('#results li a.cl').href }));
    check(/courtlistener\.com\/opinion\/\d+\/x\/$/.test(info.link) && /\/opinion\/\d+\/x\/$/.test(info.nbrLink) && /\/opinion\/\d+\/x\/$/.test(info.searchLink), 'CourtListener links in panel, neighbours and search');
    check(info.summary.length > 40, 'two-sentence summary shown'); const s = await S(); check(s.markVisible, 'marker on the pinned case');
    await page.$eval('#q', el => { el.value = ''; el.dispatchEvent(new Event('input')); });       // tidy the panel for the screenshots that follow
    return `${s.trailPoints}-point trail, ${info.nbrs} neighbours`;
  });

  await step('toggle projection at 2000 with Miranda pinned', async () => {
    const before = await S();
    await page.click('#proj button[data-track=force]');
    await until(page, () => { const s = window.__viewer.state(); return s.track === 'force' && !s.mixing && s.u === 1; }, 'the crossfade to the force track');
    await sleep(150); const f = await S();
    check(f.pinnedName === 'Miranda v. Arizona', 'Miranda still pinned'); check(f.trailTrack === 'force' && f.trailPoints > 10, 'trail redrawn in the force track');
    check(f.markVisible && Math.abs(f.mark[2]) < 1e-6, 'marker lies in the flat force layout'); check(f.labels.length > 0, 'labels kept');
    check(await page.$eval('#axes', el => getComputedStyle(el).display === 'none'), 'eigenvector picker hidden in the force projection');
    await shot(page, '02_force_miranda_2000.png');
    await page.click('#proj button[data-track=spectral]');
    await until(page, () => { const s = window.__viewer.state(); return s.track === 'spectral' && !s.mixing && s.u === 0; }, 'the crossfade back');
    await sleep(150); const s = await S();
    check(s.pinnedName === 'Miranda v. Arizona' && s.trailTrack === 'spectral', 'pinned and trail back in the spectral track');
    check(before.mark.every((v, k) => Math.abs(v - s.mark[k]) < 1e-6), 'marker returned to where it was');
    await shot(page, '03_spectral_miranda_2000.png');
    return `trail ${f.trailPoints} points in each track`;
  });

  await step('change eigenvectors without reloading', async () => {
    const opts = await page.$$eval('#ax2 option', o => o.map(x => x.text)); check(opts.length >= 3 && /λ \d\.\d+/.test(opts[0]), 'dropdowns list ev1..evK with eigenvalues', opts[0]);
    const before = await page.evaluate(() => performance.getEntriesByType('resource').length);
    await page.select('#ax2', '3'); await sleep(300);
    const a = await S(); check(a.dims.join() === '0,1,3', 'dims are [0,1,3]', a.dims.join());
    check(await page.evaluate(() => performance.getEntriesByType('resource').length) === before, 'no network request for a dimension change');
    check(a.pinnedName === 'Miranda v. Arizona' && a.trailPoints > 10, 'selection and trail survive'); await shot(page, '04_dims_0_1_3.png');
    await page.select('#ax0', '1'); check((await S()).dims.join() === '1,0,3', 'choosing an axis already in use swaps');
    const robust = await page.evaluate(() => window.__viewer.store.M.robust_dims.join());
    await page.click('#robust'); check((await S()).dims.join() === robust, 'robust set applied', robust);
    await page.click('#robust'); check((await S()).dims.join() === '0,1,2', 'robust set toggles back to the default');
    return `robust set [${robust}]`;
  });

  await step('ev4 separates the capital-punishment cases (Furman, Gregg)', async () => {
    // T2 acceptance: with dims [0,1,3] the Furman/Gregg line of cases sits at the far end of the third axis
    const z = await page.evaluate(async i2000 => {
      const v = window.__viewer; await v.setYear(i2000);
      const row = n => v.store.cases.find(c => c.name === n)?.i, furman = row('Furman v. Georgia'), gregg = row('Gregg v. Georgia');
      if (furman === undefined || gregg === undefined) return null;
      await v.select(furman); v.camera.position.set(1.9, 0.9, 1.9); v.controls.update();      // oblique, so the z axis is on screen
      const at = (dims, i) => { v.setDims(dims); const o = [0, 0, 0]; v.store.point(v.state().ti, i, o); return o[2]; };
      return { f012: at([0, 1, 2], furman), g012: at([0, 1, 2], gregg), f013: at([0, 1, 3], furman), g013: at([0, 1, 3], gregg) };
    }, await yearIdx(2000));
    check(z, 'Furman v. Georgia and Gregg v. Georgia are in the data');
    await page.evaluate(() => window.__viewer.setDims([0, 1, 2])); await sleep(500); await shot(page, '04a_furman_dims_0_1_2.png');
    await page.evaluate(() => window.__viewer.setDims([0, 1, 3])); await sleep(500); await shot(page, '04b_furman_dims_0_1_3.png');
    // coordinates are scaled so the widest displayed dimension spans [-1, 1]
    check(Math.abs(z.f013) > 0.6 && Math.abs(z.g013) > 0.6 && Math.sign(z.f013) === Math.sign(z.g013), 'both sit far out along ev4, on the same side', JSON.stringify(z));
    check(Math.abs(z.f013) > Math.abs(z.f012) && Math.abs(z.g013) > Math.abs(z.g012), 'farther out than along ev3');
    await page.evaluate(() => { const v = window.__viewer; v.setDims([0, 1, 2]); v.camera.position.set(0, 0, 2.6); v.controls.target.set(0, 0, 0); v.controls.update(); });
    return `Furman z = ${z.f012.toFixed(2)} on ev3 -> ${z.f013.toFixed(2)} on ev4; Gregg ${z.g012.toFixed(2)} -> ${z.g013.toFixed(2)}`;
  });

  await step('legend filter, hover, label details, Esc', async () => {
    await page.click('#legend .chain'); check(await page.$eval('#legend .chain', el => el.classList.contains('on')), 'legend entry isolates its lineage');
    await page.click('#legend .chain'); check(!(await page.$eval('#legend .chain', el => el.classList.contains('on'))), 'legend entry toggles off');
    await page.evaluate(() => window.__viewer.setTrack('force', false)); await page.evaluate(i => window.__viewer.setYear(i), await yearIdx(2020)); await sleep(400);
    const s = await S(); const box = s.labelBoxes.find(b => b.vis); check(box, 'a community label is on screen');
    check(/ · /.test(box.name), 'labels name the top cases', box.name);
    await page.mouse.move(box.x, box.y); await sleep(150);
    const tip = await page.$eval('#tip', el => ({ shown: el.style.display === 'block', text: el.innerText })); check(tip.shown && /cases in this community/.test(tip.text) && / v\. /.test(tip.text), 'label hover shows full names, size and chain', tip.text.replace(/\n/g, ' | '));
    // hover a point: the pinned case's marker is a point we know the screen position of
    const at = await page.evaluate(() => { const v = window.__viewer, p = v.camera.position.clone().fromArray(v.state().mark).project(v.camera); return [(p.x + 1) / 2 * innerWidth, (1 - p.y) / 2 * innerHeight]; });
    await page.mouse.move(at[0] + 1, at[1] + 1); await sleep(150);
    const ptip = await page.$eval('#tip', el => ({ shown: el.style.display === 'block', text: el.innerText })); check(ptip.shown && /\(\d{4}\)|cases in this community/.test(ptip.text), 'hovering a point names the case', ptip.text.replace(/\n/g, ' | '));
    await page.keyboard.press('Escape'); const e = await S(); check(e.pinned === -1 && e.trailPoints === 0, 'Esc clears the selection');
    await page.evaluate(i => window.__viewer.setYear(i), await yearIdx(2019)); await page.evaluate(i => window.__viewer.setYear(i), await yearIdx(2020));
    check(await page.$eval('#tip', el => el.style.display === 'none'), 'a year step hides a stale tooltip'); await page.mouse.move(8, 300); await sleep(300);
    await shot(page, '05_force_labels_2020.png');
    await page.evaluate(() => window.__viewer.setTrack('spectral', false));
    return `label "${box.name}"`;
  });

  await step('force layout screenshots at 1960 and 2020', async () => {
    await page.evaluate(() => window.__viewer.setTrack('force', false));
    for (const y of [1960, 2020]) { await page.evaluate(i => window.__viewer.setYear(i), await yearIdx(y)); await sleep(400);
      const s = await S(); check(s.aliveForce > 1000 && s.aliveForce === s.alive, `force track shows the same cases as spectral in ${y}`, `${s.aliveForce} vs ${s.alive}`); await shot(page, `06_force_${y}.png`); }
    const f = await page.evaluate(() => window.__viewer.store.force); const worst = Math.max(...Object.values(f.displacement));
    check(worst < 0.10, 'keyframe-to-keyframe displacement under 10% of the layout radius', `${(100 * worst).toFixed(1)}%`);
    await page.evaluate(() => window.__viewer.setTrack('spectral', false));
    return `${f.backend}, worst displacement ${(100 * worst).toFixed(1)}%`;
  });

  await step('playback frame rate (informational)', async () => {
    await page.evaluate(i => window.__viewer.setYear(i), await yearIdx(1990));
    const r = await page.evaluate(async () => { const v = window.__viewer, f0 = v.state().frames, t0 = performance.now(); document.querySelector('#play').click();
      await new Promise(r => setTimeout(r, 2000)); document.querySelector('#play').click(); return { fps: (v.state().frames - f0) / ((performance.now() - t0) / 1000), year: v.state().year }; });
    check(r.year > 1990, 'playback advanced'); return `${r.fps.toFixed(0)} fps with ${process.env.GL === 'gpu' ? 'GPU' : 'software'} GL, reached ${r.year}`;
  });

  // ============================== year view ==============================
  const yp = await browser.newPage(); await yp.setViewport({ width: 1400, height: 900 }); watch(yp, 'years');
  const Y = () => yp.evaluate(() => window.__years.state());

  await step('year view loads: SCOTUS 1973, Roe v. Wade first', async () => {
    await yp.goto(base + '/years.html', { waitUntil: 'load' });
    await until(yp, 'window.__years && window.__years.ready', 'the year view to be ready', 30000);
    const s = await Y(); check(s.errors === 0, 'no API errors', await yp.$eval('#status', el => el.textContent));
    check(s.year === 1973 && s.rows > 0 && s.tableRows === s.rows && s.dots === s.rows, 'table and scatter show the same rows', `${s.rows} rows`);
    check(s.first.case_name === 'Roe v. Wade', 'Roe v. Wade is the most cited 1973 SCOTUS decision that year', s.first.case_name);
    check(await yp.$$eval('#rows a.cl', a => a.length > 0 && a.every(x => /courtlistener\.com\/opinion\/\d+\/x\/$/.test(x.href))), 'every table row links to CourtListener');
    check(await yp.$$eval('#courtList label', l => l.length > 10), 'court list from /api/courts');
    await shot(yp, '07_years_scotus_1973.png'); return `${s.rows} rows in ${s.lastMs} ms`;
  });

  await step('year view: sort, search, open a decision', async () => {
    await yp.click('th[data-k=citation_count]'); const top = await yp.$eval('#rows tr td.num:last-child', el => +el.textContent.replace(/\D/g, ''));
    const second = await yp.$eval('#rows tr:nth-child(2) td.num:last-child', el => +el.textContent.replace(/\D/g, '')); check(top >= second, 'sorted by citations, descending');
    await yp.type('#q', 'roe'); await until(yp, () => { const s = window.__years.state(); return s.q === 'roe' && s.rows > 0 && s.rows < 50; }, 'the name filter to apply');
    await yp.click('#rows tr td.name'); await until(yp, () => /sitting on/i.test(document.querySelector('#case').innerText), 'the decision panel');   // headings render uppercase
    const text = await yp.$eval('#case', el => el.innerText); check(/Roe v\. Wade/.test(text) && /Blackmun/.test(text) && /CourtListener/.test(text), 'decision panel: name, author, link');
    await yp.$eval('#q', el => { el.value = ''; el.dispatchEvent(new Event('input')); }); await until(yp, () => window.__years.state().q === '' && window.__years.state().rows > 50, 'the filter to clear');
    return 'Roe v. Wade panel with opinions, authors, citing/cited decisions and the bench';
  });

  await step('year animation for 3 s (Ninth Circuit from 1950)', async () => {
    await yp.evaluate(async () => { await window.__years.setCourts(['ca9']); await window.__years.setYear(1950); });
    const a = await Y(); check(a.rows > 0 && a.courts.join() === 'ca9', 'Ninth Circuit rows for 1950', `${a.rows}`);
    await yp.click('#faster'); await yp.click('#faster'); await yp.click('#play'); await sleep(3000); await yp.click('#play');
    const b = await Y(); check(!b.playing, 'animation paused'); check(b.year >= 1953, 'the year advanced', `reached ${b.year}`);
    check(b.errors === 0 && b.loads - a.loads >= 3, 'each year refreshed without errors', `${b.loads - a.loads} refreshes, ${b.errors} errors`);
    await shot(yp, '08_years_ca9_animated.png'); return `1950 -> ${b.year}, ${b.loads - a.loads} refreshes, last ${b.lastMs} ms`;
  });
  await yp.close();

  await step('year view rows are highlighted in the 3-D cloud', async () => {
    await page.click('#yearsBtn');
    await until(page, () => { const f = document.querySelector('#yearsFrame'); return f && f.contentWindow.__years && f.contentWindow.__years.ready; }, 'the embedded year view', 30000);
    await until(page, () => window.__viewer.state().highlighted > 0, 'highlighted rows in the cloud');
    const frame = page.frames().find(f => f.url().includes('years.html'));
    await frame.evaluate(async () => { const y = window.__years; await y.setYear(1962); document.querySelector('#faster').click(); document.querySelector('#faster').click(); await y.play(1970); });
    await until(page, () => window.__viewer.state().year === 1971 && window.__viewer.state().highlightYears.length >= 8, 'the cloud to follow the animation to 1971');
    const s = await S(); check(s.highlighted > 50 && /Year view 1970/.test(s.banner), 'banner reports the highlighted rows', s.banner);
    await sleep(500); await shot(page, '09_cloud_highlights_year_view.png');
    await page.click('#yearsBtn'); check((await S()).highlighted === 0, 'closing the drawer clears the highlight');
    return s.banner;
  });
  await page.close();
} catch (e) { failed++; steps.push(`  FAIL  setup: ${e.message}`); }
finally { if (browser) await browser.close(); if (server) server.kill('SIGINT'); }

console.log(steps.join('\n'));
if (warnings.length) console.log(`\n${warnings.length} console warning(s) (not failures):\n  ` + [...new Set(warnings)].slice(0, 5).join('\n  '));
if (problems.length) console.log(`\nCONSOLE / NETWORK ERRORS (${problems.length}):\n  ` + problems.join('\n  '));
const ok = failed === 0 && problems.length === 0;
console.log(`\n${ok ? 'PASS' : 'FAIL'}: ${steps.length - failed}/${steps.length} steps, ${problems.length} console/network errors. Screenshots in ${path.relative(process.cwd(), SHOTS) || SHOTS}/`);
process.exit(ok ? 0 : 1);
