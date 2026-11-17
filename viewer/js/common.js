// Small helpers shared by the 3-D view (app.js) and the year view (years.js).
export const $ = s => document.querySelector(s);

const ESC = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
/** Escape text from the database before it goes into innerHTML. */
export const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ESC[c]);

/** Every case shown anywhere links here; CourtListener redirects /x/ to the real slug. */
export const clUrl = id => `https://www.courtlistener.com/opinion/${encodeURIComponent(id)}/x/`;
export const clLink = (id, text = 'CourtListener ↗') =>
  `<a class="cl" target="_blank" rel="noopener" href="${clUrl(id)}" title="open on CourtListener">${text}</a>`;

export async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}
export async function getBuffer(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.arrayBuffer();
}
