// ---------- Calcs ----------
// Copied from the dashboard as it stands.
dash.calc('band_counts', {
  inputs: ['new_fires'],
  title: 'New fires per 48 h risk band',
  description: 'Competition test fires counted by their 48 hour risk band.',
  fn: (newFires) => ['Low', 'Medium', 'High', 'Critical']
    .map((band) => ({ band, fires: newFires.filter((f) => f.band === band).length }))
    .concat([{ band: 'All', fires: newFires.length }]),
});
// Figures as the captions say them: rounded for the ear, still numbers from the data, so the mark check compares them.
dash.calc('said', {
  inputs: ['test_scores', 'model_cv', 'fires', 'label_counts', 'new_fires'],
  title: 'Figures as the captions say them',
  description: 'Scores rounded to 2 decimals, distances to 1 decimal, counts as they are.',
  fn: (scores, models, fires, counts, newFires) => {
    const score = (m) => +(scores.find((r) => r.metric === m) || {}).value;
    const model = (n) => +(models.find((r) => r.model === n) || {}).hybrid;
    const r2 = (v) => (Number.isFinite(v) ? Math.round(v * 100) / 100 : null);
    const r1 = (v) => (Number.isFinite(v) ? Math.round(v * 10) / 10 : null);
    const hits = fires.filter((f) => +f.hit === 1), misses = fires.filter((f) => +f.hit === 0);
    const c12 = counts.find((r) => r.horizon === '12 h') || {};
    return [{
      hybrid_test: r2(score('hybrid')),
      tuned: r2(model('Random Forest (tuned)')),
      baseline: r2(model('Baseline (average)')),
      fires: fires.length,
      hit_per_100: fires.length ? Math.round((100 * hits.length) / fires.length) : null,
      farthest_hit_km: r1(d3.max(hits, (f) => +f.distance_km)),
      closest_nohit_km: r1(d3.min(misses, (f) => +f.distance_km)),
      hits_12h: c12.hit == null ? null : +c12.hit,
      hits: hits.length,
      new_fires: newFires.length,
      critical: newFires.filter((f) => f.band === 'Critical').length,
    }];
  },
});

const root = document.querySelector('#dash-root'), vs = root.querySelector('.vs'), stage = vs.querySelector('.vs-stage');
const [chs, caps, mks] = ['.vs-ch', '.vs-cap', '.vs-mk'].map((q) => [...vs.querySelectorAll(q)]);
const still = matchMedia('(prefers-reduced-motion: reduce)');
const words = (el) => el.textContent.replace(/\s+/g, ' ').trim();
const write = (el, s) => { if (el && el.textContent !== s) el.textContent = s; };
const says = (i) => [...(caps[i]?.querySelectorAll('.vs-s:not([hidden])') ?? [])];
const called = (i) => ['.vs-label', 'h1, h2, h3'].map((q) => chs[i]?.querySelector(q)).map((el) => (el ? words(el) : '')).find(Boolean) ?? String(i + 1);
// How a figure is written, by source and field; any other as it comes.
const f2 = d3.format('.2f'), f3 = d3.format('.3f');
const fmt = { 'said.hybrid_test': f2, 'said.tuned': f2, 'said.baseline': f2, 'test_scores.value': f3, 'model_cv.hybrid': f3, 'importance.importance': f3 };
let said = [], durs = [], cur = 0, t = Infinity, playing = false, begun = false, raf = 0, last = 0, heard = null, was = null, pace = 1;
const phase = () => root.getAttribute('data-dash-read-phase');

const paced = (at) => (phase() ? at * pace : at);
const cue = (i, word) => paced(Math.max(0, said[i].indexOf(word)) / 15);
const moved = (t, from, len = 0.6) => d3.easeCubicOut(Math.max(0, Math.min(1, (t - from) / len)));
const places = (v) => { const [m, e = 0] = String(v).split('e'); return Math.min(100, Math.max(0, (m.split('.')[1] ?? '').length - e)) || 0; };
const shown = (s) => (String(s).match(/\.(\d+)/)?.[1] ?? '').length;
const count = (real, copy, p, value, f) => {
  real.style.opacity = p < 1 ? 0 : 1; copy.style.opacity = p > 0 && p < 1 ? 1 : 0;
  if (p >= 1) return;
  let s = f(value * p);
  if (shown(s) > shown(f(value))) s = f(+(value * p).toFixed(places(value)));
  write(copy, s);
};

// ---------- Drawing helpers ----------
const rowsOf = (el, src) => {
  const d = dash.data(src);
  if (d.status === 'ok' && Array.isArray(d.data) && d.data.length) return d.data;
  el.replaceChildren(); write(el, '—');
  return null;
};
const mk = (tag, cls, attrs = {}) => { const e = document.createElement(tag); if (cls) e.className = cls; for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); return e; };
const mark = (src, where, field) => { const s = mk('span', null, { 'data-source': src, 'data-where': where, 'data-field': field }); fig(s); return s; };
function tipOf(el) {
  return {
    show(ev, lines) {
      let tip = el.querySelector(':scope > .vs-tip');
      if (!tip) { tip = mk('div', 'vs-tip'); tip.hidden = true; el.append(tip); }
      tip.replaceChildren(...lines.map((s) => { const d = mk('div'); d.textContent = s; return d; }));
      tip.hidden = false;
      const b = el.getBoundingClientRect();
      let x = ev.clientX - b.left + 10, y = ev.clientY - b.top + 10;
      if (x + tip.offsetWidth > b.width) x = Math.max(0, ev.clientX - b.left - tip.offsetWidth - 10);
      if (y + tip.offsetHeight > b.height) y = Math.max(0, ev.clientY - b.top - tip.offsetHeight - 10);
      tip.style.left = x + 'px'; tip.style.top = y + 'px';
    },
    hide() { const tip = el.querySelector(':scope > .vs-tip'); if (tip) tip.hidden = true; },
  };
}
// Horizontal bars from zero: one row per item, the value at the end of its bar.
function hbars(el, items, { src, keyField, field, label, accent, tipLines, aria }) {
  el.style.setProperty('--acc', dash.colors[0]);
  const max = d3.max(items, (r) => +r[field]) || 1;
  const box = mk('div', 'hb', { role: 'img', 'aria-label': aria });
  box.style.gridAutoRows = `min(15cqh, calc(84cqh / ${items.length}))`;
  const tip = tipOf(el);
  for (const r of items) {
    const lab = mk('span', 'hb-lab'); lab.textContent = label(r);
    const track = mk('div', 'hb-track');
    const bar = mk('div', 'hb-bar' + (accent(r) ? ' acc' : ''));
    bar.style.width = Math.max(0.4, (+r[field] / max) * 78) + '%';
    const val = mark(src, `${keyField}=${r[keyField]}`, field); val.className = 'hb-val';
    track.append(bar, val);
    track.onpointermove = (ev) => tip.show(ev, tipLines(r));
    track.onpointerleave = () => tip.hide();
    box.append(lab, track);
  }
  el.replaceChildren(box);
}
// Columns from zero: value above, label below.
function columns(el, items, { src, keyField, field, label, accent, tipLines, aria }) {
  el.style.setProperty('--acc', dash.colors[0]);
  const max = d3.max(items, (r) => +r[field]) || 1;
  const box = mk('div', 'vc', { role: 'img', 'aria-label': aria });
  const tip = tipOf(el);
  for (const r of items) {
    const col = mk('div', 'vc-col');
    const val = mark(src, `${keyField}=${r[keyField]}`, field); val.className = 'vc-val';
    const bar = mk('div', 'vc-bar' + (accent(r) ? ' acc' : ''));
    bar.style.height = `calc((100cqh - 3.6em) * ${(+r[field] / max).toFixed(4)})`;
    const lab = mk('span', 'vc-lab'); lab.textContent = label(r);
    col.append(val, bar, lab);
    col.onpointermove = (ev) => tip.show(ev, tipLines(r));
    col.onpointerleave = () => tip.hide();
    box.append(col);
  }
  el.replaceChildren(box);
}

// ---------- Drawings, one per chart ----------
dash.draw('dots', (el) => {
  const rows = rowsOf(el, 'said'); if (!rows) return;
  const k = rows[0].hit_per_100;
  el.style.setProperty('--acc', dash.colors[0]);
  const grid = mk('div', 'dots', { role: 'img', 'aria-label': 'One dot per 100th of the fires; the colored dots reached an evacuation zone.' });
  for (let j = 0; j < 100; j++) grid.append(mk('div', 'dot' + (j < k ? ' hit' : '')));
  const tip = tipOf(el);
  grid.onpointermove = (ev) => {
    const d = ev.target.closest('.dot'); if (!d) return tip.hide();
    tip.show(ev, [d.classList.contains('hit') ? 'Reached a zone' : 'Did not reach a zone', `${k} in every 100 fires reached one`]);
  };
  grid.onpointerleave = () => tip.hide();
  el.replaceChildren(grid);
});

dash.draw('dist', (el) => {
  const rows = rowsOf(el, 'fires'); if (!rows) return;
  el.style.setProperty('--acc', dash.colors[0]);
  const W = el.clientWidth || 640, H = el.clientHeight || 320; // before layout the box can measure 0; it is drawn again at its real size
  const fs = parseFloat(getComputedStyle(el).fontSize) || 14;
  const m = { t: fs * 0.6, r: fs * 0.8, b: fs * 2.2, l: fs * 2.6 };
  const svg = d3.create('svg').attr('viewBox', `0 0 ${W} ${H}`).attr('role', 'img')
    .attr('aria-label', 'Each dot is a fire: distance to the evacuation zone across, hours until hit or last seen up. All fires that hit sit left of the 5 km line.');
  const x = d3.scaleLog().domain([0.05, d3.max(rows, (r) => +r.distance_km) * 1.2]).range([m.l, W - m.r]);
  const y = d3.scaleLinear().domain([0, 72]).range([H - m.b, m.t]);
  svg.append('g').attr('class', 'grid').attr('transform', `translate(${m.l},0)`).call(d3.axisLeft(y).tickValues([0, 24, 48, 72]).tickSize(-(W - m.l - m.r)).tickFormat('')).call((g) => g.select('.domain').remove());
  svg.append('g').attr('class', 'axis').attr('transform', `translate(0,${H - m.b})`).call(d3.axisBottom(x).tickValues(W < 480 ? [0.1, 1, 10, 100] : [0.1, 1, 10, 100, 500]).tickFormat((d) => d + ' km'));
  svg.append('g').attr('class', 'axis').attr('transform', `translate(${m.l},0)`).call(d3.axisLeft(y).tickValues([0, 24, 48, 72]).tickFormat((d) => d + ' h'));
  svg.append('g').selectAll('circle').data(rows.slice().sort((a, b) => a.hit - b.hit)).join('circle')
    .attr('class', (r) => 'pt' + (+r.hit === 1 ? ' h' : ''))
    .attr('cx', (r) => x(Math.max(+r.distance_km, 0.05))).attr('cy', (r) => y(+r.hours)).attr('r', Math.max(3, fs * 0.32));
  const ref = svg.append('g').attr('class', 'ref');
  ref.append('line').attr('x1', x(5)).attr('x2', x(5)).attr('y1', m.t).attr('y2', H - m.b);
  ref.append('text').attr('x', x(5) + 5).attr('y', m.t + fs).text('5 km');
  const tip = tipOf(el);
  svg.on('pointermove', (ev) => {
    const c = ev.target.closest && ev.target.closest('circle'); if (!c) return tip.hide();
    const r = d3.select(c).datum();
    tip.show(ev, [`Fire ${r.event_id}`, `${(+r.distance_km).toFixed(1)} km from a zone`, +r.hit === 1 ? `Hit after ${(+r.hours).toFixed(1)} hours` : `No hit, last seen at ${(+r.hours).toFixed(1)} hours`]);
  }).on('pointerleave', () => tip.hide());
  el.replaceChildren(svg.node());
});

dash.draw('time', (el) => {
  const rows = rowsOf(el, 'label_counts'); if (!rows) return;
  columns(el, rows, {
    src: 'label_counts', keyField: 'horizon', field: 'hit',
    label: (r) => 'within ' + r.horizon, accent: (r) => r.horizon === '12 h',
    tipLines: (r) => [`Within ${r.horizon}`, `${r.hit} fires had hit`],
    aria: 'Columns: how many fires had hit within 12, 24, 48 and 72 hours.',
  });
});

dash.draw('models', (el) => {
  const rows = rowsOf(el, 'model_cv'); if (!rows) return;
  hbars(el, rows.slice().sort((a, b) => +b.hybrid - +a.hybrid), {
    src: 'model_cv', keyField: 'model', field: 'hybrid',
    label: (r) => r.model, accent: (r) => r.model === 'Random Forest (tuned)',
    tipLines: (r) => [r.model, `Hybrid score ${(+r.hybrid).toFixed(3)}`],
    aria: 'Bars: Hybrid score of each model; the tuned random forest is highest.',
  });
});

dash.draw('bands', (el) => {
  const rows = rowsOf(el, 'band_counts'); if (!rows) return;
  columns(el, rows.filter((r) => r.band !== 'All'), {
    src: 'band_counts', keyField: 'band', field: 'fires',
    label: (r) => r.band, accent: (r) => r.band === 'Critical',
    tipLines: (r) => [`${r.band} risk at 48 hours`, `${r.fires} fires`],
    aria: 'Columns: new fires in each 48 hour risk band.',
  });
});

dash.draw('imp', (el) => {
  const rows = rowsOf(el, 'importance'); if (!rows) return;
  hbars(el, rows.slice().sort((a, b) => +b.importance - +a.importance), {
    src: 'importance', keyField: 'feature', field: 'importance',
    label: (r) => r.feature, accent: (r) => r.feature === 'dist_min_ci_0_5h',
    tipLines: (r) => [r.feature, `Score drop ${(+r.importance).toFixed(3)} when shuffled`],
    aria: 'Bars: how much each feature matters; distance is far ahead.',
  });
});

// ---------- Scenes, one per chapter ----------
const growBars = (i, t, from) => chs[i].querySelectorAll('.hb-bar').forEach((b, j) => {
  b.style.transform = `scaleX(${moved(t, from + j * 0.15)})`;
  const v = b.parentElement.querySelector('.hb-val'); if (v) v.style.opacity = moved(t, from + j * 0.15 + 0.5, 0.3);
});
const growCols = (i, t, from) => chs[i].querySelectorAll('.vc-col').forEach((c, j) => {
  const b = c.querySelector('.vc-bar'); if (b) b.style.transform = `scaleY(${moved(t, from + j * 0.2)})`;
  const v = c.querySelector('.vc-val'); if (v) v.style.opacity = moved(t, from + j * 0.2 + 0.5, 0.3);
});
const scenes = [
  () => {},
  (t) => {
    const a = cue(1, 'studied'), b = cue(1, 'reached');
    chs[1].querySelectorAll('.dot').forEach((d, j) => { d.style.opacity = moved(t, a + j * 0.012, 0.35); d.classList.toggle('on', t >= b); });
  },
  (t) => {
    const svg = chs[2].querySelector('svg'); if (!svg) return;
    svg.querySelectorAll('.pt').forEach((p, j) => { p.style.opacity = moved(t, cue(2, 'Distance') + (j % 40) * 0.01, 0.4); });
    svg.classList.toggle('lit', t >= cue(2, 'hit'));
    const ref = svg.querySelector('.ref'); if (ref) ref.style.opacity = moved(t, cue(2, 'within'), 0.5);
  },
  (t) => growCols(3, t, cue(3, 'When')),
  (t) => growBars(4, t, cue(4, 'compared')),
  (t) => growCols(5, t, cue(5, 'scored')),
  (t) => growBars(6, t, cue(6, 'leans')),
];

// ---------- Engine (from the Visual short skeleton) ----------
function fig(el) {
  const dec = (s) => { try { return decodeURIComponent(s); } catch { return s; } };
  const { source, field, where, row: n } = el.dataset, d = dash.data(source), rows = d.status === 'ok' && Array.isArray(d.data) ? d.data : [];
  const row = where == null ? rows[+(n ?? 0)] : rows.find((r) => where.split('&').every((p) => {
    const i = p.indexOf('='); return i > 0 && String(r[dec(p.slice(0, i)).trim()] ?? '') === dec(p.slice(i + 1)); }));
  const v = row?.[field];
  write(el, v == null || v === '' ? '—' : (fmt[source + '.' + field] ?? String)(v));
}
dash.draw('fig', fig);
for (const name of ['runs', 'now', 'copy']) dash.draw(name, () => {});
const due = new Map();
function timing() {
  said = chs.map((_, i) => says(i).map(words).join(' ')); durs = said.map((s) => s.length / 15 + 1);
  chs.forEach((_, i) => { let at = 0; for (const s of says(i)) { const w = words(s), j = said[i].indexOf(w, at); due.set(s, (j < 0 ? at : j) / 15); if (j >= 0) at = j + w.length; } });
  write(vs.querySelector('.vs-runs'), String(Math.round(d3.sum(durs) / 5) * 5));
}
function draw(i, t) {
  scenes[i]?.(still.matches || t >= paced(durs[i]) ? Infinity : t);
  const ss = says(i), k = begun ? ss.findLastIndex((s) => t >= paced(due.get(s))) : 0;
  ss.forEach((s, j) => s.classList.toggle('vs-on', j === Math.max(0, k)));
}
function frame() {
  draw(cur, t);
  mks.forEach((m, j) => { for (const bar of m.children) bar.style.width = (j < cur ? 1 : j > cur ? 0 : t === Infinity ? +begun : Math.min(1, t / paced(durs[cur]))) * 100 + '%'; });
}
function show(i, from) {
  if (i !== cur) { draw(cur, Infinity); if (!phase()) write(vs.querySelector('.vs-now'), called(i)); }
  [cur, t] = [i, from];
  for (const set of [chs, caps]) set.forEach((el, j) => el.classList.toggle('vs-on', j === i));
  chs.forEach((ch, j) => ch.setAttribute('aria-hidden', String(j !== i)));
  for (const set of [mks, caps]) set.forEach((el, j) => (j === i ? el.setAttribute('aria-current', 'true') : el.removeAttribute('aria-current')));
  mks.forEach((m, j) => m.getAttribute('aria-label') === called(j) || m.setAttribute('aria-label', called(j)));
  frame();
}
function tick(now) {
  raf = 0;
  if (!vs.isConnected || (phase() ? phase() !== 'playing' : !playing)) return;
  t += last ? Math.min(0.1, (now - last) / 1000) : 0; last = now;
  if (phase() || t < durs[cur]) frame();
  else if (cur < chs.length - 1) show(cur + 1, 0);
  else return play(false);
  raf = requestAnimationFrame(tick);
}
const kick = () => { if (!raf) { last = 0; raf = requestAnimationFrame(tick); } };
function look() { const on = phase() ? /starting|playing/.test(phase()) : playing; vs.classList.toggle('vs-playing', on); vs.querySelector('.vs-controls .vs-toggle')?.setAttribute('aria-label', on ? 'Pause' : 'Play'); }
function play(on) { playing = on; begun ||= on; look(); frame(); if (on) kick(); }
const from = (i) => (show(Math.max(0, Math.min(chs.length - 1, i)), 0), phase() || play(true));
const on = (q, f) => { const b = vs.querySelector('.vs-controls ' + q); if (b) b.onclick = f; };
on('.vs-toggle', () => phase() || (playing ? play(false) : t < durs[cur] ? play(true) : from(cur === chs.length - 1 ? 0 : cur)));
on('.vs-back', () => from(cur - 1));
on('.vs-next', () => from(cur + 1));
mks.forEach((m, j) => (m.onclick = () => from(j)));
on('.vs-cc', (e) => e.currentTarget.setAttribute('aria-pressed', String(!vs.classList.toggle('vs-hush'))));
stage.onpointermove = stage.onpointerdown = (e) => {
  const m = e.pointerType === 'mouse', move = e.type === 'pointermove', l = stage.classList;
  if (move && Math.abs(e.clientX - stage.x) + Math.abs(e.clientY - stage.y) < 4) return;
  [stage.x, stage.y] = [e.clientX, e.clientY];
  l.toggle('vs-idle', false); l.toggle('vs-touch', !m); l.toggle('vs-held', move && !!e.target.closest('.vs-controls'));
  clearTimeout(stage.idle); if (m) stage.idle = setTimeout(() => l.add('vs-idle'), 3000);
};
still.onchange = () => (vs.isConnected ? frame() : (still.onchange = null));

const mo = new MutationObserver(() => {
  if (!vs.isConnected) return mo.disconnect();
  const now = vs.querySelector('[data-dash-read-now]'), p = phase(), again = p === 'starting' && was !== 'starting';
  const through = heard && p === 'playing' && was === 'playing';
  const before = was; was = p;
  if (!p) { if (heard) t /= pace; heard = null; if (before) play(/starting|playing/.test(before)); return; }
  playing = false; begun = true; look();
  write(vs.querySelector('.vs-now'), '');
  if (now && (now !== heard || again)) {
    if (through) pace = Math.max(0.5, Math.min(1.5, t / (durs[cur] - 1) || 1));
    const i = caps.indexOf((heard = now));
    if (i >= 0) show(i, 0);
  }
  if (p === 'ended') show(cur, Infinity);
  kick();
});
mo.observe(root, { subtree: true, attributes: true, attributeFilter: ['data-dash-read-phase', 'data-dash-read-now'] });

function seat() {
  const top = vs.querySelector('.vs-in').getBoundingClientRect().top;
  chs.forEach((ch, i) => { const row = ch.querySelector('.vs-say, .vs-draw'); if (row && caps[i]) caps[i].style.top = row.getBoundingClientRect().top - top + 'px'; });
  if (durs.length) frame();
}

let box = vs.parentElement;
while (box.parentElement && !/auto|scroll/.test(getComputedStyle(box).overflowY)) box = box.parentElement;
const fit = () => vs.style.setProperty('--room', Math.max(240, box.getBoundingClientRect().bottom - box.scrollTop - vs.getBoundingClientRect().top
  - parseFloat(getComputedStyle(root).paddingBottom) - parseFloat(getComputedStyle(box).paddingBottom)) + 'px');
const ro = new ResizeObserver(() => (vs.isConnected ? (fit(), seat()) : ro.disconnect())); for (const el of [box, stage, ...vs.querySelectorAll('.vs-draw')]) ro.observe(el);
document.fonts.ready.then(() => vs.isConnected && seat());

dash.onData(fit);
dash.draw('settle', () => { timing(); show(cur, t); seat(); });
