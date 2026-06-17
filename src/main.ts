import './style.css';

// ----------------------------------------------------------------------------
// Types & data model
// ----------------------------------------------------------------------------
interface ModelInfo {
  key: string;
  name: string;
  dim: number;
  count: number;
  blurb: string;
}

interface Meta {
  model: string;
  name: string;
  dim: number;
  count: number;
  quant: string;
  colMin: number[];
  colScale: number[];
  tokens: { id: number; str: string }[];
}

interface Data {
  key: string;
  dim: number;
  count: number;
  tokens: { id: number; str: string }[];
  byWord: Map<string, number>; // exact trimmed word -> row index
  byLower: Map<string, number>; // lowercased word -> row index (fallback)
  emb: Float32Array; // [count * dim] dequantized
  norms: Float32Array; // [count] L2 norms
  colMin: Float32Array;
  colMax: Float32Array;
  colMean: Float32Array; // per-dimension mean (for distinctiveness ordering)
  colStd: Float32Array;
}

const Q_MAX = 65535;
let data: Data | null = null;
let selectedIndex: number | null = null;

// ----------------------------------------------------------------------------
// DOM handles
// ----------------------------------------------------------------------------
const $search = document.getElementById('search') as HTMLInputElement;
const $hint = document.getElementById('hint') as HTMLParagraphElement;
const $model = document.getElementById('model') as HTMLSelectElement;
const $selection = document.getElementById('selection') as HTMLElement;
const $selectedToken = document.getElementById('selected-token') as HTMLElement;
const $overall = document.getElementById('overall') as HTMLElement;
const $dims = document.getElementById('dims') as HTMLDivElement;
const $sort = document.getElementById('sort') as HTMLSelectElement;

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------
function escapeHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function cleanWord(str: string): string {
  return escapeHtml(str.trim() || str.replace(/ /g, '␣'));
}

function val(t: number, d: number, dim: number): number {
  return data!.emb[t * dim + d];
}

// ----------------------------------------------------------------------------
// Load data for a model
// ----------------------------------------------------------------------------
async function loadModels(): Promise<ModelInfo[]> {
  const base = import.meta.env.BASE_URL;
  const { models } = (await fetch(`${base}data/models.json`).then((r) =>
    r.json(),
  )) as { models: ModelInfo[] };
  return models;
}

async function loadData(key: string): Promise<void> {
  data = null;
  selectedIndex = null;
  nbCache = null;
  $selection.hidden = true;
  $dims.innerHTML = '<p class="loading">Gathering the dimensions…</p>';
  $search.disabled = true;

  const base = import.meta.env.BASE_URL;
  const [meta, binBuf] = await Promise.all([
    fetch(`${base}data/${key}/meta.json`).then((r) => r.json() as Promise<Meta>),
    fetch(`${base}data/${key}/embeddings.bin`).then((r) => r.arrayBuffer()),
  ]);

  const { dim, count } = meta;
  const q = new Uint16Array(binBuf);
  const colMin = Float32Array.from(meta.colMin);
  const colScale = Float32Array.from(meta.colScale);
  const colMax = new Float32Array(dim);
  for (let d = 0; d < dim; d++) colMax[d] = colMin[d] + Q_MAX * colScale[d];

  // Dequantize once for fast cosine / value lookups.
  const emb = new Float32Array(count * dim);
  for (let t = 0; t < count; t++) {
    const b = t * dim;
    for (let d = 0; d < dim; d++) {
      emb[b + d] = colMin[d] + q[b + d] * colScale[d];
    }
  }

  const norms = new Float32Array(count);
  for (let t = 0; t < count; t++) {
    let s = 0;
    const b = t * dim;
    for (let d = 0; d < dim; d++) s += emb[b + d] * emb[b + d];
    norms[t] = Math.sqrt(s) + 1e-9;
  }

  // Per-dimension mean & std for distinctiveness ordering.
  const colMean = new Float32Array(dim);
  const colStd = new Float32Array(dim);
  for (let d = 0; d < dim; d++) {
    let m = 0;
    for (let t = 0; t < count; t++) m += emb[t * dim + d];
    m /= count;
    let v = 0;
    for (let t = 0; t < count; t++) {
      const x = emb[t * dim + d] - m;
      v += x * x;
    }
    colMean[d] = m;
    colStd[d] = Math.sqrt(v / count) + 1e-9;
  }

  // Resolve a typed word to its row without a tokenizer: every token is a
  // space-prefixed common word, so we match on the trimmed string (with a
  // case-insensitive fallback). First occurrence wins.
  const byWord = new Map<string, number>();
  const byLower = new Map<string, number>();
  meta.tokens.forEach((tok, i) => {
    const w = tok.str.trim();
    if (!byWord.has(w)) byWord.set(w, i);
    const lw = w.toLowerCase();
    if (!byLower.has(lw)) byLower.set(lw, i);
  });

  data = {
    key,
    dim,
    count,
    tokens: meta.tokens,
    byWord,
    byLower,
    emb,
    norms,
    colMin,
    colMax,
    colMean,
    colStd,
  };

  $dims.innerHTML = '';
  $search.disabled = false;
}

// ----------------------------------------------------------------------------
// Word lookup & selection
// ----------------------------------------------------------------------------
function lookupWord(raw: string): number | undefined {
  if (!data) return undefined;
  return data.byWord.get(raw) ?? data.byLower.get(raw.toLowerCase());
}

function handleInput(): void {
  const raw = $search.value.trim();
  $hint.textContent = '';
  if (!raw || !data) {
    $selection.hidden = true;
    selectedIndex = null;
    return;
  }

  const idx = lookupWord(raw);
  if (idx === undefined) {
    $selection.hidden = true;
    selectedIndex = null;
    $hint.textContent = `“${raw}” isn’t a single token in ${modelName()}’s vocabulary — try a more common word.`;
    return;
  }
  selectByIndex(idx);
}

function modelName(): string {
  const opt = $model.selectedOptions[0];
  return opt ? opt.textContent || 'this model' : 'this model';
}

function selectByIndex(idx: number): void {
  if (!data) return;
  selectedIndex = idx;
  $selection.hidden = false;
  $selectedToken.innerHTML = `<span class="tok active">${cleanWord(
    data.tokens[idx].str,
  )}</span>`;
  renderOverall(idx);
  renderDimensions();
}

// Indices of every other token ordered by full-vector cosine similarity to the
// selected one, most similar first. Cached so re-sorts don't recompute it.
let nbCache: { index: number; order: Int32Array } | null = null;
function neighbourOrder(index: number): Int32Array {
  if (nbCache && nbCache.index === index) return nbCache.order;
  const { dim, count, emb, norms } = data!;
  const b = index * dim;
  const scored: { i: number; s: number }[] = [];
  for (let t = 0; t < count; t++) {
    if (t === index) continue;
    let dot = 0;
    const tb = t * dim;
    for (let d = 0; d < dim; d++) dot += emb[b + d] * emb[tb + d];
    scored.push({ i: t, s: dot / (norms[index] * norms[t]) });
  }
  scored.sort((a, c) => c.s - a.s);
  const order = Int32Array.from(scored, (x) => x.i);
  nbCache = { index, order };
  return order;
}

// Bonus context: the closest words by full-vector cosine similarity.
function renderOverall(index: number): void {
  const order = neighbourOrder(index);
  const words = Array.from(order.slice(0, 5))
    .map((i) => `<b>${cleanWord(data!.tokens[i].str)}</b>`)
    .join(', ');
  $overall.innerHTML = `nearest overall: ${words}`;
}

function renderDimensions(): void {
  if (!data || selectedIndex === null) return;
  const { dim, count, emb, colMin, colMax, colMean, colStd } = data;
  const t = selectedIndex;
  const tb = t * dim;

  // Order dimensions either by index or by how distinctive this word is along
  // each axis (|z-score| of its value within the column).
  const order = Array.from({ length: dim }, (_, d) => d);
  if ($sort.value === 'distinct') {
    order.sort((a, b) => {
      const za = Math.abs((val(t, a, dim) - colMean[a]) / colStd[a]);
      const zb = Math.abs((val(t, b, dim) - colMean[b]) / colStd[b]);
      return zb - za;
    });
  }

  // Squared distance from the target to every other word, computed once. Holding
  // all other coordinates at the target's values and moving one axis to value E
  // gives a point whose squared distance to word w is just
  //   full2(w) - (w_d - t_d)^2 + (w_d - E)^2
  // so we can find the nearest word to each axis-extreme cheaply, per dimension.
  const full2 = new Float64Array(count);
  for (let w = 0; w < count; w++) {
    if (w === t) continue;
    const wb = w * dim;
    let s = 0;
    for (let k = 0; k < dim; k++) {
      const diff = emb[wb + k] - emb[tb + k];
      s += diff * diff;
    }
    full2[w] = s;
  }

  const parts: string[] = [];
  for (const d of order) {
    const tv = emb[tb + d];
    const lo = colMin[d];
    const hi = colMax[d];

    // Push this one coordinate to the column's min, then max (the ends of the
    // axis), holding the rest at the target — which real word sits nearest each?
    let loI = -1;
    let hiI = -1;
    let loBest = Infinity;
    let hiBest = Infinity;
    for (let w = 0; w < count; w++) {
      if (w === t) continue;
      const wd = emb[w * dim + d];
      const perp = full2[w] - (wd - tv) * (wd - tv);
      const sLo = perp + (wd - lo) * (wd - lo);
      const sHi = perp + (wd - hi) * (wd - hi);
      if (sLo < loBest) {
        loBest = sLo;
        loI = w;
      }
      if (sHi < hiBest) {
        hiBest = sHi;
        hiI = w;
      }
    }

    const leftWord = cleanWord(data.tokens[loI].str);
    const rightWord = cleanWord(data.tokens[hiI].str);

    // The axis runs the full coordinate range; the target's bubble sits at its
    // value, between the min end (left) and max end (right).
    const span = hi - lo || 1;
    const targetPos = Math.max(0, Math.min(100, ((tv - lo) / span) * 100));

    parts.push(
      `<div class="row">
        <span class="dimno">${d}</span>
        <span class="word left" title="nearest at min: ${leftWord}">${leftWord}</span>
        <span class="track">
          <span class="line"></span>
          <span class="cap left" title="${leftWord} · ${lo.toFixed(3)}"></span>
          <span class="cap right" title="${rightWord} · ${hi.toFixed(3)}"></span>
          <span class="pip target" style="left:${targetPos}%" title="${cleanWord(
            data.tokens[t].str,
          )} · ${tv.toFixed(3)}"></span>
        </span>
        <span class="word right" title="nearest at max: ${rightWord}">${rightWord}</span>
      </div>`,
    );
  }
  $dims.innerHTML = parts.join('');
}

// ----------------------------------------------------------------------------
// Wire up
// ----------------------------------------------------------------------------
let debounce: number | undefined;
$search.addEventListener('input', () => {
  window.clearTimeout(debounce);
  debounce = window.setTimeout(handleInput, 140);
});

$sort.addEventListener('change', renderDimensions);

// Switching model reloads its embeddings and re-resolves the current word.
$model.addEventListener('change', () => {
  switchModel($model.value);
});

async function switchModel(key: string): Promise<void> {
  try {
    await loadData(key);
    $search.focus();
    if ($search.value.trim()) handleInput();
  } catch (err) {
    console.error(err);
    $dims.innerHTML =
      '<p class="loading">Could not load this model’s data. Please reload.</p>';
  }
}

async function init(): Promise<void> {
  const models = await loadModels();
  $model.innerHTML = models
    .map((m) => `<option value="${m.key}" title="${escapeHtml(m.blurb)}">${escapeHtml(m.name)}</option>`)
    .join('');
  await switchModel(models[0].key);
}

init().catch((err) => {
  console.error(err);
  $dims.innerHTML =
    '<p class="loading">Could not load embedding data. Please reload.</p>';
});
