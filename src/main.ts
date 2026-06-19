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
  colMean: number[]; // per-dimension mean (for distinctiveness ordering)
  colStd: number[];
  neighbours: number; // nearest-overall words stored per record
  recordBytes: number; // fixed size of one token's record in tokens.bin
  tokens: { id: number; str: string }[];
}

interface Data {
  base: string; // asset base URL (for range-fetching tokens.bin)
  key: string;
  dim: number;
  count: number;
  tokens: { id: number; str: string }[];
  byWord: Map<string, number>; // exact trimmed word -> row index
  byLower: Map<string, number>; // lowercased word -> row index (fallback)
  colMin: Float32Array;
  colMax: Float32Array;
  colScale: Float32Array;
  colMean: Float32Array; // per-dimension mean (for distinctiveness ordering)
  colStd: Float32Array;
  neighbours: number;
  recordBytes: number;
}

// One token's precomputed render payload, range-fetched from tokens.bin.
interface TokenRecord {
  values: Float32Array; // [dim] this token's own dequantized coordinates
  neighbours: number[]; // token indices, nearest first
  left: Uint16Array; // [dim] nearest word with this axis pushed to its min
  right: Uint16Array; // [dim] nearest word with this axis pushed to its max
}

const Q_MAX = 65535;
let data: Data | null = null;
let selectedIndex: number | null = null;

// Record for the token currently shown, plus a per-model cache (replaced on
// model switch). selectSeq guards against a slow fetch landing after a newer pick.
let currentRecord: TokenRecord | null = null;
let recordCache = new Map<number, TokenRecord>();
let selectSeq = 0;

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
  currentRecord = null;
  recordCache = new Map(); // drop the previous model's cached records
  $selection.hidden = true;
  $dims.innerHTML = '<p class="loading">Gathering the dimensions…</p>';
  $search.disabled = true;

  // Startup downloads only meta.json (token list + per-column stats). The matrix
  // stays on the server; each token's record is range-fetched on selection.
  const base = import.meta.env.BASE_URL;
  const meta = (await fetch(`${base}data/${key}/meta.json`).then((r) =>
    r.json(),
  )) as Meta;

  const { dim } = meta;
  const colMin = Float32Array.from(meta.colMin);
  const colScale = Float32Array.from(meta.colScale);
  const colMax = new Float32Array(dim);
  for (let d = 0; d < dim; d++) colMax[d] = colMin[d] + Q_MAX * colScale[d];

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
    base,
    key,
    dim,
    count: meta.count,
    tokens: meta.tokens,
    byWord,
    byLower,
    colMin,
    colMax,
    colScale,
    colMean: Float32Array.from(meta.colMean),
    colStd: Float32Array.from(meta.colStd),
    neighbours: meta.neighbours,
    recordBytes: meta.recordBytes,
  };

  $dims.innerHTML = '';
  $search.disabled = false;
}

// Range-fetch one token's record (offset = index * recordBytes) and unpack the
// fixed [values | neighbours | leftIdx | rightIdx] layout. Cached per model; the
// cache reference is captured up front so a mid-flight model switch can't poison
// the new model's cache.
async function fetchRecord(index: number): Promise<TokenRecord> {
  const cache = recordCache;
  const hit = cache.get(index);
  if (hit) return hit;

  const { base, key, dim, neighbours, recordBytes, colMin, colScale } = data!;
  const start = index * recordBytes;
  const end = start + recordBytes - 1;
  const res = await fetch(`${base}data/${key}/tokens.bin`, {
    headers: { Range: `bytes=${start}-${end}` },
  });
  const body = await res.arrayBuffer();
  // 206 gives just our slice; if a host ignores Range (200), index into the whole.
  const buf = res.status === 206 ? body : body.slice(start, start + recordBytes);
  const u16 = new Uint16Array(buf);

  const values = new Float32Array(dim);
  for (let d = 0; d < dim; d++) values[d] = colMin[d] + u16[d] * colScale[d];
  const nb: number[] = [];
  for (let n = 0; n < neighbours; n++) nb.push(u16[dim + n]);
  const left = u16.slice(dim + neighbours, dim + neighbours + dim);
  const right = u16.slice(dim + neighbours + dim, dim + neighbours + 2 * dim);

  const rec: TokenRecord = { values, neighbours: nb, left, right };
  cache.set(index, rec);
  return rec;
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

async function selectByIndex(idx: number): Promise<void> {
  if (!data) return;
  selectedIndex = idx;
  $selection.hidden = false;
  $selectedToken.innerHTML = `<span class="tok">${cleanWord(
    data.tokens[idx].str,
  )}</span>`;

  const seq = ++selectSeq;
  if (!recordCache.has(idx)) {
    $overall.textContent = '';
    $dims.innerHTML = '<p class="loading">Fetching this token…</p>';
  }
  let rec: TokenRecord;
  try {
    rec = await fetchRecord(idx);
  } catch {
    if (seq === selectSeq) {
      $dims.innerHTML =
        '<p class="loading">Could not load this token. Please retry.</p>';
    }
    return;
  }
  if (seq !== selectSeq) return; // a newer selection (or model switch) superseded us
  currentRecord = rec;
  renderOverall();
  renderDimensions();
}

// Bonus context: the closest words by full-vector cosine similarity, taken
// straight from the precomputed record (nearest first).
function renderOverall(): void {
  if (!currentRecord) return;
  const words = currentRecord.neighbours
    .map((i) => `<b>${cleanWord(data!.tokens[i].str)}</b>`)
    .join(', ');
  $overall.innerHTML = `nearest overall: ${words}`;
}

function renderDimensions(): void {
  if (!data || selectedIndex === null || !currentRecord) return;
  const { dim, colMin, colMax, colMean, colStd } = data;
  const { values, left, right } = currentRecord;
  const t = selectedIndex;

  // Order dimensions either by index or by how distinctive this word is along
  // each axis (|z-score| of its value within the column).
  const order = Array.from({ length: dim }, (_, d) => d);
  if ($sort.value === 'distinct') {
    order.sort((a, b) => {
      const za = Math.abs((values[a] - colMean[a]) / colStd[a]);
      const zb = Math.abs((values[b] - colMean[b]) / colStd[b]);
      return zb - za;
    });
  }

  const parts: string[] = [];
  for (const d of order) {
    const tv = values[d];
    const lo = colMin[d];
    const hi = colMax[d];

    // Precomputed: the nearest real word with this one coordinate pushed to the
    // column's min (left) / max (right), holding the rest at the target.
    const leftWord = cleanWord(data.tokens[left[d]].str);
    const rightWord = cleanWord(data.tokens[right[d]].str);

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
