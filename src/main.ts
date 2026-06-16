import './style.css';
import { encode, decode } from 'gpt-tokenizer/encoding/r50k_base';

// ----------------------------------------------------------------------------
// Types & data model
// ----------------------------------------------------------------------------
interface Meta {
  model: string;
  encoding: string;
  dim: number;
  count: number;
  quant: string;
  colMin: number[];
  colScale: number[];
  tokens: { id: number; str: string }[];
}

interface Data {
  dim: number;
  count: number;
  tokens: { id: number; str: string }[];
  idToIndex: Map<number, number>;
  q: Uint16Array; // [count * dim] quantized values
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
// The tokens the current input breaks into, and which one we're exploring.
let currentIds: number[] = [];
let selectedPos = -1;

// ----------------------------------------------------------------------------
// DOM handles
// ----------------------------------------------------------------------------
const $search = document.getElementById('search') as HTMLInputElement;
const $hint = document.getElementById('hint') as HTMLParagraphElement;
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

// Render a leading space (GPT-2's word-boundary marker) as a faint middot so
// the tokenization is legible; everything else is shown verbatim.
function showToken(str: string): string {
  return escapeHtml(str).replace(/^ /, '<span class="dot">·</span>');
}

function cleanWord(str: string): string {
  return escapeHtml(str.trim() || str.replace(/ /g, '␣'));
}

function val(t: number, d: number, dim: number): number {
  return data!.emb[t * dim + d];
}

// ----------------------------------------------------------------------------
// Load data
// ----------------------------------------------------------------------------
async function loadData(): Promise<void> {
  $dims.innerHTML = '<p class="loading">Gathering 768 dimensions…</p>';
  const base = import.meta.env.BASE_URL;
  const [meta, binBuf] = await Promise.all([
    fetch(`${base}data/meta.json`).then((r) => r.json() as Promise<Meta>),
    fetch(`${base}data/embeddings.bin`).then((r) => r.arrayBuffer()),
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
    const base2 = t * dim;
    for (let d = 0; d < dim; d++) {
      emb[base2 + d] = colMin[d] + q[base2 + d] * colScale[d];
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

  const idToIndex = new Map<number, number>();
  meta.tokens.forEach((tok, i) => idToIndex.set(tok.id, i));

  data = {
    dim,
    count,
    tokens: meta.tokens,
    idToIndex,
    q,
    emb,
    norms,
    colMin,
    colMax,
    colMean,
    colStd,
  };

  $dims.innerHTML = '';
}

// ----------------------------------------------------------------------------
// Tokenization
// ----------------------------------------------------------------------------
function handleInput(): void {
  const raw = $search.value.trim();
  $hint.textContent = '';
  if (!raw || !data) {
    $selection.hidden = true;
    selectedIndex = null;
    return;
  }

  // Treat the input as a word in running text (GPT-2's natural, space-prefixed
  // form). This is the token whose embedding the model actually uses most.
  currentIds = encode(' ' + raw);
  const firstKnown = currentIds.findIndex((id) => data!.idToIndex.has(id));

  if (firstKnown === -1) {
    $selection.hidden = true;
    selectedIndex = null;
    $hint.textContent =
      'That token isn’t in this page’s curated vocabulary. Try a more common word.';
    return;
  }

  if (currentIds.length > 1) {
    $hint.textContent =
      'This word breaks into several tokens — click a piece to explore it.';
  }
  // Default to the first in-vocabulary token; the strip lets you pick another.
  selectByPos(firstKnown);
}

// ----------------------------------------------------------------------------
// Selection & dimension rendering
// ----------------------------------------------------------------------------
// The "Now exploring" strip doubles as the token picker: every piece of the
// input is shown, the explored one is highlighted in colour, the rest are
// clickable, and out-of-vocabulary pieces are greyed out.
function renderTokenStrip(): void {
  if (!data) return;
  $selectedToken.innerHTML = currentIds
    .map((id, i) => {
      const known = data!.idToIndex.has(id);
      const cls =
        'tok' +
        (i === selectedPos ? ' active' : '') +
        (known ? '' : ' disabled');
      const title = known
        ? 'Explore this token'
        : 'Not in this page’s vocabulary';
      return `<button class="${cls}" data-pos="${i}"${
        known ? '' : ' disabled'
      } title="${title}">${showToken(decode([id]))}</button>`;
    })
    .join('');
}

function selectByPos(pos: number): void {
  if (!data) return;
  const idx = data.idToIndex.get(currentIds[pos]);
  if (idx === undefined) return;
  selectedPos = pos;
  selectedIndex = idx;
  $selection.hidden = false;
  renderTokenStrip();
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

// Clicking a piece in the "Now exploring" strip switches which token we explore.
$selectedToken.addEventListener('click', (e) => {
  const btn = (e.target as HTMLElement).closest('.tok') as HTMLElement | null;
  if (!btn || btn.classList.contains('disabled')) return;
  const pos = Number(btn.dataset.pos);
  if (!Number.isNaN(pos) && pos !== selectedPos) selectByPos(pos);
});

$sort.addEventListener('change', renderDimensions);

loadData()
  .then(() => {
    $search.disabled = false;
    $search.focus();
    if ($search.value.trim()) handleInput();
  })
  .catch((err) => {
    console.error(err);
    $dims.innerHTML =
      '<p class="loading">Could not load embedding data. Please reload.</p>';
  });
