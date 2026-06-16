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

// ----------------------------------------------------------------------------
// DOM handles
// ----------------------------------------------------------------------------
const $search = document.getElementById('search') as HTMLInputElement;
const $tokens = document.getElementById('tokens') as HTMLDivElement;
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
// Tokenization & chips
// ----------------------------------------------------------------------------
function renderTokens(): void {
  const raw = $search.value.trim();
  $tokens.innerHTML = '';
  $hint.textContent = '';
  if (!raw || !data) {
    return;
  }

  // Treat the input as a word in running text (GPT-2's natural, space-prefixed
  // form). This is the token whose embedding the model actually uses most.
  const ids = encode(' ' + raw);
  const chips: HTMLButtonElement[] = [];

  ids.forEach((id) => {
    const str = decode([id]);
    const idx = data!.idToIndex.get(id);
    const chip = document.createElement('button');
    chip.className = 'chip' + (idx === undefined ? ' disabled' : '');
    chip.innerHTML = showToken(str);
    chip.title =
      idx === undefined
        ? 'Not in this page’s vocabulary'
        : 'Explore this token';
    if (idx !== undefined) {
      chip.addEventListener('click', () => select(idx, chip));
    }
    $tokens.appendChild(chip);
    chips.push(chip);
  });

  const inVocab = ids.filter((id) => data!.idToIndex.has(id));

  if (ids.length === 1 && inVocab.length === 1) {
    // Single, known token: select it automatically.
    select(data!.idToIndex.get(ids[0])!, chips[0]);
  } else if (ids.length > 1) {
    $hint.textContent =
      'This word breaks into several tokens — click the piece you want to explore.';
  } else if (inVocab.length === 0) {
    $hint.textContent =
      'That token isn’t in this page’s curated vocabulary. Try a more common word.';
  }
}

function setActiveChip(active: HTMLElement | null): void {
  $tokens.querySelectorAll('.chip').forEach((c) => c.classList.remove('active'));
  active?.classList.add('active');
}

// ----------------------------------------------------------------------------
// Selection & dimension rendering
// ----------------------------------------------------------------------------
function select(index: number, chip?: HTMLElement | null): void {
  if (!data) return;
  selectedIndex = index;
  setActiveChip(chip ?? null);
  $selectedToken.innerHTML = showToken(data.tokens[index].str);
  $selection.hidden = false;
  renderOverall(index);
  renderDimensions();
  $selection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Bonus context: the closest words by full-vector cosine similarity.
function renderOverall(index: number): void {
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
  const words = scored
    .slice(0, 5)
    .map((x) => `<b>${cleanWord(data!.tokens[x.i].str)}</b>`)
    .join(', ');
  $overall.innerHTML = `nearest overall: ${words}`;
}

function renderDimensions(): void {
  if (!data || selectedIndex === null) return;
  const { dim, count, q, emb, colMin, colMax, colMean, colStd } = data;
  const t = selectedIndex;

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

  const parts: string[] = [];
  for (const d of order) {
    const tq = q[t * dim + d];

    // Per-dimension nearest / farthest: holding every other coordinate fixed,
    // distance collapses to |difference| along this single axis.
    let nearI = -1;
    let farI = -1;
    let nearD = Infinity;
    let farD = -1;
    for (let j = 0; j < count; j++) {
      if (j === t) continue;
      const diff = Math.abs(q[j * dim + d] - tq);
      if (diff < nearD) {
        nearD = diff;
        nearI = j;
      }
      if (diff > farD) {
        farD = diff;
        farI = j;
      }
    }

    const span = colMax[d] - colMin[d] || 1;
    const pos = (v: number) => {
      const p = ((v - colMin[d]) / span) * 100;
      return Math.max(0, Math.min(100, p));
    };
    const tv = emb[t * dim + d];
    const nv = emb[nearI * dim + d];
    const fv = emb[farI * dim + d];

    // The nearest and farthest each take the flanking column on the side of the
    // line their value falls on (smaller value -> left). Colour still tells the
    // two apart, so a word reads on the same side as its pip on the track.
    const near = {
      cls: 'near',
      label: 'nearest',
      v: nv,
      word: cleanWord(data.tokens[nearI].str),
    };
    const far = {
      cls: 'far',
      label: 'farthest',
      v: fv,
      word: cleanWord(data.tokens[farI].str),
    };
    const [left, right] = nv <= fv ? [near, far] : [far, near];
    const wordCell = (c: typeof near, side: 'left' | 'right') =>
      `<span class="word ${side} ${c.cls}" title="${c.label}: ${c.word} · ${c.v.toFixed(
        3,
      )}">${c.word}</span>`;

    parts.push(
      `<div class="row">
        <span class="dimno">${d}</span>
        ${wordCell(left, 'left')}
        <span class="track">
          <span class="line"></span>
          <span class="pip far" style="left:${pos(fv)}%" title="farthest: ${
            far.word
          } · ${fv.toFixed(3)}"></span>
          <span class="pip near" style="left:${pos(nv)}%" title="nearest: ${
            near.word
          } · ${nv.toFixed(3)}"></span>
          <span class="pip target" style="left:${pos(tv)}%" title="${cleanWord(
            data.tokens[t].str,
          )} · ${tv.toFixed(3)}"></span>
        </span>
        ${wordCell(right, 'right')}
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
  debounce = window.setTimeout(renderTokens, 140);
});

$sort.addEventListener('change', renderDimensions);

loadData()
  .then(() => {
    $search.disabled = false;
    $search.focus();
    if ($search.value.trim()) renderTokens();
  })
  .catch((err) => {
    console.error(err);
    $dims.innerHTML =
      '<p class="loading">Could not load embedding data. Please reload.</p>';
  });
