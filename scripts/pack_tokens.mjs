/**
 * Pack per-token record blobs for range-request serving, for every model.
 *
 * Consumes the matrices that `build_data.py` produced — the source matrix
 * data_src/<key>/embeddings.bin plus the served public/data/<key>/meta.json —
 * and precomputes, for every token, exactly what the UI renders on selection:
 *
 *   - the token's own quantized values (for the pip position & z-score sort)
 *   - the 5 nearest words overall (full-vector cosine)
 *   - per dimension, the nearest real word with that axis pushed to the column
 *     min / max while the rest are held at the target (target-relative extremes)
 *
 * These become fixed-size records in public/data/<key>/tokens.bin, so the
 * browser range-fetches one record (offset = index * recordBytes) instead of
 * downloading the whole matrix. meta.json is augmented with colMean/colStd
 * (formerly computed in the browser) plus the record layout. Numbers come out
 * identical to the old client-side computation — both work from the same
 * dequantized values.
 *
 *   node scripts/pack_tokens.mjs
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const DATA = join(ROOT, 'public', 'data'); // served: meta.json (read+augment), tokens.bin (write)
const SRC = join(ROOT, 'data_src'); // build-only source matrices, not deployed
const NEIGHBOURS = 5; // nearest-overall words stored per token

function packModel(key) {
  console.log(`\n=== ${key} ===`);
  const metaPath = join(DATA, key, 'meta.json');
  const meta = JSON.parse(readFileSync(metaPath, 'utf8'));
  const { dim, count } = meta;
  const colMin = Float64Array.from(meta.colMin);
  const colScale = Float64Array.from(meta.colScale);
  const colMax = Float64Array.from({ length: dim }, (_, d) => colMin[d] + 65535 * colScale[d]);

  const raw = readFileSync(join(SRC, key, 'embeddings.bin'));
  const q = new Uint16Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
  if (q.length !== count * dim) throw new Error(`${key}: bin has ${q.length} u16, expected ${count * dim}`);

  const emb = new Float32Array(count * dim);
  for (let t = 0; t < count; t++) {
    const b = t * dim;
    for (let d = 0; d < dim; d++) emb[b + d] = colMin[d] + q[b + d] * colScale[d];
  }

  // Per-dimension mean & std (was computed in the browser; precompute here).
  const colMean = new Float64Array(dim);
  const colStd = new Float64Array(dim);
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

  // Unit-normalized rows for cosine neighbour search.
  const nemb = new Float32Array(count * dim);
  for (let t = 0; t < count; t++) {
    const b = t * dim;
    let s = 0;
    for (let d = 0; d < dim; d++) s += emb[b + d] * emb[b + d];
    const inv = 1 / (Math.sqrt(s) + 1e-9);
    for (let d = 0; d < dim; d++) nemb[b + d] = emb[b + d] * inv;
  }

  // Record layout (uint16 LE): [dim values][NEIGHBOURS idx][dim leftIdx][dim rightIdx]
  const REC16 = dim + NEIGHBOURS + dim + dim;
  const recordBytes = REC16 * 2;
  const out = new Uint16Array(count * REC16);

  const full2 = new Float64Array(count);
  const topI = new Int32Array(NEIGHBOURS);
  const topS = new Float64Array(NEIGHBOURS);
  const loBest = new Float64Array(dim);
  const hiBest = new Float64Array(dim);
  const loI = new Int32Array(dim);
  const hiI = new Int32Array(dim);

  const t0 = process.hrtime.bigint();
  for (let t = 0; t < count; t++) {
    const tb = t * dim;

    // Top-NEIGHBOURS by cosine (ascending insertion, topS[0] = worst kept).
    topS.fill(-Infinity);
    topI.fill(-1);
    for (let w = 0; w < count; w++) {
      if (w === t) continue;
      const wb = w * dim;
      let dot = 0;
      for (let d = 0; d < dim; d++) dot += nemb[tb + d] * nemb[wb + d];
      if (dot <= topS[0]) continue;
      let p = 0;
      while (p < NEIGHBOURS - 1 && dot > topS[p + 1]) {
        topS[p] = topS[p + 1];
        topI[p] = topI[p + 1];
        p++;
      }
      topS[p] = dot;
      topI[p] = w;
    }

    // Squared distance to every word (once), then per-axis extremes via
    //   dist with axis d at E  =  full2(w) - (w_d - t_d)^2 + (w_d - E)^2.
    for (let w = 0; w < count; w++) {
      if (w === t) { full2[w] = Infinity; continue; }
      const wb = w * dim;
      let s = 0;
      for (let d = 0; d < dim; d++) {
        const diff = emb[wb + d] - emb[tb + d];
        s += diff * diff;
      }
      full2[w] = s;
    }

    loBest.fill(Infinity);
    hiBest.fill(Infinity);
    for (let w = 0; w < count; w++) {
      if (w === t) continue;
      const wb = w * dim;
      const f2 = full2[w];
      for (let d = 0; d < dim; d++) {
        const wd = emb[wb + d];
        const dt = wd - emb[tb + d];
        const perp = f2 - dt * dt;
        const a = wd - colMin[d];
        const sLo = perp + a * a;
        if (sLo < loBest[d]) { loBest[d] = sLo; loI[d] = w; }
        const b = wd - colMax[d];
        const sHi = perp + b * b;
        if (sHi < hiBest[d]) { hiBest[d] = sHi; hiI[d] = w; }
      }
    }

    let o = t * REC16;
    for (let d = 0; d < dim; d++) out[o++] = q[tb + d];
    for (let n = NEIGHBOURS - 1; n >= 0; n--) out[o++] = topI[n] < 0 ? 0 : topI[n]; // nearest first
    for (let d = 0; d < dim; d++) out[o++] = loI[d];
    for (let d = 0; d < dim; d++) out[o++] = hiI[d];

    if ((t + 1) % 500 === 0 || t === count - 1) {
      const secs = Number(process.hrtime.bigint() - t0) / 1e9;
      console.log(`  ${t + 1}/${count} (${secs.toFixed(1)}s)`);
    }
  }

  writeFileSync(join(DATA, key, 'tokens.bin'), Buffer.from(out.buffer));
  meta.colMean = Array.from(colMean, (x) => Number(x));
  meta.colStd = Array.from(colStd, (x) => Number(x));
  meta.neighbours = NEIGHBOURS;
  meta.recordBytes = recordBytes;
  meta.layout = 'values,neighbours,leftIdx,rightIdx';
  writeFileSync(metaPath, JSON.stringify(meta));
  console.log(`  wrote tokens.bin (${count} x ${recordBytes} = ${count * recordBytes} bytes)`);
}

const { models } = JSON.parse(readFileSync(join(DATA, 'models.json'), 'utf8'));
for (const m of models) packModel(m.key);
console.log(`\nPacked ${models.length} model(s).`);
