// Build a curated vocabulary of common English words that are SINGLE GPT-2
// (r50k_base) tokens. Words are taken in frequency order, encoded with a
// leading space (their natural in-text form), and kept only if they encode to
// exactly one token. The resulting token ids index directly into the GPT-2
// `wte` embedding matrix, so the Python extraction step and the browser
// tokenizer agree perfectly.
//
// Output: build_assets/vocab.json  ->  [{ id, str }]
import { encode, decode } from 'gpt-tokenizer/encoding/r50k_base';
import { readFileSync, writeFileSync } from 'node:fs';

const WORD_LIST = 'build_assets/google-10000-english.txt';
const OUT = 'build_assets/vocab.json';
const MAX = 6000; // upper bound on vocabulary size

const words = readFileSync(WORD_LIST, 'utf8')
  .split('\n')
  .map((w) => w.trim())
  .filter(Boolean);

const seen = new Set();
const vocab = [];

for (const word of words) {
  if (vocab.length >= MAX) break;
  // Natural in-text form carries a leading space in GPT-2's vocabulary.
  const ids = encode(' ' + word);
  if (ids.length !== 1) continue; // skip words that fragment into multiple tokens
  const id = ids[0];
  if (seen.has(id)) continue;
  seen.add(id);
  vocab.push({ id, str: decode([id]) });
}

writeFileSync(OUT, JSON.stringify(vocab));
console.log(`Selected ${vocab.length} single-token words -> ${OUT}`);
console.log('Examples:', vocab.slice(0, 8).map((v) => JSON.stringify(v.str)).join(' '));
