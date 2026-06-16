# The Embedding Explorer

An interactive, literary reading of **GPT-2's token embeddings**. Type a word
and wander its 768 coordinates — discovering, dimension by dimension, which
other words sit nearest and which lie farthest away.

Everything runs in the browser. Tokenization happens client-side; the embedding
data is a static asset. Nothing is sent anywhere.

🔗 Live (once Pages is enabled): `https://3ach.github.io/azaz/`

## How it works

- **Tokenizer.** The search box tokenizes your input with GPT-2's BPE
  (`r50k_base`) via [`gpt-tokenizer`](https://www.npmjs.com/package/gpt-tokenizer).
  A word that splits into several tokens renders as clickable chips — click the
  piece you want to explore. A single known token is selected automatically.
- **Embeddings.** The real `wte` (token-embedding) matrix of OpenAI's GPT-2
  (124M) — 50,257 tokens × 768 dimensions. The token ids produced by the
  browser tokenizer index directly into this matrix.
- **The sliders.** For the selected token, every one of the 768 dimensions gets
  a number line. Holding all *other* coordinates fixed, distance along a single
  axis is just the difference in that one value — so each line marks the
  **nearest** token and the **farthest** token in that dimension, alongside the
  selected word itself.
- **Ordering.** Dimensions are shown "most distinctive first" (where the word's
  value is most unusual versus the rest of the vocabulary) or by dimension
  number.

## Vocabulary

To stay small and fast, the page ships embeddings for ~6,000 of the most common
single-token English words (the comparison set). The word you explore must
resolve to one of these tokens; sub-tokens outside the set appear greyed out.

## Data pipeline

The embedding asset is generated once and committed under `public/data/`. To
regenerate it (requires internet access to the ONNX model + word list):

```bash
# 1. Download the source assets
curl -L -o build_assets/gpt2-10.onnx \
  https://github.com/onnx/models/raw/main/validated/text/machine_comprehension/gpt-2/model/gpt2-10.onnx
curl -L -o build_assets/google-10000-english.txt \
  https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-usa.txt

# 2. Pick the single-token common-word vocabulary (Node)
node scripts/pick_vocab.mjs

# 3. Extract + quantize the embeddings (Python)
python3 -m venv .venv && . .venv/bin/activate
pip install numpy onnx
python3 scripts/extract_embeddings.py
```

This writes `public/data/meta.json` (per-dimension scales + token list) and
`public/data/embeddings.bin` (uint16, per-column quantized, ~9 MB).

## Develop

```bash
npm install
npm run dev      # local dev server
npm run build    # production build -> dist/
npm run preview  # preview the production build
```

## Deploy

A GitHub Actions workflow (`.github/workflows/deploy.yml`) builds and publishes
`dist/` to GitHub Pages on every push to the deployment branch. **Enable it
once** under *Settings → Pages → Build and deployment → Source: GitHub Actions*.
The Vite `base` is set to `/azaz/` to match the project-site URL.
