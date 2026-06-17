# The Embedding Explorer

An interactive reading of **GPT-2's token embeddings**, running entirely in the
browser. Live at `https://azaz.zachzundel.com/`

## Develop

```bash
npm install
npm run dev      # local dev server
npm run build    # production build -> dist/
npm run preview  # preview the production build
```

## Regenerate the embedding data

The embedding asset under `public/data/` is generated once and committed. To
rebuild it (requires internet access to the ONNX model + word list):

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

This writes `public/data/meta.json` and `public/data/embeddings.bin`.

## Deploy

A GitHub Actions workflow (`.github/workflows/deploy.yml`) builds and publishes
`dist/` to GitHub Pages on every push to the deployment branch. Enable it once
under *Settings → Pages → Build and deployment → Source: GitHub Actions*, and set
the custom domain to `azaz.zachzundel.com` (the `public/CNAME` file is published
with the site). The Vite `base` is `/` because the site is served from the root
of its own domain.
