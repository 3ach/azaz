# The Embedding Explorer

An interactive reading of **language-model token embeddings**, running entirely
in the browser. Pick a model (GPT-2 or DeepSeek Coder 1.3B), type a word, and
wander its dimensions. Live at `https://azaz.zachzundel.com/`

## Develop

```bash
npm install
npm run dev      # local dev server
npm run build    # production build -> dist/
npm run preview  # preview the production build
```

## Regenerate the embedding data

The per-model assets under `public/data/` are generated once and committed (the
deploy only runs `npm run build`, so it can't download weights at deploy time).
To rebuild them:

```bash
npm run build:data   # python3 scripts/build_data.py
```

For each model in `scripts/build_data.py`'s `MODELS` list this:

1. downloads the model's `tokenizer.json` from HuggingFace,
2. curates common English words that are a single token in that model,
3. range-reads just the input-embedding tensor from the model's safetensors on
   the HuggingFace CDN (no multi-GB full-model download, no torch),
4. quantizes the selected rows to uint16 per dimension.

It writes `public/data/<model>/{meta.json,embeddings.bin}` plus
`public/data/models.json` (the index the in-page picker reads). The script needs
only Python 3 + numpy and internet access; add a model by appending to `MODELS`.
Gated repos (e.g. official Llama) additionally need an HF token.

## Deploy

A GitHub Actions workflow (`.github/workflows/deploy.yml`) builds and publishes
`dist/` to GitHub Pages on every push to the deployment branch. Enable it once
under *Settings → Pages → Build and deployment → Source: GitHub Actions*, and set
the custom domain to `azaz.zachzundel.com` (the `public/CNAME` file is published
with the site). The Vite `base` is `/` because the site is served from the root
of its own domain.
