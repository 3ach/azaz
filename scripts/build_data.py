#!/usr/bin/env python3
"""Build the per-model embedding assets the browser loads.

For each configured open-weight model this:
  1. downloads the model's `tokenizer.json` (cached under build_assets/),
  2. curates common English words that are a SINGLE token in that model,
  3. range-reads just the input-embedding tensor from the model's safetensors
     on the HuggingFace CDN (no multi-GB full-model download, no torch),
  4. quantizes the selected rows to uint16 per dimension, and
  5. writes  public/data/<key>/meta.json  (served) and
            data_src/<key>/embeddings.bin (build-only source matrix, NOT
            deployed — outside public/ so the browser never downloads it).

Finally it writes public/data/models.json — the index the picker reads.

The browser does not load these matrices directly. After this script, run

    node scripts/pack_tokens.mjs

which turns each data_src/<key>/embeddings.bin into the served
public/data/<key>/tokens.bin (fixed-size per-token records the page
range-fetches) and augments each meta.json with colMean/colStd + the layout.

Pure standard library + numpy, so it runs anywhere a recent Python does. Run:

    python3 scripts/build_data.py && node scripts/pack_tokens.mjs
"""
import json
import os
import struct
import time
import urllib.request

import numpy as np

HF = "https://huggingface.co/{repo}/resolve/main/{file}"
WORD_LIST_URL = (
    "https://raw.githubusercontent.com/first20hours/"
    "google-10000-english/master/google-10000-english-usa.txt"
)
CACHE = "build_assets"
OUT_ROOT = "public/data"
MAX_WORDS = 6000  # upper bound on curated vocabulary per model

# Each model: where to find it and which tensor holds the input embeddings.
MODELS = [
    {
        "key": "gpt2",
        "name": "GPT-2",
        "repo": "openai-community/gpt2",
        "safetensors": "model.safetensors",
        "embed_tensor": "wte.weight",
        "blurb": "OpenAI's 124M-parameter GPT-2 (2019) — 768 dimensions.",
    },
    {
        "key": "deepseek-coder-1.3b",
        "name": "DeepSeek Coder 1.3B",
        "repo": "deepseek-ai/deepseek-coder-1.3b-instruct",
        "safetensors": "model.safetensors",
        "embed_tensor": "model.embed_tokens.weight",
        "blurb": "DeepSeek's 1.3B code model — 2048 dimensions.",
    },
]


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
def http_get(url, rng=None, timeout=120, retries=5):
    """GET (optionally a byte range), retrying transient CDN/SSL hiccups."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "azaz-build"})
            if rng is not None:
                req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"    transient error ({e}); retrying in {wait}s")
            time.sleep(wait)


def cached(path, url):
    """Download `url` to build_assets/<path> once; return the local path."""
    local = os.path.join(CACHE, path)
    os.makedirs(os.path.dirname(local), exist_ok=True)
    if not os.path.exists(local):
        print(f"  downloading {url}")
        with open(local, "wb") as f:
            f.write(http_get(url))
    return local


# ---------------------------------------------------------------------------
# Byte-level BPE (GPT-2 scheme; shared by GPT-2 and DeepSeek's tokenizers)
# ---------------------------------------------------------------------------
def bytes_to_unicode():
    """GPT-2's reversible byte<->unicode table."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTE2UNI = bytes_to_unicode()
UNI2BYTE = {c: b for b, c in BYTE2UNI.items()}


def byte_encode(text):
    return "".join(BYTE2UNI[b] for b in text.encode("utf-8"))


def decode_token(tok):
    try:
        return bytearray(UNI2BYTE[c] for c in tok).decode("utf-8")
    except (KeyError, UnicodeDecodeError):
        return None


class BPE:
    """Minimal byte-level BPE encoder driven by a tokenizer.json model block."""

    def __init__(self, tokenizer_json_path):
        model = json.load(open(tokenizer_json_path))["model"]
        self.vocab = model["vocab"]  # token string -> id
        self.id_to_str = {i: t for t, i in self.vocab.items()}
        merges = model["merges"]
        pairs = (m if isinstance(m, list) else m.split(" ") for m in merges)
        self.ranks = {(a, b): i for i, (a, b) in enumerate(pairs)}

    def _merge(self, symbols):
        while len(symbols) >= 2:
            best, best_rank = None, None
            for pair in zip(symbols, symbols[1:]):
                r = self.ranks.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = pair, r
            if best is None:
                break
            a, b = best
            merged, i = [], 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    merged.append(a + b)
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            symbols = merged
        return symbols

    def single_token_id(self, word):
        """Return the token id if ' '+word encodes to exactly one token, else None."""
        symbols = self._merge(list(byte_encode(" " + word)))
        if len(symbols) != 1:
            return None
        return self.vocab.get(symbols[0])


# ---------------------------------------------------------------------------
# safetensors: read a single tensor via an HTTP range request
# ---------------------------------------------------------------------------
_ST_DTYPE = {
    "F32": ("<f4", None),
    "F16": ("<f2", None),
    "BF16": ("<u2", "bf16"),  # raw uint16, widened to f32 below
}


def read_embed_tensor(repo, fname, tensor_name):
    url = HF.format(repo=repo, file=fname)
    header_len = struct.unpack("<Q", http_get(url, (0, 7)))[0]
    header = json.loads(http_get(url, (8, 8 + header_len - 1)))
    if tensor_name not in header:
        raise RuntimeError(f"{tensor_name} not in {repo}/{fname}")
    info = header[tensor_name]
    dtype, shape = info["dtype"], info["shape"]
    start, end = info["data_offsets"]
    base = 8 + header_len
    print(f"  reading {tensor_name} {dtype}{shape} ({(end - start) >> 20} MB)")
    raw = http_get(url, (base + start, base + end - 1))
    np_dtype, special = _ST_DTYPE[dtype]
    arr = np.frombuffer(raw, dtype=np.dtype(np_dtype)).reshape(shape)
    if special == "bf16":  # widen bf16 -> f32 by placing it in the high 16 bits
        arr = (arr.astype(np.uint32) << 16).view(np.float32)
    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# Per-model build
# ---------------------------------------------------------------------------
def build_model(cfg, words):
    print(f"\n=== {cfg['name']} ({cfg['repo']}) ===")
    tok_path = cached(
        f"{cfg['key']}/tokenizer.json",
        HF.format(repo=cfg["repo"], file="tokenizer.json"),
    )
    bpe = BPE(tok_path)

    vocab, seen = [], set()
    for word in words:
        if len(vocab) >= MAX_WORDS:
            break
        tid = bpe.single_token_id(word)
        if tid is None or tid in seen:
            continue
        # round-trip the id back to a display string (" water")
        str_for_id = bpe.id_to_str.get(tid)
        text = decode_token(str_for_id) if str_for_id else None
        if text is None:
            continue
        seen.add(tid)
        vocab.append({"id": tid, "str": text})
    print(f"  curated {len(vocab)} single-token words")

    emb = read_embed_tensor(cfg["repo"], cfg["safetensors"], cfg["embed_tensor"])
    dim = emb.shape[1]
    ids = np.array([v["id"] for v in vocab])
    assert ids.max() < emb.shape[0], "token id outside embedding matrix"
    sub = emb[ids].astype(np.float32)  # [count, dim]

    # quick sanity: nearest cosine neighbours of a probe word
    normed = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-9)
    for probe in (" king", " water"):
        match = [i for i, v in enumerate(vocab) if v["str"] == probe]
        if match:
            sims = normed @ normed[match[0]]
            top = np.argsort(-sims)[1:6]
            print(f"  {probe!r} -> " + ", ".join(repr(vocab[j]["str"]) for j in top))

    # uint16 per-column quantization
    col_min = sub.min(axis=0)
    span = np.maximum(sub.max(axis=0) - col_min, 1e-9)
    scale = span / 65535.0
    q = np.round((sub - col_min) / scale).astype(np.uint16)
    max_err = np.abs((col_min + q.astype(np.float32) * scale) - sub).max()
    print(f"  max quantization error: {max_err:.3e}")

    out_dir = os.path.join(OUT_ROOT, cfg["key"])
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "embeddings.bin"), "wb") as f:
        f.write(q.tobytes(order="C"))
    meta = {
        "model": cfg["key"],
        "name": cfg["name"],
        "dim": int(dim),
        "count": len(vocab),
        "quant": "uint16-per-column",
        "colMin": [float(x) for x in col_min],
        "colScale": [float(x) for x in scale],
        "tokens": vocab,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    sz = os.path.getsize(os.path.join(out_dir, "embeddings.bin"))
    print(f"  wrote {out_dir}/  ({sz >> 10} KB bin, {len(vocab)}x{dim})")
    return {"key": cfg["key"], "name": cfg["name"], "dim": int(dim),
            "count": len(vocab), "blurb": cfg["blurb"]}


def main():
    word_path = cached("google-10000-english.txt", WORD_LIST_URL)
    words = [w.strip() for w in open(word_path) if w.strip()]
    index = [build_model(cfg, words) for cfg in MODELS]
    with open(os.path.join(OUT_ROOT, "models.json"), "w") as f:
        json.dump({"models": index}, f, indent=2)
    print(f"\nWrote {OUT_ROOT}/models.json with {len(index)} models")


if __name__ == "__main__":
    main()
