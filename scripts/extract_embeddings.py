"""Extract GPT-2 token embeddings for the curated vocabulary.

Reads the real `wte` (token embedding) matrix from the ONNX GPT-2 model,
selects the rows for our curated single-token words, quantizes them to uint16
per dimension (so within-column ordering used by the nearest/furthest feature
is preserved with high fidelity), and writes static assets the browser loads
directly:

  public/data/meta.json        metadata + per-dimension min/scale + token list
  public/data/embeddings.bin    uint16 little-endian, row-major [count x dim]
"""
import json
import struct
import numpy as np
import onnx
from onnx import numpy_helper

ONNX_PATH = "build_assets/gpt2-10.onnx"
VOCAB_PATH = "build_assets/vocab.json"
OUT_BIN = "public/data/embeddings.bin"
OUT_META = "public/data/meta.json"
EMB_NAME = "wte.weight"


def load_wte():
    model = onnx.load(ONNX_PATH)
    for init in model.graph.initializer:
        if init.name == EMB_NAME:
            return numpy_helper.to_array(init).astype(np.float32)
    raise RuntimeError(f"{EMB_NAME} not found in {ONNX_PATH}")


def sanity_check(wte, vocab):
    """Print full-vector cosine neighbours for a few probe words to confirm the
    embeddings are real and token ids line up with the tokenizer."""
    id_to_str = {v["id"]: v["str"] for v in vocab}
    ids = np.array([v["id"] for v in vocab])
    sub = wte[ids]
    normed = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-9)
    for probe in [" king", " water", " London", " happy", " three"]:
        match = [i for i, v in enumerate(vocab) if v["str"] == probe]
        if not match:
            print(f"  [{probe!r}] not in vocab")
            continue
        i = match[0]
        sims = normed @ normed[i]
        top = np.argsort(-sims)[1:6]
        neigh = ", ".join(f"{vocab[j]['str']!r}" for j in top)
        print(f"  {probe!r:>10} -> {neigh}")


def main():
    print("Loading wte ...")
    wte = load_wte()
    print("wte shape:", wte.shape)

    vocab = json.load(open(VOCAB_PATH))
    ids = np.array([v["id"] for v in vocab])
    assert ids.max() < wte.shape[0]
    dim = wte.shape[1]
    count = len(vocab)

    print("Sanity check (full-vector cosine neighbours):")
    sanity_check(wte, vocab)

    sub = wte[ids].astype(np.float32)  # [count, dim]

    # Per-dimension (column) min/max -> uint16 quantization.
    col_min = sub.min(axis=0)
    col_max = sub.max(axis=0)
    span = np.maximum(col_max - col_min, 1e-9)
    scale = span / 65535.0
    q = np.round((sub - col_min) / scale).astype(np.uint16)

    # Round-trip error report.
    recon = col_min + q.astype(np.float32) * scale
    max_err = np.abs(recon - sub).max()
    print(f"max quantization error: {max_err:.6e}")

    with open(OUT_BIN, "wb") as f:
        f.write(q.tobytes(order="C"))

    meta = {
        "model": "gpt2",
        "encoding": "r50k_base",
        "dim": int(dim),
        "count": int(count),
        "quant": "uint16-per-column",
        "colMin": [float(x) for x in col_min],
        "colScale": [float(x) for x in scale],
        "tokens": [{"id": int(v["id"]), "str": v["str"]} for v in vocab],
    }
    with open(OUT_META, "w") as f:
        json.dump(meta, f)

    import os
    print(f"Wrote {OUT_BIN} ({os.path.getsize(OUT_BIN)} bytes)")
    print(f"Wrote {OUT_META} ({os.path.getsize(OUT_META)} bytes)")


if __name__ == "__main__":
    main()
