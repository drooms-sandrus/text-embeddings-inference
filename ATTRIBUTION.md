# Attribution

This branch (`feat/cuda-bf16-pplx1`) incorporates work from several upstream
pull requests that had not yet been merged into `main` at the time this branch
was created. The contributions are listed below.

---

## PR #809 — "Add experimental BF16 support for Metal"

**Author:** Alvaro Bartolome ([@alvarobartt](https://github.com/alvarobartt))  
**PR:** https://github.com/huggingface/text-embeddings-inference/pull/809  
**Branch:** `origin/add-bfloat16-support` (also `origin/update-candle-wo-linking`)  
**Status at branch creation:** Draft / open

The bulk of this branch is built on top of PR #809. The following files were
taken from, or substantially derived from, that PR:

| File | What was taken |
|------|----------------|
| `Cargo.toml` | `candle 0.9.2` upgrade, `candle-extensions` static/dynamic linking, `cudarc 0.19` upgrade |
| `Cargo.lock` | Full dependency lock regenerated after the above |
| `Dockerfile-cuda` | Static-linking build args |
| `backends/candle/Cargo.toml` | Feature flag cleanup, `default-features = false` for candle-* |
| `backends/candle/src/compute_cap.rs` | `get_runtime_compute_cap` helper |
| `backends/candle/src/flash_attn.rs` | `use_flash_attn` / `FlashAttn` helper enum |
| `backends/candle/src/layers/index_select.rs` | BF16-exclusion on CUDA for `index_select` |
| `backends/candle/src/models/gemma3.rs` | `DType::BF16` minimum finite value / distance calc |
| `backends/candle/src/models/modernbert.rs` | `DType::BF16` minimum finite value |
| `backends/candle/src/models/mpnet.rs` | `DType::BF16` minimum finite value |
| `backends/candle/src/models/qwen3.rs` | `DType::BF16` minimum finite value; BF16 acceptance in `FlashQwen3` routing |
| `backends/src/dtype.rs` | `DType::Bfloat16` variant + `FromStr` + `Display` with wider feature-gating |
| `backends/candle/tests/common.rs` | Sequence pre-tokenizer iteration (`get_pre_tokenizers().to_vec()`) |
| `router/Cargo.toml` | `static-linking` / `dynamic-linking` forwarding to `text-embeddings-backend` |
| `router/src/lib.rs` | `dtype`/`torch_dtype` field on `ModelConfig`; `FromStr`-based dtype resolution from `config.json` |
| `router/src/main.rs` | Updated `--dtype` CLI help text |

Our branch additionally:
- Extends the CUDA BF16 path that PR #809 left as a `// NOTE: Temporarily left out`
  stub (the Metal-only BF16 guard in `backends/candle/src/lib.rs`) to be fully
  operational on CUDA with an SM ≥ 80 runtime check.
- Widens `FlashQwen3Model::load` to accept `DType::BF16` (PR #809 added a
  `FlashQwen3`-specific BF16 commit on `add-bfloat16-support` — commit `9387804`
  — which we adopted and extended).
- Makes `flash-attn` / `flash-attn-v1` imply the `cuda` feature in
  `backends/Cargo.toml` (our own addition to fix the clap enum compilation).
- Restores the `gemma3_text → Float32` default override in `router/src/lib.rs`
  that was present before PR #809 re-worked the dtype resolution block.

---

## PR #848 — "Support Flash Pplx1 model"

**Author:** kozistr ([@kozistr](https://github.com/kozistr))  
**PR:** https://github.com/huggingface/text-embeddings-inference/pull/848  
**Status at branch creation:** Closed (not merged)

The file `backends/candle/src/models/flash_pplx1.rs` is substantially based on
the model skeleton proposed in PR #848. That PR introduced a `FlashPplx1Model`
that delegated to `FlashQwen3Model` and applied the Pplx1-specific INT8
quantization head (`tanh() * 127`). Our implementation:

- Adopts the same structural pattern (inner `FlashQwen3Model` wrapper, `embed`
  delegation, `is_padded` delegation).
- Fixes a typo in the `Model` trait implementation that was present in the
  original PR.
- Changes the accepted dtype from FP16 to **BF16 only** (per alvarobartt's
  review comment on PR #848 explaining that the model was trained in BF16 and
  FP16 causes hidden-state overflow).
- Adds a defensive `vb.dtype() != DType::BF16` guard inside `load()` so the
  constraint is enforced at the wrapper boundary.

---

## `origin/update-candle-wo-linking`

**Author:** Alvaro Bartolome ([@alvarobartt](https://github.com/alvarobartt))  
**Branch:** `origin/update-candle-wo-linking`

This single-commit branch updates `candle` to the `no-default-linking` fork and
switches `candle-extensions` to the `allow-static-linking` branch. It was
incorporated into `origin/add-bfloat16-support` (commit `65080b3`) and is
therefore covered by the PR #809 attribution above. It is listed separately here
because `Cargo.toml` patch entries and the `Cargo.lock` re-generation originate
from this specific commit.
