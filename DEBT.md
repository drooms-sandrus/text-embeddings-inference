# Technical Debt

## Table of Contents

1. [Technical debt introduced in PR #809](#technical-debt-introduced-in-pr-809)
   - [Cargo.toml — dependency changes](#cargotoml--dependency-changes)
   - [Dockerfile-cuda — build environment changes](#dockerfile-cuda--build-environment-changes)
   - [backends/candle/src/layers/index_select.rs — loss of CUDA-optimised index_select](#backendscandlesrclayersindex_selectrs--loss-of-cuda-optimised-index_select)
   - [backends/src/dtype.rs — `FromStr` impl mirrors `#[cfg]` gates on the enum](#backendssrcdtypers--fromstr-impl-mirrors-cfg-gates-on-the-enum)
2. [Technical debt introduced in this branch (feat/cuda-bf16-pplx1)](#technical-debt-introduced-in-this-branch-featcuda-bf16-pplx1)
   - [backends/Cargo.toml — flash-attn features now imply cuda](#backendscargotoml--flash-attn-features-now-imply-cuda)
   - [backends/candle/tests/common.rs — Sequence::get\_pre\_tokenizers() API compatibility](#backendscandletestscommonrs--sequenceget_pre_tokenizers-api-compatibility)
   - [router/src/lib.rs — dtype resolution and gemma3 override](#routersrclibrs--dtype-resolution-and-gemma3-override)
   - [Pre-merge follow-ups](#pre-merge-follow-ups)
   - [Post-merge follow-up tickets](#post-merge-follow-up-tickets)

---

## Technical debt introduced in PR #809

### Cargo.toml — dependency changes

All changes below were introduced as part of the candle 0.8 → 0.9.2 upgrade (PR #809) applied in commit `8fdf24b`.

#### What changed and why

| Change | Reason |
|--------|--------|
| `cudarc` 0.13 → 0.19; `cuda-12020` → `cuda-version-from-build-system` | cudarc 0.14 retired the named CUDA-version features in favour of build-system detection; candle 0.9.2 requires cudarc ≥ 0.17 |
| `candle-cublaslt / candle-layer-norm / candle-rotary / candle-flash-attn-v1`: added `default-features = false` | These crates are now patched to the `allow-static-linking` branch of `candle-extensions` which splits linking into explicit `static-linking`/`dynamic-linking` feature flags; `default-features = false` suppresses the crate's old bundled defaults so the workspace controls linking centrally |
| `candle-index-select-cu` removed | Replaced by `backends/candle/src/layers/index_select.rs` using candle's native tensor op; the old crate is incompatible with cudarc 0.19 |
| `cudarc` removed from `[patch.crates-io]` | cudarc 0.19 on crates.io has everything needed; the personal `Narsil/cudarc` fork is no longer required |
| `[patch.crates-io]`: branch refs (`no-default-linking`, `allow-static-linking`) instead of pinned `rev` SHAs | PR #809 chose branch tracking for convenience while the candle 0.9.x line was still evolving |
| `candle-cublaslt / candle-layer-norm / candle-rotary / candle-flash-attn-v1` added to `[patch.crates-io]` | Published crates.io versions lack the `static-linking`/`dynamic-linking` feature gates; the `allow-static-linking` branch adds them |

#### Debt items to resolve before upstreaming

**1. Replace branch refs with pinned commit SHAs** (blocking)

`[patch.crates-io]` currently uses mutable branch names:

```toml
candle = { git = "...", branch = "no-default-linking", ... }
candle-cublaslt = { git = "...", branch = "allow-static-linking" }
# etc.
```

A push to either branch silently changes what gets compiled. `Cargo.lock` pins the resolved SHA locally, but external contributors, CI rebuilds on a clean clone, and the upstream maintainers cannot guarantee reproducibility.

**Fix**: after stabilising the desired state, resolve the actual commit SHAs from `Cargo.lock` and hard-pin them:

```toml
candle = { git = "https://github.com/huggingface/candle", rev = "<sha>", package = "candle-core" }
candle-cublaslt = { git = "https://github.com/huggingface/candle-extensions", rev = "<sha>" }
# etc.
```

**2. Publish `candle-extensions` changes to crates.io before upstreaming** (blocking)

The `allow-static-linking` branch of `huggingface/candle-extensions` adds the `static-linking`/`dynamic-linking` feature gates to `candle-cublaslt`, `candle-layer-norm`, `candle-rotary`, and `candle-flash-attn-v1`, but these are not yet on crates.io. An upstream PR that depends on an unpublished branch is difficult for maintainers to accept and breaks `cargo install` workflows.

**Fix**: either publish new patch releases of the four `candle-extensions` crates with those features, or (interim) switch to `rev`-pinned patches as per item 1 and note in the PR that publication is in progress.

#### Impact on unrelated functionality

The functional changes (cudarc bump, index_select replacement, removal of the Narsil fork) are low-risk and correct. The only runtime-observable difference for users not using this branch's new features is that `--dtype bfloat16` is now accepted on CUDA (SM ≥ 80) where it was previously rejected. All other model/dtype combinations behave identically to `main`.

---

### Dockerfile-cuda — build environment changes

All three changes were introduced together in commit `8fdf24b` as build-environment consequences of the PR #809 Cargo dependency upgrade.

#### What changed and why

##### `git` added to the `apt-get install` list

Before PR #809, every `[patch.crates-io]` entry in `Cargo.toml` used a pinned `rev = "<sha>"`. After PR #809, four new `candle-extensions` entries (`candle-cublaslt`, `candle-layer-norm`, `candle-rotary`, `candle-flash-attn-v1`) use `branch = "allow-static-linking"` — mutable branch-tracking references. When Cargo resolves branch-tracked git patches in a cold Docker layer (no pre-populated `~/.cargo/git` cache), it may invoke the system `git` binary for `fetch`/`ref-update` operations that the embedded `libgit2` delegates to the CLI in certain Docker network environments. Build scripts of the patched crates may also call `git` directly. The system `git` package was absent from the original builder image and is now explicitly required.

##### `libstdc++-13-dev` added to the `apt-get install` list

On `main` (before PR #809), `--features static-linking` in the Dockerfile was passed to `router`, which at that point only propagated the flag to `cudarc?/static-linking` and `intel-mkl-src?/...`. The `candle-extensions` crates were NOT yet patched to the `allow-static-linking` branch, so their C++ symbols were satisfied by the shared `libstdc++.so` already present in the CUDA devel image.

After PR #809:
- `router/Cargo.toml` gains `static-linking = [..., "text-embeddings-backend/static-linking"]`
- `backends/Cargo.toml` gains `static-linking = ["text-embeddings-backend-candle?/static-linking"]`
- `candle-flash-attn-v1`, `candle-cublaslt` etc. are now patched to the `allow-static-linking` branch, which exposes a `static-linking` feature that triggers full static C++ linkage

When `static-linking` is now truly propagated all the way to `candle-flash-attn-v1` and `cudarc 0.19`, the linker resolves `libstdc++.a`. The CUDA devel image (`nvidia/cuda:12.9.1-devel-ubuntu24.04`) provides `libstdc++.so` (the shared library) but not `libstdc++.a` (the static archive); `libstdc++-13-dev` provides the static archive.

##### `ln -sf "$(gcc --print-file-name=libstdc++.a)" "/usr/lib/$(gcc -print-multiarch)/libstdc++.a"`

On Ubuntu 24.04, `apt install libstdc++-13-dev` places `libstdc++.a` under the GCC-versioned directory:

```
/usr/lib/gcc/x86_64-linux-gnu/13/libstdc++.a
```

The GNU linker searches `/usr/lib/x86_64-linux-gnu/` (the multiarch path) by default, but **not** the GCC-versioned subdirectory, so `ld` fails with `cannot find -lstdc++` even with the dev package installed. The symlink places a canonical alias at the path the linker searches:

```sh
ln -sf "$(gcc --print-file-name=libstdc++.a)" \
       "/usr/lib/$(gcc -print-multiarch)/libstdc++.a"
#  expands to e.g.:
#  ln -sf /usr/lib/gcc/x86_64-linux-gnu/13/libstdc++.a \
#         /usr/lib/x86_64-linux-gnu/libstdc++.a
```

`gcc --print-file-name` resolves the exact path for the active GCC version, making this future-proof if the GCC major version changes.

#### Impact on unrelated functionality

| Concern | Assessment |
|---------|------------|
| **Runtime image** | All three changes affect only the `base-builder` stage. The final runtime image (`FROM nvidia/cuda:12.9.1-runtime-ubuntu24.04 AS base`) does not inherit the builder's apt packages or symlinks. Zero impact on image size, security surface, or runtime behaviour. |
| **Non-BF16 builds (F16 / F32)** | These changes are not specific to BF16. They are needed for any `--features static-linking` build once PR #809's feature-flag propagation is in place. F16 and F32 CUDA builds on this branch also require them. |
| **Existing model architectures** | No effect. Model code, inference logic, and API surface are unchanged by build-environment additions. |
| **Upstream contribution risk** | Low. All three changes are correct and necessary for building with Ubuntu 24.04 + GCC 13 + static C++ linking. The `ln -sf` command is idiomatic (used in the official GCC packaging docs and other CUDA-based Dockerfiles). The only open question for reviewers is whether `git` should instead be handled by setting `CARGO_NET_GIT_FETCH_WITH_CLI=false` (the Cargo default) and investigating which build script actually requires it — but adding the package is the safe and conventional fix. |

#### Debt item

**Clarify why `git` is needed** (non-blocking)

The `git` addition may be overly broad. It should be confirmed whether the system binary is needed by Cargo itself, by a build script in `candle-flash-attn-v1` or `cudarc`, or as a side-effect of another tool. If it turns out only Cargo's branch-tracking refresh needs it, consider whether pinning `[patch.crates-io]` entries to `rev` SHAs (see the Cargo.toml debt section above) would eliminate the dependency on the system git binary altogether, keeping the builder image smaller.

---

### backends/candle/src/layers/index_select.rs — loss of CUDA-optimised index_select

#### What changed

The pre-PR-#809 implementation delegated `index_select` on CUDA to the external `candle-index-select-cu` crate (a custom CUDA kernel authored by Michael Feil, originally published under MIT/Apache-2.0):

```rust
// main branch
#[cfg(not(feature = "cuda"))]
{ tensor.index_select(ids, dim) }
#[cfg(feature = "cuda")]
{ candle_index_select_cu::index_select(tensor, ids, dim) }
```

PR #809 removed the `candle-index-select-cu` dependency (incompatible with cudarc 0.19) and collapsed both branches to the candle native implementation:

```rust
// PR #809 onwards
tensor.index_select(ids, dim)
```

The SPDX header and attribution comment to the original author were dropped together with the dependency.

#### Why this matters

`index_select` is on the hot path of every embedding lookup (`nn::Embedding::forward` calls it once per token). For models that are embedding-lookup-bound (very small Bert-class encoders, batches with many short sequences), the dedicated CUDA kernel was meaningfully faster than candle's general-purpose `index_select` because it avoided a host round-trip for the dim/index validation and used a tighter launch grid.

For the BF16/Pplx1 work this branch targets, the regression is invisible — those models are bound by attention rather than embedding lookups. The cost is non-zero for the wider model zoo but no measurements have been taken in this branch.

#### Debt items

Once `candle-index-select-cu` (or an equivalent) is updated for cudarc 0.19, restore the conditional dispatch. Alternatively, contribute a faster CUDA path upstream into candle's `index_select` and drop the wrapper entirely.

---

### backends/src/dtype.rs — `FromStr` impl mirrors `#[cfg]` gates on the enum

#### What changed

PR #809 added a `FromStr` implementation for `DType` so that `router::run` can parse the `dtype`/`torch_dtype` field out of `config.json`:

```rust
impl FromStr for DType {
    type Err = DTypeParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let dtype = match s {
            "float32" => DType::Float32,
            #[cfg(any(feature = "python", all(feature = "candle", not(feature = "accelerate"))))]
            "float16" => DType::Float16,
            #[cfg(any(feature = "python", all(feature = "candle", any(feature = "metal", feature = "cuda"))))]
            "bfloat16" => DType::Bfloat16,
            _ => return Err(DTypeParseError),
        };
        Ok(dtype)
    }
}
```

#### Why this is a debt item

The `#[cfg]` predicates on the `FromStr` match arms must be kept in lock-step with the predicates on the enum variant declarations and with the `Display` impl. There are now three places (variant decl, `FromStr` arm, `Display` arm) that each enumerate the same feature combinations per variant. Adding a new dtype (or refining the feature gating for an existing one — e.g. extending BF16 to ROCm) requires changes to all three locations in sync; a mismatch produces either a hard compile error (best case) or a runtime asymmetry where a dtype string parses but its enum variant doesn't display (worst case).

#### Debt item

**Unify dtype gating with a macro** (non-blocking)

A small declarative macro `dtype_variants! { Float16 if cfg(...), ... }` could emit the enum, `FromStr` arm, and `Display` arm from a single source of truth, eliminating the three-way drift risk. Belongs to a follow-up cleanup commit; not a blocker.

---

## Technical debt introduced in this branch (feat/cuda-bf16-pplx1)

### backends/Cargo.toml — flash-attn features now imply cuda

#### What changed

In `backends/Cargo.toml`, the `flash-attn` and `flash-attn-v1` features were changed from:

```toml
flash-attn    = ["text-embeddings-backend-candle?/flash-attn"]
flash-attn-v1 = ["text-embeddings-backend-candle?/flash-attn-v1"]
```

to:

```toml
flash-attn    = ["cuda", "text-embeddings-backend-candle?/flash-attn"]
flash-attn-v1 = ["cuda", "text-embeddings-backend-candle?/flash-attn-v1"]
```

#### Why

The `DType::Bfloat16` variant in `backends/src/dtype.rs` is compiled only when the `cuda` (or `metal` / `python`) feature is active on the **`text-embeddings-backend` crate itself**:

```rust
#[cfg(any(
    feature = "python",
    all(feature = "candle", any(feature = "metal", feature = "cuda"))
))]
Bfloat16,
```

`DType` derives `clap::ValueEnum` (the trait that registers enum variants as valid CLI argument values) conditionally on the `clap` feature. Which variants are registered depends entirely on which are compiled in — if `Bfloat16` is absent from the enum (because its `#[cfg]` is not satisfied), clap simply never sees it, regardless of whether `ValueEnum` is derived.

The router activates the standard CUDA path via:

```toml
# router/Cargo.toml
candle-cuda        = ["candle", "text-embeddings-backend/flash-attn", "dep:cudarc"]
candle-cuda-turing = ["candle", "text-embeddings-backend/flash-attn-v1", "dep:cudarc"]
```

Before this fix, `text-embeddings-backend/flash-attn` forwarded the flag only to the sub-crate (`text-embeddings-backend-candle?/flash-attn`). The `text-embeddings-backend` crate's own `cuda` feature remained **off**, so `Bfloat16` was absent from the compiled `DType` enum and `--dtype bfloat16` produced a hard CLI error:

```
error: invalid value 'bfloat16' for '--dtype <DTYPE>'
  [possible values: float16, float32]
```

#### Why fix here rather than in router/Cargo.toml?

The bug could alternatively be fixed one level up by adding `text-embeddings-backend/cuda` explicitly to the router's `candle-cuda` feature. That would work, but is fragile: any future crate that depends on `text-embeddings-backend/flash-attn` would silently hit the same trap. Since flash-attn is a CUDA-only library (there is no non-CUDA flash attention), `flash-attn` implying `cuda` is semantically correct and defensive — the constraint lives next to the feature definition rather than scattered across callers.

After the fix, the chain `candle-cuda → text-embeddings-backend/flash-attn → cuda` ensures `Bfloat16` is always compiled in when flash attention is active. Note that `candle-cuda-volta` already referenced `text-embeddings-backend/cuda` directly, so Volta builds were never affected.

#### Impact on unrelated functionality

| Concern | Assessment |
|---------|------------|
| **F16 / F32 builds** | No change. F16 and F32 dtype handling was already correct under `candle-cuda`; the missing `cuda` implication only affected `Bfloat16`. |
| **Turing (`candle-cuda-turing`)** | `flash-attn-v1` now implies `cuda`, which was the correct intended state. F16 behaviour on Turing is unaffected. |
| **Volta (`candle-cuda-volta`)** | Already uses `text-embeddings-backend/cuda` directly; this change has no effect. |
| **CPU / Metal builds** | `flash-attn` / `flash-attn-v1` are never activated for CPU or Metal features; no impact. |
| **`--dtype bfloat16` now accepted by CLI on Turing/Volta** | Pre-Ampere hardware doesn't support BF16 natively. A runtime guard checks compute capability and errors before any GPU work if `compute_cap < 80`, so no incorrect results can occur. However, users on pre-Ampere hardware now see `bfloat16` in `--help` and receive a runtime error rather than an unknown-option error. This should be noted in the upstream PR description. |
| **Other model architectures** | All non-Pplx1 CUDA flash paths gate on `dtype == DType::F16`; they are transparent to this change. |

#### Debt item

**Pre-Ampere UX: `bfloat16` appears in `--help` but is runtime-rejected** (non-blocking)

Users on Turing/Volta see `bfloat16` as a valid option but receive an error only after the model download completes. The compute-cap check could be moved to an early validation step in `router::run` (before artifact download) or implemented as a custom clap validator. Cosmetic; does not block upstream contribution.

---

### backends/candle/tests/common.rs — Sequence::get\_pre\_tokenizers() API compatibility

#### What changed and why

The relevant lines in `load_tokenizer` acquire the pre-tokenizer list from a `Sequence` and iterate over it to patch any embedded `Metaspace` entry:

```rust
// Before PR #809 (main branch)
let pre_tokenizers = s.get_pre_tokenizers();
// …
let mut new_pre_tokenizers = Vec::with_capacity(s.get_pre_tokenizers().len());

// PR #809 rewrite
let pre_tokenizers: Vec<_> = s.clone().into_iter().collect();
// …
let mut new_pre_tokenizers = Vec::with_capacity(pre_tokenizers.len());

// This branch
let pre_tokenizers: Vec<_> = s.get_pre_tokenizers().to_vec();
// …
let mut new_pre_tokenizers = Vec::with_capacity(pre_tokenizers.len());
```

The root cause is a **breaking API change in the `tokenizers` crate** between 0.21.x and 0.22.x:

| Version | `Sequence` API for accessing pre-tokenizers |
|---------|---------------------------------------------|
| ≤ 0.21.1 | `get_pre_tokenizers(&self) -> &[PreTokenizerWrapper]` and `get_pre_tokenizers_mut` named methods; no `IntoIterator` impl |
| ≥ 0.22.0 | Named methods removed; `AsRef<[PreTokenizerWrapper]>` + `AsMut` + consuming `IntoIterator` added instead |

PR #809 was authored against (or targeting) tokenizers 0.22.x, where `get_pre_tokenizers()` no longer exists and the idiomatic access pattern changed to `s.as_ref()` (for a borrow) or `s.into_iter()` / `s.clone().into_iter()` (for consumption). The `s.clone().into_iter().collect()` rewrite targets that new API.

This workspace pins `tokenizers = "0.21.0"` in `Cargo.toml`, which resolves to **0.21.1** in `Cargo.lock`. On 0.21.1:

- `get_pre_tokenizers()` **exists** — calling it compiles cleanly.
- `IntoIterator` is **not implemented** for `Sequence` — `s.clone().into_iter()` fails with *"method `into_iter` not found for `Sequence`"*.

So PR #809's rewrite is correct for the tokenizers version it targets but breaks compilation against 0.21.1. This branch restores `get_pre_tokenizers()` and adds `.to_vec()` to produce an owned `Vec<PreTokenizerWrapper>` (necessary because the subsequent loop moves out of the collection rather than borrowing), making the code correct and self-consistent on 0.21.x.

#### Impact on unrelated functionality

| Concern | Assessment |
|---------|------------|
| **Correctness** | The three expressions (`s.get_pre_tokenizers().to_vec()`, `s.clone().into_iter().collect()`, and the original `s.get_pre_tokenizers()`) all yield the same logical content. |
| **Non-Metaspace tokenizers** | The block is guarded by `if let PreTokenizerWrapper::Sequence(s) = pre_tokenizer` and further by `if has_metaspace`. Models without a `Sequence`-wrapped `Metaspace` pre-tokenizer are unaffected. |
| **Upstream contribution** | **This is a blocker for upstreaming if tokenizers is bumped to 0.22.x before (or as part of) the PR.** `get_pre_tokenizers()` is not available in 0.22.x. The correct replacement for a borrow-then-clone pattern on 0.22.x is `s.as_ref().to_vec()`. Because `AsRef` is not implemented on 0.21.x, no single expression compiles against both versions without a `#[cfg]` or an intermediate variable. The least-disruptive resolution when bumping tokenizers is to change this branch's line to `s.as_ref().to_vec()` at the same time tokenizers is upgraded. |

#### Debt item

When the workspace `Cargo.toml` is updated to `tokenizers = "0.22"`, the line:

```rust
let pre_tokenizers: Vec<_> = s.get_pre_tokenizers().to_vec();
```

must be changed to:

```rust
let pre_tokenizers: Vec<_> = s.as_ref().to_vec();
```

(`IntoIterator` is also available in 0.22.x, but `as_ref().to_vec()` is the closest equivalent to the borrowed-then-cloned pattern.)

This is a two-line change that is straightforward but easy to miss because the compile error from `get_pre_tokenizers()` being absent will surface only on the 0.22.x build, not on the current pinned version.

---

### router/src/lib.rs — dtype resolution and gemma3 override

#### What changed and why

The dtype-resolution logic in `router::run` went through three states:

**State 1 — `main` branch (original):**
```rust
// NOTE: `gemma3_text` won't support Float16 but only Float32, given that with `candle-cuda`
// feature, the default `Dtype::Float16` this overrides that to prevent issues when running a
// `gemma3_text` model without specifying a `--dtype`
let dtype = if dtype.is_none() && config.model_type == "gemma3_text" {
    DType::Float32
} else {
    dtype.unwrap_or_default()
};
```

A per-model special-case: when the user omits `--dtype` and the model is `gemma3_text`, override the default (which is `Float16` under `candle-cuda`) with `Float32`, because the candle gemma3 implementation only supports F32.

**State 2 — PR #809 (applied in commit `8fdf24b`):**
```rust
let dtype = dtype.unwrap_or_else(|| {
    config
        .dtype
        .as_deref()
        .and_then(|s| DType::from_str(s).ok())
        .unwrap_or_default()
});

#[cfg(all(feature = "candle", feature = "metal"))]
if dtype == DType::Bfloat16 {
    tracing::warn!("`--dtype bfloat16` support is still experimental on Metal.");
}
```

PR #809 replaced the hardcoded gemma3 special-case with a general-purpose config.json dtype fallback. The new `ModelConfig::dtype` field (deserialized via `#[serde(alias = "torch_dtype")]`) lets any model advertise its preferred precision through `config.json`, and a newly added `DType::FromStr` impl converts the string to the enum. The `unwrap_or_default()` tail retains the existing default (Float16 under `candle-cuda`) when config.json has no `dtype`/`torch_dtype` field.

This design is preferable in principle: it is data-driven and avoids accumulating per-model `if model_type == "..."` branches. However, it silently broke the gemma3 zero-config path (see below).

**State 3 — this branch:**
```rust
let dtype = if dtype.is_none() && config.model_type == "gemma3_text" {
    DType::Float32
} else {
    dtype.unwrap_or_else(|| {
        config
            .dtype
            .as_deref()
            .and_then(|s| DType::from_str(s).ok())
            .unwrap_or_default()
    })
};
```

The gemma3 special-case is re-introduced as a short-circuit *before* step 2, while keeping PR #809's config.json fallback for all other models. This is what makes `perplexity-ai/pplx-embed-v1-0.6b` work without `--dtype`: its `config.json` declares `"torch_dtype": "bfloat16"`, which step 2 now picks up, sparing the user from having to pass `--dtype bfloat16` explicitly.

#### Why the gemma3 override was necessary

Gemma3 models on HuggingFace publish `"torch_dtype": "bfloat16"` in their `config.json` (BF16 is their intended PyTorch inference dtype). The candle backend currently rejects any non-F32 dtype for gemma3:

```rust
// backends/candle/src/lib.rs — both Cpu/Metal and Cuda arms
if dtype != DType::F32 {
    return Err(BackendError::Start(
        "Gemma3 is only supported in fp32 precision".to_string(),
    ));
}
```

This is a *temporary* constraint, not an inherent precision requirement. The TODO annotation on both guards reads `"Enable Flash Attention with BF16 once supported"`, targeting **BF16 specifically** as the future flash dtype — not F16 (which would likely produce incorrect results for the same numerical reasons as Pplx1's quantization head). The F32-only state is therefore a placeholder pending the BF16 flash implementation.

Without the router-level override:
- PR #809's config.json fallback reads `"torch_dtype": "bfloat16"` → selects `DType::Bfloat16`.
- The backend rejects BF16 with a confusing "Gemma3 is only supported in fp32 precision" error — *after* the model weights have already been downloaded and the backend process started.
- There is no user-visible indication that `--dtype float32` would fix it.

With the router-level override, the dtype is forced to F32 before backend initialisation, and the user never sees an error.

#### Was PR #809's removal of the check intentional?

The most likely explanation is **intentional deferral** in anticipation of a BF16 implementation, not an oversight. The Pplx1 model is a direct parallel: before this branch it also carried `if dtype != DType::F32 { bail!("Pplx1 is only supported in fp32 precision") }` and the same `TODO(alvarobartt): Enable Flash Attention with BF16 once supported on CUDA/Metal` annotation — identical in form to the gemma3 guards still present in `backends/candle/src/lib.rs`. This branch then added real BF16 support for Pplx1 and removed that guard. PR #809 most likely assumed the same trajectory for gemma3: once BF16 is implemented, the backend will accept it and the config.json fallback will work correctly without any special-case routing.

The practical problem is timing: PR #809 removed the safety guard before BF16 was implemented, leaving a window where the config.json fallback silently selects BF16 and the backend rejects it with a confusing error. Our re-introduction of the override closes that window for the current state of the candle gemma3 implementation.

Regardless of intent, re-introducing the override is the correct choice while the constraint holds.

#### Impact on unrelated functionality

| Concern | Assessment |
|---------|------------|
| **gemma3 zero-config** | Restored: `tei --model-id google/gemma-3-...` without `--dtype` now correctly selects F32 and starts cleanly. |
| **Pplx1 zero-config** | New: `tei --model-id perplexity-ai/pplx-embed-v1-0.6b` without `--dtype` now auto-selects BF16 from `config.json`'s `"torch_dtype": "bfloat16"`. Without PR #809's config.json fallback (step 2), this path would silently fall back to Float16, producing wrong embeddings. |
| **Models with `torch_dtype` in config.json** | PR #809's step 2 now benefits any model that advertises its preferred dtype. For example, Qwen3 models that publish `"torch_dtype": "float16"` no longer require an explicit `--dtype`. This is a net improvement, not a regression. |
| **All other models** | Unaffected. The `gemma3_text` branch is unreachable for any other `model_type`. Models with no `dtype`/`torch_dtype` in their `config.json` continue to use `DType::default()`. |

#### Debt item

**Remove the `gemma3_text` override once gemma3 BF16 is implemented** (non-blocking)

The re-introduced check is a temporary safety guard, not a permanent design. The intended lifecycle mirrors what this branch did for Pplx1:

1. Gemma3 BF16 is implemented in `backends/candle/src/models/gemma3.rs` (the `TODO(alvarobartt)` in `lib.rs` marks exactly this).
2. The `if dtype != DType::F32` rejection in `lib.rs` is replaced with flash/BF16 routing (as done for Pplx1 and Qwen3).
3. The `if dtype.is_none() && config.model_type == "gemma3_text"` block in `router/src/lib.rs` is deleted, letting PR #809's config.json fallback resolve `"torch_dtype": "bfloat16"` correctly.

Until step 1 happens, the override must stay. When filing the "implement gemma3 BF16" ticket, add removal of this router check to its acceptance criteria.

### Pre-merge follow-ups

This section contains a non-exhaustive list of cleanup and follow-up tasks to be completed before merging the `feat/cuda-bf16-pplx1` branch (apart from the blockers mentioned in other sections).

1. **Fork-internal docs retained** — `BUILD-CUDA-BF16.md`, `ATTRIBUTION.md`, and this `DEBT.md` are kept on the branch for fork-internal use but **must not** be included in any upstream PR; strip them when preparing the upstream patch.
2. **Docs**: `README.md` and `docs/source/en/supported_models.md` are missing: BF16 CUDA support note (SM 80+ / Ampere required), Pplx1 model entry, and the caveat that F16 is rejected for Pplx1.
3. **Add CI coverage for `--dtype bfloat16`** — no GitHub Actions job exercises the new code paths.

### Post-merge follow-up tickets

1. **Audit non-Pplx1 CUDA flash models for BF16 eligibility** — Bert, DistilBert, GTE, Mistral, ModernBert, NomicBert, Qwen2 all gate their flash paths on `dtype == F16` only. Determine whether BF16 is numerically safe for each and widen the conditions where it is.
2. **Add BF16 CI test matrix** — a GitHub Actions job that builds and runs tests with `--dtype bfloat16` for Pplx1 and Qwen3 on SM ≥ 80 hardware.
3. **Refactor `use_flash_attn()` call sites** — replace scattered `if dtype == F16 && use_flash_attn(...)` checks with a single helper that accepts the dtype and supported flash-attn versions as arguments.
4. **Verify OpenAPI schema includes `bfloat16`** — confirm `docs/openapi.json` and the auto-generated schema expose `bfloat16` as a valid `--dtype` enum value.
