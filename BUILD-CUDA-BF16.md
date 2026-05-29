# Building the `tei-bf16:<git-sha>` CUDA BF16 Image

This document explains, step by step, how to build a
Docker image that is based on the `feat/cuda-bf16-pplx1` branch of this fork. The
image is a CUDA build of the
[text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference)
router with bfloat16 (BF16) support enabled and a `FlashPplx1Model` wrapper
that runs `perplexity-ai/pplx-embed-v1-0.6b` natively on NVIDIA GPUs with
flash-attention.

The image that is built this way is fully functional, as verified by internal tests against an internal dataset - evaluation results are consistent with the bf16 implementation in vLLM. Further, benchmarks show that efficiency is significantly improved compared to float32.

Open technical debt is documented in [DEBT.md](DEBT.md) and includes cleanup and follow-up tasks that should be completed before merging this branch upstream. Attribution is documented in [ATTRIBUTION.md](ATTRIBUTION.md) and includes PRs that this branch is based on.

---

## Table of Contents

1. [Scope of this document](#1-scope-of-this-document)
2. [Pre-conditions](#2-pre-conditions)
    1. [Hardware](#21-hardware)
    2. [Operating system](#22-operating-system)
    3. [Software on the host](#23-software-on-the-host)
    4. [Repository access](#24-repository-access)
3. [One-time host setup](#3-one-time-host-setup)
    1. [NVIDIA driver](#31-nvidia-driver)
    2. [NVIDIA Container Toolkit](#32-nvidia-container-toolkit-only-needed-for-running--testing)
    3. [Docker BuildKit](#33-docker-buildkit)
    4. [Sanity-check the host](#34-sanity-check-the-host)
4. [Clone the repository and switch to the branch](#4-clone-the-repository-and-switch-to-the-branch)
5. [Determine the correct `CUDA_COMPUTE_CAP`](#5-determine-the-correct-cuda_compute_cap)
6. [Build the `tei-bf16:<git-sha>` image](#6-build-the-tei-bf16git-sha-image)
    1. [Recommended (with build cache)](#61-recommended-with-build-cache)
    2. [Alternative one-liner](#62-alternative-one-liner)
    3. [Expected build artifacts](#63-expected-build-artifacts)
7. [Smoke-test the image](#7-smoke-test-the-image)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Scope of this document

The procedure below produces a single OCI image tagged `tei-bf16:<git-sha>` that:

- Is based on `nvidia/cuda:12.9.1-runtime-ubuntu24.04`.
- Bundles the `text-embeddings-router` binary built with the `candle-cuda` +
  `flash-attn` + `static-linking` Cargo features.
- Accepts `--dtype bfloat16` on CUDA hardware with compute capability ≥ 8.0.
- Routes `perplexity-ai/pplx-embed-v1-0.6b` through a new `FlashPplx1Model`
  (BF16-only flash path with the model's INT8 quantization head).

Building the image itself only requires Docker and network access. Running the
image (or the test suite) additionally requires an NVIDIA GPU on the host.

---

## 2. Pre-conditions

### 2.1 Hardware

| Requirement | Build host | Runtime host |
|---|---|---|
| CPU | x86_64 or arm64 | x86_64 or arm64 (same arch as the image) |
| RAM | ≥ 16 GB recommended (Rust + CUDA compile is heavy) | ≥ 8 GB |
| Disk | ≥ 30 GB free for the build cache and intermediate layers | ≥ 5 GB for the runtime image + model |
| GPU | Not strictly required for building | **Required.** NVIDIA GPU with **compute capability ≥ 8.0** (Ampere or newer: A100, A10/A30/A40, RTX 30xx, RTX 40xx, L4, L40/L40S, H100, etc.) |
| GPU driver | n/a | NVIDIA driver supporting CUDA 12.9 (≥ 550) |

> ⚠️ BF16 + flash-attention on CUDA only works on **SM ≥ 80**. Turing (SM 75)
> and Volta (SM 70) cannot run the BF16 path. The image's runtime guard will
> bail with a clear error if started on unsupported hardware.

### 2.2 Operating system

- **Build host**: Any Linux distribution that runs Docker (Ubuntu 22.04/24.04,
  Debian 12, RHEL 9, etc.). macOS / Windows + Docker Desktop also work for the
  build step, but you cannot run the resulting image there without a Linux VM
  + GPU passthrough.
- **Runtime host**: Linux with the NVIDIA driver and the NVIDIA Container
  Toolkit installed (see [§3.2](#32-nvidia-container-toolkit-only-needed-for-running--testing)).

### 2.3 Software on the host

| Tool | Version | Required for | Install hint |
|---|---|---|---|
| Docker Engine | ≥ 24.0 (with BuildKit, default in modern Docker) | Building & running | Assumed already installed |
| `git` | any recent | Cloning the repo | `sudo apt-get install -y git` |
| `curl` | any recent | Smoke-testing the running server | `sudo apt-get install -y curl` |
| NVIDIA driver | ≥ 550 (for CUDA 12.9) | Running only | See [§3.1](#31-nvidia-driver) |
| NVIDIA Container Toolkit | ≥ 1.14 | Running only | See [§3.2](#32-nvidia-container-toolkit-only-needed-for-running--testing) |

You do **not** need a host Rust toolchain, CUDA toolkit, protobuf compiler,
sccache, or any Python — everything that compiles the binary lives inside the
multi-stage Dockerfile.

### 2.4 Repository access

The branch `feat/cuda-bf16-pplx1` lives in this fork. Make sure the user
running `docker build` can read the working tree on disk (`docker build .`
sends the context to the daemon).

---

## 3. One-time host setup

### 3.1 NVIDIA driver

Only required if you plan to **run** the image on this host.

On Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y nvidia-driver-550-server
sudo reboot
```

After reboot, verify:

```bash
nvidia-smi
```

The reported "CUDA Version" must be **≥ 12.9** for the runtime image to load
cleanly.

### 3.2 NVIDIA Container Toolkit (only needed for running / testing)

```bash
# Add the NVIDIA package repository
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install and configure Docker to use it
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify GPU is visible from a container:

```bash
docker run --rm --gpus=all nvidia/cuda:12.9.1-base-ubuntu24.04 nvidia-smi
```

### 3.3 Docker BuildKit

BuildKit is required for the cached, secret-aware multi-stage build. It has
been the default backend since Docker 23.0 and is always active on modern
installs — no configuration is needed.

> **Note:** `docker info | grep -i buildkit` returns **zero lines** on Docker
> 23+ and is not a reliable indicator. BuildKit being active is confirmed by:

```bash
docker buildx version
# Example output: github.com/docker/buildx v0.x.y ...
```

If you are on a Docker version older than 23.0, enable BuildKit explicitly
before building:

```bash
export DOCKER_BUILDKIT=1
```

### 3.4 Sanity-check the host

```bash
docker --version          # ≥ 24.x
docker buildx version     # any modern version
git --version             # any
nvidia-smi                # only if you plan to run the image
```

---

## 4. Clone the repository and switch to the branch

```bash
git clone <your-fork-url> text-embeddings-inference
cd text-embeddings-inference
git checkout feat/cuda-bf16-pplx1
```

Confirm you are on the right branch (you should see the four CUDA-BF16
commits on top of upstream `main`):

```bash
git log --oneline main..feat/cuda-bf16-pplx1
# 52f2c66 fix(cuda): restrict FlashPplx1 to BF16 and restore gemma3 fp32 default
# 558a840 test(cuda): add fp16/bf16 flash snapshots for Pplx1/Qwen3 ...
# dd5bbf5 fix(cuda): propagate cuda feature via flash-attn ...
# 8fdf24b feat(cuda): enable bf16 + FlashPplx1Model on CUDA
```

---

## 5. Determine the correct `CUDA_COMPUTE_CAP`

The build embeds CUDA code only for the target compute capability you pass in,
which keeps the image small and the build fast.

Pick the value matching the GPU(s) you will run on:

| GPU family | Example cards | `CUDA_COMPUTE_CAP` |
|---|---|---|
| Ampere (data-center) | A100 | `80` |
| Ampere (consumer / pro) | A10, A30, A40, RTX 30xx | `86` |
| Ada Lovelace | L4, L40, L40S, RTX 40xx | `89` |
| Hopper | H100, H200 | `90` |
| Blackwell (data-center) | B100/B200 | `100` |
| Blackwell (consumer) | RTX 50xx | `120` / `121` |

Auto-detect on the runtime host:

```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 \
  | tr -d '.'
# Example output: 89
```

> Only values `≥ 80` are valid for this branch (the BF16 + flash path requires
> Ampere or newer). Turing/Volta builds (`CUDA_COMPUTE_CAP < 80`) take a
> different Cargo feature path and are not exercised here.

---

## 6. Build the `tei-bf16:<git-sha>` image

The build uses the existing [Dockerfile-cuda](Dockerfile-cuda) (a multi-stage,
`cargo-chef` + `sccache`-cached build that ends on
`nvidia/cuda:12.9.1-runtime-ubuntu24.04`).

### 6.1 Recommended (with build cache)

From the repository root:

```bash
TAG="tei-bf16:$(git rev-parse --short HEAD)"

docker build \
  --file Dockerfile-cuda \
  --tag "${TAG}" \
  --build-arg CUDA_COMPUTE_CAP=89 \
  --build-arg GIT_SHA="$(git rev-parse HEAD)" \
  --build-arg DOCKER_LABEL="${TAG}" \
  .
```

Replace `89` with the value chosen in [§5](#5-determine-the-correct-cuda_compute_cap).

What this does:

1. Builds a base image with Rust 1.92 (pinned by
   [rust-toolchain.toml](rust-toolchain.toml)), `sccache 0.10.0` and
   `cargo-chef 0.1.73`.
2. Generates a `cargo-chef` recipe for the workspace and pre-builds
   dependencies with the `candle-cuda` + `static-linking` features.
3. Compiles the `text-embeddings-router` binary with `-F candle-cuda
   -F static-linking -F http --no-default-features`.
4. Copies the binary into a clean runtime image and sets the
   [cuda-entrypoint.sh](cuda-entrypoint.sh) as the container entrypoint.

> The first build is slow (heavy CUDA + Rust compile). Subsequent builds reuse
> the `cargo-chef` layer and only recompile what changed.

### 6.2 Alternative one-liner

If you do not care about cache hits and just want the image:

```bash
docker build -t "tei-bf16:$(git rev-parse --short HEAD)" \
  --build-arg CUDA_COMPUTE_CAP=89 \
  -f Dockerfile-cuda .
```

### 6.3 Expected build artifacts

After a successful build:

```bash
docker images tei-bf16
# REPOSITORY   TAG        IMAGE ID       CREATED          SIZE
# tei-bf16     52f2c66    <id>           <when>           ~3.5 GB
```

Verify the binary is in place and reports BF16 in its `--help`.

The container's entrypoint (`cuda-entrypoint.sh`) checks for `nvidia-smi`
before executing the router, so `--gpus` must be passed even for a help query.
Additionally, the router binary links against `libcuda.so.1` at load time,
which is only injected by the NVIDIA Container Toolkit when `--gpus` is
present. Omitting it produces `error while loading shared libraries:
libcuda.so.1`.

Use `--entrypoint` to bypass the entrypoint script and pass `--gpus` to
satisfy the dynamic linker:

```bash
docker run --rm --gpus=all \
  --entrypoint text-embeddings-router \
  "tei-bf16:$(git rev-parse --short HEAD)" --help 2>&1 | grep -A8 -- '--dtype'
# --dtype <DTYPE>
#     The dtype to be forced upon the model, otherwise the value set in
#     `dtype` (or `torch_dtype` as fallback) in the `config.json` file
#     is used. Note that `bfloat16` is not supported on CPU, neither for
#     Turing on CUDA, but only from Ampere onwards
#
#     [env: DTYPE=]
#     [possible values: float16, float32, bfloat16]
```

---

## 7. Smoke-test the image

This requires an NVIDIA GPU with SM ≥ 80 on the host (see
[§2.1](#21-hardware) and [§3](#3-one-time-host-setup)).

Edit the `MODEL_ID` and `volumes` entries in following `docker-compose.yml` to point at a model
directory on your host:


```yaml
services:
  tei-embedder:
    image: tei-bf16:<git-sha>  # replace with the tag printed by the build step, e.g. tei-bf16:52f2c66
    shm_size: 1g
    restart: unless-stopped
    ports:
      - "12345:80"
    environment:
      MODEL_ID: [PATH TO YOUR MODEL IN THE CONTAINER]
      DTYPE: bfloat16
      MAX_BATCH_TOKENS: 32000
      MAX_CLIENT_BATCH_SIZE: 32
    deploy:
      resources:
        reservations:
          devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]
    volumes:
      - [PATH TO YOUR MODEL ON THE HOST]:/models
```

then:

```bash
docker compose up
```

You should see:

```
Starting model backend
Starting FlashPplx1 model on Cuda(CudaDevice(...))
Ready
```

Then query it:

```bash
curl -s http://localhost:20541/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "tei",
        "input": ["What is Deep Learning?", "Hello, world"]
      }' \
  | head -c 400
```

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `cuda compute cap XX is not supported` during build | `CUDA_COMPUTE_CAP` is `< 75`, between `90` and `100` exclusive, or otherwise unhandled by `Dockerfile-cuda` | Use one of: `80`, `86`, `89`, `90`, `100`, `120`, `121`. SM `< 80` is not supported by this BF16 branch. |
| `error: invalid value 'bfloat16' for '--dtype'` when starting the container | Image was not built from this branch (or the `flash-attn ⇒ cuda` feature implication was lost) | Re-build from the `feat/cuda-bf16-pplx1` tip; verify `backends/Cargo.toml` line `flash-attn = ["cuda", ...]`. |
| `BFloat16 requires CUDA compute capability >= 8.0 ...` at startup | Runtime GPU is Turing/Volta | Use Ampere or newer hardware, or run with `--dtype float32` (Pplx1 then takes the non-flash path). |
| `FlashPplx1 requires DType::BF16 ...` | Started Pplx1 with `--dtype float16` | Use `--dtype bfloat16` (fp16 is intentionally rejected for Pplx1 — its INT8 quantization head loses precision in fp16). |
| `docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]` | NVIDIA Container Toolkit not installed/configured | Repeat [§3.2](#32-nvidia-container-toolkit-only-needed-for-running--testing). |
| Build is very slow / OOM-killed | Insufficient RAM, or too many parallel rustc jobs | `--build-arg CARGO_BUILD_JOBS=4 --build-arg RAYON_NUM_THREADS=4`. |
| Port `PORT` already in use | Another service occupies the host port | Edit the `ports:` mapping in `docker-compose.yml`. |
