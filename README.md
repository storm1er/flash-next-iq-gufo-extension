# flash_next_iq — Qwen3.8 Flash-Next IQ-quant extension for Gufo

Community (out-of-tree) source extension that lets Gufo serve the **UD-IQ4_XS** quantization family of
`Qwen3.8-Flash-Next` GGUF checkpoints on AMD Strix Halo (`gfx1151`): routed experts in **IQ3_S / IQ4_XS**
(gate/up) and **IQ4_NL** (down), logits head in **Q6_K**, with MTP speculative decoding and continuation
(prompt-cache) support. An independent repository compiled against a pinned upstream Gufo with
`-DGUFO_EXTENSION_DIRS` — Gufo's community model extension shape
(https://github.com/gufo-org/gufo/blob/main/docs/model-extensions/DISTRIBUTION.md), not a fork. The probe only claims `qwen4exp` checkpoints whose routed experts are IQ-quantized, so plain
Q4_K_XL files still resolve to the built-in provider.

## Usage

The prebuilt container is all you need for a UD-IQ4_XS-family `Qwen3.8-Flash-Next` checkpoint on `gfx1151`.

### Prebuilt container

```sh
docker run --rm --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render --ulimit memlock=-1 \
  -v /opt/rocm:/opt/rocm:ro \
  -v /path/to/model-snapshot:/models:ro -p 8080:8080 \
  ghcr.io/storm1er/flash-next-iq-gufo-extension:latest \
  gufo serve --host 0.0.0.0 --port 8080 llm \
    --model /models/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf \
    --speculative mtp --mtp-model /models/MTP/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf \
    --mmproj /models/mmproj-BF16.gguf \
    --sessions 2 --context 262144 \
    --served-model-name Qwen3.8-Flash-Next
```
ROCm userspace is **not** baked in: the host `/opt/rocm` mount keeps the image matched to the host KFD
driver (`LD_LIBRARY_PATH=/opt/rocm/lib` is preset); if the host `/opt/rocm` is only symlinks, bind the
versioned directory it points into instead, e.g. `-v /opt/rocm/core-10.0:/opt/rocm:ro`. The image ships
`video` (44)/`render` (992) groups so `--group-add` names resolve; the same keys work verbatim under
`devices`/`group_add`/`ulimits` in compose. The ROCm-baked variant (Development) is this command without the volume.

### Native binaries

```sh
gufo serve --model /models/UD-IQ4_XS --mtp-model /models/next-nvfp4-mtp.gguf \
  --port 8080 --context 262144
```
`--mtp-model` is a core CLI option, unchanged from the built-in provider; `--sessions`, `--context`
and `--cache-ram-bytes` work as usual, and `gufo prompt` / `gufo bench` support MTP too (fork-derived test matrix).

### Caveat: stock released Gufo (0.30 series) cannot claim IQ checkpoints

The built-in `qwen38_flash_next` probe accepts any `qwen4exp` checkpoint and registers before out-of-tree
extensions, so an IQ checkpoint lands on the built-in and fails with "unsupported format IQ3_S". The fix,
https://github.com/gufo-org/gufo/pull/1041 , is merged on `main` (2026-11-06) but not yet in a release; until
then use a Gufo built from `main` (the published images use a post-#1041 revision) or a selection without `qwen38_flash_next`.

Requirements: Linux x86-64 with AMD `gfx1151`; Gufo ≥ 0.30 (HIP `gfx1151` support); a ROCm ≥ 6.4
toolchain with hipcc to build (tested with ROCm 7.15 / clang 23).

#### Known limitations

- IQ gate/up use two separate RoutedF16 GEMMs instead of the paired Q4_K/Q5_K kernel (0.30
  reworked it before porting): expert-layer prefill is slightly slower than the 0.12 fork's paired
  route, decode unaffected; wide prefill builds two routed tile maps (narrow dual, 128-row down) via `Executor::RouteHints`.
- IQ4_NL wide (128-row) down tiles use the format-generic `RoutedF16GEMMKernel`, not the specialized byte-block kernel.
- The vision encoder is inherited unchanged from 0.30.0, validated only as far as loading
  (`--mmproj`); image requests follow the built-in provider's path.

## Development

### Build from source

```sh
# In a Gufo checkout (e.g. 0.30.0), without Nix. On a host without the -dev
# packages, point PKG_CONFIG_PATH at a local apt-extracted dev prefix first.
PKG_CONFIG_PATH=/path/to/devdeps/usr/lib/x86_64-linux-gnu/pkgconfig \
cmake --preset release \
  -DGUFO_EXTENSION_DIRS=/path/to/flash-next-iq-gufo-extension \
  -DGUFO_EXTENSIONS=flash_next_iq
cmake --build --preset release --target gufo --parallel 4
```
`gufo diagnose --section extensions --json` then lists `flash_next_iq` with its source directory and
provenance. For Nix, use the extension template flake shape (Gufo as a flake input, this repo via
`GUFO_EXTENSION_DIRS` / `gufo.packages...override { extensions = [ ... ]; }`) with the same selection.
There are no link-time selection restrictions: the vendored family lives in extension-private namespaces
(`gufo::hip_iq`, `gufo::models::qwen_iq`, `gufo::tokenization_iq`, `fniq_*` kernels), so it links with the
shipped `qwen38_flash_next`/Qwen models in one binary (verified: zero duplicate symbols; IQ checkpoints
served through this provider in an all-extensions build).

### Container images

Both variants are ubuntu 24.04 with `gufo` built with this extension:
- Slim (`docker/runtime.Dockerfile`): copy any `gufo` build containing this extension to
  `dist/gufo`, then `docker build -f docker/runtime.Dockerfile .` The bind-mounted host ROCm pins
  userspace exactly to the host driver; prefer this after ROCm upgrades.
- ROCm-baked (`docker/runtime-rocm.Dockerfile`): `tools/rocm-subset.sh /path/to/gufo /tmp/rocm-image`
  collects the binary's transitive `/opt/rocm` closure (HIP, hipBLAS/Lt, rocBLAS + gfx1151 tuning data)
  into `opt-rocm`, then `docker build -f docker/runtime-rocm.Dockerfile -t <tag> /tmp/rocm-image`. No
  volume needed, ~600 MB bigger, runs on any host whose KFD driver is at least that old; portable deploys.

CI (`.github/workflows/images.yml`) rebuilds both variants on every push to `main` against the
revision in `.upstream-rev` and publishes `latest`, `latest-rocm` and immutable `sha-<short>` /
`sha-<short>-rocm` tags; semver tags (e.g. `0.30.0-iq.1`) are manual stable pins.

### On-GPU operator tests

```sh
cmake --preset release ... -DGUFO_FLASH_NEXT_IQ_TESTS=ON
cmake --build --preset release --target flash_next_iq_iq_ops_test flash_next_iq_routed_wmma_ops_test
LD_LIBRARY_PATH=/opt/rocm/lib ./build/release/extensions/build/flash_next_iq/build/<test>
```
Measured on a gfx1151 machine (Nov 2026), production server still resident (both tests allocate only small
tensors): `iq_ops_test` passes all IQ3_S/IQ4_XS/IQ4_NL vector/MMQ and Q6_K head checks (worst relative error
≈ 4.6e-3 against the CPU dequantizers); `routed_wmma_ops_test` passes every IQ tile-row configuration with
F16-vs-F64 error ≤ 0.53 % of output scale (threshold 2e-3). End-to-end, real UD-IQ4_XS checkpoint + Q4_K_M MTP
draft + BF16 mmproj at 262K context, 2 sessions (cold cache): load 79 s / 80.7 GiB device, decode 51–61 tok/s
with MTP (draft acceptance 74–85 %), coherent reasoning output, working continuation checkpoints (~269 MiB)
with shared-prefix reuse; `--mtp-model` and the component cache behave exactly like the built-in provider.
The November 9, 2026 re-release (four-shard `UD-IQ4_XS` with IQ3_XXS/IQ2_S/Q5_0/Q2_0 experts, Q6_K attention
and the `Q4_K_M` MTP sidecar) verifies at 32K context, one session: load 11 s warm, prefill 415 tok/s on a
2.1K-token prompt, decode 30–39 tok/s with MTP (draft acceptance 45–57 %), coherent reasoning output.

### Vendoring

`src/qwen/` and `src/testing/compare/` are copies of Gufo's private `src/families/qwen` and
`src/testing/compare` (the sandbox cannot link those targets), **verbatim upstream after a mechanical
transform**: includes rewritten to the extension API headers and family-owned symbols renamed into private
namespaces (`gufo::hip`/`tokenization`/`speculative`/`testing` → `*_iq`, `gufo::models::qwen` →
`gufo::models::qwen_iq`, `qfn_` → `fniq_`). Core services the family calls into (dispatch telemetry, the
hipBLASLt plan database, snapshot transfers, `gufo::speculative::PromptLookup`) stay in core namespaces.
`tools/vendor.sh` owns the rules: `check <gufo-src>` fails on drift, `revendor <gufo-src> <rev>` re-emits
the trees plus delta, `regenerate <gufo-src>` re-records it. `.upstream-rev` pins the upstream revision;
`tools/iq-family.patch` carries hand edits over upstream+transform and is currently empty (all IQ code
lives in this extension's own model/device layer). Syncing to a new Gufo release is a pin bump plus
`revendor`; upstream changes to the core services show up as ordinary compile errors, never silent misbinds.

## Provenance

- Model sources: Gufo 0.30.0 `src/models/qwen38_flash_next` (MIT), with the IQ3_S/IQ4_XS/IQ4_NL +
  Q6_K work re-applied from the `gufo-iq4_xs` fork
  (https://github.com/johandevlabs/gufo-iq4_xs/compare/b13c2fa6b6abf71266529f2c878e07704f28dfd7...912679bccb9e9b91c6ae4000fc017b83a2c630a7).
- `Executor::RouteHints` "route staging drain": expert counts are awaited with a full stream drain rather
  than the download event, adopted from the mikealanni/gufo fork (https://github.com/mikealanni/gufo),
  which documents this driver defect (events completing while pinned-staging copies are still queued).
- `src/qwen/`: vendored copy of Gufo's private `src/families/qwen/*` (MIT); the external-extension sandbox
  does not link `gufo_core`/`gufo_qwen_*`, so the family layer is rebuilt as extension-private targets.
- `src/kernels/rocm/mmq/`: GGML-derived MMQ/MoE kernels (MIT, see the `VENDOR.md` inside).
- Built and validated against upstream Gufo `2c97b72d` (v0.30.0+9, pinned in `flake.nix`); vendored
  trees tracked at `b7cf081d` in `.upstream-rev` (post-#1041 main, used by the published image).
- Namespace `gufo::models::flash_next_iq`; provider id `flash-next-iq`, artifact key `gguf:qwen4exp-iq`.

License: MIT (same as Gufo); third-party notices in `LICENSE` and `src/kernels/rocm/mmq/VENDOR.md`.
