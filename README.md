# flash_next_iq — Qwen3.8 Flash-Next IQ-quant extension for Gufo

Community (out-of-tree) source extension that lets Gufo serve the **UD-IQ4_XS** quantization family of
`Qwen3.8-Flash-Next` GGUF checkpoints on AMD Strix Halo (`gfx1151`), with MTP speculative decoding and
continuation (prompt-cache) support:

- the **Unsloth Dynamic 3.1** re-release (2026-11, the current `UD-IQ4_XS`, four shards): routed gate/up in
  **IQ3_XXS / IQ4_XS / IQ2_S**, down in **Q5_0 / IQ4_NL / Q2_0**, **IQ4_XS** token embeddings, **Q6_K**
  attention, shared experts and logits head;
- the 2026-10 export (three shards): gate/up in **IQ3_S / IQ4_XS**, down in **IQ4_NL / Q8_0**, Q8_0 trunk.

An independent repository compiled against a pinned upstream Gufo with `-DGUFO_EXTENSION_DIRS` — Gufo's
community model extension shape (https://github.com/gufo-org/gufo/blob/main/docs/model-extensions/DISTRIBUTION.md),
not a fork. The probe claims `qwen4exp` checkpoints with at least one IQ/narrow-bit expert tensor whose
other expert tensors it can also run; checkpoints whose experts are all K-quant/Q8_0 (e.g. Q4_K_XL) still
resolve to the built-in provider.

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
    --speculative mtp --mtp-model /models/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf \
    --mmproj /models/mmproj-BF16.gguf \
    --sessions 2 --context 262144 \
    --served-model-name Qwen3.8-Flash-Next
```
Use the `Q8_0` MTP drafter of the same snapshot: it accepts noticeably more draft tokens than the
`Q4_K_M` one (both work; see the measurements below).
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
- IQ4_NL, Q5_0 and Q2_0 wide (128-row) down tiles use the format-generic `RoutedF16GEMMKernel`, not the
  specialized byte-block kernel.
- The `Q4_K_M` MTP drafter's Q4_K/Q5_0 projections run prefill on the tiled MMQ GEMM (dp4a), not the F16
  matrix cores; the trunk is unaffected.
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
Both tests allocate only small tensors, so they run beside a loaded server. `iq_ops_test` covers every
expert format's vector and MMQ routes (IQ3_S/IQ3_XXS/IQ2_S/IQ4_XS/IQ4_NL/Q5_0/Q2_0), the Q6_K head, the
dense Q6_K/Q5_0 MMQ GEMM, the dense Q6_K F16 matrix-core GEMM at the trunk's shapes (relative error
≤ 3e-4 against F64) and the token-embedding gather (IQ4_XS and Q8_0 rows equal to the CPU dequantizer).
`routed_wmma_ops_test` checks the F16 routed route of every format on the 16-, 48- and 128-row tiles against
the MMQ tier and F64 (F16-vs-F64 error ≤ 2e-3 of output scale; about ten times closer to F64 than MMQ).

#### End-to-end (gfx1151, 128 GB, Nov 12 2026)

Two sessions, 262,144-token context, `--cache-ram-bytes 12884901888`, BF16 mmproj, default sampling; one
workload replaying an agent session: short prompts, a cold 174K-token prompt, follow-up turns on the live
checkpoint, a second 33.5K-token conversation, a return to the first, then both sessions decoding at once.

| | 2026-10 UD-IQ4_XS + Q8_0 MTP (`gufo-runtime:edge`, Gufo PR #570 head `71bea96`) | 2026-11 UD 3.1 UD-IQ4_XS + Q8_0 MTP (this extension) |
|---|---|---|
| load / device memory | 12 s / 80.7 GiB | 11 s / 80.8 GiB |
| cold prefill, 174K tokens | 1,515 tok/s (TTFT 115.0 s) | 1,560 tok/s (TTFT 111.6 s) |
| cold prefill, 33.5K tokens | 1,511 tok/s | 1,627 tok/s |
| decode at 174K depth | 41–44 tok/s, acceptance 57–68 % | 46–55 tok/s, acceptance 66–81 % |
| follow-up TTFT (checkpoint reuse) | 0.24–0.41 s | 0.26–0.35 s |
| both sessions decoding | 50.8 tok/s aggregate | 55.0 tok/s aggregate |
| short prompt decode | 63 tok/s | 68–71 tok/s |

`gufo bench -p 16384`: 1,801 tok/s on UD 3.1, 1,800 on the 2026-10 export (same binary).
`--validate-prefill 32`: cosine 0.9974 / RMSE 0.20 on UD 3.1 (0.9943 / 0.32 on the 2026-10 export).

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
