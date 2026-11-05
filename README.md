# flash_next_iq — Qwen3.8 Flash-Next IQ-quant extension for Gufo

Community (out-of-tree) source extension that lets Gufo serve the
**UD-IQ4_XS** quantization family of `Qwen3.8-Flash-Next` GGUF checkpoints on
AMD Strix Halo (`gfx1151`): routed experts in **IQ3_S / IQ4_XS** (gate/up) and
**IQ4_NL** (down), logits head in **Q6_K**, with MTP speculative decoding and
continuation (prompt-cache) support.

It follows the [community model](https://github.com/gufo-org/gufo/blob/main/docs/model-extensions/DISTRIBUTION.md):
an independent repository compiled against a pinned upstream Gufo with
`-DGUFO_EXTENSION_DIRS`, not a fork of Gufo. The probe only claims `qwen4exp`
checkpoints whose routed experts are IQ-quantized, so plain Q4_K_XL files
still resolve to the built-in provider.

## Provenance

- Model sources: [Gufo](https://github.com/gufo-org/gufo) 0.30.0
  [`src/models/qwen38_flash_next`](https://github.com/gufo-org/gufo/tree/v0.30.0/src/models/qwen38_flash_next)
  (MIT), with the IQ3_S/IQ4_XS/IQ4_NL + Q6_K work re-applied from the
  [`gufo-iq4_xs` fork](https://github.com/johandevlabs/gufo-iq4_xs)
  ([`b13c2fa6..912679bc`](https://github.com/johandevlabs/gufo-iq4_xs/compare/b13c2fa6b6abf71266529f2c878e07704f28dfd7...912679bccb9e9b91c6ae4000fc017b83a2c630a7)).
- `src/qwen/`: vendored copy of Gufo's private
  [`src/families/qwen/*`](https://github.com/gufo-org/gufo/tree/v0.30.0/src/families/qwen)
  (MIT) — the external-extension sandbox does not link
  `gufo_core`/`gufo_qwen_*`, so the family layer is rebuilt as
  extension-private targets.
- `src/kernels/rocm/mmq/`: GGML-derived MMQ/MoE kernels (MIT, see the
  `VENDOR.md` inside).
- Built and validated against upstream Gufo
  [`2c97b72d`](https://github.com/gufo-org/gufo/tree/2c97b72d8671ddd0c74ae7d517b8cef4ebb78871)
  (v0.30.0+9), pinned in `flake.nix`.
- Namespace `gufo::models::flash_next_iq`; provider id `flash-next-iq`,
  artifact key `gguf:qwen4exp-iq`.

## Requirements

- Linux x86-64, AMD `gfx1151` (ROCm ≥ 6.4 toolchain with hipcc; tested with
  ROCm 7.15 / clang 23), HIP `gfx1151` support in Gufo ≥ 0.30.
- **Focused selection builds only** (`-DGUFO_EXTENSIONS=flash_next_iq`). The
  vendored Qwen family copies keep their original namespaces, so building this
  extension together with the shipped Qwen models would duplicate symbols.

## Build

```sh
# In a Gufo checkout (e.g. 0.30.0), without Nix. On a host without the -dev
# packages, point PKG_CONFIG_PATH at a local apt-extracted dev prefix first.
PKG_CONFIG_PATH=/path/to/devdeps/usr/lib/x86_64-linux-gnu/pkgconfig \
cmake --preset release \
  -DGUFO_EXTENSION_DIRS=/path/to/flash-next-iq-gufo-extension \
  -DGUFO_EXTENSIONS=flash_next_iq
cmake --build --preset release --target gufo --parallel 4
```

`gufo diagnose --section extensions --json` then lists `flash_next_iq` with
its source directory and provenance. For a Nix build, use the extension
template flake shape (Gufo as a flake input, this repo passed via
`GUFO_EXTENSION_DIRS` / `gufo.packages...override { extensions = [ ... ]; }`)
with the same focused selection.

## Container image

A prebuilt runtime image (ubuntu 24.04 base with the extension-built `gufo`)
is published at `ghcr.io/storm1er/flash-next-iq-gufo-extension:latest`,
alongside pinned versions (e.g. `0.30.0-iq.1`). ROCm userspace is **not**
baked in: mount the host's ROCm directory read-only so the image always
matches the host KFD driver (`LD_LIBRARY_PATH=/opt/rocm/lib` is preset).

```sh
docker run --rm --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render --ulimit memlock=-1 \
  -v /opt/rocm:/opt/rocm:ro \
  -v /path/to/model-snapshot:/models:ro \
  -p 8080:8080 \
  ghcr.io/storm1er/flash-next-iq-gufo-extension:latest \
  gufo serve --host 0.0.0.0 --port 8080 llm \
    --model /models/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
    --speculative mtp --mtp-model /models/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf \
    --mmproj /models/mmproj-BF16.gguf \
    --sessions 2 --context 262144 \
    --served-model-name Qwen3.8-Flash-Next
```

If `/opt/rocm` on the host is only a set of compatibility symlinks, bind the
versioned directory it points into instead, e.g.
`-v /opt/rocm/core-10.0:/opt/rocm:ro`. The image ships `video` (44) and
`render` (992) groups so `--group-add` names resolve; in
`docker-compose.yml` the same keys work verbatim under
`devices`/`group_add`/`ulimits`. To rebuild the image yourself, copy any
focused-selection `gufo` build to `dist/gufo` and run
`docker build -f docker/runtime.Dockerfile .`.

## Usage

```sh
gufo serve --model /models/UD-IQ4_XS --mtp-model /models/next-nvfp4-mtp.gguf \
  --port 8080 --context 262144
```

`--mtp-model` is a core CLI option, so it is unchanged from the built-in
provider; `--sessions`, `--context`, `--cache-ram-bytes` work as usual.
The `gufo prompt` / `gufo bench` paths support MTP too (validated by the
fork-derived test matrix).

## Tests

```sh
cmake --preset release ... -DGUFO_FLASH_NEXT_IQ_TESTS=ON
cmake --build --preset release --target \
  flash_next_iq_iq_ops_test flash_next_iq_routed_wmma_ops_test
LD_LIBRARY_PATH=/opt/rocm/lib ./build/release/extensions/build/flash_next_iq/build/flash_next_iq_iq_ops_test
LD_LIBRARY_PATH=/opt/rocm/lib ./build/release/extensions/build/flash_next_iq/build/flash_next_iq_routed_wmma_ops_test
```

Measured on a gfx1151 machine (Nov 2026), with the production server still
resident: `iq_ops_test` passes all IQ3_S/IQ4_XS/IQ4_NL vector/MMQ and Q6_K
head checks (worst relative error ≈ 4.6e-3 against the CPU dequantizers);
`routed_wmma_ops_test` passes every IQ tile-row configuration with F16-vs-F64
error ≤ 0.53 % of output scale (threshold 2e-3). Both tests only allocate
small tensors, so they run alongside a loaded model.

End-to-end on the real UD-IQ4_XS checkpoint + Q8_0 MTP draft + BF16 mmproj
at 262K context, 2 sessions (cold cache): load 79 s / 80.7 GiB device,
decode 51–61 tok/s with MTP (draft acceptance 74–85 %), coherent reasoning
output, and working continuation checkpoints (~269 MiB) with shared-prefix
reuse. `--mtp-model` and the component cache behave exactly like the
built-in provider.

## Limitations

- IQ types use two separate RoutedF16 GEMMs for gate/up instead of the paired
  Q4_K/Q5_K kernel (0.30 reworked that kernel before porting); prefill of the
  expert layer is slightly slower than the 0.12 fork's paired route, decode
  is unaffected. Wide prefill builds two routed tile maps (narrow for the
  dual projection, 128-row for down) via `Executor::RouteHints`.
- IQ4_NL wide (128-row) down tiles go through the format-generic
  `RoutedF16GEMMKernel` rather than the specialized byte-block kernel.
- Coexistence with the shipped `qwen38_flash_next`/Qwen family libraries in
  one binary is not supported (vendored namespaces unchanged); see
  Requirements.
- The vision encoder is inherited unchanged from 0.30.0 and was validated
  only as far as loading (`--mmproj`); image requests follow the built-in
  provider's path.

## License

MIT (same as Gufo); third-party notices in `LICENSE` and `src/kernels/rocm/mmq/VENDOR.md`.
