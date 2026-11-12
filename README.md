# flash_next_iq — Qwen3.8 Flash-Next IQ-quant extension for Gufo

Community (out-of-tree) gufo-extension that lets [Gufo](https://github.com/gufo-org/gufo) serve the
**UD-IQ4_XS** quantizations of `Qwen3.8-Flash-Next` on AMD Strix Halo (`gfx1151`), with MTP speculative
decoding, continuation (prompt cache, in-pass checkpoints) and the BF16 vision projector:

- the **Unsloth Dynamic 3.1** re-release (2026-11, the current `UD-IQ4_XS`, four shards): routed gate/up in
  **IQ3_XXS / IQ4_XS / IQ2_S**, down in **Q5_0 / IQ4_NL / Q2_0**, **IQ4_XS** token embeddings, **Q6_K**
  attention, shared experts and logits head;
- the 2026-10 export (three shards): gate/up in **IQ3_S / IQ4_XS**, down in **IQ4_NL / Q8_0**, Q8_0 trunk.

Gufo's built-in `qwen38_flash_next` provider runs the K-quant files (UD-Q4_K_XL); it cannot run IQ
experts. This provider (`flash-next-iq`, artifact key `gguf:qwen4exp-iq`) claims a `qwen4exp` GGUF only when
at least one routed-expert tensor uses an IQ/narrow-bit format and it can run every other expert tensor, so
the two never compete for a file. GGML `Q2_0` (two UD 3.1 layers) is a type Gufo core does not size: the
provider sets `accepts_unsized_tensors` and sizes and decodes it itself.

- **Maintainer:** [@storm1er](https://github.com/storm1er) — issues and pull requests on this repository.
- **Gufo:** 0.34.0 (extension API `>= 0.3.0`), tested with v0.34.0 `2845939b` and `main` `5ace393c`.
- **Platforms:** Linux x86-64, AMD `gfx1151` (Strix Halo), ROCm 10.
- **License:** MIT (source). Model weights: [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE);
  this repository ships no weights.

## Usage

### Weights

From [`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF); the UD 3.1
files are revision `084c631ac221016019d6e0a863afe92d90edd3b0` (2026-11-07):

```sh
hf download unsloth/Qwen3.8-Flash-Next-GGUF \
  --revision 084c631ac221016019d6e0a863afe92d90edd3b0 \
  --include 'UD-IQ4_XS/*' 'MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf' 'mmproj-BF16.gguf' \
  --local-dir /path/to/model-snapshot
```

That is the four `UD-IQ4_XS` shards (93.7 GB), the `Q8_0` MTP drafter (4.1 GB) and the BF16 vision
projector (0.9 GB). Use the `Q8_0` drafter: it accepts noticeably more draft tokens than the `Q4_K_M` one
(both work). The 2026-10 export is revision `38bb39ee97821de2c9009abb7e93950eec396e66`.

### Install as a module (recommended)

Into an installed Gufo 0.34 (the `gufo-runtime` container, Nix `#sdk`, or a source build's SDK; see
[Gufo extensions](https://github.com/gufo-org/gufo/blob/main/docs/extensions.md)):

```sh
gufo extension install flash_next_iq      # once listed in the Gufo extension registry
gufo extension install https://github.com/storm1er/flash-next-iq-gufo-extension@<commit>
gufo serve --host 0.0.0.0 --port 8080 llm \
  --model /models/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00004.gguf \
  --speculative mtp --mtp-model /models/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf \
  --mmproj /models/mmproj-BF16.gguf \
  --sessions 2 --context 262144 \
  --served-model-name Qwen3.8-Flash-Next
```

With the stock container, keep the module store in a volume and run the server with the same store:

```sh
IMG=ghcr.io/gufo-org/toolboxes/gufo-runtime:0.34.0
STORE="-v gufo-extensions:/ext -e GUFO_EXTENSIONS_DIR=/ext"
docker run --rm --user 0:0 $STORE -v "$PWD:/src:ro" $IMG gufo extension install /src --yes
docker run --rm --user 0:0 $STORE --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render --ulimit memlock=-1 \
  -v /path/to/model-snapshot:/models:ro -p 8080:8080 \
  $IMG gufo serve --host 0.0.0.0 --port 8080 llm ...   # as above
```

Under Docker, run as root (`--user 0:0`): as the image's unprivileged `gufo` user the 80 GiB weight upload
fails to allocate its pinned staging buffer ("weight upload staging allocation failed"). Rootless Podman
with `--userns=keep-id`, as in Gufo's toolbox README, has not been tested with this extension.

### Prebuilt containers

CI publishes `gufo` v0.34.0 with this extension compiled in, slim (host `/opt/rocm` mounted read-only)
and ROCm-baked (`-rocm`, no volume):

```sh
docker run --rm --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render --ulimit memlock=-1 \
  -v /path/to/model-snapshot:/models:ro -p 8080:8080 \
  ghcr.io/storm1er/flash-next-iq-gufo-extension:latest-rocm \
  gufo serve --host 0.0.0.0 --port 8080 llm ...   # as above
```

For the slim `latest` image add `-v /opt/rocm:/opt/rocm:ro` (or the versioned directory, e.g.
`-v /opt/rocm/core-10.0:/opt/rocm:ro`, when `/opt/rocm` is only symlinks). Both images ship the `video`
(44) and `render` (992) groups; the same keys work in compose. `gufo prompt`, `gufo chat` and
`gufo bench` take the same model options.

### Known limitations

- Prompt passes are not split by layers. Beside decoding sessions a prompt runs in 512-token passes
  (`TextSchedulingProfile{512, 1, 8}`), so other sessions wait at most one such pass; Gufo's in-tree
  Flash-Next splits its passes instead.
- IQ gate/up use two RoutedF16 GEMMs instead of the paired Q4_K/Q5_K kernel; IQ4_NL, Q5_0 and Q2_0
  wide (128-row) down tiles use the format-generic `RoutedF16GEMMKernel`.
- The `Q4_K_M` MTP drafter's Q4_K/Q5_0 projections prefill on the tiled MMQ GEMM (dp4a), not the
  F16 matrix cores; the trunk is unaffected.
- The vision encoder is Gufo's (vendored) and is validated as far as loading (`--mmproj`); image
  requests follow the built-in provider's path.

## Quality and performance

### Operator tests

`iq_ops_test` covers every expert format's vector and MMQ routes (IQ3_S/IQ3_XXS/IQ2_S/IQ4_XS/IQ4_NL/
Q5_0/Q2_0), the Q6_K head, the dense Q6_K/Q5_0 MMQ GEMM, the dense Q6_K F16 matrix-core GEMM at the trunk's
shapes (relative error ≤ 3e-4 against F64) and the token-embedding gather (IQ4_XS and Q8_0 rows equal to
the CPU dequantizer). `routed_wmma_ops_test` checks the F16 routed route of every format on the 16-, 48- and
128-row tiles against the MMQ tier and F64 (F16-vs-F64 error ≤ 2e-3 of output scale, about ten times
closer to F64 than MMQ). Both pass on gfx1151 (Gufo v0.34.0, Nov 12 2026); they allocate only small
tensors, so they run beside a loaded server.

`gufo bench --validate-prefill 32` (GPU prefill logits against the reference forward): cosine 0.9974,
RMSE 0.20, same top token on UD 3.1; 0.9943 / 0.32 on the 2026-10 export.

### End-to-end (gfx1151, 128 GB, Nov 12 2026)

Two sessions, 262,144-token context, `--cache-ram-bytes 12884901888`, BF16 mmproj, default sampling; one
workload replaying an agent session: short prompts, a cold 174K-token prompt, follow-up turns on the live
checkpoint, a second 33.5K-token conversation, a return to the first, then both sessions decoding at once.

| | 2026-10 UD-IQ4_XS, built-in IQ fork (`gufo-runtime:edge`, Gufo PR #570 head `71bea96`) | UD 3.1 UD-IQ4_XS, this extension as a module in stock `gufo-runtime:0.34.0` |
|---|---|---|
| load / device memory | 12 s / 80.7 GiB | 11 s / 80.8 GiB |
| cold prefill, 174K tokens | 1,515 tok/s (TTFT 115.0 s) | 1,530 tok/s (TTFT 113.8 s) |
| cold prefill, 33.5K tokens | 1,511 tok/s | 1,595 tok/s |
| decode at 174K depth | 41–44 tok/s, MTP acceptance 57–68 % | 46–60 tok/s, MTP acceptance 69–74 % |
| follow-up TTFT (checkpoint reuse) | 0.24–0.41 s | 0.26–0.46 s |
| both sessions decoding | 50.8 tok/s aggregate | 60.0 tok/s aggregate |

Both with the `Q8_0` MTP drafter. `gufo bench -p 16384`: 1,797 tok/s on UD 3.1 and 1,794 on the 2026-10
export (same v0.34.0 build). Failures seen while getting here, and their fixes, are in the release history.

## Development

### Build from source

In a Gufo checkout of the pinned release (`.upstream-rev`, also pinned in `flake.nix`):

```sh
cmake --preset release -DGUFO_EXTENSION_DIRS=/path/to/flash-next-iq-gufo-extension \
  -DGUFO_EXTENSIONS=flash_next_iq
cmake --build --preset release --target gufo --parallel 8
build/release/gufo diagnose --section extensions --json
```

The tests are `role: "test"` targets built with `BUILD_TESTING`; Gufo's own test tree then needs every
in-tree extension, so configure with full discovery:

```sh
cmake --preset release -DBUILD_TESTING=ON -DGUFO_EXTENSION_DIRS=/path/to/flash-next-iq-gufo-extension
cmake --build --preset release --target flash_next_iq_iq_ops_test flash_next_iq_routed_wmma_ops_test
ctest --test-dir build/release -R '^flash_next_iq\.' --output-on-failure
```

With Nix, `nix build` builds the pinned Gufo with this extension (`flake.nix`, `flake.lock`). The extension
uses only `<gufo/ext/...>` headers and the public `gufo_ext_*` targets; its helper archives are built
position-independent so the same sources link into `gufo` or into a loadable module.

### Container images

CI (`.github/workflows/images.yml`) builds `gufo` at `.upstream-rev` with this extension on every push to
`main` and publishes `latest`, `latest-rocm` and immutable `sha-<short>` / `sha-<short>-rocm` tags; its
`module` job installs the checkout with `gufo extension install` into the stock `gufo-runtime` of the same
release. Locally:

- Slim (`docker/runtime.Dockerfile`): copy a `gufo` build containing this extension to `dist/gufo`, then
  `docker build -f docker/runtime.Dockerfile .`; the bind-mounted host ROCm pins userspace to the host driver.
- ROCm-baked (`docker/runtime-rocm.Dockerfile`): `tools/rocm-subset.sh /path/to/gufo /tmp/rocm-image`
  collects the binary's `/opt/rocm` closure (HIP, hipBLAS/Lt, rocBLAS + gfx1151 tuning data), then
  `docker build -f docker/runtime-rocm.Dockerfile -t <tag> /tmp/rocm-image`.

### Vendoring

`src/qwen/` and `src/testing/compare/` are copies of Gufo's private `src/families/qwen` and
`src/testing/compare` (external extensions cannot link those targets), **verbatim upstream after a
mechanical transform**: includes rewritten to the extension API headers and family-owned symbols renamed
into private namespaces (`gufo::hip`/`tokenization`/`speculative`/`testing` → `*_iq`,
`gufo::models::qwen` → `gufo::models::qwen_iq`, `qfn_` → `fniq_`), so the extension links next to Gufo's
own Qwen models. `tools/vendor.sh` owns the rules: `check <gufo-src>` fails on drift, `revendor <gufo-src>
<rev>` re-emits the trees, `regenerate <gufo-src> <rev>` re-records the revision. `.upstream-rev` pins it
(Gufo v0.34.0); `tools/iq-family.patch` is empty: all IQ code lives in this extension's own model and
device layer.

### Updating Gufo

Pin the new release in `.upstream-rev`, `flake.nix`/`flake.lock` and the CI `RUNTIME_IMAGE`, run
`tools/vendor.sh revendor`, rebuild, rerun the operator tests, `--validate-prefill` and the end-to-end
workload, keep `api_version` at the lowest revision that has everything used, and bump `version`.

## Release history

- **0.2.0** (2026-11-12): extension API 0.3 / Gufo 0.34.0, installable as a module; GGML Q2_0 without a
  Gufo core patch (`accepts_unsized_tensors`); text scheduling profile. UD 3.1 fixes: IQ4_XS token
  embeddings decoded 12 of 16 codes wrong (prompts misread, low MTP acceptance); Q6_K trunk prefill on the
  F16 matrix cores (fused SSM/attention projections, attention output, shared-expert overlap) and
  IQ3_XXS/IQ2_S/Q5_0/Q2_0 experts on the F16 WMMA route: cold prefill 406 → 1,530 tok/s at 174K tokens.
  The probe again claims the 2026-10 export (it keeps five down-expert layers in Q8_0).
- **0.1.0** (2026-11-05 – 11-10, images `0.30.0-iq.1`–`.3`, `sha-05fff08`, `sha-87cbcca`): IQ3_S/IQ4_XS/
  IQ4_NL support on Gufo 0.30 (`b7cf081d`, with a core Q2_0 patch); first load of the UD 3.1 files.

## Provenance

- Model sources: Gufo 0.30.0 `src/models/qwen38_flash_next` (MIT), with the IQ3_S/IQ4_XS/IQ4_NL + Q6_K
  work re-applied from the `gufo-iq4_xs` fork
  (https://github.com/johandevlabs/gufo-iq4_xs/compare/b13c2fa6b6abf71266529f2c878e07704f28dfd7...912679bccb9e9b91c6ae4000fc017b83a2c630a7).
- `Executor::RouteHints` "route staging drain": expert counts are awaited with a full stream drain rather
  than the download event, adopted from the mikealanni/gufo fork (https://github.com/mikealanni/gufo),
  which documents this driver defect (events completing while pinned-staging copies are still queued).
- `src/qwen/`, `src/testing/compare/`: vendored from Gufo v0.34.0 (MIT), see Vendoring.
- `src/kernels/rocm/mmq/` and the IQ grid/value tables in `src/kernels/rocm/iq_tables.hpp`: derived from
  ggml-org/llama.cpp (MIT, The ggml authors: `licenses/llama.cpp.txt`, `src/kernels/rocm/mmq/VENDOR.md`).
- Namespace `gufo::models::flash_next_iq`; provider id `flash-next-iq`, artifact key `gguf:qwen4exp-iq`,
  model id `flash-next-iq`.

License: MIT (same as Gufo); third-party notices in `licenses/llama.cpp.txt` and
`src/kernels/rocm/mmq/VENDOR.md`. Installs ship them under `share/licenses/gufo/extensions/flash_next_iq`.
