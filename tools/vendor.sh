#!/usr/bin/env bash
# Vendored upstream tree synchronizer for the flash_next_iq extension.
#
# src/qwen/ and src/testing/compare/ are copies of Gufo's private
# src/families/qwen and src/testing/compare, transformed so the extension
# builds standalone AND coexists with the in-tree Qwen models:
#   - include paths rewritten to the extension API headers;
#   - family-owned symbols renamed into extension-private namespaces
#     (gufo::hip_iq, gufo::tokenization_iq, gufo::models::qwen_iq, ... plus
#     the fniq_ mmq prefix) so one binary can link both copies.
# The IQ-quant delta is stored separately in tools/iq-family.patch; the
# upstream revision is recorded in .upstream-rev.
#
# Usage:
#   tools/vendor.sh check     <gufo-src>          # verify tree == clean+patch
#   tools/vendor.sh regenerate <gufo-src>         # refresh tools/iq-family.patch
#                                                 # from the current working tree
#   tools/vendor.sh revendor  <gufo-src> [rev]    # re-emit src/qwen src/testing
#                                                 # from <rev> (default: .upstream-rev)
#                                                 # and re-apply tools/iq-family.patch
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STAGE="$ROOT/build/vendor-stage"
PATCH="$ROOT/tools/iq-family.patch"
REV_FILE="$ROOT/.upstream-rev"

die() { echo "vendor.sh: $*" >&2; exit 1; }

# transform_tree <dir>: upstream -> extension shape. This is the canonical
# transformation; `check` guarantees src/ equals upstream+transform+patch.
transform_tree() {
  local dir="$1"
  local files
  files=$(find "$dir" -type f \( -name '*.h' -o -name '*.hpp' -o -name '*.cpp' -o -name '*.hip' \))
  [ -n "$files" ] || die "no sources under $dir"
  # shellcheck disable=SC2086
  sed -i -E \
    -e 's|#include "src/core/crypto/sha256.hpp"|#include <gufo/ext/sha256.hpp>|g' \
    -e 's|#include "src/core/gguf_reader.hpp"|#include <gufo/ext/gguf_reader.hpp>|g' \
    -e 's|#include "src/core/hip/detail/([A-Za-z0-9_]+\.hpp)"|#include <gufo/ext/experimental/\1>|g' \
    -e 's|#include "src/core/hip/([A-Za-z0-9_]+\.hpp)"|#include <gufo/ext/experimental/\1>|g' \
    -e 's|#include "src/core/quant/ggml_dequant.hpp"|#include <gufo/ext/experimental/ggml_dequant.hpp>|g' \
    -e 's|#include "src/core/([A-Za-z0-9_]+\.hpp)"|#include <gufo/ext/\1>|g' \
    -e 's|#include "src/families/qwen/|#include "qwen/|g' \
    -e 's|#include "src/testing/compare/|#include "testing/compare/|g' \
    -e 's/\bgufo::hip\b/gufo::hip_iq/g' \
    -e 's/(^|[^:[:alnum:]_])hip::/\1hip_iq::/g' \
    -e 's/(^|[^:[:alnum:]_])detail::(Emit[A-Za-z0-9_]*|DispatchTelemetryEnabled|FindHipblasLtPlanRecord|LoadHipblasLtPlanDatabase|SaveHipblasLtPlanDatabase|InspectHipblasLtPlanDatabase|HipblasLtPlan[A-Za-z0-9_]*)/\1gufo::hip::detail::\2/g' \
    -e 's/gufo::hip_iq::detail::(Emit[A-Za-z0-9_]*|DispatchTelemetryEnabled|FindHipblasLtPlanRecord|LoadHipblasLtPlanDatabase|SaveHipblasLtPlanDatabase|InspectHipblasLtPlanDatabase|HipblasLtPlan[A-Za-z0-9_]*)/gufo::hip::detail::\1/g' \
    -e 's/(^|[^:[:alnum:]_])hip_iq::detail::(Emit[A-Za-z0-9_]*|DispatchTelemetryEnabled|FindHipblasLtPlanRecord|LoadHipblasLtPlanDatabase|SaveHipblasLtPlanDatabase|InspectHipblasLtPlanDatabase|HipblasLtPlan[A-Za-z0-9_]*)/\1gufo::hip::detail::\2/g' \
    -e 's/hip_iq::(AllocateDevice|TransferStream|SnapshotTransfer|WeightUpload|WeightAllocator|HostAllocations|CopySnapshotToHost|CopySnapshotToDevice|CreateHipComponentCacheResources|ComponentCacheHooks)/hip::\1/g' \
    -e 's/\bgufo::tokenization\b/gufo::tokenization_iq/g' \
    -e 's/(^|[^:[:alnum:]_])tokenization::/\1tokenization_iq::/g' \
    -e 's/\bgufo::speculative\b/gufo::speculative_iq/g' \
    -e 's/(^|[^:[:alnum:]_])speculative::/\1speculative_iq::/g' \
    -e 's/\bspeculative_iq::PromptLookup\b/speculative::PromptLookup/g' \
    -e 's/\bgufo::testing\b/gufo::testing_iq/g' \
    -e 's/(^|[^:[:alnum:]_])testing::/\1testing_iq::/g' \
    -e 's/\bgufo::models::qwen\b/gufo::models::qwen_iq/g' \
    -e 's/\bmodels::qwen\b/models::qwen_iq/g' \
    -e 's/(^|[^:[:alnum:]_])qwen::/\1qwen_iq::/g' \
    -e 's/\bqfn_/fniq_/g' \
    $files
}

# clean_tree <gufo-src> <rev> <dest>: upstream copy at <rev>, build files
# removed, transformation applied.
clean_tree() {
  local gufo_src="$1" rev="$2" dest="$3"
  git -C "$gufo_src" rev-parse -q --verify "$rev^{commit}" >/dev/null \
    || die "$gufo_src has no revision $rev (fetch it first)"
  rm -rf "$dest"
  mkdir -p "$dest/src"
  git -C "$gufo_src" archive "$rev" src/families/qwen src/testing/compare \
    | tar -x -C "$dest"
  ( cd "$dest/src" && mv families/qwen qwen && rm -rf families ) || die "archive layout changed"
  find "$dest" \( -name CMakeLists.txt -o -name library.json -o -name AGENTS.md \) -delete
  transform_tree "$dest/src"
}

# Current tree with the transformation applied on top of staging, to make
# diff/apply symmetric with clean_tree output.
stage_current() {
  rm -rf "$STAGE/current"
  mkdir -p "$STAGE/current/src"
  cp -a "$ROOT/src/qwen" "$STAGE/current/src/"
  mkdir -p "$STAGE/current/src/testing"
  cp -a "$ROOT/src/testing/compare" "$STAGE/current/src/testing/"
}

main() {
case "${1:?usage: vendor.sh <check|regenerate|revendor> <gufo-src> [rev]}" in
check)
  local gufo_src=${2:?gufo-src} rev=${3:-$(cat "$REV_FILE")}
  clean_tree "$gufo_src" "$rev" "$STAGE/clean"
  stage_current
  ( cd "$STAGE/clean" && patch -s -p1 --posix -i "$PATCH" ) \
    || die "tools/iq-family.patch does not apply cleanly to upstream $rev"
  diff -ruN "$STAGE/clean/src" "$STAGE/current/src" \
    || die "src/qwen + src/testing drifted from upstream $rev + transform + patch"
  echo "vendor.sh: tree matches upstream $rev + transform + patch"
  ;;
regenerate)
  local gufo_src=${2:?gufo-src} rev=${3:-$(cat "$REV_FILE")}
  clean_tree "$gufo_src" "$rev" "$STAGE/clean"
  stage_current
  ( cd "$STAGE" && diff -ruN clean/src current/src > "$PATCH" ) || true
  [ -s "$PATCH" ] || die "empty patch: current tree matches upstream?"
  echo "vendor.sh: wrote $PATCH ($(grep -c '^+++' "$PATCH") files)"
  # Verify round-trip: the regenerated patch must apply to a fresh clean tree.
  clean_tree "$gufo_src" "$rev" "$STAGE/verify"
  ( cd "$STAGE/verify" && patch -s -p1 --posix -i "$PATCH" ) || die "round-trip: patch failed"
  diff -ruN "$STAGE/verify/src" "$STAGE/current/src" || die "round-trip: drift after apply"
  echo "$rev" > "$REV_FILE"
  echo "vendor.sh: round-trip verified against upstream $rev"
  ;;
revendor)
  local gufo_src=${2:?gufo-src} rev=${3:-$(cat "$REV_FILE")}
  clean_tree "$gufo_src" "$rev" "$STAGE/verify"
  ( cd "$STAGE/verify" && patch -p1 --posix -i "$PATCH" ) \
    || die "tools/iq-family.patch conflicts with $rev: resolve in $STAGE/verify, then copy src/ back"
  echo "vendor.sh: patched tree ready in $STAGE/verify/src (copy over src/ after review)"
  echo "           then run: tools/vendor.sh regenerate $gufo_src $rev"
  ;;
esac
}

# Sourceable for transform_tree; executable as a CLI.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  main "$@"
fi
