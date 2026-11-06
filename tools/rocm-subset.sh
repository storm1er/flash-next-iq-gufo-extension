#!/usr/bin/env bash
# Assemble the baked-in ROCm subset for docker/runtime-rocm.Dockerfile.
# Scans the gufo binary's transitive /opt/rocm shared-library closure plus
# the gfx1151 tuning data of hipBLASLt/rocBLAS into <outdir>/opt-rocm.
#
#   tools/rocm-subset.sh /path/to/gufo [outdir]     (default outdir:
#   /tmp/opencode/rocm-image; the copy only reads /opt/rocm)
set -euo pipefail
BIN=${1:?usage: rocm-subset.sh /path/to/gufo [outdir]}
DEST=${2:-/tmp/opencode/rocm-image}/opt-rocm
ROCM=${ROCM:-/opt/rocm}
ARCH=${ARCH:-gfx1151}

rm -rf "$DEST"; mkdir -p "$DEST/lib"
seen="$DEST/.closure.txt"; : > "$seen"
queue=$(LD_LIBRARY_PATH="$ROCM/lib" ldd "$BIN" | awk -v r="$ROCM" 'index($3,r)==1{print $3}')
while [ -n "$queue" ]; do
  next=""
  for f in $queue; do
    case "$f" in "$ROCM"/*) ;; *) continue ;; esac
    grep -qx "$f" "$seen" && continue
    echo "$f" >> "$seen"
    rel=${f#"$ROCM"/lib/}
    mkdir -p "$DEST/lib/$(dirname "$rel")"
    cp -aL "$f" "$DEST/lib/$rel"
    next="$next $(LD_LIBRARY_PATH="$ROCM/lib" ldd "$f" 2>/dev/null | awk -v r="$ROCM" 'index($3,r)==1{print $3}')"
  done
  queue="$next"
done
for d in hipblaslt rocblas; do
  [ -d "$ROCM/lib/$d/library" ] || continue
  mkdir -p "$DEST/lib/$d/library"
  cp -aL "$ROCM"/lib/$d/library/"$ARCH"* "$DEST/lib/$d/library/" 2>/dev/null || true
done
rm -f "$seen"
echo "subset ready in $(dirname "$DEST") ($(du -sh "$DEST" | cut -f1))"
