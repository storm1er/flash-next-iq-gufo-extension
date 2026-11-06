# ROCm-baked variant: embeds the host ROCm 10.0 userspace closure (loader,
# HIP, hipBLAS/hipBLASLt/rocBLAS and the gfx1151 tuning data) so no /opt/rocm
# volume is needed. Build context must contain opt-rocm/ produced by
# tools/rocm-subset.sh (see README "Container image"):
#   docker build -f docker/runtime-rocm.Dockerfile -t <tag> <context-dir>
#
# Trade-off vs the slim image + bind mount: portable to any host whose KFD
# driver is >= the bundled userspace, at ~600 MB extra image size; the slim
# variant always matches the host driver exactly.
ARG BASE=ghcr.io/storm1er/flash-next-iq-gufo-extension:0.30.0-iq.3
FROM ${BASE}
COPY opt-rocm /opt/rocm
