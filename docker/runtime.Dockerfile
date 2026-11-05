# Runtime image for the flash_next_iq extension build of Gufo.
#
# The binary is produced by the host build documented in README.md:
#   cmake --preset release -DGUFO_EXTENSION_DIRS=<this repo> \
#     -DGUFO_EXTENSIONS=flash_next_iq && cmake --build --preset release -t gufo
# then copied to dist/gufo before `docker build`:
#   docker build -f docker/runtime.Dockerfile -t flash-next-iq-gufo:local .
#
# ROCm userspace is NOT baked in: mount the host's /opt/rocm read-only so the
# image always matches the host KFD driver exactly, e.g. in compose:
#   volumes:
#     - /opt/rocm/core-10.0:/opt/rocm:ro   # core-10.0 = host ROCm version dir
#     - <model snapshot>:/models:ro
FROM ubuntu:24.04

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl jq procps \
      libcurl4 libicu74 libjpeg-turbo8 libpng16-16t64 libwebp7 libgomp1 \
      libsndfile1 libsoxr0 \
 && rm -rf /var/lib/apt/lists/*

COPY dist/gufo /usr/local/bin/gufo

# Mirrors gufo-toolboxes gpuEnv (ROCM_PATH/HIP_PLATFORM) minus the nix paths.
ENV ROCM_PATH=/opt/rocm \
    HIP_PLATFORM=amd \
    LD_LIBRARY_PATH=/opt/rocm/lib \
    HOME=/home/gufo
# group_add: [video, render] in compose resolves names against the image's
# /etc/group, so keep entries with the host GIDs that own /dev/kfd + /dev/dri.
RUN groupadd -f -g 44 video \
 && groupadd -f -g 992 render \
 && useradd -u 10003 -m -d /home/gufo gufo
WORKDIR /home/gufo
CMD ["/bin/bash"]
