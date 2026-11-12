# Qwen3.8 Flash-Next IQ-quant extension (external, community).
#
# Model sources derive from gufo's qwen38_flash_next extension (MIT) with the
# IQ4_XS-family support from the gufo-iq4_xs fork; src/qwen/ and
# src/testing/compare/ are vendored copies of gufo's private family and test
# libraries (MIT), rebuilt as extension-private targets because the external
# sandbox does not link gufo_core/gufo_qwen_* targets. The vendored trees use
# extension-private namespaces (gufo::hip_iq, gufo::models::qwen_iq, ...; see
# tools/vendor.sh), so this extension coexists with the in-tree Qwen models
# in a single binary.

set(FNIQ "${GUFO_EXTENSION_BUILD_ROOT}/src")
set(FNIQ_QWEN "${FNIQ}/qwen")
set(FNIQ_KERNELS_DIR "${FNIQ}/kernels/rocm")
set(FNIQ_MMQ_DIR "${FNIQ_KERNELS_DIR}/mmq")

# Vendored Qwen CPU layer: config extraction, tokenizer, chat template,
# vision prompt/image-cache, and the logit comparator used by the drivers.
add_library(gufo_flash_next_iq_qwen STATIC
  ${FNIQ_QWEN}/config/qwen_config.cpp
  ${FNIQ_QWEN}/text/tokenizer.cpp
  ${FNIQ_QWEN}/text/chat_template.cpp
  ${FNIQ_QWEN}/vision/image_cache.cpp
  ${FNIQ_QWEN}/vision/prompt.cpp
  ${FNIQ}/testing/compare/logit_comparator.cpp)
target_include_directories(gufo_flash_next_iq_qwen PUBLIC "${FNIQ}")
target_link_libraries(gufo_flash_next_iq_qwen PUBLIC gufo_ext_runtime)
target_compile_features(gufo_flash_next_iq_qwen PUBLIC cxx_std_20)
# Helper archives link into the loadable module (`gufo extension install`)
# as well as into `gufo`, so they are built position-independent.
set_target_properties(gufo_flash_next_iq_qwen PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

if (ENGINE_ENABLE_HIP)
  # Vendored Qwen HIP operator library (mirrors gufo_qwen_kernels).
  set(FNIQ_QWEN_KERNEL_SOURCES
    ${FNIQ_QWEN}/kernels/attention_batched.hip
    ${FNIQ_QWEN}/kernels/attention_decode_graph.hip
    ${FNIQ_QWEN}/kernels/attention_decode.hip
    ${FNIQ_QWEN}/kernels/attention_decode_verify_wave64.hip
    ${FNIQ_QWEN}/kernels/attention_tile.hip
    ${FNIQ_QWEN}/kernels/attention_wmma.hip
    ${FNIQ_QWEN}/kernels/dflash_kernels.hip
    ${FNIQ_QWEN}/kernels/embed.hip
    ${FNIQ_QWEN}/kernels/fused.hip
    ${FNIQ_QWEN}/kernels/gemv.hip
    ${FNIQ_QWEN}/kernels/gemv_quant.hip
    ${FNIQ_QWEN}/kernels/hipblas_gemm.hip
    ${FNIQ_QWEN}/kernels/hipblaslt_gemm.hip
    ${FNIQ_QWEN}/kernels/norm.hip
    ${FNIQ_QWEN}/kernels/prefill_embed.hip
    ${FNIQ_QWEN}/kernels/prefill_fp16.hip
    ${FNIQ_QWEN}/kernels/prefill_gemm.hip
    ${FNIQ_QWEN}/kernels/prefill_norm.hip
    ${FNIQ_QWEN}/kernels/prefill_quant_gemm.hip
    ${FNIQ_QWEN}/kernels/prefill_quant_wave64.hip
    ${FNIQ_QWEN}/kernels/prefill_residual.hip
    ${FNIQ_QWEN}/kernels/prefill_rope.hip
    ${FNIQ_QWEN}/kernels/prefill_ssm.hip
    ${FNIQ_QWEN}/kernels/prefill_swiglu.hip
    ${FNIQ_QWEN}/kernels/prefill_unpack.hip
    ${FNIQ_QWEN}/kernels/qkv.hip
    ${FNIQ_QWEN}/kernels/residual.hip
    ${FNIQ_QWEN}/kernels/rope.hip
    ${FNIQ_QWEN}/kernels/sample.hip
    ${FNIQ_QWEN}/kernels/small_batch_quant16_wave64.hip
    ${FNIQ_QWEN}/kernels/small_batch_wave64.hip
    ${FNIQ_QWEN}/kernels/ssm_decode_recurrence.hip
    ${FNIQ_QWEN}/kernels/ssm.hip
    ${FNIQ_QWEN}/kernels/ssm_recurrence.hip
    ${FNIQ_QWEN}/kernels/ssm_row_split.hip
    ${FNIQ_QWEN}/kernels/swiglu.hip
    ${FNIQ_QWEN}/kernels/unpack.hip
    ${FNIQ_QWEN}/kernels/verify_gemm_27b.hip
    ${FNIQ_QWEN}/vision/encoder.hip
    ${FNIQ_QWEN}/vision/device_input.hip)
  set_source_files_properties(${FNIQ_QWEN_KERNEL_SOURCES}
    PROPERTIES LANGUAGE HIP)
  # Replicated from the gufo_qwen_kernels recipe: the wave-64 helpers and the
  # disabled-SLP prefill GEMMs keep their measured code generation. Combine
  # both properties per source: set_source_files_properties replaces.
  set_source_files_properties(
    ${FNIQ_QWEN}/kernels/small_batch_wave64.hip
    ${FNIQ_QWEN}/kernels/attention_decode_verify_wave64.hip
    PROPERTIES COMPILE_OPTIONS -mwavefrontsize64)
  set_source_files_properties(
    ${FNIQ_QWEN}/kernels/prefill_quant_wave64.hip
    PROPERTIES COMPILE_OPTIONS "-mwavefrontsize64;-fno-slp-vectorize")
  set_source_files_properties(
    ${FNIQ_QWEN}/kernels/small_batch_quant16_wave64.hip
    PROPERTIES COMPILE_OPTIONS
      "-mwavefrontsize64;-gline-tables-only;-Xarch_device;-mllvm=-amdgpu-sched-strategy=iterative-ilp")
  set_source_files_properties(
    ${FNIQ_QWEN}/kernels/prefill_quant_gemm.hip
    PROPERTIES COMPILE_OPTIONS -fno-slp-vectorize)
  add_library(gufo_flash_next_iq_qwen_hip STATIC ${FNIQ_QWEN_KERNEL_SOURCES})
  target_compile_definitions(gufo_flash_next_iq_qwen_hip PUBLIC ENGINE_ENABLE_HIP=1)
  target_include_directories(gufo_flash_next_iq_qwen_hip PUBLIC "${FNIQ}")
  target_include_directories(gufo_flash_next_iq_qwen_hip SYSTEM PUBLIC
    ${HIPCUB_INCLUDE_DIR}
    ${ROCPRIM_INCLUDE_DIR})
  target_compile_features(gufo_flash_next_iq_qwen_hip PUBLIC cxx_std_20)
  set_target_properties(gufo_flash_next_iq_qwen_hip PROPERTIES
    POSITION_INDEPENDENT_CODE ON)
  target_link_libraries(gufo_flash_next_iq_qwen_hip PUBLIC
    gufo_ext_runtime gufo_flash_next_iq_qwen hip::host roc::hipblas roc::hipblaslt)

  # Vendored GGML-derived MMQ/MoE kernels (same recipe as in-tree flash-next).
  set(FNIQ_MMQ_SOURCES
    ${FNIQ_MMQ_DIR}/fniq_ggml_stubs.hip.cpp
    ${FNIQ_MMQ_DIR}/fniq_mmq.hip.cpp
    ${FNIQ_MMQ_DIR}/mmid.hip.cpp
    ${FNIQ_MMQ_DIR}/mmvq.hip.cpp
    ${FNIQ_MMQ_DIR}/moe_wave64.hip.cpp
    ${FNIQ_MMQ_DIR}/quantize.hip.cpp)
  add_library(gufo_flash_next_iq_mmq STATIC ${FNIQ_MMQ_SOURCES})
  set_source_files_properties(${FNIQ_MMQ_SOURCES} PROPERTIES LANGUAGE HIP)
  set_source_files_properties(${FNIQ_MMQ_DIR}/moe_wave64.hip.cpp
    PROPERTIES COMPILE_OPTIONS "-mwavefrontsize64")
  set_target_properties(gufo_flash_next_iq_mmq PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
  target_compile_definitions(gufo_flash_next_iq_mmq PRIVATE
    GGML_HIP_NO_VMM ENGINE_ENABLE_HIP=1)
  target_compile_options(gufo_flash_next_iq_mmq PRIVATE
    -O3 -ffast-math -fno-finite-math-only -Wno-unused-value -Wno-unused-command-line-argument)
  target_include_directories(gufo_flash_next_iq_mmq SYSTEM PUBLIC ${FNIQ_MMQ_DIR})
  target_include_directories(gufo_flash_next_iq_mmq PUBLIC "${FNIQ}")
  target_link_libraries(gufo_flash_next_iq_mmq PUBLIC
    gufo_ext_runtime hip::host roc::hipblas)
endif()

gufo_extension_target(
  NAME gufo_flash_next_iq
  SOURCES
    src/config.cpp
    src/continuation_layout.cpp
    src/ngram.cpp
    src/rope_scaling.cpp
    src/weights.cpp
    src/bench_yarn.cpp
    src/probe.cpp
    src/continuation_identity.cpp
    src/extension.cpp
  LINK_LIBRARIES
    gufo_ext_runtime
    gufo_flash_next_iq_qwen
    Threads::Threads
)

if(TARGET gufo_flash_next_iq)
  target_include_directories(gufo_flash_next_iq PUBLIC "${FNIQ}")
  if(ENGINE_ENABLE_HIP)
    # gufo_core propagates this define in-tree; match the in-tree base target.
    target_compile_definitions(gufo_flash_next_iq PUBLIC ENGINE_ENABLE_HIP=1)
  endif()
endif()

add_library(gufo_flash_next_iq_reference STATIC EXCLUDE_FROM_ALL
  ${FNIQ}/cpu_ops.cpp
  ${FNIQ}/reference.cpp)
target_include_directories(gufo_flash_next_iq_reference PUBLIC "${FNIQ}")
set_target_properties(gufo_flash_next_iq_reference PROPERTIES
  POSITION_INDEPENDENT_CODE ON)
target_link_libraries(gufo_flash_next_iq_reference PUBLIC
  gufo_flash_next_iq)
set_source_files_properties(
  ${FNIQ}/cpu_ops.cpp
  ${FNIQ}/reference.cpp
  PROPERTIES COMPILE_OPTIONS "-O2")

if (ENGINE_ENABLE_HIP)
  gufo_extension_target(
    NAME gufo_flash_next_iq_hip
    SOURCES
      src/continuation_adapter.cpp
      src/kernels/rocm/blaslt.cpp
      src/kernels/rocm/device_model.cpp
      src/kernels/rocm/executor.cpp
      src/kernels/rocm/batch.cpp
      src/engine.cpp
      src/runner.cpp
      src/drivers.cpp
      src/provider.cpp
    LINK_LIBRARIES
      gufo_flash_next_iq
      gufo_ext_cache
      gufo_flash_next_iq_qwen_hip
      gufo_flash_next_iq_mmq
      hip::host
      roc::hipblas
      roc::hipblaslt
  )
endif()

if(TARGET gufo_flash_next_iq_hip)
  gufo_extension_hip_sources(gufo_flash_next_iq_hip
    SOURCES
      src/kernels/rocm/kernels.hip.cpp
  )
  gufo_extension_hip_sources(gufo_flash_next_iq_hip
    SOURCES
      src/kernels/rocm/w8a8_wave64.hip
  )
  set_source_files_properties(${FNIQ_KERNELS_DIR}/w8a8_wave64.hip
    PROPERTIES COMPILE_OPTIONS "-mwavefrontsize64")
  target_compile_definitions(gufo_flash_next_iq_hip PRIVATE QFN_ROCM_BUILD=1)
  target_compile_options(gufo_flash_next_iq_hip PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2>
    $<$<COMPILE_LANGUAGE:HIP>:-O3>
    $<$<COMPILE_LANGUAGE:HIP>:-ffast-math>
    $<$<COMPILE_LANGUAGE:HIP>:-fno-finite-math-only>
    $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-command-line-argument>)
  target_include_directories(gufo_flash_next_iq_hip SYSTEM PUBLIC
    ${HIPCUB_INCLUDE_DIR}
    ${ROCPRIM_INCLUDE_DIR}
    ${ROCWMMA_INCLUDE_DIR})
  target_include_directories(gufo_flash_next_iq_hip PUBLIC "${FNIQ}")
endif()

# Source license and the notices of the vendored GGML-derived kernels.
install(FILES
    "${GUFO_EXTENSION_BUILD_ROOT}/LICENSE"
    "${GUFO_EXTENSION_BUILD_ROOT}/licenses/llama.cpp.txt"
    "${FNIQ_MMQ_DIR}/VENDOR.md"
  DESTINATION share/licenses/gufo/extensions/flash_next_iq)

# Operator tests: every expert format's vector, MMQ and F16 matrix-core
# routes, the Q6_K head and dense GEMMs, and the token-embedding gather,
# against CPU/F64 references. They are role "test" targets (built with
# BUILD_TESTING), run on any gfx1151 device and allocate only small tensors,
# so they coexist with a loaded server. gufo_ext_runtime only declares the
# public services; gufo_ext_core/cache/hip link their implementations into
# these standalone executables.
set(_flash_next_iq_test_libs gufo_ext_core gufo_ext_cache)
if(ENGINE_ENABLE_HIP)
  list(APPEND _flash_next_iq_test_libs gufo_ext_hip)
endif()

foreach(test_name IN ITEMS iq_ops routed_wmma_ops)
  gufo_extension_target(
    NAME flash_next_iq_${test_name}_test
    SOURCES
      tests/${test_name}_test.cpp
    LINK_LIBRARIES
      gufo_flash_next_iq_hip
      gufo_flash_next_iq_mmq
      gufo_flash_next_iq
      gufo_ext_runtime
      ${_flash_next_iq_test_libs}
      hip::host
  )
  if(TARGET flash_next_iq_${test_name}_test)
    target_compile_definitions(flash_next_iq_${test_name}_test PRIVATE
      ENGINE_ENABLE_HIP=1)
    add_test(NAME flash_next_iq.${test_name}
      COMMAND flash_next_iq_${test_name}_test)
    set_tests_properties(flash_next_iq.${test_name} PROPERTIES
      LABELS "flash_next_iq;hip;gpu;gfx1151")
  endif()
endforeach()
