# Qwen3.8 Flash-Next IQ-quant extension (external, community).
#
# Model sources derive from gufo's qwen38_flash_next extension (MIT) with the
# IQ4_XS-family support from the gufo-iq4_xs fork; src/qwen/ and
# src/testing/compare/ are vendored copies of gufo's private family and test
# libraries (MIT), rebuilt as extension-private targets because the external
# sandbox does not link gufo_core/gufo_qwen_* targets. This extension must be
# built with a focused selection (for example -DGUFO_EXTENSIONS=flash_next_iq)
# so the in-tree flash-next model and the qwen family libraries are not
# compiled alongside these copies.

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
  target_link_libraries(gufo_flash_next_iq_qwen_hip PUBLIC
    gufo_ext_runtime gufo_flash_next_iq_qwen hip::host roc::hipblas roc::hipblaslt)

  # Vendored GGML-derived MMQ/MoE kernels (same recipe as in-tree flash-next).
  set(FNIQ_MMQ_SOURCES
    ${FNIQ_MMQ_DIR}/qfn_ggml_stubs.hip.cpp
    ${FNIQ_MMQ_DIR}/qfn_mmq.hip.cpp
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

install(FILES "${GUFO_EXTENSION_BUILD_ROOT}/LICENSE"
  DESTINATION share/licenses/gufo/extensions/flash_next_iq)

# Operator tests (UD-IQ4_XS support): accuracy of the IQ3_S/IQ4_XS/IQ4_NL
# routed-expert paths and the Q6_K logits head against CPU/F64 references.
# Opt in with -DGUFO_FLASH_NEXT_IQ_TESTS=ON; they run on any gfx1151 device
# and allocate only small tensors, so they coexist with a loaded server.
option(GUFO_FLASH_NEXT_IQ_TESTS "Build the Flash-Next IQ operator tests" OFF)
if(GUFO_FLASH_NEXT_IQ_TESTS AND ENGINE_ENABLE_HIP)
  # Tests are development artifacts and never ship. The oracle they compare
  # against (core dequantizers, GgufReader, WeightUpload) is defined in the
  # host's gufo_core archive; linking it by build-tree file path keeps the
  # production extension targets free of private targets while the tests can
  # still check exactness against the shipped CPU reference.
  set(_flash_next_iq_core "${PROJECT_BINARY_DIR}/libgufo_core.a")
  if(NOT EXISTS "${_flash_next_iq_core}")
    message(FATAL_ERROR "GUFO_FLASH_NEXT_IQ_TESTS: missing ${_flash_next_iq_core}")
  endif()
  foreach(test_name IN ITEMS iq_ops routed_wmma_ops)
    add_executable(flash_next_iq_${test_name}_test
      "${GUFO_EXTENSION_BUILD_ROOT}/tests/${test_name}_test.cpp")
    target_link_libraries(flash_next_iq_${test_name}_test PRIVATE
      gufo_flash_next_iq_hip
      gufo_flash_next_iq_mmq
      gufo_ext_runtime
      hip::host
      OpenSSL::Crypto
      ICU::uc
      ICU::i18n
      JPEG::JPEG
      PNG::PNG
      ZLIB::ZLIB
      CURL::libcurl
      PkgConfig::WEBP
      Threads::Threads
      "${_flash_next_iq_core}")
    target_compile_definitions(flash_next_iq_${test_name}_test PRIVATE
      ENGINE_ENABLE_HIP=1)
    target_include_directories(flash_next_iq_${test_name}_test PRIVATE
      "${FNIQ}")
    target_compile_features(flash_next_iq_${test_name}_test PRIVATE cxx_std_20)
    add_test(NAME flash_next_iq.${test_name}
      COMMAND flash_next_iq_${test_name}_test)
    set_tests_properties(flash_next_iq.${test_name} PROPERTIES
      LABELS "flash_next_iq;hip;gpu;gfx1151")
  endforeach()
endif()
