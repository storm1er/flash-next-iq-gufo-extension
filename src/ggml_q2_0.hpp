#pragma once

// GGML Q2_0 (ggml-org/llama.cpp `block_q2_0`, GGML_TYPE_Q2_0 = 42), used by
// two layers' down experts of the UD 3.1 UD-IQ4_XS. Gufo core does not size
// this type: the extension opens its readers with `allow_unsized_tensors`,
// accepts such files (`TextProvider::accepts_unsized_tensors`) and sizes and
// decodes Q2_0 itself.

#include <cstddef>
#include <cstdint>
#include <cstring>

#include <gufo/ext/gguf_reader.hpp>

namespace gufo::models::flash_next_iq {

inline constexpr core::GgmlType kGgmlQ2_0 = static_cast<core::GgmlType>(42);

/// Elements and bytes of one block: an F16 scale and 16 bytes of 2-bit codes.
inline constexpr std::size_t kQ2_0BlockElements = 64;
inline constexpr std::size_t kQ2_0BlockBytes = 18;

/// IEEE binary16 to binary32, subnormals included.
[[nodiscard]] inline float HalfBitsToFloat(std::uint16_t h) noexcept {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000U) << 16U;
  std::uint32_t exponent = (h >> 10U) & 0x1FU;
  std::uint32_t mantissa = h & 0x3FFU;
  std::uint32_t bits = 0;
  if (exponent == 0) {
    if (mantissa != 0) {
      // Normalize the subnormal.
      exponent = 127 - 15 + 1;
      while ((mantissa & 0x400U) == 0) {
        mantissa <<= 1U;
        --exponent;
      }
      mantissa &= 0x3FFU;
      bits = sign | (exponent << 23U) | (mantissa << 13U);
    } else {
      bits = sign;
    }
  } else if (exponent == 0x1F) {
    bits = sign | 0x7F800000U | (mantissa << 13U);
  } else {
    bits = sign | ((exponent + 127 - 15) << 23U) | (mantissa << 13U);
  }
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

/// ggml's dequantize_row_q2_0: element j of a block is bits 2 (j % 4) of
/// code byte j / 4, and codes 0-3 are the values -1, 0, +1, +2 times d.
inline void DequantizeQ2_0(const void* src, float* dst, std::size_t k) {
  const auto* bytes = static_cast<const std::uint8_t*>(src);
  for (std::size_t b = 0; b < k / kQ2_0BlockElements; ++b) {
    const std::uint8_t* block = bytes + b * kQ2_0BlockBytes;
    std::uint16_t dh = 0;
    std::memcpy(&dh, block, sizeof(dh));
    const float d = HalfBitsToFloat(dh);
    for (std::size_t j = 0; j < kQ2_0BlockElements; ++j) {
      const auto q = static_cast<int>((block[2 + j / 4] >> ((j % 4) * 2)) & 3U);
      dst[b * kQ2_0BlockElements + j] = static_cast<float>(q - 1) * d;
    }
  }
}

}  // namespace gufo::models::flash_next_iq
