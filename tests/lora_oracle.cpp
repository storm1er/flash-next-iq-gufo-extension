// Real-weight LoRA oracle: merges an adapter into the stock trunk on the CPU
// with the production routine and compares every patched tensor with an
// offline merge of the same checkpoint. Reads only; skips (exit 77) when a
// file is absent.
//
//   flash_next_iq_lora_oracle STOCK.gguf ADAPTER.gguf MERGED.gguf [SCALE]
//
// STOCK and MERGED name the first shard of each split checkpoint. A Q8_0
// target must match the offline merge byte for byte. A Q6_K target is merged
// to Q8_0 here while the offline merge re-encodes Q6_K, so both are measured
// against the exact result instead (the dequantized stock row plus the
// update, in double): RMS and maximum absolute error per tensor and overall.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <gufo/ext/experimental/ggml_dequant.hpp>
#include <iostream>
#include <string>
#include <vector>

#include "lora.hpp"
#include "weights.hpp"

namespace fniq = gufo::models::flash_next_iq;
using gufo::core::GgmlType;

namespace {

struct Error {
  double squares{0};
  double max{0};
  std::uint64_t count{0};
  std::uint64_t at{0};  ///< Element index of `max` in its tensor.
  void Add(double difference) {
    squares += difference * difference;
    if (std::fabs(difference) > max) {
      max = std::fabs(difference);
      at = count;
    }
    ++count;
  }
  void Add(const Error& other) {
    squares += other.squares;
    max = std::max(max, other.max);
    count += other.count;
  }
  [[nodiscard]] double Rms() const {
    return count == 0 ? 0 : std::sqrt(squares / static_cast<double>(count));
  }
};

void Print(const std::string& name, const Error& update, const Error& ours,
           const Error& reference) {
  std::printf("%-28s update_rms %.4e | q8_0 rms %.4e max %.4e | "
              "reference q6_k rms %.4e max %.4e\n",
              name.c_str(), update.Rms(), ours.Rms(), ours.max,
              reference.Rms(), reference.max);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4 || argc > 5) {
    std::cerr << "usage: " << argv[0]
              << " STOCK.gguf ADAPTER.gguf MERGED.gguf [SCALE]\n";
    return 2;
  }
  for (int i = 1; i < 4; ++i) {
    if (!std::filesystem::exists(argv[i])) {
      std::cout << "SKIP: " << argv[i] << " is absent\n";
      return 77;
    }
  }
  const float scale = argc == 5 ? std::strtof(argv[4], nullptr) : 1.0F;
  std::string error;
  const gufo::core::GgufOpenOptions unsized{.allow_unsized_tensors = true};
  const auto stock =
      gufo::core::GgufReader::OpenFile(argv[1], &error, unsized);
  const auto merged =
      stock ? gufo::core::GgufReader::OpenFile(argv[3], &error, unsized)
            : nullptr;
  if (!merged) {
    std::cerr << "open failed: " << error << '\n';
    return 1;
  }
  const auto weights = fniq::ModelWeights::Bind(*stock, &error);
  if (!weights) {
    std::cerr << "bind failed: " << error << '\n';
    return 1;
  }
  const auto targets = fniq::LoraTargets(*weights);
  const auto adapter = fniq::LoraAdapter::Open(
      argv[2], scale,
      stock->GetMetadataString("general.architecture").value_or(""), targets,
      &error);
  if (!adapter) {
    std::cerr << error << '\n';
    return 1;
  }
  std::cout << "adapter sha256 " << adapter->sha256() << ", " << adapter->size()
            << " patches, identity " << adapter->CacheIdentity() << '\n';

  std::uint64_t q8_tensors = 0, q8_bytes = 0, q8_mismatched = 0, changed = 0;
  std::uint64_t q6_tensors = 0;
  Error q6_update, q6_ours, q6_reference;
  std::uint64_t q6_out_of_bound = 0;
  std::chrono::duration<double> merge_time{};
  std::vector<std::uint8_t> buffer;
  std::vector<float> row(0), mine(0), theirs(0);
  for (const auto& target : targets) {
    const auto* patch = adapter->Find(target.name);
    if (patch == nullptr) {
      continue;
    }
    const auto* reference = merged->FindTensor(target.name);
    if (reference == nullptr || reference->type != target.type ||
        reference->size_bytes != target.SizeBytes()) {
      std::cerr << target.name << ": missing or reshaped in the merged file\n";
      return 1;
    }
    const bool q6_k = target.type == GgmlType::kQ6_K;
    const std::size_t row_bytes = target.RowBytes();
    const std::size_t out_bytes = patch->MergedRowBytes();
    const std::uint64_t chunk =
        std::max<std::uint64_t>(1, (4 << 20) / out_bytes);
    const std::size_t cols = target.cols;
    row.resize(cols);
    mine.resize(cols);
    theirs.resize(cols);
    const double multiplier = adapter->Multiplier(*patch);
    std::uint64_t tensor_mismatch = 0;
    Error update, ours, offline;
    // At our largest error: the other encoding's error there and the
    // 32-element block's largest exact magnitude.
    double offline_at_ours = 0, block_amax = 0;
    std::vector<double> exact_row(cols);
    for (std::uint64_t first = 0; first < target.rows; first += chunk) {
      const std::uint64_t count = std::min(chunk, target.rows - first);
      buffer.resize(count * out_bytes);
      const auto start = std::chrono::steady_clock::now();
      if (!adapter->MergeRows(*patch, first, count, buffer.data(), &error)) {
        std::cerr << error << '\n';
        return 1;
      }
      merge_time += std::chrono::steady_clock::now() - start;
      const auto* expected =
          static_cast<const std::uint8_t*>(reference->data) + first * row_bytes;
      const auto* original =
          static_cast<const std::uint8_t*>(target.data) + first * row_bytes;
      if (!q6_k) {
        for (std::size_t i = 0; i < buffer.size(); ++i) {
          tensor_mismatch += buffer[i] != expected[i];
          changed += buffer[i] != original[i];
        }
        continue;
      }
      for (std::uint64_t i = 0; i < count; ++i) {
        const std::uint64_t r = first + i;
        gufo::quant::DequantizeQ6_K(original + i * row_bytes, row.data(), cols);
        gufo::quant::DequantizeQ8_0(buffer.data() + i * out_bytes, mine.data(),
                                    cols);
        gufo::quant::DequantizeQ6_K(expected + i * row_bytes, theirs.data(),
                                    cols);
        const float* b = patch->b + r * patch->rank;
        for (std::size_t c = 0; c < cols; ++c) {
          double delta = 0;
          for (std::uint32_t k = 0; k < patch->rank; ++k)
            delta += static_cast<double>(b[k]) * patch->a[k * cols + c];
          delta *= multiplier;
          const double exact = static_cast<double>(row[c]) + delta;
          exact_row[c] = exact;
          update.Add(delta);
          const double previous = ours.max;
          ours.Add(static_cast<double>(mine[c]) - exact);
          offline.Add(static_cast<double>(theirs[c]) - exact);
          if (ours.max != previous) {
            offline_at_ours = static_cast<double>(theirs[c]) - exact;
            block_amax = -1;  // filled once the row is complete
          }
        }
        if (block_amax < 0) {
          const std::size_t c0 = ours.at % cols / 32 * 32;
          block_amax = 0;
          for (std::size_t c = c0; c < c0 + 32; ++c)
            block_amax = std::max(block_amax, std::fabs(exact_row[c]));
        }
        // Pass criterion: every value within Q8_0's rounding bound for its
        // block, half a step (amax / 254) plus the half-float scale's own
        // rounding (below amax / 1024).
        for (std::size_t c0 = 0; c0 < cols; c0 += 32) {
          double amax = 0;
          for (std::size_t c = c0; c < c0 + 32; ++c)
            amax = std::max(amax, std::fabs(exact_row[c]));
          const double bound = amax * (1.0 / 254.0 + 1.0 / 1024.0);
          for (std::size_t c = c0; c < c0 + 32; ++c)
            q6_out_of_bound += std::fabs(mine[c] - exact_row[c]) > bound;
        }
      }
    }
    if (q6_k) {
      Print(std::string(target.name), update, ours, offline);
      std::printf("%-28s   q8_0 max at row %llu col %llu (block amax %.4e, "
                  "half step %.4e); q6_k error there %.4e\n",
                  "", static_cast<unsigned long long>(ours.at / cols),
                  static_cast<unsigned long long>(ours.at % cols), block_amax,
                  block_amax / 254.0, std::fabs(offline_at_ours));
      q6_update.Add(update);
      q6_ours.Add(ours);
      q6_reference.Add(offline);
      ++q6_tensors;
      continue;
    }
    if (tensor_mismatch != 0) {
      std::cout << "MISMATCH " << target.name << ": " << tensor_mismatch
                << " of " << target.SizeBytes() << " bytes\n";
    }
    ++q8_tensors;
    q8_bytes += target.SizeBytes();
    q8_mismatched += tensor_mismatch;
  }
  if (q6_tensors != 0) {
    Print("Q6_K targets overall", q6_update, q6_ours, q6_reference);
  }
  std::cout << q8_tensors << " Q8_0 tensors, " << q8_bytes
            << " bytes compared, " << q8_mismatched << " mismatching, "
            << changed << " bytes differ from stock; " << q6_tensors
            << " Q6_K tensors measured; merge " << merge_time.count()
            << " s\n";
  // The comparison with the reference's Q6_K encoding above is a
  // measurement; correctness is the Q8_0 rounding bound.
  if (q6_tensors != 0) {
    std::cout << "Q6_K targets: " << q6_out_of_bound
              << " values outside the Q8_0 rounding bound\n";
  }
  return q8_mismatched == 0 && q6_out_of_bound == 0 &&
                 q8_tensors + q6_tensors == adapter->size()
             ? 0
             : 1;
}
