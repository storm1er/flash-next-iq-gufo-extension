// LoRA merge checks specific to this extension: Q6_K targets (the UD 3.1
// attention output) merged and re-encoded as Q8_0, which target formats an
// adapter may patch, and the cache identity's adapters line. The Q8_0
// encoder and Q8_0 merge arithmetic are ported unchanged from Gufo's
// qwen38_flash_next and covered by its lora_test.
#include "lora.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <gufo/ext/experimental/ggml_dequant.hpp>
#include <iostream>
#include <string>
#include <vector>

#include "continuation_identity.hpp"

namespace fniq = gufo::models::flash_next_iq;
using gufo::core::GgmlType;

namespace {

void Check(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

// Minimal GGUF v3 writer for adapters: string and float32 metadata, F32
// tensors, 32-byte alignment.
class GgufWriter {
public:
  void String(const std::string& key, const std::string& value) {
    Name(key);
    U32(8);
    Name(value);
    ++kv_;
  }
  void Float(const std::string& key, float value) {
    Name(key);
    U32(6);
    Raw(&value, 4);
    ++kv_;
  }
  void Tensor(const std::string& name, std::vector<std::uint64_t> dims,
              const std::vector<float>& values) {
    tensors_.push_back({name, std::move(dims), values});
  }
  [[nodiscard]] std::vector<std::uint8_t> Build() const {
    std::vector<std::uint8_t> out;
    const auto put = [&](const void* p, std::size_t n) {
      const auto* b = static_cast<const std::uint8_t*>(p);
      out.insert(out.end(), b, b + n);
    };
    const std::uint32_t magic = 0x46554747;  // "GGUF"
    const std::uint32_t version = 3;
    const std::uint64_t count = tensors_.size();
    put(&magic, 4);
    put(&version, 4);
    put(&count, 8);
    put(&kv_, 8);
    put(meta_.data(), meta_.size());
    std::uint64_t offset = 0;
    for (const auto& t : tensors_) {
      const std::uint64_t length = t.name.size();
      const auto dims = static_cast<std::uint32_t>(t.dims.size());
      const std::uint32_t f32 = 0;
      put(&length, 8);
      put(t.name.data(), t.name.size());
      put(&dims, 4);
      put(t.dims.data(), t.dims.size() * 8);
      put(&f32, 4);
      put(&offset, 8);
      offset += (t.values.size() * 4 + 31) / 32 * 32;
    }
    out.resize((out.size() + 31) / 32 * 32);
    for (const auto& t : tensors_) {
      put(t.values.data(), t.values.size() * 4);
      out.resize((out.size() + 31) / 32 * 32);
    }
    return out;
  }

private:
  struct Entry {
    std::string name;
    std::vector<std::uint64_t> dims;
    std::vector<float> values;
  };
  void Name(const std::string& text) {
    const std::uint64_t length = text.size();
    Raw(&length, 8);
    Raw(text.data(), text.size());
  }
  void U32(std::uint32_t value) { Raw(&value, 4); }
  void Raw(const void* p, std::size_t n) {
    const auto* b = static_cast<const std::uint8_t*>(p);
    meta_.insert(meta_.end(), b, b + n);
  }
  std::vector<std::uint8_t> meta_;
  std::uint64_t kv_{0};
  std::vector<Entry> tensors_;
};

// A: rank rows of `cols`, multiples of 1/4, zero in columns 16..31 of each
// 32-column block. B: small integers. Every update is exact in float32.
float A(std::uint32_t k, std::uint64_t col) {
  return col % 32 >= 16 ? 0.0F
                        : 0.25F * (static_cast<float>(col % 29) - 14.0F) *
                              (k == 0 ? 1.0F : -0.5F);
}
float B(std::uint32_t k, std::uint64_t row) {
  constexpr float kB[2][3] = {{2.0F, -1.0F, -3.0F}, {1.0F, 4.0F, -2.0F}};
  return kB[k][row];
}

std::vector<std::uint8_t> Adapter(const std::string& base, std::uint64_t cols,
                                  std::uint64_t rows, std::uint32_t rank,
                                  float alpha) {
  std::vector<float> a;
  std::vector<float> b;
  for (std::uint32_t k = 0; k < rank; ++k)
    for (std::uint64_t c = 0; c < cols; ++c)
      a.push_back(A(k, c));
  for (std::uint64_t r = 0; r < rows; ++r)
    for (std::uint32_t k = 0; k < rank; ++k)
      b.push_back(B(k, r));
  GgufWriter w;
  w.String("general.architecture", "qwen4exp");
  w.String("general.type", "adapter");
  w.String("adapter.type", "lora");
  w.Float("adapter.lora.alpha", alpha);
  w.Tensor(base + ".lora_a", {cols, rank}, a);
  w.Tensor(base + ".lora_b", {rank, rows}, b);
  return w.Build();
}

std::unique_ptr<fniq::LoraAdapter> Open(const std::vector<std::uint8_t>& bytes,
                                        std::span<const fniq::TensorRef> targets,
                                        float scale, std::string* error) {
  auto reader =
      gufo::core::GgufReader::OpenMemory(bytes.data(), bytes.size(), error);
  Check(reader != nullptr, "synthetic adapter does not parse: " + *error);
  return fniq::LoraAdapter::Open(std::move(reader), "test.gguf", scale,
                                 "qwen4exp", targets, error);
}

// A Q6_K matrix with d = 1 whose Q8_0 re-encoding keeps block scale 1:
// columns 0..15 of each 32-column block (sub-block scale 1) hold codes in
// [-20, 20]; columns 16..31 (sub-block scale 127) hold -127, 0 or 127, and
// column 31 holds 127, the block maximum.
struct Q6KMatrix {
  static constexpr std::size_t kBlockBytes = 210;
  std::uint64_t cols;
  std::uint64_t rows;
  std::vector<std::uint8_t> bytes;
  fniq::TensorRef ref;

  Q6KMatrix(const char* name, std::uint64_t c, std::uint64_t r)
      : cols(c), rows(r), bytes(c / 256 * kBlockBytes * r) {
    for (std::uint64_t row = 0; row < rows; ++row)
      for (std::uint64_t block = 0; block < cols / 256; ++block)
        Pack(row, block);
    ref.data = bytes.data();
    ref.type = GgmlType::kQ6_K;
    ref.cols = cols;
    ref.rows = rows;
    ref.name = name;
  }
  static int Value(std::uint64_t row, std::uint64_t col) {
    if (col % 32 == 31)
      return 127;
    if (col % 32 >= 16)
      return 127 * (static_cast<int>((row + col) % 3) - 1);
    return static_cast<int>((row * 7 + col * 3) % 41) - 20;
  }
  // llama.cpp block_q6_K: ql[128], qh[64], scales[16], d (half). Element j
  // uses scale j / 16 and the 6-bit code q - 32.
  void Pack(std::uint64_t row, std::uint64_t block) {
    std::array<int, 256> q{};
    for (int j = 0; j < 256; ++j) {
      const int scale = j % 32 >= 16 ? 127 : 1;
      q[j] = Value(row, block * 256 + j) / scale + 32;
    }
    auto* out = bytes.data() + (row * (cols / 256) + block) * kBlockBytes;
    std::uint8_t* ql = out;
    std::uint8_t* qh = out + 128;
    auto* scales = reinterpret_cast<std::int8_t*>(out + 192);
    for (int half = 0; half < 2; ++half) {
      for (int l = 0; l < 32; ++l) {
        const int base = half * 128 + l;
        const int q0 = q[base], q1 = q[base + 32], q2 = q[base + 64],
                  q3 = q[base + 96];
        ql[half * 64 + l] =
            static_cast<std::uint8_t>((q0 & 0xF) | ((q2 & 0xF) << 4));
        ql[half * 64 + l + 32] =
            static_cast<std::uint8_t>((q1 & 0xF) | ((q3 & 0xF) << 4));
        qh[half * 32 + l] = static_cast<std::uint8_t>(
            (q0 >> 4) | ((q1 >> 4) << 2) | ((q2 >> 4) << 4) | ((q3 >> 4) << 6));
      }
    }
    for (int s = 0; s < 16; ++s)
      scales[s] = static_cast<std::int8_t>(s % 2 == 1 ? 127 : 1);
    const std::uint16_t one = 0x3c00;
    std::memcpy(out + 208, &one, 2);
  }
};

// Expected Q8_0 code in exact arithmetic: round-half-away of the stock value
// plus the update, all multiples of 1/16 here.
int Expected(std::uint64_t row, std::uint64_t col, std::uint32_t rank,
             double multiplier) {
  double update = 0;
  for (std::uint32_t k = 0; k < rank; ++k)
    update += static_cast<double>(B(k, row)) * A(k, col);
  const long sixteenths = Q6KMatrix::Value(row, col) * 16L +
                          std::lround(update * multiplier * 16.0);
  const long magnitude = (std::labs(sixteenths) + 8) / 16;
  return static_cast<int>(sixteenths < 0 ? -magnitude : magnitude);
}

void CheckQ6KPacking() {
  Q6KMatrix m("blk.3.attn_output.weight", 512, 3);
  std::vector<float> values(512);
  for (std::uint64_t row = 0; row < 3; ++row) {
    gufo::quant::DequantizeQ6_K(m.bytes.data() + row * m.ref.RowBytes(),
                                values.data(), 512);
    for (std::uint64_t col = 0; col < 512; ++col)
      Check(values[col] == static_cast<float>(Q6KMatrix::Value(row, col)),
            "test Q6_K packing differs from the dequantizer at row " +
                std::to_string(row) + " col " + std::to_string(col));
  }
}

void CheckQ6KMerge(std::uint32_t rank, float alpha, float scale) {
  Q6KMatrix m("blk.3.attn_output.weight", 512, 3);
  const std::vector<fniq::TensorRef> targets{m.ref};
  std::string error;
  const auto bytes = Adapter("blk.3.attn_output.weight", 512, 3, rank, alpha);
  const auto adapter = Open(bytes, targets, scale, &error);
  Check(adapter != nullptr, "Q6_K target rejected: " + error);
  const auto* patch = adapter->Find("blk.3.attn_output.weight");
  Check(patch != nullptr && patch->rank == rank, "patch not found");
  // Q6_K rows of 512 are 420 bytes; their Q8_0 encoding is 544.
  Check(m.ref.RowBytes() == 420 && patch->MergedRowBytes() == 544 &&
            patch->MergedBytes() == 3 * 544,
        "merged Q8_0 size of a Q6_K target");
  // Rows one and two only, as a chunked upload asks for them.
  std::vector<std::uint8_t> out(2 * patch->MergedRowBytes());
  Check(adapter->MergeRows(*patch, 1, 2, out.data(), &error), error);
  const double multiplier = static_cast<double>(scale) * alpha / rank;
  for (std::uint64_t row = 1; row < 3; ++row) {
    for (std::uint64_t col = 0; col < 512; ++col) {
      const auto* block = out.data() + ((row - 1) * 16 + col / 32) * 34;
      Check(block[0] == 0x00 && block[1] == 0x3c,
            "Q8_0 block scale is not 1 at row " + std::to_string(row));
      const auto code = static_cast<std::int8_t>(block[2 + col % 32]);
      const int expected = Expected(row, col, rank, multiplier);
      Check(code == expected,
            "Q6_K merge rank " + std::to_string(rank) + " scale " +
                std::to_string(scale) + " row " + std::to_string(row) +
                " col " + std::to_string(col) + ": " + std::to_string(code) +
                " vs " + std::to_string(expected));
    }
  }
}

void CheckTargets() {
  Q6KMatrix q6("blk.3.attn_output.weight", 256, 3);
  Q6KMatrix stacked("blk.4.ffn_down_shexp.weight", 256, 3);
  stacked.ref.experts = 2;
  Q6KMatrix bf16("blk.5.ssm_out.weight", 256, 3);
  bf16.ref.type = GgmlType::kBF16;
  const std::vector<fniq::TensorRef> targets{q6.ref, stacked.ref, bf16.ref};
  std::string error;
  Check(Open(Adapter("blk.3.attn_output.weight", 256, 3, 1, 1.0F), targets,
             1.0F, &error) != nullptr,
        "Q6_K target rejected: " + error);
  const auto reject = [&](const std::string& base, const std::string& want) {
    std::string message;
    Check(Open(Adapter(base, 256, 3, 1, 1.0F), targets, 1.0F, &message) ==
                  nullptr &&
              message.find(want) != std::string::npos,
          "error '" + message + "' does not name '" + want + "'");
  };
  reject("blk.4.ffn_down_shexp.weight",
         "target blk.4.ffn_down_shexp.weight is Q6_K in the model, expected "
         "Q8_0 or Q6_K");
  reject("blk.5.ssm_out.weight",
         "target blk.5.ssm_out.weight is BF16 in the model, expected Q8_0 or "
         "Q6_K");
}

void CheckIdentity() {
  const std::string artifact(64, 'a');
  const auto stock = fniq::QwenFlashNextCompatibilityIdentity(
      artifact, "", false, 4096, 4, 1, "absolute-v1");
  const auto none = fniq::QwenFlashNextCompatibilityIdentity(
      artifact, "", false, 4096, 4, 1, "absolute-v1", "none");
  // The adapter's identity follows its file bytes and scale.
  Q6KMatrix m("blk.3.attn_output.weight", 512, 3);
  const std::vector<fniq::TensorRef> targets{m.ref};
  const auto identity = [&](const std::vector<std::uint8_t>& bytes,
                            float scale) {
    std::string error;
    const auto adapter = Open(bytes, targets, scale, &error);
    Check(adapter != nullptr, "adapter rejected: " + error);
    return adapter->CacheIdentity();
  };
  const auto bytes = Adapter("blk.3.attn_output.weight", 512, 3, 1, 1.0F);
  const auto copy = bytes;
  const auto other = Adapter("blk.3.attn_output.weight", 512, 3, 1, 2.0F);
  const std::string tag = identity(bytes, 1.0F);
  Check(tag == identity(copy, 1.0F), "equal adapter bytes differ in identity");
  Check(tag != identity(other, 1.0F), "different adapter bytes share one");
  Check(tag != identity(bytes, std::nextafter(1.0F, 2.0F)),
        "adjacent scales share an identity");
  Check(tag.starts_with("lora;merge=") &&
            tag.find('\n') == std::string::npos,
        "adapter identity is not one tagged line");
  const auto adapted = fniq::QwenFlashNextCompatibilityIdentity(
      artifact, "", false, 4096, 4, 1, "absolute-v1", tag);
  const std::string stock_text(stock.begin(), stock.end());
  const std::string adapted_text(adapted.begin(), adapted.end());
  Check(stock == none &&
            stock_text.find("\nadapters=none\n") != std::string::npos,
        "stock identity lost adapters=none");
  // Only the adapters line differs from the stock identity.
  std::string restored = adapted_text;
  const auto line = restored.find("\nadapters=" + tag + "\n");
  Check(line != std::string::npos, "adapter identity line missing");
  restored.replace(line, tag.size() + 11, "\nadapters=none\n");
  Check(restored == stock_text, "the adapter changed more than its line");
  for (const char* bad : {"", "a\nb"}) {
    bool threw = false;
    try {
      (void)fniq::QwenFlashNextCompatibilityIdentity(
          artifact, "", false, 4096, 4, 1, "absolute-v1", bad);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    Check(threw, "a malformed adapters line was accepted");
  }
}

}  // namespace

int main() {
  CheckQ6KPacking();
  CheckQ6KMerge(1, 1.0F, 1.0F);
  CheckQ6KMerge(2, 1.0F, 1.0F);
  CheckQ6KMerge(1, 1.0F, -1.0F);
  CheckQ6KMerge(2, 4.0F, 0.5F);
  CheckTargets();
  CheckIdentity();
  std::cout << "flash_next_iq LoRA checks passed.\n";
}
