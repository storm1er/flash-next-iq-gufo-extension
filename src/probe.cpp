#include "probe.hpp"

#include <filesystem>
#include <string>

namespace gufo::models::flash_next_iq {

bool ProbeFlashNextGguf(const core::GgufReader& reader) {
  const auto architecture =
      reader.GetMetadataString("general.architecture").value_or("");
  if (architecture != "qwen4exp") {
    return false;
  }
  // This extension owns checkpoints whose routed experts use IQ-family or
  // narrow-bit targets (Unsloth UD: IQ3_S/IQ4_XS/IQ4_NL, and the 2026-11
  // re-release mixing IQ3_XXS/IQ2_S + Q5_0/IQ4_NL/Q2_0). Every layout whose
  // experts use the K-quant/Q5_1/Q8_0 targets — including UD-Q4_K_XL —
  // belongs to the built-in flash-next provider.
  static constexpr core::GgmlType kOwned[] = {
      core::GgmlType::kIQ3_S,   core::GgmlType::kIQ4_XS, core::GgmlType::kIQ4_NL,
      core::GgmlType::kIQ3_XXS, core::GgmlType::kIQ2_S,  core::GgmlType::kQ5_0,
      core::GgmlType::kQ2_0};
  static constexpr core::GgmlType kBuiltin[] = {
      core::GgmlType::kQ4_K, core::GgmlType::kQ5_K, core::GgmlType::kQ6_K,
      core::GgmlType::kQ5_1, core::GgmlType::kQ8_0};
  auto in = [](const core::GgmlType* set, std::size_t n, core::GgmlType t) {
    for (std::size_t i = 0; i < n; ++i) {
      if (set[i] == t) {
        return true;
      }
    }
    return false;
  };
  bool saw_expert = false;
  for (const auto& tensor : reader.GetTensors()) {
    const auto name = tensor.name;
    if (name.find("ffn_gate_exps.") == std::string_view::npos &&
        name.find("ffn_up_exps.") == std::string_view::npos &&
        name.find("ffn_down_exps.") == std::string_view::npos) {
      continue;
    }
    if (in(kBuiltin, std::size(kBuiltin), tensor.type)) {
      return false;
    }
    if (!in(kOwned, std::size(kOwned), tensor.type)) {
      return false;
    }
    saw_expert = true;
  }
  return saw_expert;
}

std::optional<sampling::TextSamplingDefaults> FlashNextSamplingDefaults() {
  return sampling::PresetDefaults(sampling::TextModelPreset::kQwen38);
}

std::optional<sampling::TextSamplingDefaults> FlashNextSamplingDefaults(
    const core::GgufReader&) {
  return FlashNextSamplingDefaults();
}

bool ProbeFlashNext(std::string_view model_root) {
  try {
    std::filesystem::path p(model_root);
    if (!std::filesystem::exists(p)) {
      return false;
    }
    if (std::filesystem::is_regular_file(p)) {
      std::string err;
      auto reader = core::GgufReader::OpenFile(p.string(), &err);
      if (!reader)
        return false;
      return ProbeFlashNextGguf(*reader);
    }
    return false;
  } catch (...) {
    return false;
  }
}

}  // namespace gufo::models::flash_next_iq
