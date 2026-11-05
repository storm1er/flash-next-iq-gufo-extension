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
  // This extension owns only the IQ-quantized routed-expert targets
  // (Unsloth UD-IQ4_XS: IQ3_S/IQ4_XS/IQ4_NL experts). Every other Flash-Next
  // layout, including UD-Q4_K_XL, belongs to the built-in flash-next provider.
  for (const auto& tensor : reader.GetTensors()) {
    const auto name = tensor.name;
    if (name.find("ffn_gate_exps.") == std::string_view::npos &&
        name.find("ffn_up_exps.") == std::string_view::npos &&
        name.find("ffn_down_exps.") == std::string_view::npos) {
      continue;
    }
    return tensor.type == core::GgmlType::kIQ3_S ||
           tensor.type == core::GgmlType::kIQ4_XS ||
           tensor.type == core::GgmlType::kIQ4_NL;
  }
  return false;
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
