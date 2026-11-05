#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "gufo/ext/text.hpp"
#include "rope_scaling.hpp"

namespace gufo::models::flash_next_iq {

[[nodiscard]] std::size_t FlashNextBenchContext(
    const ext::TextBenchOptions& options);

[[nodiscard]] bool ResolveFlashNextBenchYarn(
    const ext::TextBenchOptions& options, std::uint32_t native,
    std::optional<RopeScaling>* scaling, std::string* error);

}  // namespace gufo::models::flash_next_iq

namespace gufo::cli {
using models::flash_next_iq::FlashNextBenchContext;
using models::flash_next_iq::ResolveFlashNextBenchYarn;
}  // namespace gufo::cli
