#include <gufo/ext/generated/flash_next_iq_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>

#include "probe.hpp"
#include "runner.hpp"

namespace gufo::models::flash_next_iq {

#if defined(ENGINE_ENABLE_HIP)
int RunQwen38FlashNextBenchmark(
    const ext::TextBenchOptions& opt,
    const std::shared_ptr<const core::GgufReader>& reader,
    std::chrono::steady_clock::time_point model_load_start);

int RunFlashNextPrompt(const ext::TextPromptOptions& opt,
                       const core::GgufReader& reader,
                       std::chrono::steady_clock::time_point model_load_start);

int RunFlashNextChat(const ext::TextPromptOptions& opt,
                     const core::GgufReader& reader,
                     std::chrono::steady_clock::time_point model_load_start);
#endif

}  // namespace gufo::models::flash_next_iq

void flash_next_iq_provider_register(gufo::ext::Registry& registry) {
  registry.AddProvider(gufo::ext::generated::extension_flash_next_iq::
                           kProvider_flash_next_iq);
#if defined(ENGINE_ENABLE_HIP)
  registry.AddTextProvider({
      .id = "flash-next-iq",
      .probe = &gufo::models::flash_next_iq::ProbeFlashNext,
      .probe_gguf = &gufo::models::flash_next_iq::ProbeFlashNextGguf,
      .create = &gufo::models::flash_next_iq::CreateQwenFlashNextTextRunner,
      .bench_driver =
          &gufo::models::flash_next_iq::RunQwen38FlashNextBenchmark,
      .prompt_driver = &gufo::models::flash_next_iq::RunFlashNextPrompt,
      .chat_driver = &gufo::models::flash_next_iq::RunFlashNextChat,
      .sampling_defaults =
          &gufo::models::flash_next_iq::FlashNextSamplingDefaults,
      // The UD 3.1 UD-IQ4_XS keeps two layers' down experts in GGML Q2_0,
      // which core does not size: the provider binds and decodes it itself.
      .accepts_unsized_tensors = true,
  });
#endif
}
