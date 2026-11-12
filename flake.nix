{
  description = "Qwen3.8 Flash-Next IQ-quant community source extension for Gufo";
  # Pinned to the Gufo release this extension is built and tested against
  # (v0.34.0, extension API 0.3); update the pin after the checks in README.md.
  inputs.gufo.url = "github:gufo-org/gufo/2845939b2a7d61243d407c8816c6d42652512e50";
  outputs = { self, gufo }: {
    packages.x86_64-linux.default =
      gufo.packages.x86_64-linux.default.override {
        extensions = [ self ];
      };
  };
}
