{
  description = "Qwen3.8 Flash-Next IQ-quant community source extension for Gufo";
  # Pinned to the Gufo revision this extension is ported against (v0.30.0+9).
  # The extension must be built with a focused selection because it vendors
  # Gufo's private Qwen family layer under its original namespaces:
  #   nix build --option sandbox relaxed  # then pass through CMake:
  #   cmake ... -DGUFO_EXTENSIONS=flash_next_iq  (see README.md)
  inputs.gufo.url = "github:gufo-org/gufo/2c97b72d8671ddd0c74ae7d517b8cef4ebb78871";
  outputs = { self, gufo }: {
    packages.x86_64-linux.default =
      gufo.packages.x86_64-linux.default.override {
        extensions = [ self.outPath ];
        extensionsFromLock = false;
      };
  };
}
