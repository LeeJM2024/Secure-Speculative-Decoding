"""Speculative decoding engines and device-independent verification rules."""


def __getattr__(name):
    if name == "CUDA_ARCH":
        from securesd.paths import configure_cuda_arch

        return configure_cuda_arch()
    raise AttributeError(name)
