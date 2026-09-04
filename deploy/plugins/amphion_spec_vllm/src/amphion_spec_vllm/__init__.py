"""Register the bundled AmphionSPEC architecture with vLLM."""


def register() -> None:
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "AmphionASRForConditionalGeneration",
        "amphion_spec_vllm.amphion_asr:AmphionASRForVLLM",
    )
