"""vLLM-compatible AmphionASR model implementation.

Adapts the HuggingFace ``AmphionASRForConditionalGeneration`` to work
with vLLM's PagedAttention, continuous batching, and multimodal
framework.  The architecture mirrors Ultravox in vLLM:

    Qwen3ASRAudioEncoder (tower) -> Projector -> Qwen LLM (PagedAttention)

The audio encoder runs outside vLLM's KV-cache system.  Only the LLM
backbone uses PagedAttention, loaded via ``init_vllm_registered_model``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Any, Literal, TypeAlias

import torch
from torch import nn
from transformers import BatchFeature, PretrainedConfig
from transformers.models.whisper import WhisperFeatureExtractor
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    NestedTensors,
)
from vllm.multimodal.parse import MultiModalDataDict, MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

_AUDIO_PLACEHOLDER_TOKEN = "<speech>"
_AUDIO_PLACEHOLDER_FULL = "<start_speech><speech><end_speech>"
_SAMPLING_RATE = 16_000


# ---------------------------------------------------------------------------
# Pure-PyTorch SwooshR activation (no k2 dependency)
# ---------------------------------------------------------------------------


class SwooshR(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        return torch.logaddexp(zero, x - 1.0) - 0.08 * x - 0.313261687


# ---------------------------------------------------------------------------
# Multi-modal projector (frame concatenation + Linear-SwooshR-Linear)
# ---------------------------------------------------------------------------


class AmphionASRMultiModalProjector(nn.Module):
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        proj_cfg = config.projector_config
        if isinstance(proj_cfg, dict):
            from types import SimpleNamespace

            proj_cfg = SimpleNamespace(**proj_cfg)
        self.downsample_rate = proj_cfg.downsample_rate
        self.proj = nn.Sequential(
            nn.Dropout(proj_cfg.dropout),
            nn.Linear(proj_cfg.encoder_dim * self.downsample_rate, proj_cfg.llm_dim),
            SwooshR(),
            nn.Linear(proj_cfg.llm_dim, proj_cfg.llm_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, feat_dim = x.size()
        num_frames_to_discard = seq_len % self.downsample_rate
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)
        x = x.contiguous().view(
            batch_size,
            seq_len // self.downsample_rate,
            feat_dim * self.downsample_rate,
        )
        return self.proj(x)


# ---------------------------------------------------------------------------
# TensorSchema definitions for vLLM multimodal inputs
# ---------------------------------------------------------------------------


class AmphionASRAudioFeatureInputs(TensorSchema):
    type: Literal["audio_features"]
    data: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("b", "t", "f"),
    ]
    lens: Annotated[torch.Tensor, TensorShape("b")]


class AmphionASRAudioEmbeddingInputs(TensorSchema):
    type: Literal["audio_embeds"]
    data: Annotated[
        torch.Tensor | list[torch.Tensor],
        TensorShape("b", "t", "h"),
    ]


AmphionASRAudioInputs: TypeAlias = AmphionASRAudioFeatureInputs | AmphionASRAudioEmbeddingInputs


# ---------------------------------------------------------------------------
# Processing info / dummy inputs / multimodal processor
# ---------------------------------------------------------------------------


class AmphionASRProcessingInfo(BaseProcessingInfo):
    def _get_feat_type(self) -> str:
        cfg = self.ctx.model_config.hf_config
        return getattr(cfg, "feature_extractor_type", "whisper")

    def get_feature_extractor(self, **kwargs: object) -> WhisperFeatureExtractor | None:
        if self._get_feat_type() == "kaldi_fbank":
            return None
        return WhisperFeatureExtractor.from_pretrained(
            self.ctx.model_config.model,
        )

    def get_data_parser(self):
        return MultiModalDataParser(
            target_sr=_SAMPLING_RATE,
            target_channels=1,
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # 2 supports target-speaker ASR (enrollment + mixed); 1 for plain ASR.
        return {"audio": 2}


class AmphionASRDummyInputsBuilder(
    BaseDummyInputsBuilder[AmphionASRProcessingInfo],
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        return _AUDIO_PLACEHOLDER_TOKEN * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        audio_len = 30 * _SAMPLING_RATE
        audio_overrides = mm_options.get("audio")
        return {
            "audio": self._get_dummy_audios(
                length=audio_len,
                num_audios=num_audios,
                overrides=audio_overrides,
            ),
        }


class AmphionASRMultiModalProcessor(
    BaseMultiModalProcessor[AmphionASRProcessingInfo],
):
    @staticmethod
    def _kaldi_fbank(audio_data, sr: int = 16000) -> torch.Tensor:
        """Extract Kaldi-compatible log-mel fbank features (T, F)."""
        import numpy as np
        import torchaudio

        if isinstance(audio_data, np.ndarray):
            audio_data = torch.from_numpy(audio_data).float()
        if audio_data.dim() == 1:
            audio_data = audio_data.unsqueeze(0)

        return torchaudio.compliance.kaldi.fbank(
            audio_data,
            sample_frequency=float(sr),
            num_mel_bins=80,
            frame_length=25.0,
            frame_shift=10.0,
            preemphasis_coefficient=0.97,
            window_type="povey",
            dither=0.0,
            snip_edges=False,
            energy_floor=1e-10,
            raw_energy=True,
            use_energy=False,
            low_freq=20.0,
            high_freq=-400.0,
            remove_dc_offset=True,
        )

    def _extract_features_single(self, audio_data, sr: int, feat_type: str):
        """Extract features for a single audio item.

        Returns (feats_tf, length) where feats_tf is (T, F) and length is T.
        """
        if feat_type == "kaldi_fbank":
            feats_tf = self._kaldi_fbank(audio_data, sr)
            return feats_tf, feats_tf.shape[0]

        feature_extractor = self.info.get_feature_extractor()
        feat_out = feature_extractor(
            audio_data,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        feats_btf = feat_out["input_features"].transpose(1, 2)  # (1, T, F)
        attn = feat_out["attention_mask"]
        length = int(attn.sum(dim=-1).long().item())
        return feats_btf.squeeze(0), length

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        tokenizer = self.info.get_tokenizer()

        audios = mm_data.get("audios", []) or []
        if not audios:
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        assert isinstance(audios, list) and len(audios) > 0
        feat_type = self.info._get_feat_type()

        feats_list = []
        lens_list = []
        for audio_item in audios:
            if isinstance(audio_item, tuple):
                audio_data, sr = audio_item
            else:
                audio_data, sr = audio_item, _SAMPLING_RATE
            feats_tf, length = self._extract_features_single(
                audio_data,
                sr,
                feat_type,
            )
            feats_list.append(feats_tf)
            lens_list.append(length)

        # Right-pad to T_max and stack -> (N, T_max, F)
        t_max = max(f.shape[0] for f in feats_list)
        f_dim = feats_list[0].shape[-1]
        padded = []
        for f in feats_list:
            if f.shape[0] < t_max:
                pad = f.new_zeros(t_max - f.shape[0], f_dim)
                padded.append(torch.cat([f, pad], dim=0))
            else:
                padded.append(f)
        input_features = torch.stack(padded, dim=0)  # (N, T_max, F)
        feature_lens = torch.tensor(lens_list, dtype=torch.long)

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

        return BatchFeature(
            dict(
                input_ids=[prompt_ids],
                audio_features=input_features,
                audio_lens=feature_lens,
            ),
            tensor_type="pt",
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            audio_features=MultiModalFieldConfig.batched("audio"),
            audio_lens=MultiModalFieldConfig.batched("audio"),
            audio_embeds=MultiModalFieldConfig.batched("audio"),
        )

    @staticmethod
    def _encoder_output_lengths(
        input_lengths: int,
        encoder_type: str,
    ) -> int:
        if encoder_type == "zipformer":
            return ((input_lengths - 7) // 2) // 2
        # qwen3asr / qwen3omni / qwen3omni_captioner
        leave = input_lengths % 100
        feat = (leave - 1) // 2 + 1
        return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        config = self.info.ctx.model_config.hf_config
        speech_token_id = config.default_speech_token_id

        out_mm_data = out_mm_kwargs.get_data()
        audio_lens = out_mm_data.get("audio_lens", torch.zeros(0))
        audio_embeds = out_mm_data.get("audio_embeds")

        proj_cfg = config.projector_config
        downsample_rate = (
            proj_cfg.get("downsample_rate", 1)
            if isinstance(proj_cfg, dict)
            else getattr(proj_cfg, "downsample_rate", 1)
        )
        encoder_type = getattr(config, "encoder_type", "qwen3asr")

        def get_embed_len(item_idx: int) -> int | None:
            if audio_embeds is None:
                return None
            if isinstance(audio_embeds, torch.Tensor):
                if audio_embeds.ndim == 3:
                    return int(audio_embeds[item_idx].shape[0])
                if audio_embeds.ndim == 2 and item_idx == 0:
                    return int(audio_embeds.shape[0])
                return None
            if isinstance(audio_embeds, (list, tuple)) and item_idx < len(audio_embeds):
                return int(audio_embeds[item_idx].shape[0])
            return None

        def get_replacement(item_idx: int):
            embed_len = get_embed_len(item_idx)
            if embed_len is not None:
                return [speech_token_id] * embed_len
            raw_mel_len = int(audio_lens[item_idx].item()) if audio_lens.numel() > 0 else 100
            encoder_out_len = self._encoder_output_lengths(
                raw_mel_len,
                encoder_type,
            )
            token_len = encoder_out_len // downsample_rate
            return [speech_token_id] * token_len

        return [
            PromptReplacement(
                modality="audio",
                target=_AUDIO_PLACEHOLDER_TOKEN,
                replacement=get_replacement,
            ),
        ]


# ---------------------------------------------------------------------------
# Main vLLM model
# ---------------------------------------------------------------------------


@MULTIMODAL_REGISTRY.register_processor(
    AmphionASRMultiModalProcessor,
    info=AmphionASRProcessingInfo,
    dummy_inputs=AmphionASRDummyInputsBuilder,
)
class AmphionASRForVLLM(nn.Module, SupportsMultiModal, SupportsPP):
    """AmphionASR model adapted for vLLM inference.

    Architecture: Qwen3ASRAudioEncoder -> Projector -> Qwen LLM
    """

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "audio_encoder.": "audio_tower.",
        },
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            return _AUDIO_PLACEHOLDER_FULL
        raise ValueError(f"Only audio modality is supported, got {modality}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self._vllm_config = vllm_config
        config = vllm_config.model_config.hf_config
        self.config = config

        self.configure_mm_token_handling(
            config.text_config.get("vocab_size", 151936)
            if isinstance(config.text_config, dict)
            else config.text_config.vocab_size,
            [config.default_speech_token_id],
        )

        # --- Audio tower (runs outside PagedAttention) ---
        self._build_audio_tower(config)

        # --- Projector ---
        self.multi_modal_projector = AmphionASRMultiModalProjector(config)

        # --- Language model (uses vLLM's PagedAttention) ---
        text_config = config.text_config
        if isinstance(text_config, dict):
            from transformers import AutoConfig

            text_config = AutoConfig.for_model(**text_config)

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=text_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        # --- Prompt embedding for special tokens ---
        hidden_size = text_config.hidden_size if hasattr(text_config, "hidden_size") else 2560
        self.prompt_embedding = nn.Embedding(
            config.num_prompt_tokens,
            hidden_size,
        )

        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

    # ------------------------------------------------------------------
    # Audio tower (uses unified AudioEncoderWrapper)
    # ------------------------------------------------------------------

    def _build_audio_tower(self, config: PretrainedConfig):
        import sys

        model_dir = self._vllm_config.model_config.model
        if model_dir not in sys.path:
            sys.path.insert(0, model_dir)
        from modeling_amphion_asr import build_audio_encoder

        self.audio_tower = build_audio_encoder(config)

    # ------------------------------------------------------------------
    # Encode audio
    # ------------------------------------------------------------------

    def _encode_audio(
        self,
        features: torch.Tensor,
        feature_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode audio features through the tower and projector.

        Returns (projected_features, token_lens) where token_lens
        accounts for the projector's downsample rate.
        """
        feature_lens = feature_lens.to(dtype=torch.long)

        if features.ndim == 3 and features.shape[1] in (80, 128):
            pass  # already (B, T, F)
        elif features.ndim == 3 and features.shape[-1] not in (80, 128):
            features = features.transpose(1, 2).contiguous()

        # Zipformer uses custom ops (positional embeddings, etc.) that
        # produce float32 intermediates internally, so we must run the
        # entire audio tower in float32 to avoid dtype mismatches, then
        # cast the output back to the projector's dtype.
        features = features.to(dtype=torch.float32)

        with torch.inference_mode():
            self.audio_tower.float()
            encoder_outs, encoder_lens = self.audio_tower(features, feature_lens)

        proj_dtype = next(self.multi_modal_projector.parameters()).dtype
        projected = self.multi_modal_projector(encoder_outs.to(dtype=proj_dtype))
        token_lens = encoder_lens // self.multi_modal_projector.downsample_rate

        return projected, token_lens

    # ------------------------------------------------------------------
    # Multimodal embedding interface
    # ------------------------------------------------------------------

    def _parse_audio_input(
        self,
        **kwargs: object,
    ) -> AmphionASRAudioInputs | None:
        audio_features = kwargs.pop("audio_features", None)
        audio_embeds = kwargs.pop("audio_embeds", None)
        audio_lens = kwargs.pop("audio_lens", None)

        if audio_features is None and audio_embeds is None:
            return None

        if audio_features is not None:
            if isinstance(audio_features, list) and len(audio_features) > 1:
                max_t = max(f.shape[0] for f in audio_features)
                padded = []
                for f in audio_features:
                    if f.shape[0] < max_t:
                        pad = f.new_zeros(max_t - f.shape[0], f.shape[-1])
                        padded.append(torch.cat([f, pad], dim=0))
                    else:
                        padded.append(f)
                audio_features = torch.stack(padded)
            return AmphionASRAudioFeatureInputs(
                type="audio_features",
                data=audio_features,
                lens=audio_lens,
            )

        if audio_embeds is not None:
            return AmphionASRAudioEmbeddingInputs(
                type="audio_embeds",
                data=audio_embeds,
            )

        raise AssertionError("Unreachable")

    def _process_audio_input(
        self,
        audio_input: AmphionASRAudioInputs,
    ) -> NestedTensors | tuple[torch.Tensor, ...]:
        if audio_input["type"] == "audio_embeds":
            return audio_input["data"]

        features = audio_input["data"]
        feature_lens = audio_input["lens"]

        if isinstance(features, list):
            features = torch.stack(features)
        if isinstance(feature_lens, list):
            feature_lens = torch.stack(feature_lens)

        projected, token_lens = self._encode_audio(features, feature_lens)

        max_len = projected.shape[1]
        indices = torch.arange(max_len, device=projected.device).expand(
            projected.shape[0],
            -1,
        )
        mask = indices < token_lens[:, None]
        flattened = projected[mask]

        embed_lens = token_lens.tolist()
        return flattened.split(embed_lens)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        audio_input = self._parse_audio_input(**kwargs)
        if audio_input is None:
            return []
        return self._process_audio_input(audio_input)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if multimodal_embeddings is None or is_multimodal is None:
            return super().embed_input_ids(input_ids)

        return super().embed_input_ids(
            input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    # ------------------------------------------------------------------
    # Module prefix mapping for multi-model weight loading
    # ------------------------------------------------------------------

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model.",
            connector="multi_modal_projector.",
            tower_model="audio_tower.",
        )

    # ------------------------------------------------------------------
    # Forward / compute_logits
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        language_model = self.language_model
        if hasattr(language_model, "language_model"):
            language_model = language_model.language_model

        hidden_states = language_model.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.language_model.compute_logits(hidden_states)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_prefixes=["audio_tower."],
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
