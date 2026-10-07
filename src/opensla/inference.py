"""Inference pipeline: input contract, checkpoint loading, sensor fusion, and prediction."""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .model import (
    ActionEvidenceIndex,
    DescriptorActionLabelRanker,
    MultimodalLLMWithActionHeads,
    action_group_need_indices,
    get_hidden_size,
)
from .sensors import (
    ChannelwiseNumericSignalMemory,
    FrozenWaveformEncoder,
    HTokenProjector,
    HierarchicalChannelwiseSignalMemory,
    ModalityTokenProjector,
    NumericEncoderConfig,
    NumericEventEncoder,
    SimpleWaveformPatchifier,
)

WAVEFORM_MODALITIES = {
    "clinical": (
        "II", "I", "III", "V", "AVR", "AVL", "AVF", "MCL", "MCL1", "ECG",
        "PLETH", "RESP", "ABP", "ART", "UAP", "CVP", "PAP", "ICP", "P1",
        "II_PLUS", "V_PLUS", "MCL1_PLUS", "PLETH_L", "PLETH_R", "CO2", "AOBP",
        "BAP", "LAP", "RAP", "UVP",
    ),
    "or": ("ECG", "ECG_II", "ECG_V5", "PLETH", "ART", "CVP", "FEM", "AWP", "CO2", "EEG1", "EEG2"),
    "cgm": ("CGM",),
}

# Preprocessed waveform slot geometry per domain: (max slots, samples per slot).
WAVEFORM_SLOTS = {"clinical": None, "or": None, "cgm": 8}
WAVEFORM_SAMPLES = {"clinical": 7500, "or": 7500, "cgm": 3}

SUMMARY_FEATURES = (
    "has_observed", "count_norm", "last_value_norm", "mean_value_norm",
    "min_value_norm", "max_value_norm", "slope_per_min_norm", "last_age_norm",
    "median_value_norm", "iqr_value_norm", "first_value_norm", "last_minus_first_norm",
)

# Generation sometimes ends in a run of the target header ("TARGET_TEXT"); it is cut, as in evaluation.
REPEATED_TARGET_HEADER = re.compile(r"(?:(?:TARGET_TEXT|TARGET|ARGET)\s*){2,}", re.IGNORECASE)

NUMERIC_KEYS = ("values", "measure_ids", "rel_time_min", "event_mask", "source_ids", "summary_features", "summary_measure_ids")


def numeric_summary_features(version: int = 1) -> tuple[str, ...]:
    if version not in (1, 2):
        raise ValueError(f"Unsupported numeric summary version: {version}")
    return SUMMARY_FEATURES[:8] if version == 1 else SUMMARY_FEATURES


@dataclass
class ModelConfig:
    """Architecture settings read from ``ckpt["args"]``; defaults match the training defaults."""

    domain: str
    fusion_mode: str  # "sp" = OpenSLA-B, "hcsm" = OpenSLA-H
    model_id: str
    dropout: float
    max_seq_len: int
    # language backbone
    bf16: bool = False
    fp16: bool = False
    attn_implementation: str | None = None
    trust_remote_code: bool = False
    lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: str = ""
    prompt_template: str = "plain"
    system_prompt: str = ""
    # waveform branch
    use_waveform_dino: bool = False
    waveform_encoder: str = "dino"
    waveform_reader_mode: str = "channelwise_unified"
    dino_output_mode: str = "patch"
    dino_ckpt: str | None = None
    max_waveform_lookback_min: int = 30
    max_waveform_channels: int = 0  # 0 = every modality of the domain
    # numeric branch
    use_numeric: bool = False
    numeric_encoder: str = "event_transformer"
    numeric_config: str | None = None
    numeric_hidden_dim: int = 128
    numeric_transformer_layers: int = 2
    numeric_transformer_heads: int = 4
    numeric_summary_version: int = 1
    max_numeric_events: int = 512
    max_numeric_lookback_min: int = 30
    max_signal_tokens: int = 4096
    # H memory
    hcsm_hidden_dim: int = 256
    hcsm_heads: int = 8
    hcsm_local_tokens_per_minute: int = 2
    hcsm_temporal_tokens_per_channel: int = 16
    hcsm_global_tokens: int = 64
    hcsm_dropout: float = 0.0
    numeric_hcsm: bool = False
    numeric_hcsm_architecture: str = "channelwise"
    numeric_hcsm_hidden_dim: int | None = None
    numeric_hcsm_group_minutes: float = 1.0
    numeric_hcsm_local_tokens_per_group: int | None = None
    numeric_hcsm_temporal_tokens_per_channel: int | None = None
    numeric_hcsm_global_tokens: int | None = None
    # action heads and caption decoder
    hierarchical_action_heads: bool = False
    caption_decoder_mode: str = "residual_v2"
    caption_decoder_action_conditioning: bool = True
    caption_profile_max_groups: int = 3
    caption_profile_facts_per_group: int = 3
    caption_profile_need_threshold: float = 0.5
    caption_profile_group_threshold: float = 0.15
    caption_profile_fact_threshold: float = 0.5
    caption_detail_selector: bool = False
    caption_waveform_detail_tokens_per_channel: int = 16
    caption_numeric_detail_tokens_per_measure: int = 2
    caption_waveform_detail_max_tokens: int = 0
    caption_numeric_detail_max_tokens: int = 0
    label_ranker_mode: str = "descriptor"
    label_ranking_temperature: float = 0.07

    @classmethod
    def from_checkpoint_args(cls, args: Mapping[str, Any], *, domain: str) -> "ModelConfig":
        """Build from the training args, ignoring training-only keys."""
        known = {f.name for f in fields(cls)}
        values = {key: value for key, value in dict(args).items() if key in known}
        values["domain"] = domain
        aliases = {"B": "sp", "H": "hcsm", "OpenSLA-B": "sp", "OpenSLA-H": "hcsm"}
        values["fusion_mode"] = aliases.get(values.get("fusion_mode"), values.get("fusion_mode"))
        config = cls(**values)
        config.validate(args)
        return config

    def validate(self, raw_args: Mapping[str, Any]) -> None:
        """Check that the checkpoint uses a supported configuration."""
        if self.domain not in WAVEFORM_MODALITIES:
            raise ValueError(f"Unknown domain: {self.domain}")
        if self.fusion_mode not in {"sp", "hcsm"} or raw_args.get("sensor_fusion_mode", "prefix") != "prefix":
            raise ValueError(f"Unsupported fusion mode: {self.fusion_mode}")
        if raw_args.get("lm_init_mode", "pretrained") != "pretrained" or raw_args.get("load_in_4bit", False):
            raise ValueError("Unsupported language backbone initialization.")
        if self.use_waveform_dino:
            if self.waveform_reader_mode != "channelwise_unified" or self.dino_output_mode != "patch":
                raise ValueError(f"Unsupported waveform reader/output mode: {self.waveform_reader_mode}/{self.dino_output_mode}")
            if self.waveform_encoder == "simple_patchifier":
                if self.domain != "cgm":
                    raise ValueError("The slot patchifier is only defined for the cgm domain.")
            elif self.waveform_encoder != "dino":
                raise ValueError(f"Unsupported waveform encoder: {self.waveform_encoder}")
            elif not self.dino_ckpt:
                raise ValueError("A waveform-encoder checkpoint path is required (dino_checkpoint).")
        if self.use_numeric:
            if self.numeric_encoder != "event_transformer":
                raise ValueError(f"Unsupported numeric encoder: {self.numeric_encoder}")
            if not self.numeric_config:
                raise ValueError("A numeric configuration path is required (numeric_config).")
        if self.fusion_mode == "hcsm" and not self.use_waveform_dino:
            raise ValueError("OpenSLA-H requires a waveform encoder.")
        if self.numeric_hcsm and self.numeric_hcsm_architecture != "channelwise":
            raise ValueError(f"Unsupported numeric memory: {self.numeric_hcsm_architecture}")
        if self.caption_detail_selector and self.fusion_mode != "hcsm":
            raise ValueError("The caption detail selector is only available with OpenSLA-H.")
        if self.caption_decoder_mode != "residual_v2" or not self.caption_decoder_action_conditioning:
            raise ValueError(f"Unsupported caption decoder: {self.caption_decoder_mode}")
        if self.label_ranker_mode != "descriptor":
            raise ValueError(f"Unsupported label ranker: {self.label_ranker_mode}")

    @property
    def lm_dtype(self) -> torch.dtype | None:
        return torch.bfloat16 if self.bf16 else torch.float16 if self.fp16 else None


@dataclass
class SensorBatch:
    """Preprocessed sensor inputs for one batch; see the README for the tensor contract."""

    waveform: torch.Tensor | None = None
    waveform_channel_mask: torch.Tensor | None = None
    waveform_modality_ids: torch.Tensor | None = None
    numeric: dict[str, torch.Tensor] | None = None

    def validate(self, batch_size: int, config: ModelConfig, numeric_vocab: NumericEncoderConfig | None) -> None:
        """Check shapes, dtypes, and value ranges against the checkpoint."""
        if self.waveform is None and self.numeric is None:
            raise ValueError("Provide at least one sensor modality.")
        if self.waveform is not None:
            if not config.use_waveform_dino:
                raise ValueError("This checkpoint has no waveform encoder.")
            if self.waveform.ndim != 4 or self.waveform.shape[0] != batch_size:
                raise ValueError("waveform must have shape [batch, slots, channels, samples].")
            b, slots, channels, samples = self.waveform.shape
            if self.waveform.dtype != torch.float32:
                raise ValueError("waveform must be float32.")
            max_slots = WAVEFORM_SLOTS[config.domain] or config.max_waveform_lookback_min
            if slots < 1 or slots > max_slots:
                raise ValueError("Waveform slot count is outside the checkpoint history window.")
            if samples != WAVEFORM_SAMPLES[config.domain]:
                raise ValueError(f"Expected {WAVEFORM_SAMPLES[config.domain]} preprocessed samples per slot, got {samples}.")
            modalities = WAVEFORM_MODALITIES[config.domain]
            if channels > (config.max_waveform_channels or len(modalities)):
                raise ValueError("Waveform channel count exceeds the checkpoint limit.")
            if self.waveform_channel_mask is None or self.waveform_channel_mask.shape != (b, slots, channels):
                raise ValueError("waveform_channel_mask must have shape [batch, slots, channels].")
            if self.waveform_channel_mask.dtype != torch.bool:
                raise ValueError("waveform_channel_mask must be boolean.")
            ids = self.waveform_modality_ids
            if ids is None or ids.shape != (b, channels) or ids.dtype != torch.long:
                raise ValueError("waveform_modality_ids must be int64 [batch, channels].")
            active = self.waveform_channel_mask.any(dim=1)
            if bool(((ids < 0) | (ids >= len(modalities)))[active].any()):
                raise ValueError("Observed waveform modality IDs are outside the domain vocabulary.")
            if not torch.isfinite(self.waveform).all():
                raise ValueError("Use finite normalized waveform values and mark unavailable slots in the mask.")
        if self.numeric is not None:
            if not config.use_numeric or numeric_vocab is None:
                raise ValueError("This checkpoint has no numeric encoder.")
            if set(self.numeric) != set(NUMERIC_KEYS):
                raise ValueError(f"numeric must contain exactly: {sorted(NUMERIC_KEYS)}")
            n = self.numeric
            shape = n["values"].shape
            if len(shape) != 2 or shape[0] != batch_size:
                raise ValueError("Numeric events must have shape [batch, events].")
            if shape[1] > config.max_numeric_events:
                raise ValueError("Numeric event count exceeds the checkpoint limit.")
            for key in ("measure_ids", "rel_time_min", "event_mask", "source_ids"):
                if n[key].shape != shape:
                    raise ValueError(f"numeric.{key} must match values shape.")
            if n["event_mask"].dtype != torch.bool:
                raise ValueError("numeric.event_mask must be boolean.")
            observed_times = n["rel_time_min"][n["event_mask"]]
            if bool(((observed_times > 0) | (observed_times < -float(config.max_numeric_lookback_min))).any()):
                raise ValueError("Observed numeric times must be inside the pre-decision history window.")
            features = n["summary_features"]
            if features.ndim != 3 or features.shape[0] != batch_size:
                raise ValueError("summary_features must have shape [batch, measurements, features].")
            if n["summary_measure_ids"].shape != features.shape[:2]:
                raise ValueError("summary_measure_ids must match the summary grid.")
            if features.shape[-1] != len(numeric_summary_features(config.numeric_summary_version)):
                raise ValueError("Numeric summary feature width does not match this checkpoint.")
            for key, limit in (("measure_ids", numeric_vocab.num_measures), ("summary_measure_ids", numeric_vocab.num_measures),
                               ("source_ids", numeric_vocab.num_sources)):
                if n[key].dtype != torch.long:
                    raise ValueError(f"numeric.{key} must be int64.")
                if bool(((n[key] < 0) | (n[key] >= limit)).any()):
                    raise ValueError(f"numeric.{key} contains an ID outside the checkpoint vocabulary.")
            for key in ("values", "rel_time_min", "summary_features"):
                if not torch.isfinite(n[key]).all():
                    raise ValueError(f"numeric.{key} must contain finite values.")


@dataclass
class ActionLabelIndex:
    """Fine action labels grouped under the coarse action categories."""

    label_keys: list[str]  # "<category>|<fine label>"
    label_texts: list[str]  # description used to embed each label
    group_to_label_ids: list[list[int]]

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> "ActionLabelIndex":
        return cls(
            label_keys=[str(x) for x in payload["label_keys"]],
            label_texts=[str(x) for x in payload["label_texts"]],
            group_to_label_ids=[[int(x) for x in ids] for ids in payload["group_to_label_ids"]],
        )


BATCH_KEYS = {"domain", "sample_ids", "input_text", "sensors"}


def load_prepared_batch(path: str | Path, *, domain: str) -> dict[str, Any]:
    """Load a preprocessed ``.pt`` batch and check its top-level contract."""
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) - BATCH_KEYS:
        raise ValueError(f"A prepared batch contains only {sorted(BATCH_KEYS)}.")
    if payload["domain"] != domain:
        raise ValueError(f"Batch domain {payload['domain']!r} does not match the model domain {domain!r}.")
    texts = payload["input_text"]
    if not isinstance(texts, list) or not all(isinstance(x, str) for x in texts):
        raise ValueError("input_text must be a list of strings.")
    ids = payload.get("sample_ids", [str(i) for i in range(len(texts))])
    if len(ids) != len(texts) or len(set(ids)) != len(ids):
        raise ValueError("sample_ids must be unique and match input_text length.")
    payload["sample_ids"] = [str(x) for x in ids]
    payload["sensors"] = SensorBatch(**payload["sensors"])
    return payload


def load_action_group_types(value: str | Path | Mapping[str, str] | None) -> dict[str, str] | None:
    """Accept an inline mapping or a JSON file (either a bare mapping or one with an ``action_group_types`` key)."""
    if value is None or isinstance(value, Mapping):
        return None if value is None else {str(k): str(v) for k, v in value.items()}
    payload = json.loads(Path(value).read_text())
    payload = payload.get("action_group_types", payload)
    return {str(k): str(v) for k, v in payload.items()}


def make_synthetic_batch(model: "OpenSLA", *, batch_size: int = 2, seed: int = 0) -> dict[str, Any]:
    """Random batch with the shapes the checkpoint expects, for trying the pipeline without data."""
    config = model.config
    generator = torch.Generator().manual_seed(int(seed))
    modalities = WAVEFORM_MODALITIES[config.domain]
    sensors: dict[str, Any] = {}
    if config.use_waveform_dino:
        slots = min(2, WAVEFORM_SLOTS[config.domain] or config.max_waveform_lookback_min)
        channels = min(2, len(modalities))
        sensors["waveform"] = torch.randn((batch_size, slots, channels, WAVEFORM_SAMPLES[config.domain]), generator=generator)
        sensors["waveform_channel_mask"] = torch.ones((batch_size, slots, channels), dtype=torch.bool)
        sensors["waveform_modality_ids"] = torch.arange(channels, dtype=torch.long)[None, :].repeat(batch_size, 1)
    if config.use_numeric:
        vocab = model.signals.numeric_encoder.config
        events = min(16, config.max_numeric_events)
        measures = min(4, vocab.num_measures - 1)
        features = len(numeric_summary_features(config.numeric_summary_version))
        sensors["numeric"] = {
            "values": torch.randn((batch_size, events), generator=generator),
            "rel_time_min": -torch.rand((batch_size, events), generator=generator) * float(config.max_numeric_lookback_min),
            "measure_ids": torch.randint(1, measures + 1, (batch_size, events), generator=generator),
            "source_ids": torch.zeros((batch_size, events), dtype=torch.long),
            "event_mask": torch.ones((batch_size, events), dtype=torch.bool),
            "summary_features": torch.randn((batch_size, measures, features), generator=generator),
            "summary_measure_ids": torch.arange(1, measures + 1, dtype=torch.long)[None, :].repeat(batch_size, 1),
        }
    return {
        "domain": config.domain,
        "sample_ids": [f"synthetic-{i}" for i in range(batch_size)],
        "input_text": [f"Synthetic pre-decision context for sample {i}." for i in range(batch_size)],
        "sensors": SensorBatch(**sensors),
    }


def infer_lora_target_modules(model: nn.Module) -> list[str]:
    """Same LoRA target-module selection as training."""
    common = {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
        "c_attn", "c_proj", "W_pack", "wq", "wk", "wv", "wo",
        # Hybrid DeltaNet projections; the remaining layers use the q/k/v/o names above.
        "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
    }
    found = {name.rsplit(".", 1)[-1] for name, module in model.named_modules() if isinstance(module, nn.Linear)} & common
    return sorted(found) if found else ["q_proj", "k_proj", "v_proj", "o_proj"]


def load_tokenizer(config: ModelConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=config.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_lm(config: ModelConfig) -> nn.Module:
    """Load the language backbone, with the LoRA adapters if the checkpoint uses them."""
    kwargs: dict[str, Any] = {"trust_remote_code": config.trust_remote_code}
    if config.lm_dtype is not None:
        kwargs["torch_dtype"] = config.lm_dtype
    if config.attn_implementation:
        kwargs["attn_implementation"] = config.attn_implementation
    lm = AutoModelForCausalLM.from_pretrained(config.model_id, **kwargs)
    lm.config.use_cache = False
    if config.lora:
        from peft import LoraConfig, get_peft_model

        target_modules = (
            [x.strip() for x in config.lora_target_modules.split(",") if x.strip()]
            or infer_lora_target_modules(lm)
        )
        lm = get_peft_model(lm, LoraConfig(
            r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
            bias="none", task_type="CAUSAL_LM", target_modules=target_modules,
        ))
    return lm


class CgmSlotPatchifier(SimpleWaveformPatchifier):
    """``SimpleWaveformPatchifier`` over 3-point slots: Conv1d(kernel 3, stride 3) -> 1 token per slot."""

    def __init__(self, *, output_dim: int) -> None:
        super().__init__(output_dim=output_dim, sample_rate_hz=3, window_seconds=1, patch_seconds=1)


# Embedding-table sizes used by the released checkpoints.
WAVEFORM_TOKEN_DIM = 768  # width of the frozen waveform encoder / slot patchifier
PATCHES_PER_MINUTE = 64  # H patch-position table (60 one-second patches are used)
H_MIN_MINUTES = 30  # H minute table is at least this long
B_MIN_MINUTES = 60  # B minute table is at least this long
NUMERIC_TOKEN_SLACK, NUMERIC_TOKEN_MIN = 32, 1024  # numeric position table sizing


class SignalBuilder(nn.Module):
    """Encode a ``SensorBatch`` into signal tokens for the language model (B or H fusion)."""

    def __init__(self, config: ModelConfig, *, llm_hidden_size: int) -> None:
        super().__init__()
        self.config = config
        self.waveform_modalities = WAVEFORM_MODALITIES[config.domain]
        self.is_h = config.fusion_mode == "hcsm"
        self._caption_detail_candidates: dict[str, torch.Tensor] | None = None

        self.dino = None
        if config.use_waveform_dino:
            if config.waveform_encoder == "simple_patchifier":
                self.dino = CgmSlotPatchifier(output_dim=WAVEFORM_TOKEN_DIM)
            else:
                self.dino = FrozenWaveformEncoder(config.dino_ckpt)

        self.numeric_encoder = None
        self.numeric_hcsm = None
        numeric_tokens = max(config.max_numeric_events + NUMERIC_TOKEN_SLACK, NUMERIC_TOKEN_MIN)
        if config.use_numeric:
            vocab = json.loads(Path(config.numeric_config).read_text())
            num_measures = max(int(x) for x in vocab["measure_to_id"].values()) + 1
            num_sources = max(int(x) for x in vocab["source_to_id"].values()) + 1
            numeric_tokens = max(numeric_tokens, config.max_numeric_events + num_measures - 1)
            self.numeric_encoder = NumericEventEncoder(NumericEncoderConfig(
                num_measures=num_measures,
                num_sources=num_sources,
                hidden_dim=config.numeric_hidden_dim,
                max_lookback_min=float(config.max_numeric_lookback_min),
                summary_feature_dim=len(numeric_summary_features(config.numeric_summary_version)),
                transformer_layers=config.numeric_transformer_layers,
                transformer_heads=config.numeric_transformer_heads,
                dropout=config.dropout,
            ))
            if self.is_h and config.numeric_hcsm:
                # Numeric-memory sizes fall back to the waveform-memory sizes when unset.
                self.numeric_hcsm = ChannelwiseNumericSignalMemory(
                    input_dim=config.numeric_hidden_dim,
                    hidden_dim=config.numeric_hcsm_hidden_dim or config.hcsm_hidden_dim,
                    num_measures=num_measures,
                    lookback_min=config.max_numeric_lookback_min,
                    max_events=config.max_numeric_events,
                    heads=config.hcsm_heads,
                    local_tokens_per_minute=config.numeric_hcsm_local_tokens_per_group or config.hcsm_local_tokens_per_minute,
                    temporal_tokens_per_channel=config.numeric_hcsm_temporal_tokens_per_channel or config.hcsm_temporal_tokens_per_channel,
                    global_tokens=config.numeric_hcsm_global_tokens or config.hcsm_global_tokens,
                    dropout=config.hcsm_dropout,
                    group_minutes=config.numeric_hcsm_group_minutes,
                )

        self.hcsm = None
        if not self.is_h:
            self.projector = ModalityTokenProjector(
                waveform_dim=WAVEFORM_TOKEN_DIM,
                numeric_dim=config.numeric_hidden_dim,
                llm_hidden_size=llm_hidden_size,
                max_waveform_minutes=max(config.max_waveform_lookback_min, B_MIN_MINUTES),
                num_waveform_modalities=len(self.waveform_modalities) if config.use_waveform_dino else 0,
                max_numeric_tokens=numeric_tokens,
                dropout=config.dropout,
            )
        else:
            self.hcsm = HierarchicalChannelwiseSignalMemory(
                input_dim=WAVEFORM_TOKEN_DIM,
                hidden_dim=config.hcsm_hidden_dim,
                num_modalities=len(self.waveform_modalities),
                max_minutes=max(H_MIN_MINUTES, config.max_waveform_lookback_min),
                max_patches_per_minute=PATCHES_PER_MINUTE,
                heads=config.hcsm_heads,
                local_tokens_per_minute=config.hcsm_local_tokens_per_minute,
                temporal_tokens_per_channel=config.hcsm_temporal_tokens_per_channel,
                global_tokens=config.hcsm_global_tokens,
                dropout=config.hcsm_dropout,
            )
            self.projector = HTokenProjector(
                memory_dim=config.hcsm_hidden_dim,
                numeric_dim=(self.numeric_hcsm.hierarchy.adapter.input_proj.out_features
                             if self.numeric_hcsm is not None else config.numeric_hidden_dim),
                llm_hidden_size=llm_hidden_size,
                num_waveform_modalities=len(self.waveform_modalities),
                max_waveform_minutes=max(H_MIN_MINUTES, config.max_waveform_lookback_min),
                max_numeric_tokens=numeric_tokens,
                dropout=config.dropout,
            )

    def get_caption_detail_candidates(self) -> dict[str, torch.Tensor] | None:
        """Candidates from the most recent forward, for the caption decoder only."""
        return self._caption_detail_candidates

    def forward(self, batch: SensorBatch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the signal-token prefix ``[B, S, H]`` and its mask ``[B, S]``."""
        self._caption_detail_candidates = {} if self.config.caption_detail_selector else None
        waveform = None
        if self.dino is not None and batch.waveform is not None:
            waveform = self.dino(
                batch.waveform.to(device),
                channel_mask=batch.waveform_channel_mask.to(device),
                modality_ids=batch.waveform_modality_ids.to(device),
            )
        numeric = None
        if self.numeric_encoder is not None and batch.numeric is not None:
            inputs = {key: value.to(device) for key, value in batch.numeric.items()}
            numeric = self.numeric_encoder(**inputs)
            if self._caption_detail_candidates is not None:
                # Only real observations (not per-measure summaries) are detail candidates.
                self._caption_detail_candidates.update({
                    "numeric_candidates": numeric.tokens,
                    "numeric_mask": numeric.attention_mask & numeric.token_kind_ids.eq(NumericEventEncoder.EVENT_KIND)
                    & numeric.measure_ids.gt(0),
                    "numeric_measure_ids": numeric.measure_ids,
                })
            if self.numeric_hcsm is not None:
                numeric = self.numeric_hcsm(
                    numeric.tokens, numeric.attention_mask, numeric.measure_ids,
                    token_kind_ids=numeric.token_kind_ids, rel_time_min=inputs["rel_time_min"],
                )

        if not self.is_h:
            tokens, mask = self.projector(
                waveform_tokens=waveform.tokens if waveform is not None else None,
                waveform_mask=waveform.mask if waveform is not None else None,
                waveform_modality_ids=waveform.modality_ids if waveform is not None else None,
                waveform_minute_ids=waveform.minute_ids if waveform is not None else None,
                numeric_tokens=numeric.tokens if numeric is not None else None,
                numeric_mask=numeric.attention_mask if numeric is not None else None,
            )
        else:
            memory = None
            if waveform is not None and batch.waveform.shape[2] > 0:  # numeric-only rows collate to C=0
                bsz, minutes, channels, _ = batch.waveform.shape
                patches = waveform.tokens.shape[2]
                slot_mask = waveform.mask.reshape(bsz, minutes, channels)
                modality_ids = batch.waveform_modality_ids.to(device)
                memory = self.hcsm.encode(
                    waveform.tokens.reshape(bsz, minutes, channels, patches, -1),
                    slot_mask=slot_mask,
                    modality_ids=modality_ids,
                    visible_patch_mask=slot_mask[:, :, :, None].expand(-1, -1, -1, patches),
                )
                if self._caption_detail_candidates is not None:
                    self._caption_detail_candidates.update({
                        "waveform_candidates": memory.patch_hidden.reshape(bsz, -1, memory.patch_hidden.shape[-1]),
                        "waveform_mask": slot_mask[:, :, :, None].expand(-1, -1, -1, patches).reshape(bsz, -1),
                        "waveform_channel_ids": modality_ids[:, None, :, None].expand(bsz, minutes, channels, patches).reshape(bsz, -1),
                    })
            numeric_mask = None if numeric is None else (
                numeric.memory_mask if self.numeric_hcsm is not None else numeric.attention_mask
            )
            tokens, mask = self.projector(
                memory_tokens=memory.memory if memory is not None else None,
                memory_mask=memory.memory_mask if memory is not None else None,
                memory_level_ids=memory.memory_level_ids if memory is not None else None,
                memory_modality_ids=memory.memory_modality_ids if memory is not None else None,
                memory_minute_ids=memory.memory_minute_ids if memory is not None else None,
                numeric_tokens=(numeric.memory if self.numeric_hcsm is not None else numeric.tokens) if numeric is not None else None,
                numeric_mask=numeric_mask,
                # The hierarchical numeric grid contains holes when channel counts differ.
                numeric_position_ids=(numeric_mask.long().cumsum(-1) - 1
                                      if numeric_mask is not None and self.numeric_hcsm is not None else None),
            )
        if tokens.shape[1] > self.config.max_signal_tokens:
            raise ValueError(
                f"Signal sequence exceeds max_signal_tokens: produced={tokens.shape[1]}, "
                f"budget={self.config.max_signal_tokens}."
            )
        return tokens, mask


def evidence_index_from_checkpoint(payload: Mapping[str, Any] | None) -> ActionEvidenceIndex | None:
    if not payload or not payload.get("fact_keys"):
        return None
    fact_keys = [str(x) for x in payload["fact_keys"]]
    return ActionEvidenceIndex(
        fact_keys=fact_keys,
        fact_to_id={key: idx for idx, key in enumerate(fact_keys)},
        group_fact_strengths={
            str(group): {str(key): float(value) for key, value in dict(strengths).items()}
            for group, strengths in dict(payload.get("group_fact_strengths") or {}).items()
        },
    )


def build_model_and_signal(
    *,
    ckpt: Mapping[str, Any],
    config: ModelConfig,
    tokenizer: Any,
    label_names: list[str],
    action_group_types: Mapping[str, str] | None,
    device: torch.device,
) -> tuple[MultimodalLLMWithActionHeads, SignalBuilder, DescriptorActionLabelRanker | None, ActionLabelIndex | None]:
    """Build the model, signal builder, and label ranker from a checkpoint."""
    state = ckpt["model"]
    if any(str(key).startswith("caption_action_head.") for key in state):
        raise ValueError("Unsupported checkpoint: caption-action alignment head.")
    typed_need_head = tuple(state["need_head.weight"].shape[:1]) == (2,)
    if typed_need_head and action_group_types is None:
        raise ValueError(
            "action_group_types is required for this checkpoint (category -> intervention/diagnostic mapping)."
        )
    evidence_index = evidence_index_from_checkpoint(ckpt.get("evidence_index"))
    fact_keys = evidence_index.fact_keys if evidence_index is not None else []
    lm = load_base_lm(config).to(device)
    model = MultimodalLLMWithActionHeads(
        lm,
        len(label_names),
        config.dropout,
        num_evidence_facts=len(fact_keys),
        hierarchical_action_heads=config.hierarchical_action_heads,
        enable_action_conditioned_caption_decoder=any(str(key).startswith("caption_memory.") for key in state),
        caption_decoder_mode=config.caption_decoder_mode,
        caption_profile_max_groups=config.caption_profile_max_groups,
        caption_profile_facts_per_group=config.caption_profile_facts_per_group,
        caption_profile_need_threshold=config.caption_profile_need_threshold,
        caption_profile_group_threshold=config.caption_profile_group_threshold,
        caption_profile_fact_threshold=config.caption_profile_fact_threshold,
        enable_caption_detail_selector=any(
            str(key).startswith(("caption_memory.detail_", "caption_memory.waveform_", "caption_memory.numeric_"))
            for key in state
        ),
        waveform_detail_dim=config.hcsm_hidden_dim,
        numeric_detail_dim=config.numeric_hidden_dim,
        caption_waveform_detail_tokens_per_channel=config.caption_waveform_detail_tokens_per_channel,
        caption_numeric_detail_tokens_per_measure=config.caption_numeric_detail_tokens_per_measure,
        caption_waveform_detail_max_tokens=config.caption_waveform_detail_max_tokens,
        caption_numeric_detail_max_tokens=config.caption_numeric_detail_max_tokens,
        group_fact_compatibility=(
            evidence_index.group_fact_strength_tensor(label_names, device=torch.device("cpu"))
            if evidence_index is not None else None
        ),
        need_head_mode="typed_multilabel" if typed_need_head else "legacy",
        group_need_type_indices=torch.tensor(action_group_need_indices(label_names, action_group_types or {}), dtype=torch.long),
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.configure_caption_profile_descriptors(tokenizer, label_names, fact_keys)
    signal_builder = SignalBuilder(config, llm_hidden_size=get_hidden_size(lm.config)).to(device)
    signal_builder.load_state_dict(ckpt["signal_builder"], strict=True)

    label_index = label_ranker = None
    if ckpt.get("label_index") and ckpt.get("label_ranker") is not None:
        label_index = ActionLabelIndex.from_checkpoint(ckpt["label_index"])
        label_ranker = DescriptorActionLabelRanker(
            tokenizer=tokenizer,
            label_texts=label_index.label_texts,
            hidden_size=model.need_head.in_features,
            temperature=config.label_ranking_temperature,
        ).to(device)
        label_ranker.load_state_dict(ckpt["label_ranker"], strict=True)
    for module in (model, signal_builder, label_ranker):
        if module is not None:
            module.eval().requires_grad_(False)
    return model, signal_builder, label_ranker, label_index


class OpenSLA:
    """Inference wrapper for a trained OpenSLA-B or OpenSLA-H checkpoint."""

    def __init__(
        self, *, config: ModelConfig, device: torch.device, tokenizer: Any, model: MultimodalLLMWithActionHeads,
        signals: SignalBuilder, ranker: DescriptorActionLabelRanker | None, label_index: ActionLabelIndex | None,
        category_names: list[str],
    ) -> None:
        self.config, self.device, self.tokenizer = config, device, tokenizer
        self.model, self.signals, self.ranker, self.label_index = model, signals, ranker, label_index
        self.category_names = category_names
        self.name = "OpenSLA-H" if config.fusion_mode == "hcsm" else "OpenSLA-B"

    @classmethod
    def from_checkpoint(
        cls, checkpoint: str | Path, *, domain: str, device: str = "cuda",
        model_id: str | None = None, dino_checkpoint: str | Path | None = None,
        numeric_config: str | Path | None = None,
        action_group_types: str | Path | Mapping[str, str] | None = None,
    ) -> "OpenSLA":
        """Load a checkpoint; the path arguments override the paths stored in it."""
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        required = {"args", "model", "signal_builder", "label_names"}
        if not required.issubset(ckpt):
            raise ValueError(f"Checkpoint is missing {sorted(required - set(ckpt))}")
        args = dict(ckpt["args"])
        overrides = {"model_id": model_id, "dino_ckpt": dino_checkpoint, "numeric_config": numeric_config}
        args.update({key: str(value) for key, value in overrides.items() if value is not None})
        device_obj = torch.device(device)
        if device_obj.type == "cpu":
            args.update(bf16=False, fp16=False)
        config = ModelConfig.from_checkpoint_args(args, domain=domain)
        tokenizer = load_tokenizer(config)
        names = [str(x) for x in ckpt["label_names"]]
        model, signals, ranker, index = build_model_and_signal(
            ckpt=ckpt, config=config, tokenizer=tokenizer, label_names=names,
            action_group_types=load_action_group_types(action_group_types), device=device_obj,
        )
        return cls(config=config, device=device_obj, tokenizer=tokenizer, model=model, signals=signals,
                   ranker=ranker, label_index=index, category_names=names)

    def _prompts(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the checkpoint's prompt format; return ids, mask, and each prompt's last index."""
        template, system = self.config.prompt_template, self.config.system_prompt.strip()
        prompts = []
        for text in texts:
            text = str(text).strip()
            if template == "chat":
                messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            elif template == "plain":
                text = (system + "\n\n" if system else "") + text + "\n\n"
            elif template != "preformatted":
                raise ValueError(f"Unknown prompt template: {template}")
            prompts.append(text)
        tokens = self.tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")
        if tokens["input_ids"].shape[1] > self.config.max_seq_len:
            raise ValueError("Prompt is longer than the checkpoint's max_seq_len.")
        mask = tokens["attention_mask"].to(self.device)
        if bool((mask.sum(1) == 0).any()):
            raise ValueError("Each sample needs a nonempty prompt.")
        return tokens["input_ids"].to(self.device), mask, mask.sum(1) - 1

    @torch.inference_mode()
    def predict(
        self, input_text: list[str], sensors: SensorBatch, *, max_new_tokens: int = 512,
        necessity_threshold: float = 0.5, category_threshold: float = 0.5, labels_per_category: int = 3,
        max_categories: int = 5,
    ) -> list[dict[str, Any]]:
        """Predict necessity, action categories, ranked fine labels, and (optionally) a caption per sample."""
        if not input_text:
            raise ValueError("input_text must contain at least one sample.")
        if max_new_tokens < 0 or labels_per_category < 1 or max_categories < 1:
            raise ValueError("max_new_tokens must be nonnegative and output label/category limits positive.")
        if not 0 <= necessity_threshold <= 1 or not 0 <= category_threshold <= 1:
            raise ValueError("Decision thresholds must be in [0, 1].")
        if max_new_tokens and self.model.caption_memory is None:
            raise ValueError("This checkpoint has no caption decoder; use max_new_tokens=0 for actions only.")
        numeric_vocab = self.signals.numeric_encoder.config if self.signals.numeric_encoder is not None else None
        sensors.validate(len(input_text), self.config, numeric_vocab)
        ids, mask, end = self._prompts(input_text)
        signal_tokens, signal_mask = self.signals(sensors, self.device)
        kwargs = dict(input_ids=ids, attention_mask=mask, prompt_end_indices=end,
                      signal_embeds=signal_tokens, signal_attention_mask=signal_mask)
        if max_new_tokens:
            generated, outputs = self.model.generate_action_conditioned_caption(
                **kwargs, caption_detail_candidates=self.signals.get_caption_detail_candidates(),
                max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            captions = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        else:
            outputs = self.model.forward_action_heads_only(**kwargs)
            captions = [""] * len(input_text)
        need = outputs["need_logits"].float().sigmoid()
        categories = outputs["group_logits"].float().sigmoid()
        fine = self._rank_fine_labels(outputs["decision_hidden"]) if self.ranker is not None else None
        results = []
        for i, caption in enumerate(captions):
            needed = bool(need[i] >= necessity_threshold)
            selected = [g for g in range(len(self.category_names)) if needed and categories[i, g] >= category_threshold]
            if needed and not selected:
                selected = [int(categories[i].argmax())]
            selected = sorted(selected, key=lambda g: float(categories[i, g]), reverse=True)[:max_categories]
            results.append({
                "necessity_probability": float(need[i]),
                "action_needed": needed,
                "category_probabilities": dict(zip(self.category_names, categories[i].tolist())),
                "selected_categories": [self.category_names[g] for g in selected],
                "fine_labels": {self.category_names[g]: fine(i, g, labels_per_category) for g in selected} if fine else {},
                "caption": REPEATED_TARGET_HEADER.split(caption, maxsplit=1)[0].strip(),
            })
        return results

    def _rank_fine_labels(self, decision_hidden: torch.Tensor):
        """Return ``rank(row, group, k)`` scoring every fine label of a group by cosine similarity."""
        queries = F.normalize(self.ranker.query(decision_hidden.float()), dim=-1)
        label_ids = torch.arange(len(self.label_index.label_keys), device=self.device)
        vectors = F.normalize(self.ranker.label_vectors(label_ids, self.model.lm.get_input_embeddings()).float(), dim=-1)

        def rank(row: int, group: int, k: int) -> list[dict[str, Any]]:
            candidates = self.label_index.group_to_label_ids[group]
            if not candidates:
                return []
            scores, order = (vectors[candidates] @ queries[row]).topk(min(k, len(candidates)))
            return [{"label": self.label_index.label_keys[candidates[j]].split("|", 1)[-1], "cosine_similarity": float(s)}
                    for s, j in zip(scores.tolist(), order.tolist())]

        return rank
