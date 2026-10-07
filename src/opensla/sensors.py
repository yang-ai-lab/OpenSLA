"""Sensor encoders, hierarchical signal memory, and token projectors into the language model."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import torch
import torch.nn as nn

from .vit1d import vit_base, vit_large, vit_middle, vit_nano, vit_small, vit_tiny, vit_xl


@dataclass
class WaveformTokens:
    """Patch tokens from a one-channel encoder, with token provenance preserved.

    ``tokens`` is ``[B, M*C, P, D]``: the channel axis is collapsed into the
    slot axis.  The matching slot mask, modality id, and minute offset are
    carried alongside so fusion can distinguish channels without exposing that
    identity to the frozen encoder itself.
    """

    tokens: torch.Tensor
    mask: torch.Tensor
    modality_ids: torch.Tensor
    minute_ids: torch.Tensor


def _vit_factory(name: str):
    factories = {
        "vit_nano": vit_nano,
        "vit_tiny": vit_tiny,
        "vit_small": vit_small,
        "vit_middle": vit_middle,
        "vit_base": vit_base,
        "vit_large": vit_large,
        "vit_xl": vit_xl,
    }
    if name not in factories:
        raise ValueError(f"Unknown ViT backbone: {name}")
    return factories[name]


def build_dino_backbone_from_hparams(hparams: dict[str, Any]) -> nn.Module:
    name = str(hparams.get("psg_encoder_name", "vit_base"))
    num_leads = int(hparams.get("num_leads", 3))
    sample_rate = int(round(float(hparams.get("sample_rate", 64))))
    window_size = int(round(float(hparams.get("window_size", 60))))
    patch_size_time = int(hparams.get("patch_size_time", 64))
    lead_wise = int(hparams.get("lead_wise", 0))
    patch_size_ch = int(hparams.get("patch_size_ch", 4))
    use_lead_embedding = bool(hparams.get("use_lead_embedding", True))
    if lead_wise == 0:
        patch_size_ch = num_leads
    return _vit_factory(name)(
        num_leads=num_leads,
        seq_len=sample_rate * window_size,
        patch_size=patch_size_time,
        lead_wise=lead_wise,
        patch_size_ch=patch_size_ch,
        use_lead_embedding=use_lead_embedding,
    )


def load_dino_backbone(ckpt_path: Path | str) -> tuple[nn.Module, dict[str, Any]]:
    """Load the frozen teacher backbone and its hyper-parameters from a Lightning checkpoint."""
    ckpt_path = Path(ckpt_path)
    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    hparams = dict(payload.get("hyper_parameters", {}))
    backbone = build_dino_backbone_from_hparams(hparams)
    prefix = "teacher_encoder.backbone."
    state = payload.get("state_dict", {})
    backbone_state = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
    if not backbone_state:
        raise KeyError(f"No backbone weights found with prefix {prefix!r} in {ckpt_path}")
    missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Unexpected backbone load mismatch. missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad_(False)
    return backbone, hparams


class FrozenWaveformEncoder(nn.Module):
    """Apply a frozen one-channel encoder independently to each waveform channel.

    Every observed channel-minute is encoded as ``[1, T]``; the channel axis is
    flattened only *after* encoding.  Modality identity is returned as metadata
    for the trainable fusion stack.
    """

    def __init__(
        self,
        ckpt_path: Path | str,
        *,
        layer_norm_output: bool = True,
        max_batch_windows: int = 128,
    ) -> None:
        super().__init__()
        self.backbone, self.hparams = load_dino_backbone(ckpt_path)
        self.layer_norm_output = bool(layer_norm_output)
        # The backbone is frozen, so process channel-minutes in bounded chunks
        # without changing the resulting token sequence.
        self.max_batch_windows = max(1, int(max_batch_windows))
        self.output_dim = int(getattr(self.backbone, "width", self.hparams.get("shared_emb_dim", 768)))
        self.sample_rate = int(round(float(self.hparams.get("sample_rate", 125))))
        self.window_size = int(round(float(self.hparams.get("window_size", 60))))
        self.num_leads = int(self.hparams.get("num_leads", 1))
        self.seq_len = self.sample_rate * self.window_size
        if self.num_leads != 1:
            raise ValueError(
                "FrozenWaveformEncoder requires a num_leads=1 checkpoint; "
                f"got num_leads={self.num_leads} from {ckpt_path}"
            )

    def forward(
        self,
        x: torch.Tensor,
        *,
        channel_mask: torch.Tensor | None = None,
        modality_ids: torch.Tensor | None = None,
    ) -> WaveformTokens:
        """Encode ``[B, M, C, T]`` waveform chunks independently.

        ``channel_mask`` is ``[B,M,C]`` and marks observed channel-minutes.
        ``modality_ids`` is ``[B,C]`` or ``[B,M,C]``; ``-1`` denotes padding.
        """
        if x.ndim != 4:
            raise ValueError(f"Expected channelwise waveform [B,M,C,T], got {tuple(x.shape)}")
        batch, minutes, channels, length = x.shape
        if length != self.seq_len:
            raise ValueError(
                f"Expected single-channel window length {self.seq_len} "
                f"({self.sample_rate}Hz x {self.window_size}s), got {length}"
            )
        if channel_mask is None:
            channel_mask = torch.ones(batch, minutes, channels, dtype=torch.bool, device=x.device)
        if channel_mask.shape != (batch, minutes, channels):
            raise ValueError(
                f"Expected channel_mask [B,M,C]={batch, minutes, channels}, got {tuple(channel_mask.shape)}"
            )
        if modality_ids is None:
            modality_ids = torch.zeros(batch, channels, dtype=torch.long, device=x.device)
        if modality_ids.ndim == 2:
            modality_ids = modality_ids[:, None, :].expand(batch, minutes, channels)
        if modality_ids.shape != (batch, minutes, channels):
            raise ValueError(
                f"Expected modality_ids [B,C] or [B,M,C], got {tuple(modality_ids.shape)}"
            )

        # Run the frozen ViT only on observed channel-minutes.
        flat = x.reshape(batch * minutes * channels, 1, length)
        flat_mask_all = channel_mask.reshape(batch * minutes * channels).bool()
        flat_modality_all = modality_ids.reshape(batch * minutes * channels).long()
        active = flat_mask_all & (flat_modality_all >= 0)
        patch_size = int(self.hparams.get("patch_size_time", self.sample_rate))
        patch_count = max(1, self.seq_len // max(1, patch_size))

        tokens = x.new_zeros((flat.shape[0], patch_count, self.output_dim))
        active_flat = flat[active]
        if active_flat.numel() > 0:
            with torch.no_grad():
                chunks: list[torch.Tensor] = []
                for start in range(0, int(active_flat.shape[0]), self.max_batch_windows):
                    current = active_flat[start : start + self.max_batch_windows]
                    _cls, patches = self.backbone.forward_encoding(current, return_sequence=False)
                    chunks.append(patches)
                active_tokens = torch.cat(chunks, dim=0)
            if self.layer_norm_output:
                active_tokens = torch.nn.functional.layer_norm(active_tokens, (active_tokens.shape[-1],))
            tokens[active] = active_tokens

        flat_channels = minutes * channels
        flat_tokens = tokens.reshape(batch, flat_channels, patch_count, self.output_dim)
        flat_mask = channel_mask.reshape(batch, flat_channels).bool()
        flat_modality = modality_ids.reshape(batch, flat_channels).long()
        minute_ids = (
            torch.arange(minutes, device=x.device, dtype=torch.long)[:, None]
            .expand(minutes, channels)
            .reshape(1, flat_channels)
            .expand(batch, flat_channels)
        )
        return WaveformTokens(
            tokens=flat_tokens,
            mask=flat_mask,
            modality_ids=flat_modality,
            minute_ids=minute_ids,
        )



class SimpleWaveformPatchifier(nn.Module):
    """Patch each physical waveform channel with one shallow temporal Conv1d.

    Input is the unified channel-wise waveform tensor ``[B, M, C, 7500]``.
    With the default one-second patch, output is ``[B, M*C, 60, 768]`` and
    carries the same slot mask, modality IDs, and minute IDs as the channel-wise
    waveform encoder.  Modality and time embeddings are added by the fusion
    projector.
    """

    def __init__(
        self,
        *,
        output_dim: int = 768,
        sample_rate_hz: int = 125,
        window_seconds: int = 60,
        patch_seconds: int = 1,
    ) -> None:
        super().__init__()
        self.output_dim = int(output_dim)
        self.sample_rate = int(sample_rate_hz)
        self.window_size = int(window_seconds)
        self.patch_seconds = int(patch_seconds)
        self.seq_len = self.sample_rate * self.window_size
        self.patch_samples = self.sample_rate * self.patch_seconds
        if self.patch_samples <= 0 or self.seq_len % self.patch_samples:
            raise ValueError("Waveform window length must be divisible by the patch length.")
        self.patch_count = self.seq_len // self.patch_samples
        self.patch_conv = nn.Conv1d(
            1,
            self.output_dim,
            kernel_size=self.patch_samples,
            stride=self.patch_samples,
            bias=True,
        )
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(self.output_dim)

    def forward(
        self,
        x: torch.Tensor,
        *,
        channel_mask: torch.Tensor | None = None,
        modality_ids: torch.Tensor | None = None,
    ) -> WaveformTokens:
        if x.ndim != 4:
            raise ValueError(f"Expected waveform [B,M,C,T], got {tuple(x.shape)}")
        batch, minutes, channels, length = x.shape
        if length != self.seq_len:
            raise ValueError(
                f"Expected {self.window_size}s at {self.sample_rate}Hz ({self.seq_len} samples), got {length}."
            )
        if channel_mask is None:
            channel_mask = torch.ones(batch, minutes, channels, dtype=torch.bool, device=x.device)
        if tuple(channel_mask.shape) != (batch, minutes, channels):
            raise ValueError(
                f"Expected channel_mask [B,M,C]={batch, minutes, channels}, got {tuple(channel_mask.shape)}"
            )
        if modality_ids is None:
            modality_ids = torch.zeros(batch, channels, dtype=torch.long, device=x.device)
        if modality_ids.ndim == 2:
            modality_ids = modality_ids[:, None, :].expand(batch, minutes, channels)
        if tuple(modality_ids.shape) != (batch, minutes, channels):
            raise ValueError(
                f"Expected modality_ids [B,C] or [B,M,C], got {tuple(modality_ids.shape)}"
            )

        flat = x.reshape(batch * minutes * channels, 1, length)
        patches = self.patch_conv(flat).transpose(1, 2)
        patches = self.norm(self.activation(patches))
        valid = channel_mask.bool() & modality_ids.ge(0)
        patches = patches * valid.reshape(-1, 1, 1).to(dtype=patches.dtype)

        slots = minutes * channels
        minute_ids = (
            torch.arange(minutes, device=x.device, dtype=torch.long)[:, None]
            .expand(minutes, channels)
            .reshape(1, slots)
            .expand(batch, slots)
        )
        return WaveformTokens(
            tokens=patches.reshape(batch, slots, self.patch_count, self.output_dim),
            mask=valid.reshape(batch, slots),
            modality_ids=modality_ids.reshape(batch, slots).long(),
            minute_ids=minute_ids,
        )



class NumericEncoderOutput(NamedTuple):
    tokens: torch.Tensor
    attention_mask: torch.Tensor
    measure_ids: torch.Tensor
    token_kind_ids: torch.Tensor


@dataclass
class NumericEncoderConfig:
    """Shapes of the numeric event encoder; all fields are fixed by the checkpoint."""

    num_measures: int
    num_sources: int
    hidden_dim: int
    max_lookback_min: float
    summary_feature_dim: int
    transformer_layers: int
    transformer_heads: int
    dropout: float = 0.0
    value_hidden_dim: int = 64
    time_fourier_bands: int = 8


class FourierTimeEncoder(nn.Module):
    def __init__(self, bands: int, max_lookback_min: float, output_dim: int) -> None:
        super().__init__()
        self.bands = int(bands)
        self.max_lookback_min = float(max_lookback_min)
        in_dim = 1 + 2 * self.bands
        self.proj = nn.Sequential(nn.Linear(in_dim, output_dim), nn.GELU(), nn.LayerNorm(output_dim))

    def forward(self, rel_time_min: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(rel_time_min / max(self.max_lookback_min, 1e-6), min=-1.0, max=0.0)
        feats = [x.unsqueeze(-1)]
        if self.bands > 0:
            freqs = torch.pow(2.0, torch.arange(self.bands, device=x.device, dtype=x.dtype))
            angles = x.unsqueeze(-1) * freqs * torch.pi
            feats.extend([torch.sin(angles), torch.cos(angles)])
        return self.proj(torch.cat(feats, dim=-1))


class NumericEventEncoder(nn.Module):
    """Encode irregular numeric observations into latent tokens.

    Every observed event becomes one token (value, measure, source, time); every
    measure additionally gets one summary token so that missing variables are
    explicitly represented.  Tokens are not projected to the LLM hidden size
    here; the fusion projector does that.
    """

    EVENT_KIND = 0
    SUMMARY_KIND = 1

    def __init__(self, config: NumericEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.measure_embed = nn.Embedding(config.num_measures, config.hidden_dim)
        self.source_embed = nn.Embedding(max(1, config.num_sources), config.hidden_dim)
        self.kind_embed = nn.Embedding(2, config.hidden_dim)
        self.value_mlp = nn.Sequential(
            nn.Linear(4, config.value_hidden_dim),
            nn.GELU(),
            nn.Linear(config.value_hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
        )
        self.time_encoder = FourierTimeEncoder(config.time_fourier_bands, config.max_lookback_min, config.hidden_dim)
        self.summary_mlp = nn.Sequential(
            nn.Linear(config.summary_feature_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.transformer_layers)
        self.final_norm = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        *,
        values: torch.Tensor,
        measure_ids: torch.Tensor,
        rel_time_min: torch.Tensor,
        event_mask: torch.Tensor,
        source_ids: torch.Tensor,
        summary_features: torch.Tensor,
        summary_measure_ids: torch.Tensor,
    ) -> NumericEncoderOutput:
        """Return ``[B, E + Q, D]`` tokens: observed events followed by per-measure summaries."""
        event_mask = event_mask.bool()
        finite = torch.isfinite(values) & event_mask
        safe_values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        value_features = torch.stack(
            [safe_values, torch.sign(safe_values) * torch.log1p(torch.abs(safe_values)),
             finite.to(safe_values.dtype), event_mask.to(safe_values.dtype)],
            dim=-1,
        )
        event_tokens = (
            self.value_mlp(value_features)
            + self.measure_embed(measure_ids.clamp(min=0, max=self.config.num_measures - 1))
            + self.source_embed(source_ids.clamp(min=0, max=self.config.num_sources - 1))
            + self.time_encoder(rel_time_min)
            + self.kind_embed(torch.full_like(measure_ids, self.EVENT_KIND))
        )
        summary_tokens = (
            self.summary_mlp(torch.nan_to_num(summary_features, nan=0.0, posinf=0.0, neginf=0.0))
            + self.measure_embed(summary_measure_ids.clamp(min=0, max=self.config.num_measures - 1))
            + self.kind_embed(torch.full_like(summary_measure_ids, self.SUMMARY_KIND))
        )
        tokens = torch.cat([event_tokens, summary_tokens], dim=1)
        attention_mask = torch.cat([event_mask, torch.ones_like(summary_measure_ids, dtype=torch.bool)], dim=1)
        out_measure_ids = torch.cat([measure_ids, summary_measure_ids], dim=1)
        token_kind_ids = torch.cat(
            [torch.full_like(measure_ids, self.EVENT_KIND), torch.full_like(summary_measure_ids, self.SUMMARY_KIND)], dim=1
        )
        # The transformer cannot attend over an all-masked row; summaries are always present.
        tokens = self.transformer(tokens, src_key_padding_mask=~attention_mask)
        return NumericEncoderOutput(self.final_norm(tokens), attention_mask, out_measure_ids, token_kind_ids)



@dataclass
class SignalMemoryOutput:
    memory: torch.Tensor
    memory_mask: torch.Tensor
    # 0 = local channel-minute memory, 1 = per-channel temporal memory,
    # 2 = cross-channel global memory.  The LLM projector keeps these levels
    # distinguishable after concatenation.
    memory_level_ids: torch.Tensor | None = None
    # Physical waveform identity is kept alongside every emitted token. Local
    # tokens carry exact modality + minute; temporal tokens carry modality and
    # global tokens use -1 for both fields.
    memory_modality_ids: torch.Tensor | None = None
    memory_minute_ids: torch.Tensor | None = None
    # The adapted patch grid, read by the optional caption detail selector.
    patch_hidden: torch.Tensor | None = None


class _WaveformInputAdapter(nn.Module):
    """Add physical modality/time identity without changing token count."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_modalities: int,
        max_minutes: int,
        max_patches_per_minute: int,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.modality_embed = nn.Embedding(num_modalities + 1, hidden_dim)
        self.minute_embed = nn.Embedding(max_minutes, hidden_dim)
        self.patch_embed = nn.Embedding(max_patches_per_minute, hidden_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, 1, hidden_dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        modality_ids: torch.Tensor,
        visible_patch_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return hidden features; masked-but-valid patches use mask token."""
        if tokens.ndim != 5:
            raise ValueError(f"Expected [B,M,C,P,D], got {tuple(tokens.shape)}")
        batch, minutes, channels, patches, _ = tokens.shape
        if modality_ids.shape != (batch, channels):
            raise ValueError(
                f"Expected modality_ids [B,C]={batch, channels}, got {tuple(modality_ids.shape)}"
            )
        if visible_patch_mask.shape != (batch, minutes, channels, patches):
            raise ValueError(
                "Expected visible_patch_mask [B,M,C,P] to match tokens, got "
                f"{tuple(visible_patch_mask.shape)}"
            )
        base = self.input_proj(self.input_norm(tokens))
        modality = modality_ids.clamp(min=-1, max=self.modality_embed.num_embeddings - 2) + 1
        mod_embed = self.modality_embed(modality)[:, None, :, None, :]
        minute_embed = self.minute_embed(torch.arange(minutes, device=tokens.device))[None, :, None, None, :]
        patch_embed = self.patch_embed(torch.arange(patches, device=tokens.device))[None, None, None, :, :]
        position = mod_embed + minute_embed + patch_embed
        masked = self.mask_token + position
        return torch.where(visible_patch_mask[..., None], base + position, masked)


class HierarchicalChannelwiseSignalMemory(nn.Module):
    """Learned patch -> temporal channel -> cross-channel signal memory."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_modalities: int,
        max_minutes: int,
        max_patches_per_minute: int,
        heads: int = 8,
        local_tokens_per_minute: int = 2,
        temporal_tokens_per_channel: int = 16,
        global_tokens: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.adapter = _WaveformInputAdapter(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_modalities=num_modalities,
            max_minutes=max_minutes,
            max_patches_per_minute=max_patches_per_minute,
        )
        self.local_tokens_per_minute = int(local_tokens_per_minute)
        self.temporal_tokens_per_channel = int(temporal_tokens_per_channel)
        self.global_tokens = int(global_tokens)
        self.local_queries = nn.Parameter(torch.randn(self.local_tokens_per_minute, hidden_dim) * 0.02)
        self.temporal_queries = nn.Parameter(torch.randn(self.temporal_tokens_per_channel, hidden_dim) * 0.02)
        self.global_queries = nn.Parameter(torch.randn(self.global_tokens, hidden_dim) * 0.02)
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.local_attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.temporal_attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.global_attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.local_ff = self._ff(hidden_dim, dropout)
        self.temporal_ff = self._ff(hidden_dim, dropout)
        self.global_ff = self._ff(hidden_dim, dropout)

    @staticmethod
    def _ff(hidden_dim: int, dropout: float) -> nn.Module:
        return nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def build_memory(self, hidden: torch.Tensor, *, slot_mask: torch.Tensor, visible_patch_mask: torch.Tensor,
                     modality_ids: torch.Tensor, valid_patch_mask: torch.Tensor | None = None) -> SignalMemoryOutput:
        batch, minutes, channels, patches, dim = hidden.shape
        # Numeric-only samples have no waveform channels; skip attention over an empty grid.
        if channels == 0:
            empty_memory = hidden.new_zeros((batch, 0, dim))
            empty_mask = torch.zeros((batch, 0), dtype=torch.bool, device=hidden.device)
            empty_ids = torch.zeros((batch, 0), dtype=torch.long, device=hidden.device)
            return SignalMemoryOutput(
                memory=empty_memory,
                memory_mask=empty_mask,
                memory_level_ids=empty_ids,
                memory_modality_ids=empty_ids,
                memory_minute_ids=empty_ids,
            )
        valid_patch = slot_mask[:, :, :, None].expand(batch, minutes, channels, patches).bool()
        if valid_patch_mask is not None:
            if valid_patch_mask.shape != valid_patch.shape:
                raise ValueError("valid_patch_mask must match the complete channel-minute grid")
            valid_patch = valid_patch & valid_patch_mask.bool()
            slot_mask = valid_patch.any(dim=-1)

        # Local learned queries read every valid one-second patch in a physical
        # channel-minute; invalid patches carry the learned mask embedding.
        local_kv = hidden.reshape(batch * minutes * channels, patches, dim)
        local_valid = valid_patch.reshape(batch * minutes * channels, patches)
        # Open one dummy key for all-masked rows (they are masked out again below).
        local_valid_safe = local_valid.clone()
        empty_local = ~local_valid_safe.any(dim=1)
        local_valid_safe[empty_local, 0] = True
        mod = modality_ids[:, None, :].expand(batch, minutes, channels).reshape(batch * minutes * channels)
        min_ids = torch.arange(minutes, device=hidden.device)[None, :, None].expand(batch, minutes, channels).reshape(-1)
        local_query = self.local_queries[None].expand(batch * minutes * channels, -1, -1)
        local_query = local_query + self.adapter.modality_embed(mod.clamp(min=-1) + 1)[:, None, :]
        local_query = local_query + self.adapter.minute_embed(min_ids)[:, None, :]
        local, _ = self.local_attn(
            self.norm_q(local_query),
            self.norm_kv(local_kv),
            self.norm_kv(local_kv),
            key_padding_mask=~local_valid_safe,
            need_weights=False,
        )
        local = local + self.local_ff(local)
        local = local.reshape(batch, minutes, channels, self.local_tokens_per_minute, dim)
        local_mask = slot_mask[:, :, :, None].expand(-1, -1, -1, self.local_tokens_per_minute).bool()

        # Per physical channel, learned temporal queries read all local states
        # over the complete 30-minute window.
        temporal_kv = local.permute(0, 2, 1, 3, 4).reshape(batch * channels, minutes * self.local_tokens_per_minute, dim)
        temporal_valid = slot_mask.permute(0, 2, 1).reshape(batch * channels, minutes)
        temporal_valid = temporal_valid[:, :, None].expand(-1, -1, self.local_tokens_per_minute).reshape(batch * channels, -1).bool()
        temporal_valid_safe = temporal_valid.clone()
        empty_temporal = ~temporal_valid_safe.any(dim=1)
        temporal_valid_safe[empty_temporal, 0] = True
        channel_mod = modality_ids.reshape(batch * channels)
        temporal_query = self.temporal_queries[None].expand(batch * channels, -1, -1)
        temporal_query = temporal_query + self.adapter.modality_embed(channel_mod.clamp(min=-1) + 1)[:, None, :]
        temporal, _ = self.temporal_attn(
            self.norm_q(temporal_query),
            self.norm_kv(temporal_kv),
            self.norm_kv(temporal_kv),
            key_padding_mask=~temporal_valid_safe,
            need_weights=False,
        )
        temporal = temporal + self.temporal_ff(temporal)
        temporal = temporal.reshape(batch, channels, self.temporal_tokens_per_channel, dim)
        channel_valid = slot_mask.any(dim=1)
        temporal_mask = channel_valid[:, :, None].expand(-1, -1, self.temporal_tokens_per_channel)
        channel_flat = temporal.reshape(batch, channels * self.temporal_tokens_per_channel, dim)
        channel_mask = temporal_mask.reshape(batch, channels * self.temporal_tokens_per_channel)
        local_flat = local.reshape(batch, minutes * channels * self.local_tokens_per_minute, dim)
        local_flat_mask = local_mask.reshape(batch, minutes * channels * self.local_tokens_per_minute)
        all_channel_memory = torch.cat([local_flat, channel_flat], dim=1)
        all_channel_mask = torch.cat([local_flat_mask, channel_mask], dim=1)

        # Global queries read the full adapter patch grid directly rather than
        # a second-order summary of local/temporal memory. This preserves a
        # direct route from every valid patch to cross-channel state while the
        # local and temporal paths remain available as separate output tokens.
        global_kv = hidden.reshape(batch, minutes * channels * patches, dim)
        global_valid = valid_patch.reshape(batch, minutes * channels * patches)
        global_valid_safe = global_valid.clone()
        empty_global = ~global_valid_safe.any(dim=1)
        global_valid_safe[empty_global, 0] = True
        global_query = self.global_queries[None].expand(batch, -1, -1)
        global_memory, _ = self.global_attn(
            self.norm_q(global_query),
            self.norm_kv(global_kv),
            self.norm_kv(global_kv),
            key_padding_mask=~global_valid_safe,
            need_weights=False,
        )
        global_memory = global_memory + self.global_ff(global_memory)
        global_mask = global_valid.any(dim=1, keepdim=True).expand(-1, self.global_tokens)
        level_ids = torch.cat(
            [
                torch.zeros(local_flat.shape[:2], dtype=torch.long, device=hidden.device),
                torch.ones(channel_flat.shape[:2], dtype=torch.long, device=hidden.device),
                torch.full(global_memory.shape[:2], 2, dtype=torch.long, device=hidden.device),
            ],
            dim=1,
        )
        local_modality_ids = modality_ids[:, None, :, None].expand(
            batch, minutes, channels, self.local_tokens_per_minute
        ).reshape(batch, -1)
        temporal_modality_ids = modality_ids[:, :, None].expand(
            batch, channels, self.temporal_tokens_per_channel
        ).reshape(batch, -1)
        global_modality_ids = torch.full(
            (batch, self.global_tokens), -1, dtype=torch.long, device=hidden.device
        )
        local_minute_ids = torch.arange(minutes, dtype=torch.long, device=hidden.device)[None, :, None, None].expand(
            batch, minutes, channels, self.local_tokens_per_minute
        ).reshape(batch, -1)
        summary_minute_ids = torch.full(
            (batch, channels * self.temporal_tokens_per_channel + self.global_tokens),
            -1,
            dtype=torch.long,
            device=hidden.device,
        )
        return SignalMemoryOutput(
            memory=torch.cat([all_channel_memory, global_memory], dim=1),
            memory_mask=torch.cat([all_channel_mask, global_mask], dim=1),
            memory_level_ids=level_ids,
            memory_modality_ids=torch.cat(
                [local_modality_ids, temporal_modality_ids, global_modality_ids], dim=1
            ),
            memory_minute_ids=torch.cat([local_minute_ids, summary_minute_ids], dim=1),
        )

    def encode(
        self,
        tokens: torch.Tensor,
        *,
        slot_mask: torch.Tensor,
        modality_ids: torch.Tensor,
        visible_patch_mask: torch.Tensor | None = None,
        valid_patch_mask: torch.Tensor | None = None,
    ) -> SignalMemoryOutput:
        """Encode the full patch grid into local/temporal/global memory tokens.

        Every valid one-second patch is available as key/value input to the
        local learned queries; only the emitted memory tokens are compact.
        """
        if visible_patch_mask is None:
            visible_patch_mask = slot_mask[:, :, :, None].expand(
                *slot_mask.shape, tokens.shape[3]
            )
        hidden = self.adapter(
            tokens,
            modality_ids=modality_ids,
            visible_patch_mask=visible_patch_mask.bool(),
        )
        memory = self.build_memory(
            hidden,
            slot_mask=slot_mask.bool(),
            visible_patch_mask=visible_patch_mask.bool(),
            modality_ids=modality_ids,
            valid_patch_mask=valid_patch_mask,
        )
        memory.patch_hidden = hidden
        return memory



@dataclass
class NumericMemoryOutput:
    memory: torch.Tensor
    memory_mask: torch.Tensor
    measure_ids: torch.Tensor


def _time_group_count(lookback_min, group_minutes):
    if not math.isfinite(lookback_min) or lookback_min <= 0:
        raise ValueError("Numeric lookback must be finite and positive")
    if not math.isfinite(group_minutes) or group_minutes <= 0:
        raise ValueError("Numeric group duration must be finite and positive")
    return int(math.ceil(lookback_min / group_minutes))


def pack_numeric_grid(tokens, mask, measure_ids, token_kind_ids, rel_time_min, *,
                      lookback_min, group_minutes=1.0):
    """Group by measure/time interval without averaging, interpolation or dropping points."""
    batch, count, dim = tokens.shape
    if mask.shape != (batch, count) or measure_ids.shape != mask.shape or token_kind_ids.shape != mask.shape:
        raise ValueError("Numeric token metadata must match [B,N]")
    events = mask.bool() & token_kind_ids.eq(NumericEventEncoder.EVENT_KIND) & measure_ids.gt(0)
    if rel_time_min.ndim != 2 or rel_time_min.shape[0] != batch or rel_time_min.shape[1] > count:
        raise ValueError("Numeric event times must match the encoder's event prefix")
    if events[:, rel_time_min.shape[1]:].any():
        raise ValueError("A numeric observation is missing its relative time")
    times = rel_time_min.new_zeros((batch, count))
    times[:, :rel_time_min.shape[1]] = rel_time_min
    minutes = _time_group_count(lookback_min, group_minutes)
    rows = []
    channels = patches = 0
    for b in range(batch):
        idx = events[b].nonzero().flatten()
        ids = torch.unique(measure_ids[b, idx], sorted=True)
        channels = max(channels, len(ids))
        if not len(idx):
            rows.append((idx, ids, idx, idx, idx))
            continue
        if not torch.isfinite(times[b, idx]).all():
            raise ValueError("Real numeric observations require finite relative times")
        minute = ((times[b, idx] + float(lookback_min)) / group_minutes).floor().long().clamp(0, minutes - 1)
        channel = torch.searchsorted(ids, measure_ids[b, idx])
        bucket = minute * len(ids) + channel
        order = torch.argsort(bucket, stable=True)
        idx, minute, channel = idx[order], minute[order], channel[order]
        _, lengths = torch.unique_consecutive(bucket[order], return_counts=True)
        starts = lengths.cumsum(0) - lengths
        rank = torch.arange(len(idx), device=tokens.device) - torch.repeat_interleave(starts, lengths)
        patches = max(patches, int(lengths.max()))
        rows.append((idx, ids, minute, channel, rank))
    grid = tokens.new_zeros((batch, minutes, channels, max(1, patches), dim))
    valid = torch.zeros(grid.shape[:-1], dtype=torch.bool, device=tokens.device)
    modalities = torch.full((batch, channels), -1, dtype=torch.long, device=tokens.device)
    for b, (idx, ids, minute, channel, rank) in enumerate(rows):
        modalities[b, :len(ids)] = ids
        if len(idx):
            grid[b, minute, channel, rank] = tokens[b, idx]
            valid[b, minute, channel, rank] = True
    return grid, valid, modalities


class ChannelwiseNumericSignalMemory(nn.Module):
    """Same local/temporal/global query hierarchy as the waveform memory, separate weights."""

    def __init__(self, *, input_dim, hidden_dim, num_measures, lookback_min,
                 max_events, heads=8, local_tokens_per_minute=2,
                 temporal_tokens_per_channel=16, global_tokens=64, dropout=0.0,
                 group_minutes=1.0):
        super().__init__()
        self.lookback_min = float(lookback_min)
        self.group_minutes = float(group_minutes)
        self.hierarchy = HierarchicalChannelwiseSignalMemory(
            input_dim=input_dim, hidden_dim=hidden_dim, num_modalities=num_measures,
            max_minutes=_time_group_count(self.lookback_min, self.group_minutes), max_patches_per_minute=max_events,
            heads=heads, local_tokens_per_minute=local_tokens_per_minute,
            temporal_tokens_per_channel=temporal_tokens_per_channel, global_tokens=global_tokens,
            dropout=dropout,
        )
        # Numeric's existing per-measure missingness summaries are retained separately.
        self.level_embed = nn.Embedding(4, hidden_dim)

    def forward(self, tokens, mask, measure_ids, *, token_kind_ids, rel_time_min):
        grid, valid, modalities = pack_numeric_grid(
            tokens, mask, measure_ids, token_kind_ids, rel_time_min, lookback_min=self.lookback_min,
            group_minutes=self.group_minutes,
        )
        encoded = self.hierarchy.encode(
            grid, slot_mask=valid.any(-1), modality_ids=modalities,
            visible_patch_mask=valid, valid_patch_mask=valid,
        )
        memory = encoded.memory + self.level_embed(encoded.memory_level_ids)
        memory_mask = encoded.memory_mask
        memory_ids = encoded.memory_modality_ids
        summary_positions = token_kind_ids.eq(NumericEventEncoder.SUMMARY_KIND)
        sizes = summary_positions.sum(-1)
        summaries = int(sizes.max())
        if summaries:
            summary = tokens.new_zeros((len(tokens), summaries, tokens.shape[-1]))
            ids = torch.zeros((len(tokens), summaries), dtype=torch.long, device=tokens.device)
            summary_mask = torch.zeros_like(ids, dtype=torch.bool)
            for b in range(len(tokens)):
                idx = summary_positions[b].nonzero().flatten()
                summary[b, :len(idx)] = tokens[b, idx]
                ids[b, :len(idx)] = measure_ids[b, idx]
                summary_mask[b, :len(idx)] = mask[b, idx]
            adapter = self.hierarchy.adapter
            summary = adapter.input_proj(adapter.input_norm(summary))
            summary = summary + adapter.modality_embed(ids + 1) + self.level_embed.weight[3]
            memory = torch.cat([memory, summary], dim=1)
            memory_mask = torch.cat([memory_mask, summary_mask], dim=1)
            memory_ids = torch.cat([memory_ids, ids], dim=1)
        return NumericMemoryOutput(memory, memory_mask, memory_ids)



class ModalityTokenProjector(nn.Module):
    """Project signal tokens to the hidden size of a causal LLM (the B sensor path)."""

    WAVEFORM_KIND = 0
    NUMERIC_KIND = 1

    def __init__(
        self,
        *,
        waveform_dim: int,
        numeric_dim: int,
        llm_hidden_size: int,
        max_waveform_minutes: int = 60,
        num_waveform_modalities: int = 0,
        max_waveform_patches_per_minute: int = 64,
        max_numeric_tokens: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.waveform_proj = nn.Sequential(
            nn.LayerNorm(waveform_dim),
            nn.Linear(waveform_dim, llm_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.numeric_proj = nn.Sequential(
            nn.LayerNorm(numeric_dim),
            nn.Linear(numeric_dim, llm_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.kind_embed = nn.Embedding(2, llm_hidden_size)
        self.wave_time_embed = nn.Embedding(max_waveform_minutes + 1, llm_hidden_size)
        # A patch is one second of a channel-minute.  This explicit within-minute
        # code keeps patch meaning independent of the channel count.
        self.wave_patch_time_embed = nn.Embedding(max_waveform_patches_per_minute, llm_hidden_size)
        self.wave_modality_embed = (
            nn.Embedding(int(num_waveform_modalities), llm_hidden_size)
            if int(num_waveform_modalities) > 0
            else None
        )
        self.numeric_position_embed = nn.Embedding(max_numeric_tokens + 1, llm_hidden_size)
        self.final_norm = nn.LayerNorm(llm_hidden_size)

    def project_waveform(
        self,
        waveform_tokens: torch.Tensor,
        *,
        waveform_mask: torch.Tensor | None = None,
        waveform_modality_ids: torch.Tensor | None = None,
        waveform_minute_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project waveform tokens.

        Accepts `[B, M, D]` or `[B, M, K, D]`. Output is flattened to
        `[B, M*K, H]`.
        """
        if waveform_tokens.ndim == 3:
            waveform_tokens = waveform_tokens[:, :, None, :]
        if waveform_tokens.ndim != 4:
            raise ValueError(f"Expected waveform tokens [B,M,D] or [B,M,K,D], got {tuple(waveform_tokens.shape)}")
        batch, minutes, per_min, _dim = waveform_tokens.shape
        flat = waveform_tokens.reshape(batch, minutes * per_min, _dim)
        out = self.waveform_proj(flat)
        if waveform_minute_ids is None:
            minute_ids = torch.arange(minutes, device=out.device).repeat_interleave(per_min)[None, :].expand(batch, -1)
        else:
            if waveform_minute_ids.shape != (batch, minutes):
                raise ValueError(
                    f"Expected waveform_minute_ids [B,M]={batch, minutes}, got {tuple(waveform_minute_ids.shape)}"
                )
            minute_ids = waveform_minute_ids.repeat_interleave(per_min, dim=1)
        minute_ids = minute_ids.clamp(max=self.wave_time_embed.num_embeddings - 1)
        out = out + self.kind_embed(torch.full((1, out.shape[1]), self.WAVEFORM_KIND, dtype=torch.long, device=out.device))
        out = out + self.wave_time_embed(minute_ids)
        patch_ids = torch.arange(per_min, device=out.device).repeat(minutes)[None, :].expand(batch, -1)
        patch_ids = patch_ids.clamp(max=self.wave_patch_time_embed.num_embeddings - 1)
        out = out + self.wave_patch_time_embed(patch_ids)
        if waveform_modality_ids is not None:
            if self.wave_modality_embed is None:
                raise ValueError("waveform_modality_ids were supplied but num_waveform_modalities=0")
            if waveform_modality_ids.shape != (batch, minutes):
                raise ValueError(
                    f"Expected waveform_modality_ids [B,M]={batch, minutes}, got {tuple(waveform_modality_ids.shape)}"
                )
            modality_ids = waveform_modality_ids.repeat_interleave(per_min, dim=1)
            valid_modality = modality_ids >= 0
            safe_ids = modality_ids.clamp(min=0, max=self.wave_modality_embed.num_embeddings - 1)
            out = out + self.wave_modality_embed(safe_ids) * valid_modality[:, :, None]
        if waveform_mask is None:
            mask = torch.ones(batch, out.shape[1], dtype=torch.bool, device=out.device)
        else:
            if waveform_mask.ndim == 2:
                waveform_mask = waveform_mask[:, :, None].expand(batch, minutes, per_min)
            mask = waveform_mask.reshape(batch, minutes * per_min).bool()
        return self.final_norm(out), mask

    def project_numeric(
        self,
        numeric_tokens: torch.Tensor,
        numeric_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.numeric_proj(numeric_tokens)
        positions = torch.arange(out.shape[1], device=out.device).clamp(max=self.numeric_position_embed.num_embeddings - 1)
        out = out + self.kind_embed(torch.full((1, out.shape[1]), self.NUMERIC_KIND, dtype=torch.long, device=out.device))
        out = out + self.numeric_position_embed(positions)[None, :, :]
        return self.final_norm(out), numeric_mask.bool()

    def forward(
        self,
        *,
        waveform_tokens: torch.Tensor | None = None,
        waveform_mask: torch.Tensor | None = None,
        waveform_modality_ids: torch.Tensor | None = None,
        waveform_minute_ids: torch.Tensor | None = None,
        numeric_tokens: torch.Tensor | None = None,
        numeric_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pieces: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        if waveform_tokens is not None:
            tok, mask = self.project_waveform(
                waveform_tokens,
                waveform_mask=waveform_mask,
                waveform_modality_ids=waveform_modality_ids,
                waveform_minute_ids=waveform_minute_ids,
            )
            pieces.append(tok)
            masks.append(mask)
        if numeric_tokens is not None:
            if numeric_mask is None:
                numeric_mask = torch.ones(numeric_tokens.shape[:2], dtype=torch.bool, device=numeric_tokens.device)
            tok, mask = self.project_numeric(numeric_tokens, numeric_mask)
            pieces.append(tok)
            masks.append(mask)
        if not pieces:
            raise ValueError("At least one of waveform_tokens or numeric_tokens is required.")
        return torch.cat(pieces, dim=1), torch.cat(masks, dim=1)


class HTokenProjector(nn.Module):
    """Project hierarchical waveform memory and numeric tokens into an LLM.

    H already preserves waveform modality and time identity inside its
    local/temporal attention hierarchy.  This projector adds an explicit
    memory-level code so the causal LLM can distinguish local evidence from
    channel summaries and cross-channel global state.
    """

    WAVEFORM_KIND = 0
    NUMERIC_KIND = 1

    def __init__(
        self,
        *,
        memory_dim: int,
        numeric_dim: int,
        llm_hidden_size: int,
        num_waveform_modalities: int,
        max_waveform_minutes: int,
        max_numeric_tokens: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.memory_proj = nn.Sequential(
            nn.LayerNorm(memory_dim),
            nn.Linear(memory_dim, llm_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.numeric_proj = nn.Sequential(
            nn.LayerNorm(numeric_dim),
            nn.Linear(numeric_dim, llm_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.kind_embed = nn.Embedding(2, llm_hidden_size)
        self.memory_level_embed = nn.Embedding(3, llm_hidden_size)
        self.memory_modality_embed = nn.Embedding(int(num_waveform_modalities), llm_hidden_size)
        self.memory_minute_embed = nn.Embedding(int(max_waveform_minutes), llm_hidden_size)
        self.numeric_position_embed = nn.Embedding(max_numeric_tokens + 1, llm_hidden_size)
        self.final_norm = nn.LayerNorm(llm_hidden_size)

    def forward(
        self,
        *,
        memory_tokens: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        memory_level_ids: torch.Tensor | None = None,
        memory_modality_ids: torch.Tensor | None = None,
        memory_minute_ids: torch.Tensor | None = None,
        numeric_tokens: torch.Tensor | None = None,
        numeric_mask: torch.Tensor | None = None,
        numeric_position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pieces: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        if memory_tokens is not None:
            if memory_mask is None:
                memory_mask = torch.ones(memory_tokens.shape[:2], dtype=torch.bool, device=memory_tokens.device)
            if memory_level_ids is None:
                memory_level_ids = torch.zeros(memory_tokens.shape[:2], dtype=torch.long, device=memory_tokens.device)
            if memory_level_ids.shape != memory_tokens.shape[:2]:
                raise ValueError(
                    "Expected H memory_level_ids [B,N] matching memory tokens, got "
                    f"{tuple(memory_level_ids.shape)} for {tuple(memory_tokens.shape)}"
                )
            if memory_modality_ids is not None and memory_modality_ids.shape != memory_tokens.shape[:2]:
                raise ValueError("H memory_modality_ids must match memory token shape [B,N].")
            if memory_minute_ids is not None and memory_minute_ids.shape != memory_tokens.shape[:2]:
                raise ValueError("H memory_minute_ids must match memory token shape [B,N].")
            memory = self.memory_proj(memory_tokens)
            kind = torch.full((1, memory.shape[1]), self.WAVEFORM_KIND, dtype=torch.long, device=memory.device)
            levels = memory_level_ids.clamp(min=0, max=self.memory_level_embed.num_embeddings - 1)
            memory = memory + self.kind_embed(kind) + self.memory_level_embed(levels)
            if memory_modality_ids is not None:
                valid_modality = memory_modality_ids >= 0
                safe_modality = memory_modality_ids.clamp(
                    min=0, max=self.memory_modality_embed.num_embeddings - 1
                )
                memory = memory + self.memory_modality_embed(safe_modality) * valid_modality[:, :, None]
            if memory_minute_ids is not None:
                valid_minute = memory_minute_ids >= 0
                safe_minute = memory_minute_ids.clamp(min=0, max=self.memory_minute_embed.num_embeddings - 1)
                memory = memory + self.memory_minute_embed(safe_minute) * valid_minute[:, :, None]
            pieces.append(self.final_norm(memory))
            masks.append(memory_mask.bool())
        if numeric_tokens is not None:
            if numeric_mask is None:
                numeric_mask = torch.ones(numeric_tokens.shape[:2], dtype=torch.bool, device=numeric_tokens.device)
            numeric = self.numeric_proj(numeric_tokens)
            if numeric_position_ids is None:
                positions = torch.arange(numeric.shape[1], device=numeric.device)[None, :]
            else:
                if numeric_position_ids.shape != numeric_tokens.shape[:2]:
                    raise ValueError("numeric_position_ids must match numeric tokens [B,N]")
                positions = numeric_position_ids
            positions = positions.clamp(min=0, max=self.numeric_position_embed.num_embeddings - 1)
            kind = torch.full((1, numeric.shape[1]), self.NUMERIC_KIND, dtype=torch.long, device=numeric.device)
            numeric = numeric + self.kind_embed(kind) + self.numeric_position_embed(positions)
            pieces.append(self.final_norm(numeric))
            masks.append(numeric_mask.bool())
        if not pieces:
            raise ValueError("At least one of H waveform memory or numeric tokens is required.")
        return torch.cat(pieces, dim=1), torch.cat(masks, dim=1)
