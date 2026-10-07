"""Action heads, evidence selector, label ranker, caption decoder, and the multimodal LM."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def finite_clamp(x: torch.Tensor, *, clamp: float = 80.0) -> torch.Tensor:
    """Clamp to ``[-clamp, clamp]`` and replace non-finite values."""
    return torch.nan_to_num(x, nan=0.0, posinf=float(clamp), neginf=-float(clamp)).clamp(-float(clamp), float(clamp))


def action_group_need_indices(
    label_names: Sequence[str], action_group_types: Mapping[str, str],
) -> list[int]:
    """Return Intervention=0 or Assessment=1 for each action group."""
    assessment_types = {"assessment", "diagnostic", "diagnostic_workup"}
    return [
        int(str(action_group_types.get(name) or "").lower() in assessment_types)
        for name in label_names
    ]



@dataclass
class ActionEvidenceIndex:
    """Vocabulary of evidence fact keys and their group compatibility."""

    fact_keys: list[str]
    fact_to_id: dict[str, int]
    # ``group -> fact -> strength``; facts without a group association never
    # enter the caption profile.
    group_fact_strengths: dict[str, dict[str, float]] = field(default_factory=dict)

    def group_fact_strength_tensor(self, group_names: list[str], *, device: torch.device) -> torch.Tensor:
        """Return the group-to-fact compatibility matrix [G, F]."""
        matrix = torch.zeros((len(group_names), len(self.fact_keys)), dtype=torch.float32, device=device)
        for group_id, group in enumerate(group_names):
            for key, strength in self.group_fact_strengths.get(str(group), {}).items():
                fact_id = self.fact_to_id.get(str(key))
                if fact_id is not None:
                    matrix[group_id, fact_id] = float(strength)
        return matrix


class ActionConditionedEvidenceSelector(nn.Module):
    """Predict evidence fact keys conditioned on predicted action groups."""

    def __init__(self, *, hidden_size: int, num_action_groups: int, num_evidence_facts: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_evidence_facts = int(num_evidence_facts)
        self.group_embedding = nn.Parameter(torch.empty(num_action_groups, hidden_size))
        nn.init.normal_(self.group_embedding, mean=0.0, std=0.02)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_size, max(1, self.num_evidence_facts)),
        )

    def forward(self, decision_hidden: torch.Tensor, group_logits: torch.Tensor) -> torch.Tensor:
        group_prob = torch.sigmoid(group_logits.float())
        action_context = torch.matmul(group_prob, self.group_embedding.float())
        fused = torch.cat([decision_hidden.float(), action_context], dim=-1)
        return finite_clamp(self.net(fused), clamp=50.0)



class DescriptorActionLabelRanker(nn.Module):
    """Rank fine actions from their text descriptions.

    Candidate label descriptions are tokenized once, then encoded from the LM
    input embedding table through a small shared projection.  The decision state
    is projected to a query and scored against the candidate vectors of the
    selected action group by cosine similarity.
    """

    def __init__(
        self,
        *,
        tokenizer: Any,
        label_texts: list[str],
        hidden_size: int,
        temperature: float = 0.07,
        max_description_length: int = 64,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.max_description_length = int(max_description_length)
        self.query = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.descriptor_projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        tokenized = self._tokenize(tokenizer, list(label_texts))
        self.register_buffer("label_token_ids", tokenized["input_ids"], persistent=False)
        self.register_buffer("label_token_mask", tokenized["attention_mask"].bool(), persistent=False)

    def _tokenize(self, tokenizer: Any, texts: list[str]) -> dict[str, torch.Tensor]:
        if not texts:
            pad_id = int(tokenizer.pad_token_id or tokenizer.eos_token_id or 0)
            return {
                "input_ids": torch.full((1, 1), pad_id, dtype=torch.long),
                "attention_mask": torch.zeros((1, 1), dtype=torch.long),
            }
        return tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_description_length,
            return_tensors="pt",
        )

    def label_vectors(self, label_ids: torch.Tensor, embedding_layer: nn.Module) -> torch.Tensor:
        """Encode catalog entries selected by integer IDs with the LM input embedding table."""
        token_ids = self.label_token_ids.index_select(0, label_ids).to(label_ids.device)
        token_mask = self.label_token_mask.index_select(0, label_ids).to(label_ids.device)
        embeddings = embedding_layer(token_ids).float()
        weights = token_mask.unsqueeze(-1).to(embeddings.dtype)
        pooled = (embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.descriptor_projection(pooled)



@dataclass(frozen=True)
class PreparedCaptionContext:
    """Fixed, normalized decoder memory for one generation batch."""

    normalized_profile_memory: torch.Tensor
    profile_mask: torch.Tensor
    normalized_detail_memory: torch.Tensor | None
    detail_mask: torch.Tensor | None


class ResidualActionConditionedCaptionDecoder(nn.Module):
    """Inject a detached top-k action/evidence profile into caption states."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_action_groups: int,
        num_evidence_facts: int,
        max_groups: int = 3,
        facts_per_group: int = 3,
        need_threshold: float = 0.5,
        group_threshold: float = 0.15,
        fact_threshold: float = 0.5,
        dropout: float = 0.0,
        group_fact_compatibility: torch.Tensor | None = None,
        enable_detail_selector: bool = False,
        waveform_detail_dim: int = 0,
        numeric_detail_dim: int = 0,
        waveform_detail_tokens_per_channel: int = 16,
        numeric_detail_tokens_per_measure: int = 2,
        waveform_detail_max_tokens: int = 0,
        numeric_detail_max_tokens: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_action_groups = int(num_action_groups)
        self.num_evidence_facts = int(num_evidence_facts)
        self.max_groups = max(1, int(max_groups))
        self.facts_per_group = max(0, int(facts_per_group))
        self.need_threshold = float(need_threshold)
        self.group_threshold = float(group_threshold)
        self.fact_threshold = float(fact_threshold)
        self.enable_detail_selector = bool(enable_detail_selector)
        self.waveform_detail_tokens_per_channel = max(0, int(waveform_detail_tokens_per_channel))
        self.numeric_detail_tokens_per_measure = max(0, int(numeric_detail_tokens_per_measure))
        self.waveform_detail_max_tokens = max(0, int(waveform_detail_max_tokens))
        self.numeric_detail_max_tokens = max(0, int(numeric_detail_max_tokens))

        # Profile tokens start from LM embeddings of descriptor text, filled in
        # by ``set_descriptor_embeddings`` after the checkpoint is loaded.
        self.register_buffer(
            "group_descriptor_embeddings",
            torch.zeros((self.num_action_groups, self.hidden_size), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "fact_descriptor_embeddings",
            torch.zeros((max(1, self.num_evidence_facts), self.hidden_size), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer("descriptors_initialized", torch.tensor(False), persistent=True)

        self.need_present_embedding = nn.Parameter(torch.empty(self.hidden_size))
        self.no_action_embedding = nn.Parameter(torch.empty(self.hidden_size))
        self.score_projection = nn.Sequential(nn.Linear(1, self.hidden_size), nn.Tanh())
        self.profile_norm = nn.LayerNorm(self.hidden_size)
        self.query_norm = nn.LayerNorm(self.hidden_size)
        self.cross_attention = nn.MultiheadAttention(
            self.hidden_size,
            num_heads=8,
            dropout=float(dropout),
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, 4 * self.hidden_size),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(4 * self.hidden_size, self.hidden_size),
        )
        self.residual_projection = nn.Linear(self.hidden_size, self.hidden_size)
        self.residual_gate_logit = nn.Parameter(torch.tensor(-3.0))
        self.dropout = nn.Dropout(float(dropout))

        nn.init.normal_(self.need_present_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.no_action_embedding, mean=0.0, std=0.02)
        nn.init.zeros_(self.residual_projection.weight)
        nn.init.zeros_(self.residual_projection.bias)

        self.waveform_detail_dim = int(waveform_detail_dim)
        self.numeric_detail_dim = int(numeric_detail_dim)
        if self.enable_detail_selector:
            if self.waveform_detail_dim <= 0 or self.numeric_detail_dim <= 0:
                raise ValueError("Caption detail selector requires waveform_detail_dim and numeric_detail_dim.")
            self.detail_context_norm = nn.LayerNorm(self.hidden_size)
            self.detail_profile_projection = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.waveform_query_projection = nn.Linear(self.hidden_size, self.waveform_detail_dim, bias=False)
            self.numeric_query_projection = nn.Linear(self.hidden_size, self.numeric_detail_dim, bias=False)
            self.waveform_candidate_norm = nn.LayerNorm(self.waveform_detail_dim)
            self.numeric_candidate_norm = nn.LayerNorm(self.numeric_detail_dim)
            self.waveform_value_projection = nn.Sequential(
                nn.LayerNorm(self.waveform_detail_dim), nn.Linear(self.waveform_detail_dim, self.hidden_size)
            )
            self.numeric_value_projection = nn.Sequential(
                nn.LayerNorm(self.numeric_detail_dim), nn.Linear(self.numeric_detail_dim, self.hidden_size)
            )
            self.waveform_detail_type = nn.Parameter(torch.empty(self.hidden_size))
            self.numeric_detail_type = nn.Parameter(torch.empty(self.hidden_size))
            self.detail_query_norm = nn.LayerNorm(self.hidden_size)
            self.detail_memory_norm = nn.LayerNorm(self.hidden_size)
            self.detail_cross_attention = nn.MultiheadAttention(
                self.hidden_size, num_heads=8, dropout=float(dropout), batch_first=True
            )
            self.detail_ffn = nn.Sequential(
                nn.LayerNorm(self.hidden_size),
                nn.Linear(self.hidden_size, 4 * self.hidden_size),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(4 * self.hidden_size, self.hidden_size),
            )
            self.detail_residual_projection = nn.Linear(self.hidden_size, self.hidden_size)
            self.detail_residual_gate_logit = nn.Parameter(torch.tensor(-3.0))
            nn.init.normal_(self.waveform_detail_type, mean=0.0, std=0.02)
            nn.init.normal_(self.numeric_detail_type, mean=0.0, std=0.02)
            nn.init.zeros_(self.detail_residual_projection.weight)
            nn.init.zeros_(self.detail_residual_projection.bias)
        else:
            self.detail_context_norm = None
            self.detail_profile_projection = None
            self.waveform_query_projection = None
            self.numeric_query_projection = None
            self.waveform_candidate_norm = None
            self.numeric_candidate_norm = None
            self.waveform_value_projection = None
            self.numeric_value_projection = None
            self.waveform_detail_type = None
            self.numeric_detail_type = None
            self.detail_query_norm = None
            self.detail_memory_norm = None
            self.detail_cross_attention = None
            self.detail_ffn = None
            self.detail_residual_projection = None
            self.detail_residual_gate_logit = None

        if group_fact_compatibility is None:
            group_fact_compatibility = torch.zeros(
                (self.num_action_groups, self.num_evidence_facts), dtype=torch.float32
            )
        expected = (self.num_action_groups, self.num_evidence_facts)
        if tuple(group_fact_compatibility.shape) != expected:
            raise ValueError(
                f"group_fact_compatibility shape={tuple(group_fact_compatibility.shape)} expected={expected}"
            )
        self.register_buffer("group_fact_compatibility", group_fact_compatibility.float(), persistent=True)

    @torch.no_grad()
    def set_descriptor_embeddings(self, group_embeddings: torch.Tensor, fact_embeddings: torch.Tensor) -> None:
        if tuple(group_embeddings.shape) != tuple(self.group_descriptor_embeddings.shape):
            raise ValueError(
                f"group descriptor shape={tuple(group_embeddings.shape)} expected={tuple(self.group_descriptor_embeddings.shape)}"
            )
        expected_facts = (max(1, self.num_evidence_facts), self.hidden_size)
        if tuple(fact_embeddings.shape) != expected_facts:
            raise ValueError(f"fact descriptor shape={tuple(fact_embeddings.shape)} expected={expected_facts}")
        self.group_descriptor_embeddings.copy_(group_embeddings.float().to(self.group_descriptor_embeddings.device))
        self.fact_descriptor_embeddings.copy_(fact_embeddings.float().to(self.fact_descriptor_embeddings.device))
        self.descriptors_initialized.fill_(True)

    @torch.no_grad()
    def build_profile(
        self,
        *,
        need_logits: torch.Tensor,
        group_logits: torch.Tensor,
        evidence_logits: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the detached hard profile: need token, top groups, top facts per group.

        If need is positive but no group clears the threshold, the
        highest-scoring group is used.
        """
        group_prob = torch.sigmoid(group_logits.detach().float()).clamp(0.0, 1.0)
        need_prob = torch.sigmoid(need_logits.detach().float()).clamp(0.0, 1.0)
        batch = group_prob.shape[0]
        device = group_prob.device
        dtype = group_logits.dtype
        score_dtype = self.score_projection[0].weight.dtype

        def project_score(values: torch.Tensor) -> torch.Tensor:
            return self.score_projection(values.to(dtype=score_dtype)).float()

        group_ids = torch.full((batch, self.max_groups), -1, dtype=torch.long, device=device)
        group_scores = torch.zeros((batch, self.max_groups), dtype=torch.float32, device=device)
        group_mask = torch.zeros((batch, self.max_groups), dtype=torch.bool, device=device)
        for row in range(batch):
            if float(need_prob[row]) < self.need_threshold:
                continue
            candidates = torch.nonzero(group_prob[row] >= self.group_threshold, as_tuple=False).flatten()
            if candidates.numel() == 0:
                candidates = torch.argmax(group_prob[row]).reshape(1)
            ranked = candidates[torch.argsort(group_prob[row, candidates], descending=True)][: self.max_groups]
            count = int(ranked.numel())
            group_ids[row, :count] = ranked
            group_scores[row, :count] = group_prob[row, ranked]
            group_mask[row, :count] = True

        need_token = torch.where(
            (need_prob >= self.need_threshold).unsqueeze(-1),
            self.need_present_embedding.float().unsqueeze(0),
            self.no_action_embedding.float().unsqueeze(0),
        ) + project_score(need_prob.unsqueeze(-1))
        tokens = [need_token.unsqueeze(1)]
        masks = [torch.ones((batch, 1), dtype=torch.bool, device=device)]

        safe_group_ids = group_ids.clamp_min(0)
        group_base = self.group_descriptor_embeddings.to(device=device)[safe_group_ids]
        tokens.append(group_base + project_score(group_scores.unsqueeze(-1)))
        masks.append(group_mask)

        if self.facts_per_group > 0 and self.num_evidence_facts > 0 and evidence_logits is not None:
            fact_prob = torch.sigmoid(evidence_logits.detach().float()).clamp(0.0, 1.0)
            fact_ids = torch.full(
                (batch, self.max_groups * self.facts_per_group), -1, dtype=torch.long, device=device
            )
            fact_scores = torch.zeros_like(fact_ids, dtype=torch.float32)
            fact_mask = torch.zeros_like(fact_ids, dtype=torch.bool)
            compatibility = self.group_fact_compatibility.to(device=device)
            for row in range(batch):
                for slot in range(self.max_groups):
                    if not bool(group_mask[row, slot]):
                        continue
                    compatible = torch.nonzero(compatibility[int(group_ids[row, slot])] > 0, as_tuple=False).flatten()
                    if compatible.numel() == 0:
                        continue
                    scores = fact_prob[row, compatible] * compatibility[int(group_ids[row, slot]), compatible]
                    ranked_order = torch.argsort(scores, descending=True)
                    destination = slot * self.facts_per_group
                    kept = 0
                    for candidate_idx in ranked_order.tolist():
                        score = float(scores[candidate_idx])
                        if score < self.fact_threshold:
                            continue
                        fact_ids[row, destination + kept] = int(compatible[candidate_idx])
                        fact_scores[row, destination + kept] = score
                        fact_mask[row, destination + kept] = True
                        kept += 1
                        if kept >= self.facts_per_group:
                            break
            fact_base = self.fact_descriptor_embeddings.to(device=device)[fact_ids.clamp_min(0)]
            tokens.append(fact_base + project_score(fact_scores.unsqueeze(-1)))
            masks.append(fact_mask)

        return torch.cat(tokens, dim=1).to(dtype=dtype), torch.cat(masks, dim=1)

    def _select_detail_source(
        self,
        *,
        candidates: torch.Tensor | None,
        candidate_mask: torch.Tensor | None,
        candidate_group_ids: torch.Tensor | None,
        query: torch.Tensor,
        candidate_norm: nn.Module,
        value_projection: nn.Module,
        type_embedding: torch.Tensor,
        tokens_per_group: int,
        max_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return top-k-per-group detail values gated by their retrieval scores."""
        batch = int(query.shape[0])
        empty = query.new_zeros((batch, 0, self.hidden_size))
        empty_mask = torch.zeros((batch, 0), dtype=torch.bool, device=query.device)
        if (
            candidates is None
            or candidate_mask is None
            or candidate_group_ids is None
            or tokens_per_group <= 0
            or candidates.numel() == 0
        ):
            return empty, empty_mask
        if (
            candidates.ndim != 3
            or candidate_mask.shape != candidates.shape[:2]
            or candidate_group_ids.shape != candidates.shape[:2]
        ):
            raise ValueError("Caption detail candidates/masks/group IDs must be [B,N,*] with matching [B,N] metadata.")
        if int(candidates.shape[0]) != batch:
            raise ValueError("Caption detail candidate batch does not match decoder batch.")

        score_dtype = candidate_norm.weight.dtype if isinstance(candidate_norm, nn.LayerNorm) else candidates.dtype
        keys = candidate_norm(candidates.to(dtype=score_dtype))
        scores = (keys.float() * query.float().unsqueeze(1)).sum(dim=-1) / (float(keys.shape[-1]) ** 0.5)
        valid_mask = candidate_mask.bool() & candidate_group_ids.ge(0)
        rows: list[torch.Tensor] = []
        row_scores: list[torch.Tensor] = []
        for row in range(batch):
            valid = torch.nonzero(valid_mask[row], as_tuple=False).flatten()
            if valid.numel() == 0:
                rows.append(candidates.new_zeros((0, candidates.shape[-1])))
                row_scores.append(scores.new_zeros((0,)))
                continue
            chosen: list[torch.Tensor] = []
            for group_id in torch.unique(candidate_group_ids[row, valid]).tolist():
                group_indices = valid[candidate_group_ids[row, valid] == int(group_id)]
                count = min(int(tokens_per_group), int(group_indices.numel()))
                chosen.append(group_indices[torch.topk(scores[row, group_indices], k=count).indices])
            selected = torch.cat(chosen, dim=0) if chosen else valid[:0]
            if max_tokens > 0 and selected.numel() > max_tokens:
                selected = selected[torch.topk(scores[row, selected], k=max_tokens).indices]
            rows.append(candidates[row].index_select(0, selected))
            row_scores.append(scores[row].index_select(0, selected))

        max_selected = max((int(row.shape[0]) for row in rows), default=0)
        if max_selected == 0:
            return empty, empty_mask
        selected_values = candidates.new_zeros((batch, max_selected, candidates.shape[-1]))
        selected_scores = scores.new_zeros((batch, max_selected))
        selected_mask = torch.zeros((batch, max_selected), dtype=torch.bool, device=query.device)
        for row, (values, row_score) in enumerate(zip(rows, row_scores)):
            count = int(values.shape[0])
            if count:
                selected_values[row, :count] = values
                selected_scores[row, :count] = row_score
                selected_mask[row, :count] = True
        gates = torch.sigmoid(selected_scores) * selected_mask.to(dtype=selected_scores.dtype)
        values = value_projection(selected_values.to(dtype=score_dtype))
        values = values * gates.to(dtype=values.dtype).unsqueeze(-1)
        values = values + type_embedding.to(dtype=values.dtype).view(1, 1, -1)
        values = values * selected_mask.to(dtype=values.dtype).unsqueeze(-1)
        return values, selected_mask

    def _build_detail_memory(
        self,
        *,
        decision_hidden: torch.Tensor | None,
        profile_tokens: torch.Tensor,
        profile_mask: torch.Tensor,
        detail_candidates: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self.enable_detail_selector or detail_candidates is None or decision_hidden is None:
            return None, None
        profile_weight = profile_mask.to(dtype=profile_tokens.dtype).unsqueeze(-1)
        profile_summary = (profile_tokens * profile_weight).sum(dim=1) / profile_weight.sum(dim=1).clamp_min(1.0)
        projection_dtype = self.detail_profile_projection.weight.dtype
        projected_profile = self.detail_profile_projection(profile_summary.to(dtype=projection_dtype))
        context = self.detail_context_norm(
            (decision_hidden.to(dtype=profile_summary.dtype) + projected_profile).to(
                dtype=self.detail_context_norm.weight.dtype
            )
        )
        waveform, waveform_mask = self._select_detail_source(
            candidates=detail_candidates.get("waveform_candidates"),
            candidate_mask=detail_candidates.get("waveform_mask"),
            candidate_group_ids=detail_candidates.get("waveform_channel_ids"),
            query=self.waveform_query_projection(context),
            candidate_norm=self.waveform_candidate_norm,
            value_projection=self.waveform_value_projection,
            type_embedding=self.waveform_detail_type,
            tokens_per_group=self.waveform_detail_tokens_per_channel,
            max_tokens=self.waveform_detail_max_tokens,
        )
        numeric, numeric_mask = self._select_detail_source(
            candidates=detail_candidates.get("numeric_candidates"),
            candidate_mask=detail_candidates.get("numeric_mask"),
            candidate_group_ids=detail_candidates.get("numeric_measure_ids"),
            query=self.numeric_query_projection(context),
            candidate_norm=self.numeric_candidate_norm,
            value_projection=self.numeric_value_projection,
            type_embedding=self.numeric_detail_type,
            tokens_per_group=self.numeric_detail_tokens_per_measure,
            max_tokens=self.numeric_detail_max_tokens,
        )
        memory = torch.cat([waveform, numeric], dim=1)
        mask = torch.cat([waveform_mask, numeric_mask], dim=1)
        if memory.shape[1] == 0:
            return None, None
        return memory, mask

    @torch.no_grad()
    def prepare_caption_context(
        self,
        *,
        profile_tokens: torch.Tensor,
        profile_mask: torch.Tensor,
        decision_hidden: torch.Tensor | None,
        detail_candidates: dict[str, torch.Tensor] | None,
    ) -> PreparedCaptionContext:
        """Prepare the fixed decoder memory once per generation batch."""
        compute_dtype = self.query_norm.weight.dtype
        profile_tokens = profile_tokens.detach()
        profile_mask = profile_mask.detach().bool()
        detail_memory, detail_mask = self._build_detail_memory(
            decision_hidden=decision_hidden,
            profile_tokens=profile_tokens,
            profile_mask=profile_mask,
            detail_candidates=detail_candidates,
        )
        normalized_detail_memory = None
        if detail_memory is not None:
            normalized_detail_memory = self.detail_memory_norm(detail_memory.to(dtype=compute_dtype))
        return PreparedCaptionContext(
            normalized_profile_memory=self.profile_norm(profile_tokens.to(dtype=compute_dtype)),
            profile_mask=profile_mask,
            normalized_detail_memory=normalized_detail_memory,
            detail_mask=detail_mask,
        )

    @torch.no_grad()
    def decode_with_context(
        self, caption_hidden: torch.Tensor, prepared_context: PreparedCaptionContext
    ) -> torch.Tensor:
        """Return caption states plus the gated profile (and detail) residuals."""
        # Cached decoding may expose fp32 hidden states while this branch runs
        # in bf16; keep the residual computation in the decoder's dtype.
        compute_dtype = self.query_norm.weight.dtype
        caption_hidden = caption_hidden.to(dtype=compute_dtype)
        attended, _ = self.cross_attention(
            self.query_norm(caption_hidden),
            prepared_context.normalized_profile_memory,
            prepared_context.normalized_profile_memory,
            key_padding_mask=~prepared_context.profile_mask.to(dtype=torch.bool, device=caption_hidden.device),
            need_weights=False,
        )
        delta = self.residual_projection(self.dropout(attended + self.dropout(self.ffn(attended))))
        gate = torch.sigmoid(self.residual_gate_logit).to(dtype=caption_hidden.dtype)
        output = caption_hidden + gate * delta
        detail_memory = prepared_context.normalized_detail_memory
        detail_mask = prepared_context.detail_mask
        if detail_memory is not None and detail_mask is not None:
            # MultiheadAttention cannot consume an all-masked row.
            safe_mask = detail_mask.clone()
            safe_mask[~safe_mask.any(dim=1), 0] = True
            detail_attended, _ = self.detail_cross_attention(
                self.detail_query_norm(caption_hidden),
                detail_memory,
                detail_memory,
                key_padding_mask=~safe_mask,
                need_weights=False,
            )
            detail_delta = self.detail_residual_projection(
                self.dropout(detail_attended + self.dropout(self.detail_ffn(detail_attended)))
            )
            detail_gate = torch.sigmoid(self.detail_residual_gate_logit).to(dtype=caption_hidden.dtype)
            output = output + detail_gate * detail_delta
        return finite_clamp(output, clamp=100.0)



def get_hidden_size(config: Any) -> int:
    for name in ("hidden_size", "n_embd", "d_model"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    text_config = getattr(config, "text_config", None)
    if text_config is not None and text_config is not config:
        return get_hidden_size(text_config)
    raise ValueError("Could not infer hidden size from model config.")


def prompt_only_state_mask(
    attention_mask: torch.Tensor,
    prompt_end_indices: torch.Tensor,
    prompt_length: int,
) -> torch.Tensor:
    """Mask positions that trail shorter prompts in a padded batch."""
    positions = torch.arange(int(prompt_length), device=attention_mask.device).unsqueeze(0)
    return attention_mask[:, :prompt_length] * (
        positions <= prompt_end_indices.to(device=attention_mask.device, dtype=torch.long).unsqueeze(1)
    ).to(dtype=attention_mask.dtype)


def residual_decode_position_ids(
    prompt_end_indices: torch.Tensor,
    *,
    signal_length: int,
    generated_tokens: int,
) -> torch.Tensor:
    """Continue each batched stream directly after its own prompt position."""
    return (
        prompt_end_indices.to(dtype=torch.long)
        + int(signal_length)
        + int(generated_tokens)
    ).unsqueeze(1)


def repeated_suffix_token_count(
    token_ids: list[int],
    *,
    max_span: int = 32,
    min_repeated_tokens: int = 12,
    min_repeats: int = 3,
) -> int:
    """Return the repeated suffix length when decoding enters a local loop."""
    length = len(token_ids)
    for span in range(1, min(int(max_span), length // int(min_repeats)) + 1):
        repeats = max(int(min_repeats), (int(min_repeated_tokens) + span - 1) // span)
        repeated = span * repeats
        if repeated > length:
            continue
        block = token_ids[-span:]
        if all(
            token_ids[-(repeat + 1) * span : -repeat * span] == block
            for repeat in range(1, repeats)
        ):
            return repeated
    return 0


def any_action_logit_from_typed_need(need_type_logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sigmoid(finite_clamp(need_type_logits, clamp=50.0))
    any_probability = 1.0 - (1.0 - probabilities).prod(dim=-1)
    return torch.logit(any_probability.clamp(min=1e-6, max=1.0 - 1e-6))


def hierarchical_group_logits(
    need_logits: torch.Tensor,
    conditional_group_logits: torch.Tensor,
    group_need_type_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return logit(P(need action) * P(group | need action)) stably."""
    if need_logits.ndim == 2:
        if group_need_type_indices is None:
            raise ValueError("Typed need logits require group_need_type_indices for hierarchical gating.")
        selected_need = need_logits.index_select(
            1, group_need_type_indices.to(device=need_logits.device, dtype=torch.long)
        )
        log_probability = F.logsigmoid(selected_need) + F.logsigmoid(conditional_group_logits)
    else:
        log_probability = F.logsigmoid(need_logits).unsqueeze(-1) + F.logsigmoid(conditional_group_logits)
    return log_probability - torch.log(-torch.expm1(log_probability))


class MultimodalLLMWithActionHeads(nn.Module):
    """Causal LM with need/action-group heads and an optional signal-token prefix.

    Waveform/numeric tokens are projected to the LLM hidden size and prepended
    to the textual prompt through ``inputs_embeds``.  The decision state at the
    prompt end feeds the action heads; the optional residual caption decoder
    conditions caption generation on the predicted action profile.
    """

    def __init__(
        self,
        lm: nn.Module,
        num_action_groups: int,
        dropout: float,
        num_evidence_facts: int = 0,
        hierarchical_action_heads: bool = False,
        enable_action_conditioned_caption_decoder: bool = False,
        caption_decoder_mode: str = "residual_v2",
        caption_profile_max_groups: int = 3,
        caption_profile_facts_per_group: int = 3,
        caption_profile_need_threshold: float = 0.5,
        caption_profile_group_threshold: float = 0.15,
        caption_profile_fact_threshold: float = 0.5,
        group_fact_compatibility: torch.Tensor | None = None,
        enable_caption_detail_selector: bool = False,
        waveform_detail_dim: int = 0,
        numeric_detail_dim: int = 0,
        caption_waveform_detail_tokens_per_channel: int = 16,
        caption_numeric_detail_tokens_per_measure: int = 2,
        caption_waveform_detail_max_tokens: int = 0,
        caption_numeric_detail_max_tokens: int = 0,
        need_head_mode: str = "legacy",
        group_need_type_indices: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.lm = lm
        hidden_size = get_hidden_size(lm.config)
        self.hidden_size = hidden_size
        self.hierarchical_action_heads = bool(hierarchical_action_heads)
        self.need_head_mode = str(need_head_mode)
        if self.need_head_mode not in {"legacy", "typed_multilabel"}:
            raise ValueError(f"Unsupported need_head_mode={self.need_head_mode}")
        self.caption_decoder_mode = str(caption_decoder_mode or "residual_v2")
        if enable_action_conditioned_caption_decoder and self.caption_decoder_mode != "residual_v2":
            raise ValueError(f"Unsupported caption decoder: {self.caption_decoder_mode}")
        self.dropout = nn.Dropout(dropout)
        self.need_head = nn.Linear(hidden_size, 2 if self.need_head_mode == "typed_multilabel" else 1)
        if group_need_type_indices is None:
            group_need_type_indices = torch.zeros(num_action_groups, dtype=torch.long)
        self.register_buffer(
            "group_need_type_indices",
            group_need_type_indices.to(dtype=torch.long),
            persistent=False,
        )
        self.group_head = nn.Linear(hidden_size, num_action_groups)
        if enable_action_conditioned_caption_decoder:
            self.caption_memory = ResidualActionConditionedCaptionDecoder(
                hidden_size=hidden_size,
                num_action_groups=num_action_groups,
                num_evidence_facts=num_evidence_facts,
                max_groups=caption_profile_max_groups,
                facts_per_group=caption_profile_facts_per_group,
                need_threshold=caption_profile_need_threshold,
                group_threshold=caption_profile_group_threshold,
                fact_threshold=caption_profile_fact_threshold,
                dropout=dropout,
                group_fact_compatibility=group_fact_compatibility,
                enable_detail_selector=enable_caption_detail_selector,
                waveform_detail_dim=waveform_detail_dim,
                numeric_detail_dim=numeric_detail_dim,
                waveform_detail_tokens_per_channel=caption_waveform_detail_tokens_per_channel,
                numeric_detail_tokens_per_measure=caption_numeric_detail_tokens_per_measure,
                waveform_detail_max_tokens=caption_waveform_detail_max_tokens,
                numeric_detail_max_tokens=caption_numeric_detail_max_tokens,
            )
        else:
            self.caption_memory = None
        self.evidence_selector = (
            ActionConditionedEvidenceSelector(
                hidden_size=hidden_size,
                num_action_groups=num_action_groups,
                num_evidence_facts=num_evidence_facts,
                dropout=dropout,
            )
            if num_evidence_facts > 0
            else None
        )

    @torch.no_grad()
    def configure_caption_profile_descriptors(
        self,
        tokenizer: Any,
        group_names: list[str],
        fact_keys: list[str],
    ) -> None:
        """Initialize caption profile tokens from the LM embeddings of descriptor text."""
        if self.caption_memory is None:
            return
        embedding = self.lm.get_input_embeddings()

        def encode(texts: list[str]) -> torch.Tensor:
            if not texts:
                return torch.zeros((1, self.hidden_size), dtype=torch.float32, device=embedding.weight.device)
            output: list[torch.Tensor] = []
            for start in range(0, len(texts), 64):
                tokenized = tokenizer(
                    texts[start : start + 64],
                    padding=True,
                    truncation=True,
                    max_length=48,
                    return_tensors="pt",
                )
                token_ids = tokenized["input_ids"].to(embedding.weight.device)
                mask = tokenized["attention_mask"].to(embedding.weight.device).unsqueeze(-1)
                states = embedding(token_ids).float()
                output.append((states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0))
            return torch.cat(output, dim=0)

        group_text = [f"predicted treatment group: {str(name).replace('_', ' ')}" for name in group_names]
        fact_text = [f"supporting evidence: {str(key).replace('_', ' ')}" for key in fact_keys]
        groups = encode(group_text)
        facts = encode(fact_text)
        if not fact_keys:
            facts = facts[:1]
        self.caption_memory.set_descriptor_embeddings(groups, facts)

    def _caption_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        output_embeddings = self.lm.get_output_embeddings()
        if output_embeddings is None:
            raise RuntimeError("Causal LM does not expose output embeddings for caption decoding.")
        # Cached decoding can promote the residual state to fp32 while the tied
        # output embedding stays in bf16; the LM head is the dtype boundary.
        return output_embeddings(hidden.to(dtype=output_embeddings.weight.dtype))

    def _heads_from_original_state(
        self,
        hidden: torch.Tensor,
        prompt_end_indices: torch.Tensor,
        signal_length: int,
    ) -> dict[str, torch.Tensor]:
        """Action heads read the LM state at each prompt end."""
        batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
        indices = prompt_end_indices.to(hidden.device) + int(signal_length)
        pooled_for_heads = finite_clamp(self.dropout(hidden[batch_idx, indices, :]), clamp=100.0).float()
        raw_need_logits = finite_clamp(self.need_head(pooled_for_heads), clamp=50.0)
        need_type_logits = raw_need_logits if self.need_head_mode == "typed_multilabel" else None
        need_logits = (
            any_action_logit_from_typed_need(raw_need_logits)
            if need_type_logits is not None
            else raw_need_logits.squeeze(-1)
        )
        conditional_group_logits = finite_clamp(self.group_head(pooled_for_heads), clamp=50.0)
        group_logits = finite_clamp(
            hierarchical_group_logits(
                need_type_logits if need_type_logits is not None else need_logits,
                conditional_group_logits,
                self.group_need_type_indices if need_type_logits is not None else None,
            )
            if self.hierarchical_action_heads
            else conditional_group_logits,
            clamp=50.0,
        )
        return {
            "need_logits": need_logits,
            "need_type_logits": need_type_logits,
            "group_logits": group_logits,
            "conditional_group_logits": conditional_group_logits,
            "evidence_logits": self.evidence_selector(pooled_for_heads, group_logits)
            if self.evidence_selector is not None
            else None,
            "decision_hidden": pooled_for_heads,
        }

    def _prefix_state_inputs(
        self,
        *,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        signal_embeds: torch.Tensor | None,
        signal_attention_mask: torch.Tensor | None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, int]:
        """Build ``[signal tokens | prompt tokens]`` LM inputs and their mask."""
        if signal_embeds is None:
            return {"input_ids": prompt_ids}, prompt_mask, 0
        signal_length = int(signal_embeds.shape[1])
        signal_mask = signal_attention_mask
        if signal_mask is None:
            signal_mask = torch.ones(signal_embeds.shape[:2], dtype=prompt_mask.dtype, device=signal_embeds.device)
        text_embeds = self.lm.get_input_embeddings()(prompt_ids)
        inputs = {
            "inputs_embeds": torch.cat(
                [finite_clamp(signal_embeds, clamp=100.0).to(dtype=text_embeds.dtype, device=text_embeds.device), text_embeds],
                dim=1,
            )
        }
        mask = torch.cat([signal_mask.to(dtype=prompt_mask.dtype, device=prompt_mask.device), prompt_mask], dim=1)
        return inputs, mask, signal_length

    @torch.inference_mode()
    def forward_action_heads_only(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_end_indices: torch.Tensor,
        signal_embeds: torch.Tensor | None = None,
        signal_attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the causal action path without caption decoding."""
        prompt_length = int(prompt_end_indices.max().item()) + 1
        state_inputs, state_mask, signal_length = self._prefix_state_inputs(
            prompt_ids=input_ids[:, :prompt_length],
            prompt_mask=attention_mask[:, :prompt_length],
            signal_embeds=signal_embeds,
            signal_attention_mask=signal_attention_mask,
        )
        # Only the decision-token state is needed: skip the vocabulary projection elsewhere.
        state_out = self.lm(
            **state_inputs,
            attention_mask=state_mask,
            output_hidden_states=True,
            use_cache=False,
            logits_to_keep=torch.tensor([state_mask.shape[1] - 1], dtype=torch.long, device=input_ids.device),
        )
        return self._heads_from_original_state(state_out.hidden_states[-1], prompt_end_indices, signal_length)

    @torch.no_grad()
    def generate_action_conditioned_caption(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_end_indices: torch.Tensor,
        signal_embeds: torch.Tensor | None,
        signal_attention_mask: torch.Tensor | None,
        max_new_tokens: int,
        pad_token_id: int,
        eos_token_id: int | None,
        caption_detail_candidates: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Greedy caption decoding over the LM KV cache with the residual profile head."""
        if self.caption_memory is None:
            raise ValueError("This checkpoint has no caption decoder.")
        batch_size = int(input_ids.shape[0])
        prompt_length = int(prompt_end_indices.max().item()) + 1
        prompt_only_mask = prompt_only_state_mask(attention_mask, prompt_end_indices, prompt_length)
        state_inputs, state_mask, signal_length = self._prefix_state_inputs(
            prompt_ids=input_ids[:, :prompt_length],
            prompt_mask=prompt_only_mask,
            signal_embeds=signal_embeds,
            signal_attention_mask=signal_attention_mask,
        )
        state_out = self.lm(
            **state_inputs,
            attention_mask=state_mask,
            output_hidden_states=True,
            use_cache=True,
            logits_to_keep=torch.tensor([state_mask.shape[1] - 1], dtype=torch.long, device=state_mask.device),
        )
        hidden = state_out.hidden_states[-1]
        outputs = self._heads_from_original_state(hidden, prompt_end_indices, signal_length)
        profile, profile_mask = self.caption_memory.build_profile(
            need_logits=outputs["need_logits"],
            group_logits=outputs["group_logits"],
            evidence_logits=outputs["evidence_logits"],
        )
        # Prompts are right-padded, so start each decode stream from its own
        # decision-token state rather than the last tensor position.
        batch_index = torch.arange(batch_size, device=hidden.device)
        final_state_indices = prompt_end_indices.to(device=hidden.device, dtype=torch.long) + int(signal_length)
        next_hidden = hidden[batch_index, final_state_indices, :].unsqueeze(1)
        prepared_context = self.caption_memory.prepare_caption_context(
            profile_tokens=profile,
            profile_mask=profile_mask,
            decision_hidden=outputs["decision_hidden"],
            detail_candidates=caption_detail_candidates,
        )
        generated: list[torch.Tensor] = []
        histories: list[list[int]] = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        fill_token = int(eos_token_id) if eos_token_id is not None else int(pad_token_id)
        past_key_values = state_out.past_key_values
        decode_mask = state_mask
        for _ in range(int(max_new_tokens)):
            adjusted = self.caption_memory.decode_with_context(next_hidden, prepared_context)
            next_token = torch.argmax(self._caption_logits(adjusted)[:, -1, :], dim=-1)
            is_eos = next_token.eq(int(eos_token_id)) if eos_token_id is not None else torch.zeros_like(finished)
            # Stop a row that has entered a short repeating loop.
            looping = torch.zeros_like(finished)
            for row in (~finished & ~is_eos).nonzero().flatten().tolist():
                histories[row].append(int(next_token[row]))
                looping[row] = repeated_suffix_token_count(histories[row]) > 0
            # Finished rows keep emitting the fill token (masked out) so the batch stays rectangular.
            generated.append(torch.where(finished | looping, torch.full_like(next_token, fill_token), next_token))
            finished |= is_eos | looping
            if bool(finished.all()):
                break
            decode_mask = torch.cat([decode_mask, (~finished).to(dtype=decode_mask.dtype).unsqueeze(1)], dim=1)
            decode_out = self.lm(
                input_ids=generated[-1].unsqueeze(1),
                attention_mask=decode_mask,
                past_key_values=past_key_values,
                position_ids=residual_decode_position_ids(
                    prompt_end_indices,
                    signal_length=signal_length,
                    generated_tokens=len(generated),
                ).to(device=input_ids.device),
                output_hidden_states=True,
                use_cache=True,
            )
            next_hidden = decode_out.hidden_states[-1]
            past_key_values = decode_out.past_key_values
        if not generated:
            return torch.empty((batch_size, 0), dtype=input_ids.dtype, device=input_ids.device), outputs
        return torch.stack(generated, dim=1), outputs
