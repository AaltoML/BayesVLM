from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers import CLIPModel, CLIPTextModelWithProjection, CLIPVisionModelWithProjection
from transformers.modeling_outputs import ModelOutput


def _as_optional_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
    return tensor if tensor is not None else None


def _diag_cov(
    activations: torch.Tensor,
    a_inv: torch.Tensor,
    b_diag: torch.Tensor,
    add_bias: bool,
) -> torch.Tensor | None:
    if a_inv.numel() == 0 or b_diag.numel() == 0:
        return None

    if add_bias:
        ones = torch.ones_like(activations[:, :1])
        activations = torch.cat([activations, ones], dim=-1)

    quad = torch.einsum("ij,jk,ik->i", activations, a_inv, activations)[:, None]
    return quad * b_diag


def _std_from_var(var: torch.Tensor | None) -> torch.Tensor | None:
    if var is None:
        return None
    return torch.sqrt(var)

def _get_output(outputs, name: str, index: int):
    if hasattr(outputs, name):
        return getattr(outputs, name)
    if isinstance(outputs, (tuple, list)) and len(outputs) > index:
        return outputs[index]
    return None

def _normalize_mean_and_var(
    mean: torch.Tensor,
    var: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r2 = (mean**2).sum(dim=-1, keepdim=True).clamp_min(eps)
    r = torch.sqrt(r2)
    normalized = mean / r

    # Delta-method approximation with diagonal covariance.
    y2 = normalized**2
    sum_y2v = (y2 * var).sum(dim=-1, keepdim=True)
    norm_var = (var - 2 * y2 * var + y2 * sum_y2v) / r2
    norm_var = norm_var.clamp_min(0)
    return normalized, norm_var


@dataclass
class BayesVLMEmbeddingOutput(ModelOutput):
    mean: torch.FloatTensor | None = None
    var: torch.FloatTensor | None = None
    std: torch.FloatTensor | None = None


@dataclass
class BayesVLMTextModelOutput(ModelOutput):
    text_embeds: torch.FloatTensor | None = None
    text_embeds_var: torch.FloatTensor | None = None
    text_embeds_std: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class BayesVLMVisionModelOutput(ModelOutput):
    image_embeds: torch.FloatTensor | None = None
    image_embeds_var: torch.FloatTensor | None = None
    image_embeds_std: torch.FloatTensor | None = None
    last_hidden_state: torch.FloatTensor | None = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class BayesVLMOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits_per_image: torch.FloatTensor | None = None
    logits_per_text: torch.FloatTensor | None = None
    logits_per_image_var: torch.FloatTensor | None = None
    logits_per_text_var: torch.FloatTensor | None = None
    logits_per_image_std: torch.FloatTensor | None = None
    logits_per_text_std: torch.FloatTensor | None = None
    text_embeds: torch.FloatTensor | None = None
    image_embeds: torch.FloatTensor | None = None
    text_embeds_var: torch.FloatTensor | None = None
    image_embeds_var: torch.FloatTensor | None = None
    text_embeds_std: torch.FloatTensor | None = None
    image_embeds_std: torch.FloatTensor | None = None
    text_model_output: Optional[ModelOutput] = None
    vision_model_output: Optional[ModelOutput] = None


class BayesVLMTextModel(CLIPTextModelWithProjection):
    def __init__(self, config):
        super().__init__(config)
        hidden = int(config.hidden_size)
        proj = int(config.projection_dim)
        self.register_buffer("a_inv", torch.zeros(hidden, hidden))
        self.register_buffer("b_diag", torch.zeros(proj))

    def set_covariance(self, a_inv: torch.Tensor, b_inv: torch.Tensor) -> None:
        self.a_inv = a_inv
        self.b_diag = torch.diagonal(b_inv)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if not return_dict:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        pooled_output = _get_output(text_outputs, "pooler_output", 1)
        last_hidden_state = _get_output(text_outputs, "last_hidden_state", 0)
        hidden_states = _get_output(text_outputs, "hidden_states", 2)
        attentions = _get_output(text_outputs, "attentions", 3)
        text_embeds = self.text_projection(pooled_output)

        text_var = _diag_cov(
            pooled_output,
            self.a_inv,
            self.b_diag,
            add_bias=self.text_projection.bias is not None,
        )
        if text_var is None:
            text_var = torch.zeros_like(text_embeds)
        text_std = _std_from_var(text_var)

        return BayesVLMTextModelOutput(
            text_embeds=text_embeds,
            text_embeds_var=text_var,
            text_embeds_std=text_std,
            last_hidden_state=last_hidden_state,
            hidden_states=hidden_states,
            attentions=attentions,
        )


class BayesVLMVisionModel(CLIPVisionModelWithProjection):
    def __init__(self, config):
        super().__init__(config)
        hidden = int(config.hidden_size)
        proj = int(config.projection_dim)
        self.register_buffer("a_inv", torch.zeros(hidden, hidden))
        self.register_buffer("b_diag", torch.zeros(proj))

    def set_covariance(self, a_inv: torch.Tensor, b_inv: torch.Tensor) -> None:
        self.a_inv = a_inv
        self.b_diag = torch.diagonal(b_inv)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if not return_dict:
            return super().forward(
                pixel_values=pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        pooled_output = _get_output(vision_outputs, "pooler_output", 1)
        last_hidden_state = _get_output(vision_outputs, "last_hidden_state", 0)
        hidden_states = _get_output(vision_outputs, "hidden_states", 2)
        attentions = _get_output(vision_outputs, "attentions", 3)
        image_embeds = self.visual_projection(pooled_output)

        image_var = _diag_cov(
            pooled_output,
            self.a_inv,
            self.b_diag,
            add_bias=self.visual_projection.bias is not None,
        )
        if image_var is None:
            image_var = torch.zeros_like(image_embeds)
        image_std = _std_from_var(image_var)

        return BayesVLMVisionModelOutput(
            image_embeds=image_embeds,
            image_embeds_var=image_var,
            image_embeds_std=image_std,
            last_hidden_state=last_hidden_state,
            hidden_states=hidden_states,
            attentions=attentions,
        )


class BayesVLMModel(CLIPModel):
    def __init__(self, config):
        super().__init__(config)
        text_hidden = int(config.text_config.hidden_size)
        vision_hidden = int(config.vision_config.hidden_size)
        proj = int(config.projection_dim)
        self.register_buffer("text_a_inv", torch.zeros(text_hidden, text_hidden))
        self.register_buffer("text_b_diag", torch.zeros(proj))
        self.register_buffer("image_a_inv", torch.zeros(vision_hidden, vision_hidden))
        self.register_buffer("image_b_diag", torch.zeros(proj))

    def set_covariances(
        self,
        image_a_inv: torch.Tensor,
        image_b_inv: torch.Tensor,
        text_a_inv: torch.Tensor,
        text_b_inv: torch.Tensor,
    ) -> None:
        self.image_a_inv = image_a_inv
        self.image_b_diag = torch.diagonal(image_b_inv)
        self.text_a_inv = text_a_inv
        self.text_b_diag = torch.diagonal(text_b_inv)

    def _expected_logits_and_var(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        image_acts: torch.Tensor,
        text_acts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        scale = self.logit_scale.exp()

        if self.image_a_inv.numel() == 0 or self.text_a_inv.numel() == 0:
            image_norm = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_norm = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            logits = image_norm @ text_norm.t()
            logits = logits * scale
            return logits, None

        image_diag_cov = _diag_cov(
            image_acts,
            self.image_a_inv,
            self.image_b_diag,
            add_bias=self.visual_projection.bias is not None,
        )
        text_diag_cov = _diag_cov(
            text_acts,
            self.text_a_inv,
            self.text_b_diag,
            add_bias=self.text_projection.bias is not None,
        )

        norm_image = image_embeds**2 + image_diag_cov
        norm_text = text_embeds**2 + text_diag_cov
        expect_norm_image = norm_image.sum(dim=-1, keepdim=True)
        expect_norm_text = norm_text.sum(dim=-1, keepdim=True)

        expected_similarity = torch.matmul(
            image_embeds / torch.sqrt(expect_norm_image),
            (text_embeds / torch.sqrt(expect_norm_text)).t(),
        )

        term1 = torch.matmul(norm_image, text_diag_cov.t())
        term2 = torch.matmul(image_diag_cov, (text_embeds**2).t())
        variance_similarity = (term1 + term2) / (expect_norm_image * expect_norm_text.t())

        logits_mean = expected_similarity * scale
        logits_var = variance_similarity * (scale**2)
        return logits_mean, logits_var

    def get_text_features(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_std: bool = False,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        pooled_output = _get_output(text_outputs, "pooler_output", 1)
        text_embeds = self.text_projection(pooled_output)

        text_var = _diag_cov(
            pooled_output,
            self.text_a_inv,
            self.text_b_diag,
            add_bias=self.text_projection.bias is not None,
        )
        if text_var is None:
            text_var = torch.zeros_like(text_embeds)
        text_std = _std_from_var(text_var)

        if not return_dict and not return_std:
            return text_embeds

        return BayesVLMEmbeddingOutput(mean=text_embeds, var=text_var, std=text_std)

    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_std: bool = False,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        pooled_output = _get_output(vision_outputs, "pooler_output", 1)
        image_embeds = self.visual_projection(pooled_output)

        image_var = _diag_cov(
            pooled_output,
            self.image_a_inv,
            self.image_b_diag,
            add_bias=self.visual_projection.bias is not None,
        )
        if image_var is None:
            image_var = torch.zeros_like(image_embeds)
        image_std = _std_from_var(image_var)

        if not return_dict and not return_std:
            return image_embeds

        return BayesVLMEmbeddingOutput(mean=image_embeds, var=image_var, std=image_std)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if not return_dict:
            return super().forward(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_loss=return_loss,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        text_pooled = _get_output(text_outputs, "pooler_output", 1)
        image_pooled = _get_output(vision_outputs, "pooler_output", 1)

        text_embeds = self.text_projection(text_pooled)
        image_embeds = self.visual_projection(image_pooled)

        text_var = _diag_cov(
            text_pooled,
            self.text_a_inv,
            self.text_b_diag,
            add_bias=self.text_projection.bias is not None,
        )
        image_var = _diag_cov(
            image_pooled,
            self.image_a_inv,
            self.image_b_diag,
            add_bias=self.visual_projection.bias is not None,
        )
        if text_var is None:
            text_var = torch.zeros_like(text_embeds)
        if image_var is None:
            image_var = torch.zeros_like(image_embeds)

        text_std = _std_from_var(text_var)
        image_std = _std_from_var(image_var)

        logits_mean, logits_var = self._expected_logits_and_var(
            image_embeds,
            text_embeds,
            image_pooled,
            text_pooled,
        )

        text_embeds, text_var = _normalize_mean_and_var(text_embeds, text_var)
        image_embeds, image_var = _normalize_mean_and_var(image_embeds, image_var)
        text_std = _std_from_var(text_var)
        image_std = _std_from_var(image_var)

        logits_per_image = logits_mean
        logits_per_text = logits_mean.t() if logits_mean is not None else None

        if logits_var is None and logits_mean is not None:
            logits_var = torch.zeros_like(logits_mean)
        logits_per_image_var = _as_optional_tensor(logits_var)
        logits_per_text_var = logits_var.t() if logits_var is not None else None

        logits_per_image_std = _std_from_var(logits_per_image_var)
        logits_per_text_std = _std_from_var(logits_per_text_var)

        loss = None
        if return_loss and logits_per_image is not None and logits_per_text is not None:
            labels = torch.arange(logits_per_image.shape[0], device=logits_per_image.device)
            loss_i = torch.nn.functional.cross_entropy(logits_per_image, labels)
            loss_t = torch.nn.functional.cross_entropy(logits_per_text, labels)
            loss = (loss_i + loss_t) / 2

        return BayesVLMOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            logits_per_image_var=logits_per_image_var,
            logits_per_text_var=logits_per_text_var,
            logits_per_image_std=logits_per_image_std,
            logits_per_text_std=logits_per_text_std,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_embeds_var=text_var,
            image_embeds_var=image_var,
            text_embeds_std=text_std,
            image_embeds_std=image_std,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )
