#!/usr/bin/env python3
"""
Saliency-Guided Token Importance Attribution

Uses module-level forward hooks to capture attention weights BEFORE
any dtype cast, ensuring they remain on the computation path.
"""

import torch
import torch.nn as nn
from typing import Optional, List


def _get_model_layers(model: nn.Module) -> nn.ModuleList:
    """Extract the transformer layer list from a HuggingFace CausalLM."""
    candidates = [
        lambda: model.model.layers,
        lambda: model.transformer.h,
        lambda: model.gpt_neox.layers,
        lambda: model.model.decoder.layers,
    ]
    for getter in candidates:
        try:
            layers = getter()
            if isinstance(layers, nn.ModuleList) and len(layers) > 0:
                return layers
        except (AttributeError, TypeError):
            continue
    raise RuntimeError(f"Cannot locate transformer layers in {type(model).__name__}")


def _find_attention_module(layer: nn.Module) -> nn.Module:
    for attr in ["self_attn", "attn", "attention", "self_attention"]:
        if hasattr(layer, attr):
            return getattr(layer, attr)
    return None


def compute_attention_saliency_from_ids(
    model: nn.Module,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.LongTensor] = None,
    device: Optional[torch.device] = None,
    top_k_layers: int = 2,
) -> torch.FloatTensor:
    if device is None:
        device = input_ids.device
    input_ids = input_ids.to(device)
    B, L = input_ids.shape

    if attention_mask is None:
        pad_id = getattr(model.config, "pad_token_id", None)
        if pad_id is not None:
            attention_mask = (input_ids != int(pad_id)).long()
        else:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    else:
        attention_mask = attention_mask.to(device)

    original_training = model.training
    model.eval()

    all_layers = _get_model_layers(model)
    num_layers = len(all_layers)
    top_k = min(top_k_layers, num_layers)
    target_layer_indices = list(range(num_layers - top_k, num_layers))

    try:
        importances = _compute_saliency_core(
            model, input_ids, attention_mask, target_layer_indices, device
        )
    finally:
        model.train(original_training)

    return importances


def _compute_saliency_core(model, input_ids, attention_mask, target_layer_indices, device):
    """Capture attention weights via module hooks to avoid dead-branch issue."""
    B, L = input_ids.shape
    all_layers = _get_model_layers(model)

    # ── Step 1: Register hooks to capture attention weights ───────────────
    # We hook into the attention module's forward to capture the softmax
    # output BEFORE any dtype cast. This tensor IS on the computation path.
    captured = {}
    hooks = []

    def _make_attn_hook(layer_idx):
        def hook(module, args, kwargs, output):
            # MistralAttention.forward returns (attn_output, attn_weights, past_kv)
            # The attn_weights at this point may be pre- or post-cast.
            # We intercept and replace with a version that has retain_grad.
            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]
                if attn_weights is not None and attn_weights.requires_grad:
                    attn_weights.retain_grad()
                    captured[layer_idx] = attn_weights
        return hook

    for idx in target_layer_indices:
        attn_mod = _find_attention_module(all_layers[idx])
        if attn_mod is not None:
            h = attn_mod.register_forward_hook(_make_attn_hook(idx), with_kwargs=True)
            hooks.append(h)

    # ── Step 2: Forward pass ──────────────────────────────────────────────
    embeddings = model.get_input_embeddings()(input_ids)
    embeddings = embeddings.detach().requires_grad_(True)

    try:
        outputs = model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=False,
        )
    finally:
        for h in hooks:
            h.remove()

    logits = outputs.logits

    if not captured:
        print("[Saliency] WARNING: No attention captured by hooks. Uniform fallback.")
        return torch.ones(B, L, device=device, dtype=torch.float32) * attention_mask.float()

    print(f"[Saliency] Captured attention from layers: {sorted(captured.keys())}")

    # ── Step 3: Backward ─────────────────────────────────────────────────
    lengths = attention_mask.long().sum(dim=1)
    last_pos = torch.clamp(lengths - 1, min=0)
    batch_idx = torch.arange(B, device=device)
    last_logits = logits[batch_idx, last_pos, :]
    target = last_logits.max(dim=-1).values

    model.zero_grad()
    if embeddings.grad is not None:
        embeddings.grad = None
    target.sum().backward(retain_graph=False)

    # ── Step 4: Compute A * grad_A saliency ──────────────────────────────
    saliency_accum = torch.zeros(B, L, device=device, dtype=torch.float32)
    n_valid = 0

    for layer_idx in sorted(captured.keys()):
        attn = captured[layer_idx]
        grad = attn.grad
        if grad is None:
            print(f"[Saliency] WARNING: No gradient for layer {layer_idx}")
            continue
        n_valid += 1

        attn_device = attn.device
        saliency = (attn.detach() * grad.detach()).abs().float()
        causal_mask = torch.tril(torch.ones(L, L, device=attn_device, dtype=torch.bool))
        saliency = saliency * causal_mask.unsqueeze(0).unsqueeze(0)

        for b in range(B):
            t = last_pos[b].item()
            row = saliency[b, :, t, :t + 1].sum(dim=0)
            saliency_accum[b, :t + 1] += row.to(saliency_accum.device)

    if n_valid == 0:
        print("[Saliency] WARNING: No valid saliency layers. Uniform fallback.")
        return torch.ones(B, L, device=device, dtype=torch.float32) * attention_mask.float()

    # ── Step 5: L2 normalize ─────────────────────────────────────────────
    for b in range(B):
        t = last_pos[b].item()
        vec = saliency_accum[b, :t + 1]
        norm = vec.norm(p=2)
        if norm > 1e-8:
            saliency_accum[b, :t + 1] = vec / norm

    saliency_accum = saliency_accum * attention_mask.float()
    return saliency_accum


def compute_language_model_saliency_attribution_from_ids(
    model, input_ids, attention_mask=None, device=None, top_k_layers=2,
):
    """Alias matching gradient_attribution API."""
    return compute_attention_saliency_from_ids(
        model=model, input_ids=input_ids, attention_mask=attention_mask,
        device=device, top_k_layers=top_k_layers,
    )
