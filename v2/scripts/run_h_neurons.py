#!/usr/bin/env python3
"""H-Neurons discovery and evaluation pipeline V2 for Llama-3.2-3B-Instruct.

V2 adds layer-specific alpha suppression (early/middle/late) and a two-phase
alpha sweep (coarse 64 + fine ~25 combinations).

Usage:
    python scripts/run_h_neurons.py --model-path <path> --output-dir <path>

Pipeline stages (each resume-safe with checkpoint files):
  1. Architecture probe
  2. Load questions (TriviaQA 4000 + finance QA 1000)
  3. Finance QA validation (15-35% hallucination sweet spot)
  4. Sampling (10 responses per question, temperature=1.0)
  5. Consistency filter
  6. Class balancing
  7. Token F1 span extraction (no GPT-4o)
  8. CETT feature collection
  9. Neuron selection (L1 logistic regression)
 10. Layer-specific alpha sweep (two-phase: coarse + fine)
 11. Save h_neurons.json with layer-specific alphas
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import string
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

# ---------------------------------------------------------------------------
# Project-level imports (for stage_complete / mark_complete helpers)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.utils import BASE_DIR  # noqa: E402


# ===================================================================
# Utility functions (ported from v6 utils.py)
# ===================================================================

def get_down_proj_modules(model) -> Dict[int, nn.Module]:
    """Locate all mlp.down_proj linear layers inside the model.

    Tries direct attribute traversal first, falls back to named_modules().
    """
    modules: Dict[int, nn.Module] = {}

    # Fast path: model.model.layers[i].mlp.down_proj
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        lm = getattr(model, "language_model", None)
        layers = getattr(lm, "layers", None)
    if layers is not None:
        for i, layer in enumerate(layers):
            mlp = getattr(layer, "mlp", None)
            down_proj = getattr(mlp, "down_proj", None) if mlp else None
            if isinstance(down_proj, nn.Module):
                modules[i] = down_proj

    # Fallback: walk the full module tree.
    if not modules:
        for name, module in model.named_modules():
            if not name.endswith("mlp.down_proj"):
                continue
            match = re.search(r"layers\.(\d+)\.mlp\.down_proj$", name)
            if match:
                modules[int(match.group(1))] = module

    if not modules:
        raise RuntimeError(
            "Could not locate MLP down_proj modules in this model."
        )
    return dict(sorted(modules.items(), key=lambda kv: kv[0]))


def input_device(model, fallback: str) -> torch.device:
    """Get correct input device for (possibly distributed) models."""
    if hasattr(model, "hf_device_map"):
        first_param = next(model.parameters(), None)
        if first_param is not None:
            return first_param.device
    return torch.device(fallback)


def build_chat_prompt(tokenizer, question: str) -> str:
    """Format question as a chat prompt using the model's chat template."""
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def normalize_answer(text: str) -> str:
    """Normalize an answer string following the TriviaQA standard."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(prediction: str, references: List[str]) -> bool:
    """Check whether the normalized prediction matches any reference."""
    pred_norm = normalize_answer(prediction)
    if not pred_norm:
        return False
    for ref in references:
        ref_norm = normalize_answer(ref)
        if not ref_norm:
            continue
        if ref_norm == pred_norm:
            return True
        if ref_norm in pred_norm:
            return True
    return False


def token_f1_score(prediction: str, target: str) -> float:
    """Compute token-level F1 between prediction and target."""
    pred_tokens = normalize_answer(prediction).split()
    target_tokens = normalize_answer(target).split()
    if not pred_tokens or not target_tokens:
        return 0.0
    pred_count: Dict[str, int] = {}
    target_count: Dict[str, int] = {}
    for t in pred_tokens:
        pred_count[t] = pred_count.get(t, 0) + 1
    for t in target_tokens:
        target_count[t] = target_count.get(t, 0) + 1
    common = 0
    for t, c in pred_count.items():
        common += min(c, target_count.get(t, 0))
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(target_tokens)
    return 2.0 * precision * recall / (precision + recall)


@torch.no_grad()
def generate_answer(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 80,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
) -> str:
    """Generate a single answer string from a fully-rendered prompt."""
    target = input_device(model, device)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {k: v.to(target) for k, v in encoded.items()}

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p
        generate_kwargs["top_k"] = top_k
    else:
        generate_kwargs["do_sample"] = False

    output = model.generate(**encoded, **generate_kwargs)
    prompt_len = encoded["input_ids"].shape[1]
    gen_tokens = output[0, prompt_len:]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    return text


@torch.no_grad()
def generate_answers_batch(
    model,
    tokenizer,
    prompts: List[str],
    device: str,
    max_new_tokens: int = 80,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
) -> List[str]:
    """Generate answers for a batch of prompts in a single forward pass."""
    if not prompts:
        return []

    target = input_device(model, device)

    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    encoded = {k: v.to(target) for k, v in encoded.items()}

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p
        generate_kwargs["top_k"] = top_k
    else:
        generate_kwargs["do_sample"] = False

    outputs = model.generate(**encoded, **generate_kwargs)

    prompt_len = encoded["input_ids"].shape[1]
    results = []
    for i in range(outputs.shape[0]):
        gen_tokens = outputs[i, prompt_len:]
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        results.append(text)

    tokenizer.padding_side = orig_side
    return results


def _sanitize_for_json(obj):
    """Recursively replace NaN and Inf float values with 0.0."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def save_json(path: str, data) -> None:
    """Write data to a JSON file with NaN/Inf sanitisation."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(_sanitize_for_json(data), f, ensure_ascii=False, indent=2)


def load_json(path: str):
    """Load and return a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ===================================================================
# ActivationEditor (from v6 intervention.py)
# ===================================================================

LAYER_RANGES = {
    "early": (0, 9),
    "middle": (10, 19),
    "late": (20, 27),
}


def _get_layer_group(layer_idx: int) -> str:
    """Return which group (early/middle/late) a layer belongs to."""
    for group, (lo, hi) in LAYER_RANGES.items():
        if lo <= layer_idx <= hi:
            return group
    return "late"


class ActivationEditor:
    """Inference-time H-Neuron intervention (V2: layer-specific alpha).

    Scales selected neuron activations: z_{j,t} <- alpha * z_{j,t}

    V2 enhancement: supports per-layer-group alphas via ``layer_alphas``
    dict (keys: "early", "middle", "late").

    Supports two usage modes:
      1. Context manager: ``with ActivationEditor(model, neurons, alpha=0.0):``
      2. Direct wrapper: ``model = ActivationEditor(model, neurons, alpha=1.5)``

    Args:
        model: A HuggingFace CausalLM (or compatible transformer).
        h_neurons: List of neuron descriptors, each a dict with at least
            ``"layer"`` (int) and ``"neuron"`` (int) keys.
        alpha: Global scaling factor. 0.0 = full knockout, 1.0 = no change.
        layer_alphas: Optional dict mapping layer groups to alpha values,
            e.g. ``{"early": 0.3, "middle": 0.7, "late": 1.0}``.
            When provided, overrides the global ``alpha``.
    """

    def __init__(
        self,
        model,
        h_neurons: list,
        alpha: float = 0.0,
        layer_alphas: Optional[Dict[str, float]] = None,
    ):
        self.model = model
        self.alpha = alpha
        self.layer_alphas = layer_alphas
        self._handles: list = []

        # Group neuron indices by layer.
        self._layer_neuron_idx: Dict[int, torch.Tensor] = {}
        grouped: Dict[int, List[int]] = {}
        for item in h_neurons:
            layer = int(item["layer"])
            neuron = int(item["neuron"])
            grouped.setdefault(layer, []).append(neuron)

        for layer, indices in grouped.items():
            self._layer_neuron_idx[layer] = torch.tensor(
                indices, dtype=torch.long,
            )

        # Locate down_proj modules in the model.
        self._down_proj_modules = get_down_proj_modules(model)

        # When used as a direct wrapper (not context manager), register
        # hooks immediately so the editor is always active.
        self._register_hooks()

    def _get_alpha_for_layer(self, layer_idx: int) -> float:
        """Get the alpha value for a specific layer index."""
        if self.layer_alphas is not None:
            group = _get_layer_group(layer_idx)
            return self.layer_alphas.get(group, self.alpha)
        return self.alpha

    # ------------------------------------------------------------------
    # Hook construction
    # ------------------------------------------------------------------

    def _build_pre_hook(self, layer_idx: int):
        """Return a forward_pre_hook that scales selected neurons."""

        def hook(module, inputs):
            if not inputs:
                return None
            x = inputs[0]
            if not torch.is_tensor(x) or x.ndim != 3:
                return None

            idx = self._layer_neuron_idx.get(layer_idx)
            if idx is None:
                return None

            layer_alpha = self._get_alpha_for_layer(layer_idx)
            idx = idx.to(x.device)
            x_edit = x.clone()
            x_edit[..., idx] *= layer_alpha

            if len(inputs) == 1:
                return (x_edit,)
            return (x_edit, *inputs[1:])

        return hook

    def _register_hooks(self):
        """Register forward_pre_hook on each relevant down_proj."""
        for layer_idx, module in self._down_proj_modules.items():
            if layer_idx not in self._layer_neuron_idx:
                continue
            handle = module.register_forward_pre_hook(
                self._build_pre_hook(layer_idx),
            )
            self._handles.append(handle)

    # ------------------------------------------------------------------
    # Context-manager interface
    # ------------------------------------------------------------------

    def __enter__(self):
        """When used as context manager, hooks are already registered."""
        # Hooks are registered in __init__, so nothing extra needed here.
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Remove all hooks on context-manager exit."""
        self.close()
        return False

    def close(self):
        """Remove all registered hooks. Safe to call multiple times."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    # ------------------------------------------------------------------
    # Proxy interface for direct wrapper usage
    # ------------------------------------------------------------------

    def parameters(self, *args, **kwargs):
        return self.model.parameters(*args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        return self.model.named_parameters(*args, **kwargs)

    def named_modules(self, *args, **kwargs):
        return self.model.named_modules(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def eval(self):
        self.model.eval()
        return self

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @property
    def dtype(self):
        return self.model.dtype

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def config(self):
        return self.model.config

    def __del__(self):
        self.close()

    def __getattr__(self, name):
        # Proxy any attribute not found on ActivationEditor to the model.
        # __getattr__ is only called when normal attribute lookup fails,
        # so self.model, self.alpha, etc. are found normally.
        return getattr(self.model, name)


# ===================================================================
# CETTCollector (from v6 features.py)
# ===================================================================

class CETTCollector:
    """Hook-based CETT metric collection.

    CETT_{j,t} = |z_{j,t}| * ||W_down[:, j]||_2 / ||h_t||_2

    Hooks every mlp.down_proj. For each forward pass:
    - z = input to down_proj (gated intermediate)
    - h = output of down_proj (MLP output)
    - Precompute column norms of W_down at init
    """

    def __init__(self, model):
        self.down_proj_modules: Dict[int, nn.Module] = get_down_proj_modules(model)
        self.layer_ids: List[int] = list(self.down_proj_modules.keys())
        self._handles: list = []
        self._current: Dict[int, torch.Tensor] = {}
        self._enabled: bool = False
        self._start: int = 0
        self._end: int = 0

        # Detect model dimensions from first down_proj weight.
        first_module = next(iter(self.down_proj_modules.values()))
        self.num_layers = len(self.down_proj_modules)
        self.intermediate_size = first_module.weight.shape[1]
        self.total_neurons = self.num_layers * self.intermediate_size

        # Precompute column norms for each down_proj weight matrix.
        self._col_norms: Dict[int, torch.Tensor] = {}
        for layer_id, module in self.down_proj_modules.items():
            with torch.no_grad():
                self._col_norms[layer_id] = (
                    module.weight.norm(dim=0).float().cpu()
                )

        self._register_hooks()

    def _register_hooks(self) -> None:
        for layer_id, module in self.down_proj_modules.items():
            handle = module.register_forward_hook(self._build_hook(layer_id))
            self._handles.append(handle)

    def _build_hook(self, layer_id: int):
        """Return a forward hook that records CETT for layer_id."""

        def hook(module, inputs, outputs):
            if not self._enabled:
                return
            if not inputs:
                return

            z = inputs[0]   # gated intermediate [B, T, intermediate_size]
            h = outputs      # down_proj output   [B, T, hidden_size]

            if not torch.is_tensor(z) or z.ndim != 3:
                return
            if not torch.is_tensor(h) or h.ndim != 3:
                return

            start = min(self._start, z.size(1))
            end = min(self._end, z.size(1))
            if end <= start:
                return

            z_slice = z[:, start:end, :].float()
            h_slice = h[:, start:end, :].float()

            abs_z = z_slice.abs()
            h_norms = h_slice.norm(dim=-1, keepdim=True).clamp_min(1e-8)

            col_norms = self._col_norms[layer_id].to(z_slice.device)
            cett = abs_z * col_norms.unsqueeze(0).unsqueeze(0) / h_norms

            pooled = cett.mean(dim=(0, 1)).detach().cpu()
            self._current[layer_id] = pooled

        return hook

    @torch.no_grad()
    def collect(
        self,
        model,
        tokenizer,
        device: str,
        prompt_text: str,
        response_text: str,
        token_start: int,
        token_end: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Run a forward pass and return CETT vectors for answer and non-answer spans.

        Returns:
            (cett_answer, cett_other) each of shape [total_neurons], or None.
        """
        full_text = prompt_text + response_text
        target = input_device(model, device)
        encoded = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"].to(target)
        attention_mask = torch.ones_like(input_ids)
        seq_len = input_ids.size(1)

        t_start = max(0, min(token_start, seq_len))
        t_end = max(t_start, min(token_end, seq_len))

        # Collect CETT for the answer span
        self._start = t_start
        self._end = t_end
        self._current = {}
        self._enabled = True
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
        self._enabled = False

        if len(self._current) != len(self.layer_ids):
            return None
        cett_answer = torch.cat(
            [self._current[lid] for lid in self.layer_ids], dim=0
        )

        # Collect CETT for non-answer tokens
        other_parts: List[torch.Tensor] = []
        other_token_count = 0

        for rng_start, rng_end in [(0, t_start), (t_end, seq_len)]:
            span = rng_end - rng_start
            if span <= 0:
                continue
            self._start = rng_start
            self._end = rng_end
            self._current = {}
            self._enabled = True
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
            self._enabled = False

            if len(self._current) != len(self.layer_ids):
                return None

            part = torch.cat(
                [self._current[lid] for lid in self.layer_ids], dim=0
            )
            other_parts.append(part * span)
            other_token_count += span

        if other_token_count > 0:
            cett_other = sum(other_parts) / other_token_count
        else:
            cett_other = torch.zeros(self.total_neurons, dtype=torch.float32)

        return cett_answer.float(), cett_other.float()

    def close(self) -> None:
        """Remove all forward hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles = []


# ===================================================================
# Token F1 span extraction (replaces GPT-4o)
# ===================================================================

def extract_answer_span(
    response_tokens: List[int],
    reference_answers: List[str],
    tokenizer,
    min_window: int = 1,
    max_window: int = 30,
) -> Tuple[int, int]:
    """Find token span in response that best matches references using token F1.

    Uses a sliding window approach with varying window sizes. For each
    window of tokens in the response, computes token F1 against each
    reference answer and keeps the window with highest F1.

    Args:
        response_tokens: Token IDs of the model response.
        reference_answers: List of gold-standard answer strings.
        tokenizer: HuggingFace tokenizer.
        min_window: Minimum window size in tokens.
        max_window: Maximum window size in tokens.

    Returns:
        (start_idx, end_idx) of the best matching span within the response
        token list. Indices are relative to the response_tokens list.
    """
    if not response_tokens or not reference_answers:
        return 0, min(1, len(response_tokens))

    n = len(response_tokens)
    max_window = min(max_window, n)

    best_f1 = -1.0
    best_start = 0
    best_end = min(1, n)

    # Pre-normalize reference answers for token F1 computation
    ref_token_sets = []
    for ref in reference_answers:
        ref_norm = normalize_answer(ref)
        if ref_norm:
            ref_token_sets.append(ref_norm.split())

    if not ref_token_sets:
        return 0, min(1, n)

    # Sliding window: try various window sizes
    for win_size in range(min_window, max_window + 1):
        for start in range(0, n - win_size + 1):
            end = start + win_size
            # Decode this window of tokens
            window_text = tokenizer.decode(
                response_tokens[start:end], skip_special_tokens=True
            )
            window_norm = normalize_answer(window_text)
            if not window_norm:
                continue
            window_tokens = window_norm.split()
            if not window_tokens:
                continue

            # Compute F1 against each reference
            for ref_tokens in ref_token_sets:
                # Token F1
                pred_count: Dict[str, int] = {}
                for t in window_tokens:
                    pred_count[t] = pred_count.get(t, 0) + 1
                ref_count: Dict[str, int] = {}
                for t in ref_tokens:
                    ref_count[t] = ref_count.get(t, 0) + 1

                common = 0
                for t, c in pred_count.items():
                    common += min(c, ref_count.get(t, 0))

                if common == 0:
                    continue

                precision = common / len(window_tokens)
                recall = common / len(ref_tokens)
                f1 = 2.0 * precision * recall / (precision + recall)

                if f1 > best_f1:
                    best_f1 = f1
                    best_start = start
                    best_end = end

    return best_start, best_end


# ===================================================================
# Neuron selection (from v6 neuron_select.py)
# ===================================================================

DEFAULT_C_VALUES = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]


def select_h_neurons(
    X,
    y,
    intermediate_size: int,
    model=None,
    tokenizer=None,
    device=None,
    held_out_questions: list = None,
    C_values: list = None,
    max_iter: int = 1000,
    cv_folds: int = 5,
    max_eval_questions: int = 200,
    max_new_tokens: int = 80,
) -> Tuple[List[Dict], Dict]:
    """L1 logistic regression with joint C optimization.

    For each C candidate:
    1. Fit L1 LogisticRegression on full data
    2. 5-fold cross-validation accuracy
    3. Extract positive-weight neurons
    4. Suppress at alpha=0 via ActivationEditor, evaluate on held-out
    5. Score = cv_accuracy + suppression_accuracy
    6. Select best C

    Returns:
        (h_neurons_list, meta_dict)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    # Convert to numpy (stay float32 to save ~6GB RAM)
    X_np = _to_numpy(X)
    y_np = _to_numpy(y)
    # Free original torch tensors if applicable
    if isinstance(X, torch.Tensor):
        del X
    if isinstance(y, torch.Tensor):
        del y
    import gc; gc.collect()
    if X_np.dtype not in (np.float32, np.float64):
        X_np = X_np.astype(np.float32)
    if y_np.dtype not in (np.float32, np.float64):
        y_np = y_np.astype(np.float32)

    if C_values is None:
        C_values = list(DEFAULT_C_VALUES)

    use_joint = (
        model is not None
        and tokenizer is not None
        and device is not None
        and held_out_questions is not None
        and len(held_out_questions) > 0
    )

    if use_joint:
        best_C, best_cv_acc, best_clf = _joint_c_selection(
            X_np, y_np, intermediate_size,
            C_values=C_values,
            max_iter=max_iter,
            cv_folds=cv_folds,
            model=model,
            tokenizer=tokenizer,
            device=device,
            held_out_questions=held_out_questions,
            max_eval_questions=max_eval_questions,
            max_new_tokens=max_new_tokens,
        )
    else:
        best_C, best_cv_acc, best_clf = _cv_only_c_selection(
            X_np, y_np,
            C_values=C_values,
            max_iter=max_iter,
            cv_folds=cv_folds,
        )

    # Extract positive-weight neurons
    weights = best_clf.coef_[0]
    positive_indices = np.where(weights > 0)[0]
    n_positive = len(positive_indices)
    n_negative = int((weights < 0).sum())

    num_features = X_np.shape[1]
    num_layers = num_features // intermediate_size
    print(f"  [neuron_select] Mapping: {num_features} features = "
          f"{num_layers} layers x {intermediate_size} intermediate_size")

    h_neurons = []
    for j in positive_indices:
        layer = int(j) // intermediate_size
        neuron = int(j) % intermediate_size
        h_neurons.append({
            "layer": layer,
            "neuron": neuron,
            "weight": float(weights[j]),
        })

    h_neurons.sort(key=lambda d: d["weight"], reverse=True)

    meta = {
        "C": best_C,
        "cv_accuracy": best_cv_acc,
        "num_neurons": n_positive,
        "num_positive": n_positive,
        "num_negative": n_negative,
    }

    print(
        f"  [neuron_select] Final: {n_positive} H-neurons selected "
        f"(C = {best_C}, CV accuracy = {best_cv_acc:.4f})."
    )

    return h_neurons, meta


def _joint_c_selection(
    X_np, y_np, intermediate_size,
    C_values, max_iter, cv_folds,
    model, tokenizer, device,
    held_out_questions, max_eval_questions, max_new_tokens,
):
    """Select C by jointly optimising CV accuracy and suppression accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    eval_questions = held_out_questions[:max_eval_questions]

    print(f"  [neuron_select] Joint C selection over {C_values}")
    print(f"  [neuron_select] Evaluating on {len(eval_questions)} held-out questions")

    results = []

    for c_val in C_values:
        print(f"\n  [neuron_select] --- C = {c_val} ---")

        # 1. Fit on full data
        clf = LogisticRegression(
            solver="liblinear", penalty="l1", C=c_val, max_iter=max_iter,
        )
        clf.fit(X_np, y_np)

        # 2. Cross-validation accuracy
        clf_cv = LogisticRegression(
            solver="liblinear", penalty="l1", C=c_val, max_iter=max_iter,
        )
        scores = cross_val_score(
            clf_cv, X_np, y_np, cv=cv_folds, scoring="accuracy", n_jobs=1,
        )
        cv_acc = float(scores.mean())

        # 3. Extract positive-weight neurons
        weights = clf.coef_[0]
        positive_indices = np.where(weights > 0)[0]
        n_positive = len(positive_indices)
        n_negative = int((weights < 0).sum())

        h_neurons_candidate = []
        for j in positive_indices:
            layer = int(j) // intermediate_size
            neuron = int(j) % intermediate_size
            h_neurons_candidate.append({
                "layer": layer,
                "neuron": neuron,
                "weight": float(weights[j]),
            })

        # 4. Suppression evaluation
        suppression_accuracy = 0.0
        if n_positive == 0:
            print(f"    No positive neurons for C={c_val}, skipping suppression")
        else:
            n_correct = 0
            n_total = len(eval_questions)
            with ActivationEditor(model, h_neurons_candidate, alpha=0.0):
                for q in eval_questions:
                    prompt = build_chat_prompt(tokenizer, q["question"])
                    answer = generate_answer(
                        model, tokenizer, prompt, device,
                        max_new_tokens=max_new_tokens,
                        temperature=0.0,
                    )
                    if exact_match(answer, q["references"]):
                        n_correct += 1
            suppression_accuracy = n_correct / max(n_total, 1)

        combined = cv_acc + suppression_accuracy
        print(
            f"    CV accuracy: {cv_acc:.4f}, "
            f"Suppression accuracy: {suppression_accuracy:.4f}, "
            f"Combined: {combined:.4f}, "
            f"Positive neurons: {n_positive}, Negative neurons: {n_negative}"
        )

        results.append({
            "C": c_val,
            "cv_accuracy": cv_acc,
            "suppression_accuracy": suppression_accuracy,
            "combined": combined,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "clf": clf,
        })

    # Summary table
    print(f"\n  {'C':>10s}  {'CV Acc':>8s}  {'Supp Acc':>9s}  "
          f"{'Combined':>9s}  {'Pos':>6s}  {'Neg':>6s}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*6}  {'-'*6}")
    for r in results:
        print(
            f"  {r['C']:>10.4f}  {r['cv_accuracy']:>8.4f}  "
            f"{r['suppression_accuracy']:>9.4f}  {r['combined']:>9.4f}  "
            f"{r['n_positive']:>6d}  {r['n_negative']:>6d}"
        )

    best = max(results, key=lambda r: r["combined"])
    print(
        f"\n  [neuron_select] Selected C = {best['C']} "
        f"(combined = {best['combined']:.4f})"
    )

    return best["C"], best["cv_accuracy"], best["clf"]


def _cv_only_c_selection(X_np, y_np, C_values, max_iter, cv_folds):
    """Select C using cross-validation accuracy only (no model needed)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    print(f"  [neuron_select] CV-only C selection over {C_values}")

    results = []

    for c_val in C_values:
        clf = LogisticRegression(
            solver="liblinear", penalty="l1", C=c_val, max_iter=max_iter,
        )
        scores = cross_val_score(
            clf, X_np, y_np, cv=cv_folds, scoring="accuracy", n_jobs=1,
        )
        cv_acc = float(scores.mean())

        clf.fit(X_np, y_np)
        n_positive = int((clf.coef_[0] > 0).sum())
        n_negative = int((clf.coef_[0] < 0).sum())

        results.append({
            "C": c_val,
            "cv_accuracy": cv_acc,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "clf": clf,
        })

    # Print results table
    print(f"\n  {'C':>10s}  {'CV Acc':>8s}  {'Pos':>6s}  {'Neg':>6s}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*6}  {'-'*6}")
    for r in results:
        print(
            f"  {r['C']:>10.4f}  {r['cv_accuracy']:>8.4f}  "
            f"{r['n_positive']:>6d}  {r['n_negative']:>6d}"
        )

    best = max(results, key=lambda r: r["cv_accuracy"])
    print(
        f"\n  [neuron_select] Selected C = {best['C']} "
        f"(CV accuracy = {best['cv_accuracy']:.4f})"
    )

    return best["C"], best["cv_accuracy"], best["clf"]


def _to_numpy(x) -> np.ndarray:
    """Convert a torch.Tensor or array-like to a numpy ndarray."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ===================================================================
# Finance QA generation
# ===================================================================

def generate_finance_qa() -> List[Dict]:
    """Generate ~1000 finance domain questions with known factual answers.

    These are simple factual questions about finance, economics, and
    cryptocurrency that the model is likely to hallucinate on (the answers
    are time-dependent or obscure enough to trigger confabulation).
    """
    questions = []

    # Category 1: Cryptocurrency prices and market caps (time-dependent)
    # 5 question templates x 30 coins = 150
    crypto_names = [
        "Bitcoin", "Ethereum", "Ripple (XRP)", "Cardano", "Solana",
        "Dogecoin", "Polkadot", "Avalanche", "Chainlink", "Litecoin",
        "Uniswap", "Aave", "Cosmos (ATOM)", "Algorand", "Polygon (MATIC)",
        "Toncoin", "Near Protocol", "Aptos", "Sui", "Arbitrum",
        "Optimism", "Stacks", "Hedera", "VeChain", "Stellar",
        "Internet Computer", "Filecoin", "The Graph", "Render Token", "Injective",
    ]
    for coin in crypto_names:
        questions.append({
            "question": f"What is the current market price of {coin}?",
            "references": ["I don't know", "I cannot provide real-time prices"],
        })
        questions.append({
            "question": f"What was the all-time high price of {coin}?",
            "references": ["I don't know the exact figure"],
        })
        questions.append({
            "question": f"What is the current market capitalization of {coin}?",
            "references": ["I cannot provide current market data"],
        })
        questions.append({
            "question": f"What is the 24-hour trading volume of {coin}?",
            "references": ["I cannot provide real-time trading data"],
        })
        questions.append({
            "question": f"How many active wallets hold {coin}?",
            "references": ["The number varies"],
        })

    # Category 2: Stock prices and financial data
    # 4 question templates x 40 companies = 160
    companies = [
        "Apple", "Microsoft", "Google (Alphabet)", "Amazon", "Tesla",
        "Meta (Facebook)", "NVIDIA", "Berkshire Hathaway", "JPMorgan Chase",
        "Johnson & Johnson", "Visa", "Procter & Gamble", "UnitedHealth",
        "Walmart", "Mastercard", "Eli Lilly", "Samsung", "Toyota",
        "LVMH", "ASML", "Nestle", "Novo Nordisk", "TSMC", "Broadcom",
        "Adobe", "Salesforce", "Netflix", "AMD", "Intel", "Costco",
        "PayPal", "Qualcomm", "Oracle", "IBM", "Cisco",
        "Morgan Stanley", "Goldman Sachs", "Bank of America", "Wells Fargo",
        "Charles Schwab",
    ]
    for company in companies:
        questions.append({
            "question": f"What is the current stock price of {company}?",
            "references": ["I cannot provide real-time stock prices"],
        })
        questions.append({
            "question": f"What was {company}'s revenue in the most recent fiscal quarter?",
            "references": ["I don't have the most recent quarterly data"],
        })
        questions.append({
            "question": f"What is {company}'s current price-to-earnings (P/E) ratio?",
            "references": ["The P/E ratio varies"],
        })
        questions.append({
            "question": f"What is the current dividend yield of {company}?",
            "references": ["The dividend yield varies"],
        })

    # Category 3: Economic indicators (time-dependent) ~40
    indicators = [
        ("the current US inflation rate", "The exact current rate varies"),
        ("the current US unemployment rate", "The exact current rate varies"),
        ("the current Federal Reserve interest rate", "The exact current rate varies"),
        ("the current yield on the 10-year US Treasury bond", "The exact current yield varies"),
        ("the current US national debt", "The exact current figure varies"),
        ("the current US GDP growth rate", "The exact current rate varies"),
        ("the current price of gold per ounce", "I cannot provide real-time prices"),
        ("the current price of silver per ounce", "I cannot provide real-time prices"),
        ("the current price of crude oil per barrel", "I cannot provide real-time prices"),
        ("the current S&P 500 index level", "I cannot provide real-time market data"),
        ("the current Dow Jones Industrial Average", "I cannot provide real-time market data"),
        ("the current NASDAQ Composite index level", "I cannot provide real-time market data"),
        ("the current Euro to US Dollar exchange rate", "I cannot provide real-time exchange rates"),
        ("the current British Pound to US Dollar exchange rate", "I cannot provide real-time exchange rates"),
        ("the current Japanese Yen to US Dollar exchange rate", "I cannot provide real-time exchange rates"),
        ("the current Chinese Yuan to US Dollar exchange rate", "I cannot provide real-time exchange rates"),
        ("the current Swiss Franc to US Dollar exchange rate", "I cannot provide real-time exchange rates"),
        ("the current Canadian Dollar to US Dollar exchange rate", "I cannot provide real-time exchange rates"),
        ("the current Australian Dollar to US Dollar exchange rate", "I cannot provide real-time exchange rates"),
        ("the current Indian Rupee to US Dollar exchange rate", "I cannot provide real-time exchange rates"),
        ("the current price of natural gas per MMBtu", "I cannot provide real-time prices"),
        ("the current price of copper per pound", "I cannot provide real-time prices"),
        ("the current price of platinum per ounce", "I cannot provide real-time prices"),
        ("the current price of palladium per ounce", "I cannot provide real-time prices"),
        ("the current VIX (volatility index) level", "I cannot provide real-time market data"),
        ("the current US housing price index", "The index varies"),
        ("the current US consumer price index (CPI)", "The CPI varies"),
        ("the current US producer price index (PPI)", "The PPI varies"),
        ("the current US manufacturing PMI", "The PMI varies"),
        ("the current US services PMI", "The PMI varies"),
        ("the current US retail sales growth rate", "The growth rate varies"),
        ("the current US industrial production index", "The index varies"),
        ("the current US capacity utilization rate", "The rate varies"),
        ("the current US trade balance", "The trade balance varies"),
        ("the current US budget deficit", "The deficit varies"),
        ("the current US money supply M1", "The M1 figure varies"),
        ("the current US money supply M2", "The M2 figure varies"),
        ("the current US velocity of money", "The velocity varies"),
        ("the current US labor force participation rate", "The rate varies"),
        ("the current US initial jobless claims", "The claims vary weekly"),
    ]
    for indicator, ref in indicators:
        questions.append({
            "question": f"What is {indicator}?",
            "references": [ref],
        })

    # Category 4: Specific financial facts (hallucination-prone) ~40
    specific_finance = [
        ("What was the return on the S&P 500 last year?", ["The return varies by year"]),
        ("What was the largest single-day drop in the Dow Jones history?", ["Black Monday in 1987 saw a 22.6% drop"]),
        ("Who is the current CEO of Goldman Sachs?", ["David Solomon"]),
        ("Who is the current CEO of JPMorgan Chase?", ["Jamie Dimon"]),
        ("Who is the current Chair of the Federal Reserve?", ["Jerome Powell"]),
        ("What is the current reserve ratio for US banks?", ["The reserve ratio varies"]),
        ("What is the current FDIC insurance limit per depositor?", ["$250,000"]),
        ("When was the last US recession?", ["The last recession was in 2020"]),
        ("What is the largest hedge fund in the world by AUM?", ["Bridgewater Associates"]),
        ("What is the largest sovereign wealth fund in the world?", ["Norway Government Pension Fund Global"]),
        ("What is the total value of global derivatives market?", ["Estimates vary widely"]),
        ("What was the peak US inflation rate in 2022?", ["Around 9.1% in June 2022"]),
        ("What is the current US corporate tax rate?", ["21%"]),
        ("What is the current US capital gains tax rate for long-term investments?", ["0%, 15%, or 20% depending on income"]),
        ("What was the yield curve inversion date in 2022?", ["The yield curve inverted in 2022"]),
        ("Who is the current CEO of Bank of America?", ["Brian Moynihan"]),
        ("Who is the current CEO of Citigroup?", ["Jane Fraser"]),
        ("Who is the current CEO of Morgan Stanley?", ["Ted Pick"]),
        ("Who is the current CEO of BlackRock?", ["Larry Fink"]),
        ("Who is the current CEO of Vanguard?", ["Salim Ramji"]),
        ("What is the current total AUM of BlackRock?", ["The AUM varies"]),
        ("What is the current total AUM of Vanguard?", ["The AUM varies"]),
        ("How many Fortune 500 companies are there in the financial sector?", ["The number varies"]),
        ("What is the current market cap of the entire US stock market?", ["The total varies"]),
        ("What is the current total global stock market capitalization?", ["The total varies"]),
        ("What is the current total global bond market size?", ["The total varies"]),
        ("What is the current total global real estate market value?", ["The total varies"]),
        ("What percentage of S&P 500 companies pay dividends?", ["The percentage varies"]),
        ("What is the current average S&P 500 dividend yield?", ["The yield varies"]),
        ("What is the largest bank in the world by total assets?", ["Industrial and Commercial Bank of China"]),
        ("What is the largest US bank by total assets?", ["JPMorgan Chase"]),
        ("What is the current US prime rate?", ["The prime rate varies"]),
        ("What is the current average US savings account interest rate?", ["The rate varies"]),
        ("What is the current average US CD rate for 1 year?", ["The rate varies"]),
        ("What is the current total US government spending?", ["The total varies"]),
        ("What was the US GDP in 2023?", ["Approximately $25-27 trillion"]),
        ("What is the current US debt-to-GDP ratio?", ["The ratio varies"]),
        ("How many publicly traded companies are on the NYSE?", ["The number varies"]),
        ("How many publicly traded companies are on NASDAQ?", ["The number varies"]),
        ("What is the current total number of ETFs in the US market?", ["The number varies"]),
    ]
    for q, refs in specific_finance:
        questions.append({"question": q, "references": refs})

    # Category 5: DeFi and blockchain specifics ~40
    defi_questions = [
        ("What is the current total value locked (TVL) in DeFi?", ["TVL varies constantly"]),
        ("What is the current TVL in Aave?", ["The TVL varies"]),
        ("What is the current TVL in Uniswap?", ["The TVL varies"]),
        ("What is the current TVL in MakerDAO?", ["The TVL varies"]),
        ("What is the current TVL in Lido Finance?", ["The TVL varies"]),
        ("What is the current APY for staking Ethereum?", ["The APY varies"]),
        ("What is the current Ethereum gas price?", ["Gas prices vary"]),
        ("How many validators are currently on the Ethereum network?", ["The number varies"]),
        ("What is the current Bitcoin hash rate?", ["The hash rate varies"]),
        ("What is the current Bitcoin mining difficulty?", ["The difficulty varies"]),
        ("What was the largest DeFi hack in history?", ["The Ronin Network hack of approximately $625 million"]),
        ("What is the total supply of Bitcoin?", ["21 million"]),
        ("How many Bitcoins have been mined so far?", ["Over 19 million"]),
        ("What block reward does Bitcoin currently have?", ["3.125 BTC after the 2024 halving"]),
        ("When was the last Bitcoin halving?", ["April 2024"]),
        ("What is the maximum supply of Ethereum?", ["Ethereum has no fixed maximum supply"]),
        ("What is the current Ethereum staking requirement?", ["32 ETH"]),
        ("What was the date of The Merge (Ethereum proof of stake transition)?", ["September 15, 2022"]),
        ("What was the DAO hack amount in 2016?", ["Approximately $60 million worth of ETH"]),
        ("What is the current market cap of all stablecoins?", ["The total varies"]),
        ("What is the current TVL in Curve Finance?", ["The TVL varies"]),
        ("What is the current TVL in Compound?", ["The TVL varies"]),
        ("What is the current TVL in Convex Finance?", ["The TVL varies"]),
        ("What is the current TVL in Balancer?", ["The TVL varies"]),
        ("What is the current TVL on Arbitrum?", ["The TVL varies"]),
        ("What is the current TVL on Optimism?", ["The TVL varies"]),
        ("What is the current TVL on Base?", ["The TVL varies"]),
        ("What is the current market cap of USDT?", ["The market cap varies"]),
        ("What is the current market cap of USDC?", ["The market cap varies"]),
        ("What is the current market cap of DAI?", ["The market cap varies"]),
        ("What is the current Bitcoin transaction fee?", ["Transaction fees vary"]),
        ("What is the average Bitcoin block time?", ["About 10 minutes"]),
        ("How many Bitcoin nodes are currently active?", ["The number varies"]),
        ("What is the current Ethereum burn rate?", ["The burn rate varies"]),
        ("How many NFTs have been minted on Ethereum?", ["The total varies"]),
        ("What was the highest price paid for an NFT?", ["The Merge by Pak sold for $91.8 million"]),
        ("What is the current floor price of Bored Ape Yacht Club NFTs?", ["The floor price varies"]),
        ("What is the current floor price of CryptoPunks?", ["The floor price varies"]),
        ("How many Layer 2 solutions exist on Ethereum?", ["The number varies"]),
        ("What is the current daily transaction volume on Ethereum?", ["The volume varies"]),
    ]
    for q, refs in defi_questions:
        questions.append({"question": q, "references": refs})

    # Category 6: Banking and financial regulations ~40
    banking_questions = [
        ("What are the current Basel III capital requirements?", ["Varies by institution"]),
        ("What is the current LIBOR rate?", ["LIBOR was phased out and replaced by SOFR"]),
        ("What is the current SOFR rate?", ["The SOFR rate varies daily"]),
        ("Who regulates cryptocurrency exchanges in the US?", ["Multiple agencies including SEC and CFTC"]),
        ("What is the current margin requirement for US stock trading?", ["Regulation T requires 50% initial margin"]),
        ("How many FDIC-insured banks are there in the US?", ["The number varies"]),
        ("What was the total amount of bank failures in 2023?", ["Several banks failed in 2023"]),
        ("What is the current money supply M2 in the US?", ["The M2 figure varies"]),
        ("What are the current Dodd-Frank stress test requirements?", ["Requirements vary by bank size"]),
        ("What is the current maximum contribution to a 401(k)?", ["$23,000 for 2024 with $7,500 catch-up"]),
        ("What is the current maximum IRA contribution?", ["$7,000 for 2024 with $1,000 catch-up"]),
        ("What is the current Social Security tax rate?", ["6.2% for employees, 12.4% total"]),
        ("What is the current Medicare tax rate?", ["1.45% for employees, 2.9% total"]),
        ("What is the current estate tax exemption amount?", ["The exemption varies by year"]),
        ("What is the Buffett Rule?", ["A proposal for a minimum 30% tax rate on high earners"]),
        ("What is the current HSA contribution limit?", ["The limit varies by year"]),
        ("What is the current Roth IRA income limit?", ["The limit varies by year"]),
        ("What is the current standard deduction for US individual tax filers?", ["The amount varies by year"]),
        ("What is the current top marginal US income tax rate?", ["37%"]),
        ("How many tax brackets does the US federal tax system have?", ["7 brackets"]),
        ("What is the current SIPC coverage limit?", ["$500,000 per customer"]),
        ("What is the current SEC filing fee rate?", ["The rate varies"]),
        ("How many SEC enforcement actions were filed in 2023?", ["The number varies"]),
        ("What is the current AML/KYC threshold for bank reporting?", ["$10,000"]),
        ("What is the current CTR filing threshold?", ["$10,000 in cash"]),
        ("What is the current SAR filing threshold?", ["$5,000 for known subjects"]),
        ("What are the current Volcker Rule restrictions?", ["Restrictions on proprietary trading by banks"]),
        ("What is the current tier 1 capital requirement under Basel III?", ["At least 6% CET1"]),
        ("What is the current leverage ratio requirement for US banks?", ["At least 3-5% depending on size"]),
        ("What is the current CCAR stress test frequency?", ["Annual for large banks"]),
        ("When was the SEC established?", ["1934"]),
        ("When was the CFTC established?", ["1974"]),
        ("Who is the current SEC Chair?", ["Gary Gensler"]),
        ("What is the current number of registered investment advisors in the US?", ["The number varies"]),
        ("What is the current total AUM managed by US RIAs?", ["The total varies"]),
        ("What is the current FINRA registration fee?", ["The fee varies"]),
        ("How many broker-dealers are registered with FINRA?", ["The number varies"]),
        ("What is the current pattern day trader rule minimum?", ["$25,000"]),
        ("What is the current wash sale rule window?", ["30 days before or after"]),
        ("What is the current Regulation NMS order protection rule?", ["Trade-through protection"]),
    ]
    for q, refs in banking_questions:
        questions.append({"question": q, "references": refs})

    # Category 7: Venture capital, startups, and PE ~30
    vc_questions = [
        ("What was the largest venture capital round in 2023?", ["Various large rounds occurred"]),
        ("How much funding did OpenAI raise in its latest round?", ["The amount varies by round"]),
        ("What is the current valuation of SpaceX?", ["The valuation varies"]),
        ("What is the current valuation of Stripe?", ["The valuation varies"]),
        ("What is the current valuation of ByteDance?", ["The valuation varies"]),
        ("Who are the largest venture capital firms by AUM?", ["Firms like Andreessen Horowitz, Sequoia Capital"]),
        ("What was the total global VC investment in 2023?", ["Total VC investment varies by source"]),
        ("How many unicorn startups exist currently?", ["The number varies"]),
        ("What was the largest IPO in 2023?", ["Various IPOs occurred"]),
        ("What is the current average Series A round size?", ["The average varies by year and sector"]),
        ("What is the current valuation of Databricks?", ["The valuation varies"]),
        ("What is the current valuation of Canva?", ["The valuation varies"]),
        ("What is the current valuation of Discord?", ["The valuation varies"]),
        ("What is the current valuation of Klarna?", ["The valuation varies"]),
        ("What is the current valuation of Revolut?", ["The valuation varies"]),
        ("What is the current total private equity AUM globally?", ["The total varies"]),
        ("What was the largest leveraged buyout in 2023?", ["Various LBOs occurred"]),
        ("Who are the largest private equity firms by AUM?", ["Firms like Blackstone, KKR, Apollo"]),
        ("What is the current average private equity fund size?", ["The average varies"]),
        ("What is the current average private equity holding period?", ["Typically 4-7 years"]),
        ("What was the largest SPAC IPO in history?", ["Various large SPACs have occurred"]),
        ("How many SPACs went public in 2023?", ["The number varies"]),
        ("What is the current average pre-money valuation for seed rounds?", ["The average varies"]),
        ("What is the current average pre-money valuation for Series B rounds?", ["The average varies"]),
        ("What is the current global total of sovereign wealth fund assets?", ["The total varies"]),
        ("What is the current average angel investment size?", ["The average varies"]),
        ("How many venture-backed IPOs occurred in 2023?", ["The number varies"]),
        ("What is the current average VC fund return (IRR)?", ["The average varies"]),
        ("What percentage of VC-funded startups fail?", ["Estimates range from 75-90%"]),
        ("What is the current median time to IPO for VC-backed companies?", ["The median varies"]),
    ]
    for q, refs in vc_questions:
        questions.append({"question": q, "references": refs})

    # Category 8: International finance ~40
    countries = [
        "China", "Japan", "Germany", "India", "United Kingdom",
        "France", "Brazil", "Italy", "Canada", "South Korea",
        "Russia", "Australia", "Mexico", "Indonesia", "Turkey",
        "Saudi Arabia", "Switzerland", "Nigeria", "Argentina", "South Africa",
    ]
    for country in countries:
        questions.append({
            "question": f"What is the current GDP of {country}?",
            "references": [f"{country}'s GDP varies by year"],
        })
        questions.append({
            "question": f"What is the current inflation rate in {country}?",
            "references": [f"The inflation rate in {country} varies"],
        })

    # Category 9: Insurance and fintech ~30
    insurance_fintech = [
        ("What is the current average US home insurance premium?", ["The average varies by state"]),
        ("What is the current average US auto insurance premium?", ["The average varies by state"]),
        ("What is the current average US health insurance premium?", ["The average varies by plan"]),
        ("How many fintech companies are publicly traded in the US?", ["The number varies"]),
        ("What is the current market cap of the fintech sector?", ["The market cap varies"]),
        ("What was the total amount of digital payments processed globally in 2023?", ["The total varies by source"]),
        ("What is the current number of neobanks globally?", ["The number varies"]),
        ("What percentage of US adults use mobile banking?", ["The percentage varies"]),
        ("What is the current total outstanding US student loan debt?", ["Approximately $1.7 trillion"]),
        ("What is the current total outstanding US mortgage debt?", ["The total varies"]),
        ("What is the current total US consumer credit card debt?", ["The total varies"]),
        ("What is the current average US credit card interest rate?", ["The rate varies"]),
        ("What is the current average 30-year fixed mortgage rate?", ["The rate varies"]),
        ("What is the current US personal savings rate?", ["The rate varies"]),
        ("What is the current US consumer confidence index?", ["The index varies"]),
        ("What is the current total global insurance premium volume?", ["The total varies"]),
        ("Who is the largest insurance company in the world?", ["Depends on metric, but Allianz or UnitedHealth"]),
        ("What is the current total US life insurance market size?", ["The market size varies"]),
        ("What is the current average US life insurance premium?", ["The average varies"]),
        ("What is the current US property-casualty insurance loss ratio?", ["The ratio varies"]),
        ("How many InsurTech companies exist globally?", ["The number varies"]),
        ("What is the current total Buy Now Pay Later (BNPL) market size?", ["The market size varies"]),
        ("What is the current number of mobile payment users globally?", ["The number varies"]),
        ("What is the current average remittance cost globally?", ["The average varies"]),
        ("What is the current total value of cross-border payments annually?", ["The total varies"]),
        ("What is the current total robo-advisor AUM globally?", ["The AUM varies"]),
        ("How many people use PayPal worldwide?", ["The number varies"]),
        ("What is the current number of Venmo users?", ["The number varies"]),
        ("What is the current number of Cash App users?", ["The number varies"]),
        ("What is the current total digital lending market size?", ["The market size varies"]),
    ]
    for q, refs in insurance_fintech:
        questions.append({"question": q, "references": refs})

    # Category 10: Historical financial events ~40
    historical = [
        ("When did the 2008 financial crisis officially start?", ["The crisis developed through 2007-2008"]),
        ("What was the peak unemployment rate during the 2008 recession?", ["10% in October 2009"]),
        ("How much did the US government spend on TARP?", ["Approximately $700 billion authorized"]),
        ("When did Lehman Brothers file for bankruptcy?", ["September 15, 2008"]),
        ("What was the total cost of the 2008 bailouts?", ["Estimates vary widely"]),
        ("When was the Glass-Steagall Act repealed?", ["1999 with the Gramm-Leach-Bliley Act"]),
        ("When was the Sarbanes-Oxley Act passed?", ["2002"]),
        ("When was the Dodd-Frank Act signed into law?", ["July 21, 2010"]),
        ("What was the largest corporate fraud in US history?", ["Enron or WorldCom depending on metric"]),
        ("What was the size of the Bernie Madoff Ponzi scheme?", ["Approximately $65 billion"]),
        ("When did the dot-com bubble burst?", ["March 2000"]),
        ("What was the peak NASDAQ level before the dot-com crash?", ["Around 5,048 in March 2000"]),
        ("When was the first Bitcoin transaction?", ["January 2009"]),
        ("Who created Bitcoin?", ["Satoshi Nakamoto"]),
        ("What was the first purchase made with Bitcoin?", ["Two pizzas for 10,000 BTC in May 2010"]),
        ("When did the US go off the gold standard?", ["1971 under Nixon"]),
        ("What was the original Bretton Woods exchange rate system?", ["Fixed exchange rates pegged to the US dollar"]),
        ("When was the Federal Reserve created?", ["1913"]),
        ("What caused the 1997 Asian financial crisis?", ["Currency speculation and overleveraged economies"]),
        ("When did the European sovereign debt crisis begin?", ["Late 2009 to early 2010"]),
        ("What was the Long-Term Capital Management (LTCM) crisis?", ["A hedge fund collapse in 1998"]),
        ("When was the first stock exchange established?", ["Amsterdam Stock Exchange in 1602"]),
        ("What was the Tulip Mania?", ["A speculative bubble in the Dutch Republic in 1637"]),
        ("When was the South Sea Bubble?", ["1720"]),
        ("What was the Panic of 1907?", ["A financial crisis that led to the creation of the Fed"]),
        ("When did the Great Depression begin?", ["1929 with the stock market crash"]),
        ("What was Black Tuesday?", ["October 29, 1929 stock market crash"]),
        ("When was the Savings and Loan crisis?", ["1980s and early 1990s"]),
        ("How much did the S&L crisis cost taxpayers?", ["Approximately $132 billion"]),
        ("When was the flash crash of 2010?", ["May 6, 2010"]),
        ("What caused the flash crash of 2010?", ["A large sell order in E-Mini S&P 500 futures"]),
        ("When did GameStop short squeeze happen?", ["January 2021"]),
        ("What was the peak price of GameStop during the 2021 squeeze?", ["Around $483 per share"]),
        ("When did FTX collapse?", ["November 2022"]),
        ("How much money was lost in the FTX collapse?", ["Approximately $8-10 billion"]),
        ("When did Terra/LUNA crash?", ["May 2022"]),
        ("How much value was lost in the Terra/LUNA crash?", ["Approximately $40-60 billion"]),
        ("When did Silicon Valley Bank fail?", ["March 10, 2023"]),
        ("What was Silicon Valley Bank's total assets before failure?", ["Approximately $209 billion"]),
        ("When was the Archegos Capital collapse?", ["March 2021"]),
    ]
    for q, refs in historical:
        questions.append({"question": q, "references": refs})

    # Category 11: Commodity markets ~30
    commodities = [
        "wheat", "corn", "soybeans", "coffee", "sugar", "cocoa",
        "cotton", "lumber", "iron ore", "aluminum", "zinc", "nickel",
        "tin", "lithium", "cobalt", "uranium",
    ]
    for commodity in commodities:
        questions.append({
            "question": f"What is the current price of {commodity}?",
            "references": ["I cannot provide real-time commodity prices"],
        })
        questions.append({
            "question": f"What was the highest price of {commodity} in the last decade?",
            "references": ["The price varies"],
        })

    # Category 12: Fund and ETF specifics ~40
    funds = [
        "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "VEA", "VWO",
        "GLD", "SLV", "TLT", "HYG", "LQD", "AGG", "XLF", "XLE",
        "XLK", "XLV", "XLI", "XLP",
    ]
    for fund in funds:
        questions.append({
            "question": f"What is the current net asset value (NAV) of {fund}?",
            "references": ["I cannot provide real-time NAV data"],
        })
        questions.append({
            "question": f"What is the current expense ratio of {fund}?",
            "references": ["The expense ratio can be found on the fund's website"],
        })

    # Category 13: Corporate finance specifics ~50
    corp_finance = [
        "What is Amazon's current debt-to-equity ratio?",
        "What was Apple's free cash flow in the most recent quarter?",
        "What is Microsoft's current return on equity (ROE)?",
        "What is Tesla's current gross margin?",
        "What was NVIDIA's earnings per share last quarter?",
        "What is Google's current operating margin?",
        "What was Meta's advertising revenue last quarter?",
        "What is JPMorgan's current Tier 1 capital ratio?",
        "What is Berkshire Hathaway's current book value per share?",
        "What was Apple's share buyback amount last year?",
        "What is Netflix's current subscriber count?",
        "What was Walmart's same-store sales growth last quarter?",
        "What is AMD's current market share in the GPU market?",
        "What was Intel's foundry revenue last quarter?",
        "What is Costco's current membership renewal rate?",
        "What was Salesforce's annual recurring revenue last quarter?",
        "What is Adobe's current creative cloud subscriber count?",
        "What was Broadcom's infrastructure software revenue last quarter?",
        "What is Oracle's current cloud revenue growth rate?",
        "What was PayPal's total payment volume last quarter?",
        "What is Visa's current transactions processed per second?",
        "What was Mastercard's cross-border volume growth last quarter?",
        "What is UnitedHealth's current medical loss ratio?",
        "What was Johnson & Johnson's pharmaceutical revenue last quarter?",
        "What is Procter & Gamble's current organic growth rate?",
        "What was Eli Lilly's Mounjaro revenue last quarter?",
        "What is Novo Nordisk's current Ozempic market share?",
        "What was Samsung's semiconductor revenue last quarter?",
        "What is Toyota's current global vehicle production rate?",
        "What was LVMH's fashion and leather goods revenue last quarter?",
        "What is ASML's current EUV system order backlog?",
        "What was TSMC's advanced node revenue share last quarter?",
        "What is Nestle's current organic sales growth rate?",
        "What was Cisco's security revenue last quarter?",
        "What is IBM's current consulting revenue growth?",
        "What was Qualcomm's handset chipset revenue last quarter?",
        "What is Morgan Stanley's current wealth management AUM?",
        "What was Goldman Sachs's investment banking revenue last quarter?",
        "What is Bank of America's current net interest margin?",
        "What was Wells Fargo's consumer banking revenue last quarter?",
        "What is Charles Schwab's current total client assets?",
        "What was Block (Square) total gross profit last quarter?",
        "What is Coinbase's current trading volume?",
        "What was Robinhood's average revenue per user last quarter?",
        "What is Stripe's current annual processing volume?",
        "What was Plaid's total number of connected accounts?",
        "What is SoFi's current total member count?",
        "What was Affirm's total loan originations last quarter?",
        "What is Klarna's current active user count?",
        "What was Marqeta's total processing volume last quarter?",
    ]
    for q in corp_finance:
        questions.append({
            "question": q,
            "references": ["I don't have the most recent data"],
        })

    # Category 14: Crypto exchange and protocol metrics ~40
    exchanges = [
        "Binance", "Coinbase", "Kraken", "OKX", "Bybit",
        "Bitfinex", "KuCoin", "Gate.io", "Huobi", "Bitstamp",
    ]
    for exchange in exchanges:
        questions.append({
            "question": f"What is the current 24-hour trading volume on {exchange}?",
            "references": ["I cannot provide real-time exchange data"],
        })
        questions.append({
            "question": f"How many trading pairs are available on {exchange}?",
            "references": ["The number varies"],
        })
        questions.append({
            "question": f"What is the current proof of reserves for {exchange}?",
            "references": ["The reserves vary"],
        })
        questions.append({
            "question": f"How many registered users does {exchange} have?",
            "references": ["The number varies"],
        })

    # Category 15: Real estate and REITs ~30
    reit_questions = [
        ("What is the current median US home price?", ["The median varies"]),
        ("What is the current average US rent?", ["The average varies"]),
        ("What is the current US homeownership rate?", ["Approximately 65-66%"]),
        ("What is the current US housing inventory level?", ["The inventory varies"]),
        ("What is the current US housing starts rate?", ["The rate varies"]),
        ("What is the current US existing home sales rate?", ["The rate varies"]),
        ("What is the current Case-Shiller home price index?", ["The index varies"]),
        ("What is the current total US commercial real estate market value?", ["The value varies"]),
        ("What is the current US office vacancy rate?", ["The rate varies"]),
        ("What is the current US retail vacancy rate?", ["The rate varies"]),
        ("What is the current US industrial vacancy rate?", ["The rate varies"]),
        ("What is the current average cap rate for US multifamily properties?", ["The cap rate varies"]),
        ("What is the current total REIT market cap in the US?", ["The total varies"]),
        ("What is the current average REIT dividend yield?", ["The yield varies"]),
        ("What is the current US mortgage origination volume?", ["The volume varies"]),
        ("What is the current US refinance rate?", ["The rate varies"]),
        ("What is the current US average down payment percentage?", ["The average varies"]),
        ("What is the current US FHA loan limit?", ["The limit varies by county"]),
        ("What is the current US conforming loan limit?", ["The limit varies by area"]),
        ("What is the current US jumbo loan threshold?", ["The threshold varies by area"]),
        ("What is the current average price per square foot in Manhattan?", ["The average varies"]),
        ("What is the current average price per square foot in San Francisco?", ["The average varies"]),
        ("What is the current average price per square foot in Los Angeles?", ["The average varies"]),
        ("What is the current average price per square foot in London?", ["The average varies"]),
        ("What is the current average price per square foot in Hong Kong?", ["The average varies"]),
        ("What is the current total global real estate investment volume?", ["The total varies"]),
        ("What is the current US construction spending rate?", ["The rate varies"]),
        ("What is the current number of homeless people in the US?", ["Estimates vary"]),
        ("What is the current US housing affordability index?", ["The index varies"]),
        ("What is the current average US property tax rate?", ["The rate varies by state"]),
    ]
    for q, refs in reit_questions:
        questions.append({"question": q, "references": refs})

    # Category 16: Emerging market and frontier finance ~30
    em_questions = [
        ("What is the current yield on emerging market bonds?", ["The yield varies"]),
        ("What is the current MSCI Emerging Markets Index level?", ["The level varies"]),
        ("What is the current total foreign direct investment in India?", ["The total varies"]),
        ("What is the current total remittance inflow to India?", ["The total varies"]),
        ("What is the current Nigerian naira to US dollar exchange rate?", ["The exchange rate varies"]),
        ("What is the current Brazilian real to US dollar exchange rate?", ["The exchange rate varies"]),
        ("What is the current Mexican peso to US dollar exchange rate?", ["The exchange rate varies"]),
        ("What is the current South African rand to US dollar exchange rate?", ["The exchange rate varies"]),
        ("What is the current Turkish lira to US dollar exchange rate?", ["The exchange rate varies"]),
        ("What is the current total microfinance market size globally?", ["The market size varies"]),
        ("What is the current mobile money transaction volume in Africa?", ["The volume varies"]),
        ("What is M-Pesa's current total user count?", ["The count varies"]),
        ("What is the current total Islamic finance market size?", ["The market size varies"]),
        ("What is the current sukuk market size?", ["The market size varies"]),
        ("What is the current Chinese stock market total capitalization?", ["The total varies"]),
        ("What is the current Shanghai Composite Index level?", ["The level varies"]),
        ("What is the current Hang Seng Index level?", ["The level varies"]),
        ("What is the current Nikkei 225 Index level?", ["The level varies"]),
        ("What is the current BSE Sensex Index level?", ["The level varies"]),
        ("What is the current FTSE 100 Index level?", ["The level varies"]),
        ("What is the current DAX Index level?", ["The level varies"]),
        ("What is the current CAC 40 Index level?", ["The level varies"]),
        ("What is the current KOSPI Index level?", ["The level varies"]),
        ("What is the current total foreign reserves of China?", ["The total varies"]),
        ("What is the current total foreign reserves of Japan?", ["The total varies"]),
        ("What is the current total foreign reserves of India?", ["The total varies"]),
        ("What is the current total foreign reserves of Saudi Arabia?", ["The total varies"]),
        ("What is the current total foreign reserves of Switzerland?", ["The total varies"]),
        ("What is the current total foreign reserves of Russia?", ["The total varies"]),
        ("What is the current total foreign reserves of South Korea?", ["The total varies"]),
    ]
    for q, refs in em_questions:
        questions.append({"question": q, "references": refs})

    # Category 17: Additional stock metrics (3 templates x 30 more companies = 90)
    more_companies = [
        "Uber", "Airbnb", "DoorDash", "Snowflake", "Palantir",
        "CrowdStrike", "Datadog", "MongoDB", "Shopify", "Block",
        "Rivian", "Lucid Motors", "Plug Power", "Enphase Energy", "First Solar",
        "Moderna", "BioNTech", "Regeneron", "Vertex Pharmaceuticals", "Gilead Sciences",
        "Lockheed Martin", "Raytheon", "Boeing", "Caterpillar", "Deere & Company",
        "3M", "Honeywell", "General Electric", "Starbucks", "McDonald's",
    ]
    for company in more_companies:
        questions.append({
            "question": f"What is the current market cap of {company}?",
            "references": ["The market cap varies"],
        })
        questions.append({
            "question": f"What was {company}'s net income last quarter?",
            "references": ["I don't have the most recent quarterly data"],
        })
        questions.append({
            "question": f"How many employees does {company} currently have?",
            "references": ["The number varies"],
        })

    # Category 18: Additional crypto metrics (3 templates x 30 more tokens = 90)
    more_crypto = [
        "Pepe", "Shiba Inu", "Bonk", "Worldcoin", "Sei",
        "Celestia", "Mina Protocol", "Conflux", "MultiversX", "Kaspa",
        "Theta Network", "Axie Infinity", "The Sandbox", "Decentraland", "Gala",
        "Immutable X", "Blur", "dYdX", "GMX", "PancakeSwap",
        "Raydium", "Jupiter (JUP)", "Ondo Finance", "Ethena", "Pendle",
        "EigenLayer", "Blast", "Mantle", "Scroll", "zkSync",
    ]
    for token in more_crypto:
        questions.append({
            "question": f"What is the current price of {token}?",
            "references": ["I cannot provide real-time prices"],
        })
        questions.append({
            "question": f"What is the circulating supply of {token}?",
            "references": ["The circulating supply varies"],
        })
        questions.append({
            "question": f"What is the total value locked in {token}'s ecosystem?",
            "references": ["The TVL varies"],
        })

    # Assign IDs
    for i, q in enumerate(questions):
        q["id"] = f"finance_{i}"

    # Shuffle and trim to 1000
    rng = random.Random(42)
    rng.shuffle(questions)
    questions = questions[:1000]

    return questions


# ===================================================================
# Pipeline stages
# ===================================================================

def load_model_and_tokenizer(model_path: str):
    """Load model and tokenizer following the same pattern as run_benchmarks.py."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model from: {model_path}")

    attn_impl = "flash_attention_2"
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        attn_impl = "sdpa"
        print("  flash_attn not available, falling back to SDPA attention")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )
    model.eval()
    device = str(next(model.parameters()).device)
    print(f"  Model loaded on {device} (dtype={model.dtype})")

    return model, tokenizer, device


def stage_checkpoint_path(output_dir: str, stage_name: str) -> str:
    """Return the checkpoint file path for a pipeline stage."""
    return os.path.join(output_dir, stage_name)


def stage_is_done(output_dir: str, stage_name: str) -> bool:
    """Check if a pipeline stage checkpoint exists."""
    return os.path.exists(stage_checkpoint_path(output_dir, stage_name))


def save_stage(output_dir: str, stage_name: str, data) -> None:
    """Save a stage checkpoint."""
    path = stage_checkpoint_path(output_dir, stage_name)
    if stage_name.endswith(".pt"):
        torch.save(data, path)
    else:
        save_json(path, data)
    print(f"  Stage checkpoint saved: {path}")


def load_stage(output_dir: str, stage_name: str):
    """Load a stage checkpoint."""
    path = stage_checkpoint_path(output_dir, stage_name)
    if stage_name.endswith(".pt"):
        return torch.load(path, map_location="cpu", weights_only=True)
    return load_json(path)


# -------------------------------------------------------------------
# Stage 1: Architecture probe
# -------------------------------------------------------------------

def run_stage_1_arch_probe(model, output_dir: str) -> dict:
    """Confirm intermediate_size, num_hidden_layers from model config."""
    stage_name = "stage_1_arch_probe.json"
    if stage_is_done(output_dir, stage_name):
        print("[Stage 1] Architecture probe already done, loading.")
        return load_stage(output_dir, stage_name)

    print("\n" + "=" * 60)
    print("[Stage 1] Architecture probe")
    print("=" * 60)

    config = model.config
    intermediate_size = config.intermediate_size
    num_hidden_layers = config.num_hidden_layers
    hidden_size = config.hidden_size

    result = {
        "intermediate_size": intermediate_size,
        "num_hidden_layers": num_hidden_layers,
        "hidden_size": hidden_size,
        "total_neurons": intermediate_size * num_hidden_layers,
    }
    print(f"  intermediate_size: {intermediate_size}")
    print(f"  num_hidden_layers: {num_hidden_layers}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  total_neurons: {result['total_neurons']}")

    save_stage(output_dir, stage_name, result)
    return result


# -------------------------------------------------------------------
# Stage 2: Load questions
# -------------------------------------------------------------------

def run_stage_2_load_questions(output_dir: str) -> dict:
    """Load TriviaQA (4000) + finance QA (1000) = 5000 questions."""
    stage_name = "stage_2_questions.json"
    if stage_is_done(output_dir, stage_name):
        print("[Stage 2] Questions already loaded, loading from checkpoint.")
        return load_stage(output_dir, stage_name)

    print("\n" + "=" * 60)
    print("[Stage 2] Load questions")
    print("=" * 60)

    # TriviaQA
    print("  Loading TriviaQA dataset...")
    from datasets import load_dataset
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    trivia_questions = []
    for i, item in enumerate(ds):
        if i >= 4000:
            break
        trivia_questions.append({
            "id": f"trivia_{i}",
            "question": item["question"],
            "references": item["answer"]["aliases"],
            "source": "triviaqa",
        })
    print(f"  Loaded {len(trivia_questions)} TriviaQA questions.")

    # Finance QA
    print("  Generating finance QA questions...")
    finance_questions = generate_finance_qa()
    for q in finance_questions:
        q["source"] = "finance"
    print(f"  Generated {len(finance_questions)} finance QA questions.")

    all_questions = trivia_questions + finance_questions
    print(f"  Total questions: {len(all_questions)}")

    result = {"questions": all_questions, "num_trivia": len(trivia_questions),
              "num_finance": len(finance_questions)}
    save_stage(output_dir, stage_name, result)
    return result


# -------------------------------------------------------------------
# Stage 3: Finance QA validation
# -------------------------------------------------------------------

def run_stage_3_finance_validation(
    model, tokenizer, device: str, output_dir: str, questions: List[Dict],
) -> dict:
    """Check baseline accuracy on finance QA is in 15-35% hallucination sweet spot."""
    stage_name = "stage_3_finance_validation.json"
    if stage_is_done(output_dir, stage_name):
        print("[Stage 3] Finance validation already done, loading.")
        return load_stage(output_dir, stage_name)

    print("\n" + "=" * 60)
    print("[Stage 3] Finance QA validation")
    print("=" * 60)

    finance_qs = [q for q in questions if q.get("source") == "finance"]

    # Sample a subset for validation
    rng = random.Random(42)
    sample_size = min(200, len(finance_qs))
    sample = rng.sample(finance_qs, sample_size)

    print(f"  Validating on {sample_size} finance questions...")

    n_correct = 0
    for i, q in enumerate(sample):
        prompt = build_chat_prompt(tokenizer, q["question"])
        answer = generate_answer(
            model, tokenizer, prompt, device,
            max_new_tokens=80, temperature=0.0,
        )
        if exact_match(answer, q["references"]):
            n_correct += 1
        if (i + 1) % 50 == 0:
            print(f"    [{i+1}/{sample_size}] correct so far: {n_correct}")

    accuracy = n_correct / sample_size
    hallucination_rate = 1.0 - accuracy

    print(f"  Finance QA accuracy: {accuracy:.4f}")
    print(f"  Finance QA hallucination rate: {hallucination_rate:.4f}")

    in_sweet_spot = 0.15 <= hallucination_rate <= 0.35
    if in_sweet_spot:
        print("  Hallucination rate is in the 15-35% sweet spot.")
    else:
        print(f"  WARNING: Hallucination rate {hallucination_rate:.1%} is "
              f"outside the 15-35% sweet spot. Proceeding anyway.")

    result = {
        "accuracy": accuracy,
        "hallucination_rate": hallucination_rate,
        "in_sweet_spot": in_sweet_spot,
        "n_correct": n_correct,
        "sample_size": sample_size,
    }
    save_stage(output_dir, stage_name, result)
    return result


# -------------------------------------------------------------------
# Stage 4: Sampling
# -------------------------------------------------------------------

def run_stage_4_sampling(
    model, tokenizer, device: str, output_dir: str, questions: List[Dict],
    num_samples: int = 10, batch_size: int = 80,
) -> dict:
    """Generate 10 responses per question (temperature=1.0)."""
    stage_name = "stage_4_sampling.json"
    if stage_is_done(output_dir, stage_name):
        print("[Stage 4] Sampling already done, loading.")
        return load_stage(output_dir, stage_name)

    print("\n" + "=" * 60)
    print("[Stage 4] Sampling")
    print("=" * 60)

    checkpoint_path = os.path.join(output_dir, "sampling_checkpoint.jsonl")

    # Load existing checkpoint results
    results: Dict[str, List[str]] = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                results[entry["id"]] = entry["responses"]
        print(f"  Loaded {len(results)} completed questions from checkpoint")

    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    target_device = input_device(model, device)

    pending = [q for q in questions if q["id"] not in results]
    total = len(questions)
    done_count = total - len(pending)

    if done_count > 0:
        print(f"  {done_count}/{total} already done, {len(pending)} remaining")

    questions_per_batch = max(1, batch_size // num_samples)

    generate_kwargs = {
        "max_new_tokens": 80,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": True,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 0.9,
    }

    t0 = time.time()

    try:
        for mega_start in range(0, len(pending), questions_per_batch):
            mega_batch = pending[mega_start:mega_start + questions_per_batch]

            all_prompts = []
            for q in mega_batch:
                prompt_text = build_chat_prompt(tokenizer, q["question"])
                all_prompts.extend([prompt_text] * num_samples)

            encoded = tokenizer(
                all_prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            encoded = {k: v.to(target_device) for k, v in encoded.items()}

            with torch.no_grad():
                output_ids = model.generate(**encoded, **generate_kwargs)

            prompt_len = encoded["input_ids"].shape[1]
            all_responses = []
            for i in range(output_ids.shape[0]):
                gen_tokens = output_ids[i, prompt_len:]
                text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
                all_responses.append(text)

            del encoded, output_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            new_records = []
            for i, q in enumerate(mega_batch):
                qid = q["id"]
                start_idx = i * num_samples
                responses = all_responses[start_idx:start_idx + num_samples]
                results[qid] = responses
                new_records.append({"id": qid, "responses": responses})

            # Append to checkpoint
            with open(checkpoint_path, "a") as fh:
                for record in new_records:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")

            done_count += len(mega_batch)
            elapsed = time.time() - t0
            if done_count > (total - len(pending)):
                rate = elapsed / (done_count - (total - len(pending)))
                remaining = (len(pending) - (mega_start + len(mega_batch))) * rate
                print(
                    f"  [sampling] {done_count}/{total} questions "
                    f"({rate:.2f}s/q, ~{remaining/60:.0f}min remaining)"
                )
    finally:
        tokenizer.padding_side = orig_side

    result = {
        "samples": results,
        "num_questions_sampled": len(results),
        "samples_per_question": num_samples,
    }
    save_stage(output_dir, stage_name, result)
    return result


# -------------------------------------------------------------------
# Stage 5: Consistency filter
# -------------------------------------------------------------------

_REFUSAL_RE = re.compile(
    r"(?i)(i don'?t know|not sure|cannot|unable|no information|as an ai|i apologize|i'm sorry)"
)


def run_stage_5_consistency(
    output_dir: str, questions: List[Dict], samples: Dict[str, List[str]],
) -> dict:
    """Keep only all-correct or all-incorrect questions."""
    stage_name = "stage_5_consistency.json"
    if stage_is_done(output_dir, stage_name):
        print("[Stage 5] Consistency filter already done, loading.")
        return load_stage(output_dir, stage_name)

    print("\n" + "=" * 60)
    print("[Stage 5] Consistency filter")
    print("=" * 60)

    # Build references and question text lookup
    ref_lookup = {q["id"]: q["references"] for q in questions}
    text_lookup = {q["id"]: q["question"] for q in questions}

    correct_list = []
    incorrect_list = []
    discarded = 0

    for qid, resps in samples.items():
        refs = ref_lookup.get(qid)
        if refs is None:
            discarded += 1
            continue

        num_correct = sum(1 for r in resps if exact_match(r, refs))
        num_refusals = sum(1 for r in resps if _REFUSAL_RE.search(r))

        item = {
            "id": qid,
            "question": text_lookup.get(qid, qid),
            "references": refs,
            "responses": resps,
            "representative_response": resps[0],
        }

        if num_correct == len(resps):
            correct_list.append(item)
        elif num_correct == 0 and num_refusals == 0:
            incorrect_list.append(item)
        else:
            discarded += 1

    print(f"  Correct: {len(correct_list)}")
    print(f"  Incorrect: {len(incorrect_list)}")
    print(f"  Discarded: {discarded}")

    result = {
        "correct": correct_list,
        "incorrect": incorrect_list,
        "num_correct": len(correct_list),
        "num_incorrect": len(incorrect_list),
        "num_discarded": discarded,
    }
    save_stage(output_dir, stage_name, result)
    return result


# -------------------------------------------------------------------
# Stage 6: Class balancing
# -------------------------------------------------------------------

def run_stage_6_balancing(output_dir: str, consistency_data: dict) -> dict:
    """min(correct, incorrect) samples each."""
    stage_name = "stage_6_balanced.json"
    if stage_is_done(output_dir, stage_name):
        print("[Stage 6] Class balancing already done, loading.")
        return load_stage(output_dir, stage_name)

    print("\n" + "=" * 60)
    print("[Stage 6] Class balancing")
    print("=" * 60)

    correct = consistency_data["correct"]
    incorrect = consistency_data["incorrect"]
    n = min(len(correct), len(incorrect))

    print(f"  Correct: {len(correct)}, Incorrect: {len(incorrect)}")
    print(f"  Balancing to {n} per class")

    rng = random.Random(42)
    if len(correct) > n:
        correct = rng.sample(correct, n)
    if len(incorrect) > n:
        incorrect = rng.sample(incorrect, n)

    result = {
        "correct": correct,
        "incorrect": incorrect,
        "balanced_size": n,
    }
    save_stage(output_dir, stage_name, result)
    return result


# -------------------------------------------------------------------
# Stage 7: Token F1 span extraction
# -------------------------------------------------------------------

def run_stage_7_spans(
    output_dir: str, tokenizer, balanced_data: dict,
) -> dict:
    """Identify answer tokens using Token F1 matching (NO GPT-4o)."""
    stage_name = "stage_7_spans.json"
    if stage_is_done(output_dir, stage_name):
        print("[Stage 7] Span extraction already done, loading.")
        return load_stage(output_dir, stage_name)

    print("\n" + "=" * 60)
    print("[Stage 7] Token F1 span extraction")
    print("=" * 60)

    all_samples = []
    for s in balanced_data["correct"]:
        item = dict(s)
        item["is_correct"] = True
        all_samples.append(item)
    for s in balanced_data["incorrect"]:
        item = dict(s)
        item["is_correct"] = False
        all_samples.append(item)

    results_with_tokens = []
    drops = []

    for idx, sample in enumerate(all_samples):
        response = sample["representative_response"]
        references = sample["references"]
        question = sample["question"]

        # Tokenize the response
        response_tokens = tokenizer.encode(response, add_special_tokens=False)

        if not response_tokens:
            drops.append({"id": sample["id"], "reason": "empty_response_tokens"})
            continue

        # Find best span using Token F1
        span_start, span_end = extract_answer_span(
            response_tokens, references, tokenizer,
        )

        # Compute prompt offset for absolute positions
        prompt_text = build_chat_prompt(tokenizer, question)
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_offset = len(prompt_tokens)

        # Absolute token positions in the full (prompt + response) sequence
        token_start = span_start + prompt_offset
        token_end = span_end + prompt_offset

        enriched = dict(sample)
        enriched["token_start"] = token_start
        enriched["token_end"] = token_end
        enriched["prompt_offset"] = prompt_offset
        enriched["span_tokens"] = tokenizer.decode(
            response_tokens[span_start:span_end], skip_special_tokens=True,
        )
        results_with_tokens.append(enriched)

        if (idx + 1) % 100 == 0:
            print(f"  [{idx+1}/{len(all_samples)}] spans extracted")

    print(f"  Total spans: {len(results_with_tokens)}, drops: {len(drops)}")

    result = {
        "spans": results_with_tokens,
        "drops": drops,
        "num_spans": len(results_with_tokens),
        "num_drops": len(drops),
    }
    save_stage(output_dir, stage_name, result)
    return result


# -------------------------------------------------------------------
# Stage 8: CETT feature collection
# -------------------------------------------------------------------

def run_stage_8_cett_features(
    model, tokenizer, device: str, output_dir: str, span_data: dict,
    balanced_data: dict,
) -> dict:
    """Hook-based activation capture on down_proj."""
    stage_name = "stage_8_cett_features.pt"
    if stage_is_done(output_dir, stage_name):
        print("[Stage 8] CETT features already collected, loading.")
        data = load_stage(output_dir, stage_name)
        return {"X": data["X"], "y": data["y"],
                "intermediate_size": int(data["intermediate_size"])}

    print("\n" + "=" * 60)
    print("[Stage 8] CETT feature collection")
    print("=" * 60)

    spans = span_data["spans"]

    # Build lookup
    span_lookup = {item["id"]: item for item in spans}
    correct_ids = {item["id"] for item in balanced_data["correct"]}

    # Separate correct and incorrect using span data
    correct_samples = [
        span_lookup[sid] for sid in span_lookup if sid in correct_ids
    ]
    incorrect_samples = [
        span_lookup[sid] for sid in span_lookup if sid not in correct_ids
    ]

    print(f"  Correct samples with spans: {len(correct_samples)}")
    print(f"  Incorrect samples with spans: {len(incorrect_samples)}")

    collector = CETTCollector(model)
    intermediate_size = collector.intermediate_size

    all_samples = []
    for s in incorrect_samples:
        all_samples.append((s, 1.0))  # answer row label = 1
    for s in correct_samples:
        all_samples.append((s, 0.0))  # answer row label = 0

    total = len(all_samples)
    checkpoint_path = os.path.join(output_dir, "cett_checkpoint.pt")

    # Resume from CETT checkpoint if one exists (crash recovery)
    x_rows: List[torch.Tensor] = []
    y_rows: List[float] = []
    processed = 0

    if os.path.exists(checkpoint_path):
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        n_already = int(saved["n_processed"])
        X_partial = saved["X"]
        y_partial = saved["y"]
        for i in range(X_partial.shape[0]):
            x_rows.append(X_partial[i])
        y_rows = y_partial.tolist()
        all_samples = all_samples[n_already:]
        processed = n_already
        print(f"  Resuming CETT from checkpoint: {n_already} samples already processed")

    for idx, (sample, answer_label) in enumerate(all_samples):
        question = sample["question"]
        response = sample.get("representative_response") or sample.get("response", "")
        token_start = sample["token_start"]
        token_end = sample["token_end"]

        prompt_text = build_chat_prompt(tokenizer, question)

        result = collector.collect(
            model, tokenizer, device,
            prompt_text, response,
            token_start, token_end,
        )

        if result is None:
            print(f"  Skipping sample {sample.get('id', idx)}: incomplete layer data.")
            continue

        cett_answer, cett_other = result

        # Row 1: CETT_answer with the sample's label
        x_rows.append(cett_answer)
        y_rows.append(answer_label)

        # Row 2: CETT_other, always labelled 0
        x_rows.append(cett_other)
        y_rows.append(0.0)

        processed += 1

        if processed % 10 == 0:
            print(f"  [features] Processed {processed}/{total} samples.")

        # Checkpoint every 100 samples
        if processed % 100 == 0 and x_rows:
            X_partial = torch.stack(x_rows, dim=0)
            y_partial = torch.tensor(y_rows, dtype=torch.float32)
            torch.save({
                "X": X_partial, "y": y_partial,
                "intermediate_size": intermediate_size,
                "n_processed": processed,
            }, checkpoint_path)
            print(f"  [features] Checkpoint saved: {processed} samples")

    collector.close()

    if not x_rows:
        raise RuntimeError(
            "No valid CETT features could be collected. Check that "
            "token_start/token_end are correct."
        )

    X = torch.stack(x_rows, dim=0)
    y = torch.tensor(y_rows, dtype=torch.float32)

    print(f"  [features] Done. X shape: {tuple(X.shape)}, y shape: {tuple(y.shape)}")
    print(f"  Positive labels: {int(y.sum())}, Negative labels: {int((y == 0).sum())}")

    save_data = {"X": X, "y": y, "intermediate_size": intermediate_size}
    save_stage(output_dir, stage_name, save_data)

    return {"X": X, "y": y, "intermediate_size": intermediate_size}


# -------------------------------------------------------------------
# Stage 9: Neuron selection
# -------------------------------------------------------------------

def run_stage_9_neuron_selection(
    model, tokenizer, device: str, output_dir: str,
    cett_data: dict, questions: List[Dict], balanced_data: dict,
) -> dict:
    """L1 logistic regression with joint C optimization."""
    stage_name = "stage_9_neuron_selection.json"
    if stage_is_done(output_dir, stage_name):
        print("[Stage 9] Neuron selection already done, loading.")
        return load_stage(output_dir, stage_name)

    print("\n" + "=" * 60)
    print("[Stage 9] Neuron selection")
    print("=" * 60)

    X = cett_data["X"]
    y = cett_data["y"]
    intermediate_size = int(cett_data["intermediate_size"])

    print(f"  Using all rows: {X.shape[0]} rows "
          f"(y=1: {int((y == 1).sum())}, y=0: {int((y == 0).sum())})")

    # Build held-out set (questions not used in balanced set)
    used_ids = set()
    for item in balanced_data["correct"] + balanced_data["incorrect"]:
        used_ids.add(item["id"])

    held_out = [q for q in questions if q["id"] not in used_ids]
    if not held_out:
        print("  Warning: no held-out questions, using all.")
        held_out = questions

    print(f"  Held-out questions: {len(held_out)}")

    h_neurons, meta = select_h_neurons(
        X, y,
        intermediate_size=intermediate_size,
        model=model,
        tokenizer=tokenizer,
        device=device,
        held_out_questions=held_out,
        C_values=DEFAULT_C_VALUES,
        max_eval_questions=200,
        max_new_tokens=80,
    )

    print(f"\n  Selected {len(h_neurons)} H-Neurons")
    if h_neurons:
        print("  Top 10 neurons:")
        for n in h_neurons[:10]:
            print(f"    layer={n['layer']:2d}, neuron={n['neuron']:4d}, "
                  f"weight={n['weight']:.6f}")

    result = {
        "h_neurons": h_neurons,
        "selection_meta": meta,
    }
    save_stage(output_dir, stage_name, result)
    return result


# -------------------------------------------------------------------
# Stage 10: Layer-specific alpha sweep (V2 two-phase)
# -------------------------------------------------------------------

COARSE_ALPHAS = [0.0, 0.3, 0.7, 1.0]  # 4^3 = 64 combinations


def _evaluate_alpha_combo(
    model, tokenizer, device: str,
    h_neurons: List[Dict], eval_questions: List[Dict],
    layer_alphas: Dict[str, float],
    batch_size: int = 8,
) -> dict:
    """Evaluate a single layer-alpha combination on held-out questions."""
    total = len(eval_questions)
    n_correct = 0

    with ActivationEditor(model, h_neurons, layer_alphas=layer_alphas):
        for batch_start in range(0, total, batch_size):
            batch_items = eval_questions[batch_start:batch_start + batch_size]
            prompts = [
                build_chat_prompt(tokenizer, item["question"])
                for item in batch_items
            ]

            if batch_size > 1:
                predictions = generate_answers_batch(
                    model, tokenizer, prompts, device,
                    max_new_tokens=80, temperature=0.0,
                )
            else:
                predictions = [
                    generate_answer(
                        model, tokenizer, prompts[0], device,
                        max_new_tokens=80, temperature=0.0,
                    )
                ]

            for item, pred in zip(batch_items, predictions):
                if exact_match(pred, item["references"]):
                    n_correct += 1

    accuracy = n_correct / total if total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "hallucination_rate": 1.0 - accuracy,
        "total": total,
        "n_correct": n_correct,
        "layer_alphas": layer_alphas,
    }


def run_stage_10_alpha_sweep(
    model, tokenizer, device: str, output_dir: str,
    h_neurons: List[Dict], questions: List[Dict], balanced_data: dict,
    batch_size: int = 8,
) -> dict:
    """Two-phase layer-specific alpha sweep.

    Phase 1: Coarse sweep (4x4x4 = 64 combos)
    Phase 2: Fine refinement (~25 combos around top-3)
    """
    stage_name = "stage_10_alpha_sweep.json"
    if stage_is_done(output_dir, stage_name):
        print("[Stage 10] Alpha sweep already done, loading.")
        return load_stage(output_dir, stage_name)

    print("\n" + "=" * 60)
    print("[Stage 10] Layer-specific alpha sweep (V2 two-phase)")
    print("=" * 60)

    # Build held-out set
    used_ids = set()
    for item in balanced_data["correct"] + balanced_data["incorrect"]:
        used_ids.add(item["id"])

    held_out = [q for q in questions if q["id"] not in used_ids]
    if not held_out:
        held_out = questions

    eval_questions = held_out[:1000]
    total = len(eval_questions)
    print(f"  Evaluating on {total} held-out questions")

    # --- Phase 1: Coarse sweep (64 combinations) ---
    # Resume support: load partial results if available
    sweep_ckpt_path = os.path.join(output_dir, "sweep_checkpoint.json")
    coarse_results = []
    coarse_done = {}  # (early, mid, late) -> result

    if os.path.exists(sweep_ckpt_path):
        with open(sweep_ckpt_path, "r") as fh:
            sweep_ckpt = json.load(fh)
        for r in sweep_ckpt.get("coarse_results", []):
            la = r["layer_alphas"]
            key = (la["early"], la["middle"], la["late"])
            coarse_done[key] = r
            coarse_results.append(r)
        print(f"  Resumed {len(coarse_results)} coarse results from checkpoint")

    print(f"\n  Phase 1: Coarse sweep ({len(COARSE_ALPHAS)}^3 = "
          f"{len(COARSE_ALPHAS)**3} combinations)")

    combo_idx = 0
    total_combos = len(COARSE_ALPHAS) ** 3

    for early_a in COARSE_ALPHAS:
        for middle_a in COARSE_ALPHAS:
            for late_a in COARSE_ALPHAS:
                combo_idx += 1
                la = {"early": early_a, "middle": middle_a, "late": late_a}
                key = (early_a, middle_a, late_a)

                if key in coarse_done:
                    print(f"\n  [{combo_idx}/{total_combos}] "
                          f"early={early_a:.1f} mid={middle_a:.1f} late={late_a:.1f} [cached]")
                    continue

                print(f"\n  [{combo_idx}/{total_combos}] "
                      f"early={early_a:.1f} mid={middle_a:.1f} late={late_a:.1f}")

                res = _evaluate_alpha_combo(
                    model, tokenizer, device,
                    h_neurons, eval_questions, la,
                    batch_size=batch_size,
                )
                coarse_results.append(res)
                print(f"    accuracy={res['accuracy']:.4f}, "
                      f"hall_rate={res['hallucination_rate']:.4f}")

                # Save checkpoint after each combo
                with open(sweep_ckpt_path, "w") as fh:
                    json.dump({"coarse_results": [
                        {"layer_alphas": r["layer_alphas"],
                         "accuracy": r["accuracy"],
                         "hallucination_rate": r["hallucination_rate"],
                         "total": r["total"],
                         "n_correct": r["n_correct"]}
                        for r in coarse_results
                    ]}, fh)

    # Sort by hallucination rate (lower is better)
    coarse_results.sort(key=lambda r: r["hallucination_rate"])

    # Baseline: all alphas = 1.0
    baseline = next(
        (r for r in coarse_results
         if r["layer_alphas"] == {"early": 1.0, "middle": 1.0, "late": 1.0}),
        coarse_results[-1],
    )
    baseline_acc = baseline["accuracy"]
    baseline_hall = baseline["hallucination_rate"]

    print(f"\n  Phase 1 Results (top 5):")
    print(f"  {'Early':>7s} {'Middle':>7s} {'Late':>7s} "
          f"{'Acc':>8s} {'HallRate':>9s}")
    print(f"  {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*9}")
    for r in coarse_results[:5]:
        la = r["layer_alphas"]
        print(f"  {la['early']:>7.1f} {la['middle']:>7.1f} "
              f"{la['late']:>7.1f} {r['accuracy']:>8.4f} "
              f"{r['hallucination_rate']:>9.4f}")
    print(f"\n  Baseline (1.0/1.0/1.0): acc={baseline_acc:.4f}, "
          f"hall={baseline_hall:.4f}")

    # --- Phase 2: Fine refinement around top-3 ---
    top3 = coarse_results[:3]
    print(f"\n  Phase 2: Fine refinement around top-3 (±0.15 per group)")

    fine_results = list(coarse_results)  # include coarse results
    seen_combos = set()
    for r in coarse_results:
        la = r["layer_alphas"]
        seen_combos.add((la["early"], la["middle"], la["late"]))

    # Resume fine results from checkpoint
    fine_done = {}
    if os.path.exists(sweep_ckpt_path):
        with open(sweep_ckpt_path, "r") as fh:
            sweep_ckpt = json.load(fh)
        for r in sweep_ckpt.get("fine_results", []):
            la = r["layer_alphas"]
            key = (la["early"], la["middle"], la["late"])
            fine_done[key] = r
            if key not in seen_combos:
                fine_results.append(r)
                seen_combos.add(key)
        if fine_done:
            print(f"  Resumed {len(fine_done)} fine results from checkpoint")

    fine_combo_idx = 0
    for top_r in top3:
        la = top_r["layer_alphas"]
        base_early = la["early"]
        base_middle = la["middle"]
        base_late = la["late"]

        # Generate refinements: ±0.15 on each dimension
        deltas = [-0.15, 0.0, 0.15]
        for de in deltas:
            for dm in deltas:
                for dl in deltas:
                    if de == 0.0 and dm == 0.0 and dl == 0.0:
                        continue  # already evaluated
                    new_e = max(0.0, min(3.0, round(base_early + de, 2)))
                    new_m = max(0.0, min(3.0, round(base_middle + dm, 2)))
                    new_l = max(0.0, min(3.0, round(base_late + dl, 2)))

                    combo_key = (new_e, new_m, new_l)
                    if combo_key in seen_combos:
                        continue
                    seen_combos.add(combo_key)

                    fine_combo_idx += 1
                    new_la = {"early": new_e, "middle": new_m, "late": new_l}

                    if combo_key in fine_done:
                        print(f"\n  [Fine {fine_combo_idx}] "
                              f"early={new_e:.2f} mid={new_m:.2f} late={new_l:.2f} [cached]")
                        fine_results.append(fine_done[combo_key])
                        continue

                    print(f"\n  [Fine {fine_combo_idx}] "
                          f"early={new_e:.2f} mid={new_m:.2f} late={new_l:.2f}")

                    res = _evaluate_alpha_combo(
                        model, tokenizer, device,
                        h_neurons, eval_questions, new_la,
                        batch_size=batch_size,
                    )
                    fine_results.append(res)
                    print(f"    accuracy={res['accuracy']:.4f}, "
                          f"hall_rate={res['hallucination_rate']:.4f}")

                    # Save fine checkpoint
                    fine_all = [r for r in fine_results if (r["layer_alphas"]["early"], r["layer_alphas"]["middle"], r["layer_alphas"]["late"]) not in coarse_done]
                    with open(sweep_ckpt_path, "r") as fh:
                        sweep_ckpt = json.load(fh)
                    sweep_ckpt["fine_results"] = [
                        {"layer_alphas": r["layer_alphas"],
                         "accuracy": r["accuracy"],
                         "hallucination_rate": r["hallucination_rate"],
                         "total": r["total"],
                         "n_correct": r["n_correct"]}
                        for r in fine_all
                    ]
                    with open(sweep_ckpt_path, "w") as fh:
                        json.dump(sweep_ckpt, fh)

    print(f"\n  Phase 2 evaluated {fine_combo_idx} additional combinations")

    # --- Select best combination ---
    # Among combos where accuracy doesn't drop >5pp from baseline,
    # pick lowest hallucination rate.
    valid_results = [
        r for r in fine_results
        if r["accuracy"] >= baseline_acc - 0.05
    ]
    if not valid_results:
        valid_results = fine_results

    valid_results.sort(key=lambda r: r["hallucination_rate"])
    best = valid_results[0]
    best_la = best["layer_alphas"]

    print(f"\n  {'='*60}")
    print(f"  Best layer alphas: early={best_la['early']:.2f}, "
          f"middle={best_la['middle']:.2f}, late={best_la['late']:.2f}")
    print(f"  accuracy={best['accuracy']:.4f}, "
          f"hallucination_rate={best['hallucination_rate']:.4f}")
    print(f"  {'='*60}")

    # Compute equivalent global alpha (average of layer alphas)
    best_global = round(
        (best_la["early"] + best_la["middle"] + best_la["late"]) / 3.0, 4
    )

    result = {
        "coarse_results": [
            {
                "layer_alphas": r["layer_alphas"],
                "accuracy": r["accuracy"],
                "hallucination_rate": r["hallucination_rate"],
            }
            for r in coarse_results[:10]  # save top 10
        ],
        "fine_combo_count": fine_combo_idx,
        "best_layer_alphas": best_la,
        "best_global_alpha": best_global,
        "best_accuracy": best["accuracy"],
        "best_hallucination_rate": best["hallucination_rate"],
        "baseline_accuracy": baseline_acc,
        "baseline_hallucination_rate": baseline_hall,
    }
    save_stage(output_dir, stage_name, result)
    return result


# -------------------------------------------------------------------
# Stage 11: Save h_neurons.json with layer-specific alphas
# -------------------------------------------------------------------

def run_stage_11_save_final(
    output_dir: str, model_path: str,
    h_neurons: List[Dict], selection_meta: dict,
    alpha_sweep_data: dict,
) -> str:
    """Save final h_neurons.json with layer-specific alphas."""
    output_path = os.path.join(output_dir, "h_neurons.json")

    if os.path.exists(output_path):
        print("[Stage 11] h_neurons.json already exists, skipping.")
        return output_path

    best_layer_alphas = alpha_sweep_data["best_layer_alphas"]
    best_global = alpha_sweep_data["best_global_alpha"]

    final_output = {
        "neuron_indices": h_neurons,
        "alpha": best_global,
        "layer_alphas": best_layer_alphas,
        "layer_ranges": {
            group: list(rng) for group, rng in LAYER_RANGES.items()
        },
        "model_path": model_path,
        "num_neurons": len(h_neurons),
        "selection_meta": {
            "C": selection_meta.get("C"),
            "cv_accuracy": selection_meta.get("cv_accuracy"),
            "best_accuracy": alpha_sweep_data.get("best_accuracy"),
            "best_hallucination_rate": alpha_sweep_data.get("best_hallucination_rate"),
            "baseline_accuracy": alpha_sweep_data.get("baseline_accuracy"),
            "baseline_hallucination_rate": alpha_sweep_data.get("baseline_hallucination_rate"),
        },
    }

    save_json(output_path, final_output)
    print(f"\n  Final h_neurons.json saved: {output_path}")
    print(f"  Neurons: {len(h_neurons)}")
    print(f"  Layer alphas: early={best_layer_alphas['early']:.2f}, "
          f"middle={best_layer_alphas['middle']:.2f}, "
          f"late={best_layer_alphas['late']:.2f}")
    print(f"  Global equivalent alpha: {best_global}")

    return output_path


# ===================================================================
# Main pipeline
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="H-Neurons discovery and evaluation pipeline for "
                    "Llama-3.2-3B-Instruct."
    )
    parser.add_argument(
        "--model-path", required=True,
        help="HuggingFace model ID or local path to model",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to save pipeline outputs and checkpoints",
    )
    parser.add_argument(
        "--batch-size", type=int, default=20,
        help="Batch size for sampling (default: 20)",
    )
    parser.add_argument(
        "--eval-batch-size", type=int, default=8,
        help="Batch size for evaluation (default: 8)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=10,
        help="Number of responses to generate per question (default: 10)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print("=" * 70)
    print("INVISIBLE LOGIC V2 -- H-NEURONS PIPELINE")
    print("  (Layer-specific alpha, two-phase sweep)")
    print(f"  Model:      {args.model_path}")
    print(f"  Output:     {output_dir}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Seed:       {args.seed}")
    print("=" * 70)

    t0 = time.time()

    # Load model
    model, tokenizer, device = load_model_and_tokenizer(args.model_path)

    # Stage 1: Architecture probe
    arch = run_stage_1_arch_probe(model, output_dir)

    # Stage 2: Load questions
    questions_data = run_stage_2_load_questions(output_dir)
    questions = questions_data["questions"]

    # Stage 3: Finance QA validation
    finance_val = run_stage_3_finance_validation(
        model, tokenizer, device, output_dir, questions,
    )

    # Stage 4: Sampling
    sampling_data = run_stage_4_sampling(
        model, tokenizer, device, output_dir, questions,
        num_samples=args.num_samples, batch_size=args.batch_size,
    )

    # Stage 5: Consistency filter
    consistency_data = run_stage_5_consistency(
        output_dir, questions, sampling_data["samples"],
    )

    # Stage 6: Class balancing
    balanced_data = run_stage_6_balancing(output_dir, consistency_data)

    # Stage 7: Token F1 span extraction
    span_data = run_stage_7_spans(output_dir, tokenizer, balanced_data)

    # Stage 8: CETT feature collection
    cett_data = run_stage_8_cett_features(
        model, tokenizer, device, output_dir, span_data, balanced_data,
    )

    # Stage 9: Neuron selection
    # Pre-extract CETT data and free the large tensor dict to save ~3.3GB RAM
    import gc
    selection_data = run_stage_9_neuron_selection(
        model, tokenizer, device, output_dir,
        cett_data, questions, balanced_data,
    )
    del cett_data
    gc.collect()

    # Stage 10: Alpha sweep
    alpha_sweep_data = run_stage_10_alpha_sweep(
        model, tokenizer, device, output_dir,
        selection_data["h_neurons"], questions, balanced_data,
        batch_size=args.eval_batch_size,
    )

    # Stage 11: Save final h_neurons.json
    final_path = run_stage_11_save_final(
        output_dir, args.model_path,
        selection_data["h_neurons"], selection_data["selection_meta"],
        alpha_sweep_data,
    )

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"H-NEURONS PIPELINE COMPLETE")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  Output: {final_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
