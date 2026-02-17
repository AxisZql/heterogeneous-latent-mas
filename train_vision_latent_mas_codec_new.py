   

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor, AutoTokenizer

from models import ModelWrapper, _past_length

                                                                                                 
from methods.vision_latent_mas_codec_new import (
    LatentToUniversalEncoder,
    UniversalToVisionDecoder,
)


                               
               
                               


def _parse_model_list(raw: str) -> List[str]:
    return [s.strip() for s in (raw or "").split(",") if s.strip()]


def _parse_int_csv(raw: str) -> List[int]:
    vals: List[int] = []
    for part in str(raw or "").split(","):
        p = part.strip()
        if not p:
            continue
        vals.append(int(p))
    return vals


def _expand_override_list(vals: List[int], n: int, arg_name: str) -> Optional[List[int]]:
    if not vals:
        return None
    if len(vals) == 1 and n > 1:
        return vals * n
    if len(vals) != n:
        raise ValueError(f"{arg_name} expects either 1 value or {n} values; got {len(vals)}")
    return vals


def _load_json_object(raw_or_path: str, *, arg_name: str) -> Dict[str, Any]:
    payload = str(raw_or_path or "").strip()
    if not payload:
        return {}
    if os.path.exists(payload):
        with open(payload, "r", encoding="utf-8") as f:
            payload = f.read()
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError(f"{arg_name} must decode to a JSON object")
    return data


@dataclass
class _DummyImageSpec:
    count: int
    size: int


def _resolve_dummy_image_specs(model_names: Sequence[str], args: argparse.Namespace) -> Dict[str, _DummyImageSpec]:
    n = len(model_names)
    default_count = max(1, _safe_int(getattr(args, "vision_codec_dummy_image_count", 1), 1))
    default_size = max(32, _safe_int(getattr(args, "vision_codec_dummy_image_size", 224), 224))

    counts = _expand_override_list(
        _parse_int_csv(getattr(args, "vision_codec_dummy_image_counts", "")),
        n,
        "--vision_codec_dummy_image_counts",
    )
    sizes = _expand_override_list(
        _parse_int_csv(getattr(args, "vision_codec_dummy_image_sizes", "")),
        n,
        "--vision_codec_dummy_image_sizes",
    )

    specs: Dict[str, _DummyImageSpec] = {}
    for i, name in enumerate(model_names):
        c = default_count if counts is None else max(1, int(counts[i]))
        s = default_size if sizes is None else max(32, int(sizes[i]))
        specs[name] = _DummyImageSpec(count=c, size=s)

    raw_map = getattr(args, "vision_codec_dummy_image_spec_json", "")
    if raw_map:
        data = _load_json_object(raw_map, arg_name="--vision_codec_dummy_image_spec_json")
        for name, cfg in data.items():
            if name not in specs:
                continue
            cur = specs[name]
            if isinstance(cfg, dict):
                c = max(1, _safe_int(cfg.get("count", cur.count), cur.count))
                s = max(32, _safe_int(cfg.get("size", cur.size), cur.size))
            elif isinstance(cfg, (list, tuple)) and len(cfg) >= 2:
                c = max(1, _safe_int(cfg[0], cur.count))
                s = max(32, _safe_int(cfg[1], cur.size))
            elif isinstance(cfg, int):
                c = max(1, int(cfg))
                s = cur.size
            else:
                raise ValueError(
                    "--vision_codec_dummy_image_spec_json entries must be {count,size}, [count,size], or int count"
                )
            specs[name] = _DummyImageSpec(count=c, size=s)

    return specs


def _safe_int(x: Any, default: int) -> int:
    try:
        return default if x is None else int(x)
    except Exception:
        return default


def _safe_float(x: Any, default: float) -> float:
    try:
        return default if x is None else float(x)
    except Exception:
        return default


def _sanitize_logits(logits: torch.Tensor, clip: float) -> torch.Tensor:
                                                                                  
    x = torch.nan_to_num(logits.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    if clip > 0:
        x = x.clamp(min=-clip, max=clip)
    return x


def _normalize_kl_mode(mode: str) -> str:
    m = str(mode or "full").strip().lower()
    aliases = {
        "kl_div": "full",
        "kl": "k1",
        "k1": "k1",
        "abs": "abs",
        "mse": "k2",
        "k2": "k2",
        "low_var_kl": "k3",
        "k3": "k3",
        "k3+": "k3+",
        "low_var_kl+": "k3+",
        "full": "full",
    }
    if m not in aliases:
        raise ValueError(f"Unsupported KL mode: {mode!r}")
    return aliases[m]


def _kl_penalty_forward(logprob: torch.Tensor, ref_logprob: torch.Tensor, mode: str) -> torch.Tensor:
                                                               
    if mode == "k1":
        return logprob - ref_logprob

    if mode == "abs":
        return (logprob - ref_logprob).abs()

    if mode == "k2":
        return 0.5 * (logprob - ref_logprob).square()

                                                                              
    if mode in ("k3", "k3+"):
        kl = torch.clamp(ref_logprob - logprob, min=-20.0, max=20.0)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1.0).contiguous()
        return torch.clamp(kld, min=-10.0, max=10.0)

    raise ValueError(f"Unsupported KL mode for forward penalty: {mode!r}")


def _kl_penalty(logprob: torch.Tensor, ref_logprob: torch.Tensor, mode: str) -> torch.Tensor:
       
    forward_score = _kl_penalty_forward(logprob, ref_logprob, mode)
    if not mode.endswith("+") or mode in ("k2",):
        return forward_score

    backward_score = 0.5 * (logprob - ref_logprob).square()
    return backward_score - backward_score.detach() + forward_score.detach()


def _compute_kl_loss(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temp: float,
    mode: str,
    logit_clip: float,
    topk: int,
) -> torch.Tensor:
       
    t_logits = _sanitize_logits(teacher_logits, clip=logit_clip)
    s_logits = _sanitize_logits(student_logits, clip=logit_clip)

    if topk > 0 and topk < int(t_logits.shape[-1]):
        idx = torch.topk(t_logits, k=int(topk), dim=-1).indices
        t_logits = torch.gather(t_logits, dim=-1, index=idx)
        s_logits = torch.gather(s_logits, dim=-1, index=idx)

    T = max(1e-6, float(temp))
    log_tgt = F.log_softmax(t_logits / T, dim=-1)
    logp = F.log_softmax(s_logits / T, dim=-1)

    mode_n = _normalize_kl_mode(mode)
    if mode_n == "full":
        loss = F.kl_div(logp, log_tgt, reduction="batchmean", log_target=True)
        return loss * (T * T)

    penalty = _kl_penalty(logp, log_tgt, mode_n)
    return penalty.mean() * (T * T)


def _maybe_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return out


def _module_accepts_kwarg(module: nn.Module, kw: str) -> bool:
    try:
        sig = inspect.signature(module.forward)
    except Exception:
        return False
    params = sig.parameters
    if kw in params:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def _forward_with_supported_kwargs(module: nn.Module, kwargs: Dict[str, Any]):
    try:
        sig = inspect.signature(module.forward)
    except Exception:
        return module(**kwargs)
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return module(**kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in params}
    return module(**filtered)


def _get_causal_lm_head(module: nn.Module) -> Optional[nn.Module]:
    head = None
    if hasattr(module, "get_output_embeddings"):
        try:
            head = module.get_output_embeddings()
        except Exception:
            head = None
    if head is None:
        head = getattr(module, "lm_head", None)
    return head if isinstance(head, nn.Module) else None


def _get_causal_backbone(module: nn.Module) -> Optional[nn.Module]:
    for attr in ("model", "language_model", "llm", "transformer"):
        sub = getattr(module, attr, None)
        if isinstance(sub, nn.Module):
            return sub
    return None


def _apply_logit_softcap_if_configured(logits: torch.Tensor, module: nn.Module) -> torch.Tensor:
    cfg = getattr(module, "config", None)
    if cfg is None:
        return logits
    softcap = getattr(cfg, "final_logit_softcapping", None)
    if softcap is None:
        softcap = getattr(cfg, "logit_softcapping", None)
    try:
        s = float(softcap)
    except Exception:
        return logits
    if s <= 0:
        return logits
    return torch.tanh(logits / s) * s


def _student_forward_last_token_only(
    *,
    causal_module: nn.Module,
    forward_kwargs: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
       
    backbone = _get_causal_backbone(causal_module)
    lm_head = _get_causal_lm_head(causal_module)

    if backbone is None or lm_head is None:
                                                               
        fwd = dict(forward_kwargs)
        fwd["use_cache"] = False
        fwd["output_hidden_states"] = True
        fwd["return_dict"] = True
        if _module_accepts_kwarg(causal_module, "logits_to_keep"):
            fwd["logits_to_keep"] = 1
        out = _forward_with_supported_kwargs(causal_module, fwd)
        hs = _get_hidden_states_tuple(out)
        if not hs:
            raise RuntimeError("Student fallback forward did not return hidden_states")
        student_h = hs[-1][:, -1, :]
        logits = out.logits
        if logits is None:
            raise RuntimeError("Student fallback forward did not return logits")
        if logits.ndim == 3:
            logits = logits[:, -1, :]
        return student_h, logits

    bkw = dict(forward_kwargs)
    bkw["use_cache"] = False
    bkw["output_hidden_states"] = False
    bkw["return_dict"] = True
    out_b = _forward_with_supported_kwargs(backbone, bkw)

    hidden = getattr(out_b, "last_hidden_state", None)
    if hidden is None and isinstance(out_b, (tuple, list)) and out_b:
        v0 = out_b[0]
        if torch.is_tensor(v0):
            hidden = v0
    if hidden is None:
        hs = _get_hidden_states_tuple(out_b)
        if hs:
            hidden = hs[-1]
    if hidden is None:
        raise RuntimeError("Backbone forward did not return last hidden state")

    student_h = hidden[:, -1, :]
    student_logits = lm_head(student_h)
    student_logits = _apply_logit_softcap_if_configured(student_logits, causal_module)
    if student_logits.ndim == 3:
        student_logits = student_logits[:, -1, :]
    return student_h, student_logits


def _make_dummy_image(size: int) -> Image.Image:
    return Image.new("RGB", (size, size), color=(255, 255, 255))


def _infer_hidden_size(wrapper: ModelWrapper) -> int:
    cfg = getattr(wrapper.model, "config", None)
    h = getattr(cfg, "hidden_size", None) if cfg is not None else None
    if h is None:
        h = int(wrapper.model.get_input_embeddings().weight.shape[1])
        if cfg is not None:
            try:
                setattr(cfg, "hidden_size", int(h))
            except Exception:
                pass
    return int(h)


def _get_hidden_states_tuple(out: Any) -> Optional[Tuple[torch.Tensor, ...]]:
    hs = getattr(out, "hidden_states", None)
    if hs is None:
        hs = getattr(out, "decoder_hidden_states", None)
    return hs


def _ridge_fit(X: torch.Tensor, Y: torch.Tensor, ridge: float = 1e-3) -> Tuple[torch.Tensor, torch.Tensor]:
                                                                               
    Xc = X.detach().float().cpu()
    Yc = Y.detach().float().cpu()
    if Xc.ndim != 2 or Yc.ndim != 2:
        raise ValueError("X and Y must be 2D")
    if Xc.shape != Yc.shape:
        raise ValueError("X and Y must have same shape")
    D = Xc.shape[1]
    Xm = Xc.mean(0, keepdim=True)
    Ym = Yc.mean(0, keepdim=True)
    X0 = Xc - Xm
    Y0 = Yc - Ym
    XtX = X0.T @ X0
    XtX = XtX + float(ridge) * torch.eye(D, dtype=XtX.dtype)
    W = torch.linalg.solve(XtX, X0.T @ Y0)
    b = (Ym - Xm @ W).squeeze(0)
    return W.float(), b.float()


def _ridge_eval_mse(
    X: torch.Tensor,
    Y: torch.Tensor,
    W: torch.Tensor,
    b: torch.Tensor,
    *,
    max_rows: int = 16384,
) -> float:
       
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D for ridge eval")
    if X.shape != Y.shape:
        raise ValueError("X and Y must have the same shape for ridge eval")

    n = int(X.shape[0])
    if n == 0:
        return 0.0

    k = max(1, min(int(max_rows), n))
    if k < n:
        idx = torch.linspace(0, n - 1, steps=k, dtype=torch.long)
        Xs = X.index_select(0, idx)
        Ys = Y.index_select(0, idx)
    else:
        Xs = X
        Ys = Y

    pred = Xs.float().cpu() @ W.float().cpu() + b.float().cpu().unsqueeze(0)
    mse = F.mse_loss(pred, Ys.float().cpu(), reduction="mean")
    return float(mse.detach().cpu())


def _codec_config_from_args(cfg: argparse.Namespace) -> Dict[str, Any]:
    return {
        "codec_dim": int(cfg.vision_codec_dim),
        "codec_tokens": int(cfg.vision_codec_tokens),
        "codec_img_tokens": int(cfg.vision_codec_img_tokens),
        "codec_heads": int(cfg.vision_codec_heads),
        "codec_layers": int(cfg.vision_codec_layers),
        "codec_dropout": float(cfg.vision_codec_dropout),
        "codec_gate_init_bias": float(cfg.vision_codec_gate_init_bias),
    }


def _identity_alignment(
    *,
    model_names: Sequence[str],
    codec_dim: int,
    ref_idx: int = 0,
    ref_name: Optional[str] = None,
) -> Dict[str, Any]:
    names = list(model_names)
    if not names:
        return {"ref_idx": 0, "ref_model_name": "", "out": {}, "in": {}}

    ridx = max(0, min(int(ref_idx), len(names) - 1))
    rname = ref_name if ref_name in names else names[ridx]
    ridx = names.index(rname)

    I = torch.eye(int(codec_dim), dtype=torch.float32)
    z = torch.zeros(int(codec_dim), dtype=torch.float32)
    out = {n: {"W": I.clone(), "b": z.clone()} for n in names}
    in_map = {n: {"W": I.clone(), "b": z.clone()} for n in names}
    return {"ref_idx": int(ridx), "ref_model_name": rname, "out": out, "in": in_map}


def _build_codec_checkpoint(
    *,
    model_names: Sequence[str],
    ref_model_name: str,
    cfg: argparse.Namespace,
    encoders: Dict[str, Dict[str, torch.Tensor]],
    decoders: Dict[str, Dict[str, torch.Tensor]],
    align: Dict[str, Any],
    is_partial: bool,
) -> Dict[str, Any]:
    names = [n for n in model_names if n in encoders and n in decoders]
    if names and ref_model_name not in names:
        ref_model_name = names[0]

    return {
        "version": 2,
        "models": names,
        "ref_model_name": ref_model_name,
        "config": _codec_config_from_args(cfg),
        "encoders": {n: encoders[n] for n in names},
        "decoders": {n: decoders[n] for n in names},
        "align": align,
        "is_partial": bool(is_partial),
    }


def _atomic_torch_save(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _derive_partial_ckpt_path(path: str) -> str:
    root, ext = os.path.splitext(path)
    if ext:
        return f"{root}.partial{ext}"
    return f"{path}.partial.pt"


def _is_internvl_wrapper(wrapper: ModelWrapper) -> bool:
    if bool(getattr(wrapper, "is_internvl", False)):
        return True
    model_type = str(getattr(getattr(wrapper.model, "config", None), "model_type", "")).lower()
    cls_name = wrapper.model.__class__.__name__.lower()
    return ("internvl" in model_type) or ("internvl" in cls_name)


def _is_minicpm_wrapper(wrapper: ModelWrapper) -> bool:
    if bool(getattr(wrapper, "is_minicpm_v", False)):
        return True
    model_type = str(getattr(getattr(wrapper.model, "config", None), "model_type", "")).lower()
    cls_name = wrapper.model.__class__.__name__.lower()
    return ("minicpmv" in model_type) or ("minicpmv" in cls_name) or (
        "minicpm" in model_type and "vl" in model_type
    )


def _get_text_backbone(wrapper: ModelWrapper):
    if _is_internvl_wrapper(wrapper) and hasattr(wrapper.model, "language_model"):
        return wrapper.model.language_model
    if _is_minicpm_wrapper(wrapper) and hasattr(wrapper.model, "llm"):
        return wrapper.model.llm
    return wrapper.model


def _patch_minicpm_batchfeature_if_needed(processor: Any) -> None:
    mods = set()
    try:
        mods.add(inspect.getmodule(processor.__class__))
    except Exception:
        pass
    try:
        mods.add(inspect.getmodule(getattr(processor, "image_processor", None).__class__))
    except Exception:
        pass
    for mod in mods:
        if mod is None:
            continue
        bf = getattr(mod, "MiniCPMVBatchFeature", None)
        if bf is None:
            continue
        try:
            sig = inspect.signature(bf.convert_to_tensors)
        except Exception:
            continue
        if "skip_tensor_conversion" in sig.parameters:
            continue
        orig = bf.convert_to_tensors

        def _make_patch(orig_fn):
            def _patched(self, tensor_type=None, skip_tensor_conversion=False, **kwargs):
                if skip_tensor_conversion:
                    return self
                return orig_fn(self, tensor_type=tensor_type)

            return _patched

        bf.convert_to_tensors = _make_patch(orig)


def _minicpm_pack_prompt(prompt: str) -> str:
    placeholder = "(<image>./</image>)"
    if placeholder in prompt:
        return prompt
    if "<image>" in prompt:
        return prompt.replace("<image>", placeholder)
    return placeholder + "\n" + prompt


def _build_position_ids_from_attention(attention_mask: torch.Tensor) -> torch.Tensor:
    pos = attention_mask.long().cumsum(dim=-1) - 1
    pos = pos.clamp_min(0)
    pos = pos.masked_fill(attention_mask == 0, 0)
    return pos.to(dtype=torch.long)


def _fallback_image_bounds_from_unk(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    unk_id: int,
    query_num: int,
) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    q = max(1, int(query_num))
    for row_ids, row_mask in zip(input_ids.cpu(), attention_mask.cpu()):
        vals = row_ids.tolist()
        msk = row_mask.tolist()
        runs: List[Tuple[int, int]] = []
        i = 0
        n = len(vals)
        while i < n:
            if msk[i] and vals[i] == int(unk_id):
                j = i + 1
                while j < n and msk[j] and vals[j] == int(unk_id):
                    j += 1
                runs.append((i, j))
                i = j
            else:
                i += 1
        bounds: List[List[int]] = []
        for s, e in runs:
            run_len = e - s
            if run_len >= q and run_len % q == 0:
                for k in range(0, run_len, q):
                    bounds.append([s + k, s + k + q])
            elif run_len >= 8:
                bounds.append([s, e])
        if bounds:
            out.append(torch.tensor(bounds, dtype=torch.long))
        else:
            out.append(torch.zeros((0, 2), dtype=torch.long))
    return out


def _minicpm_prepare_multimodal_batch(
    *,
    wrapper: ModelWrapper,
    processor: Any,
    prompts: List[str],
    images: List[Image.Image],
    device: torch.device,
) -> Dict[str, Any]:
    _patch_minicpm_batchfeature_if_needed(processor)
    packed = [_minicpm_pack_prompt(p) for p in prompts]
    img_lists = [[img.convert("RGB")] for img in images]
    enc = processor(text=packed, images=img_lists, return_tensors="pt", padding=True)
    if "input_ids" not in enc or "attention_mask" not in enc:
        raise RuntimeError("MiniCPM processor did not return input_ids/attention_mask")

    enc["input_ids"] = enc["input_ids"].to(device)
    enc["attention_mask"] = enc["attention_mask"].to(device)

    image_bound = enc.get("image_bound", None)
    missing = (
        image_bound is None
        or not isinstance(image_bound, list)
        or len(image_bound) != int(enc["input_ids"].shape[0])
        or all((not isinstance(b, torch.Tensor)) or b.numel() == 0 for b in image_bound)
    )
    if missing:
        tok = getattr(processor, "tokenizer", None)
        unk_id = int(getattr(tok, "unk_token_id", 0)) if tok is not None else 0
        q = int(getattr(getattr(wrapper.model, "config", None), "query_num", 64))
        enc["image_bound"] = _fallback_image_bounds_from_unk(
            enc["input_ids"], enc["attention_mask"], unk_id=unk_id, query_num=q
        )
    return enc


def _minicpm_positions_from_bounds(bounds: Any) -> List[int]:
    if not isinstance(bounds, torch.Tensor) or bounds.ndim != 2 or bounds.shape[-1] != 2:
        return []
    out: List[int] = []
    for r in bounds.tolist():
        s = int(r[0])
        e = int(r[1])
        if e > s:
            out.extend(list(range(s, e)))
    return out


@torch.no_grad()
def _minicpm_build_inputs_embeds(wrapper: ModelWrapper, mm: Dict[str, Any]) -> torch.Tensor:
    if not hasattr(wrapper.model, "get_vllm_embedding"):
        raise RuntimeError("MiniCPM wrapper.model is missing get_vllm_embedding")
    data = {
        "input_ids": mm["input_ids"],
        "pixel_values": mm.get("pixel_values"),
        "image_bound": mm.get("image_bound"),
        "tgt_sizes": mm.get("tgt_sizes"),
        "position_ids": _build_position_ids_from_attention(mm["attention_mask"]),
    }
    embeds, _ = wrapper.model.get_vllm_embedding(data)
    return embeds


def _get_tokenizer_like(processor) -> Optional[Any]:
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        return tok
    if hasattr(processor, "convert_tokens_to_ids") and hasattr(processor, "__call__"):
        return processor
    return None


def _internvl_num_image_tokens(wrapper: ModelWrapper) -> int:
    n = getattr(wrapper.model, "num_image_token", None)
    if n is not None:
        return int(n)
    cfg = getattr(wrapper.model, "config", None)
    image_size = getattr(cfg, "force_image_size", None)
    if image_size is None:
        image_size = getattr(getattr(cfg, "vision_config", None), "image_size", 224)
    patch = getattr(getattr(cfg, "vision_config", None), "patch_size", 14)
    downsample = float(getattr(cfg, "downsample_ratio", 0.5))
    return int((int(image_size) // int(patch)) ** 2 * (downsample ** 2))


def _internvl_image_size(wrapper: ModelWrapper) -> int:
    cfg = getattr(wrapper.model, "config", None)
    image_size = getattr(cfg, "force_image_size", None)
    if image_size is None:
        image_size = getattr(getattr(cfg, "vision_config", None), "image_size", 224)
    return int(image_size)


def _internvl_preprocess_images(
    wrapper: ModelWrapper,
    images: List[Image.Image],
    device: torch.device,
) -> torch.Tensor:
    size = _internvl_image_size(wrapper)
    arrs: List[torch.Tensor] = []
    for img in images:
        x = img.convert("RGB").resize((size, size), resample=Image.BICUBIC)
        x = torch.from_numpy(np.array(x, copy=True)).permute(2, 0, 1).float() / 255.0
        arrs.append(x)
    batch = torch.stack(arrs, dim=0)                      
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    batch = (batch - mean) / std
    vision_dtype = wrapper.model.vision_model.embeddings.patch_embedding.weight.dtype
    return batch.to(device=device, dtype=vision_dtype)


def _internvl_pack_prompt(prompt: str, wrapper: ModelWrapper) -> str:
    image_tokens = "<img>" + ("<IMG_CONTEXT>" * _internvl_num_image_tokens(wrapper)) + "</img>"
    if "<image>" in prompt:
        return prompt.replace("<image>", image_tokens, 1)
    return image_tokens + "\n" + prompt


def _internvl_prepare_multimodal_batch(
    *,
    wrapper: ModelWrapper,
    tokenizer,
    prompts: List[str],
    images: List[Image.Image],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    img_ctx = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    if not isinstance(img_ctx, int) or img_ctx < 0:
        raise RuntimeError("InternVL tokenizer is missing <IMG_CONTEXT> token id")
    packed = [_internvl_pack_prompt(p, wrapper) for p in prompts]
    text = tokenizer(packed, return_tensors="pt", padding=True)
    pixel_values = _internvl_preprocess_images(wrapper, images, device=device)
    image_flags = torch.ones((len(images), 1), dtype=torch.long, device=device)
    wrapper.model.img_context_token_id = int(img_ctx)
    return {
        "input_ids": text["input_ids"].to(device),
        "attention_mask": text["attention_mask"].to(device),
        "pixel_values": pixel_values,
        "image_flags": image_flags,
        "img_context_id": torch.tensor(int(img_ctx), device=device, dtype=torch.long),
    }


                               
                                            
                               


@dataclass
class _SpecialTokenIds:
    image_token_id: Optional[int] = None
    image_pad_id: Optional[int] = None
    image_context_id: Optional[int] = None


def _infer_special_token_ids(processor) -> _SpecialTokenIds:
    ids = _SpecialTokenIds()
    tok = _get_tokenizer_like(processor)
    if tok is None:
        return ids

    for attr in ("image_token_id", "image_pad_token_id", "image_pad_id"):
        if hasattr(tok, attr):
            val = getattr(tok, attr)
            if val is None:
                continue
            if "pad" in attr and ids.image_pad_id is None:
                ids.image_pad_id = int(val)
            if attr == "image_token_id" and ids.image_token_id is None:
                ids.image_token_id = int(val)

                                          
    special_strings: List[str] = []
    for attr in ("additional_special_tokens", "all_special_tokens"):
        seq = getattr(tok, attr, None)
        if isinstance(seq, (list, tuple)):
            special_strings.extend([str(s) for s in seq])
    special_strings.extend(
        [
            "<image>",
            "<img>",
            "<image_pad>",
            "<imgpad>",
            "<img_pad>",
            "<im_start>",
            "<im_end>",
        ]
    )

    def _try_id(token_str: str) -> Optional[int]:
        try:
            tid = tok.convert_tokens_to_ids(token_str)
            if isinstance(tid, int) and tid >= 0:
                return int(tid)
        except Exception:
            return None
        return None

    if ids.image_pad_id is None:
        for s in special_strings:
            sl = s.lower()
            if "imgpad" in sl or ("image" in sl and "pad" in sl):
                tid = _try_id(s)
                if tid is not None:
                    ids.image_pad_id = tid
                    break

    if ids.image_token_id is None:
        for s in special_strings:
            sl = s.lower()
            if sl in {"<image>", "<img>"} or (
                ("image" in sl or "img" in sl) and "pad" not in sl
            ):
                tid = _try_id(s)
                if tid is not None:
                    ids.image_token_id = tid
                    break
    if ids.image_context_id is None:
        tid = _try_id("<IMG_CONTEXT>")
        if tid is not None:
            ids.image_context_id = tid

    return ids


def _find_image_positions(
    input_ids: torch.Tensor,
    ids: _SpecialTokenIds,
    *,
    tokenizer=None,
    batch_index: int = 0,
) -> List[int]:
                                                                
    if input_ids.ndim == 2:
        row = input_ids[batch_index]
    else:
        row = input_ids
    toks = row.tolist()

    if ids.image_pad_id is not None and ids.image_pad_id in toks:
        return [i for i, t in enumerate(toks) if t == ids.image_pad_id]
    if ids.image_context_id is not None and ids.image_context_id in toks:
        return [i for i, t in enumerate(toks) if t == ids.image_context_id]
    if ids.image_token_id is not None and ids.image_token_id in toks:
        return [i for i, t in enumerate(toks) if t == ids.image_token_id]

                           
    if tokenizer is not None and hasattr(tokenizer, "convert_ids_to_tokens"):
        try:
            ss = tokenizer.convert_ids_to_tokens(toks)
            cand = [
                i
                for i, s in enumerate(ss)
                if isinstance(s, str)
                and (
                    "imgpad" in s.lower() or ("image" in s.lower() and "pad" in s.lower())
                )
            ]
            if cand:
                return cand
        except Exception:
            pass

                          
    best: List[int] = []
    cur: List[int] = []
    for i, t in enumerate(toks):
        if not cur or t == toks[cur[-1]]:
            cur.append(i)
        else:
            if len(cur) >= 8 and len(cur) > len(best):
                best = cur
            cur = [i]
    if len(cur) >= 8 and len(cur) > len(best):
        best = cur
    return best


def _build_mm_user_content(*, num_images: int, text: str) -> List[Dict[str, str]]:
    n = max(1, int(num_images))
    content: List[Dict[str, str]] = [{"type": "image"} for _ in range(n)]
    content.append({"type": "text", "text": str(text)})
    return content


def _processor_encode_multimodal(
    processor: Any,
    *,
    texts: List[str],
    dummy_imgs: List[Image.Image],
) -> Dict[str, Any]:
    if not texts:
        raise ValueError("texts must be non-empty")
    if not dummy_imgs:
        raise ValueError("dummy_imgs must be non-empty")

    B = len(texts)
    candidates: List[Tuple[Any, Any]] = []
    if B == 1:
        t1 = [texts[0]]
        candidates.append((t1, list(dummy_imgs)))
        candidates.append((t1, [list(dummy_imgs)]))
        candidates.append((texts[0], list(dummy_imgs)))
        if len(dummy_imgs) == 1:
            candidates.append((t1, [dummy_imgs[0]]))
    else:
        nested = [[img for img in dummy_imgs] for _ in range(B)]
        candidates.append((texts, nested))
        if len(dummy_imgs) == 1:
            candidates.append((texts, [dummy_imgs[0]] * B))
        flat = [img for _ in range(B) for img in dummy_imgs]
        candidates.append((texts, flat))

    last_exc: Optional[Exception] = None
    for text_arg, image_arg in candidates:
        try:
            return processor(text=text_arg, images=image_arg, return_tensors="pt", padding=True)
        except Exception as e:                 
            last_exc = e

    raise RuntimeError(
        f"Processor could not encode multimodal batch with {len(dummy_imgs)} image(s) per sample. "
        f"Last error: {repr(last_exc)}"
    )


def _resample_tokens(x: torch.Tensor, target_len: int) -> torch.Tensor:
                                                                      
    if target_len <= 0:
        raise ValueError("target_len must be > 0")
    orig_dtype = x.dtype
    xf = x.float()
    if xf.ndim == 2:
        y = F.interpolate(xf.unsqueeze(0).transpose(1, 2), size=target_len, mode="linear", align_corners=False)
        return y.transpose(1, 2)[0].to(dtype=orig_dtype)
    if xf.ndim == 3:
        y = F.interpolate(xf.transpose(1, 2), size=target_len, mode="linear", align_corners=False)
        return y.transpose(1, 2).to(dtype=orig_dtype)
    raise ValueError(f"bad shape: {tuple(x.shape)}")


                               
                                           
                               


def _top_p_filtering(probs: torch.Tensor, top_p: float) -> torch.Tensor:
                                                                          
    if top_p >= 1.0:
        return probs
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cdf = torch.cumsum(sorted_probs, dim=-1)
                           
    mask = cdf > top_p
    mask[..., 0] = False
    sorted_probs = sorted_probs.masked_fill(mask, 0.0)
            
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                  
    out = torch.zeros_like(probs)
    out.scatter_(-1, sorted_idx, sorted_probs)
    return out


@torch.no_grad()
def _sample_token_rollouts(
    wrapper: ModelWrapper,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    latent_steps: int,
    num_rollouts: int,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
                                                                                       
    device = input_ids.device
    B = int(input_ids.shape[0])
    H = _infer_hidden_size(wrapper)
    if latent_steps <= 0 or num_rollouts <= 0:
        return torch.empty((B, 0, 0, H), device=device, dtype=torch.float32)

    text_model = _get_text_backbone(wrapper)
    is_minicpm = _is_minicpm_wrapper(wrapper)

                                                        
    fwd0 = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": True,
        "output_hidden_states": True,
        "return_dict": True,
    }
    if is_minicpm:
        fwd0["position_ids"] = _build_position_ids_from_attention(attention_mask)
    out0 = text_model(**fwd0)
    past0 = out0.past_key_values
    logits0 = out0.logits[:, -1, :]
    hs0 = _get_hidden_states_tuple(out0)
    if not hs0:
        raise RuntimeError("Model did not return hidden_states")
    last_hidden0 = hs0[-1][:, -1, :]

                                                                                                                  
    latents_all: List[torch.Tensor] = []

    for _r in range(int(num_rollouts)):
        past = past0
        logits = logits0
        last_hidden = last_hidden0
        steps: List[torch.Tensor] = []

        for _t in range(int(latent_steps)):
                          
            temp = max(1e-6, float(temperature))
            probs = F.softmax(logits.float() / temp, dim=-1)
            probs = _top_p_filtering(probs, float(top_p))
            token = torch.multinomial(probs, num_samples=1)         

                               
            past_len = _past_length(past)
            attn = torch.ones((B, past_len + 1), device=device, dtype=attention_mask.dtype)
            fwd = {
                "input_ids": token,
                "attention_mask": attn,
                "past_key_values": past,
                "use_cache": True,
                "output_hidden_states": True,
                "return_dict": True,
            }
            if is_minicpm:
                fwd["position_ids"] = torch.full(
                    (B, 1),
                    past_len,
                    dtype=torch.long,
                    device=device,
                )
            out = text_model(**fwd)
            past = out.past_key_values
            logits = out.logits[:, -1, :]
            hs = _get_hidden_states_tuple(out)
            if not hs:
                raise RuntimeError("Model did not return hidden_states during rollout")
            last_hidden = hs[-1][:, -1, :]

            latent_vec = wrapper._apply_latent_realignment(last_hidden, text_model)
            steps.append(latent_vec.detach().float())

        latents_all.append(torch.stack(steps, dim=1))                        

    return torch.stack(latents_all, dim=1)                           


                               
                              
                               


@torch.no_grad()
def _extract_dummy_image_tokens(
    wrapper: ModelWrapper,
    processor,
    special_ids: _SpecialTokenIds,
    dummy_imgs: List[Image.Image],
) -> torch.Tensor:
    tok = _get_tokenizer_like(processor)
    if tok is None:
        raise RuntimeError("AutoProcessor has no tokenizer")
    if not dummy_imgs:
        raise RuntimeError("dummy_imgs must be non-empty")
    n_imgs = max(1, len(dummy_imgs))
    primary_img = dummy_imgs[0]

    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": _build_mm_user_content(num_images=n_imgs, text=" "),
        },
    ]

    if _is_minicpm_wrapper(wrapper):
        mini_msgs = [{"role": "user", "content": "(<image>./</image>)\n "}]
        if hasattr(tok, "apply_chat_template"):
            chat = tok.apply_chat_template(mini_msgs, tokenize=False, add_generation_prompt=True)
        else:
            chat = "(<image>./</image>)\n "
    elif hasattr(tok, "apply_chat_template"):
        chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        chat = "You are a helpful assistant.\n\n[IMAGE]\n "

    if _is_internvl_wrapper(wrapper):
        mm = _internvl_prepare_multimodal_batch(
            wrapper=wrapper,
            tokenizer=tok,
            prompts=[chat],
            images=[primary_img],
            device=wrapper.model.device,
        )
        out = wrapper.model(
            input_ids=mm["input_ids"],
            attention_mask=mm["attention_mask"],
            pixel_values=mm["pixel_values"],
            image_flags=mm["image_flags"],
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        pos = _find_image_positions(mm["input_ids"], special_ids, tokenizer=tok, batch_index=0)
    elif _is_minicpm_wrapper(wrapper):
        mm = _minicpm_prepare_multimodal_batch(
            wrapper=wrapper,
            processor=processor,
            prompts=[chat],
            images=[primary_img],
            device=wrapper.model.device,
        )
        base_embeds = _minicpm_build_inputs_embeds(wrapper, mm).detach()
        pos = _minicpm_positions_from_bounds(mm.get("image_bound", [None])[0])
        if not pos:
            pos = _find_image_positions(mm["input_ids"], special_ids, tokenizer=tok, batch_index=0)
        if not pos:
            raise RuntimeError("Could not locate image token span in MiniCPM dummy prompt")
        return base_embeds[0, pos, :].contiguous().float()
    else:
        enc = _processor_encode_multimodal(processor, texts=[chat], dummy_imgs=dummy_imgs)
        enc = _maybe_to_device(enc, wrapper.model.device)
        model_type = str(getattr(getattr(wrapper.model, "config", None), "model_type", "")).lower()
        if model_type == "gemma3":
            pv = enc.get("pixel_values", None)
            exp_img = getattr(getattr(wrapper.model, "config", None), "vision_config", None)
            exp_img_size = getattr(exp_img, "image_size", None)
            if (
                torch.is_tensor(pv)
                and pv.ndim >= 4
                and isinstance(exp_img_size, int)
                and exp_img_size > 0
                and int(pv.shape[-1]) != int(exp_img_size)
            ):
                raise RuntimeError(
                    "Gemma3 pixel_values image size mismatch before forward: "
                    f"got {tuple(pv.shape)}, expected H/W={int(exp_img_size)}. "
                    "Use fast processor (use_fast=True) or set --vision_codec_dummy_image_size to match Gemma vision size."
                )
        out = wrapper.model(**enc, output_hidden_states=True, use_cache=False, return_dict=True)
        pos = _find_image_positions(enc["input_ids"], special_ids, tokenizer=tok, batch_index=0)

    hs = _get_hidden_states_tuple(out)
    if not hs:
        raise RuntimeError("Model did not return hidden_states")
    hs0 = hs[0]                                                                      

    if not pos:
        raise RuntimeError("Could not locate image token span in dummy prompt")

    return hs0[0, pos, :].contiguous().float()                      


                               
             
                               


def _ddp_is_active() -> bool:
    return dist.is_available() and dist.is_initialized()


def _ddp_rank() -> int:
    return int(dist.get_rank()) if _ddp_is_active() else 0


def _ddp_world_size() -> int:
    return int(dist.get_world_size()) if _ddp_is_active() else 1


def _ddp_barrier() -> None:
    if _ddp_is_active():
        dist.barrier()


def _ddp_all_true(flag: bool, device: torch.device) -> bool:
    if not _ddp_is_active():
        return bool(flag)
    x = torch.tensor(1 if flag else 0, device=device, dtype=torch.int32)
    dist.all_reduce(x, op=dist.ReduceOp.MIN)
    return bool(int(x.item()) == 1)


def _looks_like_timeout_error(exc: Exception) -> bool:
    cur: Optional[BaseException] = exc
    hops = 0
    while cur is not None and hops < 12:
        name = cur.__class__.__name__.lower()
        msg = str(cur).lower()
        if "timeout" in name or "timed out" in msg:
            return True
        nxt = getattr(cur, "__cause__", None)
        if nxt is None:
            nxt = getattr(cur, "__context__", None)
        cur = nxt
        hops += 1
    return False


def _load_pretrained_with_cache_warmup(
    *,
    label: str,
    load_fn,
    network_retries: int = 3,
) -> Any:
       
    rank = _ddp_rank()
    world = _ddp_world_size()

                                  
    try:
        return load_fn(local_files_only=True)
    except Exception:
        pass

                                                              
    if world > 1:
        if rank == 0:
            for attempt in range(max(1, int(network_retries))):
                try:
                    _ = load_fn(local_files_only=False)
                    break
                except Exception as e:
                    if attempt + 1 >= max(1, int(network_retries)):
                        break
                    sleep_s = min(30.0, 4.0 * (attempt + 1))
                    print(
                        f"[processor-load][rank0] {label} warmup failed (attempt {attempt + 1}/{network_retries}): "
                        f"{type(e).__name__}: {e}. Retrying in {sleep_s:.1f}s..."
                    )
                    time.sleep(sleep_s)
        _ddp_barrier()
        try:
            return load_fn(local_files_only=True)
        except Exception:
            pass

                                                                    
    if world > 1 and rank > 0:
        time.sleep(min(8.0, float(rank)))
    last_err: Optional[Exception] = None
    for attempt in range(max(1, int(network_retries))):
        try:
            return load_fn(local_files_only=False)
        except Exception as e:
            last_err = e
            if attempt + 1 < max(1, int(network_retries)):
                sleep_s = min(45.0, 5.0 * (attempt + 1))
                time.sleep(sleep_s)
    if last_err is None:
        raise RuntimeError(f"{label} failed with unknown error")
    raise last_err


def _load_mm_processor_or_tokenizer(name: str, wrapper: ModelWrapper):
    is_gemma3 = "gemma-3" in str(name).lower()

    def _load_processor(local_files_only: bool):
        kwargs = {
            "trust_remote_code": True,
            "local_files_only": bool(local_files_only),
        }
                                                                                          
                                                                                           
                                                                                           
        if is_gemma3:
            try:
                return AutoProcessor.from_pretrained(name, use_fast=True, **kwargs)
            except Exception:
                pass
        return AutoProcessor.from_pretrained(name, **kwargs)

    try:
        proc = _load_pretrained_with_cache_warmup(
            label=f"AutoProcessor({name})",
            load_fn=_load_processor,
            network_retries=3,
        )
        if _is_minicpm_wrapper(wrapper):
            _patch_minicpm_batchfeature_if_needed(proc)
        return proc
    except Exception as e_proc:
        if _is_internvl_wrapper(wrapper):
            def _load_tokenizer(local_files_only: bool):
                return AutoTokenizer.from_pretrained(
                    name,
                    trust_remote_code=True,
                    local_files_only=bool(local_files_only),
                )

            try:
                return _load_pretrained_with_cache_warmup(
                    label=f"AutoTokenizer({name})",
                    load_fn=_load_tokenizer,
                    network_retries=3,
                )
            except Exception as e_tok:
                raise RuntimeError(
                    f"Could not load processor/tokenizer for InternVL model '{name}'.\n"
                    f"AutoProcessor error: {repr(e_proc)}\n"
                    f"AutoTokenizer error: {repr(e_tok)}"
                ) from e_tok

        if _looks_like_timeout_error(e_proc):
            raise RuntimeError(
                f"Could not load AutoProcessor for model '{name}' due to Hugging Face network timeout.\n"
                "This is not necessarily a text-only model issue. "
                "Retry the job after cache warmup or with larger HF timeouts.\n"
                f"Original error: {repr(e_proc)}"
            ) from e_proc

        raise RuntimeError(
            f"Could not load AutoProcessor for model '{name}'. "
            "This model may be text-only, or processor files may be unavailable locally.\n"
            f"Original error: {repr(e_proc)}"
        ) from e_proc


                               
                          
                               


def _train_one_model(
    *,
    wrapper: ModelWrapper,
    processor,
    special_ids: _SpecialTokenIds,
    dummy_imgs: List[Image.Image],
    anchor_texts: List[str],
    cfg: argparse.Namespace,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], int]:
                                                                   

    device = wrapper.model.device
    tok = _get_tokenizer_like(processor) or getattr(wrapper, "tokenizer", None)
    if tok is None:
        raise RuntimeError("processor has no tokenizer")
    text_model = _get_text_backbone(wrapper)

    codec_bf16_requested = int(getattr(cfg, "codec_uses_bf16", 0)) == 1
    codec_dtype = torch.bfloat16 if (codec_bf16_requested and device.type == "cuda") else torch.float32
    if codec_bf16_requested and codec_dtype != torch.bfloat16 and _ddp_rank() == 0:
        print(
            "[codec-train][warn] --codec_uses_bf16=1 requested but CUDA is unavailable; falling back to float32 codec.",
            flush=True,
        )
    avoid_full_logits_in_student = int(getattr(cfg, "avoid_full_logits_in_student", 0)) == 1
    if _ddp_rank() == 0:
        print(
            f"[codec-train] codec_dtype={str(codec_dtype)} avoid_full_logits_in_student={int(avoid_full_logits_in_student)}",
            flush=True,
        )

                
    wrapper.model.eval()
    for p in wrapper.model.parameters():
        p.requires_grad_(False)

    H = _infer_hidden_size(wrapper)
    enc = LatentToUniversalEncoder(
        h_in=H,
        d_univ=int(cfg.vision_codec_dim),
        k_univ=int(cfg.vision_codec_tokens),
        n_heads=int(cfg.vision_codec_heads),
        n_layers=int(cfg.vision_codec_layers),
        dropout=float(cfg.vision_codec_dropout),
    ).to(device=device, dtype=codec_dtype)

    dec = UniversalToVisionDecoder(
        d_univ=int(cfg.vision_codec_dim),
        h_out=H,
        k_img=int(cfg.vision_codec_img_tokens),
        n_heads=int(cfg.vision_codec_heads),
        n_layers=int(cfg.vision_codec_layers),
        dropout=float(cfg.vision_codec_dropout),
        gate_init_bias=float(cfg.vision_codec_gate_init_bias),
    ).to(device=device, dtype=codec_dtype)

                                                                      
    if _ddp_is_active() and torch.cuda.is_available():
        enc = nn.parallel.DistributedDataParallel(enc, device_ids=[device.index] if device.type == "cuda" else None)
        dec = nn.parallel.DistributedDataParallel(dec, device_ids=[device.index] if device.type == "cuda" else None)

    enc.train()
    dec.train()

    opt = torch.optim.AdamW(
        list(enc.parameters()) + list(dec.parameters()),
        lr=float(cfg.vision_codec_train_lr),
    )

    if not dummy_imgs:
        raise RuntimeError("dummy_imgs must be non-empty")
    dummy_img_primary = dummy_imgs[0]
    if (_is_internvl_wrapper(wrapper) or _is_minicpm_wrapper(wrapper)) and len(dummy_imgs) > 1 and _ddp_rank() == 0:
        print(
            f"[codec-train][warn] {wrapper.model_name} currently uses one dummy image per sample in this path; "
            f"received {len(dummy_imgs)} and will use the first image only.",
            flush=True,
        )

    dummy_tokens = _extract_dummy_image_tokens(wrapper, processor, special_ids, dummy_imgs)                     
    dummy_rms = dummy_tokens.pow(2).mean().sqrt().clamp_min(1e-6)

    steps = int(cfg.vision_codec_train_steps)
    bs = int(cfg.vision_codec_train_batch_size)
    mc = int(cfg.vision_codec_mc_rollouts)
    rollout_mode = str(cfg.vision_codec_rollout_mode).lower()
    loss_mse_w = float(cfg.vision_codec_loss_mse)
    loss_kl_w = float(cfg.vision_codec_loss_kl)
    loss_stats_w = float(cfg.vision_codec_loss_stats)
    kl_mode = str(getattr(cfg, "vision_codec_kl_mode", "auto")).strip().lower()
    kl_topk = int(getattr(cfg, "vision_codec_kl_topk", 0))
    if kl_mode == "auto":
        if _is_internvl_wrapper(wrapper):
            kl_mode = str(getattr(cfg, "vision_codec_internvl_kl_mode", "low_var_kl")).strip().lower()
            if kl_topk <= 0:
                kl_topk = int(getattr(cfg, "vision_codec_internvl_kl_topk", 256))
        else:
            kl_mode = "full"

                                                       
    if _is_internvl_wrapper(wrapper) and int(getattr(cfg, "vision_codec_internvl_disable_kl", 0)) == 1 and loss_kl_w > 0:
        if _ddp_rank() == 0:
            print(
                "[codec-train] Deprecated flag --vision_codec_internvl_disable_kl=1 used; "
                "KL is disabled for this run.",
                flush=True,
            )
        loss_kl_w = 0.0

    if _ddp_rank() == 0 and loss_kl_w > 0:
        print(
            f"[codec-train] KL config: mode={kl_mode}, topk={kl_topk}, temp={float(cfg.vision_codec_kl_temp):.4f}",
            flush=True,
        )

    kl_logit_clip = float(getattr(cfg, "vision_codec_kl_logit_clip", 80.0))
    latent_clip = float(getattr(cfg, "vision_codec_latent_clip", 50.0))
    inj_clip = float(getattr(cfg, "vision_codec_inj_clip", 20.0))

    pbar = range(steps)
    if _ddp_rank() == 0:
        pbar = tqdm(pbar, desc=f"[codec-train] {wrapper.model_name}")

    codec_params = list(enc.parameters()) + list(dec.parameters())

    all_bad_grad_streak = 0

    for _step in pbar:
        batch_texts = random.sample(anchor_texts, k=min(bs, len(anchor_texts)))
        B = len(batch_texts)
        if B == 0:
            continue

                                                
        teacher_msgs = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Message:\n{t}\n\nAcknowledge."},
            ]
            for t in batch_texts
        ]
        _, teacher_ids, teacher_mask, _ = wrapper.prepare_chat_batch(teacher_msgs, add_generation_prompt=True)
        teacher_ids = teacher_ids.to(device)
        teacher_mask = teacher_mask.to(device)

        with torch.no_grad():
            out_t = text_model(
                input_ids=teacher_ids,
                attention_mask=teacher_mask,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            hs_t = _get_hidden_states_tuple(out_t)
            if not hs_t:
                raise RuntimeError("Teacher forward did not return hidden_states")
            last_pos = (teacher_mask.sum(dim=1) - 1).long()
            teacher_h = hs_t[-1][torch.arange(B, device=device), last_pos, :]
            teacher_logits = out_t.logits[torch.arange(B, device=device), last_pos, :]

                                    
        if rollout_mode == "token":
            latents = _sample_token_rollouts(
                wrapper,
                input_ids=teacher_ids,
                attention_mask=teacher_mask,
                latent_steps=int(cfg.latent_steps),
                num_rollouts=mc,
                temperature=float(cfg.vision_codec_rollout_temperature),
                top_p=float(cfg.vision_codec_rollout_top_p),
            )                 
        else:
                                                                                         
            _, lat = wrapper.generate_latent_batch(
                teacher_ids,
                attention_mask=teacher_mask,
                latent_steps=int(cfg.latent_steps),
                return_latent_embeds=True,
            )             
            lat = lat.detach().float()
            if mc <= 1:
                latents = lat.unsqueeze(1)             
            else:
                noise_std = float(getattr(cfg, "vision_codec_rollout_noise_std", 0.0))
                latents = lat.unsqueeze(1).repeat(1, mc, 1, 1)
                if noise_std > 0:
                    latents = latents + noise_std * torch.randn_like(latents)

                                                                                                     
        latents = torch.nan_to_num(latents.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        if latent_clip > 0:
            latents = latents.clamp(min=-latent_clip, max=latent_clip)

        B2, R, L, H2 = latents.shape
        assert B2 == B and H2 == H

        lat_flat = latents.reshape(B * R, L, H).to(device=device, dtype=codec_dtype)
        U_flat = torch.nan_to_num(
            enc(lat_flat),
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        )               
        delta_flat, gate_flat = dec(U_flat)                              
        delta_flat = torch.nan_to_num(delta_flat, nan=0.0, posinf=1e4, neginf=-1e4)
        gate_flat = torch.nan_to_num(gate_flat, nan=0.0, posinf=1.0, neginf=0.0)
        delta = delta_flat.reshape(B, R, delta_flat.shape[1], H)
        gate = gate_flat.reshape(B, R, 1, 1)
        inj = (gate * delta).mean(dim=1)                 
        inj = torch.nan_to_num(inj, nan=0.0, posinf=1e4, neginf=-1e4)
        if inj_clip > 0:
            inj = inj.clamp(min=-inj_clip, max=inj_clip)

                                                                             
        student_msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": _build_mm_user_content(
                    num_images=(1 if (_is_internvl_wrapper(wrapper) or _is_minicpm_wrapper(wrapper)) else len(dummy_imgs)),
                    text="Message:\n\nAcknowledge.",
                ),
            },
        ]
        if _is_minicpm_wrapper(wrapper):
            mini_msgs = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "(<image>./</image>)\nMessage:\n\nAcknowledge."},
            ]
            if hasattr(tok, "apply_chat_template"):
                chat = tok.apply_chat_template(mini_msgs, tokenize=False, add_generation_prompt=True)
            else:
                chat = "You are a helpful assistant.\n\n(<image>./</image>)\nMessage:\n\nAcknowledge."
        elif hasattr(tok, "apply_chat_template"):
            chat = tok.apply_chat_template(student_msgs, tokenize=False, add_generation_prompt=True)
        else:
            chat = "You are a helpful assistant.\n\n[IMAGE]\nMessage:\n\nAcknowledge."

        if _is_internvl_wrapper(wrapper):
            mm = _internvl_prepare_multimodal_batch(
                wrapper=wrapper,
                tokenizer=tok,
                prompts=[chat] * B,
                images=[dummy_img_primary] * B,
                device=device,
            )
            input_ids = mm["input_ids"]
            emb = wrapper.model.language_model.get_input_embeddings()
            emb_dtype = emb.weight.dtype
            base_embeds = emb(input_ids).detach()
            inputs_embeds = base_embeds.clone()
            with torch.no_grad():
                vit_embeds = wrapper.model.extract_feature(mm["pixel_values"]).detach().float()
            for b in range(B):
                pos = _find_image_positions(input_ids, special_ids, tokenizer=tok, batch_index=b)
                if not pos:
                    continue
                base_img = _resample_tokens(vit_embeds[b], len(pos)).to(dtype=emb_dtype, device=device)
                add = _resample_tokens(inj[b], len(pos)).to(dtype=emb_dtype, device=device)
                inputs_embeds[b, pos, :] = base_img + add
            student_model = wrapper.model.language_model
            student_fwd = {
                "inputs_embeds": inputs_embeds,
                "attention_mask": mm["attention_mask"],
            }
            attn_s = mm["attention_mask"]
        elif _is_minicpm_wrapper(wrapper):
            mm = _minicpm_prepare_multimodal_batch(
                wrapper=wrapper,
                processor=processor,
                prompts=[chat] * B,
                images=[dummy_img_primary] * B,
                device=device,
            )
            base_embeds = _minicpm_build_inputs_embeds(wrapper, mm).detach()
            emb_dtype = base_embeds.dtype
            inputs_embeds = base_embeds.clone()
            for b in range(B):
                bounds = mm.get("image_bound", None)
                pos = _minicpm_positions_from_bounds(bounds[b] if isinstance(bounds, list) and b < len(bounds) else None)
                if not pos:
                    pos = _find_image_positions(mm["input_ids"], special_ids, tokenizer=tok, batch_index=b)
                if not pos:
                    continue
                base_img = base_embeds[b, pos, :]
                add = _resample_tokens(inj[b], len(pos)).to(dtype=emb_dtype, device=device)
                inputs_embeds[b, pos, :] = base_img + add
            pos_ids = _build_position_ids_from_attention(mm["attention_mask"])
            student_model = wrapper.model.llm
            student_fwd = {
                "inputs_embeds": inputs_embeds,
                "attention_mask": mm["attention_mask"],
                "position_ids": pos_ids,
            }
            attn_s = mm["attention_mask"]
        else:
            mm = _processor_encode_multimodal(processor, texts=[chat] * B, dummy_imgs=dummy_imgs)
            mm = _maybe_to_device(mm, device)
            input_ids = mm.get("input_ids", None)
            if input_ids is None:
                raise RuntimeError("processor did not return input_ids")
            emb = wrapper.model.get_input_embeddings()
            emb_dtype = emb.weight.dtype
            base_embeds = emb(input_ids).detach()
            inputs_embeds = base_embeds.clone()
            for b in range(B):
                pos = _find_image_positions(input_ids, special_ids, tokenizer=tok, batch_index=b)
                if not pos:
                    continue
                base_img = _resample_tokens(dummy_tokens, len(pos)).to(dtype=emb_dtype, device=device)
                add = _resample_tokens(inj[b], len(pos)).to(dtype=emb_dtype, device=device)
                inputs_embeds[b, pos, :] = base_img + add
            fwd = {k: v for k, v in mm.items() if k not in ("input_ids", "pixel_values")}
            fwd["inputs_embeds"] = inputs_embeds
            student_model = wrapper.model
            student_fwd = dict(fwd)
            attn_s = mm.get("attention_mask", None)
        if avoid_full_logits_in_student:
            student_h, student_logits = _student_forward_last_token_only(
                causal_module=student_model,
                forward_kwargs=student_fwd,
            )
        else:
            out_s = student_model(
                **student_fwd,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            hs_s = _get_hidden_states_tuple(out_s)
            if not hs_s:
                raise RuntimeError("Student forward did not return hidden_states")
            last_pos_s = (
                (attn_s.sum(dim=1) - 1).long()
                if attn_s is not None
                else torch.full((B,), inputs_embeds.shape[1] - 1, device=device, dtype=torch.long)
            )
            student_h = hs_s[-1][torch.arange(B, device=device), last_pos_s, :]
            student_logits = out_s.logits[torch.arange(B, device=device), last_pos_s, :]
        teacher_h = torch.nan_to_num(teacher_h.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        student_h = torch.nan_to_num(student_h.float(), nan=0.0, posinf=1e4, neginf=-1e4)

                
        loss = torch.zeros((), device=device, dtype=torch.float32)
        loss_mse = torch.zeros((), device=device, dtype=torch.float32)
        loss_kl = torch.zeros((), device=device, dtype=torch.float32)
        loss_stats = torch.zeros((), device=device, dtype=torch.float32)
        if loss_mse_w > 0:
            loss_mse = F.mse_loss(student_h, teacher_h)
            loss = loss + loss_mse_w * loss_mse

        if loss_kl_w > 0:
            loss_kl = _compute_kl_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                temp=float(cfg.vision_codec_kl_temp),
                mode=kl_mode,
                logit_clip=kl_logit_clip,
                topk=kl_topk,
            )
            loss = loss + loss_kl_w * loss_kl

        if loss_stats_w > 0:
            inj_rms = inj.float().pow(2).mean().sqrt().clamp_min(1e-6)
            loss_stats = F.mse_loss(inj_rms, dummy_rms.expand_as(inj_rms))
            loss = loss + loss_stats_w * loss_stats

        local_loss_ok = bool(
            torch.isfinite(loss).item()
            and torch.isfinite(loss_mse).item()
            and torch.isfinite(loss_kl).item()
            and torch.isfinite(loss_stats).item()
        )
        global_loss_ok = _ddp_all_true(local_loss_ok, device=device)
        if not global_loss_ok:
            print(
                f"[codec-train][rank{_ddp_rank()}] Non-finite loss at step {_step}: "
                f"total={float(loss.detach().cpu())}, mse={float(loss_mse.detach().cpu())}, "
                f"kl={float(loss_kl.detach().cpu())}, stats={float(loss_stats.detach().cpu())}. Skipping step.",
                flush=True,
            )
            opt.zero_grad(set_to_none=True)
            continue

        opt.zero_grad(set_to_none=True)
        loss.backward()

        nonfinite_grad_elems = 0
        total_grad_elems = 0
        for p in codec_params:
            g = p.grad
            if g is None:
                continue
            total_grad_elems += int(g.numel())
            bad_mask = ~torch.isfinite(g)
            bad = int(bad_mask.sum().item())
            if bad > 0:
                nonfinite_grad_elems += bad
                p.grad = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

        if nonfinite_grad_elems > 0:
            print(
                f"[codec-train][rank{_ddp_rank()}] Non-finite gradient elements at step {_step}: "
                f"{nonfinite_grad_elems}/{max(1, total_grad_elems)}. Sanitized to zero before clip.",
                flush=True,
            )

        if total_grad_elems > 0 and nonfinite_grad_elems == total_grad_elems:
            all_bad_grad_streak += 1
        else:
            all_bad_grad_streak = 0

        if all_bad_grad_streak >= 5:
            raise RuntimeError(
                f"All codec gradients became non-finite for {all_bad_grad_streak} consecutive steps "
                f"(last step {_step}). This usually indicates numerical instability; try lowering "
                f"--vision_codec_train_lr and/or --vision_codec_inj_clip."
            )

        grad_norm = torch.nn.utils.clip_grad_norm_(codec_params, max_norm=1.0, error_if_nonfinite=False)
        local_grad_ok = bool(torch.isfinite(grad_norm).item())
        global_grad_ok = _ddp_all_true(local_grad_ok, device=device)
        if not global_grad_ok:
            print(
                f"[codec-train][rank{_ddp_rank()}] Non-finite grad norm at step {_step}: "
                f"grad_norm={float(grad_norm.detach().cpu())}. Skipping optimizer step.",
                flush=True,
            )
            opt.zero_grad(set_to_none=True)
            continue

        opt.step()

        if _ddp_rank() == 0 and isinstance(pbar, tqdm):
            pbar.set_postfix({"loss": f"{float(loss.detach().cpu()):.6f}"})
            log_every = max(1, int(getattr(cfg, "vision_codec_log_every", 10)))
            if ((_step + 1) % log_every) == 0:
                print(
                    f"[codec-train][{wrapper.model_name}] step={_step + 1}/{steps} "
                    f"loss={float(loss.detach().cpu()):.6f} "
                    f"mse={float(loss_mse.detach().cpu()):.6f} "
                    f"kl={float(loss_kl.detach().cpu()):.6f} "
                    f"stats={float(loss_stats.detach().cpu()):.6f}",
                    flush=True,
                )

                
    enc_mod = enc.module if hasattr(enc, "module") else enc
    dec_mod = dec.module if hasattr(dec, "module") else dec

                                                                              
    enc_sd = {k: v.detach().float().cpu() for k, v in enc_mod.state_dict().items()}
    dec_sd = {k: v.detach().float().cpu() for k, v in dec_mod.state_dict().items()}
    return enc_sd, dec_sd, int(dummy_tokens.shape[0])


                               
                             
                               


@torch.no_grad()
def _collect_anchor_U(
    *,
    wrapper: ModelWrapper,
    enc_sd: Dict[str, torch.Tensor],
    anchor_texts: List[str],
    cfg: argparse.Namespace,
) -> torch.Tensor:
                                             
    device = wrapper.model.device
    H = _infer_hidden_size(wrapper)

    enc = LatentToUniversalEncoder(
        h_in=H,
        d_univ=int(cfg.vision_codec_dim),
        k_univ=int(cfg.vision_codec_tokens),
        n_heads=int(cfg.vision_codec_heads),
        n_layers=int(cfg.vision_codec_layers),
        dropout=float(cfg.vision_codec_dropout),
    ).to(device=device, dtype=torch.float32)
    enc.load_state_dict(enc_sd, strict=True)
    enc.eval()

    out_list: List[torch.Tensor] = []
    align_bs = _safe_int(getattr(cfg, "vision_codec_align_batch_size", 0), 0)
    if align_bs > 0:
        bs = max(1, align_bs)
    else:
        bs = max(1, min(4, int(cfg.vision_codec_train_batch_size)))
    it = range(0, len(anchor_texts), bs)
    if len(anchor_texts) > bs:
        it = tqdm(it, desc=f"[align-U] {wrapper.model_name}", leave=False)
    for s in it:
        texts = anchor_texts[s : s + bs]
        msgs = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Message:\n{t}\n\nAcknowledge."},
            ]
            for t in texts
        ]
        _, ids, mask, _ = wrapper.prepare_chat_batch(msgs, add_generation_prompt=True)
        ids = ids.to(device)
        mask = mask.to(device)
        _, lat = wrapper.generate_latent_batch(ids, attention_mask=mask, latent_steps=int(cfg.latent_steps), return_latent_embeds=True)
        U = enc(lat.detach().float()).detach().float().cpu()
        out_list.append(U)
    return torch.cat(out_list, dim=0) if out_list else torch.empty((0, int(cfg.vision_codec_tokens) + 2, int(cfg.vision_codec_dim)), dtype=torch.float32)


                               
      
                               


def main() -> None:
    p = argparse.ArgumentParser()

    p.add_argument("--model_name", type=str, default="", help="Single model name")
    p.add_argument("--agent_model_names", type=str, default="", help="Comma-separated model names")
    p.add_argument("--vision_codec_path", type=str, required=True, help="Where to save the codec checkpoint")
    p.add_argument("--seed", type=int, default=42)

                                               
    p.add_argument("--latent_steps", type=int, default=1024)


    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--generate_bs", type=int, default=20, help="Batch size for generation")
    p.add_argument("--text_mas_context_length", type=int, default=-1, help="TextMAS context length limit")
    p.add_argument("--think", action="store_true", help="Manually add think token in the prompt for LatentMAS")
    p.add_argument("--latent_space_realign", action="store_true")
                  
    p.add_argument("--use_vllm", action="store_true", help="Use vLLM backend for generation")
    p.add_argument("--enable_prefix_caching", action="store_true", help="Enable prefix caching in vLLM for latent_mas")
    p.add_argument("--use_second_HF_model", action="store_true", help="Use a second HF model for latent generation in latent_mas")
    p.add_argument("--device2", type=str, default="cuda:1")
    p.add_argument("--tensor_parallel_size", type=int, default=1, help="How many GPUs vLLM should shard the model across")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="Target GPU memory utilization for vLLM")


                        
    p.add_argument("--vision_codec_dim", type=int, default=256)
    p.add_argument("--vision_codec_tokens", type=int, default=1024)
    p.add_argument("--vision_codec_img_tokens", type=int, default=256)
    p.add_argument("--vision_codec_heads", type=int, default=8)
    p.add_argument("--vision_codec_layers", type=int, default=4)
    p.add_argument("--vision_codec_dropout", type=float, default=0.)
    p.add_argument("--vision_codec_gate_init_bias", type=float, default=-4.0)

                          
    p.add_argument("--vision_codec_train_steps", type=int, default=200)
    p.add_argument("--vision_codec_train_batch_size", type=int, default=8)
    p.add_argument("--vision_codec_train_lr", type=float, default=5e-4)
    p.add_argument(
        "--codec_uses_bf16",
        type=int,
        default=0,
        help="If 1, train codec modules in bfloat16 on CUDA (checkpoint is still saved in fp32).",
    )
    p.add_argument(
        "--avoid_full_logits_in_student",
        type=int,
        default=0,
        help=(
            "If 1, use memory-optimized student forward that avoids full-sequence logits/"
            "all-layer hidden-state outputs."
        ),
    )
    p.add_argument("--vision_codec_loss_mse", type=float, default=1.0)
    p.add_argument("--vision_codec_loss_kl", type=float, default=0.25)
    p.add_argument("--vision_codec_loss_stats", type=float, default=0.1)
    p.add_argument(
        "--vision_codec_kl_mode",
        type=str,
        default="auto",
        help="KL variant: auto|full|k1|abs|k2|k3|k3+ (aliases: kl,mse,low_var_kl).",
    )
    p.add_argument(
        "--vision_codec_kl_topk",
        type=int,
        default=0,
        help="If >0, compute KL-style loss on teacher top-k logits only.",
    )
    p.add_argument(
        "--vision_codec_internvl_kl_mode",
        type=str,
        default="low_var_kl",
        help="KL mode override used when --vision_codec_kl_mode=auto and model is InternVL.",
    )
    p.add_argument(
        "--vision_codec_internvl_kl_topk",
        type=int,
        default=256,
        help="Top-k override used when --vision_codec_kl_mode=auto and model is InternVL.",
    )
    p.add_argument(
        "--vision_codec_internvl_disable_kl",
        type=int,
        default=0,
        help="Deprecated: force-disable KL on InternVL (1/0). Prefer --vision_codec_kl_mode.",
    )
    p.add_argument("--vision_codec_kl_temp", type=float, default=1.0)
    p.add_argument("--vision_codec_kl_logit_clip", type=float, default=80.0, help="Clamp teacher/student logits to +/- this value before KL")
    p.add_argument("--vision_codec_latent_clip", type=float, default=50.0, help="Clamp sampled latents to +/- this value after rollout")
    p.add_argument("--vision_codec_inj_clip", type=float, default=20.0, help="Clamp decoded injection tokens to +/- this value before student forward")

                 
    p.add_argument("--vision_codec_mc_rollouts", type=int, default=1, help="N rollouts per sample")
    p.add_argument("--vision_codec_rollout_mode", type=str, default="latent", choices=["latent", "token"], help="latent = deterministic; token = sampled")
    p.add_argument("--vision_codec_rollout_temperature", type=float, default=1.0)
    p.add_argument("--vision_codec_rollout_top_p", type=float, default=1.0)
    p.add_argument("--vision_codec_rollout_noise_std", type=float, default=0.0, help="Only used in latent mode when mc_rollouts>1")
    p.add_argument("--vision_codec_log_every", type=int, default=10, help="Print stable scalar losses every N steps (rank0)")

          
    p.add_argument("--vision_codec_dummy_image_count", type=int, default=1)
    p.add_argument(
        "--vision_codec_dummy_image_counts",
        type=str,
        default="",
        help="Comma-separated dummy-image counts per model (aligned with --agent_model_names).",
    )
    p.add_argument("--vision_codec_dummy_image_size", type=int, default=224)
    p.add_argument(
        "--vision_codec_dummy_image_sizes",
        type=str,
        default="",
        help="Comma-separated dummy-image sizes per model (aligned with --agent_model_names).",
    )
    p.add_argument(
        "--vision_codec_dummy_image_spec_json",
        type=str,
        default="",
        help="JSON string/path mapping model_name -> {count,size} (overrides list/global dummy-image args).",
    )
    p.add_argument(
        "--vision_codec_check_dummy_img_tokens",
        type=int,
        default=0,
        help="If 1, check whether dummy image token counts are equal across models and report mismatches.",
    )
    p.add_argument(
        "--vision_codec_require_dummy_img_tokens_match",
        type=int,
        default=0,
        help="If 1 with --vision_codec_check_dummy_img_tokens=1, fail on mismatched dummy image token counts.",
    )
    p.add_argument("--vision_codec_anchor_texts_path", type=str, default="")
    p.add_argument(
        "--vision_codec_align_max_anchors",
        type=int,
        default=0,
        help="If >0, use at most this many anchor texts for universal alignment fitting.",
    )
    p.add_argument(
        "--vision_codec_align_batch_size",
        type=int,
        default=0,
        help=(
            "If >0, use this batch size for align-U anchor collection. "
            "If 0, defaults to min(4, vision_codec_train_batch_size)."
        ),
    )
    p.add_argument(
        "--vision_codec_align_loss_eval_max_rows",
        type=int,
        default=16384,
        help="Max rows used to estimate alignment ridge fit MSE (for logging).",
    )
    p.add_argument(
        "--vision_codec_save_per_model",
        type=int,
        default=1,
        help="If 1, save a progressive checkpoint after each model codec finishes training.",
    )
    p.add_argument(
        "--vision_codec_partial_ckpt_path",
        type=str,
        default="",
        help="Optional path for progressive per-model saves; default is <vision_codec_path>.partial.<ext>.",
    )
    p.add_argument(
        "--vision_codec_skip_alignment_if_single",
        type=int,
        default=1,
        help="If 1 and only one model is trained, skip expensive alignment fitting and use identity maps.",
    )
    p.add_argument("--vision_codec_ref_idx", type=int, default=0)
    p.add_argument("--vision_codec_ridge", type=float, default=1e-3)

    args = p.parse_args()

                                                                
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

                                                
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1

    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

                                                      
    seed = int(args.seed) + 1000 * _ddp_rank()
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

                 
    model_names = _parse_model_list(args.agent_model_names) if args.agent_model_names else ([] if not args.model_name else [args.model_name])
    if not model_names:
        raise ValueError("Provide --model_name or --agent_model_names")
    dummy_specs = _resolve_dummy_image_specs(model_names, args)
    if _ddp_rank() == 0:
        for i, name in enumerate(model_names):
            spec = dummy_specs[name]
            print(
                f"[dummy-spec] model[{i}]={name} count={spec.count} size={spec.size}",
                flush=True,
            )
    target_ref_idx = max(0, min(int(args.vision_codec_ref_idx), len(model_names) - 1))
    target_ref_name = model_names[target_ref_idx]

    partial_ckpt_path = ""
    if int(args.vision_codec_save_per_model) == 1:
        partial_ckpt_path = (args.vision_codec_partial_ckpt_path or "").strip() or _derive_partial_ckpt_path(args.vision_codec_path)
        if _ddp_rank() == 0:
            print(f"Progressive per-model checkpoint path: {partial_ckpt_path}", flush=True)

                  
    anchor_texts = _load_anchor_texts(args.vision_codec_anchor_texts_path)
    if _ddp_rank() == 0:
        print(f"Loaded {len(anchor_texts)} anchor texts for training.", flush=True)

                                                     
    encoders: Dict[str, Dict[str, torch.Tensor]] = {}
    decoders: Dict[str, Dict[str, torch.Tensor]] = {}
    specials: Dict[str, _SpecialTokenIds] = {}
    dummy_token_lens: Dict[str, int] = {}

    for mi, name in enumerate(model_names):
        _ddp_barrier()
        if _ddp_rank() == 0:
            print(f"\n==============================\nTraining codec for model[{mi}]: {name}\n==============================")

        device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
        wrapper = ModelWrapper(name, device, use_vllm=False, args=args)

        processor = _load_mm_processor_or_tokenizer(name, wrapper)

        special_ids = _infer_special_token_ids(processor)
        specials[name] = special_ids
        spec = dummy_specs[name]
        dummy_imgs = [_make_dummy_image(spec.size) for _ in range(spec.count)]

        enc_sd, dec_sd, dummy_token_len = _train_one_model(
            wrapper=wrapper,
            processor=processor,
            special_ids=special_ids,
            dummy_imgs=dummy_imgs,
            anchor_texts=anchor_texts,
            cfg=args,
        )

        if _ddp_rank() == 0:
            encoders[name] = enc_sd
            decoders[name] = dec_sd
            dummy_token_lens[name] = int(dummy_token_len)
            print(
                f"[dummy-spec] trained model={name} dummy_image_tokens={int(dummy_token_len)} "
                f"(count={spec.count}, size={spec.size})",
                flush=True,
            )

            if int(getattr(args, "vision_codec_check_dummy_img_tokens", 0)) == 1 and len(dummy_token_lens) >= 2:
                lens_set = set(dummy_token_lens.values())
                if len(lens_set) != 1:
                    msg = (
                        "Dummy image token counts do not match across models: "
                        + ", ".join([f"{k}={v}" for k, v in dummy_token_lens.items()])
                    )
                    if int(getattr(args, "vision_codec_require_dummy_img_tokens_match", 0)) == 1:
                        raise RuntimeError(msg)
                    print(f"[dummy-spec][warn] {msg}", flush=True)

            if partial_ckpt_path:
                done_models = [m for m in model_names if m in encoders and m in decoders]
                if done_models:
                    if target_ref_name in done_models:
                        partial_ref_name = target_ref_name
                        partial_ref_idx = done_models.index(target_ref_name)
                    else:
                        partial_ref_name = done_models[0]
                        partial_ref_idx = 0
                    partial_align = _identity_alignment(
                        model_names=done_models,
                        codec_dim=int(args.vision_codec_dim),
                        ref_idx=partial_ref_idx,
                        ref_name=partial_ref_name,
                    )
                    partial_ckpt = _build_codec_checkpoint(
                        model_names=done_models,
                        ref_model_name=partial_ref_name,
                        cfg=args,
                        encoders=encoders,
                        decoders=decoders,
                        align=partial_align,
                        is_partial=True,
                    )
                    _atomic_torch_save(partial_ckpt, partial_ckpt_path)
                    print(
                        f"[checkpoint] Saved progressive codec checkpoint after model[{mi}] "
                        f"to: {partial_ckpt_path} (models={len(done_models)})",
                        flush=True,
                    )

                                                 
        del wrapper
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    _ddp_barrier()

                                                             
    align: Dict[str, Any] = {}
    ref_idx = int(target_ref_idx)
    ref_name = target_ref_name

    if _ddp_rank() == 0:
        D = int(args.vision_codec_dim)
        if len(model_names) == 1 and int(args.vision_codec_skip_alignment_if_single) == 1:
            print(
                "\nSkipping universal-space alignment: only one model is being trained "
                "(using identity affine maps).",
                flush=True,
            )
            align = _identity_alignment(
                model_names=model_names,
                codec_dim=D,
                ref_idx=0,
                ref_name=model_names[0],
            )
            ref_idx = 0
            ref_name = model_names[0]
        else:
            align_anchor_texts = anchor_texts
            max_align_anchors = int(args.vision_codec_align_max_anchors)
            if max_align_anchors > 0 and len(align_anchor_texts) > max_align_anchors:
                rng = random.Random(int(args.seed) + 314159)
                sel = rng.sample(range(len(align_anchor_texts)), k=max_align_anchors)
                sel.sort()
                align_anchor_texts = [align_anchor_texts[i] for i in sel]
                print(
                    f"\nFitting universal-space alignment (ridge regression) on sampled anchors: "
                    f"{len(align_anchor_texts)}/{len(anchor_texts)}",
                    flush=True,
                )
            else:
                print(
                    f"\nFitting universal-space alignment (ridge regression) on anchors: "
                    f"{len(align_anchor_texts)} texts",
                    flush=True,
                )
            align_bs = _safe_int(getattr(args, "vision_codec_align_batch_size", 0), 0)
            if align_bs > 0:
                align_bs_eff = max(1, align_bs)
            else:
                align_bs_eff = max(1, min(4, int(args.vision_codec_train_batch_size)))
            print(f"[align-U] batch_size={align_bs_eff}", flush=True)

                                                         
            U_by_name: Dict[str, torch.Tensor] = {}
            for name in model_names:
                if name not in encoders:
                    continue
                device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
                wrapper = ModelWrapper(name, device, use_vllm=False, args=args)
                U_by_name[name] = _collect_anchor_U(
                    wrapper=wrapper,
                    enc_sd=encoders[name],
                    anchor_texts=align_anchor_texts,
                    cfg=args,
                )
                del wrapper
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            if ref_name not in U_by_name:
                raise RuntimeError(f"Reference model {ref_name} missing U anchors; did training skip it?")
            X_ref = U_by_name[ref_name].reshape(-1, D)

            out_map: Dict[str, Dict[str, torch.Tensor]] = {}
            in_map: Dict[str, Dict[str, torch.Tensor]] = {}
            loss_eval_rows = int(getattr(args, "vision_codec_align_loss_eval_max_rows", 16384))
            for name, U in U_by_name.items():
                X = U.reshape(-1, D)
                W_out, b_out = _ridge_fit(X, X_ref, ridge=float(args.vision_codec_ridge))
                W_in, b_in = _ridge_fit(X_ref, X, ridge=float(args.vision_codec_ridge))
                out_map[name] = {"W": W_out, "b": b_out}
                in_map[name] = {"W": W_in, "b": b_in}
                mse_out = _ridge_eval_mse(X, X_ref, W_out, b_out, max_rows=loss_eval_rows)
                mse_in = _ridge_eval_mse(X_ref, X, W_in, b_in, max_rows=loss_eval_rows)
                eval_rows = min(int(loss_eval_rows), int(X.shape[0]))
                print(
                    f"[align-U][loss] model={name} mse_out={mse_out:.6f} "
                    f"mse_in={mse_in:.6f} eval_rows={eval_rows}",
                    flush=True,
                )

            ref_idx = model_names.index(ref_name) if ref_name in model_names else 0
            align = {
                "ref_idx": int(ref_idx),
                "ref_model_name": ref_name,
                "out": out_map,
                "in": in_map,
            }

                               
        ckpt = _build_codec_checkpoint(
            model_names=model_names,
            ref_model_name=ref_name,
            cfg=args,
            encoders=encoders,
            decoders=decoders,
            align=align,
            is_partial=False,
        )
        _atomic_torch_save(ckpt, args.vision_codec_path)
        print(f"\nSaved codec checkpoint to: {args.vision_codec_path}")

    _ddp_barrier()
    if distributed:
        dist.destroy_process_group()


def _load_anchor_texts(path: str) -> List[str]:
    default = [
        "Summarize the following in one sentence: The mitochondrion is the powerhouse of the cell.",
        "Give a step-by-step plan to solve a two-digit multiplication problem.",
        "Explain what 'gradient descent' is in simple terms.",
        "Write a short critique of an argument that confuses correlation with causation.",
        "List three potential failure modes in multi-agent reasoning systems.",
        "State the Pythagorean theorem and one practical use-case.",
        "You are given: A=17, B=5. Compute A*B and show your reasoning.",
        "Define Bayes' rule and describe one intuition for it.",
        "Provide a counterexample to the claim: 'All prime numbers are odd.'",
        "Explain what an embedding is in machine learning.",
    ]
    if not path:
        return default
    if not os.path.exists(path):
        return default
    try:
        if path.endswith(".jsonl"):
            out: List[str] = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, str):
                        out.append(obj)
                    elif isinstance(obj, dict):
                        for k in ("text", "prompt", "message"):
                            if k in obj and isinstance(obj[k], str):
                                out.append(obj[k])
                                break
            return out if len(out) >= 4 else default
                   
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            out = [x for x in obj if isinstance(x, str)]
            return out if len(out) >= 4 else default
    except Exception:
        return default
    return default


if __name__ == "__main__":
    main()
