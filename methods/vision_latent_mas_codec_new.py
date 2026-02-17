   

from __future__ import annotations

import argparse
import copy
import inspect
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from transformers import AutoProcessor, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from . import default_agents
from models import ModelWrapper, _past_length
from prompts import (
    build_agent_message_hierarchical_latent_mas,
    build_agent_message_sequential_latent_mas,
)
from utils import extract_answer_with_meta


                                                                               
                         
                                                                               


def _safe_int(x: Any, default: int) -> int:
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _safe_float(x: Any, default: float) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


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
    default_size = max(
        32,
        _safe_int(getattr(args, "vision_codec_dummy_image_size", getattr(args, "vision_dummy_image_size", 224)), 224),
    )

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


def _load_role_model_map(raw: str) -> Dict[str, int]:
    if not raw:
        return {}
    payload = raw
    if os.path.exists(raw):
        with open(raw, "r", encoding="utf-8") as f:
            payload = f.read()
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("--role_model_map must be a JSON object mapping role -> model index.")
    out = {str(k).lower(): int(v) for k, v in data.items()}
    if "solver" in out and "judger" not in out:
        out["judger"] = out["solver"]
    return out


def _maybe_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return out


def _make_dummy_image(size: int = 224) -> Image.Image:
    return Image.new("RGB", (size, size), color=(255, 255, 255))


def _is_internvl_wrapper(wrapper: ModelWrapper) -> bool:
    if bool(getattr(wrapper, "is_internvl", False)):
        return True
    cfg = getattr(wrapper.model, "config", None)
    model_type = str(getattr(cfg, "model_type", "")).lower()
    cls_name = wrapper.model.__class__.__name__.lower()
    return ("internvl" in model_type) or ("internvl" in cls_name)


def _is_minicpm_wrapper(wrapper: ModelWrapper) -> bool:
    if bool(getattr(wrapper, "is_minicpm_v", False)):
        return True
    cfg = getattr(wrapper.model, "config", None)
    model_type = str(getattr(cfg, "model_type", "")).lower()
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


def _get_tokenizer_like(processor: Any) -> Optional[Any]:
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        return tok
    if hasattr(processor, "convert_tokens_to_ids") and hasattr(processor, "__call__"):
        return processor
    return None


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


def _extract_generated_ids(sequences: torch.Tensor, prompt_len: int) -> torch.Tensor:
       
    seq = sequences[0]
    if seq.dim() != 1:
        seq = seq.reshape(-1)
    if prompt_len >= 0 and seq.shape[0] > prompt_len:
        return seq[prompt_len:]
    return seq


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
    mm = processor(text=packed, images=img_lists, return_tensors="pt", padding=True)
    if "input_ids" not in mm or "attention_mask" not in mm:
        raise RuntimeError("MiniCPM processor did not return input_ids/attention_mask")
    mm["input_ids"] = mm["input_ids"].to(device)
    mm["attention_mask"] = mm["attention_mask"].to(device)

    image_bound = mm.get("image_bound", None)
    missing = (
        image_bound is None
        or not isinstance(image_bound, list)
        or len(image_bound) != int(mm["input_ids"].shape[0])
        or all((not isinstance(b, torch.Tensor)) or b.numel() == 0 for b in image_bound)
    )
    if missing:
        tok = getattr(processor, "tokenizer", None)
        unk_id = int(getattr(tok, "unk_token_id", 0)) if tok is not None else 0
        q = int(getattr(getattr(wrapper.model, "config", None), "query_num", 64))
        mm["image_bound"] = _fallback_image_bounds_from_unk(
            mm["input_ids"], mm["attention_mask"], unk_id=unk_id, query_num=q
        )
    return mm


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


def _normalize_minicpm_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for m in messages:
        role = str(m.get("role", "user"))
        content = m.get("content", "")
        if isinstance(content, list):
            parts: List[str] = []
            for c in content:
                if isinstance(c, dict):
                    ctype = str(c.get("type", "")).lower()
                    if ctype == "image":
                        parts.append("(<image>./</image>)")
                    elif ctype == "text":
                        parts.append(str(c.get("text", "")))
                    else:
                        parts.append(str(c))
                else:
                    parts.append(str(c))
            text = "\n".join([p for p in parts if p])
        else:
            text = str(content)
        if "<image>" in text:
            text = text.replace("<image>", "(<image>./</image>)")
        out.append({"role": role, "content": text})
    return out


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
    tokenizer: Any,
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


def _infer_hidden_size(wrapper: ModelWrapper) -> int:
    cfg = getattr(wrapper.model, "config", None)
    h = getattr(cfg, "hidden_size", None) if cfg is not None else None
    if h is None:
                                          
        h = int(wrapper.model.get_input_embeddings().weight.shape[1])
    return int(h)


def _get_hidden_states_tuple(out: Any) -> Optional[Tuple[torch.Tensor, ...]]:
    hs = getattr(out, "hidden_states", None)
    if hs is None:
        hs = getattr(out, "decoder_hidden_states", None)
    return hs


def _resample_tokens(x: torch.Tensor, target_len: int) -> torch.Tensor:
       
    if target_len <= 0:
        raise ValueError("target_len must be > 0")
    if x.dim() not in (2, 3):
        raise ValueError(f"x must be 2D or 3D, got {tuple(x.shape)}")

    orig_dtype = x.dtype
    xf = x.float()
    if xf.dim() == 2:
        xf = xf.unsqueeze(0)
        y = F.interpolate(xf.transpose(1, 2), size=target_len, mode="linear", align_corners=False).transpose(1, 2)
        return y[0].to(dtype=orig_dtype)
    y = F.interpolate(xf.transpose(1, 2), size=target_len, mode="linear", align_corners=False).transpose(1, 2)
    return y.to(dtype=orig_dtype)


def _apply_affine(U: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                                     
    return U @ W + b.view(1, 1, -1)


                                                                               
                                                                       
                                                                               


class _ResamplerBlock(nn.Module):
                                                                          

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ln_q = nn.LayerNorm(dim)
        self.ln_kv = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        kv_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q2 = self.ln_q(q)
        kv2 = self.ln_kv(kv)
        out, _ = self.attn(q2, kv2, kv2, key_padding_mask=kv_key_padding_mask, need_weights=False)
        q = q + out
        q = q + self.ff(q)
        return q


class LatentToUniversalEncoder(nn.Module):
                                                                       

    def __init__(
        self,
        h_in: int,
        d_univ: int,
        k_univ: int,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.h_in = int(h_in)
        self.d_univ = int(d_univ)
        self.k_univ = int(k_univ)

        self.in_proj = nn.Linear(self.h_in, self.d_univ)
        self.q_sem = nn.Parameter(torch.randn(self.k_univ, self.d_univ) * 0.02)
        self.q_global = nn.Parameter(torch.randn(1, self.d_univ) * 0.02)
        self.q_style = nn.Parameter(torch.randn(1, self.d_univ) * 0.02)

        self.blocks = nn.ModuleList([_ResamplerBlock(self.d_univ, n_heads, dropout) for _ in range(n_layers)])
        self.out_ln = nn.LayerNorm(self.d_univ)

        self.style_mlp = nn.Sequential(
            nn.Linear(3, self.d_univ),
            nn.GELU(),
            nn.Linear(self.d_univ, self.d_univ),
        )

    def forward(self, latents: torch.Tensor, latents_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if latents.dim() != 3:
            raise ValueError(f"latents must be [B,L,H], got {tuple(latents.shape)}")
        B, L, _ = latents.shape

        if L == 0:
            kv = latents.new_zeros((B, 1, self.d_univ))
            latents_key_padding_mask = None
            style = latents.new_zeros((B, 3), dtype=kv.dtype)
        else:
            x = self.in_proj(latents)
            kv = x
            lat_fp32 = latents.float()
            mean = lat_fp32.mean(dim=(1, 2))
            std = lat_fp32.std(dim=(1, 2), unbiased=False)
            norm = lat_fp32.norm(dim=-1).mean(dim=1)
            style = torch.stack([mean, std, norm], dim=-1).to(x.dtype)

        q = torch.cat(
            [
                self.q_sem.unsqueeze(0).expand(B, -1, -1),
                self.q_global.unsqueeze(0).expand(B, -1, -1),
                self.q_style.unsqueeze(0).expand(B, -1, -1),
            ],
            dim=1,
        )

        for blk in self.blocks:
            q = blk(q, kv, kv_key_padding_mask=latents_key_padding_mask)

        q[:, -1:, :] = q[:, -1:, :] + self.style_mlp(style).unsqueeze(1)
        return self.out_ln(q)


class UniversalToVisionDecoder(nn.Module):
                                                                                 

    def __init__(
        self,
        d_univ: int,
        h_out: int,
        k_img: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.0,
        gate_init_bias: float = -4.0,
    ):
        super().__init__()
        self.d_univ = int(d_univ)
        self.h_out = int(h_out)
        self.k_img = int(k_img)

        self.kv_ln = nn.LayerNorm(self.d_univ)
        self.q_img = nn.Parameter(torch.randn(self.k_img, self.d_univ) * 0.02)
        self.blocks = nn.ModuleList([_ResamplerBlock(self.d_univ, n_heads, dropout) for _ in range(n_layers)])
        self.out_ln = nn.LayerNorm(self.d_univ)
        self.out_proj = nn.Linear(self.d_univ, self.h_out)

        self.gate_mlp = nn.Sequential(
            nn.Linear(self.d_univ, self.d_univ),
            nn.GELU(),
            nn.Linear(self.d_univ, 1),
        )
        nn.init.constant_(self.gate_mlp[-1].bias, float(gate_init_bias))

    def forward(self, U: torch.Tensor, U_key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if U.dim() != 3:
            raise ValueError(f"U must be [B,K,D], got {tuple(U.shape)}")
        B, K, _ = U.shape
        if K == 0:
            delta = U.new_zeros((B, self.k_img, self.h_out))
            gate = U.new_zeros((B, 1, 1))
            return delta, gate

        kv = self.kv_ln(U)
        q = self.q_img.unsqueeze(0).expand(B, -1, -1)
        for blk in self.blocks:
            q = blk(q, kv, kv_key_padding_mask=U_key_padding_mask)
        q = self.out_ln(q)
        delta = self.out_proj(q)

                            
        if U_key_padding_mask is not None:
            valid = (~U_key_padding_mask).unsqueeze(-1).to(dtype=U.dtype)
            pooled = (U * valid).sum(dim=1)
            denom = valid.sum(dim=1).clamp_min(1.0)
            pooled = pooled / denom
        else:
            pooled = U.mean(dim=1)
        gate = torch.sigmoid(self.gate_mlp(pooled)).view(B, 1, 1)
        return delta, gate


                                                                               
                                 
                                                                               


@dataclass
class SpecialTokenIds:
    image_token_id: Optional[int] = None
    image_pad_id: Optional[int] = None
    image_context_id: Optional[int] = None


def _infer_special_token_ids(tokenizer: Any) -> SpecialTokenIds:
    ids = SpecialTokenIds()
    if tokenizer is None:
        return ids

                                
    for attr in ("image_token_id", "image_pad_token_id", "image_pad_id", "image_context_token_id"):
        if hasattr(tokenizer, attr):
            val = getattr(tokenizer, attr)
            if val is None:
                continue
            if "pad" in attr and ids.image_pad_id is None:
                ids.image_pad_id = int(val)
            if attr == "image_token_id" and ids.image_token_id is None:
                ids.image_token_id = int(val)
            if "context" in attr and ids.image_context_id is None:
                ids.image_context_id = int(val)

                                   
    special_strings: List[str] = []
    for attr in ("additional_special_tokens", "all_special_tokens"):
        seq = getattr(tokenizer, attr, None)
        if isinstance(seq, (list, tuple)):
            special_strings.extend([str(s) for s in seq])
    special_strings.extend(
        [
            "<image>",
            "<img>",
            "<image_token>",
            "<image_pad>",
            "<imgpad>",
            "<img_pad>",
            "<im_start>",
            "<im_end>",
        ]
    )

    def _try_id(tok_str: str) -> Optional[int]:
        try:
            tid = tokenizer.convert_tokens_to_ids(tok_str)
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
            if sl in {"<image>", "<img>"} or (("image" in sl or "img" in sl) and "pad" not in sl):
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
    input_ids: Optional[torch.Tensor],
    tokenizer: Any,
    special_ids: SpecialTokenIds,
    *,
    batch_index: int = 0,
) -> List[int]:
                                                                               
    if input_ids is None:
        return []
    ids_1d = input_ids[batch_index] if input_ids.dim() == 2 else input_ids
    ids_list = ids_1d.tolist()

    if special_ids.image_pad_id is not None and special_ids.image_pad_id in ids_list:
        return [i for i, t in enumerate(ids_list) if t == special_ids.image_pad_id]
    if special_ids.image_context_id is not None and special_ids.image_context_id in ids_list:
        return [i for i, t in enumerate(ids_list) if t == special_ids.image_context_id]
    if special_ids.image_token_id is not None and special_ids.image_token_id in ids_list:
        return [i for i, t in enumerate(ids_list) if t == special_ids.image_token_id]

                            
    if tokenizer is not None and hasattr(tokenizer, "convert_ids_to_tokens"):
        try:
            toks = tokenizer.convert_ids_to_tokens(ids_list)
            cand = [
                i
                for i, s in enumerate(toks)
                if isinstance(s, str)
                and ("imgpad" in s.lower() or ("image" in s.lower() and "pad" in s.lower()))
            ]
            if cand:
                return cand
            cand = [
                i
                for i, s in enumerate(toks)
                if isinstance(s, str)
                and (
                    s.lower() in {"<image>", "<img>"}
                    or ("image" in s.lower() or "img" in s.lower())
                )
            ]
            if cand:
                return cand
        except Exception:
            pass

                                                             
    best: List[int] = []
    cur: List[int] = []
    for i, t in enumerate(ids_list):
        if not cur or t == ids_list[cur[-1]]:
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


@torch.no_grad()
def _extract_dummy_image_tokens(
    wrapper: ModelWrapper,
    processor: Any,
    tokenizer: Any,
    special_ids: SpecialTokenIds,
    dummy_imgs: List[Image.Image],
) -> Optional[torch.Tensor]:
       
    if tokenizer is None:
        return None
    if not dummy_imgs:
        return None
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
        if hasattr(tokenizer, "apply_chat_template"):
            chat = tokenizer.apply_chat_template(mini_msgs, tokenize=False, add_generation_prompt=True)
        else:
            chat = "(<image>./</image>)\n "
    elif hasattr(tokenizer, "apply_chat_template"):
        chat = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    elif hasattr(processor, "apply_chat_template"):
        chat = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        chat = "You are a helpful assistant.\n\n[IMAGE]\n "

    if _is_internvl_wrapper(wrapper):
        enc = _internvl_prepare_multimodal_batch(
            wrapper=wrapper,
            tokenizer=tokenizer,
            prompts=[chat],
            images=[primary_img],
            device=wrapper.model.device,
        )
        out = wrapper.model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            pixel_values=enc["pixel_values"],
            image_flags=enc["image_flags"],
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    elif _is_minicpm_wrapper(wrapper):
        if processor is None:
            return None
        enc = _minicpm_prepare_multimodal_batch(
            wrapper=wrapper,
            processor=processor,
            prompts=[chat],
            images=[primary_img],
            device=wrapper.model.device,
        )
        base_embeds = _minicpm_build_inputs_embeds(wrapper, enc).detach()
        pos = _minicpm_positions_from_bounds(enc.get("image_bound", [None])[0])
        if not pos:
            pos = _find_image_positions(enc.get("input_ids"), tokenizer, special_ids, batch_index=0)
        if not pos:
            return None
        return base_embeds[0, pos, :].contiguous().float().cpu()
    else:
        if processor is None:
            return None
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
    hs = _get_hidden_states_tuple(out)
    if not hs:
        return None
    hs0 = hs[0]
    if hs0.dim() != 3:
        return None

    pos = _find_image_positions(enc.get("input_ids"), tokenizer, special_ids, batch_index=0)
    if not pos:
        return None

    img_tokens = hs0[0, pos, :].contiguous().float().cpu()
    return img_tokens


                                                                               
                            
                                                                               


class StopOnRegex(StoppingCriteria):
    def __init__(self, tokenizer, regexes: Sequence[str], check_every: int = 4):
        super().__init__()
        import re as _re

        self.tokenizer = tokenizer
        self.regexes = [_re.compile(r) for r in regexes]
        self.check_every = int(check_every)
        self._step = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        self._step += 1
        if self.check_every > 1 and (self._step % self.check_every != 0):
            return False
        texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        for t in texts:
            for r in self.regexes:
                if r.search(t):
                    return True
        return False


                                                                               
               
                                                                               


def _load_codec_checkpoint(path: str) -> Dict[str, Any]:
    if not path:
        raise RuntimeError("--vision_codec_path is required (inference is checkpoint-only)")
    if not os.path.exists(path):
        raise RuntimeError(f"Codec checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError("Codec checkpoint is not a dict")
    return ckpt


                                                                               
             
                                                                               


class VisionLatentMASMethodCODECNew:
                                                                                  

    def __init__(self, args: argparse.Namespace, models: List[ModelWrapper]):
        self.args = args
        self.models = models
        self.model_names = [m.model_name for m in models]
        self.role_model_map = _load_role_model_map(getattr(args, "role_model_map", ""))

                      
        self.agents = copy.deepcopy(default_agents)
        self.mode = getattr(args, "mode", getattr(args, "prompt", "sequential"))
        self.hier_comm_mode = str(getattr(args, "vision_hier_comm_mode", "chained") or "chained").strip().lower()
        if self.hier_comm_mode not in {"chained", "independent_join"}:
            self.hier_comm_mode = "chained"
        self.hier_join_agg = str(getattr(args, "vision_hier_join_agg", "concat") or "concat").strip().lower()
        if self.hier_join_agg not in {"concat", "mean"}:
            self.hier_join_agg = "concat"

                                    
        raw_latent_steps = _safe_int(getattr(args, "latent_steps", 8), 8)
        self.latent_steps = raw_latent_steps if raw_latent_steps > 0 else 8

                                                                 
                                                                           
                                                                                      
        self.mc_rollouts = max(1, _safe_int(getattr(args, "vision_codec_mc_rollouts", 1), 1))
        self.rollout_mode = str(getattr(args, "vision_codec_rollout_mode", "latent")).lower().strip()
        self.rollout_temperature = float(getattr(args, "vision_codec_rollout_temperature", 1.0))
        self.rollout_top_p = float(getattr(args, "vision_codec_rollout_top_p", 1.0))
        self.rollout_noise_std = float(getattr(args, "vision_codec_rollout_noise_std", 0.0))

        if self.rollout_mode not in {"latent", "token"}:
            self.rollout_mode = "latent"

        if self.rollout_temperature <= 0:
            self.rollout_temperature = 1.0

                                       
        if not (0.0 < self.rollout_top_p <= 1.0):
            self.rollout_top_p = 1.0
        self.max_new_tokens = max(1, _safe_int(getattr(args, "max_new_tokens", 256), 256))
        self.temperature = _safe_float(getattr(args, "temperature", 0.0), 0.0)
        self.top_p = _safe_float(getattr(args, "top_p", 1.0), 1.0)
        do_sample_arg = getattr(args, "do_sample", None)
        if do_sample_arg is None:
            self.do_sample = (self.temperature > 0.0) and (self.top_p > 0.0)
        else:
            self.do_sample = bool(do_sample_arg)
                                                                             
                                                                          
        self._has_minicpm_in_pool = any(_is_minicpm_wrapper(m) for m in models)
        self.total_infer_time_sec = 0.0
        self.total_infer_batches = 0
        self.total_infer_items = 0

                          
        self.codec_ckpt_path = getattr(args, "vision_codec_path", "") or getattr(args, "vision_codec_ckpt", "")
        ckpt = _load_codec_checkpoint(self.codec_ckpt_path)

                                                                          
        cfg = ckpt.get("config", {}) if isinstance(ckpt.get("config", {}), dict) else {}
        self.codec_dim = int(cfg.get("codec_dim", _safe_int(getattr(args, "vision_codec_dim", 256), 256)))
        self.codec_tokens = int(cfg.get("codec_tokens", _safe_int(getattr(args, "vision_codec_tokens", 16), 16)))
        self.codec_img_tokens = int(cfg.get("codec_img_tokens", _safe_int(getattr(args, "vision_codec_img_tokens", 256), 256)))
        self.codec_heads = int(cfg.get("codec_heads", _safe_int(getattr(args, "vision_codec_heads", 8), 8)))
        self.codec_layers = int(cfg.get("codec_layers", _safe_int(getattr(args, "vision_codec_layers", 2), 2)))
        self.codec_dropout = float(cfg.get("codec_dropout", _safe_float(getattr(args, "vision_codec_dropout", 0.0), 0.0)))
        self.codec_gate_init_bias = float(
            cfg.get("codec_gate_init_bias", _safe_float(getattr(args, "vision_codec_gate_init_bias", -4.0), -4.0))
        )
                                              
                                                                   
                                                                                              
        self.codec_decode_chunks = max(1, _safe_int(getattr(args, "vision_codec_decode_chunks", 1), 1))

                                                                                    
        self.dummy_specs = _resolve_dummy_image_specs(self.model_names, args)
        self.dummy_imgs_per_model: List[List[Image.Image]] = []
        for idx, name in enumerate(self.model_names):
            spec = self.dummy_specs[name]
            self.dummy_imgs_per_model.append([_make_dummy_image(spec.size) for _ in range(spec.count)])
            print(
                f"[codec-new][dummy-spec] model[{idx}]={name} count={spec.count} size={spec.size}",
                flush=True,
            )

                                                                  
        self.processors: List[Any] = [None] * len(models)
        self.tokenizers: List[Any] = [None] * len(models)
        self.special_ids: List[SpecialTokenIds] = [SpecialTokenIds() for _ in models]
        self.dummy_img_tokens: List[Optional[torch.Tensor]] = [None] * len(models)                         

                                 
        self.encoders: List[LatentToUniversalEncoder] = []
        self.decoders: List[UniversalToVisionDecoder] = []

                                                
        self.codec_ref_idx = int(ckpt.get("align", {}).get("ref_idx", _safe_int(getattr(args, "vision_codec_ref_idx", 0), 0)))
        self.align_out_W: List[Optional[torch.Tensor]] = [None] * len(models)
        self.align_out_b: List[Optional[torch.Tensor]] = [None] * len(models)
        self.align_in_W: List[Optional[torch.Tensor]] = [None] * len(models)
        self.align_in_b: List[Optional[torch.Tensor]] = [None] * len(models)

        self._init_processors_and_dummy_tokens()
        self._init_codec_modules_and_load(ckpt)

        self._load_alignment_from_ckpt(ckpt)
        self._ensure_identity_alignment()

                               
          
                               

    def _init_processors_and_dummy_tokens(self) -> None:
        token_lens: Dict[str, int] = {}
        for i, wrapper in enumerate(self.models):
            proc = None
            is_internvl = _is_internvl_wrapper(wrapper)
            is_minicpm = _is_minicpm_wrapper(wrapper)
            is_gemma3 = "gemma-3" in str(wrapper.model_name).lower()
            if is_internvl:
                try:
                    if is_gemma3:
                        try:
                            proc = AutoProcessor.from_pretrained(
                                wrapper.model_name, trust_remote_code=True, use_fast=True
                            )
                        except Exception:
                            proc = AutoProcessor.from_pretrained(wrapper.model_name, trust_remote_code=True)
                    else:
                        proc = AutoProcessor.from_pretrained(wrapper.model_name, trust_remote_code=True)
                except Exception:
                    try:
                        proc = AutoTokenizer.from_pretrained(wrapper.model_name, trust_remote_code=True)
                    except Exception:
                        proc = None
            elif is_minicpm:
                try:
                    if is_gemma3:
                        try:
                            proc = AutoProcessor.from_pretrained(
                                wrapper.model_name, trust_remote_code=True, use_fast=True
                            )
                        except Exception:
                            proc = AutoProcessor.from_pretrained(wrapper.model_name, trust_remote_code=True)
                    else:
                        proc = AutoProcessor.from_pretrained(wrapper.model_name, trust_remote_code=True)
                    _patch_minicpm_batchfeature_if_needed(proc)
                except Exception:
                    proc = None
            else:
                try:
                    if is_gemma3:
                        try:
                            proc = AutoProcessor.from_pretrained(
                                wrapper.model_name, trust_remote_code=True, use_fast=True
                            )
                        except Exception:
                            proc = AutoProcessor.from_pretrained(wrapper.model_name, trust_remote_code=True)
                    else:
                        proc = AutoProcessor.from_pretrained(wrapper.model_name, trust_remote_code=True)
                except Exception:
                    proc = None
            self.processors[i] = proc

            tok = _get_tokenizer_like(proc) if proc is not None else None
            if tok is None:
                tok = getattr(wrapper, "tokenizer", None)
            self.tokenizers[i] = tok
            self.special_ids[i] = _infer_special_token_ids(tok)

                                                          
            if tok is not None and (proc is not None or is_internvl):
                dummy_imgs = self.dummy_imgs_per_model[i]
                if (is_internvl or is_minicpm) and len(dummy_imgs) > 1:
                    print(
                        f"[codec-new][warn] {wrapper.model_name} currently uses one dummy image per sample in this path; "
                        f"received {len(dummy_imgs)} and will use the first image only.",
                        flush=True,
                    )
                self.dummy_img_tokens[i] = _extract_dummy_image_tokens(
                    wrapper,
                    proc,
                    tok,
                    self.special_ids[i],
                    dummy_imgs,
                )
                if self.dummy_img_tokens[i] is not None:
                    token_lens[wrapper.model_name] = int(self.dummy_img_tokens[i].shape[0])
                    spec = self.dummy_specs.get(wrapper.model_name, _DummyImageSpec(count=1, size=224))
                    print(
                        f"[codec-new][dummy-spec] model={wrapper.model_name} dummy_image_tokens={token_lens[wrapper.model_name]} "
                        f"(count={spec.count}, size={spec.size})",
                        flush=True,
                    )

        if int(getattr(self.args, "vision_codec_check_dummy_img_tokens", 0)) == 1 and len(token_lens) >= 2:
            lens_set = set(token_lens.values())
            if len(lens_set) != 1:
                msg = (
                    "Dummy image token counts do not match across models: "
                    + ", ".join([f"{k}={v}" for k, v in token_lens.items()])
                )
                if int(getattr(self.args, "vision_codec_require_dummy_img_tokens_match", 0)) == 1:
                    raise RuntimeError(msg)
                print(f"[codec-new][warn] {msg}", flush=True)

    def _init_codec_modules_and_load(self, ckpt: Dict[str, Any]) -> None:
        enc_sd = ckpt.get("encoders", {}) if isinstance(ckpt.get("encoders", {}), dict) else {}
        dec_sd = ckpt.get("decoders", {}) if isinstance(ckpt.get("decoders", {}), dict) else {}

        for idx, wrapper in enumerate(self.models):
            H = _infer_hidden_size(wrapper)
            device = wrapper.model.device

            enc = LatentToUniversalEncoder(
                h_in=H,
                d_univ=self.codec_dim,
                k_univ=self.codec_tokens,
                n_heads=self.codec_heads,
                n_layers=self.codec_layers,
                dropout=self.codec_dropout,
            ).to(device=device, dtype=torch.float32)

            dec = UniversalToVisionDecoder(
                d_univ=self.codec_dim,
                h_out=H,
                k_img=self.codec_img_tokens,
                n_heads=self.codec_heads,
                n_layers=self.codec_layers,
                dropout=self.codec_dropout,
                gate_init_bias=self.codec_gate_init_bias,
            ).to(device=device, dtype=torch.float32)

                                                                
            key_name = wrapper.model_name
            loaded = False
            if key_name in enc_sd and key_name in dec_sd:
                enc.load_state_dict(enc_sd[key_name], strict=True)
                dec.load_state_dict(dec_sd[key_name], strict=True)
                loaded = True
            elif str(idx) in enc_sd and str(idx) in dec_sd:
                enc.load_state_dict(enc_sd[str(idx)], strict=True)
                dec.load_state_dict(dec_sd[str(idx)], strict=True)
                loaded = True

            if not loaded:
                raise RuntimeError(
                    f"Checkpoint is missing encoder/decoder weights for model[{idx}]={wrapper.model_name}. "
                    "Re-train the codec checkpoint with the same model list/order."
                )

            enc.eval()
            dec.eval()
            self.encoders.append(enc)
            self.decoders.append(dec)

    def _load_alignment_from_ckpt(self, ckpt: Dict[str, Any]) -> None:
        align = ckpt.get("align", None)
        if not isinstance(align, dict):
            return

        out_map = align.get("out", {}) if isinstance(align.get("out", {}), dict) else {}
        in_map = align.get("in", {}) if isinstance(align.get("in", {}), dict) else {}

        for i, wrapper in enumerate(self.models):
            key_name = wrapper.model_name

            oi = out_map.get(key_name, out_map.get(str(i)))
            if isinstance(oi, dict) and "W" in oi and "b" in oi:
                self.align_out_W[i] = oi["W"].detach().float().cpu()
                self.align_out_b[i] = oi["b"].detach().float().cpu()

            ii = in_map.get(key_name, in_map.get(str(i)))
            if isinstance(ii, dict) and "W" in ii and "b" in ii:
                self.align_in_W[i] = ii["W"].detach().float().cpu()
                self.align_in_b[i] = ii["b"].detach().float().cpu()

    def _ensure_identity_alignment(self) -> None:
        for i in range(len(self.models)):
            if self.align_out_W[i] is None:
                self.align_out_W[i] = torch.eye(self.codec_dim, dtype=torch.float32)
                self.align_out_b[i] = torch.zeros(self.codec_dim, dtype=torch.float32)
            if self.align_in_W[i] is None:
                self.align_in_W[i] = torch.eye(self.codec_dim, dtype=torch.float32)
                self.align_in_b[i] = torch.zeros(self.codec_dim, dtype=torch.float32)

    def _get_role_model_idx(self, role: str, default_idx: int) -> int:
        if self.role_model_map and role in self.role_model_map:
            idx = int(self.role_model_map[role])
        else:
            idx = int(default_idx)
        if idx < 0 or idx >= len(self.models):
            raise ValueError(f"role_model_map index {idx} out of range for role {role}")
        return idx

                               
                
                               

    def run_batch(self, batch: List[Dict[str, Any]]) -> List[str]:
        infer_start = time.time()
        outputs = [self.run_item(item) for item in batch]
        infer_total = time.time() - infer_start
        per_item = infer_total / max(1, len(batch))
        self.total_infer_time_sec += infer_total
        self.total_infer_batches += 1
        self.total_infer_items += len(batch)
        avg_all = self.total_infer_time_sec / max(1, self.total_infer_items)
        print(
            f"[vision_latent_mas_codec_new] batch_infer_time_sec={infer_total:.4f} "
            f"per_item_sec={per_item:.4f} batch_size={len(batch)} "
            f"total_infer_time_sec={self.total_infer_time_sec:.4f} avg_per_item_sec={avg_all:.4f}"
        )
        return outputs

    def run_item(self, item: Dict[str, Any]) -> str:
        question = item.get("question", item.get("prompt", ""))

                                                                                
        memory: List[torch.Tensor] = []
        sender_bank: List[torch.Tensor] = []

                                                     
        planner_idx = self._get_role_model_idx("planner", 0)
        critic_idx = self._get_role_model_idx("critic", 1 if len(self.models) > 1 else 0)
        refiner_idx = self._get_role_model_idx("refiner", 2 if len(self.models) > 2 else critic_idx)
        judger_idx = self._get_role_model_idx("judger", len(self.models) - 1)

        comm_mode = self.hier_comm_mode if str(self.mode).lower() == "hierarchical" else "chained"
        for model_idx, role in [(planner_idx, "planner"), (critic_idx, "critic"), (refiner_idx, "refiner")]:
                                                            
                                                                                                     
            sender_input_memory = memory if comm_mode == "chained" else []
            lat = self._run_agent_latent(model_idx, role, question, sender_input_memory)
            if lat is not None and lat.numel() > 0:
                U_ref = self._encode_latents_to_universal_ref(model_idx, lat)
                chunks = self._split_memory_chunks(U_ref)
                sender_bank.extend(chunks)
                if comm_mode == "chained":
                    memory.extend(chunks)

        if comm_mode == "chained":
            judger_memory = memory
        else:
            judger_memory = self._build_judger_memory_from_senders(sender_bank)

        final_text = self._run_judger_text(judger_idx, question, judger_memory)
        return final_text

    @staticmethod
    def _split_memory_chunks(U_ref: torch.Tensor) -> List[torch.Tensor]:
                                                     
        if U_ref.dim() == 3 and U_ref.shape[0] > 1:
            return [U_ref[i : i + 1] for i in range(int(U_ref.shape[0]))]
        return [U_ref]

    def _build_judger_memory_from_senders(self, sender_bank: List[torch.Tensor]) -> List[torch.Tensor]:
        if not sender_bank:
            return []
        if self.hier_join_agg == "concat":
            return sender_bank
                                                 
        target_len = max(int(x.shape[1]) for x in sender_bank if x.dim() == 3)
        aligned = []
        for x in sender_bank:
            xx = _resample_tokens(x, target_len) if int(x.shape[1]) != target_len else x
            aligned.append(xx.float())
        mean_mem = torch.cat(aligned, dim=0).mean(dim=0, keepdim=True).detach().float().cpu()
        return [mean_mem]

                               
                    
                               

    def _build_agent_messages(self, role: str, question: str, memory_placeholder: str = "") -> List[Dict[str, Any]]:
        if str(self.mode).lower() == "hierarchical":
            return build_agent_message_hierarchical_latent_mas(
                role=role,
                question=question,
                context=memory_placeholder,
                method="vision_latent_mas",                                
                args=self.args,
            )
        return build_agent_message_sequential_latent_mas(
            role=role,
            question=question,
            context=memory_placeholder,
            method="vision_latent_mas_codec_new",
            args=self.args,
        )

    def _messages_to_mm(self, messages: List[Dict[str, Any]], *, num_images: int = 1) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for m in messages:
            if m.get("role") == "user":
                content = m.get("content", "")
                if isinstance(content, list):
                    out.append(m)
                else:
                    out.append(
                        {
                            "role": "user",
                            "content": _build_mm_user_content(num_images=max(1, int(num_images)), text=str(content)),
                        }
                    )
            else:
                out.append(m)
        return out

                               
                          
                               

    @torch.no_grad()
    def _encode_latents_to_universal_ref(self, model_idx: int, latents: torch.Tensor) -> torch.Tensor:
        wrapper = self.models[model_idx]
        device = wrapper.model.device
        lat = latents.to(device=device, dtype=torch.float32)
        U = self.encoders[model_idx](lat)                   

        W = self.align_out_W[model_idx]
        b = self.align_out_b[model_idx]
        if W is None or b is None:
            return U.detach().float().cpu()
        U_ref = _apply_affine(U, W.to(device=device, dtype=U.dtype), b.to(device=device, dtype=U.dtype))
        return U_ref.detach().float().cpu()

    @torch.no_grad()
    def _decode_universal_ref_to_delta(self, model_idx: int, U_ref: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        wrapper = self.models[model_idx]
        device = wrapper.model.device
        U_ref_dev = U_ref.to(device=device, dtype=torch.float32)

        W = self.align_in_W[model_idx]
        b = self.align_in_b[model_idx]
        if W is not None and b is not None:
            U_i = _apply_affine(U_ref_dev, W.to(device=device, dtype=torch.float32), b.to(device=device, dtype=torch.float32))
        else:
            U_i = U_ref_dev
                                                                
        if self.codec_decode_chunks <= 1 or U_i.shape[1] <= 1:
            delta, gate = self.decoders[model_idx](U_i)
            return delta, gate

                                                                                  
                                                                                
                                                          
        chunk_count = max(1, min(self.codec_decode_chunks, int(U_i.shape[1])))
        if chunk_count <= 1:
            delta, gate = self.decoders[model_idx](U_i)
            return delta, gate

        injected_parts: List[torch.Tensor] = []
        for U_chunk in torch.chunk(U_i, chunks=chunk_count, dim=1):
            if U_chunk.numel() == 0:
                continue
            delta_i, gate_i = self.decoders[model_idx](U_chunk)
            injected_parts.append(gate_i * delta_i)

        if not injected_parts:
            delta, gate = self.decoders[model_idx](U_i)
            return delta, gate

        delta_cat = torch.cat(injected_parts, dim=1)
        gate_one = torch.ones(
            (delta_cat.shape[0], 1, 1),
            device=delta_cat.device,
            dtype=delta_cat.dtype,
        )
        return delta_cat, gate_one

                               
                           
                               

    def _inject_memory_into_mm_inputs(
        self,
        model_idx: int,
        mm: Dict[str, torch.Tensor],
        memory_U_ref: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
           
        wrapper = self.models[model_idx]
        device = wrapper.model.device
        tok = self.tokenizers[model_idx]
        ids = self.special_ids[model_idx]

        if _is_internvl_wrapper(wrapper):
            input_ids = mm.get("input_ids")
            if input_ids is None:
                return {"attention_mask": mm.get("attention_mask")}
            if "pixel_values" not in mm:
                return {"input_ids": input_ids, "attention_mask": mm.get("attention_mask")}

            text_model = _get_text_backbone(wrapper)
            emb_layer = text_model.get_input_embeddings()
            base_embeds = emb_layer(input_ids).detach()
            emb_dtype = base_embeds.dtype

            with torch.no_grad():
                vit_embeds = wrapper.model.extract_feature(mm["pixel_values"]).detach().float()

            delta = gate = None
            if memory_U_ref is not None and memory_U_ref.numel() > 0:
                delta, gate = self._decode_universal_ref_to_delta(model_idx, memory_U_ref)
                if delta.shape[0] != input_ids.shape[0]:
                    if delta.shape[0] == 1:
                        delta = delta.expand(input_ids.shape[0], -1, -1)
                        gate = gate.expand(input_ids.shape[0], -1, -1)
                    else:
                        raise RuntimeError(
                            f"Decoded memory batch ({delta.shape[0]}) does not match input batch ({input_ids.shape[0]})."
                        )

            inputs_embeds = base_embeds.clone()
            for b in range(input_ids.shape[0]):
                pos = _find_image_positions(input_ids, tok, ids, batch_index=b)
                if not pos:
                    continue
                base_img = _resample_tokens(vit_embeds[b], len(pos)).to(device=device, dtype=emb_dtype)
                if delta is None:
                    inputs_embeds[b, pos, :] = base_img
                else:
                    dlt = _resample_tokens(delta[b], len(pos)).to(device=device, dtype=emb_dtype)
                    g = gate[b].to(device=device, dtype=emb_dtype)
                    inputs_embeds[b, pos, :] = base_img + g * dlt

            return {
                "inputs_embeds": inputs_embeds,
                "attention_mask": mm.get("attention_mask"),
                "position_ids": _build_position_ids_from_attention(mm["attention_mask"])
                if mm.get("attention_mask") is not None
                else None,
            }

        if _is_minicpm_wrapper(wrapper):
            input_ids = mm.get("input_ids")
            if input_ids is None:
                return {"attention_mask": mm.get("attention_mask")}
            base_embeds = _minicpm_build_inputs_embeds(wrapper, mm).detach()
            emb_dtype = base_embeds.dtype
            inputs_embeds = base_embeds.clone()

            delta = gate = None
            if memory_U_ref is not None and memory_U_ref.numel() > 0:
                delta, gate = self._decode_universal_ref_to_delta(model_idx, memory_U_ref)
                if delta.shape[0] != input_ids.shape[0]:
                    if delta.shape[0] == 1:
                        delta = delta.expand(input_ids.shape[0], -1, -1)
                        gate = gate.expand(input_ids.shape[0], -1, -1)
                    else:
                        raise RuntimeError(
                            f"Decoded memory batch ({delta.shape[0]}) does not match input batch ({input_ids.shape[0]})."
                        )

            bounds_all = mm.get("image_bound", None)
            for b in range(input_ids.shape[0]):
                bounds = bounds_all[b] if isinstance(bounds_all, list) and b < len(bounds_all) else None
                pos = _minicpm_positions_from_bounds(bounds)
                if not pos:
                    pos = _find_image_positions(input_ids, tok, ids, batch_index=b)
                if not pos:
                    continue
                base_img = base_embeds[b, pos, :]
                if delta is None:
                    inputs_embeds[b, pos, :] = base_img
                else:
                    dlt = _resample_tokens(delta[b], len(pos)).to(device=device, dtype=emb_dtype)
                    g = gate[b].to(device=device, dtype=emb_dtype)
                    inputs_embeds[b, pos, :] = base_img + g * dlt

            return {
                "inputs_embeds": inputs_embeds,
                "attention_mask": mm.get("attention_mask"),
            }

        input_ids = mm.get("input_ids")
        if input_ids is None:
            return {k: v for k, v in mm.items() if k != "pixel_values"}
        B, _L = input_ids.shape

        dummy_tokens = self.dummy_img_tokens[model_idx]
        if dummy_tokens is None or dummy_tokens.numel() == 0:
            return {k: v for k, v in mm.items() if k != "pixel_values"}

        emb_layer = wrapper.model.get_input_embeddings()
        base_embeds = emb_layer(input_ids).detach()           
        emb_dtype = base_embeds.dtype

                                                 
        delta = gate = None
        if memory_U_ref is not None and memory_U_ref.numel() > 0:
            delta, gate = self._decode_universal_ref_to_delta(model_idx, memory_U_ref)

        inputs_embeds = base_embeds.clone()
        for b in range(B):
            pos = _find_image_positions(input_ids, tok, ids, batch_index=b)
            if not pos:
                continue
            base_img = _resample_tokens(dummy_tokens, len(pos)).to(device=device, dtype=emb_dtype)
            if delta is None:
                inputs_embeds[b, pos, :] = base_img
            else:
                dlt = _resample_tokens(delta[b], len(pos)).to(device=device, dtype=emb_dtype)
                g = gate[b].to(device=device, dtype=emb_dtype)
                inputs_embeds[b, pos, :] = base_img + g * dlt

        fwd = {k: v for k, v in mm.items() if k not in ("input_ids", "pixel_values")}
        fwd["inputs_embeds"] = inputs_embeds
        return fwd

                               
                                                
                               

    @torch.no_grad()
    def _generate_latents(
        self,
        wrapper: ModelWrapper,
        *,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple[Optional[Tuple], torch.Tensor]:
                                                                        
        text_model = _get_text_backbone(wrapper)
        if latent_steps <= 0:
            if inputs_embeds is not None:
                B, _L, H = inputs_embeds.shape
                device = inputs_embeds.device
                dtype = inputs_embeds.dtype
            else:
                assert input_ids is not None
                B = int(input_ids.shape[0])
                H = _infer_hidden_size(wrapper)
                device = input_ids.device
                dtype = text_model.get_input_embeddings().weight.dtype
            empty = torch.empty((B, 0, H), device=device, dtype=dtype)
            return past_key_values, empty

        if attention_mask is None:
            if inputs_embeds is not None:
                attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
            else:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
        if position_ids is None and _is_minicpm_wrapper(wrapper):
            position_ids = _build_position_ids_from_attention(attention_mask)

                         
        init_fwd = {
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": True,
            "output_hidden_states": True,
            "return_dict": True,
        }
        if position_ids is not None:
            init_fwd["position_ids"] = position_ids
        if inputs_embeds is None:
            init_fwd["input_ids"] = input_ids
        else:
            init_fwd["inputs_embeds"] = inputs_embeds
        out = text_model(**init_fwd)
        past = out.past_key_values
        hs = _get_hidden_states_tuple(out)
        if not hs:
            raise RuntimeError("Model did not return hidden_states; cannot run latent rollout")
        last_hidden = hs[-1][:, -1, :]

        latents: List[torch.Tensor] = []
        for _ in range(int(latent_steps)):
            latent_vec = wrapper._apply_latent_realignment(last_hidden, text_model)
            latents.append(latent_vec.detach())

            latent_embed = latent_vec.unsqueeze(1)
            past_len = _past_length(past)
            latent_mask = torch.ones((latent_embed.shape[0], past_len + 1), dtype=attention_mask.dtype, device=latent_embed.device)
            step_fwd = {
                "inputs_embeds": latent_embed,
                "attention_mask": latent_mask,
                "past_key_values": past,
                "use_cache": True,
                "output_hidden_states": True,
                "return_dict": True,
            }
            if _is_minicpm_wrapper(wrapper):
                step_fwd["position_ids"] = torch.full(
                    (latent_embed.shape[0], 1),
                    past_len,
                    dtype=torch.long,
                    device=latent_embed.device,
                )
            out2 = text_model(**step_fwd)
            past = out2.past_key_values
            hs2 = _get_hidden_states_tuple(out2)
            if not hs2:
                raise RuntimeError("Model did not return hidden_states during latent rollout")
            last_hidden = hs2[-1][:, -1, :]

        latent_stack = torch.stack(latents, dim=1) if latents else torch.empty((last_hidden.shape[0], 0, last_hidden.shape[-1]), device=last_hidden.device, dtype=last_hidden.dtype)
        return past, latent_stack

    def _sample_from_logits(
        self,
        logits: torch.Tensor,
        *,
        temperature: float,
        top_p: float,
    ) -> torch.Tensor:
           
        if temperature <= 0:
            return torch.argmax(logits, dim=-1)

        if temperature != 1.0:
            logits = logits / float(temperature)

        probs = torch.softmax(logits, dim=-1)

                                  
        if 0 < top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            cdf = torch.cumsum(sorted_probs, dim=-1)
            mask = cdf > float(top_p)
            mask[..., 0] = False
            sorted_probs = sorted_probs.masked_fill(mask, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            sampled_in_sorted = torch.multinomial(sorted_probs, num_samples=1).squeeze(-1)
            return sorted_idx.gather(-1, sampled_in_sorted.unsqueeze(-1)).squeeze(-1)

        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @torch.no_grad()
    def _sample_token_rollouts(
        self,
        wrapper: ModelWrapper,
        *,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        latent_steps: int,
        num_rollouts: int,
        temperature: float,
        top_p: float,
    ) -> torch.Tensor:
           
        text_model = _get_text_backbone(wrapper)
        if num_rollouts <= 1:
                                                       
            _, lat = self._generate_latents(
                wrapper,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                latent_steps=latent_steps,
            )
            return lat

                                                           
        init_fwd = {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "use_cache": True,
            "output_hidden_states": True,
            "return_dict": True,
        }
        if position_ids is not None:
            init_fwd["position_ids"] = position_ids
        out = text_model(**init_fwd)
        hs = _get_hidden_states_tuple(out)
        if not hs:
            raise RuntimeError("Model did not return hidden_states for token rollouts")

        last_logits = out.logits[:, -1, :]
        init_past = out.past_key_values

        B = last_logits.shape[0]
        latents_all: List[torch.Tensor] = []

        for _r in range(int(num_rollouts)):
            past = init_past
            logits = last_logits
            latents: List[torch.Tensor] = []

            for _s in range(int(latent_steps)):
                tok = self._sample_from_logits(logits, temperature=temperature, top_p=top_p)
                tok = tok.to(device=last_logits.device, dtype=torch.long).view(B, 1)

                past_len = _past_length(past)
                step_mask = torch.ones((B, past_len + 1), dtype=attention_mask.dtype, device=last_logits.device)

                step_fwd = {
                    "input_ids": tok,
                    "attention_mask": step_mask,
                    "past_key_values": past,
                    "use_cache": True,
                    "output_hidden_states": True,
                    "return_dict": True,
                }
                if _is_minicpm_wrapper(wrapper):
                    step_fwd["position_ids"] = torch.full(
                        (B, 1),
                        past_len,
                        dtype=torch.long,
                        device=last_logits.device,
                    )
                out2 = text_model(**step_fwd)
                past = out2.past_key_values
                hs2 = _get_hidden_states_tuple(out2)
                if not hs2:
                    raise RuntimeError("Model did not return hidden_states during token rollout")

                h = hs2[-1][:, -1, :]
                latent_vec = wrapper._apply_latent_realignment(h, text_model)
                latents.append(latent_vec.detach())
                logits = out2.logits[:, -1, :]

            latents_all.append(torch.stack(latents, dim=1))

                                     
        lat = torch.stack(latents_all, dim=0).permute(1, 0, 2, 3).contiguous()
        return lat.view(B * int(num_rollouts), int(latent_steps), lat.shape[-1])

    @torch.no_grad()
    def _generate_latents_for_message(
        self,
        wrapper: ModelWrapper,
        *,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        latent_steps: int,
    ) -> torch.Tensor:
           
        if self.mc_rollouts <= 1:
            _, lat = self._generate_latents(
                wrapper,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                latent_steps=latent_steps,
            )
            return lat

        if self.rollout_mode == "token":
            return self._sample_token_rollouts(
                wrapper,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                latent_steps=latent_steps,
                num_rollouts=self.mc_rollouts,
                temperature=self.rollout_temperature,
                top_p=self.rollout_top_p,
            )

                                                                                   
        _, lat = self._generate_latents(
            wrapper,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            latent_steps=latent_steps,
        )
        B = lat.shape[0]
        lat = lat.unsqueeze(1).repeat(1, int(self.mc_rollouts), 1, 1)
        if self.rollout_noise_std > 0:
            lat = lat + torch.randn_like(lat) * float(self.rollout_noise_std)
        return lat.view(B * int(self.mc_rollouts), lat.shape[2], lat.shape[3])

    @torch.no_grad()
    def _generate_latents_mc(
        self,
        wrapper: ModelWrapper,
        *,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        latent_steps: int,
    ) -> torch.Tensor:
                                                                  
        return self._generate_latents_for_message(
            wrapper,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            latent_steps=latent_steps,
        )

                               
                     
                               

    @torch.no_grad()
    def _run_agent_latent(self, model_idx: int, role: str, question: str, memory: List[torch.Tensor]) -> Optional[torch.Tensor]:
        wrapper = self.models[model_idx]
        proc = self.processors[model_idx]
        dummy_imgs = self.dummy_imgs_per_model[model_idx]
        dummy_img_primary = dummy_imgs[0]

        messages = self._build_agent_messages(role, question, memory_placeholder="")
        is_internvl = _is_internvl_wrapper(wrapper)
        is_minicpm = _is_minicpm_wrapper(wrapper)

        if is_internvl:
            tok = self.tokenizers[model_idx] or getattr(wrapper, "tokenizer", None)
            if tok is None:
                raise RuntimeError(f"Tokenizer unavailable for InternVL model: {wrapper.model_name}")

            mm_msgs = self._messages_to_mm(messages, num_images=1)
            if hasattr(tok, "apply_chat_template"):
                chat = tok.apply_chat_template(mm_msgs, tokenize=False, add_generation_prompt=True)
            else:
                chat = "\n".join([str(m.get("content", "")) for m in mm_msgs])

            mm = _internvl_prepare_multimodal_batch(
                wrapper=wrapper,
                tokenizer=tok,
                prompts=[chat],
                images=[dummy_img_primary],
                device=wrapper.model.device,
            )

            memory_U_ref = torch.cat(memory, dim=1) if memory else None
            fwd = self._inject_memory_into_mm_inputs(model_idx, mm, memory_U_ref)
            return self._generate_latents_mc(
                wrapper,
                inputs_embeds=fwd.get("inputs_embeds"),
                attention_mask=fwd.get("attention_mask", mm.get("attention_mask")),
                position_ids=fwd.get("position_ids"),
                latent_steps=self.latent_steps,
            )

        if is_minicpm:
            tok = self.tokenizers[model_idx] or getattr(wrapper, "tokenizer", None)
            if tok is None or proc is None:
                raise RuntimeError(f"Processor/tokenizer unavailable for MiniCPM model: {wrapper.model_name}")

            mini_msgs = _normalize_minicpm_messages(messages)
            if hasattr(tok, "apply_chat_template"):
                chat = tok.apply_chat_template(mini_msgs, tokenize=False, add_generation_prompt=True)
            else:
                chat = "\n".join([str(m.get("content", "")) for m in mini_msgs])

            mm = _minicpm_prepare_multimodal_batch(
                wrapper=wrapper,
                processor=proc,
                prompts=[chat],
                images=[dummy_img_primary],
                device=wrapper.model.device,
            )

            memory_U_ref = torch.cat(memory, dim=1) if memory else None
            fwd = self._inject_memory_into_mm_inputs(model_idx, mm, memory_U_ref)
            return self._generate_latents_mc(
                wrapper,
                inputs_embeds=fwd.get("inputs_embeds"),
                attention_mask=fwd.get("attention_mask", mm.get("attention_mask")),
                position_ids=fwd.get("position_ids"),
                latent_steps=self.latent_steps,
            )

                                            
        if proc is None:
            _, input_ids, attention_mask, _ = wrapper.prepare_chat_batch([messages], add_generation_prompt=True)
            return self._generate_latents_mc(
                wrapper,
                input_ids=input_ids,
                attention_mask=attention_mask,
                latent_steps=self.latent_steps,
            )

        tok = self.tokenizers[model_idx]
        mm_msgs = self._messages_to_mm(messages, num_images=len(dummy_imgs))
        if tok is not None and hasattr(tok, "apply_chat_template"):
            chat = tok.apply_chat_template(mm_msgs, tokenize=False, add_generation_prompt=True)
        elif proc is not None and hasattr(proc, "apply_chat_template"):
            chat = proc.apply_chat_template(mm_msgs, tokenize=False, add_generation_prompt=True)
        else:
                            
            chat = "\n".join([str(m.get("content", "")) for m in mm_msgs])

                        
        mm = _processor_encode_multimodal(proc, texts=[chat], dummy_imgs=dummy_imgs)
        mm = _maybe_to_device(mm, wrapper.model.device)

        memory_U_ref = torch.cat(memory, dim=1) if memory else None       
        fwd = self._inject_memory_into_mm_inputs(model_idx, mm, memory_U_ref)

        return self._generate_latents_mc(
            wrapper,
            inputs_embeds=fwd.get("inputs_embeds"),
            attention_mask=fwd.get("attention_mask", mm.get("attention_mask")),
            position_ids=fwd.get("position_ids"),
            latent_steps=self.latent_steps,
        )

    @torch.no_grad()
    def _run_judger_text(self, model_idx: int, question: str, memory: List[torch.Tensor]) -> str:
        wrapper = self.models[model_idx]
        proc = self.processors[model_idx]
        dummy_imgs = self.dummy_imgs_per_model[model_idx]
        dummy_img_primary = dummy_imgs[0]
        messages = self._build_agent_messages("judger", question, memory_placeholder="")
        is_internvl = _is_internvl_wrapper(wrapper)
        is_minicpm = _is_minicpm_wrapper(wrapper)
        judger_do_sample = self.do_sample and (not is_internvl) and (not self._has_minicpm_in_pool)
        judger_temperature = self.temperature if judger_do_sample else None
        judger_top_p = self.top_p if judger_do_sample else None
        remove_invalid_values = bool(self._has_minicpm_in_pool)

        if is_internvl:
            tok = self.tokenizers[model_idx] or getattr(wrapper, "tokenizer", None)
            if tok is None:
                raise RuntimeError(f"Tokenizer unavailable for InternVL model: {wrapper.model_name}")

            mm_msgs = self._messages_to_mm(messages, num_images=1)
            if hasattr(tok, "apply_chat_template"):
                chat = tok.apply_chat_template(mm_msgs, tokenize=False, add_generation_prompt=True)
            else:
                chat = "\n".join([str(m.get("content", "")) for m in mm_msgs])

            mm = _internvl_prepare_multimodal_batch(
                wrapper=wrapper,
                tokenizer=tok,
                prompts=[chat],
                images=[dummy_img_primary],
                device=wrapper.model.device,
            )

            memory_U_ref = torch.cat(memory, dim=1) if memory else None
            fwd = self._inject_memory_into_mm_inputs(model_idx, mm, memory_U_ref)

            stop_regexes = getattr(self.args, "stop_regexes", None)
            stopping = StoppingCriteriaList([StopOnRegex(tok, stop_regexes)]) if stop_regexes else None
            text_model = _get_text_backbone(wrapper)

            try:
                out = text_model.generate(
                    **fwd,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=judger_do_sample,
                    temperature=judger_temperature,
                    top_p=judger_top_p,
                    stopping_criteria=stopping,
                    pad_token_id=getattr(tok, "pad_token_id", None),
                    remove_invalid_values=remove_invalid_values,
                )
                prompt_len = int(mm.get("attention_mask", torch.ones_like(mm["input_ids"])).sum(dim=1)[0].item())
                gen_ids = _extract_generated_ids(out, prompt_len)
                return tok.decode(gen_ids, skip_special_tokens=True)
            except Exception:
                out = text_model.generate(
                    input_ids=mm["input_ids"],
                    attention_mask=mm.get("attention_mask"),
                    max_new_tokens=self.max_new_tokens,
                    do_sample=judger_do_sample,
                    temperature=judger_temperature,
                    top_p=judger_top_p,
                    stopping_criteria=stopping,
                    pad_token_id=getattr(tok, "pad_token_id", None),
                    remove_invalid_values=remove_invalid_values,
                )
                prompt_len = int(mm.get("attention_mask", torch.ones_like(mm["input_ids"])).sum(dim=1)[0].item())
                gen_ids = _extract_generated_ids(out, prompt_len)
                return tok.decode(gen_ids, skip_special_tokens=True)

        if is_minicpm:
            tok = self.tokenizers[model_idx] or getattr(wrapper, "tokenizer", None)
            if tok is None or proc is None:
                raise RuntimeError(f"Processor/tokenizer unavailable for MiniCPM model: {wrapper.model_name}")

            mini_msgs = _normalize_minicpm_messages(messages)
            if hasattr(tok, "apply_chat_template"):
                chat = tok.apply_chat_template(mini_msgs, tokenize=False, add_generation_prompt=True)
            else:
                chat = "\n".join([str(m.get("content", "")) for m in mini_msgs])

            mm = _minicpm_prepare_multimodal_batch(
                wrapper=wrapper,
                processor=proc,
                prompts=[chat],
                images=[dummy_img_primary],
                device=wrapper.model.device,
            )

            memory_U_ref = torch.cat(memory, dim=1) if memory else None
            fwd = self._inject_memory_into_mm_inputs(model_idx, mm, memory_U_ref)
            stop_regexes = getattr(self.args, "stop_regexes", None)
            stopping = StoppingCriteriaList([StopOnRegex(tok, stop_regexes)]) if stop_regexes else None
            text_model = _get_text_backbone(wrapper)

            out = text_model.generate(
                **fwd,
                max_new_tokens=self.max_new_tokens,
                do_sample=judger_do_sample,
                temperature=judger_temperature,
                top_p=judger_top_p,
                stopping_criteria=stopping,
                pad_token_id=getattr(tok, "pad_token_id", 0),
                remove_invalid_values=remove_invalid_values,
            )
            prompt_len = int(mm.get("attention_mask", torch.ones_like(mm["input_ids"])).sum(dim=1)[0].item())
            gen_ids = _extract_generated_ids(out, prompt_len)
            return tok.decode(gen_ids, skip_special_tokens=True)

                            
        if proc is None:
            _, input_ids, attention_mask, _ = wrapper.prepare_chat_batch([messages], add_generation_prompt=True)
            out_ids = wrapper.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=judger_do_sample,
                temperature=judger_temperature,
                top_p=judger_top_p,
                pad_token_id=getattr(wrapper.tokenizer, "pad_token_id", None),
                remove_invalid_values=remove_invalid_values,
            )
            return wrapper.tokenizer.decode(out_ids[0], skip_special_tokens=True)

        tok = self.tokenizers[model_idx]
        mm_msgs = self._messages_to_mm(messages, num_images=len(dummy_imgs))
        if tok is not None and hasattr(tok, "apply_chat_template"):
            chat = tok.apply_chat_template(mm_msgs, tokenize=False, add_generation_prompt=True)
        elif proc is not None and hasattr(proc, "apply_chat_template"):
            chat = proc.apply_chat_template(mm_msgs, tokenize=False, add_generation_prompt=True)
        else:
            chat = "\n".join([str(m.get("content", "")) for m in mm_msgs])

        mm = _processor_encode_multimodal(proc, texts=[chat], dummy_imgs=dummy_imgs)
        mm = _maybe_to_device(mm, wrapper.model.device)

        memory_U_ref = torch.cat(memory, dim=1) if memory else None
        fwd = self._inject_memory_into_mm_inputs(model_idx, mm, memory_U_ref)

        stop_regexes = getattr(self.args, "stop_regexes", None)
        stopping = StoppingCriteriaList([StopOnRegex(tok, stop_regexes)]) if (stop_regexes and tok is not None) else None

                                                                    
        try:
            out = wrapper.model.generate(
                **fwd,
                max_new_tokens=self.max_new_tokens,
                do_sample=judger_do_sample,
                temperature=judger_temperature,
                top_p=judger_top_p,
                stopping_criteria=stopping,
                pad_token_id=getattr(tok, "pad_token_id", None) if tok is not None else None,
                remove_invalid_values=remove_invalid_values,
            )
            prompt_len = int(mm.get("attention_mask", torch.ones_like(mm["input_ids"])).sum(dim=1)[0].item())
            gen_ids = _extract_generated_ids(out, prompt_len)
            return tok.decode(gen_ids, skip_special_tokens=True) if tok is not None else str(gen_ids)
        except Exception:
            out = wrapper.model.generate(
                input_ids=mm["input_ids"],
                attention_mask=mm.get("attention_mask"),
                max_new_tokens=self.max_new_tokens,
                do_sample=judger_do_sample,
                temperature=judger_temperature,
                top_p=judger_top_p,
                stopping_criteria=stopping,
                pad_token_id=getattr(tok, "pad_token_id", None) if tok is not None else None,
                remove_invalid_values=remove_invalid_values,
            )
            prompt_len = int(mm.get("attention_mask", torch.ones_like(mm["input_ids"])).sum(dim=1)[0].item())
            gen_ids = _extract_generated_ids(out, prompt_len)
            return tok.decode(gen_ids, skip_special_tokens=True) if tok is not None else str(gen_ids)
