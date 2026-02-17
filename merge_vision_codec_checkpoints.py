#!/usr/bin/env python3
   

from __future__ import annotations

import argparse
import copy
import multiprocessing as mp
import os
import queue as queue_mod
import random
import shutil
import tempfile
import traceback
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

from models import ModelWrapper
from train_vision_latent_mas_codec_new import (
    _collect_anchor_U,
    _load_anchor_texts,
    _parse_model_list,
    _ridge_eval_mse,
    _ridge_fit,
)


_ARCH_KEYS = (
    "codec_dim",
    "codec_tokens",
    "codec_img_tokens",
    "codec_heads",
    "codec_layers",
    "codec_dropout",
    "codec_gate_init_bias",
)

_INT_ARCH_KEYS = (
    "codec_dim",
    "codec_tokens",
    "codec_img_tokens",
    "codec_heads",
    "codec_layers",
)

_FLOAT_ARCH_KEYS = (
    "codec_dropout",
    "codec_gate_init_bias",
)


def _load_codec_checkpoint(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Codec checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Codec checkpoint is not a dict: {path}")
    return ckpt


def _normalize_arch_config(cfg: Dict, path: str) -> Dict[str, float]:
    if not isinstance(cfg, dict):
        raise RuntimeError(f"Checkpoint config is not a dict: {path}")
    missing = [k for k in _ARCH_KEYS if k not in cfg]
    if missing:
        raise RuntimeError(f"Checkpoint missing config keys {missing}: {path}")

    out: Dict[str, float] = {}
    for k in _INT_ARCH_KEYS:
        out[k] = int(cfg[k])
    for k in _FLOAT_ARCH_KEYS:
        out[k] = float(cfg[k])
    return out


def _arch_equal(a: Dict[str, float], b: Dict[str, float], atol: float = 1e-8) -> bool:
    for k in _INT_ARCH_KEYS:
        if int(a[k]) != int(b[k]):
            return False
    for k in _FLOAT_ARCH_KEYS:
        if abs(float(a[k]) - float(b[k])) > atol:
            return False
    return True


def _assert_compatible_arch(all_arch: Sequence[Tuple[str, Dict[str, float]]]) -> Dict[str, float]:
    if not all_arch:
        raise RuntimeError("No checkpoint architecture to compare")
    ref_path, ref_arch = all_arch[0]
    for path, arch in all_arch[1:]:
        if not _arch_equal(ref_arch, arch):
            raise RuntimeError(
                "Codec architecture mismatch.\n"
                f"Reference ({ref_path}): {ref_arch}\n"
                f"Current   ({path}): {arch}"
            )
    return ref_arch


def _get_model_list_from_ckpt(ckpt: Dict, path: str) -> List[str]:
    models = ckpt.get("models", None)
    if not isinstance(models, list) or not all(isinstance(x, str) for x in models):
        raise RuntimeError(
            f"Checkpoint missing valid `models` list: {path}. "
            "Use checkpoints produced by train_vision_latent_mas_codec_new.py."
        )
    return [m.strip() for m in models if m and m.strip()]


def _lookup_sd(sd_map: Dict, model_name: str, model_idx: int) -> Optional[Dict[str, torch.Tensor]]:
    if not isinstance(sd_map, dict):
        return None
    obj = sd_map.get(model_name, sd_map.get(str(model_idx)))
    return obj if isinstance(obj, dict) else None


def _build_merge_namespace(
    *,
    arch_cfg: Dict[str, float],
    latent_steps: int,
    train_batch_size: int,
    align_batch_size: int,
    align_loss_eval_max_rows: int,
    ridge: float,
    latent_space_realign: bool,
) -> argparse.Namespace:
    ns = argparse.Namespace()
    ns.latent_steps = int(latent_steps)
    ns.vision_codec_dim = int(arch_cfg["codec_dim"])
    ns.vision_codec_tokens = int(arch_cfg["codec_tokens"])
    ns.vision_codec_img_tokens = int(arch_cfg["codec_img_tokens"])
    ns.vision_codec_heads = int(arch_cfg["codec_heads"])
    ns.vision_codec_layers = int(arch_cfg["codec_layers"])
    ns.vision_codec_dropout = float(arch_cfg["codec_dropout"])
    ns.vision_codec_gate_init_bias = float(arch_cfg["codec_gate_init_bias"])
    ns.vision_codec_train_batch_size = int(train_batch_size)
    ns.vision_codec_align_batch_size = int(align_batch_size)
    ns.vision_codec_align_loss_eval_max_rows = int(align_loss_eval_max_rows)
    ns.vision_codec_ridge = float(ridge)
    ns.latent_space_realign = bool(latent_space_realign)
    return ns


def _parse_align_devices(spec: str) -> List[str]:
    out: List[str] = []
    for raw in str(spec or "").split(","):
        tok = raw.strip().lower()
        if not tok:
            continue
        if tok == "cpu":
            out.append("cpu")
            continue
        if tok.startswith("cuda:"):
            idx_s = tok.split(":", 1)[1].strip()
            if not idx_s.isdigit():
                raise ValueError(f"Invalid CUDA device spec: {raw}")
            out.append(f"cuda:{int(idx_s)}")
            continue
        if tok.isdigit():
            out.append(f"cuda:{int(tok)}")
            continue
        raise ValueError(f"Unsupported device token in --vision_codec_align_devices: {raw}")
    return out


def _collect_anchor_u_worker(
    *,
    model_name: str,
    enc_sd: Dict[str, torch.Tensor],
    anchor_texts: List[str],
    cfg_state: Dict,
    device_str: str,
    out_path: str,
    result_queue,
) -> None:
    try:
        cfg = argparse.Namespace(**cfg_state)
        device = torch.device(device_str)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        wrapper = ModelWrapper(model_name, device, use_vllm=False, args=cfg)
        try:
            u = _collect_anchor_U(
                wrapper=wrapper,
                enc_sd=enc_sd,
                anchor_texts=anchor_texts,
                cfg=cfg,
            )
        finally:
            del wrapper
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        torch.save(u, out_path)
        result_queue.put({"ok": True, "model_name": model_name, "out_path": out_path, "error": ""})
    except Exception:
        result_queue.put(
            {
                "ok": False,
                "model_name": model_name,
                "out_path": out_path,
                "error": traceback.format_exc(),
            }
        )


def _collect_anchor_u_parallel(
    *,
    model_names: Sequence[str],
    encoders: Dict[str, Dict[str, torch.Tensor]],
    anchor_texts: List[str],
    cfg: argparse.Namespace,
    devices: Sequence[str],
) -> Dict[str, torch.Tensor]:
    if len(devices) < len(model_names):
        raise ValueError(
            f"Need at least one align device per model for parallel collection: "
            f"models={len(model_names)} devices={len(devices)}"
        )

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    tmpdir = tempfile.mkdtemp(prefix="merge_u_collect_")
    cfg_state = vars(cfg).copy()
    procs = []
    chosen_devices = list(devices[: len(model_names)])

    try:
        for i, model_name in enumerate(model_names):
            out_path = os.path.join(tmpdir, f"U_{i:03d}.pt")
            device_str = chosen_devices[i]
            print(f"[merge][parallel] launch model[{i}]={model_name} on {device_str}", flush=True)
            p = ctx.Process(
                target=_collect_anchor_u_worker,
                kwargs={
                    "model_name": model_name,
                    "enc_sd": encoders[model_name],
                    "anchor_texts": anchor_texts,
                    "cfg_state": cfg_state,
                    "device_str": device_str,
                    "out_path": out_path,
                    "result_queue": result_queue,
                },
            )
            p.start()
            procs.append((model_name, device_str, out_path, p))

        results: Dict[str, Dict] = {}
        expected = len(model_names)
        while len(results) < expected:
            try:
                msg = result_queue.get(timeout=5.0)
            except queue_mod.Empty:
                if all(not p.is_alive() for _, _, _, p in procs):
                    break
                continue
            if not isinstance(msg, dict):
                continue
            model_name = str(msg.get("model_name", ""))
            if model_name:
                results[model_name] = msg
                status = "ok" if bool(msg.get("ok", False)) else "error"
                print(f"[merge][parallel] finished {model_name}: {status}", flush=True)

        for _, _, _, p in procs:
            p.join()

        failures: List[str] = []
        for model_name, _device_str, _out_path, p in procs:
            msg = results.get(model_name)
            if msg is None:
                failures.append(
                    f"model={model_name}: no worker result (exitcode={p.exitcode})"
                )
                continue
            if not bool(msg.get("ok", False)):
                failures.append(f"model={model_name} failed:\n{msg.get('error', '')}")
                continue
            if p.exitcode not in (0, None):
                failures.append(
                    f"model={model_name}: worker exitcode={p.exitcode} (result marked ok)"
                )
        if failures:
            raise RuntimeError(
                "Parallel align-U collection failed:\n" + "\n\n".join(failures)
            )

        u_by_name: Dict[str, torch.Tensor] = {}
        for model_name in model_names:
            out_path = str(results[model_name]["out_path"])
            u_by_name[model_name] = torch.load(out_path, map_location="cpu")
        return u_by_name
    finally:
        for _model_name, _device_str, _out_path, p in procs:
            if p.is_alive():
                p.terminate()
                p.join(timeout=2.0)
        shutil.rmtree(tmpdir, ignore_errors=True)


def _fit_alignment(
    *,
    model_names: Sequence[str],
    encoders: Dict[str, Dict[str, torch.Tensor]],
    anchor_texts: List[str],
    cfg: argparse.Namespace,
    ref_model_name: str,
    devices: Sequence[str],
) -> Dict[str, Dict]:
    if ref_model_name not in model_names:
        raise ValueError(f"Reference model not in merged model list: {ref_model_name}")

    u_by_name: Dict[str, torch.Tensor] = {}
    use_parallel = len(model_names) > 1 and len(devices) > 1
    if use_parallel:
        print(
            f"[merge] Collecting anchor U in parallel across devices: {list(devices[:len(model_names)])}",
            flush=True,
        )
        u_by_name = _collect_anchor_u_parallel(
            model_names=model_names,
            encoders=encoders,
            anchor_texts=anchor_texts,
            cfg=cfg,
            devices=devices,
        )
    else:
        device = torch.device(devices[0])
        for i, name in enumerate(tqdm(model_names, desc="[merge] collect U", unit="model")):
            print(f"[merge] Collecting anchor U for model[{i}]={name} on {device}")
            wrapper = ModelWrapper(name, device, use_vllm=False, args=cfg)
            try:
                u_by_name[name] = _collect_anchor_U(
                    wrapper=wrapper,
                    enc_sd=encoders[name],
                    anchor_texts=anchor_texts,
                    cfg=cfg,
                )
            finally:
                del wrapper
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    x_ref = u_by_name[ref_model_name].reshape(-1, int(cfg.vision_codec_dim))
    out_map: Dict[str, Dict[str, torch.Tensor]] = {}
    in_map: Dict[str, Dict[str, torch.Tensor]] = {}
    loss_eval_rows = int(getattr(cfg, "vision_codec_align_loss_eval_max_rows", 16384))
    for name in tqdm(model_names, desc="[merge] ridge fit", unit="model"):
        x = u_by_name[name].reshape(-1, int(cfg.vision_codec_dim))
        w_out, b_out = _ridge_fit(x, x_ref, ridge=float(cfg.vision_codec_ridge))
        w_in, b_in = _ridge_fit(x_ref, x, ridge=float(cfg.vision_codec_ridge))
        out_map[name] = {"W": w_out, "b": b_out}
        in_map[name] = {"W": w_in, "b": b_in}
        mse_out = _ridge_eval_mse(x, x_ref, w_out, b_out, max_rows=loss_eval_rows)
        mse_in = _ridge_eval_mse(x_ref, x, w_in, b_in, max_rows=loss_eval_rows)
        eval_rows = min(max(1, loss_eval_rows), int(x.shape[0]))
        print(
            f"[merge][align-U][loss] model={name} mse_out={mse_out:.6f} "
            f"mse_in={mse_in:.6f} eval_rows={eval_rows}",
            flush=True,
        )

    return {
        "ref_idx": int(model_names.index(ref_model_name)),
        "ref_model_name": ref_model_name,
        "out": out_map,
        "in": in_map,
    }


def _tensor_nonfinite_counts(t: torch.Tensor) -> Tuple[int, int]:
    if not torch.is_tensor(t):
        return 0, 0
    nan = int(torch.isnan(t).sum().item())
    inf = int(torch.isinf(t).sum().item())
    return nan, inf


def _inspect_affine_nonfinite(align: Dict, *, label: str) -> Tuple[int, int]:
                                                                          
    if not isinstance(align, dict):
        print(f"[merge][warn] {label}: no align dict to inspect")
        return 0, 0

    total_nan = 0
    total_inf = 0
    for direction in ("out", "in"):
        section = align.get(direction, {})
        if not isinstance(section, dict):
            continue
        for model_name, affine in section.items():
            if not isinstance(affine, dict):
                continue
            for key in ("W", "b"):
                t = affine.get(key, None)
                if not torch.is_tensor(t):
                    continue
                nan, inf = _tensor_nonfinite_counts(t)
                total_nan += nan
                total_inf += inf
                if nan or inf:
                    print(
                        f"[merge][warn] {label}: align[{direction}][{model_name}][{key}] "
                        f"has nan={nan} inf={inf}"
                    )

    if total_nan == 0 and total_inf == 0:
        print(f"[merge] {label}: affine mapping is finite")
    else:
        print(f"[merge][warn] {label}: affine mapping has total nan={total_nan}, inf={total_inf}")
    return total_nan, total_inf


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--codec_paths", nargs="+", required=True, help="Input codec checkpoints to merge")
    p.add_argument("--vision_codec_path", type=str, required=True, help="Output merged checkpoint path")
    p.add_argument(
        "--agent_model_names",
        type=str,
        default="",
        help="Optional comma-separated final model order; defaults to first-seen order from inputs",
    )
    p.add_argument(
        "--duplicate_policy",
        type=str,
        default="last",
        choices=["first", "last", "error"],
        help="How to handle duplicate model codecs across input checkpoints",
    )
    p.add_argument(
        "--ref_model_name",
        type=str,
        default="",
        help="Reference model for alignment. Defaults to first model in final merged order",
    )
    p.add_argument("--vision_codec_anchor_texts_path", type=str, default="")
    p.add_argument(
        "--vision_codec_align_max_anchors",
        type=int,
        default=0,
        help="If >0, shuffle anchor texts and keep only this many for merge alignment.",
    )
    p.add_argument(
        "--vision_codec_align_seed",
        type=int,
        default=42,
        help="Random seed used for shuffled anchor subsampling in merge alignment.",
    )
    p.add_argument("--vision_codec_ridge", type=float, default=1e-3)
    p.add_argument("--latent_steps", type=int, default=1024)
    p.add_argument("--vision_codec_train_batch_size", type=int, default=8)
    p.add_argument(
        "--vision_codec_align_batch_size",
        type=int,
        default=0,
        help=(
            "If >0, use this batch size for align-U anchor collection during merge. "
            "If 0, fallback logic in _collect_anchor_U is used."
        ),
    )
    p.add_argument(
        "--vision_codec_align_loss_eval_max_rows",
        type=int,
        default=16384,
        help="Max rows used to estimate ridge fit MSE during merge alignment logging.",
    )
    p.add_argument(
        "--vision_codec_align_devices",
        type=str,
        default="",
        help=(
            "Comma-separated devices used for align-U collection, e.g. '0,1' or 'cuda:0,cuda:1'. "
            "When multiple are provided, merge collects U in parallel with one model per device."
        ),
    )
    p.add_argument("--latent_space_realign", action="store_true")
    p.add_argument(
        "--inspect_affine_nonfinite",
        action="store_true",
        help="Inspect source/merged align affine tensors for NaN/Inf.",
    )
    p.add_argument(
        "--fail_on_affine_nonfinite",
        action="store_true",
        help="Fail when inspected align affine tensors contain NaN/Inf.",
    )
    args = p.parse_args()

    codec_paths = [os.path.abspath(x) for x in args.codec_paths]
    if len(codec_paths) < 2:
        raise ValueError("Provide at least 2 checkpoints in --codec_paths")

    print("[merge] Loading checkpoints...")
    loaded = []
    arch_list: List[Tuple[str, Dict[str, float]]] = []
    for path in tqdm(codec_paths, desc="[merge] load ckpts", unit="ckpt"):
        ckpt = _load_codec_checkpoint(path)
        arch = _normalize_arch_config(ckpt.get("config", {}), path)
        models = _get_model_list_from_ckpt(ckpt, path)
        enc_sd = ckpt.get("encoders", {})
        dec_sd = ckpt.get("decoders", {})
        if not isinstance(enc_sd, dict) or not isinstance(dec_sd, dict):
            raise RuntimeError(f"Checkpoint missing dict encoders/decoders: {path}")
        if args.inspect_affine_nonfinite:
            nan, inf = _inspect_affine_nonfinite(ckpt.get("align", {}), label=f"source:{path}")
            if args.fail_on_affine_nonfinite and (nan > 0 or inf > 0):
                raise RuntimeError(
                    f"Non-finite affine mapping found in source checkpoint {path}: nan={nan}, inf={inf}"
                )
        loaded.append((path, ckpt, models, enc_sd, dec_sd))
        arch_list.append((path, arch))

    merged_arch = _assert_compatible_arch(arch_list)
    print(f"[merge] Compatible architecture confirmed: {merged_arch}")

    discovered_order: List[str] = []
    encoders: Dict[str, Dict[str, torch.Tensor]] = {}
    decoders: Dict[str, Dict[str, torch.Tensor]] = {}
    source_by_model: Dict[str, str] = {}

    for path, _ckpt, models, enc_sd, dec_sd in tqdm(loaded, desc="[merge] merge models", unit="ckpt"):
        for idx, model_name in enumerate(tqdm(models, desc=f"[merge] {os.path.basename(path)}", unit="model", leave=False)):
            if model_name not in discovered_order:
                discovered_order.append(model_name)
            enc = _lookup_sd(enc_sd, model_name, idx)
            dec = _lookup_sd(dec_sd, model_name, idx)
            if enc is None or dec is None:
                print(f"[merge][warn] Missing encoder/decoder for {model_name} in {path}; skipping")
                continue

            if model_name in encoders:
                if args.duplicate_policy == "error":
                    raise RuntimeError(f"Duplicate model codec found for {model_name}: {source_by_model[model_name]} and {path}")
                if args.duplicate_policy == "first":
                    continue

            encoders[model_name] = copy.deepcopy(enc)
            decoders[model_name] = copy.deepcopy(dec)
            source_by_model[model_name] = path

    if args.agent_model_names:
        model_names = _parse_model_list(args.agent_model_names)
    else:
        model_names = discovered_order

    if not model_names:
        raise RuntimeError("No models found for merge")

    missing = [m for m in model_names if (m not in encoders or m not in decoders)]
    if missing:
        raise RuntimeError(
            "Merged checkpoint would be incomplete. Missing per-model codec weights for: "
            f"{missing}"
        )

    ref_model_name = args.ref_model_name.strip() if args.ref_model_name else model_names[0]
    if ref_model_name not in model_names:
        raise RuntimeError(f"--ref_model_name '{ref_model_name}' is not in merged model list")

    cfg = _build_merge_namespace(
        arch_cfg=merged_arch,
        latent_steps=int(args.latent_steps),
        train_batch_size=int(args.vision_codec_train_batch_size),
        align_batch_size=int(args.vision_codec_align_batch_size),
        align_loss_eval_max_rows=int(args.vision_codec_align_loss_eval_max_rows),
        ridge=float(args.vision_codec_ridge),
        latent_space_realign=bool(args.latent_space_realign),
    )

    anchor_texts = _load_anchor_texts(args.vision_codec_anchor_texts_path)
    max_align_anchors = int(args.vision_codec_align_max_anchors)
    if max_align_anchors > 0 and len(anchor_texts) > max_align_anchors:
        rng = random.Random(int(args.vision_codec_align_seed))
        idx = list(range(len(anchor_texts)))
        rng.shuffle(idx)
        sel = sorted(idx[:max_align_anchors])
        anchor_texts = [anchor_texts[i] for i in sel]
        print(
            f"[merge] Shuffled+sampled anchors: {len(anchor_texts)}/{len(idx)} "
            f"(seed={int(args.vision_codec_align_seed)})"
        )
    align_devices = _parse_align_devices(args.vision_codec_align_devices)
    if not align_devices:
        align_devices = ["cuda:0"] if torch.cuda.is_available() else ["cpu"]
    if any(d.startswith("cuda:") for d in align_devices) and not torch.cuda.is_available():
        raise RuntimeError(
            f"--vision_codec_align_devices requested CUDA devices {align_devices}, but CUDA is unavailable."
        )
    print(
        f"[merge] Fitting alignment on {len(anchor_texts)} anchor texts "
        f"(align_devices={align_devices})"
    )
    align = _fit_alignment(
        model_names=model_names,
        encoders=encoders,
        anchor_texts=anchor_texts,
        cfg=cfg,
        ref_model_name=ref_model_name,
        devices=align_devices,
    )
    if args.inspect_affine_nonfinite:
        nan, inf = _inspect_affine_nonfinite(align, label="merged")
        if args.fail_on_affine_nonfinite and (nan > 0 or inf > 0):
            raise RuntimeError(f"Non-finite affine mapping found in merged checkpoint: nan={nan}, inf={inf}")

    merged_ckpt = {
        "version": 2,
        "models": list(model_names),
        "ref_model_name": ref_model_name,
        "config": {
            "codec_dim": int(merged_arch["codec_dim"]),
            "codec_tokens": int(merged_arch["codec_tokens"]),
            "codec_img_tokens": int(merged_arch["codec_img_tokens"]),
            "codec_heads": int(merged_arch["codec_heads"]),
            "codec_layers": int(merged_arch["codec_layers"]),
            "codec_dropout": float(merged_arch["codec_dropout"]),
            "codec_gate_init_bias": float(merged_arch["codec_gate_init_bias"]),
        },
        "encoders": {m: encoders[m] for m in model_names},
        "decoders": {m: decoders[m] for m in model_names},
        "align": align,
        "merge_info": {
            "source_checkpoints": codec_paths,
            "duplicate_policy": args.duplicate_policy,
            "source_by_model": source_by_model,
            "anchor_text_count": len(anchor_texts),
        },
    }

    out_path = os.path.abspath(args.vision_codec_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(merged_ckpt, out_path)
    print(f"[merge] Saved merged checkpoint to: {out_path}")
    print(f"[merge] Models: {model_names}")
    print(f"[merge] Reference model: {ref_model_name}")


if __name__ == "__main__":
    main()
