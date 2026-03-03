"""vision_latent_mas_ocr.py

OCR-image multi-agent communication method.

This is a *drop-in* method intended for the same evaluation harness as other methods in this
repository (see run.py). It renders inter-agent notes as images and lets the receiver VLM read
them through the vision pathway (OCR-like behavior in the model).

Pragmatic design (so it works with off-the-shelf VLMs):
  - Each agent produces a short "note" (text) locally.
  - We render the note(s) into a single image (PIL) using a deterministic layout.
  - The next agent receives that image as context (plus the original question as normal text).

So: agents do not exchange text tokens; they exchange an OCR note image. The receiver ingests it through
its vision encoder, meaning the communicated information is carried by **visual latent tokens**.

This file intentionally avoids any heavyweight training.

References:
  - Render-of-Thought: Rendering Textual Chain-of-Thought as Images for Visual Latent Reasoning.

"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont

# transformers is required for actual multimodal generation. We keep the import
# optional so this file can still be imported (e.g., for unit-testing the PIL
# rendering helpers) in minimal environments.
try:  # pragma: no cover
    from transformers import AutoProcessor
except Exception:  # pragma: no cover
    AutoProcessor = None  # type: ignore

# default_agents exists in the full repo, but keep a small fallback list so this
# file can be used standalone during development.
try:
    from . import default_agents  # type: ignore
    _DEFAULT_AGENTS = copy.deepcopy(default_agents)
except Exception:  # pragma: no cover
    _DEFAULT_AGENTS = [
        {"name": "Planner", "role": "planner"},
        {"name": "Critic", "role": "critic"},
        {"name": "Refiner", "role": "refiner"},
        {"name": "Judger", "role": "judger"},
    ]
from prompts import (
    build_agent_message_hierarchical_latent_mas,
    build_agent_message_sequential_latent_mas,
)


# -----------------------------
# Small helpers
# -----------------------------


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
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _make_blank_image(size: int = 384) -> Image.Image:
    return Image.new("RGB", (size, size), color=(255, 255, 255))


def _load_font(font_size: int) -> ImageFont.FreeTypeFont:
    """Best-effort font loader with safe fallback.

    We prefer a mono font for consistent wrapping. If DejaVuSansMono isn't available,
    fall back to PIL's default bitmap font.
    """
    try:
        return ImageFont.truetype("DejaVuSansMono.ttf", font_size)
    except Exception:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", font_size)
        except Exception:
            return ImageFont.load_default()


def _prepare_render_text(text: str, max_chars: int) -> str:
    t = "" if text is None else str(text)
    if max_chars is not None and max_chars > 0 and len(t) > max_chars:
        t = t[: max_chars - 1] + "…"
    return t.replace("\t", "  ")


def _line_height_for_font(font: ImageFont.FreeTypeFont, font_size: int) -> int:
    try:
        bbox = font.getbbox("Ag")
        return int((bbox[3] - bbox[1]) * 1.25)
    except Exception:
        return int(font_size * 1.3)


def _wrap_text_for_font(
    text: str,
    *,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.FreeTypeFont,
    font_size: int,
    usable_w: int,
    max_chars_per_line: int,
) -> List[str]:
    hard_char_cap = int(max_chars_per_line) if (max_chars_per_line is not None and int(max_chars_per_line) > 0) else 0

    def _text_px_width(s: str) -> int:
        try:
            return int(draw.textlength(s, font=font))
        except Exception:
            try:
                b = draw.textbbox((0, 0), s, font=font)
                return max(0, int(b[2] - b[0]))
            except Exception:
                return max(0, len(s) * max(1, int(font_size // 2)))

    def _fits_line(s: str) -> bool:
        if hard_char_cap > 0 and len(s) > hard_char_cap:
            return False
        return _text_px_width(s) <= usable_w

    def _wrap_paragraph_auto(para: str) -> List[str]:
        if not para:
            return [""]
        words = para.split()
        if not words:
            return [""]

        out: List[str] = []
        cur = ""
        for w in words:
            cand = w if not cur else f"{cur} {w}"
            if _fits_line(cand):
                cur = cand
                continue

            if cur:
                out.append(cur)

            if _fits_line(w):
                cur = w
                continue

            # Long token: split by characters so no information is dropped.
            chunk = ""
            for ch in w:
                cand2 = chunk + ch
                if _fits_line(cand2):
                    chunk = cand2
                else:
                    if chunk:
                        out.append(chunk)
                    chunk = ch
            cur = chunk

        if cur:
            out.append(cur)
        return out or [""]

    wrapped_lines: List[str] = []
    for para in text.splitlines() or [""]:
        if not para.strip():
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(_wrap_paragraph_auto(para))
    return wrapped_lines


def estimate_max_ocr_font_size(
    text: str,
    *,
    size: int,
    pad: int,
    max_chars: int,
    max_chars_per_line: int,
    min_font_size: int,
    max_font_size: int,
) -> int:
    size_i = int(size)
    pad_i = int(pad)
    usable_w = size_i - 2 * pad_i
    usable_h = size_i - 2 * pad_i

    render_text = _prepare_render_text(text, max_chars=max_chars)
    min_fs = max(6, int(min_font_size))
    max_fs = max(min_fs, int(max_font_size))

    # Approximation first, then fit-check search for exact max fitting font.
    est = int(((max(1, usable_w) * max(1, usable_h)) / max(1, len(render_text))) ** 0.5)
    lo = max(min_fs, min(est, max_fs))
    hi = max_fs
    probe = Image.new("RGB", (size_i, size_i), color=(255, 255, 255))
    probe_draw = ImageDraw.Draw(probe)

    def _fits(fs: int) -> bool:
        f = _load_font(int(fs))
        wrapped = _wrap_text_for_font(
            render_text,
            draw=probe_draw,
            font=f,
            font_size=int(fs),
            usable_w=usable_w,
            max_chars_per_line=max_chars_per_line,
        )
        line_h = _line_height_for_font(f, int(fs))
        max_lines = max(1, usable_h // max(1, line_h))
        return len(wrapped) <= max_lines

    # Ensure search lower bound is feasible.
    if not _fits(lo):
        lo2, hi2 = min_fs, max(lo - 1, min_fs)
        best = min_fs
        while lo2 <= hi2:
            mid = (lo2 + hi2) // 2
            if _fits(mid):
                best = mid
                lo2 = mid + 1
            else:
                hi2 = mid - 1
        return best

    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        if _fits(mid):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def render_text_as_ocr_image(
    text: str,
    *,
    size: int = 384,
    font_size: int = 16,
    pad: int = 14,
    max_chars: int = 1400,
    max_chars_per_line: int = 0,
    auto_font_size: bool = False,
    min_font_size: int = 8,
    max_font_size: int = 16,
) -> Image.Image:
    """Render text into a square OCR-note image.

    The goal is robustness (no crashes), not artistic typography.
    """
    render_text = _prepare_render_text(text, max_chars=max_chars)
    size_i = int(size)
    pad_i = int(pad)
    usable_w = size_i - 2 * pad_i
    usable_h = size_i - 2 * pad_i

    chosen_font_size = int(font_size)
    if auto_font_size:
        chosen_font_size = estimate_max_ocr_font_size(
            render_text,
            size=size_i,
            pad=pad_i,
            max_chars=max_chars,
            max_chars_per_line=max_chars_per_line,
            min_font_size=min_font_size,
            max_font_size=max_font_size,
        )

    font = _load_font(chosen_font_size)
    img = Image.new("RGB", (size_i, size_i), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    wrapped_lines = _wrap_text_for_font(
        render_text,
        draw=draw,
        font=font,
        font_size=chosen_font_size,
        usable_w=usable_w,
        max_chars_per_line=max_chars_per_line,
    )

    line_h = _line_height_for_font(font, chosen_font_size)
    max_lines = max(1, usable_h // max(1, line_h))

    if len(wrapped_lines) > max_lines:
        wrapped_lines = wrapped_lines[: max_lines]
        # Add truncation marker on the last line while respecting pixel width.
        ell = " …"
        last = wrapped_lines[-1].rstrip()
        def _final_px_width(s: str) -> int:
            try:
                return int(draw.textlength(s, font=font))
            except Exception:
                try:
                    b = draw.textbbox((0, 0), s, font=font)
                    return max(0, int(b[2] - b[0]))
                except Exception:
                    return len(s) * max(1, int(chosen_font_size // 2))

        while last and _final_px_width(last + ell) > usable_w:
            last = last[:-1].rstrip()
        wrapped_lines[-1] = (last + ell) if last else "…"

    y = pad_i
    x = pad_i
    for line in wrapped_lines:
        draw.text((x, y), line, font=font, fill=(0, 0, 0))
        y += line_h

    return img


def _strip_for_note(text: str) -> str:
    """Heuristic clean-up so notes are less likely to leak a 'final answer' line."""
    t = "" if text is None else str(text)
    # Common patterns; keep conservative so we don't delete useful content.
    for marker in ("Final Answer:", "FINAL ANSWER:", "Answer:", "ANSWER:"):
        if marker in t:
            t = t.split(marker, 1)[0].rstrip()
    return t.strip()


@dataclass
class _OCRConfig:
    # Rendering
    image_size: int = 1280
    font_size: int = 10
    auto_font_size: bool = True
    min_font_size: int = 8
    max_font_size: int = 16
    pad: int = 12
    max_chars: int = 12000
    # <=0 means auto-wrap by rendered pixel width.
    max_chars_per_line: int = 0

    # Memory policy
    max_messages: int = 6

    # Generation
    # Separate budgets: notes should be short; final answer can be longer.
    msg_max_new_tokens: int = 1024
    judger_max_new_tokens: int = 1024
    temperature: float = 0.6
    top_p: float = 0.95
    do_sample: bool = True

    # Prompt hint
    protocol_hint: str = (
        "The attached image is NOT a photo. It contains TEXT messages from other agents. "
        "Read the image as a note and use it as context."
    )


# -----------------------------
# Main method
# -----------------------------


class VisionLatentMASMethodOCR:
    """Multi-agent method: communicate between agents via OCR-note images."""

    def __init__(self, args: argparse.Namespace, models: List[Any]):
        self.args = args
        self.models = models
        self.role_model_map = _load_role_model_map(getattr(args, "role_model_map", ""))

        # Agents are mainly used for role ordering; we keep the repo's defaults
        # when available, otherwise fall back to a minimal list.
        self.agents = copy.deepcopy(_DEFAULT_AGENTS)
        self.mode = getattr(args, "mode", getattr(args, "prompt", "sequential"))

        # Pull defaults from existing args where possible.
        cfg = _OCRConfig()

        # Reuse existing proto args if present in this repo.
        cfg.protocol_hint = str(getattr(args, "vision_comm_protocol_hint", cfg.protocol_hint))
        cfg.max_messages = max(1, _safe_int(getattr(args, "vision_comm_max_messages", cfg.max_messages), cfg.max_messages))
        cfg.max_chars = max(100, _safe_int(getattr(args, "vision_comm_max_chars", cfg.max_chars), cfg.max_chars))

        # Use text-image size args if present, otherwise fall back.
        cfg.image_size = max(128, _safe_int(getattr(args, "vision_t2v_text_image_size", cfg.image_size), cfg.image_size))

        # Generation knobs (keep consistent with other methods)
        msg_budget = _safe_int(getattr(args, "vision_ocr_msg_max_new_tokens", 0), 0)
        if msg_budget <= 0:
            latent_steps = _safe_int(getattr(args, "latent_steps", 0), 0)
            if latent_steps > 0:
                msg_budget = latent_steps
            else:
                msg_budget = _safe_int(
                    getattr(args, "vision_comm_msg_max_new_tokens", cfg.msg_max_new_tokens),
                    cfg.msg_max_new_tokens,
                )
        cfg.msg_max_new_tokens = max(1, msg_budget)
        cfg.judger_max_new_tokens = max(1, _safe_int(getattr(args, "max_new_tokens", cfg.judger_max_new_tokens), cfg.judger_max_new_tokens))
        cfg.temperature = _safe_float(getattr(args, "temperature", cfg.temperature), cfg.temperature)
        cfg.top_p = _safe_float(getattr(args, "top_p", cfg.top_p), cfg.top_p)
        cfg.do_sample = bool(getattr(args, "do_sample", False))
        # Safety: HF generate will error if do_sample=True with temperature<=0
        if cfg.do_sample and cfg.temperature <= 0:
            cfg.do_sample = False

        # Optional direct overrides for OCR method (rot args kept as backward-compatible aliases).
        cfg.image_size = max(
            128,
            _safe_int(
                getattr(args, "vision_ocr_image_size", getattr(args, "vision_rot_image_size", cfg.image_size)),
                cfg.image_size,
            ),
        )
        cfg.font_size = max(
            8,
            _safe_int(
                getattr(args, "vision_ocr_font_size", getattr(args, "vision_rot_font_size", cfg.font_size)),
                cfg.font_size,
            ),
        )
        cfg.auto_font_size = bool(
            _safe_int(
                getattr(args, "vision_ocr_auto_font_size", getattr(args, "vision_rot_auto_font_size", 1)),
                1,
            )
        )
        cfg.min_font_size = max(
            6,
            _safe_int(
                getattr(args, "vision_ocr_min_font_size", getattr(args, "vision_rot_min_font_size", cfg.min_font_size)),
                cfg.min_font_size,
            ),
        )
        cfg.max_font_size = max(
            cfg.min_font_size,
            _safe_int(
                getattr(args, "vision_ocr_max_font_size", getattr(args, "vision_rot_max_font_size", cfg.max_font_size)),
                cfg.max_font_size,
            ),
        )
        cfg.pad = max(
            0,
            _safe_int(
                getattr(args, "vision_ocr_pad", getattr(args, "vision_rot_pad", cfg.pad)),
                cfg.pad,
            ),
        )
        cfg.max_chars = max(
            100,
            _safe_int(
                getattr(args, "vision_ocr_max_chars", getattr(args, "vision_rot_max_chars", cfg.max_chars)),
                cfg.max_chars,
            ),
        )
        _ocr_line_cap = _safe_int(
            getattr(
                args,
                "vision_ocr_max_chars_per_line",
                getattr(args, "vision_rot_max_chars_per_line", cfg.max_chars_per_line),
            ),
            cfg.max_chars_per_line,
        )
        cfg.max_chars_per_line = 0 if _ocr_line_cap <= 0 else max(16, _ocr_line_cap)
        cfg.max_messages = max(
            1,
            _safe_int(
                getattr(args, "vision_ocr_max_messages", getattr(args, "vision_rot_max_messages", cfg.max_messages)),
                cfg.max_messages,
            ),
        )

        # High message-token budgets need a larger OCR canvas and denser line budget;
        # otherwise text gets heavily truncated before the receiver can read it.
        if cfg.msg_max_new_tokens >= 1000:
            cfg.image_size = max(cfg.image_size, 1280)
            cfg.max_chars = max(cfg.max_chars, 12000)
            if cfg.auto_font_size:
                cfg.min_font_size = min(cfg.min_font_size, 8)
                cfg.max_font_size = max(cfg.max_font_size, 16)
            if cfg.max_chars_per_line > 0:
                cfg.max_chars_per_line = max(cfg.max_chars_per_line, 120)

        self.cfg = cfg

        # Load processors per model (best effort).
        self.processors: List[Optional[Any]] = []
        for w in self.models:
            try:
                proc = AutoProcessor.from_pretrained(w.model_name, trust_remote_code=True)
                tok = getattr(proc, "tokenizer", None)
                if tok is not None and hasattr(tok, "padding_side"):
                    # Batched decoder-only generation should use left padding.
                    tok.padding_side = "left"
                self.processors.append(proc)
            except Exception:
                self.processors.append(None)

        self.blank_img = _make_blank_image(self.cfg.image_size)
        self.total_infer_time_sec = 0.0
        self.total_infer_batches = 0
        self.total_infer_items = 0

    def _get_role_model_idx(self, role: str, default_idx: int) -> int:
        if self.role_model_map and role in self.role_model_map:
            idx = int(self.role_model_map[role])
        else:
            idx = int(default_idx)
        if idx < 0 or idx >= len(self.models):
            raise ValueError(f"role_model_map index {idx} out of range for role {role}")
        return idx

    # -----------------------------
    # Prompts
    # -----------------------------

    def _build_agent_messages(self, role: str, question: str, memory_placeholder: str = "") -> List[Dict[str, Any]]:
        """Build role-specific prompts using the existing prompt builders."""
        if str(self.mode).lower() == "hierarchical":
            # prompts.py asserts on the method name list for hierarchical; use a compatible one.
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
            method="vision_latent_mas_ocr",
            args=self.args,
        )

    def _messages_to_mm(self, messages: List[Dict[str, Any]], user_suffix: str) -> List[Dict[str, Any]]:
        """Convert a normal chat list into multimodal chat by injecting one image placeholder."""
        mm: List[Dict[str, Any]] = []
        for m in messages:
            if m.get("role") == "user":
                content = m.get("content", "")
                # Keep existing list-format user content if already present.
                if isinstance(content, list):
                    mm.append(m)
                else:
                    txt = str(content)
                    if user_suffix:
                        txt = txt.rstrip() + "\n\n" + user_suffix
                    mm.append({
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": txt},
                        ],
                    })
            else:
                mm.append(m)
        return mm

    # -----------------------------
    # Rendering / memory
    # -----------------------------

    def _render_memory_image(self, notes: List[Tuple[str, str]]) -> Image.Image:
        """Render the current memory (list of (role, note_text)) into one image."""
        if not notes:
            return render_text_as_ocr_image(
                "(no prior messages)",
                size=self.cfg.image_size,
                font_size=self.cfg.font_size,
                auto_font_size=self.cfg.auto_font_size,
                min_font_size=self.cfg.min_font_size,
                max_font_size=self.cfg.max_font_size,
                pad=self.cfg.pad,
                max_chars=self.cfg.max_chars,
                max_chars_per_line=self.cfg.max_chars_per_line,
            )

        # Keep only the last N messages.
        notes = notes[-int(self.cfg.max_messages) :]

        chunks: List[str] = []
        for role, note in notes:
            role = str(role).strip() or "agent"
            note = _strip_for_note(note)
            if not note:
                continue
            chunks.append(f"[{role.upper()}]\n{note}")

        text = "\n\n-----\n\n".join(chunks) if chunks else "(no prior messages)"
        return render_text_as_ocr_image(
            text,
            size=self.cfg.image_size,
            font_size=self.cfg.font_size,
            auto_font_size=self.cfg.auto_font_size,
            min_font_size=self.cfg.min_font_size,
            max_font_size=self.cfg.max_font_size,
            pad=self.cfg.pad,
            max_chars=self.cfg.max_chars,
            max_chars_per_line=self.cfg.max_chars_per_line,
        )

    # -----------------------------
    # Generation helpers
    # -----------------------------

    def _apply_chat_template(self, proc: Any, messages: List[Dict[str, Any]]) -> str:
        """Best-effort chat template application for multimodal messages."""
        tok = getattr(proc, "tokenizer", None)
        if tok is not None and hasattr(tok, "apply_chat_template"):
            try:
                return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        # Some processors expose apply_chat_template directly.
        if hasattr(proc, "apply_chat_template"):
            try:
                return proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass

        # Fallback: crude textual format.
        parts: List[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):
                # Represent images with a placeholder.
                s = []
                for it in content:
                    if isinstance(it, dict) and it.get("type") == "image":
                        s.append("[IMAGE]")
                    elif isinstance(it, dict) and it.get("type") == "text":
                        s.append(str(it.get("text", "")))
                content_str = "\n".join(s)
            else:
                content_str = str(content)
            parts.append(f"<{role}>\n{content_str}\n</{role}>")
        parts.append("<assistant>\n")
        return "\n".join(parts)

    def _messages_with_user_suffix(
        self,
        base_messages: List[Dict[str, Any]],
        user_suffix: str,
    ) -> List[Dict[str, Any]]:
        """Append role instruction suffix to the latest user turn."""
        messages = copy.deepcopy(base_messages)
        if not user_suffix:
            return messages
        for m in reversed(messages):
            if m.get("role") == "user":
                c = m.get("content", "")
                if isinstance(c, str):
                    m["content"] = c + str(user_suffix)
                break
        return messages

    @torch.no_grad()
    def _generate_text_only_batch(
        self,
        model_idx: int,
        *,
        base_messages_batch: List[List[Dict[str, Any]]],
        user_suffix: str,
        max_new_tokens: int,
    ) -> List[str]:
        """Text-only batched generation fallback."""
        wrapper = self.models[model_idx]
        messages_batch = [
            self._messages_with_user_suffix(messages, user_suffix)
            for messages in base_messages_batch
        ]
        _prompts, input_ids, attention_mask, _tokens = wrapper.prepare_chat_batch(
            messages_batch,
            add_generation_prompt=True,
        )
        temperature = float(self.cfg.temperature) if self.cfg.do_sample else 0.0
        out_texts, _ = wrapper.generate_text_batch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=float(self.cfg.top_p),
        )
        return [str(x) for x in out_texts]

    @torch.no_grad()
    def _generate_with_optional_image_batch(
        self,
        model_idx: int,
        *,
        base_messages_batch: List[List[Dict[str, Any]]],
        memory_imgs: Optional[List[Optional[Image.Image]]],
        user_suffix: str,
        max_new_tokens: int,
        want_image: bool,
    ) -> List[str]:
        """Batched generation with multimodal path when available."""
        wrapper = self.models[model_idx]
        proc = self.processors[model_idx]
        bsz = len(base_messages_batch)

        # Text-only fallback conditions.
        if (proc is None) or (not want_image) or (memory_imgs is None):
            return self._generate_text_only_batch(
                model_idx,
                base_messages_batch=base_messages_batch,
                user_suffix=user_suffix,
                max_new_tokens=max_new_tokens,
            )
        if len(memory_imgs) != bsz or any(img is None for img in memory_imgs):
            return self._generate_text_only_batch(
                model_idx,
                base_messages_batch=base_messages_batch,
                user_suffix=user_suffix,
                max_new_tokens=max_new_tokens,
            )

        # Multimodal path (real batch).
        tok = getattr(proc, "tokenizer", None)
        if tok is not None and hasattr(tok, "padding_side"):
            tok.padding_side = "left"

        mm_messages_batch = [
            self._messages_to_mm(messages, user_suffix=user_suffix)
            for messages in base_messages_batch
        ]
        chats = [self._apply_chat_template(proc, mm_messages) for mm_messages in mm_messages_batch]
        images_flat = [img for img in memory_imgs if img is not None]
        images_nested = [[img] for img in images_flat]

        mm_inputs = None
        for images_arg in (images_flat, images_nested):
            try:
                mm_inputs = proc(text=chats, images=images_arg, return_tensors="pt", padding=True)
                break
            except Exception:
                mm_inputs = None
        if mm_inputs is None:
            return self._generate_text_only_batch(
                model_idx,
                base_messages_batch=base_messages_batch,
                user_suffix=user_suffix,
                max_new_tokens=max_new_tokens,
            )

        mm_inputs = _maybe_to_device(mm_inputs, wrapper.device)
        try:
            gen_kwargs: Dict[str, Any] = {
                **mm_inputs,
                "max_new_tokens": max_new_tokens,
                "do_sample": self.cfg.do_sample,
                "pad_token_id": getattr(getattr(proc, "tokenizer", None), "pad_token_id", None)
                or getattr(wrapper.tokenizer, "pad_token_id", None),
            }
            if self.cfg.do_sample:
                gen_kwargs["temperature"] = float(self.cfg.temperature)
                gen_kwargs["top_p"] = float(self.cfg.top_p)
            out_ids = wrapper.model.generate(**gen_kwargs)
        except Exception:
            return self._generate_text_only_batch(
                model_idx,
                base_messages_batch=base_messages_batch,
                user_suffix=user_suffix,
                max_new_tokens=max_new_tokens,
            )

        attn = mm_inputs.get("attention_mask", None)
        if attn is not None:
            prompt_lens = [int(x) for x in attn.sum(dim=1).tolist()]
        else:
            prompt_lens = [int(mm_inputs["input_ids"].shape[1])] * bsz

        tok = getattr(proc, "tokenizer", None) or wrapper.tokenizer
        outs: List[str] = []
        for i in range(bsz):
            gen_ids = out_ids[i, prompt_lens[i] :]
            outs.append(tok.decode(gen_ids, skip_special_tokens=True))
        return outs

    def _run_batch_impl(self, batch: List[Dict[str, Any]]) -> List[str]:
        if not batch:
            return []

        questions = [item.get("question", item.get("prompt", "")) for item in batch]
        notes_batch: List[List[Tuple[str, str]]] = [[] for _ in questions]

        planner_idx = self._get_role_model_idx("planner", 0)
        critic_idx = self._get_role_model_idx("critic", 1 if len(self.models) > 1 else 0)
        refiner_idx = self._get_role_model_idx("refiner", 2 if len(self.models) > 2 else critic_idx)
        judger_idx = self._get_role_model_idx("judger", len(self.models) - 1)

        planner_msgs_batch = [
            self._build_agent_messages("planner", q, memory_placeholder="")
            for q in questions
        ]
        planner_imgs = [self._render_memory_image(notes) for notes in notes_batch]
        planner_suffix = (
            f"{self.cfg.protocol_hint}\n\n"
            "Write a compact reasoning note for the other agents. "
            f"Keep it under {self.cfg.max_chars} characters. Do NOT include the final answer."
        )
        planner_outs = self._generate_with_optional_image_batch(
            planner_idx,
            base_messages_batch=planner_msgs_batch,
            memory_imgs=planner_imgs,
            user_suffix=planner_suffix,
            max_new_tokens=self.cfg.msg_max_new_tokens,
            want_image=True,
        )
        for i, out in enumerate(planner_outs):
            notes_batch[i].append(("planner", out))

        critic_msgs_batch = [
            self._build_agent_messages("critic", q, memory_placeholder="")
            for q in questions
        ]
        critic_imgs = [self._render_memory_image(notes) for notes in notes_batch]
        critic_suffix = (
            f"{self.cfg.protocol_hint}\n\n"
            "Critique the plan and identify possible mistakes. "
            f"Keep it under {self.cfg.max_chars} characters. Do NOT include the final answer."
        )
        critic_outs = self._generate_with_optional_image_batch(
            critic_idx,
            base_messages_batch=critic_msgs_batch,
            memory_imgs=critic_imgs,
            user_suffix=critic_suffix,
            max_new_tokens=self.cfg.msg_max_new_tokens,
            want_image=True,
        )
        for i, out in enumerate(critic_outs):
            notes_batch[i].append(("critic", out))

        refiner_msgs_batch = [
            self._build_agent_messages("refiner", q, memory_placeholder="")
            for q in questions
        ]
        refiner_imgs = [self._render_memory_image(notes) for notes in notes_batch]
        refiner_suffix = (
            f"{self.cfg.protocol_hint}\n\n"
            "Propose an improved solution approach based on the notes. "
            f"Keep it under {self.cfg.max_chars} characters. Do NOT include the final answer."
        )
        refiner_outs = self._generate_with_optional_image_batch(
            refiner_idx,
            base_messages_batch=refiner_msgs_batch,
            memory_imgs=refiner_imgs,
            user_suffix=refiner_suffix,
            max_new_tokens=self.cfg.msg_max_new_tokens,
            want_image=True,
        )
        for i, out in enumerate(refiner_outs):
            notes_batch[i].append(("refiner", out))

        judger_msgs_batch = [
            self._build_agent_messages("judger", q, memory_placeholder="")
            for q in questions
        ]
        judger_imgs = [self._render_memory_image(notes) for notes in notes_batch]
        judger_suffix = (
            f"{self.cfg.protocol_hint}\n\n"
            "Use the notes in the image as context. Produce ONLY the final answer."
        )
        final_texts = self._generate_with_optional_image_batch(
            judger_idx,
            base_messages_batch=judger_msgs_batch,
            memory_imgs=judger_imgs,
            user_suffix=judger_suffix,
            max_new_tokens=self.cfg.judger_max_new_tokens,
            want_image=True,
        )

        # Return raw model text. Evaluation/extraction is handled centrally in run.py.
        return final_texts

    # -----------------------------
    # Public API (used by run.py)
    # -----------------------------

    def run_batch(self, batch: List[Dict[str, Any]]) -> List[str]:
        infer_start = time.time()
        outputs = self._run_batch_impl(batch)
        infer_total = time.time() - infer_start
        per_item = infer_total / max(1, len(batch))
        self.total_infer_time_sec += infer_total
        self.total_infer_batches += 1
        self.total_infer_items += len(batch)
        avg_all = self.total_infer_time_sec / max(1, self.total_infer_items)
        print(
            f"[vision_latent_mas_ocr] batch_infer_time_sec={infer_total:.4f} "
            f"per_item_sec={per_item:.4f} batch_size={len(batch)} "
            f"total_infer_time_sec={self.total_infer_time_sec:.4f} avg_per_item_sec={avg_all:.4f}"
        )
        return outputs

    def run_item(self, item: Dict[str, Any]) -> str:
        outs = self._run_batch_impl([item])
        return outs[0] if outs else ""


# Backward-compatible aliases for old name.
render_text_as_rot_image = render_text_as_ocr_image
VisionLatentMASMethodRoT = VisionLatentMASMethodOCR
