#!/usr/bin/env python3
   
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, Iterable, List, Optional

from datasets import load_dataset


FORCE_STREAMING_DATASETS = {
    "bytedance-seed/code-contests-plus",
}

OPEN_CODE_REASONING_DATASET = "nvidia/opencodereasoning"
COS_E_DATASET = "salesforce/cos_e"

def _build_text_from_conversations(conversations: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for turn in conversations:
        role = str(turn.get("from", "")).strip()
        value = str(turn.get("value", "")).strip()
        if not value:
            continue
        if role:
            parts.append(f"{role}: {value}")
        else:
            parts.append(value)
    return "\n\n".join(parts).strip()

def _pick_first(item: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for k in keys:
        v = item.get(k, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _join_nonempty(parts: List[Optional[str]], sep: str = "\n\n") -> str:
    out: List[str] = []
    for p in parts:
        if isinstance(p, str):
            p = p.strip()
            if p:
                out.append(p)
    return sep.join(out).strip()


def _join_steps(steps: Any) -> Optional[str]:
    if not isinstance(steps, list) or not steps:
        return None
    parts: List[str] = []
    for idx, s in enumerate(steps):
        if isinstance(s, str) and s.strip():
            parts.append(s.strip())
            continue
        if isinstance(s, dict):
            txt = _pick_first(s, ["text", "content", "step", "analysis", "response", "completion"])
            if txt:
                parts.append(txt)
                continue
                              
    if not parts:
        return None
    return "\n".join(f"{i+1}. {p}" for i, p in enumerate(parts))


def _build_text_from_prm800k(item: Dict[str, Any]) -> Optional[str]:
    question = item.get("question", {}) if isinstance(item.get("question", {}), dict) else {}
    problem = _pick_first(question, ["problem", "question", "prompt", "input", "instruction"])
    final = _pick_first(question, ["ground_truth_answer", "answer", "final", "output"])

    label = item.get("label", {}) if isinstance(item.get("label", {}), dict) else {}
    steps_raw = label.get("steps", None)
    steps_out: List[str] = []
    if isinstance(steps_raw, list):
        for step in steps_raw:
            if not isinstance(step, dict):
                continue
            human = step.get("human_completion", None)
            if isinstance(human, str) and human.strip():
                steps_out.append(human.strip())
                continue
            comps = step.get("completions", None)
            chosen = step.get("chosen_completion", None)
            if isinstance(comps, list) and comps:
                comp = None
                if isinstance(chosen, int) and 0 <= chosen < len(comps):
                    comp = comps[chosen]
                else:
                    comp = comps[0]
                if isinstance(comp, dict):
                    text = comp.get("text", None)
                    if isinstance(text, str) and text.strip():
                        steps_out.append(text.strip())

    steps = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps_out)) if steps_out else None

    if not any([problem, steps, final]):
        return None

                                                         
    return _join_nonempty([problem, steps, final])


def _build_text_from_code_contests(item: Dict[str, Any]) -> Optional[str]:
    title = _pick_first(item, ["title"])
    description = _pick_first(item, ["description", "statement", "problem"])
    time_limit = item.get("time_limit", None)
    memory_limit = item.get("memory_limit", None)

    if not any([title, description]):
        return None

                                     
    return _join_nonempty(
        [
            title,
            description,
            str(time_limit) if isinstance(time_limit, (int, float)) else None,
            str(memory_limit) if isinstance(memory_limit, (int, float)) else None,
        ]
    )


def _build_text_from_ifeval(item: Dict[str, Any]) -> Optional[str]:
    prompt = _pick_first(item, ["prompt"])
    if not prompt:
        return None
    return prompt


def _build_text_from_open_code_reasoning(item: Dict[str, Any]) -> Optional[str]:
    problem = _pick_first(item, ["input"])
                                                                                    
    solution = _pick_first(item, ["solution"])
    if not problem:
        return None
                                     
    return _join_nonempty([problem, solution])

def _build_text_from_cos_e(item: Dict[str, Any]) -> Optional[str]:
    question = _pick_first(item, ["question"])
    choices = item.get("choices", None)
    ans = _pick_first(item, ["answer"])
    abs_exp = _pick_first(item, ["abstractive_explanation"])
    ext_exp = _pick_first(item, ["extractive_explanation"])

    if not question:
        return None

    choices_str = None
    if isinstance(choices, list) and choices:
        choices_str = " | ".join(str(c) for c in choices)

                                                                      
                                                                                
    explanation = abs_exp or ext_exp
                                
    return _join_nonempty([question, choices_str, explanation, ans])


def _build_text_from_item(dataset: str, item: Dict[str, Any]) -> Optional[str]:
                              
    ds = dataset.lower()
    if "prm800k" in ds:
        return _build_text_from_prm800k(item)
    if "code-contests-plus" in ds or "code_contests" in ds:
        return _build_text_from_code_contests(item)
    if "ifeval" in ds:
        return _build_text_from_ifeval(item)
    if "opencodereasoning" in ds:
        return _build_text_from_open_code_reasoning(item)
    if ds == COS_E_DATASET:
        return _build_text_from_cos_e(item)

                                           
    conversations = item.get("conversations", None)
    if isinstance(conversations, list) and conversations:
        return _build_text_from_conversations(conversations)

                                                
    return _pick_first(item, ["prompt", "question", "instruction", "input", "text"])


def _iter_examples(ds: Iterable[Dict[str, Any]], dataset: str, limit: int) -> Iterable[str]:
    count = 0
    for item in ds:
        text = _build_text_from_item(dataset, item)
        if text:
            yield text
            count += 1
        if limit > 0 and count >= limit:
            break
    return


def _parse_dataset_spec(spec: str) -> Tuple[str, Optional[str]]:
       
    s = (spec or "").strip()
    if "@" in s:
        name, cfg = s.split("@", 1)
        name = name.strip()
        cfg = cfg.strip()
        return name, (cfg or None)
    return s, None


def _iter_dataset_texts(dataset_spec: str, split: str, streaming: bool, limit: int) -> Iterable[str]:
    name, cfg = _parse_dataset_spec(dataset_spec)
    name_l = name.lower()

                                                                                                
    if name_l == OPEN_CODE_REASONING_DATASET:
        if cfg is None:
                                                                                    
            cfg = split if split in ("split_0", "split_1") else "split_0"
        if split not in ("split_0", "split_1"):
            split = cfg
        ds = load_dataset(name, cfg, split=split, streaming=streaming)
    elif name_l == COS_E_DATASET:
                                                         
        cfg = cfg or "v1.11"
        ds = load_dataset(name, cfg, split=split, streaming=streaming)
    else:
        ds = load_dataset(name, cfg, split=split, streaming=streaming) if cfg else load_dataset(name, split=split, streaming=streaming)

    yield from _iter_examples(ds, name, limit)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        type=str,
        default="NovaSky-AI/Sky-T1_data_17k",
        help="HF dataset name (single). Use --datasets for multiple.",
    )
    ap.add_argument("--datasets", type=str, default="", help="Comma-separated dataset names.")
    ap.add_argument("--split", type=str, default="train", help="Split name (single). Use --splits for multiple.")
    ap.add_argument("--splits", type=str, default="", help="Comma-separated split names aligned to --datasets.")
    ap.add_argument("--out", type=str, default="data/vision_codec_anchor_text/sky_t1_17k.txt")
    ap.add_argument("--limit", type=int, default=500, help="Max number of examples to write (<=0 for all).")
    ap.add_argument("--limit_per_dataset", type=int, default=500, help="Max per dataset when using --datasets.")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle examples within each dataset before writing.")
    ap.add_argument("--shuffle_seed", type=int, default=42)
    ap.add_argument("--format", type=str, choices=["txt", "jsonl"], default="txt")
    ap.add_argument("--streaming", action="store_true", help="Use streaming dataset loading.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.datasets:
        datasets = [s.strip() for s in args.datasets.split(",") if s.strip()]
    else:
        datasets = [args.dataset]

    if args.splits:
        splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    else:
        splits = [args.split] * len(datasets)

    if len(splits) == 1 and len(datasets) > 1:
        splits = splits * len(datasets)
    if len(splits) != len(datasets):
        raise ValueError("--splits must have the same length as --datasets (or be a single value).")

    rng = random.Random(int(args.shuffle_seed))

    if args.format == "jsonl":
        with open(args.out, "w", encoding="utf-8") as f:
            for ds_name, split in zip(datasets, splits):
                base_name, _ = _parse_dataset_spec(ds_name)
                ds_streaming = args.streaming or base_name.lower() in FORCE_STREAMING_DATASETS
                limit = args.limit_per_dataset if args.datasets else args.limit
                texts = list(_iter_dataset_texts(ds_name, split, ds_streaming, limit))
                if args.shuffle:
                    rng.shuffle(texts)
                for text in texts:
                    f.write(json.dumps({"text": text, "dataset": ds_name}, ensure_ascii=False) + "\n")
    else:
        with open(args.out, "w", encoding="utf-8") as f:
            for ds_name, split in zip(datasets, splits):
                base_name, _ = _parse_dataset_spec(ds_name)
                ds_streaming = args.streaming or base_name.lower() in FORCE_STREAMING_DATASETS
                limit = args.limit_per_dataset if args.datasets else args.limit
                texts = list(_iter_dataset_texts(ds_name, split, ds_streaming, limit))
                if args.shuffle:
                    rng.shuffle(texts)
                for text in texts:
                    f.write(text.replace("\r\n", "\n").replace("\r", "\n") + "\n")


if __name__ == "__main__":
    main()
