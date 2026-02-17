import os
import random
import re
from typing import Dict, Optional, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def auto_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

                                       
_NUM_PATTERN = r"[-+]?\d+(?:\.\d+)?"
_HASH_LINE_RE = re.compile(rf"^####\s*({_NUM_PATTERN})\s*$", re.MULTILINE)
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_NUM_RE = re.compile(_NUM_PATTERN)


def _last_match(pattern: re.Pattern, text: str) -> Optional[re.Match]:
    last = None
    for match in pattern.finditer(text):
        last = match
    return last


def _parse_boxed_content(content: str, task: str) -> Optional[str]:
    content = content.strip()
    numbers = _NUM_RE.findall(content)
    if numbers:
        return numbers[-1]
    if task in {"arc_easy", "arc_challenge", "gpqa", "medqa"}:
        if re.fullmatch(r"[A-Da-d]", content):
            return content.lower()
    if task == "winogrande":
        if content in {"1", "2"}:
            return content
    return None


def extract_answer_with_meta(text: str, task: str) -> Tuple[Optional[str], Dict[str, Optional[str]]]:
    task = (task or "").lower()

    def _boxed() -> Tuple[Optional[str], Optional[str]]:
        match = _last_match(_BOXED_RE, text)
        if not match:
            return None, None
        content = match.group(1)
        pred = _parse_boxed_content(content, task)
        if pred is None and task in {"aime2024", "aime2025", "gsm8k"}:
            return None, None
        return pred, match.group(0)

    def _hashes() -> Tuple[Optional[str], Optional[str]]:
        match = _last_match(_HASH_LINE_RE, text)
        if not match:
            return None, None
        return match.group(1), match.group(0)

    def _fallback_number() -> Tuple[Optional[str], Optional[str]]:
        numbers = _NUM_RE.findall(text)
        if not numbers:
            return None, None
        return numbers[-1], numbers[-1]

    def _fallback_choice() -> Tuple[Optional[str], Optional[str]]:
        matches = re.findall(r"\b([A-Da-d])\b", text)
        if matches:
            value = matches[-1].lower()
            return value, value
        matches = re.findall(r"\b([12])\b", text)
        if matches:
            value = matches[-1]
            return value, value
        return None, None

    if task == "gsm8k":
        order = [("hashes", _hashes), ("boxed", _boxed), ("fallback", _fallback_number)]
    elif task in {"aime2024", "aime2025"}:
        order = [("boxed", _boxed), ("hashes", _hashes), ("fallback", _fallback_number)]
    elif task in {"arc_easy", "arc_challenge", "gpqa", "medqa", "winogrande"}:
        order = [("boxed", _boxed), ("hashes", _hashes), ("fallback", _fallback_choice)]
    else:
        order = [("boxed", _boxed), ("hashes", _hashes), ("fallback", _fallback_number)]

    for source, fn in order:
        pred, span = fn()
        if pred:
            return pred, {"extraction_source": source, "extracted_span": span}

    return None, {"extraction_source": "unparsable", "extracted_span": None}


def extract_gsm8k_answer(text: str) -> Optional[str]:
    pred, _ = extract_answer_with_meta(text, "gsm8k")
    return pred


def extract_gold(text: str) -> Optional[str]:
    match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", text)
    return match.group(1) if match else None


def get_stop_regexes(task: str) -> list[str]:
    task = (task or "").lower()
    if task == "gsm8k":
        return [rf"^####\s*{_NUM_PATTERN}\s*$"]
    if task in {"aime2024", "aime2025"}:
        return [r"\\boxed\{[^}]*\}", rf"^####\s*{_NUM_PATTERN}\s*$"]
    if task in {"arc_easy", "arc_challenge", "gpqa", "medqa", "winogrande"}:
        return [r"\\boxed\{[^}]*\}"]
    return []


def normalize_answer(ans: Optional[str]) -> Optional[str]:
    if ans is None:
        return None
    return ans.strip().lower()


def extract_markdown_python_block(text: str) -> Optional[str]:
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None


               
import traceback
from multiprocessing import Process, Manager
def run_with_timeout(code, timeout):
    def worker(ns, code):
        try:
            local_ns = {}
            exec(code, local_ns)
            ns['ok'] = True
            ns['error'] = None
        except Exception:
            ns['ok'] = False
            ns['error'] = traceback.format_exc()
    with Manager() as manager:
        ns = manager.dict()
        p = Process(target=worker, args=(ns, code))
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            ns['ok'] = False
            ns['error'] = f"TimeoutError: Execution exceeded {timeout} seconds"
        return ns.get('ok', False), ns.get('error', None)
