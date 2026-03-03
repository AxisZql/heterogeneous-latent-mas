#!/usr/bin/env python3
"""Merge partition JSONLs into canonical experiment JSONLs.

Example:
  python scripts/merge_partition_jsonl.py \
    --jsonl_root preds_jsonl/exp_qwen8b_gemma4b_pool2g \
    --interval_sec 15

This script is read-mostly on partition files and writes only canonical files.
It is safe for concurrent readers (e.g. rsync) because updates are atomic (tmp + replace).
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PARTITION_RE = re.compile(r"^(?P<stem>.+?)(?:_world_(?P<world>\d+))?_partition_(?P<pid>\d+)\.jsonl$")


def _iter_jsonl_files(root: Path, recursive: bool) -> List[Path]:
    out: List[Path] = []
    if not root.is_dir():
        return out
    if recursive:
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                if name.endswith(".jsonl"):
                    out.append(Path(dirpath) / name)
    else:
        for name in os.listdir(root):
            p = root / name
            if p.is_file() and name.endswith(".jsonl"):
                out.append(p)
    out.sort()
    return out


def _group_partition_files(root: Path, recursive: bool, world_size: int) -> Dict[Path, List[Tuple[int, Path]]]:
    groups: Dict[Path, List[Tuple[int, Path]]] = {}
    for path in _iter_jsonl_files(root, recursive=recursive):
        m = PARTITION_RE.match(path.name)
        if not m:
            continue
        part_world_raw = m.group("world")
        part_world = int(part_world_raw) if part_world_raw is not None else 4
        if part_world != world_size:
            continue
        pid = int(m.group("pid"))
        base_path = path.with_name(f"{m.group('stem')}.jsonl")
        groups.setdefault(base_path, []).append((pid, path))
    for base in groups:
        groups[base].sort(key=lambda x: x[0])
    return groups


def _stable_row_key(row: Dict) -> Optional[int]:
    idx = row.get("problem_idx")
    if isinstance(idx, int) and idx > 0:
        return idx
    return None


def _read_jsonl_rows(path: Path) -> Tuple[List[Dict], int]:
    """Return (rows, skipped_partial_or_invalid_count)."""
    rows: List[Dict] = []
    skipped = 0
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                # Ignore trailing partial line while writer is still appending.
                if not raw.endswith("\n"):
                    skipped += 1
                    continue
                s = raw.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    skipped += 1
                    continue
                if not isinstance(obj, dict):
                    skipped += 1
                    continue
                if _stable_row_key(obj) is None:
                    skipped += 1
                    continue
                rows.append(obj)
    except Exception:
        return [], 1
    return rows, skipped


def _read_partition_rows(path: Path) -> Tuple[List[Dict], int]:
    return _read_jsonl_rows(path)


def _read_canonical_rows(path: Path) -> Tuple[List[Dict], int]:
    if not path.exists():
        return [], 0
    return _read_jsonl_rows(path)


def _row_fingerprint(row: Dict) -> str:
    try:
        return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return repr(row)


def _merge_rows(base_path: Path, partitions: List[Tuple[int, Path]]) -> Tuple[List[Dict], int, int]:
    """Return merged rows sorted by problem_idx, conflict_count, skipped_count."""
    by_idx: Dict[int, Dict] = {}
    by_idx_fp: Dict[int, str] = {}
    conflicts = 0
    skipped_total = 0

    # Seed from canonical cache first so partition merges never discard existing rows.
    base_rows, skipped = _read_canonical_rows(base_path)
    skipped_total += skipped
    for row in base_rows:
        idx = _stable_row_key(row)
        if idx is None:
            continue
        fp = _row_fingerprint(row)
        by_idx[idx] = row
        by_idx_fp[idx] = fp

    for _pid, part_path in partitions:
        rows, skipped = _read_partition_rows(part_path)
        skipped_total += skipped
        for row in rows:
            idx = _stable_row_key(row)
            if idx is None:
                continue
            fp = _row_fingerprint(row)
            prev = by_idx.get(idx)
            if prev is None:
                by_idx[idx] = row
                by_idx_fp[idx] = fp
                continue
            if by_idx_fp.get(idx) != fp:
                conflicts += 1
                # Last-seen wins. In healthy partitioned runs this should not happen.
                by_idx[idx] = row
                by_idx_fp[idx] = fp

    merged = [by_idx[idx] for idx in sorted(by_idx.keys())]
    return merged, conflicts, skipped_total


def _rows_to_text(rows: List[Dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def _read_text_if_exists(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None


def _atomic_write_text(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.tmp.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _run_once(root: Path, recursive: bool, world_size: int) -> Dict[str, int]:
    groups = _group_partition_files(root, recursive=recursive, world_size=world_size)
    updated_files = 0
    skipped_files = 0
    conflicts = 0
    skipped_lines = 0

    for base_path, part_list in sorted(groups.items(), key=lambda kv: str(kv[0])):
        merged_rows, c, s = _merge_rows(base_path, part_list)
        conflicts += c
        skipped_lines += s

        new_text = _rows_to_text(merged_rows)
        old_text = _read_text_if_exists(base_path)
        if old_text == new_text:
            skipped_files += 1
            continue
        _atomic_write_text(base_path, new_text)
        updated_files += 1

    return {
        "groups": len(groups),
        "updated_files": updated_files,
        "skipped_files": skipped_files,
        "conflicts": conflicts,
        "skipped_lines": skipped_lines,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge *_partition_k.jsonl files into canonical JSONLs.")
    ap.add_argument("--jsonl_root", type=str, default="preds_jsonl", help="Root folder to scan.")
    ap.add_argument(
        "--world_size",
        type=int,
        default=4,
        help="Only merge partitions for this world size. Legacy names without world tag are treated as world=4.",
    )
    ap.add_argument("--interval_sec", type=int, default=30, help="Polling interval.")
    ap.add_argument("--once", action="store_true", help="Run one merge pass and exit.")
    ap.add_argument(
        "--no_recursive",
        action="store_true",
        help="Only scan top-level files under --jsonl_root.",
    )
    args = ap.parse_args()
    if args.world_size < 1:
        raise ValueError("--world_size must be >= 1.")

    root = Path(args.jsonl_root).resolve()
    lock_path = root / ".merge_partition_jsonl.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fh = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"[lock] another merger instance is already running for {root}")
        return

    print(
        f"[start] root={root} recursive={not args.no_recursive} "
        f"world_size={args.world_size} interval_sec={args.interval_sec}"
    )
    try:
        while True:
            t0 = time.time()
            stats = _run_once(root=root, recursive=not bool(args.no_recursive), world_size=int(args.world_size))
            elapsed = time.time() - t0
            print(
                "[merge] "
                f"groups={stats['groups']} updated={stats['updated_files']} "
                f"unchanged={stats['skipped_files']} conflicts={stats['conflicts']} "
                f"skipped_lines={stats['skipped_lines']} elapsed_sec={elapsed:.2f}"
            )
            if args.once:
                break
            time.sleep(max(0.0, float(args.interval_sec) - elapsed))
    finally:
        try:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        lock_fh.close()


if __name__ == "__main__":
    main()
