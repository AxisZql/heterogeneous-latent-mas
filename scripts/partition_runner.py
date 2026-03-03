#!/usr/bin/env python3
"""Partition scheduler for partitioned JSONL runs.

Supports two schedulers:
  - wave: strict barrier per (task, method)
  - queue: each GPU worker immediately pulls the next partition from later waves

Each partition writes:
  <jsonl_dir>/<task>_<method_tag>_world_<N>_partition_<k>.jsonl
Legacy files without world tag are treated as world=4 for compatibility.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


TASK_MAX_NEW_TOKENS: Dict[str, int] = {
    "gsm8k": 2048,
    "arc_easy": 2048,
    "arc_challenge": 2048,
    "gpqa": 8192,
    "medqa": 4096,
    "mbppplus": 4096,
    "humanevalplus": 4096,
    "aime2024": 20000,
    "aime2025": 20000,
}

METHODS = (
    "vision_latent_mas_codec_new",
    "vision_latent_mas_rot",
    "vision_latent_mas_ocr",
    "text_mas",
    "latent_mas_hybrid",
    "baseline",
)
DEFAULT_BS_LEVELS_DESC = [12, 8, 4, 2, 1]


def _task_iter(task: str) -> Iterable[Dict[str, Any]]:
    # Lazy import to avoid loading datasets module unless this completion check is needed.
    from data import (
        load_aime2024,
        load_aime2025,
        load_arc_challenge,
        load_arc_easy,
        load_gpqa_diamond,
        load_gsm8k,
        load_humanevalplus,
        load_mbppplus,
        load_medqa,
    )

    if task == "gsm8k":
        return load_gsm8k(split="test")
    if task == "aime2024":
        return load_aime2024(split="train")
    if task == "aime2025":
        return load_aime2025(split="train")
    if task == "gpqa":
        return load_gpqa_diamond(split="test")
    if task == "arc_easy":
        return load_arc_easy(split="test")
    if task == "arc_challenge":
        return load_arc_challenge(split="test")
    if task == "mbppplus":
        return load_mbppplus(split="test")
    if task == "humanevalplus":
        return load_humanevalplus(split="test")
    if task == "medqa":
        return load_medqa(split="test")
    raise ValueError(f"Unsupported task for completion check: {task}")


def _count_task_samples(task: str) -> int:
    return sum(1 for _ in _task_iter(task))


def _parse_bs_levels(raw: str) -> List[int]:
    vals: List[int] = []
    for s in (raw or "").split(","):
        s = s.strip()
        if not s:
            continue
        try:
            v = int(s)
        except Exception:
            continue
        if v <= 0:
            continue
        if v not in vals:
            vals.append(v)
    vals.sort(reverse=True)
    return vals


def _initial_bs_for_max_new_tokens(max_new_tokens: int, auto_generate_bs: bool, fallback_bs: int) -> int:
    if not auto_generate_bs:
        return max(1, int(fallback_bs))
    if max_new_tokens <= 2048:
        return 12
    if max_new_tokens <= 4096:
        return 8
    return 4


def _build_bs_retry_plan(initial_bs: int, levels_desc: List[int]) -> List[int]:
    bs0 = max(1, int(initial_bs))
    out: List[int] = [bs0]
    for v in levels_desc:
        if v < bs0 and v not in out:
            out.append(v)
    if 1 not in out:
        out.append(1)
    return out


def _partition_target_count(max_samples: int, num_partitions: int, partition_id: int) -> int:
    if max_samples <= 0:
        return 0
    if num_partitions <= 1:
        return max_samples
    start = partition_id + 1
    if start > max_samples:
        return 0
    return ((max_samples - start) // num_partitions) + 1


def _partition_expected_ids(max_samples: int, num_partitions: int, partition_id: int) -> Set[int]:
    if max_samples <= 0:
        return set()
    start = partition_id + 1
    if start > max_samples:
        return set()
    return set(range(start, max_samples + 1, num_partitions))


def _method_tag(method: str) -> str:
    if method == "vision_latent_mas_codec_new":
        return "vision_codec_new"
    # Keep OCR/ROT file naming unified for backward compatibility with existing JSONL paths.
    if method in {"vision_latent_mas_rot", "vision_latent_mas_ocr"}:
        return "vision_rot"
    if method == "latent_mas_hybrid":
        return "latent_mas_hybrid"
    if method == "baseline":
        return "baseline"
    return "text_mas"


def _partition_jsonl_path(
    jsonl_dir: Path,
    task: str,
    method_tag: str,
    world_size: int,
    partition_id: int,
) -> Path:
    return jsonl_dir / f"{task}_{method_tag}_world_{world_size}_partition_{partition_id}.jsonl"


def _partition_legacy_jsonl_path(
    jsonl_dir: Path,
    task: str,
    method_tag: str,
    partition_id: int,
) -> Path:
    return jsonl_dir / f"{task}_{method_tag}_partition_{partition_id}.jsonl"


def _partition_read_paths(
    jsonl_dir: Path,
    task: str,
    method_tag: str,
    world_size: int,
    partition_id: int,
) -> List[Path]:
    paths = [_partition_jsonl_path(jsonl_dir, task, method_tag, world_size, partition_id)]
    # Backward compatibility: legacy naming implies world=4.
    if world_size == 4:
        legacy = _partition_legacy_jsonl_path(jsonl_dir, task, method_tag, partition_id)
        if legacy not in paths:
            paths.append(legacy)
    return paths


def _partition_log_path(
    log_dir: Path,
    task: str,
    method_tag: str,
    world_size: int,
    partition_id: int,
) -> Path:
    return log_dir / f"{task}_{method_tag}_world_{world_size}_partition_{partition_id}.log"


def parse_csv(raw: str) -> List[str]:
    return [x.strip() for x in (raw or "").split(",") if x.strip()]


def build_gpu_groups(gpus: Sequence[str], gpus_per_job: int) -> Tuple[List[List[str]], List[str]]:
    if gpus_per_job <= 0:
        raise ValueError("--gpus_per_job must be >= 1.")
    n_full = len(gpus) // gpus_per_job
    used = n_full * gpus_per_job
    groups = [list(gpus[i : i + gpus_per_job]) for i in range(0, used, gpus_per_job)]
    dropped = list(gpus[used:])
    return groups, dropped


@dataclass
class PartitionJob:
    wave_idx: int
    task: str
    method: str
    partition_id: int
    num_partitions: int
    gpu_group: List[str]
    cmd: List[str]
    log_path: Path
    partition_jsonl_path: Path

    @property
    def name(self) -> str:
        return (
            f"{self.task}_{_method_tag(self.method)}_world_{self.num_partitions}_"
            f"partition_{self.partition_id}"
        )


@dataclass
class PartitionResult:
    job: PartitionJob
    returncode: int
    elapsed_sec: float
    attempts: List[Dict[str, Any]]
    start_ts: float
    end_ts: float


def _get_cmd_arg(cmd: List[str], flag: str) -> Optional[str]:
    for i in range(len(cmd) - 1):
        if cmd[i] == flag:
            return cmd[i + 1]
    return None


def _set_cmd_arg(cmd: List[str], flag: str, value: str) -> List[str]:
    out = list(cmd)
    for i in range(len(out) - 1):
        if out[i] == flag:
            out[i + 1] = value
            return out
    out.extend([flag, value])
    return out


def _is_cuda_oom_log(log_path: Path, from_offset: int = 0) -> bool:
    try:
        data = log_path.read_bytes()
        if from_offset > 0:
            if from_offset >= len(data):
                return False
            data = data[from_offset:]
        tail = data[-300000:].decode("utf-8", errors="ignore").lower()
    except Exception:
        return False
    needles = (
        "torch.outofmemoryerror",
        "outofmemoryerror",
        "cuda out of memory",
        "cublas_status_alloc_failed",
        "hip out of memory",
    )
    return any(n in tail for n in needles)


def _run_one_attempt(cmd: List[str], cwd: Path, env: Dict[str, str], log_path: Path) -> Tuple[int, float, int]:
    start = time.time()
    try:
        pre_size = log_path.stat().st_size
    except Exception:
        pre_size = 0
    with log_path.open("a", encoding="utf-8") as f:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.time() - start
    return proc.returncode, elapsed, pre_size


def _stable_row_key(row: Dict[str, Any]) -> Optional[int]:
    idx = row.get("problem_idx")
    if isinstance(idx, int) and idx > 0:
        return idx
    return None


def _load_jsonl_by_idx(path: Path, max_samples: int = -1) -> Dict[int, Dict[str, Any]]:
    by_idx: Dict[int, Dict[str, Any]] = {}
    if not path.exists():
        return by_idx
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                if not raw.endswith("\n"):
                    continue
                s = raw.strip()
                if not s:
                    continue
                try:
                    row = json.loads(s)
                except Exception:
                    continue
                if not isinstance(row, dict):
                    continue
                idx = _stable_row_key(row)
                if idx is None:
                    continue
                if max_samples > 0 and idx > max_samples:
                    continue
                by_idx[idx] = row
    except Exception:
        return {}
    return by_idx


def _rows_to_text(by_idx: Dict[int, Dict[str, Any]]) -> str:
    return "".join(json.dumps(by_idx[idx], ensure_ascii=False) + "\n" for idx in sorted(by_idx.keys()))


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


def _seed_partition_cache_from_canonical(
    jsonl_dir: Path,
    task: str,
    method: str,
    world_size: int,
    max_samples: int,
) -> Dict[str, int]:
    tag = _method_tag(method)
    canonical_path = jsonl_dir / f"{task}_{tag}.jsonl"
    canonical = _load_jsonl_by_idx(canonical_path, max_samples=max_samples)
    if not canonical:
        return {
            "canonical_rows": 0,
            "partition_updates": 0,
            "partition_seeded": 0,
            "partition_rows_added": 0,
        }

    canonical_by_pid: Dict[int, Dict[int, Dict[str, Any]]] = {pid: {} for pid in range(world_size)}
    for idx, row in canonical.items():
        pid = (idx - 1) % world_size
        canonical_by_pid[pid][idx] = row

    partition_updates = 0
    partition_seeded = 0
    partition_rows_added = 0
    for pid in range(world_size):
        part_path = _partition_jsonl_path(
            jsonl_dir=jsonl_dir,
            task=task,
            method_tag=tag,
            world_size=world_size,
            partition_id=pid,
        )
        part_rows: Dict[int, Dict[str, Any]] = {}
        for read_path in _partition_read_paths(
            jsonl_dir=jsonl_dir,
            task=task,
            method_tag=tag,
            world_size=world_size,
            partition_id=pid,
        ):
            part_rows.update(_load_jsonl_by_idx(read_path, max_samples=max_samples))
        before_cnt = len(part_rows)
        added_here = 0
        for idx, row in canonical_by_pid[pid].items():
            if idx not in part_rows:
                part_rows[idx] = row
                added_here += 1
        if added_here <= 0:
            continue
        old_text = _read_text_if_exists(part_path)
        new_text = _rows_to_text(part_rows)
        if old_text == new_text:
            continue
        _atomic_write_text(part_path, new_text)
        partition_updates += 1
        if before_cnt == 0 and len(part_rows) > 0:
            partition_seeded += 1
        partition_rows_added += added_here

    return {
        "canonical_rows": len(canonical),
        "partition_updates": partition_updates,
        "partition_seeded": partition_seeded,
        "partition_rows_added": partition_rows_added,
    }


def _resolve_target_samples_for_task(
    task: str,
    max_samples: int,
    task_size_cache: Dict[str, Optional[int]],
) -> Optional[int]:
    if max_samples > 0:
        return int(max_samples)
    if task in task_size_cache:
        return task_size_cache[task]
    try:
        total = int(_count_task_samples(task))
        task_size_cache[task] = total
        print(f"[task-size] task={task} total={total}")
        return total
    except Exception as e:
        task_size_cache[task] = None
        print(f"[warn] task-size unresolved task={task}: {e}")
        return None


def _is_canonical_complete(canonical_path: Path, target_total: int) -> bool:
    if target_total <= 0:
        return False
    rows = _load_jsonl_by_idx(canonical_path, max_samples=target_total)
    if len(rows) < target_total:
        return False
    for idx in range(1, target_total + 1):
        if idx not in rows:
            return False
    return True


def _is_partition_complete(
    jsonl_dir: Path,
    task: str,
    method_tag: str,
    target_total: int,
    world_size: int,
    partition_id: int,
) -> bool:
    expected = _partition_expected_ids(target_total, world_size, partition_id)
    if not expected:
        return True
    rows: Dict[int, Dict[str, Any]] = {}
    for p in _partition_read_paths(
        jsonl_dir=jsonl_dir,
        task=task,
        method_tag=method_tag,
        world_size=world_size,
        partition_id=partition_id,
    ):
        rows.update(_load_jsonl_by_idx(p, max_samples=target_total))
    have = {idx for idx in rows.keys() if ((idx - 1) % world_size) == partition_id}
    return expected.issubset(have)


def _write_log_header(job: PartitionJob, cwd: Path) -> None:
    gpu_str = ",".join(job.gpu_group)
    with job.log_path.open("w", encoding="utf-8") as f:
        f.write(f"[start] {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"[wave_idx] {job.wave_idx}\n")
        f.write(f"[job] {job.name}\n")
        f.write(f"[gpu_group] {gpu_str}\n")
        f.write(f"[cwd] {cwd}\n")
        f.write(f"[partition_jsonl] {job.partition_jsonl_path}\n")
        f.write(f"[cmd] {' '.join(shlex.quote(x) for x in job.cmd)}\n")
        f.write("-" * 80 + "\n")
        f.flush()


def _run_partition_job(
    job: PartitionJob,
    cwd: Path,
    base_env: Dict[str, str],
    oom_retry_on_cuda_oom: bool,
    bs_retry_levels_desc: List[int],
) -> PartitionResult:
    job_start_ts = time.time()
    _write_log_header(job, cwd)
    env = dict(base_env)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(job.gpu_group)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    print(f"[dispatch] wave={job.wave_idx} gpu={','.join(job.gpu_group)} job={job.name}")
    bs_raw = _get_cmd_arg(job.cmd, "--generate_bs")
    try:
        bs_initial = int(bs_raw) if bs_raw is not None else 1
    except Exception:
        bs_initial = 1
    bs_plan = _build_bs_retry_plan(bs_initial, bs_retry_levels_desc)

    attempts: List[Dict[str, Any]] = []
    total_elapsed = 0.0
    final_code = 1
    current_cmd = list(job.cmd)
    attempt_idx = 0
    current_bs = bs_plan[0]
    for candidate_bs in bs_plan:
        if attempt_idx == 0:
            current_bs = candidate_bs
            current_cmd = _set_cmd_arg(current_cmd, "--generate_bs", str(current_bs))
        else:
            current_bs = candidate_bs
            current_cmd = _set_cmd_arg(current_cmd, "--generate_bs", str(current_bs))
            current_cmd = _set_cmd_arg(current_cmd, "--resume_preds_jsonl", "1")
            with job.log_path.open("a", encoding="utf-8") as f:
                f.write("\n" + "=" * 80 + "\n")
                f.write(
                    f"[oom-retry] detected CUDA OOM; retrying with --generate_bs "
                    f"{bs_plan[attempt_idx - 1]} -> {current_bs}\n"
                )
                f.write(f"[oom-retry-cmd] {' '.join(shlex.quote(x) for x in current_cmd)}\n")
                f.write("=" * 80 + "\n")
            print(
                f"[retry] wave={job.wave_idx} gpu={','.join(job.gpu_group)} job={job.name} "
                f"reason=oom generate_bs={bs_plan[attempt_idx - 1]}->{current_bs}"
            )

        code, elapsed, pre_size = _run_one_attempt(current_cmd, cwd=cwd, env=env, log_path=job.log_path)
        total_elapsed += elapsed
        attempt_idx += 1
        oom_here = _is_cuda_oom_log(job.log_path, from_offset=pre_size)
        attempts.append(
            {
                "attempt": attempt_idx,
                "returncode": code,
                "elapsed_sec": elapsed,
                "generate_bs": str(current_bs),
                "reason": "initial" if attempt_idx == 1 else "cuda_oom_retry",
                "oom": bool(oom_here),
            }
        )
        final_code = code
        if final_code == 0:
            break
        if not oom_retry_on_cuda_oom:
            break
        if not oom_here:
            break
        if attempt_idx >= len(bs_plan):
            break

    status = "ok" if final_code == 0 else f"fail({final_code})"
    if final_code == 0 and len(attempts) > 1:
        status = "ok_after_retry"
    print(
        f"[done] wave={job.wave_idx} gpu={','.join(job.gpu_group)} job={job.name} "
        f"status={status} attempts={len(attempts)} t={total_elapsed/60:.1f}m"
    )
    return PartitionResult(
        job=job,
        returncode=final_code,
        elapsed_sec=total_elapsed,
        attempts=attempts,
        start_ts=job_start_ts,
        end_ts=time.time(),
    )


def _build_partition_cmd(
    args: argparse.Namespace,
    task: str,
    method: str,
    max_new_tokens: int,
    generate_bs: int,
    partition_id: int,
    num_partitions: int,
    partition_jsonl_path: Path,
) -> List[str]:
    cmd = [
        args.python_exec,
        "-u",
        args.run_py,
        "--method",
        method,
        "--task",
        task,
        "--prompt",
        args.prompt,
        "--agent_model_names",
        args.agent_model_names,
        "--role_model_map",
        args.role_model_map,
        "--max_new_tokens",
        str(max_new_tokens),
        "--generate_bs",
        str(generate_bs),
        "--save_preds_jsonl",
        "--preds_jsonl_path",
        str(partition_jsonl_path),
        "--resume_preds_jsonl",
        "1",
        "--num_partitions",
        str(num_partitions),
        "--partition_id",
        str(partition_id),
    ]
    if int(getattr(args, "text_mas_context_length", -1)) > 0:
        cmd.extend(["--text_mas_context_length", str(int(args.text_mas_context_length))])
    if method == "baseline":
        cmd.extend(["--model_name", args.model_name])
    if args.max_samples > 0:
        cmd.extend(["--max_samples", str(args.max_samples)])
    if args.agent_devices:
        cmd.extend(["--agent_devices", args.agent_devices])
    if method == "vision_latent_mas_codec_new":
        cmd.extend(
            [
                "--vision_codec_path",
                args.vision_codec_path,
                "--vision_hier_comm_mode",
                args.vision_hier_comm_mode,
                "--vision_hier_join_agg",
                args.vision_hier_join_agg,
                "--vision_codec_dummy_image_count",
                str(args.vision_codec_dummy_image_count),
                "--vision_codec_dummy_image_size",
                str(args.vision_codec_dummy_image_size),
                "--vision_codec_check_dummy_img_tokens",
                str(args.vision_codec_check_dummy_img_tokens),
                "--vision_codec_require_dummy_img_tokens_match",
                str(args.vision_codec_require_dummy_img_tokens_match),
                "--vision_codec_decode_chunks",
                str(args.vision_codec_decode_chunks),
            ]
        )
        if args.vision_codec_dummy_image_counts:
            cmd.extend(["--vision_codec_dummy_image_counts", args.vision_codec_dummy_image_counts])
        if args.vision_codec_dummy_image_sizes:
            cmd.extend(["--vision_codec_dummy_image_sizes", args.vision_codec_dummy_image_sizes])
        if args.vision_codec_dummy_image_spec_json:
            cmd.extend(["--vision_codec_dummy_image_spec_json", args.vision_codec_dummy_image_spec_json])
    elif method == "latent_mas_hybrid":
        cmd.extend(["--latent_steps", str(args.latent_steps)])
    else:
        cmd.extend(
            [
                "--latent_steps",
                str(args.latent_steps),
                "--prefix_postproc",
                args.prefix_postproc,
                "--prefix_len",
                str(args.prefix_len),
            ]
        )
    return cmd


def _build_skipped_result(job: PartitionJob, wave_bs: int, reason: str) -> PartitionResult:
    now_ts = time.time()
    return PartitionResult(
        job=job,
        returncode=0,
        elapsed_sec=0.0,
        attempts=[
            {
                "attempt": 0,
                "returncode": 0,
                "elapsed_sec": 0.0,
                "generate_bs": str(wave_bs),
                "reason": reason,
            }
        ],
        start_ts=now_ts,
        end_ts=now_ts,
    )


def _wave_elapsed_sec(results: List[PartitionResult]) -> float:
    if not results:
        return 0.0
    starts = [r.start_ts for r in results]
    ends = [r.end_ts for r in results]
    return max(0.0, max(ends) - min(starts))


def _summarize_wave(
    wave_idx: int,
    task: str,
    method: str,
    wave_results: List[PartitionResult],
) -> Tuple[Dict[str, Any], List[PartitionResult]]:
    wave_results.sort(key=lambda r: r.job.partition_id)
    skipped = sum(
        1
        for r in wave_results
        if r.attempts and str(r.attempts[0].get("reason", "")).startswith("skipped_")
    )
    ok = sum(1 for r in wave_results if r.returncode == 0)
    failed = [r for r in wave_results if r.returncode != 0]
    wave_elapsed = _wave_elapsed_sec(wave_results)
    print(
        f"[wave-summary] wave={wave_idx} task={task} method={method} "
        f"ok={ok}/{len(wave_results)} failed={len(failed)} skipped={skipped} "
        f"elapsed_min={wave_elapsed/60:.1f}"
    )
    for r in failed:
        print(
            f"  [wave-failed] partition={r.job.partition_id} rc={r.returncode} "
            f"log={r.job.log_path}"
        )

    summary = {
        "wave_idx": wave_idx,
        "task": task,
        "method": method,
        "elapsed_sec": wave_elapsed,
        "ok_partitions": ok,
        "failed_partitions": len(failed),
        "skipped_partitions": skipped,
        "results": [
            {
                "partition_id": r.job.partition_id,
                "gpu_group": r.job.gpu_group,
                "returncode": r.returncode,
                "elapsed_sec": r.elapsed_sec,
                "attempts": r.attempts,
                "partition_jsonl_path": str(r.job.partition_jsonl_path),
                "log_path": str(r.job.log_path),
            }
            for r in wave_results
        ],
    }
    return summary, failed


def _prepare_wave_jobs(
    args: argparse.Namespace,
    *,
    wave_idx: int,
    task: str,
    method: str,
    gpu_groups: List[List[str]],
    jsonl_dir: Path,
    log_dir: Path,
    task_target_total: Dict[str, Optional[int]],
) -> Tuple[List[PartitionJob], List[PartitionResult]]:
    max_new_tokens = TASK_MAX_NEW_TOKENS[task]
    wave_bs = _initial_bs_for_max_new_tokens(
        max_new_tokens=max_new_tokens,
        auto_generate_bs=bool(args.auto_generate_bs),
        fallback_bs=int(args.generate_bs),
    )
    target_total = task_target_total.get(task)
    tag = _method_tag(method)
    seed_cap = int(target_total) if target_total is not None else int(args.max_samples)
    seed_stats = _seed_partition_cache_from_canonical(
        jsonl_dir=jsonl_dir,
        task=task,
        method=method,
        world_size=len(gpu_groups),
        max_samples=seed_cap,
    )
    if seed_stats["canonical_rows"] > 0:
        print(
            f"[seed] wave={wave_idx} task={task} method={method} "
            f"canonical_rows={seed_stats['canonical_rows']} "
            f"partition_updates={seed_stats['partition_updates']} "
            f"added_rows={seed_stats['partition_rows_added']}"
        )

    canonical_complete = False
    partition_complete_map: Dict[int, bool] = {}
    if target_total is not None and target_total > 0:
        canonical_path = jsonl_dir / f"{task}_{tag}.jsonl"
        canonical_complete = _is_canonical_complete(canonical_path, target_total=target_total)
        for partition_id in range(len(gpu_groups)):
            partition_complete_map[partition_id] = _is_partition_complete(
                jsonl_dir=jsonl_dir,
                task=task,
                method_tag=tag,
                target_total=target_total,
                world_size=len(gpu_groups),
                partition_id=partition_id,
            )
        if canonical_complete:
            print(
                f"[skip-wave] wave={wave_idx} task={task} method={method} "
                f"reason=canonical_complete target_total={target_total}"
            )
        elif all(partition_complete_map.values()):
            print(
                f"[skip-wave] wave={wave_idx} task={task} method={method} "
                f"reason=all_partitions_complete target_total={target_total}"
            )

    jobs: List[PartitionJob] = []
    pre_results: List[PartitionResult] = []
    for partition_id, gpu_group in enumerate(gpu_groups):
        part_path = _partition_jsonl_path(
            jsonl_dir=jsonl_dir,
            task=task,
            method_tag=tag,
            world_size=len(gpu_groups),
            partition_id=partition_id,
        )
        log_path = _partition_log_path(
            log_dir=log_dir,
            task=task,
            method_tag=tag,
            world_size=len(gpu_groups),
            partition_id=partition_id,
        )
        cmd = _build_partition_cmd(
            args,
            task=task,
            method=method,
            max_new_tokens=max_new_tokens,
            generate_bs=wave_bs,
            partition_id=partition_id,
            num_partitions=len(gpu_groups),
            partition_jsonl_path=part_path,
        )
        job = PartitionJob(
            wave_idx=wave_idx,
            task=task,
            method=method,
            partition_id=partition_id,
            num_partitions=len(gpu_groups),
            gpu_group=gpu_group,
            cmd=cmd,
            log_path=log_path,
            partition_jsonl_path=part_path,
        )
        skip_partition = bool(canonical_complete) or bool(partition_complete_map.get(partition_id, False))
        if skip_partition:
            reason = "skipped_canonical_complete" if canonical_complete else "skipped_partition_complete"
            if target_total is not None and target_total > 0:
                exp_cnt = _partition_target_count(target_total, len(gpu_groups), partition_id)
            else:
                exp_cnt = None
            print(
                f"[skip-partition] wave={wave_idx} task={task} method={method} "
                f"partition={partition_id} reason={reason} expected={exp_cnt if exp_cnt is not None else 'unknown'}"
            )
            pre_results.append(_build_skipped_result(job=job, wave_bs=wave_bs, reason=reason))
            continue
        jobs.append(job)

    return jobs, pre_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Wave scheduler with partitioned JSONL writes.")
    parser.add_argument("--python_exec", type=str, default=sys.executable)
    parser.add_argument("--run_py", type=str, default="run.py")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7")
    parser.add_argument("--gpus_per_job", type=int, default=1, help="Number of GPUs reserved per partition process.")
    parser.add_argument("--tasks", type=str, default="gsm8k,arc_easy,arc_challenge,gpqa,medqa,mbppplus,humanevalplus,aime2024,aime2025")
    parser.add_argument("--methods", type=str, default="vision_latent_mas_codec_new,text_mas")
    parser.add_argument("--prompt", type=str, default="sequential")
    parser.add_argument("--model_name", type=str, default="", help="Single model name used by baseline method.")
    parser.add_argument("--agent_model_names", type=str, default="Qwen/Qwen3-VL-2B-Thinking,google/gemma-3-4b-it")
    parser.add_argument("--role_model_map", type=str, default='{"planner":1,"critic":0,"refiner":1,"judger":0}')
    parser.add_argument(
        "--agent_devices",
        type=str,
        default="",
        help=(
            "Optional comma-separated model device map passed to run.py. "
            "If omitted and gpus_per_job>1, auto round-robin mapping is used."
        ),
    )
    parser.add_argument("--vision_codec_path", type=str, default="checkpoints/codec.pt")
    parser.add_argument("--vision_hier_comm_mode", type=str, default="chained", choices=["chained", "independent_join"])
    parser.add_argument("--vision_hier_join_agg", type=str, default="concat", choices=["concat", "mean"])
    parser.add_argument(
        "--vision_codec_decode_chunks",
        type=int,
        default=1,
        help="Decode memory in N chunks for codec_new (1 keeps old single-decode behavior).",
    )
    parser.add_argument("--vision_codec_dummy_image_count", type=int, default=1)
    parser.add_argument("--vision_codec_dummy_image_size", type=int, default=224)
    parser.add_argument(
        "--vision_codec_dummy_image_counts",
        type=str,
        default="",
        help="Comma-separated dummy-image counts per model (aligned with --agent_model_names).",
    )
    parser.add_argument(
        "--vision_codec_dummy_image_sizes",
        type=str,
        default="",
        help="Comma-separated dummy-image sizes per model (aligned with --agent_model_names).",
    )
    parser.add_argument(
        "--vision_codec_dummy_image_spec_json",
        type=str,
        default="",
        help="JSON string/path mapping model_name -> {count,size}.",
    )
    parser.add_argument(
        "--vision_codec_check_dummy_img_tokens",
        type=int,
        default=0,
        choices=[0, 1],
    )
    parser.add_argument(
        "--vision_codec_require_dummy_img_tokens_match",
        type=int,
        default=0,
        choices=[0, 1],
    )
    parser.add_argument("--generate_bs", type=int, default=4)
    parser.add_argument(
        "--auto_generate_bs",
        type=int,
        default=1,
        choices=[0, 1],
        help="Auto-set generate_bs by task max_new_tokens: <=2048->12, <=4096->8, else 4.",
    )
    parser.add_argument(
        "--oom_retry_levels",
        type=str,
        default="12,8,4,2,1",
        help="Descending batch-size levels used for OOM retries (comma-separated).",
    )
    parser.add_argument("--latent_steps", type=int, default=1024)
    parser.add_argument("--prefix_postproc", type=str, default="mean_and_norm_match")
    parser.add_argument("--prefix_len", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=-1, help="Passed through to run.py when >0.")
    parser.add_argument(
        "--text_mas_context_length",
        type=int,
        default=-1,
        help="Passed through to run.py; when >0, truncates text_mas inter-agent context.",
    )
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--jsonl_dir", type=str, default="preds_jsonl")
    parser.add_argument("--summary_json", type=str, default="logs/run_exp_partition_pool_summary.json")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="queue",
        choices=["queue", "wave"],
        help="queue: workers immediately pull next partition from later waves; wave: strict barrier between waves.",
    )
    parser.add_argument(
        "--oom_retry_on_cuda_oom",
        type=int,
        default=1,
        choices=[0, 1],
        help="Auto-retry failed partition jobs caused by CUDA OOM.",
    )
    parser.add_argument("--oom_retry_from_bs", type=int, default=4, help="Retry trigger batch size.")
    parser.add_argument("--oom_retry_to_bs", type=int, default=2, help="Retry batch size after OOM.")
    parser.add_argument(
        "--seed_only",
        action="store_true",
        help="Only seed partition JSONLs from canonical cache and exit.",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    # Validate role map early.
    try:
        parsed_role_map = json.loads(args.role_model_map)
    except json.JSONDecodeError as e:
        raise ValueError(f"--role_model_map must be valid JSON. Got: {args.role_model_map}") from e
    if not isinstance(parsed_role_map, dict):
        raise ValueError("--role_model_map must be a JSON object.")

    tasks = parse_csv(args.tasks)
    methods = parse_csv(args.methods)
    unknown_tasks = [t for t in tasks if t not in TASK_MAX_NEW_TOKENS]
    if unknown_tasks:
        raise ValueError(f"Unknown tasks: {unknown_tasks}. Supported: {sorted(TASK_MAX_NEW_TOKENS)}")
    unknown_methods = [m for m in methods if m not in METHODS]
    if unknown_methods:
        raise ValueError(f"Unknown methods: {unknown_methods}. Supported: {list(METHODS)}")
    if "baseline" in methods and not args.model_name.strip():
        raise ValueError("--model_name is required when --methods includes baseline.")

    gpus = parse_csv(args.gpus)
    if not gpus:
        raise ValueError("No GPUs provided in --gpus.")
    gpu_groups, dropped = build_gpu_groups(gpus, int(args.gpus_per_job))
    if not gpu_groups:
        raise ValueError(
            f"Not enough GPUs to form a single worker group: gpus={gpus}, gpus_per_job={args.gpus_per_job}"
        )
    if dropped:
        print(f"[plan] dropping remainder GPUs (group size {args.gpus_per_job}): {dropped}")

    model_names = parse_csv(args.agent_model_names)
    if args.agent_devices:
        agent_devs = parse_csv(args.agent_devices)
        if model_names and len(agent_devs) not in (1, len(model_names)):
            raise ValueError(
                "--agent_devices must provide either 1 device or exactly one per model in --agent_model_names."
            )
    elif int(args.gpus_per_job) > 1 and len(model_names) > 1:
        auto_agent_devs = [f"cuda:{i % int(args.gpus_per_job)}" for i in range(len(model_names))]
        args.agent_devices = ",".join(auto_agent_devs)
        print(f"[plan] auto agent_devices={args.agent_devices}")

    cwd = Path.cwd()
    log_dir = Path(args.log_dir)
    jsonl_dir = Path(args.jsonl_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    jsonl_dir.mkdir(parents=True, exist_ok=True)

    bs_levels_desc = _parse_bs_levels(args.oom_retry_levels)
    if not bs_levels_desc:
        bs_levels_desc = list(DEFAULT_BS_LEVELS_DESC)
    if args.oom_retry_from_bs != 4 or args.oom_retry_to_bs != 2:
        print(
            "[plan] note: --oom_retry_from_bs/--oom_retry_to_bs are deprecated in partition mode; "
            "using --oom_retry_levels instead."
        )

    combos: List[Tuple[str, str]] = [(task, method) for task in tasks for method in methods]
    task_size_cache: Dict[str, Optional[int]] = {}
    task_target_total: Dict[str, Optional[int]] = {}
    for task in tasks:
        task_target_total[task] = _resolve_target_samples_for_task(
            task=task,
            max_samples=int(args.max_samples),
            task_size_cache=task_size_cache,
        )

    print(
        f"[plan] waves={len(combos)} gpu_groups={gpu_groups} "
        f"oom_retry={bool(args.oom_retry_on_cuda_oom)} bs_levels={bs_levels_desc} "
        f"auto_generate_bs={bool(args.auto_generate_bs)} scheduler={args.scheduler}"
    )
    for i, (task, method) in enumerate(combos):
        max_new_tokens = TASK_MAX_NEW_TOKENS[task]
        wave_bs = _initial_bs_for_max_new_tokens(
            max_new_tokens=max_new_tokens,
            auto_generate_bs=bool(args.auto_generate_bs),
            fallback_bs=int(args.generate_bs),
        )
        target_total = task_target_total.get(task)
        target_txt = str(target_total) if target_total is not None else "unknown"
        print(
            f"[wave {i:02d}] task={task} method={method} partitions={len(gpu_groups)} "
            f"max_new_tokens={max_new_tokens} init_bs={wave_bs} target_total={target_txt}"
        )

    if args.seed_only:
        total_added = 0
        total_updates = 0
        total_seeded = 0
        for wave_idx, (task, method) in enumerate(combos):
            target_total = task_target_total.get(task)
            seed_cap = int(target_total) if target_total is not None else int(args.max_samples)
            seed_stats = _seed_partition_cache_from_canonical(
                jsonl_dir=jsonl_dir,
                task=task,
                method=method,
                world_size=len(gpu_groups),
                max_samples=seed_cap,
            )
            total_added += seed_stats["partition_rows_added"]
            total_updates += seed_stats["partition_updates"]
            total_seeded += seed_stats["partition_seeded"]
            print(
                f"[seed-only] wave={wave_idx} task={task} method={method} "
                f"canonical_rows={seed_stats['canonical_rows']} "
                f"partition_updates={seed_stats['partition_updates']} "
                f"added_rows={seed_stats['partition_rows_added']}"
            )
        print(
            f"[seed-only-summary] combos={len(combos)} partition_updates={total_updates} "
            f"partition_seeded={total_seeded} added_rows={total_added}"
        )
        return

    if args.dry_run:
        for i, (task, method) in enumerate(combos):
            tag = _method_tag(method)
            max_new_tokens = TASK_MAX_NEW_TOKENS[task]
            wave_bs = _initial_bs_for_max_new_tokens(
                max_new_tokens=max_new_tokens,
                auto_generate_bs=bool(args.auto_generate_bs),
                fallback_bs=int(args.generate_bs),
            )
            for partition_id, gpu_group in enumerate(gpu_groups):
                part_path = _partition_jsonl_path(
                    jsonl_dir=jsonl_dir,
                    task=task,
                    method_tag=tag,
                    world_size=len(gpu_groups),
                    partition_id=partition_id,
                )
                log_path = _partition_log_path(
                    log_dir=log_dir,
                    task=task,
                    method_tag=tag,
                    world_size=len(gpu_groups),
                    partition_id=partition_id,
                )
                cmd = _build_partition_cmd(
                    args,
                    task=task,
                    method=method,
                    max_new_tokens=max_new_tokens,
                    generate_bs=wave_bs,
                    partition_id=partition_id,
                    num_partitions=len(gpu_groups),
                    partition_jsonl_path=part_path,
                )
                print(f"[dry-run] wave={i} gpu={','.join(gpu_group)} log={log_path}")
                print("  " + " ".join(shlex.quote(x) for x in cmd))
        print("[dry_run] Exiting without launching.")
        return

    base_env = dict(os.environ)
    all_results: List[PartitionResult] = []
    wave_summaries: List[Dict[str, Any]] = []
    if args.scheduler == "wave":
        for wave_idx, (task, method) in enumerate(combos):
            jobs, wave_results = _prepare_wave_jobs(
                args,
                wave_idx=wave_idx,
                task=task,
                method=method,
                gpu_groups=gpu_groups,
                jsonl_dir=jsonl_dir,
                log_dir=log_dir,
                task_target_total=task_target_total,
            )
            lock = threading.Lock()
            threads: List[threading.Thread] = []

            def _runner(j: PartitionJob) -> None:
                res = _run_partition_job(
                    j,
                    cwd=cwd,
                    base_env=base_env,
                    oom_retry_on_cuda_oom=bool(args.oom_retry_on_cuda_oom),
                    bs_retry_levels_desc=bs_levels_desc,
                )
                with lock:
                    wave_results.append(res)

            if jobs:
                for job in jobs:
                    t = threading.Thread(target=_runner, args=(job,), daemon=True)
                    threads.append(t)
                    t.start()
                for t in threads:
                    t.join()

            all_results.extend(sorted(wave_results, key=lambda r: r.job.partition_id))
            wave_summary, _failed = _summarize_wave(
                wave_idx=wave_idx,
                task=task,
                method=method,
                wave_results=wave_results,
            )
            wave_summaries.append(wave_summary)
    else:
        results_by_wave: Dict[int, List[PartitionResult]] = {i: [] for i in range(len(combos))}
        jobs_by_worker: Dict[int, List[PartitionJob]] = {i: [] for i in range(len(gpu_groups))}

        for wave_idx, (task, method) in enumerate(combos):
            jobs, pre_results = _prepare_wave_jobs(
                args,
                wave_idx=wave_idx,
                task=task,
                method=method,
                gpu_groups=gpu_groups,
                jsonl_dir=jsonl_dir,
                log_dir=log_dir,
                task_target_total=task_target_total,
            )
            results_by_wave[wave_idx].extend(pre_results)
            for job in jobs:
                jobs_by_worker[job.partition_id].append(job)

        total_queued = sum(len(v) for v in jobs_by_worker.values())
        print(f"[queue] queued_partition_jobs={total_queued} workers={len(gpu_groups)}")
        for wid in sorted(jobs_by_worker.keys()):
            if jobs_by_worker[wid]:
                print(f"  [queue-worker] worker={wid} jobs={len(jobs_by_worker[wid])}")

        lock = threading.Lock()
        worker_threads: List[threading.Thread] = []

        def _worker(wid: int) -> None:
            for j in jobs_by_worker.get(wid, []):
                res = _run_partition_job(
                    j,
                    cwd=cwd,
                    base_env=base_env,
                    oom_retry_on_cuda_oom=bool(args.oom_retry_on_cuda_oom),
                    bs_retry_levels_desc=bs_levels_desc,
                )
                with lock:
                    results_by_wave[j.wave_idx].append(res)

        for wid in sorted(jobs_by_worker.keys()):
            t = threading.Thread(target=_worker, args=(wid,), daemon=True)
            worker_threads.append(t)
            t.start()
        for t in worker_threads:
            t.join()

        for wave_idx, (task, method) in enumerate(combos):
            wave_results = results_by_wave.get(wave_idx, [])
            if len(wave_results) != len(gpu_groups):
                print(
                    f"[warn] wave={wave_idx} task={task} method={method} "
                    f"results={len(wave_results)} expected={len(gpu_groups)}"
                )
            all_results.extend(sorted(wave_results, key=lambda r: r.job.partition_id))
            wave_summary, _failed = _summarize_wave(
                wave_idx=wave_idx,
                task=task,
                method=method,
                wave_results=wave_results,
            )
            wave_summaries.append(wave_summary)

    total_ok = sum(1 for r in all_results if r.returncode == 0)
    total_failed = [r for r in all_results if r.returncode != 0]
    total_retried = sum(1 for r in all_results if len(r.attempts) > 1)
    summary = {
        "scheduler": args.scheduler,
        "total_waves": len(combos),
        "total_partition_jobs": len(all_results),
        "ok_partition_jobs": total_ok,
        "failed_partition_jobs": len(total_failed),
        "retried_partition_jobs": total_retried,
        "wave_summaries": wave_summaries,
    }
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[summary] waves={len(combos)} ok={total_ok} failed={len(total_failed)} "
        f"retried={total_retried} written={summary_path}"
    )
    if total_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
