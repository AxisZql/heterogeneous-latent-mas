#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-${ROOT:-$PWD}}"
if [[ ! -d "$ROOT" ]]; then
  echo "[error] project root does not exist: $ROOT" >&2
  exit 1
fi
cd "$ROOT"

PYTHON="${PYTHON:-python}"
if ! "$PYTHON" -V >/dev/null 2>&1; then
  echo "[error] PYTHON is not executable: $PYTHON" >&2
  exit 1
fi

RUN_NAME="${RUN_NAME:-}"
RUN_PY="${RUN_PY:-run.py}"
if [[ "$RUN_PY" != /* ]]; then
  RUN_PY="$ROOT/$RUN_PY"
fi
if [[ ! -f "$RUN_PY" ]]; then
  echo "[error] run.py entry not found: $RUN_PY" >&2
  exit 1
fi

GPUS="${GPUS:-}"
GPUS_PER_JOB="${GPUS_PER_JOB:-}"
TASKS="${TASKS:-}"
METHODS="${METHODS:-}"
MODEL_NAME="${MODEL_NAME:-}"
AGENT_MODEL_NAMES="${AGENT_MODEL_NAMES:-}"
AGENT_DEVICES="${AGENT_DEVICES:-}"
ROLE_MODEL_MAP="${ROLE_MODEL_MAP:-}"
VISION_CODEC_PATH="${VISION_CODEC_PATH:-}"
VISION_CODEC_DECODE_CHUNKS="${VISION_CODEC_DECODE_CHUNKS:-1}"
VISION_CODEC_DUMMY_IMAGE_COUNT="${VISION_CODEC_DUMMY_IMAGE_COUNT:-1}"
VISION_CODEC_DUMMY_IMAGE_SIZE="${VISION_CODEC_DUMMY_IMAGE_SIZE:-224}"
VISION_CODEC_DUMMY_IMAGE_COUNTS="${VISION_CODEC_DUMMY_IMAGE_COUNTS:-}"
VISION_CODEC_DUMMY_IMAGE_SIZES="${VISION_CODEC_DUMMY_IMAGE_SIZES:-}"
VISION_CODEC_DUMMY_IMAGE_SPEC_JSON="${VISION_CODEC_DUMMY_IMAGE_SPEC_JSON:-}"
VISION_CODEC_CHECK_DUMMY_IMG_TOKENS="${VISION_CODEC_CHECK_DUMMY_IMG_TOKENS:-0}"
VISION_CODEC_REQUIRE_DUMMY_IMG_TOKENS_MATCH="${VISION_CODEC_REQUIRE_DUMMY_IMG_TOKENS_MATCH:-0}"

GENERATE_BS="${GENERATE_BS:-4}"
LATENT_STEPS="${LATENT_STEPS:-1024}"
PREFIX_POSTPROC="${PREFIX_POSTPROC:-mean_and_norm_match}"
PREFIX_LEN="${PREFIX_LEN:-64}"
TEXT_MAS_CONTEXT_LENGTH="${TEXT_MAS_CONTEXT_LENGTH:--1}"
AUTO_GENERATE_BS="${AUTO_GENERATE_BS:-1}"
OOM_RETRY_LEVELS="${OOM_RETRY_LEVELS:-12,8,4,2,1}"
SCHEDULER="${SCHEDULER:-queue}"

required_vars=(
  RUN_NAME
  GPUS
  GPUS_PER_JOB
  TASKS
  METHODS
)
for var_name in "${required_vars[@]}"; do
  if [[ -z "${!var_name:-}" ]]; then
    echo "[error] required env var is missing: ${var_name}" >&2
    exit 1
  fi
done

contains_method() {
  local method="$1"
  [[ ",${METHODS}," == *",${method},"* ]]
}

if contains_method "baseline"; then
  if [[ -z "$MODEL_NAME" ]]; then
    echo "[error] MODEL_NAME is required when METHODS includes baseline" >&2
    exit 1
  fi
fi

if contains_method "vision_latent_mas_codec_new"; then
  if [[ -z "$VISION_CODEC_PATH" ]]; then
    echo "[error] VISION_CODEC_PATH is required when METHODS includes vision_latent_mas_codec_new" >&2
    exit 1
  fi
fi

if contains_method "vision_latent_mas_codec_new" || contains_method "vision_latent_mas_rot" || contains_method "vision_latent_mas_ocr" || contains_method "text_mas" || contains_method "latent_mas_hybrid"; then
  if [[ -z "$AGENT_MODEL_NAMES" ]]; then
    echo "[error] AGENT_MODEL_NAMES is required for multi-agent methods in METHODS" >&2
    exit 1
  fi
  if [[ -z "$ROLE_MODEL_MAP" ]]; then
    echo "[error] ROLE_MODEL_MAP is required for multi-agent methods in METHODS" >&2
    exit 1
  fi
fi

if [[ -z "$ROLE_MODEL_MAP" ]]; then
  ROLE_MODEL_MAP='{}'
fi

LOG_DIR="${LOG_DIR:-logs/${RUN_NAME}_logs_rerun_partitions}"
JSONL_DIR="${JSONL_DIR:-preds_jsonl/${RUN_NAME}}"
SUMMARY_JSON="${SUMMARY_JSON:-logs/${RUN_NAME}_partition_pool_summary.json}"

MERGE_INTERVAL_SEC="${MERGE_INTERVAL_SEC:-60}"
MERGE_LOG="${MERGE_LOG:-logs/merge_partition_${RUN_NAME}.log}"
POOL_LOG="${POOL_LOG:-logs/${RUN_NAME}_partition_pool.launch.log}"
MERGE_PID_FILE="${MERGE_PID_FILE:-logs/pids/merge_partition_${RUN_NAME}.pid}"
POOL_PID_FILE="${POOL_PID_FILE:-logs/pids/${RUN_NAME}_partition_pool.pid}"

mkdir -p "$LOG_DIR" "$JSONL_DIR" logs/pids

# Validate ROLE_MODEL_MAP early so nohup pool doesn't fail immediately on bad JSON.
if [[ -n "$ROLE_MODEL_MAP" ]]; then
if ! "$PYTHON" - "$ROLE_MODEL_MAP" <<'PY'
import json
import sys

s = sys.argv[1]
try:
    obj = json.loads(s)
except Exception as e:
    print(f"[error] ROLE_MODEL_MAP is not valid JSON: {s}\n{e}", file=sys.stderr)
    sys.exit(1)
if not isinstance(obj, dict):
    print(f"[error] ROLE_MODEL_MAP must decode to a JSON object: {s}", file=sys.stderr)
    sys.exit(1)
PY
then
  exit 1
fi
fi

if ! [[ "$GPUS_PER_JOB" =~ ^[0-9]+$ ]] || [[ "$GPUS_PER_JOB" -le 0 ]]; then
  echo "[error] GPUS_PER_JOB must be a positive integer, got: $GPUS_PER_JOB" >&2
  exit 1
fi

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
GPU_COUNT=0
for g in "${GPU_LIST[@]}"; do
  gg="${g//[[:space:]]/}"
  if [[ -n "$gg" ]]; then
    GPU_COUNT=$((GPU_COUNT + 1))
  fi
done
WORLD_SIZE=$((GPU_COUNT / GPUS_PER_JOB))
if [[ "$WORLD_SIZE" -le 0 ]]; then
  echo "[error] computed WORLD_SIZE is 0 (GPUS='$GPUS', GPUS_PER_JOB='$GPUS_PER_JOB')" >&2
  exit 1
fi
echo "[plan] gpus=$GPUS gpus_per_job=$GPUS_PER_JOB world_size=$WORLD_SIZE"
echo "[env] root=$ROOT python=$PYTHON run_py=$RUN_PY"
echo "[config] run_name=$RUN_NAME scheduler=$SCHEDULER"
echo "[config] tasks=$TASKS"
echo "[config] methods=$METHODS"
if [[ -n "$MODEL_NAME" ]]; then
  echo "[config] model_name=$MODEL_NAME"
fi
echo "[config] agent_model_names=$AGENT_MODEL_NAMES"
if [[ -n "$AGENT_DEVICES" ]]; then
  echo "[config] agent_devices=$AGENT_DEVICES"
fi
echo "[config] role_model_map=$ROLE_MODEL_MAP"
echo "[config] vision_codec_path=$VISION_CODEC_PATH"
echo "[config] vision_codec_decode_chunks=$VISION_CODEC_DECODE_CHUNKS"
echo "[config] vision_codec_dummy_image_count=$VISION_CODEC_DUMMY_IMAGE_COUNT"
echo "[config] vision_codec_dummy_image_size=$VISION_CODEC_DUMMY_IMAGE_SIZE"
if [[ -n "$VISION_CODEC_DUMMY_IMAGE_COUNTS" ]]; then
  echo "[config] vision_codec_dummy_image_counts=$VISION_CODEC_DUMMY_IMAGE_COUNTS"
fi
if [[ -n "$VISION_CODEC_DUMMY_IMAGE_SIZES" ]]; then
  echo "[config] vision_codec_dummy_image_sizes=$VISION_CODEC_DUMMY_IMAGE_SIZES"
fi
if [[ -n "$VISION_CODEC_DUMMY_IMAGE_SPEC_JSON" ]]; then
  echo "[config] vision_codec_dummy_image_spec_json=$VISION_CODEC_DUMMY_IMAGE_SPEC_JSON"
fi
echo "[config] vision_codec_check_dummy_img_tokens=$VISION_CODEC_CHECK_DUMMY_IMG_TOKENS"
echo "[config] vision_codec_require_dummy_img_tokens_match=$VISION_CODEC_REQUIRE_DUMMY_IMG_TOKENS_MATCH"
echo "[config] text_mas_context_length=$TEXT_MAS_CONTEXT_LENGTH"
echo "[config] jsonl_dir=$JSONL_DIR log_dir=$LOG_DIR summary_json=$SUMMARY_JSON"

POOL_EXTRA_ARGS=()
if [[ -n "$AGENT_DEVICES" ]]; then
  POOL_EXTRA_ARGS+=(--agent_devices "$AGENT_DEVICES")
fi

echo "[1/3] seed partition cache from canonical jsonl"
"$PYTHON" scripts/partition_runner.py \
  --python_exec "$PYTHON" \
  --run_py "$RUN_PY" \
  --gpus "$GPUS" \
  --gpus_per_job "$GPUS_PER_JOB" \
  --tasks "$TASKS" \
  --methods "$METHODS" \
  --model_name "$MODEL_NAME" \
  --jsonl_dir "$JSONL_DIR" \
  --seed_only

echo "[2/3] start merger (nohup)"
nohup "$PYTHON" scripts/merge_partition_jsonl.py \
  --jsonl_root "$JSONL_DIR" \
  --world_size "$WORLD_SIZE" \
  --interval_sec "$MERGE_INTERVAL_SEC" \
  > "$MERGE_LOG" 2>&1 &
MERGE_PID=$!
echo "$MERGE_PID" > "$MERGE_PID_FILE"
echo "  merger pid=$MERGE_PID log=$MERGE_LOG"

echo "[3/3] start partition pool run (nohup)"
nohup "$PYTHON" scripts/partition_runner.py \
  --python_exec "$PYTHON" \
  --run_py "$RUN_PY" \
  --gpus "$GPUS" \
  --gpus_per_job "$GPUS_PER_JOB" \
  --tasks "$TASKS" \
  --methods "$METHODS" \
  --model_name "$MODEL_NAME" \
  --agent_model_names "$AGENT_MODEL_NAMES" \
  "${POOL_EXTRA_ARGS[@]}" \
  --role_model_map "$ROLE_MODEL_MAP" \
  --vision_codec_path "$VISION_CODEC_PATH" \
  --vision_codec_decode_chunks "$VISION_CODEC_DECODE_CHUNKS" \
  --vision_codec_dummy_image_count "$VISION_CODEC_DUMMY_IMAGE_COUNT" \
  --vision_codec_dummy_image_size "$VISION_CODEC_DUMMY_IMAGE_SIZE" \
  --vision_codec_dummy_image_counts "$VISION_CODEC_DUMMY_IMAGE_COUNTS" \
  --vision_codec_dummy_image_sizes "$VISION_CODEC_DUMMY_IMAGE_SIZES" \
  --vision_codec_dummy_image_spec_json "$VISION_CODEC_DUMMY_IMAGE_SPEC_JSON" \
  --vision_codec_check_dummy_img_tokens "$VISION_CODEC_CHECK_DUMMY_IMG_TOKENS" \
  --vision_codec_require_dummy_img_tokens_match "$VISION_CODEC_REQUIRE_DUMMY_IMG_TOKENS_MATCH" \
  --generate_bs "$GENERATE_BS" \
  --auto_generate_bs "$AUTO_GENERATE_BS" \
  --oom_retry_levels "$OOM_RETRY_LEVELS" \
  --scheduler "$SCHEDULER" \
  --latent_steps "$LATENT_STEPS" \
  --prefix_postproc "$PREFIX_POSTPROC" \
  --prefix_len "$PREFIX_LEN" \
  --text_mas_context_length "$TEXT_MAS_CONTEXT_LENGTH" \
  --log_dir "$LOG_DIR" \
  --jsonl_dir "$JSONL_DIR" \
  --summary_json "$SUMMARY_JSON" \
  > "$POOL_LOG" 2>&1 &
POOL_PID=$!
echo "$POOL_PID" > "$POOL_PID_FILE"
echo "  pool pid=$POOL_PID log=$POOL_LOG"

echo "[done]"
echo "tail logs:"
echo "  tail -f \"$POOL_LOG\""
echo "  tail -f \"$MERGE_LOG\""
echo "summary:"
echo "  cat \"$SUMMARY_JSON\""
