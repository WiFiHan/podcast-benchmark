#!/usr/bin/env bash
#SBATCH -J incomplete_retry
#SBATCH -A m5187
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -t 04:00:00
#SBATCH -o logs/slurm/%x_%j.out
#SBATCH -e logs/slurm/%x_%j.err

set -euo pipefail

# -----------------------------------------------------------------------------
# Retry runner for missing entries in notebooks/results_0423/missing_report.csv
#
# Important:
# - This script is intentionally SINGLE-NODE. It does GPU-level scheduling within
#   one node by assigning CUDA_VISIBLE_DEVICES per launched command.
# - For multi-node execution, launch multiple shards (SHARD_INDEX/SHARD_COUNT)
#   across nodes, e.g. with commands/run_4node_incomplete_plus_volume.sh.
#
# Scheduling model:
# - Each GPU has capacity 3 slots.
# - Short jobs use 1 slot (up to 3 concurrent per GPU).
# - Long jobs use 3 slots (exclusive on that GPU).
#
# Defaults:
# - LONG_TASKS_CSV="llm_embedding_pretraining,whisper_embedding"
# - TASKS_CSV="" (optional; restrict to specific tasks, e.g. "word_embedding")
# - GPU_IDS_CSV="0,1,2,3"
# - MAX_PARALLEL_LONG=1       (parallelism for long tasks, already GPU-bound)
# - MAX_PARALLEL_SHORT=4      (parallelism for short tasks, can exploit idle cores)
# - DRY_RUN=0                 (set 1 to print plan only)
#
# Example:
#   sbatch commands/run_sbatch_for_incomplete.sh
#   MAX_PARALLEL_SHORT=8 sbatch commands/run_sbatch_for_incomplete.sh
# Example to run across multiple nodes(suppose you got 2 nodes by salloc):
# in node1: SHARD_INDEX=0 SHARD_COUNT=2 GPU_IDS_CSV=0,1,2,3 SHORT_SLOTS=1 LONG_SLOTS=1 GPU_CAPACITY=1 MAX_PARALLEL_SHORT=1 MAX_PARALLEL_LONG=1 DRY_RUN=0 bash commands/run_sbatch_for_incomplete.sh
# in node2: SHARD_INDEX=1 SHARD_COUNT=2 GPU_IDS_CSV=0,1,2,3 SHORT_SLOTS=1 LONG_SLOTS=1 GPU_CAPACITY=1 MAX_PARALLEL_SHORT=1 MAX_PARALLEL_LONG=1 DRY_RUN=0 bash commands/run_sbatch_for_incomplete.sh
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/pyproject.toml" ]]; then
    PROJECT_ROOT="${SCRIPT_DIR}"
else
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${PROJECT_ROOT}"

mkdir -p logs/slurm logs/incomplete_retry

if [[ -f "${PROJECT_ROOT}/decoding_env/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${PROJECT_ROOT}/decoding_env/bin/activate"
fi

MISSING_CSV="${PROJECT_ROOT}/notebooks/results_0423/missing_report.csv"
if [[ ! -f "${MISSING_CSV}" ]]; then
    echo "[ERROR] Missing file: ${MISSING_CSV}"
    exit 1
fi

GPU_IDS_CSV="${GPU_IDS_CSV:-0,1,2,3}"
LONG_TASKS_CSV="${LONG_TASKS_CSV:-llm_embedding_pretraining,whisper_embedding}"
TASKS_CSV="${TASKS_CSV:-}"
MAX_PARALLEL_PER_CMD="${MAX_PARALLEL_PER_CMD:-1}"
DRY_RUN="${DRY_RUN:-0}"
SHORT_SLOTS="${SHORT_SLOTS:-1}"
LONG_SLOTS="${LONG_SLOTS:-3}"
GPU_CAPACITY="${GPU_CAPACITY:-3}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"

if (( SHORT_SLOTS < 1 || LONG_SLOTS < 1 || GPU_CAPACITY < 1 )); then
    echo "[ERROR] SHORT_SLOTS, LONG_SLOTS, GPU_CAPACITY must be >= 1"
    exit 1
fi

if (( LONG_SLOTS > GPU_CAPACITY )); then
    echo "[ERROR] LONG_SLOTS (${LONG_SLOTS}) cannot exceed GPU_CAPACITY (${GPU_CAPACITY})"
    exit 1
fi

if (( SHARD_COUNT < 1 )); then
    echo "[ERROR] SHARD_COUNT must be >= 1"
    exit 1
fi

if (( SHARD_INDEX < 0 || SHARD_INDEX >= SHARD_COUNT )); then
    echo "[ERROR] SHARD_INDEX (${SHARD_INDEX}) must satisfy 0 <= SHARD_INDEX < SHARD_COUNT (${SHARD_COUNT})"
    exit 1
fi

IFS=',' read -r -a GPU_IDS <<< "${GPU_IDS_CSV}"
NUM_GPUS=${#GPU_IDS[@]}

if (( NUM_GPUS == 0 )); then
    echo "[ERROR] No GPUs configured. Set GPU_IDS_CSV."
    exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT
PLAN_TSV="${TMP_DIR}/retry_plan.tsv"

python - <<'PY' > "${PLAN_TSV}"
import csv
import os
from collections import defaultdict
from pathlib import Path

project_root = Path.cwd()
csv_path = project_root / 'notebooks' / 'results_0423' / 'missing_report.csv'
long_tasks = {t.strip() for t in os.environ.get('LONG_TASKS_CSV', 'llm_embedding_pretraining,whisper_embedding').split(',') if t.strip()}
task_filters = {t.strip() for t in os.environ.get('TASKS_CSV', '').split(',') if t.strip()}
max_parallel_long = os.environ.get('MAX_PARALLEL_LONG', '1').strip() or '1'
max_parallel_short = os.environ.get('MAX_PARALLEL_SHORT', '2').strip() or '2'
shard_index = int(os.environ.get('SHARD_INDEX', '0'))
shard_count = int(os.environ.get('SHARD_COUNT', '1'))

task_to_dir = {
    'content_noncontent_task': 'content_noncontent',
    'sentence_onset_task': 'sentence_onset',
    'iu_boundary_task': 'iu_boundary',
    'gpt_surprise_task': 'gpt_surprise',
    'gpt_surprise_multiclass_task': 'gpt_surprise_multiclass',
    'pos_task': 'pos',
    'word_embedding_decoding_task': 'word_embedding',
    'whisper_embedding_decoding_task': 'whisper_embedding',
    'volume_level_decoding_task': 'volume_level',
    'llm_decoding_task': 'llm_decoding',
    'llm_embedding_pretraining_task': 'llm_embedding_pretraining',
}

super_lags = defaultdict(set)            # (model, task_dir) -> lags
subject_groups = defaultdict(set)         # (model, task_dir, lag) -> subjects

with csv_path.open() as f:
    reader = csv.DictReader(f)
    for r in reader:
        task_dir = task_to_dir.get(r['task'])
        if not task_dir:
            continue
        if task_filters and (task_dir not in task_filters and r['task'] not in task_filters):
            continue

        cond = r['condition']
        model = r['model']
        lag = r['lag']
        lags = {'0', '250'} if lag == 'all' else {lag}

        if cond == 'persubject_concat':
            for lg in lags:
                super_lags[(model, task_dir)].add(lg)
        elif cond == 'subject_full':
            subj = r['subject']
            if subj in ('', 'all'):
                continue
            for lg in lags:
                subject_groups[(model, task_dir, lg)].add(int(subj))

rows = []

for (model, task), lags in sorted(super_lags.items()):
    lag_arg = ','.join(sorted(lags, key=lambda x: int(x)))
    is_long = task in long_tasks
    max_p = max_parallel_long if is_long else max_parallel_short
    cmd = (
        f"bash job_script_in_interactive_supersubject.sh "
        f"--models {model} --tasks {task} --lags {lag_arg} --max-parallel {max_p}"
    )
    is_long_flag = '1' if is_long else '0'
    rows.append((is_long_flag, task, cmd))

for (model, task, lag), subjects in sorted(subject_groups.items(), key=lambda x: (x[0][0], x[0][1], int(x[0][2]))):
    sub_arg = ','.join(str(s) for s in sorted(subjects))
    is_long = task in long_tasks
    max_p = max_parallel_long if is_long else max_parallel_short
    cmd = (
        f"bash commands/job_script_in_interactive_persubject.sh "
        f"--models {model} --tasks {task} --lags {lag} --subjects {sub_arg} --max-parallel {max_p}"
    )
    is_long_flag = '1' if is_long else '0'
    rows.append((is_long_flag, task, cmd))

# Deduplicate while preserving order
seen = set()
kept_idx = 0
for is_long, task, cmd in rows:
    if cmd in seen:
        continue
    seen.add(cmd)
    if (kept_idx % shard_count) != shard_index:
        kept_idx += 1
        continue
    print(f"{is_long}\t{task}\t{cmd}")
    kept_idx += 1
PY

mapfile -t PLAN_LINES < "${PLAN_TSV}"
TOTAL=${#PLAN_LINES[@]}
if (( TOTAL == 0 )); then
    echo "No retry commands generated from ${MISSING_CSV}"
    exit 0
fi

echo "Generated ${TOTAL} retry commands"
echo "GPU ids: ${GPU_IDS[*]}"
echo "Long tasks: ${LONG_TASKS_CSV}"
if [[ -n "${TASKS_CSV}" ]]; then
    echo "Task filter: ${TASKS_CSV}"
fi
echo "MAX_PARALLEL: long=${MAX_PARALLEL_LONG:-1}, short=${MAX_PARALLEL_SHORT:-4}"
echo "Shard: index=${SHARD_INDEX}/${SHARD_COUNT}"
echo "Slots: short=${SHORT_SLOTS}, long=${LONG_SLOTS}, gpu_capacity=${GPU_CAPACITY}"
echo "DRY_RUN: ${DRY_RUN}"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo ""
    echo "[DRY RUN] Planned commands:"
    nl -ba "${PLAN_TSV}"
    exit 0
fi

# Track capacity usage per GPU
for idx in "${!GPU_IDS[@]}"; do
    gpu_used[$idx]=0
    gpu_jobs[$idx]=0
done

running_pids=()
running_gpu_idx=()
running_slots=()
running_cmd=()

success=0
failed=0
job_index=0
ts="$(date +%Y%m%d_%H%M%S)"

reap_finished() {
    local new_pids=()
    local new_gpu_idx=()
    local new_slots=()
    local new_cmd=()

    for i in "${!running_pids[@]}"; do
        local pid="${running_pids[$i]}"
        local gidx="${running_gpu_idx[$i]}"
        local slots="${running_slots[$i]}"
        local cmd="${running_cmd[$i]}"

        if kill -0 "${pid}" 2>/dev/null; then
            new_pids+=("${pid}")
            new_gpu_idx+=("${gidx}")
            new_slots+=("${slots}")
            new_cmd+=("${cmd}")
        else
            if wait "${pid}"; then
                success=$((success + 1))
                echo "[DONE] ${cmd}"
            else
                failed=$((failed + 1))
                echo "[FAIL] ${cmd}"
            fi
            gpu_used[$gidx]=$(( gpu_used[$gidx] - slots ))
        fi
    done

    running_pids=("${new_pids[@]:-}")
    running_gpu_idx=("${new_gpu_idx[@]:-}")
    running_slots=("${new_slots[@]:-}")
    running_cmd=("${new_cmd[@]:-}")
}

find_gpu_for_slots() {
    local need="$1"
    local i
    for i in "${!GPU_IDS[@]}"; do
        local free=$(( GPU_CAPACITY - gpu_used[$i] ))
        if (( free >= need )); then
            echo "$i"
            return 0
        fi
    done
    return 1
}

for line in "${PLAN_LINES[@]}"; do
    IFS=$'\t' read -r is_long task cmd <<< "${line}"
    slots=${SHORT_SLOTS}
    if [[ "${is_long}" == "1" ]]; then
        slots=${LONG_SLOTS}
    fi

    gpu_idx=""
    while true; do
        if gpu_idx="$(find_gpu_for_slots "${slots}")"; then
            break
        fi
        reap_finished
        sleep 2
    done

    gpu_id="${GPU_IDS[$gpu_idx]}"
    log_file="${PROJECT_ROOT}/logs/incomplete_retry/${ts}_job$(printf '%03d' "${job_index}")_gpu${gpu_id}.log"

    echo "[START] gpu=${gpu_id} slots=${slots} task=${task}"
    echo "        ${cmd}"

    CUDA_VISIBLE_DEVICES="${gpu_id}" bash -lc "${cmd}" > "${log_file}" 2>&1 &
    pid=$!

    running_pids+=("${pid}")
    running_gpu_idx+=("${gpu_idx}")
    running_slots+=("${slots}")
    running_cmd+=("${cmd}")

    gpu_used[$gpu_idx]=$(( gpu_used[$gpu_idx] + slots ))
    gpu_jobs[$gpu_idx]=$(( gpu_jobs[$gpu_idx] + 1 ))
    job_index=$((job_index + 1))

    reap_finished

done

while (( ${#running_pids[@]} > 0 )); do
    reap_finished
    if (( ${#running_pids[@]} > 0 )); then
        sleep 2
    fi
done

echo "========================================"
echo "Retry run complete"
echo "  Total:   ${TOTAL}"
echo "  Success: ${success}"
echo "  Failed:  ${failed}"
for i in "${!GPU_IDS[@]}"; do
    echo "  GPU ${GPU_IDS[$i]} jobs launched: ${gpu_jobs[$i]}"
done
echo "Logs: ${PROJECT_ROOT}/logs/incomplete_retry"
echo "========================================"

if (( failed > 0 )); then
    exit 1
fi
