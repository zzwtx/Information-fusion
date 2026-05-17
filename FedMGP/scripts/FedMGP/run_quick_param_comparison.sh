#!/usr/bin/env bash

set -euo pipefail

# Quick baseline-vs-parameter comparison for FedMGP base-to-novel experiments.
#
# Usage examples:
#
#   # Use the default TCVP setting in this script
#   bash scripts/FedMGP/run_quick_param_comparison.sh
#
#   # Compare baseline with a custom parameter set
#   EXP_NAME=tcvp_localmeta EXP_METHOD=TCVP-localmeta bash scripts/FedMGP/run_quick_param_comparison.sh \
#     TRAINER.FEDMGP.TCVP_ENABLED True \
#     TRAINER.FEDMGP.TCVP_ALPHA 0.3 \
#     TRAINER.FEDMGP.TCVP_HIDDEN_DIM 32 \
#     TRAINER.FEDMGP.TCVP_USE_ALIGN_LOSS False \
#     TRAINER.FEDMGP.TCVP_REG_WEIGHT 0.0 \
#     TRAINER.FEDMGP.TCVP_AGGREGATE_META_NET False
#
# Defaults:
# - Datasets: small downloaded datasets only, excluding Food101
# - Two GPU workers: GPU 0 and GPU 1, each runs at most one experiment process
# - Each task: train -> base_test -> new_test
# - Summary: writes Dataset/Method/Local/Base/Novel/HM/CM to summary.tsv and summary.md

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
SHOTS="${SHOTS:-16}"
CFG_FILE="${CFG_FILE:-base2novel_vit_b16}"
DATA_ROOT="${COOP_DATASET:-${ROOT_DIR}/DATA}"
VENV_BIN="${VENV_BIN:-/home/xsf/Information-fusion/.venv/bin}"

BASE_METHOD="${BASE_METHOD:-baseline}"
BASE_RUN_TAG="${BASE_RUN_TAG:-_nctx4}"

EXP_NAME="${EXP_NAME:-tcvp_a03_h32_noloss}"
EXP_METHOD="${EXP_METHOD:-${EXP_NAME}}"
EXP_RUN_TAG="${EXP_RUN_TAG:-_nctx4_${EXP_NAME}}"

LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/logs/quick_param_comparison_${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"
SUMMARY_TSV="${SUMMARY_TSV:-${LOG_ROOT}/summary.tsv}"
SUMMARY_MD="${SUMMARY_MD:-${LOG_ROOT}/summary.md}"

SKIP_DONE="${SKIP_DONE:-true}"
DRY_RUN="${DRY_RUN:-false}"

if [[ -x "${VENV_BIN}/python" ]]; then
  export PATH="${VENV_BIN}:${PATH}"
fi

DEFAULT_EXP_OPTS=(
  TRAINER.FEDMGP.TCVP_ENABLED True
  TRAINER.FEDMGP.TCVP_ALPHA 0.3
  TRAINER.FEDMGP.TCVP_HIDDEN_DIM 32
  TRAINER.FEDMGP.TCVP_USE_ALIGN_LOSS False
  TRAINER.FEDMGP.TCVP_REG_WEIGHT 0.0
)

if [[ "$#" -gt 0 ]]; then
  EXP_OPTS_ARRAY=("$@")
elif [[ -n "${EXP_OPTS:-}" ]]; then
  read -r -a EXP_OPTS_ARRAY <<< "${EXP_OPTS}"
else
  EXP_OPTS_ARRAY=("${DEFAULT_EXP_OPTS[@]}")
fi

mkdir -p "${LOG_ROOT}"

dataset_ready() {
  local dataset="$1"

  case "${dataset}" in
    caltech101)
      [[ -d "${DATA_ROOT}/caltech-101/101_ObjectCategories" && -f "${DATA_ROOT}/caltech-101/split_zhou_Caltech101.json" ]]
      ;;
    dtd)
      [[ -d "${DATA_ROOT}/dtd/images" && -f "${DATA_ROOT}/dtd/split_zhou_DescribableTextures.json" ]]
      ;;
    oxford_pets)
      [[ -d "${DATA_ROOT}/oxford_pets/images" && -d "${DATA_ROOT}/oxford_pets/annotations" && -f "${DATA_ROOT}/oxford_pets/split_zhou_OxfordPets.json" ]]
      ;;
    oxford_flowers)
      [[ -d "${DATA_ROOT}/oxford_flowers/jpg" && -f "${DATA_ROOT}/oxford_flowers/imagelabels.mat" && -f "${DATA_ROOT}/oxford_flowers/cat_to_name.json" && -f "${DATA_ROOT}/oxford_flowers/split_zhou_OxfordFlowers.json" ]]
      ;;
    eurosat)
      [[ -d "${DATA_ROOT}/eurosat/2750" && -f "${DATA_ROOT}/eurosat/split_zhou_EuroSAT.json" ]]
      ;;
    food101)
      [[ -d "${DATA_ROOT}/food-101/images" && -d "${DATA_ROOT}/food-101/meta" && -f "${DATA_ROOT}/food-101/split_zhou_Food101.json" ]]
      ;;
    *)
      return 1
      ;;
  esac
}

discover_datasets() {
  local candidates=(
    caltech101
    dtd
    oxford_pets
    oxford_flowers
    eurosat
  )

  local found=()
  local dataset
  for dataset in "${candidates[@]}"; do
    if dataset_ready "${dataset}"; then
      found+=("${dataset}")
    else
      echo "[skip] ${dataset}: dataset files are incomplete under ${DATA_ROOT}" >&2
    fi
  done

  printf '%s\n' "${found[@]}"
}

if [[ -n "${DATASETS:-}" ]]; then
  read -r -a DATASET_LIST <<< "${DATASETS}"
else
  mapfile -t DATASET_LIST < <(discover_datasets)
fi

if [[ "${#DATASET_LIST[@]}" -eq 0 ]]; then
  echo "No ready datasets found. Set DATASETS=\"caltech101 dtd ...\" or check ${DATA_ROOT}." >&2
  exit 1
fi

variant_run_tag() {
  case "$1" in
    baseline) echo "${BASE_RUN_TAG}" ;;
    experiment) echo "${EXP_RUN_TAG}" ;;
    *) echo "Unknown variant: $1" >&2; return 1 ;;
  esac
}

variant_method() {
  case "$1" in
    baseline) echo "${BASE_METHOD}" ;;
    experiment) echo "${EXP_METHOD}" ;;
    *) echo "Unknown variant: $1" >&2; return 1 ;;
  esac
}

output_base_dir() {
  local dataset="$1"
  local run_tag="$2"
  echo "output/${dataset}/FedMGP/${CFG_FILE}${run_tag}/${SHOTS}shots/seed1"
}

task_done() {
  local dataset="$1"
  local run_tag="$2"
  local base_dir
  base_dir="$(output_base_dir "${dataset}" "${run_tag}")"

  [[ -f "${base_dir}/base/log.txt" \
    && -f "${base_dir}/base_test/log.txt" \
    && -f "${base_dir}/new_test/log.txt" \
    && ( -f "${base_dir}/base/best_model.pt" || -f "${base_dir}/base/last_model.pt" ) ]]
}

run_phase() {
  local run_tag="$1"
  shift

  echo
  echo ">>> RUN_TAG=${run_tag} $*"
  if [[ "${DRY_RUN}" == "true" ]]; then
    return 0
  fi
  RUN_TAG="${run_tag}" COOP_DATASET="${DATA_ROOT}" "$@"
}

run_task() {
  local gpu="$1"
  local dataset="$2"
  local variant="$3"
  local run_tag
  local method
  local extra_opts=()

  run_tag="$(variant_run_tag "${variant}")"
  method="$(variant_method "${variant}")"

  if [[ "${variant}" == "experiment" ]]; then
    extra_opts=("${EXP_OPTS_ARRAY[@]}")
  fi

  if [[ "${SKIP_DONE}" == "true" ]] && task_done "${dataset}" "${run_tag}"; then
    echo "[gpu ${gpu}] skip completed ${dataset}/${method} (${CFG_FILE}${run_tag})"
    return 0
  fi

  echo
  echo "============================================================"
  echo "[gpu ${gpu}] ${dataset}/${method} start"
  echo "config=${CFG_FILE}${run_tag}, shots=${SHOTS}"
  echo "============================================================"

  run_phase "${run_tag}" bash scripts/FedMGP/base2novel_train.sh \
    "${gpu}" "${dataset}" "${SHOTS}" "${CFG_FILE}" "${extra_opts[@]}"

  run_phase "${run_tag}" bash scripts/FedMGP/base2novel_test.sh \
    "${gpu}" "${dataset}" base "${SHOTS}" "${CFG_FILE}" "${extra_opts[@]}"

  run_phase "${run_tag}" bash scripts/FedMGP/base2novel_test.sh \
    "${gpu}" "${dataset}" new "${SHOTS}" "${CFG_FILE}" "${extra_opts[@]}"

  echo "[gpu ${gpu}] ${dataset}/${method} done"
}

worker() {
  local gpu="$1"
  shift

  local task dataset variant
  for task in "$@"; do
    dataset="${task%%:*}"
    variant="${task#*:}"
    run_task "${gpu}" "${dataset}" "${variant}"
  done
}

write_summary() {
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "DRY_RUN=true: skip summary generation"
    return 0
  fi

  DATASETS_JOINED="${DATASET_LIST[*]}" \
  CFG_FILE="${CFG_FILE}" \
  SHOTS="${SHOTS}" \
  BASE_RUN_TAG="${BASE_RUN_TAG}" \
  EXP_RUN_TAG="${EXP_RUN_TAG}" \
  BASE_METHOD="${BASE_METHOD}" \
  EXP_METHOD="${EXP_METHOD}" \
  SUMMARY_TSV="${SUMMARY_TSV}" \
  SUMMARY_MD="${SUMMARY_MD}" \
  python - <<'PY'
from pathlib import Path
import os
import re

datasets = os.environ["DATASETS_JOINED"].split()
cfg_file = os.environ["CFG_FILE"]
shots = os.environ["SHOTS"]
summary_tsv = Path(os.environ["SUMMARY_TSV"])
summary_md = Path(os.environ["SUMMARY_MD"])

variants = [
    (os.environ["BASE_METHOD"], os.environ["BASE_RUN_TAG"]),
    (os.environ["EXP_METHOD"], os.environ["EXP_RUN_TAG"]),
]

def parse_average(path: Path):
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    rows = []
    for line in text.splitlines():
        if "|  Average" in line or "| Average" in line:
            nums = [float(x) for x in re.findall(r"\d+\.\d+", line)]
            rows.append(nums)
    return rows[-1] if rows else None

def result_for(dataset: str, method: str, run_tag: str):
    root = Path("output") / dataset / "FedMGP" / f"{cfg_file}{run_tag}" / f"{shots}shots" / "seed1"
    base_avg = parse_average(root / "base_test" / "log.txt")
    new_avg = parse_average(root / "new_test" / "log.txt")
    if not base_avg or not new_avg or len(base_avg) < 2:
        return None
    local = base_avg[0]
    base = base_avg[1]
    novel = new_avg[-1]
    hm = 0.0 if base + novel == 0 else 2 * base * novel / (base + novel)
    cm = (local + hm) / 2
    return [dataset, method, local, base, novel, hm, cm]

rows = []
missing = []
for dataset in datasets:
    for method, run_tag in variants:
        row = result_for(dataset, method, run_tag)
        if row is None:
            missing.append((dataset, method, run_tag))
        else:
            rows.append(row)

summary_tsv.parent.mkdir(parents=True, exist_ok=True)
with summary_tsv.open("w", encoding="utf-8") as f:
    f.write("Dataset\tMethod\tLocal\tBase\tNovel\tHM\tCM\n")
    for dataset, method, local, base, novel, hm, cm in rows:
        f.write(f"{dataset}\t{method}\t{local:.4f}\t{base:.4f}\t{novel:.4f}\t{hm:.4f}\t{cm:.4f}\n")

with summary_md.open("w", encoding="utf-8") as f:
    f.write("| Dataset | Method | Local | Base | Novel | HM | CM |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|\n")
    for dataset, method, local, base, novel, hm, cm in rows:
        f.write(f"| {dataset} | {method} | {local:.4f} | {base:.4f} | {novel:.4f} | {hm:.4f} | {cm:.4f} |\n")
    if missing:
        f.write("\nIncomplete runs skipped:\n\n")
        for dataset, method, run_tag in missing:
            f.write(f"- {dataset} / {method} ({cfg_file}{run_tag})\n")

print(f"Summary written to {summary_tsv}")
print(f"Markdown summary written to {summary_md}")
if missing:
    print("Incomplete runs skipped:")
    for dataset, method, run_tag in missing:
        print(f"  - {dataset} / {method} ({cfg_file}{run_tag})")
PY
}

queue0=()
queue1=()
dataset_index=0

for dataset in "${DATASET_LIST[@]}"; do
  if (( dataset_index % 2 == 0 )); then
    queue0+=("${dataset}:baseline")
    queue1+=("${dataset}:experiment")
  else
    queue0+=("${dataset}:experiment")
    queue1+=("${dataset}:baseline")
  fi
  ((dataset_index += 1))
done

{
  echo "DATA_ROOT=${DATA_ROOT}"
  echo "SHOTS=${SHOTS}"
  echo "CFG_FILE=${CFG_FILE}"
  echo "BASE_METHOD=${BASE_METHOD}"
  echo "BASE_RUN_TAG=${BASE_RUN_TAG}"
  echo "EXP_METHOD=${EXP_METHOD}"
  echo "EXP_RUN_TAG=${EXP_RUN_TAG}"
  echo "EXP_OPTS=${EXP_OPTS_ARRAY[*]}"
  echo "DATASETS=${DATASET_LIST[*]}"
  echo "GPU ${GPU0} queue: ${queue0[*]:-<empty>}"
  echo "GPU ${GPU1} queue: ${queue1[*]:-<empty>}"
  echo "SKIP_DONE=${SKIP_DONE}"
  echo "DRY_RUN=${DRY_RUN}"
  echo "SUMMARY_TSV=${SUMMARY_TSV}"
  echo "SUMMARY_MD=${SUMMARY_MD}"
  echo "python=$(command -v python)"
} | tee "${LOG_ROOT}/plan.txt"

pids=()

if [[ "${#queue0[@]}" -gt 0 ]]; then
  worker "${GPU0}" "${queue0[@]}" > >(tee "${LOG_ROOT}/gpu${GPU0}.log") 2>&1 &
  pids+=("$!")
fi

if [[ "${#queue1[@]}" -gt 0 ]]; then
  worker "${GPU1}" "${queue1[@]}" > >(tee "${LOG_ROOT}/gpu${GPU1}.log") 2>&1 &
  pids+=("$!")
fi

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${status}" -eq 0 ]]; then
  write_summary
  echo "All quick comparison tasks finished. Logs: ${LOG_ROOT}"
else
  write_summary || true
  echo "Some quick comparison tasks failed. Check logs: ${LOG_ROOT}" >&2
fi

exit "${status}"
