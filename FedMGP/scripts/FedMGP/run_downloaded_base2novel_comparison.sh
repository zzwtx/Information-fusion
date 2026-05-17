#!/usr/bin/env bash

set -euo pipefail

# Run baseline-vs-TCVP base-to-novel comparisons on downloaded datasets.
#
# Default comparison:
#   base2novel_vit_b16_nctx4
#   base2novel_vit_b16_nctx4_tcvp_a03_h32_noloss
#
# Each task is: train -> base_test -> new_test.
# Two GPU workers are launched, one for GPU 0 and one for GPU 1. Each worker
# runs tasks sequentially, so each GPU has at most one experiment process.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
SHOTS="${SHOTS:-16}"
CFG_FILE="${CFG_FILE:-base2novel_vit_b16}"
DATA_ROOT="${COOP_DATASET:-${ROOT_DIR}/DATA}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/logs/base2novel_comparison_$(date +%Y%m%d_%H%M%S)}"
SKIP_DONE="${SKIP_DONE:-false}"
DRY_RUN="${DRY_RUN:-false}"
VENV_BIN="${VENV_BIN:-/home/xsf/Information-fusion/.venv/bin}"

if [[ -x "${VENV_BIN}/python" ]]; then
  export PATH="${VENV_BIN}:${PATH}"
fi

BASE_RUN_TAG="${BASE_RUN_TAG:-_nctx4}"
TCVP_RUN_TAG="${TCVP_RUN_TAG:-_nctx4_tcvp_a03_h32_noloss}"

TCVP_OPTS=(
  TRAINER.FEDMGP.TCVP_ENABLED True
  TRAINER.FEDMGP.TCVP_ALPHA 0.3
  TRAINER.FEDMGP.TCVP_HIDDEN_DIM 32
  TRAINER.FEDMGP.TCVP_USE_ALIGN_LOSS False
  TRAINER.FEDMGP.TCVP_REG_WEIGHT 0.0
)

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
    food101)
      [[ -d "${DATA_ROOT}/food-101/images" && -d "${DATA_ROOT}/food-101/meta" && -f "${DATA_ROOT}/food-101/split_zhou_Food101.json" ]]
      ;;
    eurosat)
      [[ -d "${DATA_ROOT}/eurosat/2750" && -f "${DATA_ROOT}/eurosat/split_zhou_EuroSAT.json" ]]
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
    food101
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
    tcvp) echo "${TCVP_RUN_TAG}" ;;
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
  local extra_opts=()

  run_tag="$(variant_run_tag "${variant}")"

  if [[ "${variant}" == "tcvp" ]]; then
    extra_opts=("${TCVP_OPTS[@]}")
  fi

  if [[ "${SKIP_DONE}" == "true" ]] && task_done "${dataset}" "${run_tag}"; then
    echo "[gpu ${gpu}] skip completed ${dataset}/${variant} (${CFG_FILE}${run_tag})"
    return 0
  fi

  echo
  echo "============================================================"
  echo "[gpu ${gpu}] ${dataset}/${variant} start"
  echo "config=${CFG_FILE}${run_tag}, shots=${SHOTS}"
  echo "============================================================"

  run_phase "${run_tag}" bash scripts/FedMGP/base2novel_train.sh \
    "${gpu}" "${dataset}" "${SHOTS}" "${CFG_FILE}" "${extra_opts[@]}"

  run_phase "${run_tag}" bash scripts/FedMGP/base2novel_test.sh \
    "${gpu}" "${dataset}" base "${SHOTS}" "${CFG_FILE}" "${extra_opts[@]}"

  run_phase "${run_tag}" bash scripts/FedMGP/base2novel_test.sh \
    "${gpu}" "${dataset}" new "${SHOTS}" "${CFG_FILE}" "${extra_opts[@]}"

  echo "[gpu ${gpu}] ${dataset}/${variant} done"
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

queue0=()
queue1=()
dataset_index=0

for dataset in "${DATASET_LIST[@]}"; do
  if (( dataset_index % 2 == 0 )); then
    queue0+=("${dataset}:baseline")
    queue1+=("${dataset}:tcvp")
  else
    queue0+=("${dataset}:tcvp")
    queue1+=("${dataset}:baseline")
  fi
  ((dataset_index += 1))
done

{
  echo "DATA_ROOT=${DATA_ROOT}"
  echo "SHOTS=${SHOTS}"
  echo "CFG_FILE=${CFG_FILE}"
  echo "BASE_RUN_TAG=${BASE_RUN_TAG}"
  echo "TCVP_RUN_TAG=${TCVP_RUN_TAG}"
  echo "GPU ${GPU0} queue: ${queue0[*]:-<empty>}"
  echo "GPU ${GPU1} queue: ${queue1[*]:-<empty>}"
  echo "SKIP_DONE=${SKIP_DONE}"
  echo "DRY_RUN=${DRY_RUN}"
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
  echo "All comparison tasks finished. Logs: ${LOG_ROOT}"
else
  echo "Some comparison tasks failed. Check logs: ${LOG_ROOT}" >&2
fi

exit "${status}"
