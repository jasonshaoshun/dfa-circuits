#!/bin/bash
set -euo pipefail

mkdir -p logs/alignment logs/err

MODEL_PAIRS=(
  # "llama3-1b:llama3-3b"
  # "llama3-1b:llama3-8b"
  # "llama3-3b:llama3-8b"
  # "qwen2.5-0.5b:qwen2.5-1.5b"
  # "qwen2.5-0.5b:qwen2.5-3b"
  # "qwen2.5-1.5b:qwen2.5-3b"
  "gpt2-small:gpt2-medium"
  "gpt2-small:gpt2-large"
  "gpt2-medium:gpt2-large"
  # "gemma2-2b:gemma2-9b"
  "gemma2-2b:gemma2-27b"
  "gemma2-9b:gemma2-27b"
)

LEVEL="node"

METHODS_LEVEL_ABLATIONS=(
  "EAP-IG-inputs:${LEVEL}:patching"
  # "EAP:${LEVEL}:patching"
  # "EAP-IG-activations:${LEVEL}:patching"
)

ALL_TASKS=(
  "ioi"
  "mcqa"
  "arithmetic_addition"
  "arithmetic_subtraction"
  "arc_easy"
  "arc_challenge"
)

TRAIN_SPECS=(
  "LOO_ioi"
  "LOO_mcqa"
  "LOO_arithmetic_addition"
  "LOO_arithmetic_subtraction"
  "LOO_arc_easy"
  "LOO_arc_challenge"
  "Train_ioi"
  "Train_mcqa"
  "Train_arithmetic_addition"
  "Train_arithmetic_subtraction"
  "Train_arc_easy"
  "Train_arc_challenge"
)

TRAIN_CONTROLS=("none")

LOO_CONTROLS=(
  "none"
  # "random_W"
  # "permuted_W"
  # "scrambled_s"
  # "heuristic_depth_mean"
)

epochs=3
lr=1e-1
batch_size=1

safe_tag() {
  echo "$1" | sed 's/[^A-Za-z0-9]/_/g'
}

normalize_train_spec() {
  local train_spec="$1"
  if [[ "${train_spec}" == "All" ]]; then
    echo "All"
  elif [[ "${train_spec}" == LOO_* ]]; then
    echo "LOO-${train_spec#LOO_}"
  elif [[ "${train_spec}" == Train_* ]]; then
    echo "TRAIN-${train_spec#Train_}"
  else
    echo "${train_spec}"
  fi
}

submit_job() {
  local small_model="$1"
  local large_model="$2"
  local method="$3"
  local level="$4"
  local ablation="$5"
  local train_spec="$6"
  local control="$7"

  local train_tag method_tag circuit_parent marker_dir marker_matrix
  train_tag="$(normalize_train_spec "${train_spec}")"
  method_tag="${method}__kind-diffalign__src-${small_model}__train-${train_tag}__ctrl-${control}"
  circuit_parent="circuits/${method_tag}_${ablation}_${level}"
  marker_dir="${circuit_parent}/ioi_${large_model}"
  marker_matrix="${circuit_parent}/alignment_matrix__${small_model}_to_${large_model}.pt"

  if [[ -d "${marker_dir}" && -f "${marker_matrix}" ]]; then
    echo "SKIP training ${small_model} -> ${large_model} | ${method} | ${train_spec} | ${control} (found ${marker_dir} and ${marker_matrix})"
    return 0
  fi

  local safe_small safe_large safe_method safe_level safe_ablation safe_train safe_control
  safe_small="$(safe_tag "${small_model}")"
  safe_large="$(safe_tag "${large_model}")"
  safe_method="$(safe_tag "${method}")"
  safe_level="$(safe_tag "${level}")"
  safe_ablation="$(safe_tag "${ablation}")"
  safe_train="$(safe_tag "${train_spec}")"
  safe_control="$(safe_tag "${control}")"

  local experiment_id="align_${safe_small}_to_${safe_large}_${safe_method}_${safe_level}_${safe_ablation}_${safe_train}_${safe_control}"
  echo "Submitting ${experiment_id}"

  sbatch \
    -J "${experiment_id}" \
    -o logs/alignment/%x.%j.out \
    -e logs/err/%x.%j.err \
    isambard.sh run_dat.py \
      --small-model "${small_model}" \
      --large-model "${large_model}" \
      --tasks "${ALL_TASKS[@]}" \
      --circuit-dir circuits \
      --method "${method}" \
      --level "${level}" \
      --ablation "${ablation}" \
      --train-mode "${train_spec}" \
      --ablation-tag "${control}" \
      --epochs "${epochs}" \
      --lr "${lr}" \
      --batch-size "${batch_size}" \
      --output circuits/
}

for pair in "${MODEL_PAIRS[@]}"; do
  IFS=':' read -r small_model large_model <<< "${pair}"

  for mla in "${METHODS_LEVEL_ABLATIONS[@]}"; do
    IFS=':' read -r method level ablation <<< "${mla}"

    for train_spec in "${TRAIN_SPECS[@]}"; do
      if [[ "${train_spec}" == LOO_* ]]; then
        controls=("${LOO_CONTROLS[@]}")
      else
        controls=("${TRAIN_CONTROLS[@]}")
      fi

      for control in "${controls[@]}"; do
        submit_job \
          "${small_model}" \
          "${large_model}" \
          "${method}" \
          "${level}" \
          "${ablation}" \
          "${train_spec}" \
          "${control}"
      done
    done
  done
done
