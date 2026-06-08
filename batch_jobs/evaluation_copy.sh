#!/bin/bash
set -euo pipefail

mkdir -p logs/evaluation logs/err logs/err_evaluation results

TASKS=(
  "ioi"
  "mcqa"
  "arithmetic_addition"
  "arithmetic_subtraction"
  "arc_easy"
  "arc_challenge"
)

marker_task="ioi"
marker_task="${marker_task//_/-}"
# Submitting evaluation: eval_align_llama3_1b_to_llama3_8b_EAP_IG_inputs_LOO_ioi_none

MODEL_PAIRS=(
  "llama3-1b:llama3-3b"
  "llama3-1b:llama3-8b"
  "llama3-3b:llama3-8b"
  "qwen2.5-0.5b:qwen2.5-1.5b"
  "qwen2.5-0.5b:qwen2.5-3b"
  "qwen2.5-1.5b:qwen2.5-3b"
  "gpt2-small:gpt2-medium"
  "gpt2-small:gpt2-large"
  "gpt2-medium:gpt2-large"
  "gemma2-2b:gemma2-9b"
  "gemma2-2b:gemma2-27b"
  "gemma2-9b:gemma2-27b"
)

LEVEL="node"
METHODS_LEVEL_ABLATIONS=(
  "EAP-IG-inputs:${LEVEL}:patching"
  "EAP:${LEVEL}:patching"
  "EAP-IG-activations:${LEVEL}:patching"
  # "exact:${LEVEL}:patching"
)

batch_size=8
split="test"
absolute="False"

TRAIN_SPECS=(
  "LOO-ioi"
  "LOO-mcqa"
  "LOO-arithmetic_addition"
  "LOO-arithmetic_subtraction"
  "LOO-arc_easy"
  "LOO-arc_challenge"
  "TRAIN-ioi"
  "TRAIN-mcqa"
  "TRAIN-arithmetic_addition"
  "TRAIN-arithmetic_subtraction"
  "TRAIN-arc_easy"
  "TRAIN-arc_challenge"
)

LOO_CONTROLS=("none" "random_W" "permuted_W" "scrambled_s")
TRAIN_CONTROLS=("none")

safe_tag() {
  echo "$1" | sed 's/[^A-Za-z0-9]/_/g'
}

submit_eval() {
  local job_name="$1"
  shift

  echo "Submitting evaluation: ${job_name}"
  sbatch \
    -J "${job_name}" \
    -o "logs/evaluation/%x.%j.out" \
    -e "logs/err_evaluation/%x.%j.err" \
    isambard_single.sh run_evaluation.py "$@"
}

for pair in "${MODEL_PAIRS[@]}"; do
  IFS=':' read -r source_model target_model <<< "${pair}"

  for mla in "${METHODS_LEVEL_ABLATIONS[@]}"; do
    IFS=':' read -r method level ablation <<< "${mla}"

    safe_source="$(safe_tag "${source_model}")"
    safe_target="$(safe_tag "${target_model}")"
    safe_method="$(safe_tag "${method}")"
    safe_ablation="$(safe_tag "${ablation}")"
    safe_level="$(safe_tag "${level}")"

    gold_result_dir="results/${method}_${ablation}_${level}"
    gold_marker_file="${gold_result_dir}/${marker_task}_${target_model}_${split}_abs-${absolute}.pkl"
    if [[ -f "${gold_marker_file}" ]]; then
      echo "SKIP evaluation gold ${target_model} | ${method} | ${ablation} | ${level} (found ${gold_marker_file})"
    else
    
      echo "file not found: ${gold_marker_file}"
      submit_eval \
        "eval_gold_${safe_target}_${safe_method}_${safe_ablation}_${safe_level}" \
        --models "${target_model}" \
        --tasks "${TASKS[@]}" \
        --method "${method}" \
        --ablation "${ablation}" \
        --batch-size "${batch_size}" \
        --level "${level}" \
        --split "${split}" \
        --circuit-dir circuits \
        --output-dir results
      

    fi

    for train_spec in "${TRAIN_SPECS[@]}"; do
      if [[ "${train_spec}" == LOO-* ]]; then
        controls=("${LOO_CONTROLS[@]}")
      else
        controls=("${TRAIN_CONTROLS[@]}")
      fi

      for control in "${controls[@]}"; do
        method_tag="${method}__kind-diffalign__src-${source_model}__train-${train_spec}__ctrl-${control}"
        result_dir="results/${method_tag}_${ablation}_${level}"
        marker_file="${result_dir}/${marker_task}_${target_model}_${split}_abs-${absolute}.pkl"

        if [[ -f "${marker_file}" ]]; then
          echo "SKIP evaluation ${source_model} -> ${target_model} | ${method} | ${train_spec} | ${control} (found ${marker_file})"
          continue
        fi

        safe_train="$(safe_tag "${train_spec}")"
        safe_control="$(safe_tag "${control}")"

        submit_eval \
          "eval_align_${safe_source}_to_${safe_target}_${safe_method}_${safe_train}_${safe_control}" \
          --models "${target_model}" \
          --tasks "${TASKS[@]}" \
          --method "${method_tag}" \
          --ablation "${ablation}" \
          --batch-size "${batch_size}" \
          --level "${level}" \
          --split "${split}" \
          --circuit-dir circuits \
          --output-dir results
      done
    done


  done
done

echo "All Table 3 evaluation checks complete."
