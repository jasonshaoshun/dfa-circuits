#!/bin/bash

mkdir -p logs/run_attribution
mkdir -p logs/evaluation
mkdir -p logs/err

# Define your experimental configurations
MODELS=(
  # "llama3-1b"
  # "llama3-3b"
  # "llama3-8b"
  # "qwen2.5-0.5b"
  # "qwen2.5-1.5b"
  # "qwen2.5-3b"
  # "gpt2-small"
  # "gpt2-medium"
  # "gpt2-large"
  "gemma2-2b"
  "gemma2-9b"
  "gemma2-27b"
)

LEVEL="node"

# METHODS_LEVEL_ABLATIONS=(
#   # NAP (EAP @ node level)
#   "EAP:${LEVEL}:patching"
#   "EAP:${LEVEL}:mean"
#   "EAP:${LEVEL}:zero"
# #   "EAP:${LEVEL}:mean-positional"
# #   "EAP:${LEVEL}:optimal"

#   # NAP-IG-inputs (EAP-IG-inputs @ node level) -- patching only
#   "EAP-IG-inputs:${LEVEL}:patching"

#   # NAP-IG-activations (EAP-IG-activations @ node level)
#   "EAP-IG-activations:${LEVEL}:patching"
#   "EAP-IG-activations:${LEVEL}:mean"
#   "EAP-IG-activations:${LEVEL}:zero"
# #   "EAP-IG-activations:${LEVEL}:mean-positional"
# #   "EAP-IG-activations:${LEVEL}:optimal"

#   # "exact" at node level (implemented, though README claims exact is edge-level)
#   "exact:${LEVEL}:patching"
#   "exact:${LEVEL}:mean"
#   "exact:${LEVEL}:zero"
# #   "exact:${LEVEL}:mean-positional"
# #   "exact:${LEVEL}:optimal"
# )

METHODS_LEVEL_ABLATIONS=(
  "EAP-IG-inputs:${LEVEL}:patching"
  "EAP:${LEVEL}:patching"
  "EAP-IG-activations:${LEVEL}:patching"
)

# Define tasks
TASKS=("ioi" "mcqa" "arithmetic_addition" "arithmetic_subtraction" "arc_easy" "arc_challenge")

# Fixed batch size for all experiments
BATCH_SIZE=1

# Loop through all combinations
for model in "${MODELS[@]}"; do
  for method_level_ablation in "${METHODS_LEVEL_ABLATIONS[@]}"; do
    IFS=':' read -r method level ablation <<< "$method_level_ablation"

    for task in "${TASKS[@]}"; do
      experiment_id="run_${model}_${task}_${method}_${level}_${ablation}"

      output_dir="circuits/${method}_${ablation}_${level}/${task//_/-}_${model}"
      marker_file="${output_dir}/importances.json"

      if [ -f "$marker_file" ]; then
        echo "SKIP gold circuit: ${experiment_id} (found ${marker_file})"
        continue
      fi

      echo "Submitting: ${experiment_id}"
      echo "Output dir: ${output_dir}"

      sbatch \
        -J "${experiment_id}" \
        -o "logs/run_attribution/%x.%j.out" \
        -e "logs/err/%x.%j.err" \
        isambard_single.sh run_attribution.py \
          --models "$model" \
          --tasks "$task" \
          --method "$method" \
          --level "$level" \
          --ablation "$ablation" \
          --batch-size "$BATCH_SIZE" \
          --circuit-dir "circuits/"
    done
  done
done

echo "All jobs checked/submitted!"
