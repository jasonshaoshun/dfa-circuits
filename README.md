# Differentiable Circuit Alignment for MIB

This repository builds on the circuit localization track from the
[Mechanistic Interpretability Benchmark (MIB)](https://arxiv.org/abs/2504.13151).
It keeps the original MIB circuit discovery and evaluation workflow, and adds a
differentiable alignment pipeline for transferring node-level circuits from a
source model to a target model.

The usual workflow is:

1. Compute gold MIB circuits with `run_attribution.py`.
2. Train or construct source-to-target alignments with `run_dat.py`.
3. Evaluate both gold and aligned circuits with `run_evaluation.py`.
4. Summarize results as tables with `paper_builders/build_table*.py`.
5. Generate paper figures with `paper_builders/build_figure*.py`.

Batch wrappers for the common runs live in `batch_jobs/`.

## Repository Layout

```text
MIB_circuit_track/       Core MIB dataset, model, metric, and evaluation code
EAP-IG/                  Local EAP/EAP-IG implementation used by attribution
differentiable_alignment/ Alignment model, hooks, and loss helpers
paper_builders/          Scripts for paper tables, figures, and shared helpers
batch_jobs/              SLURM submission scripts for the full pipeline
circuits/                Generated circuit importances and alignment matrices
results/                 Generated evaluation pickles
tables/                  Generated CSV/TeX result tables
figures/                 Generated paper figures
logs/                    SLURM logs
```

## Setup

Create an environment from one of the requirement files, then install the local
project and make sure the EAP-IG sources are present:

```bash
conda env create -f environment.yml
conda activate mib-subgraph
pip install -e .
```

The SLURM wrappers `isambard.sh` and `isambard_single.sh` currently activate a
conda environment named `mib`. Either create that environment name or edit the
wrappers to activate your local environment.

Most runs download datasets and model weights from Hugging Face. For cluster
jobs, set:

```bash
export HUGGINGFACE_TOKEN=...
```

The supported tasks in this code are:

```text
ioi
mcqa
arithmetic_addition
arithmetic_subtraction
arc_easy
arc_challenge
```

Common model ids are:

```text
llama3-1b
llama3-3b
llama3-8b
qwen2.5-0.5b
qwen2.5-1.5b
qwen2.5-3b
```

## 1. Compute Gold Circuits

Gold circuits are computed with `run_attribution.py`. The script loads a model
and MIB task data, computes attribution scores, and writes an `importances.json`
graph under `circuits/`.

Example:

```bash
python run_attribution.py \
  --models llama3-1b \
  --tasks ioi mcqa arithmetic_addition arithmetic_subtraction arc_easy arc_challenge \
  --method EAP-IG-inputs \
  --level node \
  --ablation patching \
  --split train \
  --batch-size 1 \
  --circuit-dir circuits
```

Output layout:

```text
circuits/EAP-IG-inputs_patching_node/ioi_llama3-1b/importances.json
circuits/EAP-IG-inputs_patching_node/mcqa_llama3-1b/importances.json
...
```

Useful options:

```text
--method       EAP, EAP-IG-inputs, EAP-IG-activations, exact
--level        node, edge, or neuron
--ablation     patching, zero, mean, mean-positional, optimal
--split        train, validation, or test
--ig-steps     integrated-gradient steps for EAP-IG methods
--head         limit the dataset after filtering
--num-examples number of examples loaded by the attribution dataset
```

For the default batch run, edit the model/method/task arrays in
`batch_jobs/circuits.sh`, then submit:

```bash
bash batch_jobs/circuits.sh
```

This script skips any circuit whose `importances.json` already exists.

## 2. Calculate Differentiable Alignment

Alignment is handled by `run_dat.py`. It learns a source-node to target-node
matrix from source-model circuits and target-model task data, then uses that
matrix to predict target-model circuit importances.

Important: `run_dat.py` currently supports `--level node` only.

Example:

```bash
python run_dat.py \
  --small-model llama3-1b \
  --large-model llama3-3b \
  --tasks ioi mcqa arithmetic_addition arithmetic_subtraction arc_easy arc_challenge \
  --circuit-dir circuits \
  --method EAP-IG-inputs \
  --ablation patching \
  --level node \
  --train-mode LOO_ioi \
  --ablation-tag none \
  --epochs 3 \
  --lr 1e-1 \
  --batch-size 4 \
  --device-small cuda:0 \
  --device-large cuda:1 \
  --output circuits
```

The `--train-mode` controls which tasks train the alignment:

```text
All                         train on all provided tasks
LOO_<task>                  leave one task out, train on the rest
Train_<task>                train only on one task
```

The `--ablation-tag` controls the alignment/control variant:

```text
none                        learned alignment
random_W                    random matrix baseline
permuted_W                  learned matrix with permuted target nodes
scrambled_s                 scrambled source circuit scores
heuristic_depth_mean        fixed depth-matching heuristic
```

Output layout:

```text
circuits/EAP-IG-inputs__kind-diffalign__src-llama3-1b__train-LOO-ioi__ctrl-none_patching_node/
  alignment_matrix__llama3-1b_to_llama3-3b.pt
  inference_alignment_matrix__llama3-1b_to_llama3-3b.pt
  run_meta_llama3-3b.json
  ioi_llama3-3b/importances.json
  mcqa_llama3-3b/importances.json
  ...
```

For the default source-to-target batch run, edit `MODEL_PAIRS`,
`METHODS_LEVEL_ABLATIONS`, `TRAIN_SPECS`, and controls in:

```bash
bash batch_jobs/alignment.sh
```

For reverse or small-model transfer experiments, use:

```bash
bash batch_jobs/alignment_small_model.sh
```

## 3. Evaluate Circuits

`run_evaluation.py` evaluates a circuit on MIB task data and writes one pickle
per method/task/model/split. Evaluation is used for both original gold circuits
and aligned circuits.

Evaluate a gold circuit:

```bash
python run_evaluation.py \
  --models llama3-3b \
  --tasks ioi mcqa arithmetic_addition arithmetic_subtraction arc_easy arc_challenge \
  --method EAP-IG-inputs \
  --ablation patching \
  --level node \
  --split test \
  --batch-size 12 \
  --circuit-dir circuits \
  --output-dir results
```

Evaluate an aligned circuit by passing the packed method tag produced by
`run_dat.py`:

```bash
python run_evaluation.py \
  --models llama3-3b \
  --tasks ioi mcqa arithmetic_addition arithmetic_subtraction arc_easy arc_challenge \
  --method EAP-IG-inputs__kind-diffalign__src-llama3-1b__train-LOO-ioi__ctrl-none \
  --ablation patching \
  --level node \
  --split test \
  --batch-size 12 \
  --circuit-dir circuits \
  --output-dir results
```

Output layout:

```text
results/EAP-IG-inputs_patching_node/ioi_llama3-3b_test_abs-False.pkl
results/EAP-IG-inputs__kind-diffalign__src-llama3-1b__train-LOO-ioi__ctrl-none_patching_node/ioi_llama3-3b_test_abs-False.pkl
```

Each result pickle contains:

```text
weighted_edge_counts
area_under
area_from_1
average
faithfulnesses
meta
```

For the default batch evaluation, edit the arrays in:

```bash
bash batch_jobs/evaluation.sh
```

For reverse or small-model transfer experiments:

```bash
bash batch_jobs/evaluation_small_model.sh
```

## 4. Print and Build Result Tables

This checkout does not use a standalone `print_results.py`; tables are produced
from evaluation pickles by the scripts in `paper_builders/`.

The main table wrappers are:

```bash
bash batch_jobs/run_build_table1_leaderboard.sh
bash batch_jobs/run_build_table2_detail.sh
bash batch_jobs/run_build_table2_summary.sh
bash batch_jobs/run_build_table3_detail.sh
bash batch_jobs/run_build_table3_summary.sh
bash batch_jobs/run_build_table4_heatmap.sh
```

You can also call the builders directly:

```bash
python paper_builders/build_table1_leaderboard.py \
  --results-dir results \
  --output-dir tables \
  --metric-key area_under \
  --ablation patching \
  --level node \
  --split test

python paper_builders/build_table3_summary.py \
  --results-dir results \
  --output-dir tables \
  --metric-key area_under \
  --ablation patching \
  --level node \
  --split test
```

Typical outputs are written as both CSV and TeX:

```text
tables/table1_leaderboard_<model-pair>.csv
tables/table1_leaderboard_<model-pair>.tex
tables/table3_summary_<model-pair>.csv
tables/table3_summary_<model-pair>.tex
```

The aggregate wrapper is:

```bash
bash batch_jobs/run_build_all_tables.sh
```

Check that file before running it; some table jobs may be commented out during
active experiments.

## 5. Create Paper Figures

Figure scripts also live in `paper_builders/`, with wrappers in `batch_jobs/`.

Common wrappers:

```bash
bash batch_jobs/run_figure1A.sh
bash batch_jobs/run_figure1B.sh
bash batch_jobs/run_figure1C.sh
bash batch_jobs/run_figure2_cross_model_by_method.sh
bash batch_jobs/run_figure4_heatmap.sh
```

Direct examples:

```bash
python paper_builders/build_figure1A_gold_sizes.py \
  --results-dir results \
  --out-dir figures/figure1A_gold_sizes \
  --split test \
  --absolute False \
  --ablation patching \
  --level node

python paper_builders/build_figure2_cross_model_by_method.py \
  --tables-dir tables \
  --results-dir results \
  --out-dir figures/figure2_cross_model_by_method \
  --split test \
  --absolute False \
  --ablation patching \
  --level node \
  --select-by area_under \
  --modes zero_shot,in_distribution,near_distribution,best

python paper_builders/build_figure4_heatmap.py \
  --results-dir results \
  --output-dir figures \
  --metric-key area_under \
  --ablation patching \
  --level node \
  --split test
```

The aggregate wrapper is:

```bash
bash batch_jobs/run_build_all_figures.sh
```

Check that file before running it; it may intentionally enable only a subset of
figures.

## End-to-End Batch Workflow

A standard cluster run looks like:

```bash
# 1. Gold source/target circuits
bash batch_jobs/circuits.sh

# 2. Differentiable source-to-target alignments
bash batch_jobs/alignment.sh

# 3. Gold and aligned evaluation
bash batch_jobs/evaluation.sh

# 4. Tables
bash batch_jobs/run_build_all_tables.sh

# 5. Figures
bash batch_jobs/run_build_all_figures.sh
```

Before launching, edit the arrays in the batch scripts to choose model pairs,
methods, controls, task splits, and batch sizes. The scripts are idempotent where
marker files are checked, so rerunning them usually skips completed jobs.

## Naming Conventions

Gold circuits and results use:

```text
<method>_<ablation>_<level>
```

Example:

```text
EAP-IG-inputs_patching_node
```

Aligned circuits and results use:

```text
<method>__kind-diffalign__src-<source-model>__train-<train-spec>__ctrl-<control>_<ablation>_<level>
```

Example:

```text
EAP-IG-inputs__kind-diffalign__src-llama3-1b__train-LOO-ioi__ctrl-none_patching_node
```

Task/model directories use hyphens for task names:

```text
arithmetic-addition_llama3-3b
arc-easy_qwen2.5-1.5b
```

Command-line task arguments use underscores:

```text
arithmetic_addition
arc_easy
```

## Notes

- Alignment requires source-model gold circuits to exist before `run_dat.py`.
- `run_dat.py` trains on the target model's train split and writes predicted
  target-model circuit importances for each available task.
- `run_evaluation.py` reads circuits from `circuits/` unless explicit
  `--circuit-files` are provided.
- Figure 2 builders use both `results/` and table summaries from `tables/`.
- If Python bytecode cache creation fails on a read-only filesystem, set
  `PYTHONPYCACHEPREFIX` to a writable path such as `/tmp/dfa_node_pycache`.

## Citation

If you use the MIB components, cite the original benchmark:

```bibtex
@article{mib-2025,
  title = {{MIB}: A Mechanistic Interpretability Benchmark},
  author = {Aaron Mueller and Atticus Geiger and Sarah Wiegreffe and Dana Arad and Iv{\'a}n Arcuschin and Adam Belfki and Yik Siu Chan and Jaden Fiotto-Kaufman and Tal Haklay and Michael Hanna and Jing Huang and Rohan Gupta and Yaniv Nikankin and Hadas Orgad and Nikhil Prakash and Anja Reusch and Aruna Sankaranarayanan and Shun Shao and Alessandro Stolfo and Martin Tutek and Amir Zur and David Bau and Yonatan Belinkov},
  year = {2025},
  journal = {CoRR},
  volume = {arXiv:2504.13151},
  url = {https://arxiv.org/abs/2504.13151v1}
}
```
