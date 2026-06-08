import argparse
import os
import pickle
import sys
from functools import partial
from typing import Dict

import torch
from huggingface_hub import hf_hub_download
from transformer_lens import HookedTransformer

sys.path.append(os.path.join(os.path.dirname(__file__), 'EAP-IG', 'src'))
from eap.graph import Graph

from MIB_circuit_track.dataset import HFEAPDataset
from MIB_circuit_track.evaluation import evaluate_area_under_curve, evaluate_area_under_roc
from MIB_circuit_track.metrics import get_metric
from MIB_circuit_track.utils import COL_MAPPING, MODEL_NAME_TO_FULLNAME, TASKS_TO_HF_NAMES
from run_attribution import load_interpbench_model


MEMORY_EFFICIENT_MODEL_NAMES = {
    'qwen2.5',
    'qwen2.5-0.5b',
    'qwen2.5-1.5b',
    'qwen2.5-3b',
    'gemma2',
    'gemma2-2b',
    'gemma2-9b',
    'gemma2-27b',
    'llama3',
    'llama3-1b',
    'llama3-3b',
    'llama3-8b',
}


def parse_packed_method_tag(method_tag: str | None) -> Dict[str, str]:
    """Parse Option-A packed tags like:
    EAP-IG-inputs__kind-diffalign__src-llama3-1b__train-LOO-ioi__ctrl-none
    """
    if not method_tag:
        return {
            'method_base': 'unknown',
            'kind': 'unknown',
            'source_model': '',
            'train_spec': '',
            'control': '',
        }

    parts = method_tag.split('__')
    info: Dict[str, str] = {
        'method_base': parts[0],
        'kind': 'gold',
        'source_model': '',
        'train_spec': '',
        'control': '',
    }

    for segment in parts[1:]:
        if '-' not in segment:
            continue
        key, value = segment.split('-', 1)
        if key == 'kind':
            info['kind'] = value
        elif key == 'src':
            info['source_model'] = value
        elif key == 'train':
            info['train_spec'] = value
        elif key == 'ctrl':
            info['control'] = value
        else:
            info[key] = value

    return info


def resolve_model(model_name: str) -> HookedTransformer:
    if model_name in MEMORY_EFFICIENT_MODEL_NAMES:
        model = HookedTransformer.from_pretrained(
            MODEL_NAME_TO_FULLNAME[model_name],
            attn_implementation='eager',
            dtype=torch.bfloat16,
        )
    elif model_name == 'interpbench':
        model = load_interpbench_model()
    else:
        model = HookedTransformer.from_pretrained(MODEL_NAME_TO_FULLNAME[model_name])

    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.ungroup_grouped_query_attention = True
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, nargs='+', required=True)
    parser.add_argument('--tasks', type=str, nargs='+', required=True)
    parser.add_argument('--ablation', type=str, choices=['patching', 'zero', 'mean', 'mean-positional', 'optimal'], default='patching')
    parser.add_argument('--split', type=str, choices=['train', 'validation', 'test'], default='validation')
    parser.add_argument('--method', type=str, default=None, help='Method tag used to infer circuit file name')
    parser.add_argument('--level', type=str, choices=['edge', 'node', 'neuron'], default='edge')
    parser.add_argument('--absolute', action='store_true')
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--head', type=int, default=None)
    parser.add_argument('--circuit-dir', type=str, default='circuits')
    parser.add_argument('--circuit-files', type=str, nargs='+', default=None)
    parser.add_argument('--output-dir', type=str, default='results')
    args = parser.parse_args()

    parsed_method = parse_packed_method_tag(args.method)
    method_base = parsed_method['method_base'] if args.method else 'custom'
    apply_greedy = method_base in ['information-flow-routes']

    reference_graph = None
    if 'interpbench' in args.models:
        reference_graph = Graph.from_json(
            hf_hub_download('mib-bench/interpbench', filename='interpbench_graph.json')
        )

    circuit_file_index = 0
    for model_name in args.models:
        model = resolve_model(model_name)

        for task in args.tasks:
            task_dir = f"{task.replace('_', '-')}_{model_name}"
            if task_dir not in COL_MAPPING:
                print(f"Skipping {task} for {model_name}: unsupported in COL_MAPPING")
                continue

            if args.circuit_files is not None:
                p = args.circuit_files[circuit_file_index]
                circuit_file_index += 1
            else:
                if args.method is None:
                    raise ValueError('--method is required when --circuit-files is not provided')
                method_name_saveable = f"{args.method}_{args.ablation}_{args.level}"
                p = os.path.join(args.circuit_dir, method_name_saveable, task_dir, 'importances.json')

            print(f"Loading circuit from {p}")
            if not os.path.exists(p):
                print(f"Warning: missing circuit file {p}; skipping")
                continue

            if p.endswith('.json'):
                graph = Graph.from_json(p)
            elif p.endswith('.pt'):
                graph = Graph.from_pt(p)
            else:
                raise ValueError(f"Invalid circuit file extension for {p}")

            hf_task_name = f'mib-bench/{TASKS_TO_HF_NAMES[task]}'
            dataset = HFEAPDataset(
                hf_task_name,
                model.tokenizer,
                split=args.split,
                task=task,
                model_name=model_name,
            )

            print(f"Task={task} Model={model_name} Split={args.split} Dataset size after filtering = {len(dataset)}")
            if len(dataset) == 0:
                print(f"WARNING: empty dataset after filtering for task={task}, model={model_name}, split={args.split}. Skipping.")
                continue
            else:
                print(f"Sample datapoint after filtering: {dataset[0]}")
                
            if args.head is not None:
                head = min(args.head, len(dataset))
                if head < len(dataset):
                    dataset.head(head)
            
            if len(dataset) == 0:
                print(
                    f"Warning: dataset became empty after --head for task={task}, "
                    f"model={model_name}; skipping this task."
                )
                continue

            dataloader = dataset.to_dataloader(batch_size=args.batch_size)

            metric = get_metric('logit_diff', task, model.tokenizer, model)
            attribution_metric = partial(metric, mean=False, loss=False)

            if model_name == 'interpbench':
                if reference_graph is None:
                    raise RuntimeError('reference_graph was not initialised for interpbench')
                d = evaluate_area_under_roc(reference_graph, graph)
            else:
                weighted_edge_counts, area_under, area_from_1, average, faithfulnesses, raw_scores = evaluate_area_under_curve(
                    model,
                    graph,
                    dataloader,
                    attribution_metric,
                    level=args.level,
                    absolute=args.absolute,
                    apply_greedy=apply_greedy,
                )
                d = {
                    'weighted_edge_counts': weighted_edge_counts,
                    'area_under': area_under,
                    'area_from_1': area_from_1,
                    'average': average,
                    'faithfulnesses': faithfulnesses,
                    'raw_scores': raw_scores,
                }

            meta = {
                'method_tag': args.method if args.method is not None else 'custom',
                'method_base': method_base,
                'kind': parsed_method.get('kind', 'gold') if args.method is not None else 'custom',
                'source_model': parsed_method.get('source_model', ''),
                'train_spec': parsed_method.get('train_spec', ''),
                'control': parsed_method.get('control', ''),
                'ablation': args.ablation,
                'level': args.level,
                'task': task,
                'model': model_name,
                'split': args.split,
                'absolute': args.absolute,
                'circuit_path': p,
            }
            d['meta'] = meta

            output_method_name = args.method if args.method is not None else 'custom'
            method_name_saveable = f"{output_method_name}_{args.ablation}_{args.level}"
            output_path = os.path.join(args.output_dir, method_name_saveable)
            os.makedirs(output_path, exist_ok=True)
            out_file = os.path.join(
                output_path,
                f"{task.replace('_', '-')}_{model_name}_{args.split}_abs-{args.absolute}.pkl",
            )
            with open(out_file, 'wb') as f:
                pickle.dump(d, f)
            print(f"Saved evaluation to {out_file}")


if __name__ == '__main__':
    main()
