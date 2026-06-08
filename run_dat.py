import argparse
import json
import math
import os
import sys
from typing import Dict, List, Tuple

import torch
import torch.optim as optim
from tqdm import tqdm
from transformer_lens import HookedTransformer

sys.path.append(os.path.join(os.path.dirname(__file__), 'EAP-IG', 'src'))
from eap.graph import Graph

from MIB_circuit_track.dataset import HFEAPDataset
from MIB_circuit_track.utils import TASKS_TO_HF_NAMES, MODEL_NAME_TO_FULLNAME

from differentiable_alignment.model import AlignmentModel
from differentiable_alignment.hooks import get_faithfulness_hooks
from differentiable_alignment.loss import faithfulness_loss


CONTROL_CHOICES = ['none', 'random_W', 'permuted_W', 'scrambled_s', 'heuristic_depth_mean']


def get_all_nodes(model: HookedTransformer) -> List[str]:
    nodes: List[str] = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            nodes.append(f"a{layer}.h{head}")
        nodes.append(f"m{layer}")
    return nodes


def parse_train_mode(all_tasks: List[str], train_mode: str) -> Tuple[List[str], str]:
    if train_mode == 'All':
        return list(all_tasks), 'All'
    if train_mode.startswith('LOO_'):
        held_out = train_mode[len('LOO_'):]
        return [t for t in all_tasks if t != held_out], f"LOO-{held_out}"
    if train_mode.startswith('Train_'):
        source_task = train_mode[len('Train_'):]
        return [source_task], f"TRAIN-{source_task}"
    raise ValueError(
        f"Unsupported --train-mode={train_mode}. Expected All, LOO_<task>, or Train_<task>."
    )


def build_aligned_method_tag(method_base: str, small_model: str, train_spec: str, control: str) -> str:
    return (
        f"{method_base}"
        f"__kind-diffalign"
        f"__src-{small_model}"
        f"__train-{train_spec}"
        f"__ctrl-{control}"
    )


def deterministic_task_seed(task: str, base_seed: int) -> int:
    acc = base_seed
    for i, ch in enumerate(task):
        acc += (i + 1) * ord(ch)
    return acc


def load_small_circuit_scores(
    task: str,
    small_model_name: str,
    method: str,
    ablation: str,
    level: str,
    circuit_dir: str,
    small_nodes: List[str],
    device: str,
) -> torch.Tensor:
    circuit_path = os.path.join(
        circuit_dir,
        f"{method}_{ablation}_{level}",
        f"{task.replace('_', '-')}_{small_model_name}",
        'importances.json',
    )
    if not os.path.exists(circuit_path):
        raise FileNotFoundError(f"Missing source circuit: {circuit_path}")

    g = Graph.from_json(circuit_path)
    s_vector = torch.zeros(len(small_nodes), device=device, dtype=torch.float32)

    found = 0
    for i, node_name in enumerate(small_nodes):
        if node_name in g.nodes:
            node = g.nodes[node_name]
            if node.score is not None:
                val = float(node.score)
                if not torch.isnan(torch.tensor(val)):
                    s_vector[i] = val
                    found += 1

    print(f"Loaded small circuit for task={task}: found scores for {found}/{len(small_nodes)} nodes")
    return s_vector


def build_eval_matrix(base_W: torch.Tensor, control: str, seed: int) -> torch.Tensor:
    if control == 'permuted_W':
        generator = torch.Generator(device=base_W.device)
        generator.manual_seed(seed)
        perm = torch.randperm(base_W.shape[1], generator=generator, device=base_W.device)
        return base_W[:, perm]
    return base_W


def maybe_scramble_input(s_vector: torch.Tensor, control: str, seed: int) -> torch.Tensor:
    if control == 'scrambled_s':
        generator = torch.Generator(device=s_vector.device)
        generator.manual_seed(seed)
        perm = torch.randperm(s_vector.shape[0], generator=generator, device=s_vector.device)
        return s_vector[perm]
    return s_vector


def parse_alignment_node_name(node_name: str) -> Tuple[str, int, int]:
    if node_name.startswith('a') and '.h' in node_name:
        layer_part, head_part = node_name.split('.h')
        layer = int(layer_part[1:])
        head = int(head_part)
        return 'attn', layer, head
    if node_name.startswith('m'):
        layer = int(node_name[1:])
        return 'mlp', layer, -1
    raise ValueError(f"Unrecognized alignment node name: {node_name}")


def build_depth_mean_matrix(
    source_nodes: List[str],
    target_nodes: List[str],
    n_source_layers: int,
    n_target_layers: int,
    device: str,
) -> torch.Tensor:
    W = torch.zeros((len(source_nodes), len(target_nodes)), device=device, dtype=torch.float32)

    target_heads_by_layer: Dict[int, List[int]] = {}
    target_mlp_by_layer: Dict[int, int] = {}

    for j, node_name in enumerate(target_nodes):
        node_type, layer, _ = parse_alignment_node_name(node_name)
        if node_type == 'attn':
            target_heads_by_layer.setdefault(layer, []).append(j)
        elif node_type == 'mlp':
            target_mlp_by_layer[layer] = j

    def depth_interp(src_layer: int, Ls: int, Lt: int) -> Tuple[int, int, float, float]:
        if Ls <= 1:
            return 0, 0, 1.0, 0.0

        p = src_layer * (Lt - 1) / (Ls - 1)
        j0 = math.floor(p)
        j1 = math.ceil(p)

        if j0 == j1:
            return j0, j1, 1.0, 0.0

        w0 = j1 - p
        w1 = p - j0
        return j0, j1, w0, w1

    for i, src_name in enumerate(source_nodes):
        src_type, src_layer, _ = parse_alignment_node_name(src_name)
        j0, j1, w0, w1 = depth_interp(src_layer, n_source_layers, n_target_layers)

        if src_type == 'attn':
            heads0 = target_heads_by_layer.get(j0, [])
            heads1 = target_heads_by_layer.get(j1, [])

            if j0 == j1:
                if not heads0:
                    raise RuntimeError(f"No target heads found in target layer {j0}")
                W[i, heads0] = 1.0 / len(heads0)
            else:
                if not heads0 or not heads1:
                    raise RuntimeError(f"Missing target heads in target layers {j0}, {j1}")
                W[i, heads0] += w0 / len(heads0)
                W[i, heads1] += w1 / len(heads1)

        elif src_type == 'mlp':
            idx0 = target_mlp_by_layer.get(j0, None)
            idx1 = target_mlp_by_layer.get(j1, None)

            if j0 == j1:
                if idx0 is None:
                    raise RuntimeError(f"No target MLP found in target layer {j0}")
                W[i, idx0] = 1.0
            else:
                if idx0 is None or idx1 is None:
                    raise RuntimeError(f"Missing target MLP in target layers {j0}, {j1}")
                W[i, idx0] += w0
                W[i, idx1] += w1

    row_sums = W.sum(dim=1, keepdim=True)
    if torch.any(row_sums <= 0):
        bad_rows = torch.nonzero((row_sums <= 0).squeeze(-1)).flatten().tolist()
        raise RuntimeError(f"Heuristic W has zero-sum rows for source indices: {bad_rows[:10]}")
    W = W / row_sums

    return W


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--small-model', type=str, required=True)
    parser.add_argument('--large-model', type=str, required=True)
    parser.add_argument('--tasks', nargs='+', required=True)
    parser.add_argument('--circuit-dir', type=str, default='circuits')
    parser.add_argument('--method', type=str, default='EAP-IG-inputs')
    parser.add_argument('--ablation', type=str, default='patching')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lambda-sparse', type=float, default=0.1)
    parser.add_argument('--device-small', type=str, default='cuda:0')
    parser.add_argument('--device-large', type=str, default='cuda:1')
    parser.add_argument('--output', type=str, default='circuits')
    parser.add_argument('--level', type=str, choices=['node', 'neuron', 'edge'], default='node')
    parser.add_argument('--train-mode', type=str, default='All')
    parser.add_argument('--ablation-tag', type=str, default='none', choices=CONTROL_CHOICES)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.level != 'node':
        raise NotImplementedError(f"This script currently supports node-level alignment only; got --level={args.level}.")

    torch.manual_seed(args.seed)

    all_tasks = list(args.tasks)
    train_tasks, train_spec = parse_train_mode(all_tasks, args.train_mode)

    print('Starting Differentiable Alignment')
    print(f'small={args.small_model} large={args.large_model}')
    print(f'method={args.method} ablation={args.ablation} level={args.level}')
    print(f'train_mode={args.train_mode} -> train_tasks={train_tasks} -> train_spec={train_spec}')
    print(f'control={args.ablation_tag}')

    print(f"Loading small model {args.small_model} on {args.device_small}")
    small_model = HookedTransformer.from_pretrained(
        MODEL_NAME_TO_FULLNAME[args.small_model],
        device=args.device_small,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
    )
    small_model.eval()

    print(f"Loading large model {args.large_model} on {args.device_large}")
    large_model = HookedTransformer.from_pretrained(
        MODEL_NAME_TO_FULLNAME[args.large_model],
        device=args.device_large,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
    )
    large_model.eval()

    small_nodes = get_all_nodes(small_model)
    large_nodes = get_all_nodes(large_model)

    alignment_model = AlignmentModel(small_nodes, large_nodes)
    alignment_model.to(args.device_large)
    optimizer = optim.Adam(alignment_model.parameters(), lr=args.lr)

    tasks_data: List[Dict] = []
    print('Loading source circuits and large-model datasets...')
    for task in all_tasks:
        try:
            s_vector = load_small_circuit_scores(
                task=task,
                small_model_name=args.small_model,
                method=args.method,
                ablation=args.ablation,
                level=args.level,
                circuit_dir=args.circuit_dir,
                small_nodes=small_nodes,
                device=args.device_large,
            )
        except FileNotFoundError as e:
            print(f"Warning: {e}. Skipping task={task}")
            continue

        hf_task_name = f"mib-bench/{TASKS_TO_HF_NAMES[task]}"
        dataset = HFEAPDataset(
            hf_task_name,
            large_model.tokenizer,
            split='train',
            task=task,
            model_name=args.large_model,
        )
        if len(dataset) == 0:
            print(f"Warning: empty train dataset for task={task}, model={args.large_model}; skipping task")
            continue

        dataloader = dataset.to_dataloader(batch_size=args.batch_size)
        tasks_data.append({'task': task, 's_vector': s_vector, 'dataloader': dataloader})

    if not tasks_data:
        raise RuntimeError('No valid tasks found after loading source circuits and train datasets.')

    tasks_data_by_name = {td['task']: td for td in tasks_data}
    train_tasks_data = [tasks_data_by_name[t] for t in train_tasks if t in tasks_data_by_name]

    if args.ablation_tag not in ('random_W', 'heuristic_depth_mean') and not train_tasks_data:
        raise RuntimeError(
            f"No training tasks available for train_mode={args.train_mode}. Requested={train_tasks}, available={sorted(tasks_data_by_name.keys())}"
        )

    if args.ablation_tag == 'random_W':
        print('Control=random_W: skipping optimisation, using the randomly initialised W.')
    elif args.ablation_tag == 'heuristic_depth_mean':
        print('Control=heuristic_depth_mean: skipping optimisation, using fixed proportional-depth + mean-within-layer mapping.')
    else:
        print(f"Training W for {args.epochs} epochs on tasks: {[td['task'] for td in train_tasks_data]}")
        large_node_map = alignment_model.get_large_component_mapping()

        for epoch in range(args.epochs):
            total_loss = 0.0
            total_kl = 0.0
            total_sparse = 0.0
            total_steps = 0

            pbar = tqdm(train_tasks_data, desc=f"Epoch {epoch + 1}/{args.epochs}")
            for task_info in pbar:
                task_name = task_info['task']
                s_vector = task_info['s_vector']
                dataloader = task_info['dataloader']

                for batch_idx, (clean_prompts, corrupt_prompts, _) in enumerate(dataloader):
                    optimizer.zero_grad(set_to_none=True)
                    mask = alignment_model.predict_mask(s_vector)

                    with torch.no_grad():
                        clean_logits, clean_cache = large_model.run_with_cache(
                            clean_prompts,
                            names_filter=lambda n: 'hook_result' in n or 'hook_mlp_out' in n,
                        )

                    hooks = get_faithfulness_hooks(large_model, mask, large_node_map, clean_cache)
                    intervened_logits = large_model.run_with_hooks(corrupt_prompts, fwd_hooks=hooks)

                    loss, kl, sparse = faithfulness_loss(
                        clean_logits,
                        intervened_logits,
                        mask,
                        lambda_sparsity=args.lambda_sparse,
                    )

                    loss.backward()
                    optimizer.step()

                    total_loss += float(loss.item())
                    total_kl += float(kl.item())
                    total_sparse += float(sparse.item())
                    total_steps += 1

                    if batch_idx % 10 == 0:
                        avg_loss = total_loss / max(total_steps, 1)
                        pbar.set_postfix({
                            'task': task_name,
                            'loss': f"{avg_loss:.4f}",
                            'last_kl': f"{float(kl.item()):.4f}",
                        })

            if total_steps == 0:
                print(f"Warning: epoch {epoch + 1} had zero optimisation steps.")
            else:
                print(
                    f"Epoch {epoch + 1} summary | "
                    f"loss={total_loss / total_steps:.4f} "
                    f"kl={total_kl / total_steps:.4f} "
                    f"sparse={total_sparse / total_steps:.4f}"
                )

    output_method_name = build_aligned_method_tag(
        method_base=args.method,
        small_model=args.small_model,
        train_spec=train_spec,
        control=args.ablation_tag,
    )
    parent_dir_name = f"{output_method_name}_{args.ablation}_{args.level}"
    base_output_dir = os.path.join(args.output, parent_dir_name)
    os.makedirs(base_output_dir, exist_ok=True)

    base_W = alignment_model.W.detach()
    if args.ablation_tag == 'heuristic_depth_mean':
        eval_W = build_depth_mean_matrix(
            source_nodes=small_nodes,
            target_nodes=large_nodes,
            n_source_layers=small_model.cfg.n_layers,
            n_target_layers=large_model.cfg.n_layers,
            device=args.device_large,
        )
    else:
        eval_W = build_eval_matrix(base_W, args.ablation_tag, args.seed)

    run_meta = {
        'kind': 'aligned',
        'method_base': args.method,
        'method_tag': output_method_name,
        'small_model': args.small_model,
        'large_model': args.large_model,
        'tasks': all_tasks,
        'available_tasks': sorted(tasks_data_by_name.keys()),
        'train_mode': args.train_mode,
        'train_spec': train_spec,
        'train_tasks': train_tasks,
        'train_tasks_available': [td['task'] for td in train_tasks_data],
        'control': args.ablation_tag,
        'ablation': args.ablation,
        'level': args.level,
        'epochs': args.epochs,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'lambda_sparse': args.lambda_sparse,
        'seed': args.seed,
        'permutation_axis': 'large_nodes' if args.ablation_tag == 'permuted_W' else '',
        'heuristic': 'proportional_depth_plus_mean_within_layer' if args.ablation_tag == 'heuristic_depth_mean' else '',
    }

    with open(os.path.join(base_output_dir, f'run_meta_{args.large_model}.json'), 'w') as f:
        json.dump(run_meta, f, indent=2, sort_keys=True)

    torch.save(
        {
            'W': base_W.cpu(),
            'small_nodes': small_nodes,
            'large_nodes': large_nodes,
            'meta': run_meta,
        },
        os.path.join(base_output_dir, f'alignment_matrix__{args.small_model}_to_{args.large_model}.pt'),
    )
    torch.save(
        {
            'W': eval_W.detach().cpu(),
            'small_nodes': small_nodes,
            'large_nodes': large_nodes,
            'meta': dict(run_meta, matrix_role='inference'),
        },
        os.path.join(base_output_dir, f'inference_alignment_matrix__{args.small_model}_to_{args.large_model}.pt'),
    )

    print(f"Saving aligned circuits to: {base_output_dir}")

    large_graph_template = Graph.from_model(large_model, node_scores=True)
    if getattr(large_graph_template, 'nodes_scores', None) is None:
        large_graph_template.nodes_scores = torch.zeros(large_graph_template.n_forward, device='cpu')

    for task_info in tasks_data:
        task = task_info['task']
        s_vector = task_info['s_vector'].to(base_W.device)
        task_seed = deterministic_task_seed(task, args.seed)
        s_eval = maybe_scramble_input(s_vector, args.ablation_tag, task_seed)

        with torch.no_grad():
            predicted_scores = torch.matmul(s_eval, eval_W).detach().float().cpu()

        if getattr(large_graph_template, 'nodes_scores', None) is not None:
            large_graph_template.nodes_scores[:] = 0.0

        for i, node_name in enumerate(large_nodes):
            if node_name in large_graph_template.nodes:
                large_graph_template.nodes[node_name].score = float(predicted_scores[i].item())

        task_output_dir = os.path.join(base_output_dir, f"{task.replace('_', '-')}_{args.large_model}")
        os.makedirs(task_output_dir, exist_ok=True)

        json_path = os.path.join(task_output_dir, 'importances.json')
        large_graph_template.to_json(json_path)

        task_meta = dict(run_meta)
        task_meta.update({
            'eval_task': task,
            'task_dir': os.path.basename(task_output_dir),
            'importances_path': json_path,
            'input_scrambled': args.ablation_tag == 'scrambled_s',
            'matrix_permuted': args.ablation_tag == 'permuted_W',
            'heuristic_depth_mean': args.ablation_tag == 'heuristic_depth_mean',
        })
        with open(os.path.join(task_output_dir, 'meta.json'), 'w') as f:
            json.dump(task_meta, f, indent=2, sort_keys=True)

        print(f"Saved predicted graph to {json_path}")


if __name__ == '__main__':
    main()