import argparse
import os
import sys
import torch
import torch.optim as optim
from typing import List, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), 'EAP-IG', 'src'))
from eap.graph import Graph

from differentiable_alignment.model import AlignmentModel
from differentiable_alignment.loss import supervised_mse_loss

def build_aligned_method_tag(method_base: str, small_model: str, train_spec: str, control: str) -> str:
    return (
        f"{method_base}"
        f"__kind-regression"
        f"__src-{small_model}"
        f"__train-{train_spec}"
        f"__ctrl-{control}"
    )

def load_graph_scores(task: str, model_name: str, method: str, ablation: str, level: str, circuit_dir: str, device: str) -> Tuple[torch.Tensor, List[str], Graph]:
    path = os.path.join(circuit_dir, f"{method}_{ablation}_{level}", f"{task.replace('_', '-')}_{model_name}", 'importances.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing circuit: {path}")
    g = Graph.from_json(path)
    nodes = list(g.nodes.keys())
    s_vector = torch.zeros(len(nodes), device=device)
    for i, node_name in enumerate(nodes):
        if g.nodes[node_name].score is not None:
            val = float(g.nodes[node_name].score)
            if not torch.isnan(torch.tensor(val)):
                s_vector[i] = val
    return s_vector, nodes, g

def parse_train_mode(all_tasks: List[str], train_mode: str) -> Tuple[List[str], str]:
    if train_mode == 'All': return list(all_tasks), 'All'
    if train_mode.startswith('LOO_'): return [t for t in all_tasks if t != train_mode[4:]], f"LOO-{train_mode[4:]}"
    if train_mode.startswith('Train_'): return [train_mode[6:]], f"TRAIN-{train_mode[6:]}"
    raise ValueError("Invalid train mode")

def main():
    parser = argparse.add_argument_group("Regression Baseline")
    parser.add_argument('--small-model', type=str, required=True)
    parser.add_argument('--large-model', type=str, required=True)
    parser.add_argument('--tasks', nargs='+', required=True)
    parser.add_argument('--method', type=str, default='EAP-IG-inputs')
    parser.add_argument('--level', type=str, default='node')
    parser.add_argument('--ablation', type=str, default='patching')
    parser.add_argument('--train-mode', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--circuit-dir', type=str, default='circuits/')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_tasks, train_spec_tag = parse_train_mode(args.tasks, args.train_mode)

    # Note: We must load one task to get node templates
    _, small_nodes, _ = load_graph_scores(args.tasks[0], args.small_model, args.method, args.ablation, args.level, args.circuit_dir, device)
    _, large_nodes, large_template = load_graph_scores(args.tasks[0], args.large_model, args.method, args.ablation, args.level, args.circuit_dir, device)

    # Always linear for supervised regression baseline as per prompt
    model = AlignmentModel(small_nodes, large_nodes, architecture='linear').to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 1. Gather all training data
    s_small_train_list, s_large_train_list = [], []
    for t in train_tasks:
        s_small, _, _ = load_graph_scores(t, args.small_model, args.method, args.ablation, args.level, args.circuit_dir, device)
        s_large, _, _ = load_graph_scores(t, args.large_model, args.method, args.ablation, args.level, args.circuit_dir, device)
        s_small_train_list.append(s_small)
        s_large_train_list.append(s_large)
    
    s_small_batch = torch.stack(s_small_train_list)
    s_large_batch = torch.stack(s_large_train_list)

    # 2. Train
    model.train()
    for ep in range(args.epochs):
        optimizer.zero_grad()
        logits_large = torch.matmul(s_small_batch, model.W)
        loss = supervised_mse_loss(logits_large, s_large_batch)
        loss.backward()
        optimizer.step()
        if ep % 10 == 0:
            print(f"Epoch {ep} | MSE Loss: {loss.item():.4f}")

    # 3. Evaluate & Save for all tasks
    model.eval()
    method_tag = build_aligned_method_tag(args.method, args.small_model, train_spec_tag, 'none')
    base_output_dir = os.path.join(args.circuit_dir, f"{method_tag}_{args.ablation}_{args.level}")

    for t in args.tasks:
        s_small_eval, _, _ = load_graph_scores(t, args.small_model, args.method, args.ablation, args.level, args.circuit_dir, device)
        with torch.no_grad():
            pred_large = torch.matmul(s_small_eval, model.W).detach().float().cpu()
        
        # Populate template
        for i, node_name in enumerate(large_nodes):
            if node_name in large_template.nodes:
                large_template.nodes[node_name].score = float(pred_large[i].item())
        
        task_out = os.path.join(base_output_dir, f"{t.replace('_', '-')}_{args.large_model}")
        os.makedirs(task_out, exist_ok=True)
        large_template.to_json(os.path.join(task_out, 'importances.json'))

if __name__ == "__main__":
    main()