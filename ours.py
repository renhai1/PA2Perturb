"""
PA2Perturb: Prompt-Activated Amplified Perturbation against Unauthorized
Topological Dataset Usage.

Full pipeline: hard node selection -> stable region construction ->
CIS generation with bi-level optimization -> legal/illegal model training ->
ownership verification.
"""

import argparse
import os
import pickle
import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Flickr
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, k_hop_subgraph, degree
from sklearn.metrics.pairwise import pairwise_kernels

from model import GCN, GAT, GIN, GraphSAGE
from NodeClassifier import NodeClassifier
from prompt_graph.MyPrompt import DualPrompt

# ============================================================
# CLI
# ============================================================
parser = argparse.ArgumentParser(description="PA2Perturb pipeline")
parser.add_argument('--seed', type=int, default=10)
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--device_id', type=int, default=0)

# model / data
parser.add_argument('--model', type=str, default='GCN',
                    choices=['GCN', 'GAT', 'GraphSage', 'GIN'])
parser.add_argument('--dataset', type=str, default='Cora',
                    choices=['Cora', 'PubMed', 'Citeseer', 'Flickr'])
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--num_layer', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.5)

# training
parser.add_argument('--train_lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--epochs', type=int, default=200,
                    help='Epochs for GNN training (stages 4-5)')
parser.add_argument('--surrogate_epochs', type=int, default=200,
                    help='Epochs for surrogate model training')

# hard node / region selection  (Section 5.2)
parser.add_argument('--total_select', type=int, default=80,
                    help='Total number of hard nodes N_h')
parser.add_argument('--tau_s', type=float, default=0.7,
                    help='Semantic similarity threshold')
parser.add_argument('--tau_d', type=float, default=0.3,
                    help='Structural consistency threshold')
parser.add_argument('--K_hop', type=int, default=2,
                    help='K-hop neighborhood size')

# CIS / prompt optimization  (Section 5.3)
parser.add_argument('--trigger_size', type=int, default=3,
                    help='Number of CIS nodes per hard node')
parser.add_argument('--beta', type=float, default=10.0,
                    help='Hard node correction weight in L_prompt')
parser.add_argument('--gamma', type=float, default=10.0,
                    help='Semantic loss weight in L_c')
parser.add_argument('--outer_epochs', type=int, default=200,
                    help='Outer loop epochs for bi-level optimization')
parser.add_argument('--n_inner', type=int, default=10,
                    help='Inner loop steps per outer epoch')

# verification  (Section 5.4)
parser.add_argument('--zeta', type=float, default=2.0,
                    help='Amplification coefficient (>1)')
parser.add_argument('--alpha', type=float, default=0.05,
                    help='Significance level for permutation test')
parser.add_argument('--sigma', type=float, default=1.0,
                    help='Gaussian kernel bandwidth for MMD')
parser.add_argument('--n_perm', type=int, default=1000,
                    help='Number of permutations for MMD test')

args = parser.parse_args()
args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device(
    f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# ============================================================
# Logger
# ============================================================
os.makedirs('logs', exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join('logs', f'pa2perturb_{timestamp}.log')
logging.basicConfig(filename=log_path, filemode='w', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger('').addHandler(console)

logging.info("=== PA2Perturb Experiment ===")
logging.info(f"Args: {vars(args)}")

# ============================================================
# 0. Data Loading
# ============================================================
transform = T.Compose([T.NormalizeFeatures()])
if args.dataset in ('Cora', 'Citeseer', 'PubMed'):
    dataset = Planetoid(root='./data/Planetoid', name=args.dataset,
                        transform=transform)
elif args.dataset == 'Flickr':
    dataset = Flickr(root='./data/Flickr/', transform=transform)

data = dataset[0].to(device)
num_classes = dataset.num_classes
feature_dim = data.x.size(1)

train_mask = data.train_mask
test_mask = data.test_mask
val_mask = data.val_mask if hasattr(data, 'val_mask') and data.val_mask is not None else test_mask
train_idx = train_mask.nonzero(as_tuple=False).view(-1)
test_idx = test_mask.nonzero(as_tuple=False).view(-1)
unlabeled_mask = ~train_mask

logging.info(f"Dataset: {args.dataset} | Nodes: {data.num_nodes} | "
             f"Edges: {data.edge_index.size(1)} | Features: {feature_dim} | "
             f"Classes: {num_classes}")


# ============================================================
# Helper: build GNN
# ============================================================
def build_gnn(input_dim, hid_dim, num_layer, dropout):
    if args.model == 'GCN':
        return GCN(input_dim, hid_dim, num_layer=num_layer,
                   drop_ratio=dropout).to(device)
    elif args.model == 'GAT':
        return GAT(input_dim, hid_dim, num_layer=num_layer,
                   drop_ratio=dropout).to(device)
    elif args.model == 'GraphSage':
        return GraphSAGE(input_dim, hid_dim, num_layer=num_layer,
                         drop_ratio=dropout).to(device)
    elif args.model == 'GIN':
        return GIN(input_dim, hid_dim, num_layer=num_layer,
                   drop_ratio=dropout).to(device)


# ============================================================
# CIS Generator: 2-layer MLP  (Appendix A, "f_c is a 2-layer MLP")
# ============================================================
class CISGenerator(nn.Module):
    """Generate CIS node features from stable-region context."""

    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, region_feats):
        """
        Args:
            region_feats: [N_h, d] mean feature of each stable region
        Returns:
            cis_feats: [N_h * trigger_size, d]
        """
        return self.mlp(region_feats)


# ============================================================
# 1. Train Surrogate Model  (Section 5.2, Eq. 3)
# ============================================================
logging.info("=== Stage 1: Train Surrogate Model ===")

surrogate_gnn = GCN(input_dim=feature_dim, hid_dim=args.hidden,
                     num_layer=args.num_layer, drop_ratio=args.dropout).to(device)
surrogate_cls = NodeClassifier(hid_dim=args.hidden, num_classes=num_classes,
                               dropout=args.dropout, inner_dim=args.hidden).to(device)
opt_gnn = torch.optim.Adam(surrogate_gnn.parameters(), lr=args.train_lr,
                           weight_decay=args.weight_decay)
opt_cls = torch.optim.Adam(surrogate_cls.parameters(), lr=args.train_lr,
                           weight_decay=args.weight_decay)

for epoch in range(1, args.surrogate_epochs + 1):
    surrogate_gnn.train(); surrogate_cls.train()
    opt_gnn.zero_grad(); opt_cls.zero_grad()
    h = surrogate_gnn(data.x, data.edge_index)
    logits, _ = surrogate_cls(h)
    loss = F.cross_entropy(logits[train_idx], data.y[train_idx])
    loss.backward()
    opt_gnn.step(); opt_cls.step()

surrogate_gnn.eval(); surrogate_cls.eval()
with torch.no_grad():
    h_all = surrogate_gnn(data.x, data.edge_index)
    logits_all, _ = surrogate_cls(h_all)
    preds_all = logits_all.argmax(dim=1)
    train_acc = (preds_all[train_idx] == data.y[train_idx]).float().mean().item()
    test_acc = (preds_all[test_idx] == data.y[test_idx]).float().mean().item()
logging.info(f"Surrogate Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")


# ============================================================
# 2. Entropy-Based Hard Node Selection  (Section 5.2, Eq. 4-5)
# ============================================================
def select_hard_nodes_by_entropy():
    """Select N_h hard nodes with highest prediction entropy per class."""
    save_dir = './saved_indices/'
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir,
                             f'{args.dataset}_entropy_{args.total_select}_hard_nodes.pkl')
    if os.path.exists(save_file):
        with open(save_file, 'rb') as f:
            result = pickle.load(f)
        logging.info(f"Loaded cached hard nodes from {save_file}")
        return result

    with torch.no_grad():
        probs = F.softmax(logits_all, dim=1)
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=1)  # H(v)
        pseudo_labels_local = probs.argmax(dim=1)

    per_class = args.total_select // num_classes
    remainder = args.total_select % num_classes
    allocations = [per_class + (1 if i < remainder else 0)
                   for i in range(num_classes)]

    hard_nodes_per_class = {}
    for c in range(num_classes):
        mask_c = (pseudo_labels_local == c) & unlabeled_mask
        candidates = mask_c.nonzero(as_tuple=False).view(-1)
        if len(candidates) == 0:
            hard_nodes_per_class[c] = torch.tensor([], dtype=torch.long,
                                                   device=device)
            continue
        ent_c = entropy[candidates]
        k = min(allocations[c], len(candidates))
        _, topk_idx = ent_c.topk(k)
        hard_nodes_per_class[c] = candidates[topk_idx]

    with open(save_file, 'wb') as f:
        pickle.dump(hard_nodes_per_class, f)
    logging.info(f"Hard nodes saved to {save_file}")
    return hard_nodes_per_class


logging.info("=== Stage 2: Hard Node Selection (entropy-based) ===")
hard_nodes_per_class = select_hard_nodes_by_entropy()

hard_node_list = []
for c in range(num_classes):
    nodes = hard_nodes_per_class[c]
    if isinstance(nodes, np.ndarray):
        nodes = torch.tensor(nodes, dtype=torch.long, device=device)
    hard_node_list.append(nodes.to(device))

hard_nodes = torch.cat(hard_node_list)
hard_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
hard_mask[hard_nodes] = True
soft_mask = ~hard_mask

with torch.no_grad():
    pseudo_labels = logits_all.argmax(dim=1)

logging.info(f"Selected {len(hard_nodes)} hard nodes across {num_classes} classes")


# ============================================================
# 3. Context-Aware Stable Region Selection  (Section 5.2, Eq. 6-9)
# ============================================================
logging.info("=== Stage 3: Stable Region Selection ===")

with torch.no_grad():
    embeddings = surrogate_gnn(data.x, data.edge_index)

deg = degree(data.edge_index[0], num_nodes=data.num_nodes)

stable_regions = {}
for vh in hard_nodes.tolist():
    subset, sub_edge_index, mapping, _ = k_hop_subgraph(
        vh, args.K_hop, data.edge_index, relabel_nodes=False,
        num_nodes=data.num_nodes)

    h_vh = embeddings[vh]
    h_neighbors = embeddings[subset]

    # semantic filtering (Eq. 6)
    cos_sim = F.cosine_similarity(
        h_vh.unsqueeze(0).expand(len(subset), -1), h_neighbors, dim=1)
    sem_mask = cos_sim >= args.tau_s
    v_sem = subset[sem_mask]

    # structural filtering (Eq. 7-8)
    deg_vh = deg[vh]
    if deg_vh > 0:
        deg_diff = ((deg[v_sem] - deg_vh).abs() / deg_vh)
        str_mask = deg_diff <= args.tau_d
        v_sem = v_sem[str_mask]

    if len(v_sem) == 0:
        v_sem = torch.tensor([vh], device=device)

    stable_regions[vh] = v_sem

total_region_nodes = sum(len(v) for v in stable_regions.values())
logging.info(f"Stable regions: {len(stable_regions)} regions, "
             f"{total_region_nodes} total nodes")


# ============================================================
# 4. CIS Generation + Bi-Level Optimization  (Section 5.3, Eq. 10-19)
# ============================================================
logging.info("=== Stage 4: Bi-Level Optimization (CIS + Dual Prompt) ===")

# freeze surrogate
for p in surrogate_gnn.parameters():
    p.requires_grad = False
for p in surrogate_cls.parameters():
    p.requires_grad = False

num_orig = data.num_nodes
total_cis_nodes = len(hard_nodes) * args.trigger_size
hard_nodes_list = hard_nodes.tolist()

# --- CIS Generator (2-layer MLP, Appendix Section 14) ---
cis_gen = CISGenerator(input_dim=feature_dim, hidden_dim=args.hidden).to(device)

# precompute stable-region mean features as input to generator
region_mean_feats = []
for vh in hard_nodes_list:
    v_sem = stable_regions[vh]
    region_mean_feats.append(data.x[v_sem].mean(dim=0))
region_mean_feats = torch.stack(region_mean_feats)  # [N_h, d]

# --- Build CIS topology (fixed throughout optimization) ---
cis_edges_list = []
for i, vh in enumerate(hard_nodes_list):
    base = i * args.trigger_size + num_orig
    v_sem = stable_regions[vh]

    # internal CIS clique
    for a in range(args.trigger_size):
        for b in range(a + 1, args.trigger_size):
            cis_edges_list.append([base + a, base + b])
            cis_edges_list.append([base + b, base + a])

    # attach edges: CIS nodes -> stable region nodes (Eq. 10)
    for a in range(args.trigger_size):
        for u in v_sem.tolist():
            cis_edges_list.append([base + a, u])
            cis_edges_list.append([u, base + a])

if cis_edges_list:
    cis_edges = torch.tensor(cis_edges_list, dtype=torch.long, device=device).t()
else:
    cis_edges = torch.zeros((2, 0), dtype=torch.long, device=device)

# precompute perturbed edge_index (topology never changes)
perturbed_edge_index = to_undirected(
    torch.cat([data.edge_index, cis_edges], dim=1))

# --- Dual Prompt ---
prompt = DualPrompt(feature_dim).to(device)
prompt.init_from_data(data.x, soft_mask, hard_mask)

# masks for the full perturbed graph  (N_orig + N_cis nodes)
total_nodes = num_orig + total_cis_nodes
hard_mask_full = torch.zeros(total_nodes, dtype=torch.bool, device=device)
hard_mask_full[hard_nodes] = True

# optimizers
cis_optimizer = torch.optim.Adam(cis_gen.parameters(), lr=args.train_lr)
prompt_optimizer = torch.optim.Adam(prompt.parameters(), lr=args.train_lr)

# --- precompute original-graph embedding statistics (constant target) ---
with torch.no_grad():
    mu_ori = embeddings.mean(dim=0)
    diff_ori = embeddings - mu_ori
    cov_ori = (diff_ori.t() @ diff_ori) / max(embeddings.size(0) - 1, 1)
    # eigendecompose cov_ori once (PSD, use eigh)
    eigvals_ori, Q_ori = torch.linalg.eigh(
        cov_ori + 1e-6 * torch.eye(cov_ori.size(0), device=device))
    sqrt_cov_ori = Q_ori @ torch.diag(eigvals_ori.clamp(min=0).sqrt()) @ Q_ori.t()


def generate_cis_features():
    """Run CIS generator to produce features for all CIS nodes."""
    # generator input: region mean feature replicated per trigger_size
    per_node_feats = cis_gen(region_mean_feats)          # [N_h, d]
    cis_feats = per_node_feats.repeat_interleave(args.trigger_size, dim=0)  # [N_h*ts, d]
    return cis_feats


def compute_w2_loss(h_cis):
    """Squared 2-Wasserstein distance between CIS and original embeddings (Eq. 12).

    Uses the closed-form for Gaussians:
      W2^2 = ||mu_cis - mu_ori||^2
           + Tr(Sigma_cis) + Tr(Sigma_ori)
           - 2 * Tr( (Sigma_ori^{1/2} Sigma_cis Sigma_ori^{1/2})^{1/2} )
    """
    mu_cis = h_cis.mean(dim=0)
    diff_cis = h_cis - mu_cis
    cov_cis = (diff_cis.t() @ diff_cis) / max(h_cis.size(0) - 1, 1)

    mean_diff = (mu_cis - mu_ori).pow(2).sum()

    # M = Sigma_ori^{1/2} Sigma_cis Sigma_ori^{1/2}  (guaranteed PSD)
    M = sqrt_cov_ori @ cov_cis @ sqrt_cov_ori
    # symmetrise for numerical stability
    M = (M + M.t()) / 2
    eigvals_M = torch.linalg.eigvalsh(M).clamp(min=0)
    trace_sqrt_M = eigvals_M.sqrt().sum()

    w2 = mean_diff + cov_cis.trace() + cov_ori.trace() - 2 * trace_sqrt_M
    return w2.clamp(min=0)


def compute_semantic_loss(h_cis_all):
    """Semantic coherence loss (Eq. 13): cosine sim to stable-region centroid."""
    losses = []
    for i, vh in enumerate(hard_nodes_list):
        base = i * args.trigger_size
        v_sem = stable_regions[vh]
        centroid = embeddings[v_sem].mean(dim=0, keepdim=True)  # frozen

        for j in range(args.trigger_size):
            h_j = h_cis_all[base + j].unsqueeze(0)
            cos_sim = F.cosine_similarity(h_j, centroid)
            losses.append(1.0 - cos_sim)

    if losses:
        return torch.stack(losses).mean()
    return torch.tensor(0.0, device=device)


# ---- Bi-level alternating optimization (Eq. 17-19) ----
cis_indices = torch.arange(num_orig, num_orig + total_cis_nodes, device=device)

for outer_epoch in range(1, args.outer_epochs + 1):

    # === Inner loop: fix CIS generator, optimize prompt (Eq. 18) ===
    cis_gen.eval()
    with torch.no_grad():
        cis_feats_detached = generate_cis_features().detach()
        inner_x = torch.cat([data.x, cis_feats_detached], dim=0)

    for inner_step in range(args.n_inner):
        prompt.train()
        prompt_optimizer.zero_grad()

        x_modulated = prompt(inner_x, hard_mask_full, zeta=1.0)
        h = surrogate_gnn(x_modulated, perturbed_edge_index)
        logits, _ = surrogate_cls(h)

        # L_prompt (Eq. 16)
        loss_labeled = F.cross_entropy(logits[train_idx], data.y[train_idx])
        loss_hard = F.cross_entropy(logits[hard_nodes],
                                    pseudo_labels[hard_nodes])
        loss_prompt = loss_labeled + args.beta * loss_hard
        loss_prompt.backward()
        prompt_optimizer.step()

    # === Outer loop: fix prompt, optimize CIS generator (Eq. 19) ===
    cis_gen.train()
    cis_optimizer.zero_grad()

    cis_feats = generate_cis_features()
    new_x = torch.cat([data.x, cis_feats], dim=0)
    h_all_nodes = surrogate_gnn(new_x, perturbed_edge_index)
    h_cis = h_all_nodes[cis_indices]

    l_dist = compute_w2_loss(h_cis)
    l_sem = compute_semantic_loss(h_cis)
    l_c = l_dist + args.gamma * l_sem
    l_c.backward()
    cis_optimizer.step()

    if outer_epoch % 20 == 0 or outer_epoch == 1:
        logging.info(f"[Bi-Level] Epoch {outer_epoch:03d} | "
                     f"L_prompt: {loss_prompt.item():.4f} | "
                     f"L_dist: {l_dist.item():.4f} | "
                     f"L_sem: {l_sem.item():.4f} | "
                     f"L_c: {l_c.item():.4f}")


# ============================================================
# 5. Build Final Perturbed Graph
# ============================================================
logging.info("=== Stage 5: Build Perturbed Graph ===")

cis_gen.eval()
with torch.no_grad():
    final_cis_feats = generate_cis_features().detach()
    final_x = torch.cat([data.x, final_cis_feats], dim=0)

cis_labels = []
for vh in hard_nodes_list:
    label = pseudo_labels[vh].item()
    cis_labels.extend([label] * args.trigger_size)
cis_labels = torch.tensor(cis_labels, device=device)
final_y = torch.cat([data.y, cis_labels])

perturbed_data = Data(x=final_x, edge_index=perturbed_edge_index,
                      y=final_y).to(device)

logging.info(f"Perturbed graph: {perturbed_data.num_nodes} nodes, "
             f"{perturbed_data.edge_index.size(1)} edges "
             f"(+{total_cis_nodes} CIS nodes)")


# ============================================================
# 6. Train Legal Model (GNN + Classifier)
# ============================================================
logging.info("=== Stage 6: Train Legal Model ===")

cis_node_indices = torch.arange(num_orig, perturbed_data.num_nodes, device=device)
legal_train_idx = torch.cat([train_idx, cis_node_indices])

legal_gnn = build_gnn(feature_dim, args.hidden, args.num_layer, args.dropout)
legal_cls = NodeClassifier(hid_dim=args.hidden, num_classes=num_classes,
                           dropout=args.dropout, inner_dim=args.hidden).to(device)
opt_legal_gnn = torch.optim.Adam(legal_gnn.parameters(), lr=args.train_lr,
                                 weight_decay=args.weight_decay)
opt_legal_cls = torch.optim.Adam(legal_cls.parameters(), lr=args.train_lr,
                                 weight_decay=args.weight_decay)

for epoch in range(1, args.epochs + 1):
    legal_gnn.train(); legal_cls.train()
    opt_legal_gnn.zero_grad(); opt_legal_cls.zero_grad()

    h = legal_gnn(perturbed_data.x, perturbed_data.edge_index)
    out, _ = legal_cls(h)
    loss = F.cross_entropy(out[legal_train_idx],
                           perturbed_data.y[legal_train_idx])
    loss.backward()
    opt_legal_gnn.step(); opt_legal_cls.step()

    if epoch % 50 == 0 or epoch == args.epochs:
        legal_gnn.eval(); legal_cls.eval()
        with torch.no_grad():
            logits_e, _ = legal_cls(legal_gnn(perturbed_data.x,
                                              perturbed_data.edge_index))
            pred = logits_e.argmax(dim=1)
            acc = (pred[test_idx] == perturbed_data.y[test_idx]).float().mean().item()
        logging.info(f"[Legal] Epoch {epoch:03d} | Loss: {loss.item():.4f} | "
                     f"Test Acc: {acc:.4f}")


# ============================================================
# 7. Train Illegal Model (suspected, no prompt knowledge)
# ============================================================
logging.info("=== Stage 7: Train Illegal Model ===")

illegal_gnn = build_gnn(feature_dim, args.hidden, args.num_layer, args.dropout)
illegal_cls = NodeClassifier(hid_dim=args.hidden, num_classes=num_classes,
                             dropout=args.dropout, inner_dim=args.hidden).to(device)
opt_ill_gnn = torch.optim.Adam(illegal_gnn.parameters(), lr=args.train_lr,
                               weight_decay=args.weight_decay)
opt_ill_cls = torch.optim.Adam(illegal_cls.parameters(), lr=args.train_lr,
                               weight_decay=args.weight_decay)

for epoch in range(1, args.epochs + 1):
    illegal_gnn.train(); illegal_cls.train()
    opt_ill_gnn.zero_grad(); opt_ill_cls.zero_grad()

    out, _ = illegal_cls(illegal_gnn(perturbed_data.x,
                                     perturbed_data.edge_index))
    loss = F.cross_entropy(out[legal_train_idx],
                           perturbed_data.y[legal_train_idx])
    loss.backward()
    opt_ill_gnn.step(); opt_ill_cls.step()

    if epoch % 50 == 0 or epoch == args.epochs:
        illegal_gnn.eval(); illegal_cls.eval()
        with torch.no_grad():
            logits_e, _ = illegal_cls(illegal_gnn(perturbed_data.x,
                                                  perturbed_data.edge_index))
            pred = logits_e.argmax(dim=1)
            acc = (pred[test_idx] == perturbed_data.y[test_idx]).float().mean().item()
        logging.info(f"[Illegal] Epoch {epoch:03d} | Loss: {loss.item():.4f} | "
                     f"Test Acc: {acc:.4f}")


# ============================================================
# 8. Train Clean Model (FPR baseline, trained on CLEAN data)
# ============================================================
logging.info("=== Stage 8: Train Clean Model (FPR baseline) ===")

clean_gnn = build_gnn(feature_dim, args.hidden, args.num_layer, args.dropout)
clean_cls = NodeClassifier(hid_dim=args.hidden, num_classes=num_classes,
                           dropout=args.dropout, inner_dim=args.hidden).to(device)
opt_c_gnn = torch.optim.Adam(clean_gnn.parameters(), lr=args.train_lr,
                             weight_decay=args.weight_decay)
opt_c_cls = torch.optim.Adam(clean_cls.parameters(), lr=args.train_lr,
                             weight_decay=args.weight_decay)

for epoch in range(1, args.epochs + 1):
    clean_gnn.train(); clean_cls.train()
    opt_c_gnn.zero_grad(); opt_c_cls.zero_grad()

    out, _ = clean_cls(clean_gnn(data.x, data.edge_index))
    loss = F.cross_entropy(out[train_idx], data.y[train_idx])
    loss.backward()
    opt_c_gnn.step(); opt_c_cls.step()

    if epoch == args.epochs:
        clean_gnn.eval(); clean_cls.eval()
        with torch.no_grad():
            logits_e, _ = clean_cls(clean_gnn(data.x, data.edge_index))
            pred = logits_e.argmax(dim=1)
            acc = (pred[test_idx] == data.y[test_idx]).float().mean().item()
        logging.info(f"[Clean] Test Acc: {acc:.4f}")


# ============================================================
# 9. Ownership Verification  (Section 5.4, Eq. 20-23)
# ============================================================
logging.info("=== Stage 9: Ownership Verification ===")


def compute_mmd(X, Y, gamma=1.0):
    """Empirical MMD^2 with Gaussian kernel (Eq. 22)."""
    XX = pairwise_kernels(X, X, metric='rbf', gamma=gamma)
    YY = pairwise_kernels(Y, Y, metric='rbf', gamma=gamma)
    XY = pairwise_kernels(X, Y, metric='rbf', gamma=gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()


def permutation_test(x, y, n_perm=1000, gamma=1.0):
    """Permutation test for MMD (Eq. 23)."""
    combined = np.vstack([x, y])
    n = len(x)
    observed = compute_mmd(x, y, gamma)
    count = 0
    for _ in range(n_perm):
        perm = np.random.permutation(len(combined))
        if compute_mmd(combined[perm[:n]], combined[perm[n:]], gamma) >= observed:
            count += 1
    return observed, count / n_perm


def classify_verdict(a_label, p_value, alpha):
    """Verification criteria (Section 5.4)."""
    if a_label >= 0.5 and p_value > alpha:
        return "CONFIRMED INFRINGEMENT"
    elif a_label < 0.5 and p_value <= alpha:
        return "NO INFRINGEMENT"
    elif a_label >= 0.5 and p_value <= alpha:
        return "FALSE POSITIVE"
    return "FALSE NEGATIVE"


def verify_ownership(legal_gnn, legal_cls, suspect_gnn, suspect_cls,
                     query_data, prompt_module, hard_nodes, hard_mask_q,
                     zeta, alpha, sigma, n_perm, label="Suspect"):
    """Ownership verification via label agreement + MMD test.

    Both models receive the SAME query graph with prompt-modulated features.
    """
    legal_gnn.eval(); legal_cls.eval()
    suspect_gnn.eval(); suspect_cls.eval()
    prompt_module.eval()

    gamma = 1.0 / (2 * sigma ** 2)

    with torch.no_grad():
        # amplified prompt modulation (Eq. 20)
        x_mod = prompt_module(query_data.x, hard_mask_q, zeta=zeta)

        # legal model on perturbed graph
        logits_legal, _ = legal_cls(legal_gnn(x_mod, query_data.edge_index))
        probs_legal = F.softmax(logits_legal[hard_nodes], dim=1)
        preds_legal = logits_legal[hard_nodes].argmax(dim=1)

        # suspected model on the SAME query graph
        logits_suspect, _ = suspect_cls(suspect_gnn(x_mod, query_data.edge_index))
        probs_suspect = F.softmax(logits_suspect[hard_nodes], dim=1)
        preds_suspect = logits_suspect[hard_nodes].argmax(dim=1)

    # label agreement (Eq. 21)
    a_label = (preds_legal == preds_suspect).float().mean().item()

    # MMD + permutation test (Eq. 22-23)
    mmd_val, p_value = permutation_test(
        probs_legal.cpu().numpy(), probs_suspect.cpu().numpy(),
        n_perm=n_perm, gamma=gamma)

    verdict = classify_verdict(a_label, p_value, alpha)

    logging.info(f"[{label}] Label Agreement: {a_label:.4f} | "
                 f"MMD: {mmd_val:.6f} | p-value: {p_value:.4f} | "
                 f"Verdict: {verdict}")
    return a_label, mmd_val, p_value, verdict


# ---- Verify: Legal vs Illegal (should be CONFIRMED INFRINGEMENT) ----
print("\n" + "=" * 60)
print("Verification: Legal vs Illegal (trained on perturbed data)")
print("=" * 60)
verify_ownership(legal_gnn, legal_cls, illegal_gnn, illegal_cls,
                 perturbed_data, prompt, hard_nodes, hard_mask_full,
                 args.zeta, args.alpha, args.sigma, args.n_perm,
                 label="Illegal")

# ---- Verify: Legal vs Clean (should be NO INFRINGEMENT) ----
print("\n" + "=" * 60)
print("Verification: Legal vs Clean (trained on clean data)")
print("=" * 60)
verify_ownership(legal_gnn, legal_cls, clean_gnn, clean_cls,
                 perturbed_data, prompt, hard_nodes, hard_mask_full,
                 args.zeta, args.alpha, args.sigma, args.n_perm,
                 label="Clean")

logging.info("=== Experiment Complete ===")
