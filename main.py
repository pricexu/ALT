import sys, argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_geometric import seed_everything

from models import *
from utils import *
from loader import loader

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora',
                    choices=['Cora', 'Citeseer', 'Pubmed', 'DBLP', 'Computers', 'Photo', 'CS', 'Physics',
                    'Chameleon', 'Squirrel', 'Texas', 'Wisconsin', 'Cornell', 'Film', 'Penn94', 'Cornell5'],
                    help="The dataset to be used.")
parser.add_argument('--seed', type=int, default=1234,
                    help="random seed")
parser.add_argument('--alt', type=str, default='adj',
                    choices=['adj', 'adj_norm'],
                    help="Decompose the original adj or the normalized adj.")
parser.add_argument('--backbone', type=str, default='APPNP',
                    choices=['APPNP'],
                    help="The backbone GNN works with the ALT.")
parser.add_argument('--K', type=int, default=5,
                    help="Propagation step of APPNP")
parser.add_argument('--alpha', type=float, default=0.1,
                    help="Restart prob of APPNP")
parser.add_argument('--hidden_unit', type=int, default=32,
                    help="Hidden dim")
parser.add_argument('--amp', type=float, default=0.5,
                    help="The parameter of the high-pass filters in the augmenter.")
parser.add_argument('--wd', type=float, default=5e-5,
                    help="Weight decay.")
parser.add_argument('--lr', type=float, default=0.01,
                    help="Learning rate.")
parser.add_argument('--MLPA', type=float, default=0.001,
                    help="The ratio of signals from the MLP(A) part.")
parser.add_argument('--device', type=str, default='3',
                    choices=['cpu', '0', '1', '2', '3'],
                    help="The GPU device to be used.")
parser.add_argument('--run', type=int, default=10,
                    help="The # of test runs.")

args = parser.parse_args()

if args.device == 'cpu':
    device = 'cpu'
else:
    device = 'cuda:' + args.device
torch.autograd.set_detect_anomaly(True)

seed = args.seed
seed_everything(seed)

def train(model, augmenter, optimizer, optimizer_aug, data):
    model.train()
    augmenter.train()
    optimizer.zero_grad()
    optimizer_aug.zero_grad()

    edge_weight = augmenter(data)
    out = model(data, edge_weight)
    loss = F.nll_loss(out[data.train_idx], data.y[data.train_idx])
    loss.backward()

    optimizer.step()
    optimizer_aug.step()

@torch.no_grad()
def test(model, augmenter, data):
    model.eval()
    augmenter.eval()

    edge_weight = augmenter(data)
    out, accs = model(data, edge_weight), []
    for _, idx in data('train_idx', 'val_idx', 'test_idx'):
        acc = accuracy(out[idx], data.y[idx])
        accs.append(acc.cpu())
    return accs

    
""" Experimental settings """
hidden_unit = args.hidden_unit

""" Load dataset """
data = loader(args.dataset)
if args.dataset in ['Penn94', 'Cornell5']:
    all_idx = torch.nonzero(data.y != -1).flatten().tolist()
    num_nodes = len(all_idx)
else:
    num_nodes = data.y.shape[0]
    all_idx = np.array(range(num_nodes))

num_features = data.x.shape[1]
num_classes = data.y.max().item()+1
data.A = torch_geometric.utils.to_dense_adj(data.edge_index)[0]
data = data.to(device)

print("Device: {}".format(args.device))
print("ALT: {}".format(args.alt))
print("Backbone GNN: {}".format(args.backbone))
print("Propagation Steps: {}".format(args.K))
print("Dataset: {}".format(args.dataset))
print("amp: {}".format(args.amp))
print("wd: {}".format(args.wd))
print("lr: {}".format(args.lr))
print("hidden_unit: {}".format(hidden_unit))
print("MLP_A: {}".format(args.MLPA))

final_tests = []
dataset_split = [0.2, 0.2, 0.6]
for i in range(args.run):
    print("Run: {}".format(i))
    
    """ For 'Cora', 'Citeseer', 'Pubmed' we use their default split. For the others, we randomly split them. """
    if args.dataset not in ['Cora', 'Citeseer', 'Pubmed']:
        np.random.shuffle(all_idx)
        data.train_idx = torch.LongTensor(all_idx[:int(num_nodes*dataset_split[0])])
        data.val_idx = torch.LongTensor(all_idx[int(num_nodes*dataset_split[0]):int(num_nodes*dataset_split[0])+int(num_nodes*dataset_split[1])])
        data.test_idx = torch.LongTensor(all_idx[int(num_nodes*dataset_split[0])+int(num_nodes*dataset_split[1]):])
    else:
        data.train_idx = data.train_mask.nonzero(as_tuple=True)[0]
        data.val_idx = data.val_mask.nonzero(as_tuple=True)[0]
        data.test_idx = data.test_mask.nonzero(as_tuple=True)[0]

    """ To decompose the original adj or normalized adj """
    if args.alt == 'adj':
        model = DualGNN(num_features, hidden_unit, num_classes, args.backbone, MLPA=args.MLPA, A_dim=data.A.shape[0]).to(device)
        decompose_norm = False
    elif args.alt == 'adj_norm':
        model = DualGNN_norm(num_features, hidden_unit, num_classes, args.backbone, MLPA=args.MLPA, A_dim=data.A.shape[0]).to(device)
        decompose_norm = True

    augmenter = Augmenter(num_features, hidden_unit, amp=args.amp, decompose_norm=decompose_norm).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    optimizer_aug = torch.optim.Adam(augmenter.parameters(), lr=args.lr, weight_decay=args.wd)

    best_test_acc = 0
    best_val_acc = 0
    for epoch in tqdm(range(1, 201)):
        train(model, augmenter, optimizer, optimizer_aug, data)
        if epoch % 10 == 0:
            train_acc, val_acc, test_acc = test(model, augmenter, data)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
    final_tests.append(best_test_acc)
print("Mean: {:.4f}, Std: {:.4f}".format(np.array(final_tests).mean(axis=0),
        np.array(final_tests).std(axis=0)))
print()
