import argparse
import os.path as osp
import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader
import numpy as np
import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid, Actor, WebKB
from torch_geometric.utils import dropout_adj, subgraph
from torch_geometric.nn import GCNConv
from model import Model, LogReg
from utils import knn_graph
from torch_geometric.utils import dropout_adj, homophily

def train(model: Model, x, edge_index, kg_edge_index, optimizer):
    model.train()
    optimizer.zero_grad()
    edge_index = dropout_adj(edge_index, p=args.drop_edge_rate)[0]
    kg_edge_index = dropout_adj(kg_edge_index, p=args.drop_kg_edge_rate)[0]
    h0, h1, z1, z2 = model(x, edge_index, kg_edge_index)
    loss = model.loss(h0, h1, z1, z2)
    loss.backward()
    optimizer.step()

    return loss.item()

def run(data, num_epochs, r):
    model = Model(dataset.num_features, args.num_hidden, args.tau1, args.tau2, args.l1, args.l2).to(device)
     # optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_p, weight_decay=args.wd_p, momentum=0.9)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_p, weight_decay=args.wd_p)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_p, weight_decay=args.wd_p)

    start = t()
    prev = start

    cnt_wait = 0
    best = 1e9
    best_t = 0
    patience = 20
    for epoch in range(1, num_epochs + 1):

        loss = train(model, data.x, data.edge_index, data.kg_edge_index, optimizer)
        now = t()
        if loss < best:
            best = loss
            cnt_wait = 0
            torch.save(model.state_dict(), 'model.pkl')
        else:
            cnt_wait += 1

        #print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
              #    f'this epoch {now - prev:.4f}, total {now - start:.4f}')
        prev = now
           
        if cnt_wait == patience:
            #print('Early stopping!')
            break

    #print("===", r, "th run finished===")
    model.load_state_dict(torch.load('model.pkl'))
    embeds = model.embed(data.x, data.edge_index, data.kg_edge_index)
    return embeds

def evaluate(model, embeds, data):
    model.eval()

    with torch.no_grad():
        logits = model(embeds)

    outs = {}
    for key in ['train', 'val', 'test']:
        mask = data['{}_mask'.format(key)]
        loss = F.nll_loss(logits[mask], data.y[mask]).item()
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

        outs['{}_loss'.format(key)] = loss
        outs['{}_acc'.format(key)] = acc

    return outs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--SEED', type=int, default=0)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--lr_p', type=float, default=0.001)
    parser.add_argument('--lr_m', type=float, default=0.01)
    parser.add_argument('--wd_p', type=float, default=0.0)
    parser.add_argument('--wd_m', type=float, default=0.01)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--num_hidden', type=int, default=256)
    parser.add_argument('--tau1', type=float, default=1.1)
    parser.add_argument('--tau2', type=float, default=1.1)
    parser.add_argument('--l1', type=float, default=1.0)
    parser.add_argument('--l2', type=float, default=1.0)
    parser.add_argument('--metric', type=str, default='cosine')
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--drop_edge_rate', type=float, default=0.0)
    parser.add_argument('--drop_kg_edge_rate', type=float, default=0.0)
    args = parser.parse_args()

    def get_dataset(path, name):
        if name in ['cora', 'citeseer', 'pubmed']:
            return Planetoid(path, name, transform=T.NormalizeFeatures())
        elif name in ['Cornell', 'Texas', 'Wisconsin']:
            return WebKB(path, name)
        elif name in ['Actor']:
            return Actor(path, transform=T.NormalizeFeatures())
        
    torch.manual_seed(args.SEED)
    torch.cuda.manual_seed(args.SEED)
    np.random.seed(args.SEED)
    torch.backends.cudnn.deterministic = True

    path = osp.join(osp.expanduser('..'), 'data', args.dataset)
    dataset = get_dataset(path, args.dataset)
    data = dataset[0]
 #   h = homophily(data.edge_index, data.y, method = 'edge')
#    print(h)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data.kg_edge_index = knn_graph(data.x, k=args.K, metric=args.metric)
    data = data.to(device)
    f_accs = []
  
    for r in range(5):
        
        if len(data.train_mask.size())>1:
            data.train_mask = data.train_mask[:, r]
            data.val_mask = data.val_mask[:, r]
            data.test_mask = data.test_mask[:, r]
        embeds = run(data, args.num_epochs, r)
        train_lbls = data.y[data.train_mask]
        test_lbls = data.y[data.test_mask]
        accs = []
        for _ in range(100):
            log = LogReg(args.num_hidden, dataset.num_classes)
            opt = torch.optim.Adam(log.parameters(), lr=args.lr_m, weight_decay=args.wd_m)
            log.cuda()

            best_val_loss = float('inf')
            test_acc = 0
            val_loss_history = []
            
            for _ in range(200):
                log.train()
                opt.zero_grad()

                logits = log(embeds[data.train_mask])
                loss = F.nll_loss(logits, train_lbls)

                loss.backward()
                opt.step()
                
                eval_info = evaluate(log, embeds, data)
                if eval_info['val_loss'] < best_val_loss:
                    best_val_loss = eval_info['val_loss']
                    test_acc = eval_info['test_acc']
                    val_acc = eval_info['val_acc']
                val_loss_history.append(eval_info['val_loss'])

                if _ > _ // 2:
                    tmp = torch.tensor(val_loss_history[-(20 + 1):-1])
                    if eval_info['val_loss'] > tmp.mean().item():
                        break

            accs.append(test_acc * 100)
            print(test_acc * 100)
        f_accs.append(np.mean(accs))

    print(np.mean(f_accs), np.std(f_accs))
