import numpy as np
import torch
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops

def knn_graph(X, k=20, metric='minkowski'):
    X = X.cpu().detach().numpy()
    A = kneighbors_graph(X, n_neighbors=k, metric=metric)
    edge_index = sparse_mx_to_edge_index(A)
    edge_index, _ = remove_self_loops(edge_index)
    return edge_index

def sparse_mx_to_edge_index(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    row = torch.from_numpy(sparse_mx.row.astype(np.int64))
    col = torch.from_numpy(sparse_mx.col.astype(np.int64))
    edge_index = torch.stack([row, col], dim=0)
 
    return edge_index

if __name__ == '__main__':
    path = osp.join(osp.expanduser('~'), 'data', 'cora')
    dataset = Planetoid(path, 'cora')
    data = dataset[0]
    knn_graph = knn_graph(data.x)
    print(knn_graph.size())