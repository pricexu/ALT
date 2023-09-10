import sys
from typing import Optional, Tuple
import numpy as np
import torch
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor, matmul, mul
from torch.nn import Sequential, Linear, ReLU, Dropout
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn.conv import APPNP
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn.inits import zeros

class HighPassConv(MessagePassing):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, amp=0.5, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.amp = amp

        self._cached_edge_index = None
        self._cached_adj_t = None

        # self.lin = torch_geometric.nn.dense.linear.Linear(in_channels, out_channels, bias=False, weight_initializer='glorot')
        self.lin = Linear(in_channels, out_channels)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
            self.lin.reset_parameters()
            zeros(self.bias)
            self._cached_edge_index = None
            self._cached_adj_t = None


    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x = self.lin(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        out = self.amp * x - out # the high-pass part

        if self.bias is not None:
            out += self.bias

        return out, edge_index


    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

class APPNP_(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, softmax=True, normalize=True, K=5, alpha=0.1):
        super().__init__()
        self.conv1 = APPNP(K=K, alpha=alpha, normalize=normalize)
        self.softmax = softmax
        self.normalize = normalize

        self.mlp = Sequential(
            Dropout(p=0.5),
            Linear(input_dim, hidden_dim),
            ReLU(),
            Dropout(p=0.5),
            Linear(hidden_dim, output_dim)
        )

    def forward(self, data, edge_weight=None, output_emb=False):
        # x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x, edge_index = data.x, data.edge_index
        if not self.normalize:
            edge_index, _ = add_remaining_self_loops(edge_index, None, 1, x.size(0))
        x = self.mlp(x)
        # x = F.dropout(self.mlp(x), training=self.training)
        if edge_weight == None:
            x = self.conv1(x, edge_index)
        else:
            x = self.conv1(x, edge_index, 0.0001+0.9999*edge_weight)

        if output_emb:
            return x
        elif self.softmax:
            return F.log_softmax(x, dim=1)
        else:
            return x


class DualGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, backbone, MLPA=0.001, A_dim=1, K=5, alpha=0.1):
        super().__init__()

        if backbone == 'APPNP':
            self.net1 = APPNP_(input_dim, hidden_dim, output_dim, softmax=False, K=K, alpha=alpha)
            self.net2 = APPNP_(input_dim, hidden_dim, output_dim, softmax=False, K=K, alpha=alpha)
        else:
            print("Unknown model.")
            sys.exit()

        self.offset_mlp = Sequential(
            Linear(input_dim, hidden_dim*2),
            ReLU(),
            Linear(hidden_dim*2, output_dim)
        )

        self.mlp_A = Sequential(
            Linear(A_dim, hidden_dim*2),
            ReLU(),
            Linear(hidden_dim*2, output_dim)
        )

        self.MLPA = MLPA

        self.backbone = backbone

    def forward(self, data, edge_weight=None):
        x1 = self.net1(data, 0.00001 + edge_weight*0.99998) # for smoothness
        x2 = self.net2(data, 0.99999 - edge_weight*0.99998)
        x_offset = self.offset_mlp(data.x)

        x = x1 - x2 + 0.001 * x_offset
        if self.MLPA:
            x += self.MLPA * self.mlp_A(data.A)

        return F.log_softmax(x, dim=1)

class DualGNN_norm(torch.nn.Module):
    # the model decomposing the normalized adj matrix
    def __init__(self, input_dim, hidden_dim, output_dim, backbone, MLPA=0.001, A_dim=1, K=5, alpha=0.1):
        super().__init__()

        if backbone == 'APPNP':
            self.net1 = APPNP_(input_dim, hidden_dim, output_dim, softmax=False, normalize=False, K=K, alpha=alpha)
            self.net2 = APPNP_(input_dim, hidden_dim, output_dim, softmax=False, normalize=False, K=K, alpha=alpha)
        else:
            print("Unknown model.")
            sys.exit()

        self.offset_mlp = Sequential(
            Linear(input_dim, hidden_dim*2),
            ReLU(),
            Linear(hidden_dim*2, output_dim)
        )

        self.mlp_A = Sequential(
            Linear(A_dim, hidden_dim*2),
            ReLU(),
            Linear(hidden_dim*2, output_dim)
        )

        self.MLPA = MLPA

        self.backbone = backbone
        self._cached_edge_index = None
        self.add_self_loops = True

    def forward(self, data, re_edge_weight=None, output_emb=False):
        x, edge_index = data.x, data.edge_index

        cache = self._cached_edge_index
        if cache is None:
            edge_weight = None
            edge_index, edge_weight = gcn_norm(
                edge_index, edge_weight, x.size(0), self.add_self_loops)
            self._cached_edge_index = (edge_index, edge_weight)
        else:
            edge_index, edge_weight = cache[0], cache[1]

        x1 = self.net1(data, (0.0001 + re_edge_weight.flatten()*0.9998)*edge_weight) # for smoothness
        x2 = self.net2(data, (0.9999 - re_edge_weight.flatten()*0.9998)*edge_weight)
        x_offset = self.offset_mlp(x)
        x = x1 - x2 + 0.001 * x_offset

        if self.MLPA != 0:
            x += self.MLPA * self.mlp_A(data.A)

        if output_emb:
            return x
        else:
            return F.log_softmax(x, dim=1)

class Augmenter(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, amp=0.5, decompose_norm=False):

        super().__init__()
        self.conv1 = HighPassConv(input_dim, hidden_dim, cached=True, amp=amp)
        self.conv2 = HighPassConv(hidden_dim, hidden_dim, cached=True, amp=amp)

        self.mlp_edge_model = Sequential(
            Dropout(p=0.5),
            Linear(hidden_dim, hidden_dim*2),
            ReLU(),
            Dropout(p=0.5),
            Linear(hidden_dim*2, 1)
        )
        self.init_emb()

        self.decompose_norm = decompose_norm

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(1)

    def forward(self, data, edge_weight=None):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index)[0])
        x, self_looped_edge_index = self.conv2(x, edge_index)

        # if we aim to decompose the normalized adj, we need to update the edge list
        # cuz in most cases we will add self-loop
        if self.decompose_norm:
            src, dst = self_looped_edge_index[0], self_looped_edge_index[1]
        else:
            src, dst = edge_index[0], edge_index[1]

        emb_src = x[src]
        emb_dst = x[dst]

        edge_emb = emb_src + emb_dst
        edge_logits = self.mlp_edge_model(edge_emb) # aim to keep most of the edges at initialization

        return torch.sigmoid(edge_logits)
