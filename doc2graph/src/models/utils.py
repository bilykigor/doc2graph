"""
Author: Wesley (liwanshui12138@gmail.com)
Date: 2021/6/15
"""
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import dgl.function as fn
import math
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops  #, softmax
#from torch_scatter import scatter
from torch_geometric.nn import GCNConv, SAGEConv#, GATConv, GATv2Conv
from doc2graph.src.models.gatconv import GATConv
from doc2graph.src.models.gatv2conv import GATv2Conv


import math


class GcnSAGELayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 bias=True,
                 use_pp=False,
                 use_lynorm=True):
        super(GcnSAGELayer, self).__init__()
        self.linear = nn.Linear(2 * in_feats, out_feats, bias=bias)
        self.activation = activation
        self.use_pp = use_pp
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        if use_lynorm:
            self.lynorm = nn.LayerNorm(out_feats, elementwise_affine=True)
        else:
            self.lynorm = lambda x: x
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.linear.weight.size(1))
        self.linear.weight.data.uniform_(-stdv, stdv)
        if self.linear.bias is not None:
            self.linear.bias.data.uniform_(-stdv, stdv)

    def forward(self, g, h):
        g = g.local_var()
        
        if not self.use_pp:
            # norm = self.get_norm(g)
            norm = g.ndata['norm']
            g.ndata['h'] = h
            g.update_all(fn.u_mul_e('h', 'weights', 'm'),
                        fn.sum(msg='m', out='h'))
            ah = g.ndata.pop('h')
            h = self.concat(h, ah, norm)

        if self.dropout:
            h = self.dropout(h)

        h = self.linear(h)
        h = self.lynorm(h)
        if self.activation:
            h = self.activation(h)
        return h

    def concat(self, h, ah, norm):
        ah = ah * norm
        h = torch.cat((h, ah), dim=1)
        return h

    def get_norm(self, g):
        norm = 1. / g.in_degrees().float().unsqueeze(1)
        norm[torch.isinf(norm)] = 0
        norm = norm.to(self.linear.weight.device)
        return norm

class InputProjectorSimple(nn.Module):
    def __init__(self, in_size : list, out_size : int, dropout, doNorm) -> None:
        super().__init__()
        
        if doNorm:
            self.node_pred = nn.Sequential(
                    nn.Linear(in_size, in_size),
                    nn.LayerNorm(in_size),
                    nn.LeakyReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(in_size, out_size),
                )
        else:
            self.node_pred = nn.Sequential(
                    nn.Linear(in_size, in_size),
                    nn.LeakyReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(in_size, out_size),
                )
        
    
    def forward(self, x):
        return self.node_pred(x)
    
class InputProjector(nn.Module):
    def __init__(self, in_chunks : list, out_chunks : int, device, doIt = True, dropout=0.2) -> None:
        super().__init__()
        
        if not doIt:
            self.output_length = sum(in_chunks)
            self.doIt = doIt
            return

        self.output_length = len(in_chunks)*out_chunks
        self.doIt = doIt
        self.chunks = in_chunks
        modules = []
        self.device = device

        for chunk in in_chunks:
            chunk_module = []
            chunk_module.append(nn.Linear(chunk, out_chunks))
            chunk_module.append(nn.LayerNorm(out_chunks))
            chunk_module.append(nn.LeakyReLU())
            chunk_module.append(nn.Dropout(dropout))
            modules.append(nn.Sequential(*chunk_module))
        
        self.modalities = nn.Sequential(*modules)
        self.chunks.insert(0, 0)
    
    def get_out_lenght(self):
        return self.output_length
    
    def forward(self, h):

        if not self.doIt:
            return h

        mid = []

        for name, module in self.modalities.named_children():
            num = int(name)
            if num + 1 == len(self.chunks): break
            start = self.chunks[num] + sum(self.chunks[:num])
            end = start + self.chunks[num+1]
            input = h[:, start:end].to(self.device)
            mid.append(module(input))

        return torch.cat(mid, dim=1)

class MLPPredictor(nn.Module):
    def __init__(self, in_features, hidden_dim, out_classes, dropout):
        super().__init__()
        self.out = out_classes
        self.W1 = nn.Linear(in_features*2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.W2 = nn.Linear(hidden_dim + 6, out_classes)
        self.drop = nn.Dropout(dropout)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        polar = edges.data['feat']

        x = F.relu(self.norm(self.W1(torch.cat((h_u, h_v), dim=1))))
        x = torch.cat((x, polar), dim=1)
        score = self.drop(self.W2(x))

        return {'score': score}

    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']

class MLPPredictor_E2E(nn.Module):
    def __init__(self, in_features, hidden_dim, out_classes, dropout,  edge_pred_features):
        super().__init__()
        self.out = out_classes
        self.W1 = nn.Linear(in_features*2 +  edge_pred_features, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.W2 = nn.Linear(hidden_dim, out_classes)
        self.drop = nn.Dropout(dropout)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        cls_u = F.softmax(edges.src['cls'], dim=1)
        cls_v = F.softmax(edges.dst['cls'], dim=1)
        polar = edges.data['feat']

        x = F.relu(self.norm(self.W1(torch.cat((h_u, cls_u, polar, h_v, cls_v), dim=1))))
        score = self.drop(self.W2(x))

        return {'score': score}

    def forward(self, graph, h, cls):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.ndata['cls'] = cls
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']
    
class Attention(nn.Module):
    # single head attention
    def __init__(self, in_features, out_features, alpha):
        super(Attention, self).__init__()
        self.alpha = alpha

        self.W = nn.Linear(in_features, out_features, bias = False)
        self.a_T = nn.Linear(2 * out_features, 1, bias = False)

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_T.weight)

    def forward(self, h, adj):
        # h : a tensor with size [N, F] where N be a number of nodes and F be a number of features
        N = h.size(0)
        Wh = self.W(h) # h -> Wh : [N, F] -> [N, F']
        
        # H1 : [N, N, F'], H2 : [N, N, F'], attn_input = [N, N, 2F']

        # H1 = [[h1 h1 ... h1]   |  H2 = [[h1 h2 ... hN]   |   attn_input = [[h1||h1 h1||h2 ... h1||hN]
        #       [h2 h2 ... h2]   |        [h1 h2 ... hN]   |                 [h2||h1 h2||h2 ... h2||hN]
        #            ...         |             ...         |                         ...
        #       [hN hN ... hN]]  |        [h1 h2 ... hN]]  |                 [hN||h1 hN||h2 ... hN||hN]]
        
        H1 = Wh.unsqueeze(1).repeat(1,N,1)
        H2 = Wh.unsqueeze(0).repeat(N,1,1)
        attn_input = torch.cat([H1, H2], dim = -1)

        e = F.leaky_relu(self.a_T(attn_input).squeeze(-1), negative_slope = self.alpha) # [N, N]
        
        attn_mask = -1e18*torch.ones_like(e)
        masked_e = torch.where(adj > 0, e, attn_mask)
        attn_scores = F.softmax(masked_e, dim = -1) # [N, N]

        h_prime = torch.mm(attn_scores, Wh) # [N, F']

        return F.elu(h_prime) # [N, F']

class GraphAttentionLayer(nn.Module):
    # multi head attention
    def __init__(self, in_features, out_features, num_heads, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.concat = concat
        self.attentions = nn.ModuleList([Attention(in_features, out_features, alpha) for _ in range(num_heads)])
        
    def forward(self, input, adj):
        # input (= X) : a tensor with size [N, F]

        if self.concat :
            # concatenate
            outputs = []
            for attention in self.attentions:
                outputs.append(attention(input, adj))
            
            return torch.cat(outputs, dim = -1) # [N, KF']

        else :
            # average
            output = None
            for attention in self.attentions:
                if output == None:
                    output = attention(input, adj)
                else:
                    output += attention(input, adj)
            
            return output/len(self.attentions) # [N, F']
        
        
def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


# def softmax(src, index, num_nodes):
#     """
#     Given a value tensor: `src`, this function first groups the values along the first dimension
#     based on the indices specified in: `index`, and then proceeds to compute the softmax individually for each group.
#     """
#     print('src', src)
#     print('index', index)
#     print('num_nodes', num_nodes)
#     N = int(index.max()) + 1 if num_nodes is None else num_nodes
#     print('N', N)
#     print(f"{scatter(src, index, dim=0, dim_size=N, reduce='max')}")
#     print(f"{scatter(src, index, dim=0, dim_size=N, reduce='max')[index]}")
#     out = src - scatter(src, index, dim=0, dim_size=N, reduce='max')[index]
#     print('out', out)
#     out = out.exp()
#     print('out', out)
#     out_sum = scatter(out, index, dim=0, dim_size=N, reduce='sum')[index]
#     print('out_sum', out_sum)
#     print(f'return: {out / (out_sum + 1e-16)}')
#     return out / (out_sum + 1e-16)


# class Edge_GATConv(MessagePassing):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  edge_dim,  # new
#                  heads=1,
#                  negative_slope=0.2,
#                  dropout=0.,
#                  bias=True):
#         super(Edge_GATConv, self).__init__(node_dim=0, aggr='add')  # "Add" aggregation.

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.edge_dim = edge_dim  # new
#         self.heads = heads
#         self.negative_slope = negative_slope
#         self.dropout = dropout

#         self.weight = Parameter(torch.Tensor(in_channels, heads * out_channels))    # emb(in) x [H*emb(out)]
#         self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels + edge_dim))   # 1 x H x [2*emb(out)+edge_dim]    # new
#         self.edge_update = Parameter(torch.Tensor(out_channels + edge_dim, out_channels))   # [emb(out)+edge_dim] x emb(out)  # new

#         if bias:
#             self.bias = Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('bias', None)

#         self.reset_parameters()

#     def reset_parameters(self):
#         glorot(self.weight)
#         glorot(self.att)
#         glorot(self.edge_update)  # new
#         zeros(self.bias)

#     def forward(self, x, edge_index, edge_attr, size=None):
#         # 1. Linearly transform node feature matrix (XΘ)
#         x = torch.mm(x, self.weight).view(-1, self.heads, self.out_channels)   # N x H x emb(out)
#         print('x', x)

#         # 2. Add self-loops to the adjacency matrix (A' = A + I)
#         if size is None and torch.is_tensor(x):
#             edge_index, _ = remove_self_loops(edge_index)   # 2 x E
#             print('edge_index', edge_index)
#             edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))   # 2 x (E+N)
#             print('edge_index', edge_index)

#         # 2.1 Add node's self information (value=0) to edge_attr
#         self_loop_edges = torch.zeros(x.size(0), edge_attr.size(1)).to(edge_index.device)   # N x edge_dim   # new
#         print('self_loop_edges', self_loop_edges)
#         edge_attr = torch.cat([edge_attr, self_loop_edges], dim=0)  # (E+N) x edge_dim  # new
#         print('edge_attr', edge_attr)

#         # 3. Start propagating messages
#         return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)  # new
#                             # 2 x (E+N), N x H x emb(out), (E+N) x edge_dim, None

#     def message(self, x_i, x_j, size_i, edge_index_i, edge_attr):  # Compute normalization (concatenate + softmax)
#         # x_i, x_j: after linear x and expand edge (N+E) x H x emb(out)
#         # = N x emb(in) @ emb(in) x [H*emb(out)] (+) E x H x emb(out)
#         # edge_index_i: the col part of index  [E+N]
#         # size_i: number of nodes
#         # edge_attr: edge values of 1->0, 2->0, 3->0.   (E+N) x edge_dim
#         print('x_i', x_i)
#         print('x_j', x_j)
#         print('size_i', size_i)
#         print('edge_index_i', edge_index_i)
#         print('edge_attr', edge_attr)

#         edge_attr = edge_attr.unsqueeze(1).repeat(1, self.heads, 1)  # (E+N) x H x edge_dim  # new
#         print('edge_attr', edge_attr)
#         x_j = torch.cat([x_j, edge_attr], dim=-1)  # (E+N) x H x (emb(out)+edge_dim)   # new
#         print('x_j', x_j)

#         x_i = x_i.view(-1, self.heads, self.out_channels)  # (E+N) x H x emb(out)
#         print('x_i', x_i)
#         print(torch.cat([x_i, x_j], dim=-1))   # (E+N) x H x [emb(out)+(emb(out)+edge_dim)]
#         print('self.att', self.att)   # 1 x H x [2*emb(out)+edge_dim]
#         alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)  # (E+N) x H
#         print('alpha', alpha)

#         alpha = F.leaky_relu(alpha, self.negative_slope)
#         print('alpha', alpha)
#         alpha = softmax(alpha, edge_index_i, num_nodes=size_i)   # Computes a sparsely evaluated softmax
#         print('alpha', alpha)

#         if self.training and self.dropout > 0:
#             alpha = F.dropout(alpha, p=self.dropout, training=True)

#         print(f'x_j*alpha {x_j * alpha.view(-1, self.heads, 1)}')
#         return x_j * alpha.view(-1, self.heads, 1)   # (E+N) x H x (emb(out)+edge_dim)

#     def update(self, aggr_out):   # 4. Return node embeddings (average heads)
#         # for Node 0: Based on the directed graph, Node 0 gets message from three edges and one self_loop
#         # for Node 1, 2, 3: since they do not get any message from others, so only self_loop

#         print('aggr_out', aggr_out)   # N x H x (emb(out)+edge_dim)
#         aggr_out = aggr_out.mean(dim=1)
#         print('aggr_out', aggr_out)   # N x (emb(out)+edge_dim)
#         print('self.edge_update', self.edge_update)   # (emb(out)+edge_dim) x emb(out)
#         aggr_out = torch.mm(aggr_out, self.edge_update)
#         print('aggr_out', aggr_out)   # N x emb(out)  # new

#         if self.bias is not None:
#             aggr_out = aggr_out + self.bias
#         return aggr_out


import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GCNConvM2(MessagePassing):
    def __init__(self):
        super(GCNConvM, self).__init__(aggr='add') # "Add" aggregation.
        #self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [num_nodes, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        #edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        #x = self.lin(x)

        # Step 3-5: Start propagating messages.
        out = self.propagate(edge_index, x=x)
        #out = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
        return out

    def message(self, x_j, edge_index, size):
        # x_j has shape [num_edges, out_channels]

        # Step 3: Normalize node features.
        # row, col = edge_index
        # deg = degree(row, size[0], dtype=x_j.dtype)
        # deg_inv_sqrt = deg.pow(-0.5)
        # norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return x_j #norm.view(-1, 1) * 

    def update(self, aggr_out, x):
        # aggr_out has shape [num_nodes, out_channels]

        # Step 5: Return new node embeddings.
        out = torch.cat([x,aggr_out],dim=1)
        #out = x + aggr_out
        return out
    
class GCNConvM(GCNConv):
    def __init__(self, in_channels, out_channels,**kwargs):
        super().__init__(in_channels, out_channels,**kwargs)
        self.message_projector = InputProjectorSimple(2*out_channels, out_channels, dropout = 0.1, doNorm = True)
        
    def update(self, aggr_out, x):
        out = torch.cat([x,aggr_out],dim=1)
        return out
    
    # def message(self, x_i, x_j):
    #     # x_i has shape [E, in_channels]
    #     # x_j has shape [E, in_channels]

    #     # add edge features also
    #     tmp = torch.cat([x_i, x_j], dim=1)  # tmp has shape [E, 2 * in_channels]
    #     out = self.message_projector(tmp)
    #     return out
    
class SAGEConvM(SAGEConv):
    def update(self, aggr_out, x):
        out = torch.cat([x[0],aggr_out],dim=1)
        return out
    
class GATConvM(GATConv):
    pass
    # def update(self, aggr_out, x):
    #     out = torch.cat([x[0],aggr_out],dim=1)
    #     return out
    
class GATv2ConvM(GATv2Conv):
    pass
    # def update(self, aggr_out, x):
    #     out = torch.cat([x[0],aggr_out],dim=1)
    #     return out