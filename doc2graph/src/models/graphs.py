import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GATv2Conv, FAConv
import dgl.function as fn
import math
import numpy as np

from doc2graph.src.paths import CFGM
from doc2graph.src.utils import get_config
from doc2graph.src.models.utils import GCNConvM, SAGEConvM, InputProjectorSimple, GATConvM, GATv2ConvM


class SetModel():
    def __init__(self, name='e2e', device = 'cpu'):
        """ Create a SetModel object, that handles dinamically different version of Doc2Graph Model. Default "end-to-end" (e2e)

        Args:
            name (str) : Which model to train / test. Default: e2e [gcn, edge].
        
        Returns:
            SetModel object.
        """

        self.cfg_model = get_config(CFGM / name)
        self.name = self.cfg_model.name
        self.total_params = 0
        self.device = device
    
    def get_name(self) -> str:
        """ Returns model name.
        """
        return self.name
    
    def get_total_params(self) -> int:
        """ Returns number of model parameteres.
        """
        return self.total_params

    def get_model(self, nodes : int, edges : int, chunks : list, verbatim : bool = True) -> nn.Module:
        """Return the DGL model defined in the setting file

        Args:
            nodes (int) : number of nodes target class
            edges (int) : number of edges target class
            chunks (list) : list of indeces of chunks

        Returns:
            A PyTorch nn.Module, your DGL model.
        """
        print("\n### MODEL ###")
        print(f"-> Using {self.name}")

        if self.name == 'GCN':
            m = NodeClassifier(chunks, self.cfg_model.out_chunks, nodes, self.cfg_model.num_layers, F.relu, False, self.device)
        
        elif self.name == 'EDGE':
            m = EdgeClassifier(edges, self.cfg_model.num_layers, self.cfg_model.dropout, chunks, self.cfg_model.out_chunks, self.cfg_model.hidden_dim, self.device, self.cfg_model.doProject)

        elif self.name == 'E2E':
            edge_pred_features = int((math.log2(get_config('preprocessing').FEATURES.num_polar_bins) + nodes)*2)
            m = E2E(nodes, edges, self.cfg_model.num_layers, self.cfg_model.dropout, chunks, self.cfg_model.out_chunks, self.cfg_model.hidden_dim, self.device,  edge_pred_features, self.cfg_model.doProject)
        
        elif self.name == 'GAT':
            m = GAT(nodes, edges, self.cfg_model.n_heads, self.cfg_model.dropout, chunks, self.cfg_model.node_projector_dim, self.cfg_model.edge_projector_dim, self.device, self.cfg_model.doProject, self.cfg_model.doNorm)

        else:
            raise Exception(f"Error! Model {self.name} do not exists.")
        
        m.to(self.device)
        self.total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"-> Total params: {self.total_params}")
        print("-> Device: " + str(next(m.parameters()).is_cuda) + "\n")
        if verbatim: print(m)

        return m

################
##### GCNS #####

class NodeClassifier(nn.Module):
    def __init__(self,
                 in_chunks,
                 out_chunks,
                 n_classes,
                 n_layers,
                 activation,
                 dropout=0,
                 use_pp=False,
                 device='cuda:0'):
        super(NodeClassifier, self).__init__()

        self.projector = InputProjector(in_chunks, out_chunks, device)
        self.layers = nn.ModuleList()
        # self.dropout = nn.Dropout(dropout)
        self.n_layers = n_layers

        n_hidden = self.projector.get_out_lenght()

        # mp layers
        for i in range(0, n_layers - 1):
            self.layers.append(GcnSAGELayer(n_hidden, n_hidden, activation=activation, 
                        dropout=dropout, use_pp=False, use_lynorm=True))

        self.layers.append(GcnSAGELayer(n_hidden, n_classes, activation=None,
                                    dropout=False, use_pp=False, use_lynorm=False))

    def forward(self, g, h):
        
        h = self.projector(h)

        for l in range(self.n_layers):
            h = self.layers[l](g, h)
        
        return h

################
##### EDGE #####

class EdgeClassifier(nn.Module):

    def __init__(self, edge_classes, m_layers, dropout, in_chunks, out_chunks, hidden_dim, device, doProject=True):
        super().__init__()

        # Project inputs into higher space
        self.projector = InputProjector(in_chunks, out_chunks, device, doProject)

        # Perform message passing
        m_hidden = self.projector.get_out_lenght()
        self.message_passing = nn.ModuleList()
        self.m_layers = m_layers
        for l in range(m_layers):
            self.message_passing.append(GcnSAGELayer(m_hidden, m_hidden, F.relu, 0.))

        # Define edge predictori layer
        self.edge_pred = MLPPredictor(m_hidden, hidden_dim, edge_classes, dropout)  

    def forward(self, g, h):

        h = self.projector(h)

        for l in range(self.m_layers):
            h = self.message_passing[l](g, h)
        
        e = self.edge_pred(g, h)

        return e

################
###### E2E #####

class E2E(nn.Module):
    def __init__(self, node_classes, 
                       edge_classes, 
                       m_layers, 
                       dropout, 
                       in_chunks, 
                       out_chunks, 
                       hidden_dim, 
                       device,
                       edge_pred_features,
                       doProject=True):

        super().__init__()

        # Project inputs into higher space
        self.projector = InputProjector(in_chunks, out_chunks, device, doProject, dropout)

        # Perform message passing
        m_hidden = self.projector.get_out_lenght()
        # self.message_passing = nn.ModuleList()
        # self.m_layers = m_layers
        # for l in range(m_layers):
        #     self.message_passing.append(GcnSAGELayer(m_hidden, m_hidden, F.relu, 0.))
        self.message_passing = GcnSAGELayer(m_hidden, m_hidden, F.relu, dropout)

        # Define edge predictor layer
        self.edge_pred = MLPPredictor_E2E(m_hidden, hidden_dim, edge_classes, dropout,  edge_pred_features)

        # Define node predictor layer
        node_pred = []
        node_pred.append(nn.Linear(m_hidden, node_classes))
        node_pred.append(nn.LayerNorm(node_classes))
        self.node_pred = nn.Sequential(*node_pred)
        self.drop = nn.Dropout(dropout)

    def forward(self, g, h):

        h = self.projector(h)
        # for l in range(self.m_layers):
        #     h = self.message_passing[l](g, h)
        h = self.message_passing(g,h)
        n = self.node_pred(h)
        n = self.drop(n)
        e = self.edge_pred(g, h, n)
        
        return n, e
    
################
###### GAT #####

class GAT(nn.Module):
    def __init__(self, node_classes, 
                       num_edge_features, 
                       n_heads, 
                       dropout, 
                       in_chunks, 
                       node_projector_dim, 
                       edge_projector_dim,
                       device,
                       doProject=True, 
                       doNorm = True):

        super().__init__()
        
        self.nclasses = node_classes
        self.num_edge_features = num_edge_features
        self.num_node_features = sum(in_chunks)
        self.in_chunks = in_chunks
        self.edge_projector_dim = edge_projector_dim
        self.node_projector_dim = node_projector_dim
        self.doProject = doProject
        self.doNorm = doNorm
        
        if dropout>0:
            self.drop = nn.Dropout(dropout)
        else:
            self.drop = None

        m_hidden = sum(in_chunks)
        self.node_dim = node_projector_dim if self.node_projector_dim>0 else self.num_node_features
        edge_dim = edge_projector_dim if self.edge_projector_dim>0 else num_edge_features
        
        if doProject:
            
            self.node_projector = InputProjectorSimple(m_hidden, self.node_dim, dropout, doNorm)
            self.edge_projector = InputProjectorSimple(num_edge_features, edge_dim, dropout, doNorm)
                
            #self.message_passing = GcnSAGELayer(m_hidden, m_hidden, F.relu, dropout)
            #self.message_passing = GCNConv(self.node_dim, self.node_dim, edge_dim = edge_dim, heads=n_heads, dropout = dropout, v2 = True, improved = True, normalize = False, add_self_loops = False, bias = False, aggr='add')
            self.message_passing = GATv2ConvM(self.node_dim, self.node_dim, edge_dim = edge_dim, heads=n_heads, dropout = dropout, v2 = True, add_self_loops = False, aggr='add', bias=False)
            #self.message_passing = SAGEConvM(self.node_dim, self.node_dim, project = True)
        else:
            self.message_passing = GATv2ConvM(m_hidden, m_hidden, edge_dim = edge_dim, heads=n_heads, dropout = dropout, v2 = True, add_self_loops = False, aggr='add', bias=False)
            
        self.node_pred = InputProjectorSimple(2* n_heads*self.node_dim, node_classes, dropout, doNorm)
        

    def forward(self, data, return_attention_weights = None):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        if self.doProject:
            x = self.node_projector(x)
            e = self.edge_projector(edge_attr)
        else:
            e = edge_attr
            
        x = self.message_passing(x, edge_index)
        #x = self.message_passing(x, edge_index, edge_attr=e)
        x = F.relu(x)
        if self.doNorm:
            x = F.layer_norm(x, x.shape)
            
        if self.drop:
            x = self.drop(x)

        x = self.node_pred(x)
        
        return x


class GATLSTM(nn.Module):
    def __init__(self, node_classes, 
                       num_edge_features, 
                       n_heads, 
                       dropout, 
                       in_chunks, 
                       node_projector_dim, 
                       edge_projector_dim,
                       device,
                       edge_pred_features,
                       doProject=True, 
                       doNorm = True):

        super().__init__()
        
        self.nclasses = node_classes
        self.num_edge_features = num_edge_features
        self.num_node_features = sum(in_chunks)
        self.in_chunks = in_chunks
        self.in_chunks_cumsum = [0]
        self.in_chunks_cumsum.extend(list(np.cumsum(in_chunks)))
        self.edge_projector_dim = edge_projector_dim
        self.node_projector_dim = node_projector_dim
        self.doNorm = doNorm
        
        print(f'in_chunks: {in_chunks}')
        
        self.drop = nn.Dropout(dropout)

        edge_dim = edge_projector_dim if self.edge_projector_dim>0 else num_edge_features
        
        self.node_projector = [InputProjectorSimple(in_chunk, self.node_projector_dim, dropout, doNorm).to(device) for in_chunk in in_chunks]
        self.edge_projector = InputProjectorSimple(num_edge_features, edge_dim, dropout, doNorm)
        
        self.message_passing = [GATv2ConvM(3*self.node_projector_dim, self.node_projector_dim, edge_dim = edge_dim, heads=n_heads, dropout = dropout, v2 = True, add_self_loops = False, aggr='add', bias=False).to(device) for in_chunk in in_chunks]
                
        self.LSTM = nn.LSTM(2*self.node_projector_dim,2*self.node_projector_dim,bidirectional=True, batch_first = True)
        
        self.node_pred = InputProjectorSimple(2* n_heads*self.node_projector_dim, node_classes, dropout, doNorm)
        

    def forward(self, data,return_attention_weights = None):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        e = self.edge_projector(edge_attr)
        combined_tensor = []
        for i,node_projector in enumerate(self.node_projector):
            sub_x = x[:,self.in_chunks_cumsum[i]:self.in_chunks_cumsum[i+1]]
            sub_x = self.node_projector[i](sub_x)
            
            combined_tensor.append(sub_x)

        combined_tensor = torch.cat(combined_tensor, dim=1)
        #sub_x = self.message_passing[i](sub_x, edge_index)
        combined_tensor = self.message_passing[i](combined_tensor, edge_index, edge_attr=e)
        combined_tensor = F.relu(combined_tensor)
        if self.doNorm:
            combined_tensor = F.layer_norm(combined_tensor, combined_tensor.shape)
        combined_tensor = self.drop(combined_tensor)
        #print(sub_x.shape)
        
        #print(combined_tensor.shape)

        #output, (last_hidden, cell_state) = self.LSTM(combined_tensor)
        #x = self.node_pred(torch.cat((last_hidden[0], last_hidden[1]), dim=1))
        x = self.node_pred(combined_tensor)

        return x
