import torch
from torch_geometric.data import Data

edge_index = torch.tensor([[0, 1, 2],
                           [0, 1, 2]], dtype=torch.long)
x = torch.tensor([[1], [1], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)

from torch_geometric.nn import GCNConv
from src.main import seed_everything
seed_everything(42)

conv1 = GCNConv(data.num_node_features, 1, add_self_loops = False, normalize = True, _explain = False, _apply_sigmoid = True)
z = conv1(x, edge_index)

print(z)