from models.gnan import TensorGNAN
import torch
import torch_geometric as pyg
import torch.nn as nn


x = torch.randn((10, 5))
print(x)
edge_index = torch.tensor([[0,1,2,3,4,5], [6,7,8,9,1,2]])
node_distances = torch.randn((10, 10))
inputs = pyg.data.Data(x=x, edge_index=edge_index, node_distances=node_distances)

model = TensorGNAN(in_channels=5, out_channels=3, n_layers=2, hidden_channels=4)
print(model)

out = model(inputs)
print(out)


