import torch
from torch_geometric.nn import GraphConv, GINConv, GATv2Conv, GraphSAGE, TransformerConv
from torch_geometric.nn import global_mean_pool
import torch.nn as nn
import torch_geometric as pyg
import torch.nn.functional as F


class TensorGNAN(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers, hidden_channels=None, bias=True, dropout=0.0,
                 device='cpu', rho_per_feature=False, normalize_rho=False, is_graph_task=False, readout_n_layers=1):
        super().__init__()

        self.device = device
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.bias = bias
        self.dropout = dropout
        self.rho_per_feature = rho_per_feature
        self.normalize_rho = normalize_rho
        self.fs = nn.ModuleList()
        self.is_graph_task = is_graph_task

        #Create n fully connected layers, one for each input feature.
        #This is one fully connected model for each input feature.
        #hidden_channels is the number of neurons in the "middle" layers. 
        #Output the amount of values to be returned.
        for _ in range(in_channels):
            if n_layers == 1:
                curr_f = [nn.Linear(1, self.out_channels, bias=bias)]
            else:
                curr_f = [nn.Linear(1, hidden_channels, bias=bias), nn.ReLU(), nn.Dropout(p=dropout)]
                for _ in range(1, n_layers - 1):
                    curr_f.append(nn.Linear(hidden_channels, hidden_channels, bias=bias))
                    curr_f.append(nn.ReLU())
                    curr_f.append(nn.Dropout(p=dropout))
                curr_f.append(nn.Linear(hidden_channels, self.out_channels, bias=bias))
            self.fs.append(nn.Sequential(*curr_f))

        rho_bias = True
        if is_graph_task:  rho_bias = False



        #Rho feature, did i delete the if statement for per feature?
        if rho_per_feature:
            assert("rho_per_feature not implemented yet")

        #Again create a fully connected model, for the rho function.
        if n_layers == 1:
            self.rho = [nn.Linear(1, self.out_channels, bias=rho_bias)]
        else:
            self.rho = [nn.Linear(1, hidden_channels, bias=rho_bias), nn.ReLU()]
            for _ in range(1, n_layers - 1):
                self.rho.append(nn.Linear(hidden_channels, hidden_channels, bias=rho_bias))
                self.rho.append(nn.ReLU())
            self.rho.append(nn.Linear(hidden_channels, self.out_channels, bias=rho_bias))
        self.rho = nn.Sequential(*self.rho)

        for name, param in self.named_parameters():
            if 'weight' in name:
                #changed from 0.01
                nn.init.xavier_normal_(param, gain=1.0)
            elif 'bias' in name:
                nn.init.constant_(param, 0)


#It doesnt even use the idnex?
    def forward(self, inputs):
        x, edge_index, node_distances = inputs.x, inputs.edge_index, inputs.node_distances

        #Create a tensor of size (num_nodes, num_features, out_channels)
        fx = torch.empty(x.size(0), x.size(1), self.out_channels).to(self.device)
        #For each feature,
        for feature_index in range(x.size(1)):
            #This creates a row vector of the feature values for all nodes.
            feature_col = x[:, feature_index]
            #Reshape it to be a column vector, so (num_nodes, 1)
            feature_col = feature_col.view(-1, 1)

            #fs is a modulelist (a "list"), index into spot k,
            #and add the column vector to the list
            feature_col = self.fs[feature_index](feature_col)
            #Also add this column to the tensor fx,
            #Thus the k'th column of the tensor has the k'th feature of all nodes.
            fx[:, feature_index] = feature_col
        #We now have a tensor of size (num_nodes, features, out_channels)
        #out channels is still zero.

        #Reshape it into (out_channels, num_nodes, features)
        fx_perm = torch.permute(fx, (2, 0, 1))
        if self.normalize_rho:
            node_distances = torch.div(node_distances, inputs.normalization_matrix)
        
        
        #First take the n x n matrix of node distances
        #flatten it so it becomes a list of all rows concatenated
        #reshape it into a column vector (num_nodes*num_nodes, 1)
        #Send it through the rho function,
            #  which outputs (num_nodes*num_nodes, out_channels)
        #Then reshape it back into (num_nodes, num_nodes, out_channels)
        m_dist = self.rho(node_distances.flatten().view(-1, 1)).view(x.size(0), x.size(0), self.out_channels)
        
        #Reshape it into (out_channels, num_nodes, num_nodes)
        #Can now index it as [feature, node_i, node_j] to get the contribution of node i,j. 
        m_dist_perm = torch.permute(m_dist, (2, 0, 1))

        #for each feature k,
        #Take the product:
        #m dist has shape (out_channels, num_nodes, num_nodes)
        #fx_perm has shape (out_channels, num_nodes, features)
        #Take the row for each node and its distances to the other nodes
        #multiply it by the feature values of the other nodes.
        mf = torch.matmul(m_dist_perm, fx_perm)
        #This means we have a tensor for each feature k,
        #Inside of each feature we have a "matrix",
        #This matrix has the feature value of all neighbouring nodes,
        #Multiplied by their distance
        
        #Sum over dim1, gives the total value of feature k in the graph.
        #Sum over dim2, sums the contributions of all features into one value in R^1.?
        if not self.is_graph_task:
            out = torch.sum(mf, dim=2)
        #Hence we sum over the features, and we get the weighted sum
        #Of the neighbouring nodes, for each feature.

        else:
            hidden = torch.sum(mf, dim=1)

            out = torch.sum(hidden, dim=1).view(1, -1)
        return out.T

        
    def forward_batched(self, x_batch, dist_batch):
        """x_batch: (S, N, F), dist_batch: (N, N). Same graph, different features."""
        #Batch size, num nodes, num features
        S, N, F = x_batch.shape


        #Create an empty tensor of dimension (Batch*nodes, feature, out_channels)
        fx = torch.empty(S * N, F, self.out_channels, device=self.device)

        #can it not just do it directly without unpacking?
    

        for feat_idx in range(F):
            #Take all feature values for the k'th feature, 
            #For all nodes across all batches.
            #Stack these columns into a column vector
            feat_col = x_batch[:, :, feat_idx].reshape(-1, 1)
            #Run the column vector through the MLP for the k'th feature
            #Add it to the respective column in the tensor fx
            fx[:, feat_idx, :] = self.fs[feat_idx](feat_col)
        
        #Reshape it back into (Batch, nodes, feature, out_channels)
        fx = fx.reshape(S, N, F, self.out_channels)

        # rho: computed once
        #Flatten the n x n distance matrix, and turn it into a column vector,
        #and run it through the rho nn, to get a vector of size (num_nodes*num_nodes, out_channels)
        dist_embed = self.rho(dist_batch.flatten().view(-1, 1))
        #View vs reshape, is just if it is contiguous in memory???
        #Ive just used them interchangeably, but maybe it matters.
        #Reshape it back into (num_nodes, num_nodes, out_channels)
        dist_embed = dist_embed.view(N, N, self.out_channels)

        # batched matmul
        fx_perm = fx.permute(0, 3, 1, 2)                    # (B, out_channels, N, F)
        m_dist = dist_embed.permute(2, 0, 1).unsqueeze(0)   # (1, out_channels, N, N)

        #Torch multiplication only cares about the last two dimensions,
        #So since m_dist has shape 1 in the first dimension, it will broadcast (repeat it B times https://docs.pytorch.org/docs/stable/notes/broadcasting.html)
        #Then for each batch, for each out_channel, take the weighte sum of the features of the neighboring nodes,
        
        #After this step mf, is a matrix of size batch, out_channels, num_nodes, features
        #The entry of the inner two dimensions means [i,j] is the sum over all neighbors of i on the j'th feature, weighted by the distance to the neighbor and the rho function.
        mf = torch.matmul(m_dist, fx_perm)                  # (B, out_channels, N, F)

        #We however expect a single output per out_channel, so we sum the features together
        mf = mf.sum(dim=3).permute(0, 2, 1)                 # (B, N, out_channels)
        return mf


class GNAN(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers, hidden_channels=None, bias=True, dropout=0.0,
                 device='cpu', normalize_rho=True, rho_per_feature=False):
        super().__init__()

        self.device = device
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.num_layers = n_layers
        self.bias = bias
        self.dropout = dropout
        self.rho_per_feature = rho_per_feature
        self.normalize_rho = normalize_rho
        self.fs = nn.ModuleList()

        for _ in range(in_channels):
            if n_layers == 1:
                curr_f = [nn.Linear(1, out_channels, bias=bias)]
            else:

                #Take a layer, with 1D input feature
                curr_f = [nn.Linear(1, hidden_channels, bias=bias), nn.ReLU(), nn.Dropout(p=dropout)]
                for _ in range(1, n_layers - 1):
                    curr_f.append(nn.Linear(hidden_channels, hidden_channels, bias=bias))
                    curr_f.append(nn.ReLU())
                    curr_f.append(nn.Dropout(p=dropout))
                curr_f.append(nn.Linear(hidden_channels, out_channels, bias=bias))
            self.fs.append(nn.Sequential(*curr_f))

        if rho_per_feature:
            self.rhos = nn.ModuleList()
            for _ in range(in_channels):
                if n_layers == 1:
                    self.rho = [nn.Linear(1, out_channels, bias=bias)]
                else:
                    self.rho = [nn.Linear(1, hidden_channels, bias=bias), nn.ReLU()]
                    for _ in range(1, n_layers - 1):
                        self.rho.append(nn.Linear(hidden_channels, hidden_channels, bias=bias))
                        self.rho.append(nn.ReLU())
                    if not rho_per_feature:
                        self.rho.append(nn.Linear(hidden_channels, 1, bias=bias))
                    else:
                        self.rho.append(nn.Linear(hidden_channels, out_channels, bias=bias))

                self.rhos.append(nn.Sequential(*self.rho))
        else:
            if n_layers == 1:
                self.rho = [nn.Linear(1, out_channels, bias=bias)]
            else:
                self.rho = [nn.Linear(1, hidden_channels, bias=bias), nn.ReLU()]
                for _ in range(1, n_layers - 1):
                    self.rho.append(nn.Linear(hidden_channels, hidden_channels, bias=bias))
                    self.rho.append(nn.ReLU())
                if not rho_per_feature:
                    self.rho.append(nn.Linear(hidden_channels, 1, bias=bias))
                else:
                    self.rho.append(nn.Linear(hidden_channels, out_channels, bias=bias))

        self.rho = nn.Sequential(*self.rho)

    def init_params(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.init_std)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, inputs, node_ids=None):
        x, edge_index, node_distances = inputs.x, inputs.edge_index, inputs.node_distances
        if node_ids is None:
            node_ids = range(x.size(0))
        fx = torch.empty(x.size(0), x.size(1), self.out_channels).to(self.device)
        for feature_index in range(x.size(1)):
            feature_col = x[:, feature_index]
            feature_col = feature_col.view(-1, 1)
            feature_col = self.fs[feature_index](feature_col)
            fx[:, feature_index] = feature_col

        f_sums = fx.sum(dim=1)
        stacked_results = torch.empty(len(node_ids), self.out_channels).to(self.device)
        for j, node in enumerate(node_ids):
            node_dists = node_distances[node]
            normalization = inputs.normalization_matrix[node]
            rho_dist = self.rho(node_dists.view(-1, 1))
            if self.normalize_rho:
                if rho_dist.size(1) == 1:
                    rho_dist = torch.div(rho_dist, normalization.view(-1, 1))
                else:
                    for i in range(rho_dist.size(1)):
                        rho_dist[:, i] = torch.div(rho_dist[:, i], normalization)
            pred_for_node = torch.sum(torch.mul(rho_dist, f_sums), dim=0)
            stacked_results[j] = pred_for_node.view(1, -1)

        return stacked_results

    def print_rho_params(self):
        for name, param in self.rho.named_parameters():
            print(name, param)
    