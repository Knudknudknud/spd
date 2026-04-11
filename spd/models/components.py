import einops
import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F
from spd.module_utils import init_param_


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
        

        self.self_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),)
        
        # Create n fully connected layers, one for each input feature.
        # This is one fully connected model for each input feature.
        # hidden_channels is the number of neurons in the "middle" layers. 
        # Output the amount of values to be returned.
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
    
        return mf + self.self_mlp(x_batch)




# class Gate(nn.Module):
#     """A gate that maps a single input to a single output."""

#     def __init__(self, C: int):
#         super().__init__()
#         self.weight = nn.Parameter(torch.empty((C,)))
#         self.bias = nn.Parameter(torch.zeros((C,)))
#         fan_val = 1  # Since each weight gets applied independently
#         init_param_(self.weight, fan_val=fan_val, nonlinearity="linear")

#     def forward(self, x: Float[Tensor, "... C"]) -> Float[Tensor, "... C"]:
#         return x * self.weight + self.bias




# class GateMLP(nn.Module):
#     """A gate with a hidden layer that maps a single input to a single output."""

#     def __init__(self, C: int, n_ci_mlp_neurons: int):
#         super().__init__()
#         self.n_ci_mlp_neurons = n_ci_mlp_neurons

#         self.mlp_in = nn.Parameter(torch.empty((C, n_ci_mlp_neurons)))
#         self.in_bias = nn.Parameter(torch.zeros((C, n_ci_mlp_neurons)))
#         self.mlp_out = nn.Parameter(torch.empty((C, n_ci_mlp_neurons)))
#         self.out_bias = nn.Parameter(torch.zeros((C,)))

#         init_param_(self.mlp_in, fan_val=1, nonlinearity="relu")
#         init_param_(self.mlp_out, fan_val=n_ci_mlp_neurons, nonlinearity="linear")

#     def forward(self, x: Float[Tensor, "... C"]) -> Float[Tensor, "... C"]:
#         hidden = (
#             einops.einsum(
#                 x,
#                 self.mlp_in,
#                 "... C, C n_ci_mlp_neurons -> ... C n_ci_mlp_neurons",
#             )
#             + self.in_bias
#         )
#         hidden = F.gelu(hidden)

#         out = (
#             einops.einsum(
#                 hidden,
#                 self.mlp_out,
#                 "... C n_ci_mlp_neurons, C n_ci_mlp_neurons -> ... C",
#             )
#             + self.out_bias
#         )
#         return out


class LinearComponent(nn.Module):
    """A linear transformation made from A and B matrices for SPD.

    NOTE: In the paper, we use V and U for A and B, respectively.

    The weight matrix W is decomposed as W = B^T @ A^T, where A and B are learned parameters.
    """

    def __init__(self, d_in: int, d_out: int, C: int, k: int, bias: Tensor | None):
        super().__init__()
        self.C = C
        self.k = k

        self.A = nn.Parameter(torch.empty(d_in, C, k))
        self.B = nn.Parameter(torch.empty(C, k, d_out))
        self.bias = bias

        init_param_(self.A, fan_val=d_out, nonlinearity="linear")
        init_param_(self.B, fan_val=C, nonlinearity="linear")

        self.mask: Float[Tensor, "... C"] | None = None  # Gets set on sparse forward passes

    @property
    def weight(self) -> Float[Tensor, "d_out d_in"]:
        """B^T @ A^T"""
        return einops.einsum(self.A, self.B, "d_in C k, C k d_out -> d_out d_in")

    # @torch.compile
    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        """Forward pass through A and B matrices.

        Args:
            x: Input tensor
            mask: Tensor which masks parameter components. May be boolean or float.
        Returns:
            output: The summed output across all components
        """
        component_acts = einops.einsum(x, self.A, "... d_in, d_in C k -> ... C k")

        if self.mask is not None:
            
            #component_acts *= self.mask
            component_acts = component_acts * self.mask.unsqueeze(-1)


        #out = einops.einsum(component_acts, self.B, "... C, C d_out -> ... d_out")
        out = einops.einsum(component_acts, self.B, "... C k, C k d_out -> ... d_out")
        if self.bias is not None:
            out += self.bias

        return out


class EmbeddingComponent(nn.Module):
    """An efficient embedding component for SPD that avoids one-hot encoding."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        C: int,
    ):
        super().__init__()
        self.C = C

        self.A = nn.Parameter(torch.empty(vocab_size, C))
        self.B = nn.Parameter(torch.empty(C, embedding_dim))

        init_param_(self.A, fan_val=embedding_dim, nonlinearity="linear")
        init_param_(self.B, fan_val=C, nonlinearity="linear")

        # For masked forward passes
        self.mask: Float[Tensor, "batch pos C"] | None = None

    @property
    def weight(self) -> Float[Tensor, "vocab_size embedding_dim"]:
        """A @ B"""
        return einops.einsum(
            self.A, self.B, "vocab_size C, ... C embedding_dim -> vocab_size embedding_dim"
        )

    # @torch.compile
    def forward(self, x: Float[Tensor, "batch pos"]) -> Float[Tensor, "batch pos embedding_dim"]:
        """Forward through the embedding component using nn.Embedding for efficient lookup

        NOTE: Unlike a LinearComponent, here we alter the mask with an instance attribute rather
        than passing it in the forward pass. This is just because we only use this component in the
        newer lm_decomposition.py setup which does monkey-patching of the modules rather than using
        a SPDModel object.

        Args:
            x: Input tensor of token indices
        """
        # From https://github.com/pytorch/pytorch/blob/main/torch/_decomp/decompositions.py#L1211
        component_acts = self.A[x]  # (batch pos C)

        if self.mask is not None:
            component_acts *= self.mask

        out = einops.einsum(
            component_acts, self.B, "batch pos C, ... C embedding_dim -> batch pos embedding_dim"
        )
        return out
