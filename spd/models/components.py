import einops
import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F
from spd.module_utils import init_param_



import torch
import torch.nn as nn
import torch.nn.functional as F



#Temp code, used to see if we can use fast attention that is linear in N https://arxiv.org/abs/2009.14794
# class PerformerAttention(nn.Module):
#     def __init__(self, in_channels, out_channels, hidden_channels, n_features=None, dropout=0.0):
#         """
#         Args:
#             in_channels:     input feature dim (F in your x_batch)
#             out_channels:    output dim
#             hidden_channels: dim of Q/K/V (d)
#             n_features:      number of random features (m). Defaults to hidden_channels.
#         """
#         super().__init__()
#         self.hidden_channels = hidden_channels
#         self.n_features = n_features if n_features is not None else hidden_channels

#         self.W_q = nn.Linear(in_channels, hidden_channels)
#         self.W_k = nn.Linear(in_channels, hidden_channels)
#         self.W_v = nn.Linear(in_channels, hidden_channels)

#         # Random projection matrix — fixed, not trained
#         # Shape: (m, d)
#         self.register_buffer(
#             "random_features",
#             torch.randn(self.n_features, hidden_channels) / (hidden_channels ** 0.25),
#         )

#         self.out_projection = nn.Sequential(
#             nn.Linear(in_channels + hidden_channels, hidden_channels),
#             nn.ReLU(),
#             nn.Dropout(p=dropout),
#             nn.Linear(hidden_channels, out_channels),
#         )

#     def favor_plus(self, x):
#         """Positive random features approximating exp(q·k).
#         x: (S, N, d) → (S, N, m)
#         """
#         # Shift for numerical stability: subtract max before exp
#         projected = x @ self.random_features.T  # (S, N, m)
#         norm = (x ** 2).sum(dim=-1, keepdim=True) / 2  # (S, N, 1)
#         # Subtract max across features for stability
#         stabilizer = projected.max(dim=-1, keepdim=True).values
#         return torch.exp(projected - norm - stabilizer) / (self.n_features ** 0.5)

#     def forward_batched(self, x_batch):
#         # x_batch: (S, N, F)
#         Q = self.W_q(x_batch)  # (S, N, d)
#         K = self.W_k(x_batch)
#         V = self.W_v(x_batch)

#         # Scale Q by 1/sqrt(d) to match softmax convention
#         Q = Q / (self.hidden_channels ** 0.25)
#         K = K / (self.hidden_channels ** 0.25)

#         # Apply positive random features
#         phi_Q = self.favor_plus(Q)  # (S, N, m)
#         phi_K = self.favor_plus(K)  # (S, N, m)

#         # Compute S = sum_j phi(K_j) V_j^T, shape (S, m, d)
#         KV = phi_K.transpose(-2, -1) @ V  # (S, m, d)

#         # Compute z = sum_k phi(K_k), shape (S, m)
#         z = phi_K.sum(dim=1)  # (S, m)

#         # Numerator: phi(Q) @ KV, shape (S, N, d)
#         numerator = phi_Q @ KV

#         # Denominator: phi(Q) @ z, shape (S, N)
#         denominator = (phi_Q @ z.unsqueeze(-1)).squeeze(-1) + 1e-6

#         out = numerator / denominator.unsqueeze(-1)  # (S, N, d)

#         combined = torch.cat([x_batch, out], dim=-1)
#         return self.out_projection(combined)


from performer_pytorch import SelfAttention

# class PerformerAttention(nn.Module):
#     def __init__(self, in_channels, out_channels, hidden_channels, nb_features=None, dropout=0.0):
#         super().__init__()
#         Performer expects dim = input dim, and projects internally
#         self.attn = SelfAttention(
#             dim=in_channels,
#             heads=1,
#             dim_head=hidden_channels,
#             nb_features=nb_features,  # None defaults to d*log(d)
#             causal=False,
#         )

#         self.out_projection = nn.Sequential(
#             nn.Linear(2 * in_channels, hidden_channels),
#             nn.ReLU(),
#             nn.Dropout(p=dropout),
#             nn.Linear(hidden_channels, out_channels),
#         )

#     def forward_batched(self, x_batch):
#         x_batch: (S, N, F)
#         out = self.attn(x_batch)  # (S, N, F) — same shape
#         combined = torch.cat([x_batch, out], dim=-1)  # (S, N, 2F)
#         return self.out_projection(combined)  # (S, N, 1)

from performer_pytorch import FastAttention

# class PerformerAttention(nn.Module):
#     def __init__(self, in_channels, out_channels, hidden_channels, nb_features=None):
#         super().__init__()
#         self.W_q = nn.Linear(in_channels, hidden_channels)
#         self.W_k = nn.Linear(in_channels, hidden_channels)
#         self.W_v = nn.Linear(in_channels, hidden_channels)

#         self.fast_attn = FastAttention(
#             dim_heads=hidden_channels,
#             nb_features=nb_features,
#             causal=False,
#         )

#         self.out_projection = nn.Sequential(
#             nn.Linear(in_channels + hidden_channels, hidden_channels),
#             nn.ReLU(),
#             nn.Linear(hidden_channels, out_channels),
#         )
        

#     def forward_batched(self, x_batch):
#         Q = self.W_q(x_batch).unsqueeze(1)  # (S, 1, N, d)
#         K = self.W_k(x_batch).unsqueeze(1)
#         V = self.W_v(x_batch).unsqueeze(1)

#         out = self.fast_attn(Q, K, V).squeeze(1)  # (S, N, d)

#         combined = torch.cat([x_batch, out], dim=-1)
#         return self.out_projection(combined)
    


class PerformerAttention(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, nb_features=None):
        super().__init__()
        self.W_q = nn.Linear(in_channels, hidden_channels)
        self.W_k = nn.Linear(in_channels, hidden_channels)
        self.W_v = nn.Linear(in_channels, hidden_channels)

        self.fast_attn = FastAttention(
            dim_heads=hidden_channels,
            nb_features=nb_features,
            causal=False,
        )

        # Passthrough path: in_channels -> 1
        self.x_proj = nn.Linear(in_channels, out_channels)

        # Attention correction: hidden_channels -> 1, zero-init
        self.attn_proj = nn.Linear(hidden_channels, out_channels)
        nn.init.zeros_(self.attn_proj.weight)
        nn.init.zeros_(self.attn_proj.bias)

    def forward_batched(self, x_batch):
        Q = self.W_q(x_batch).unsqueeze(1)
        K = self.W_k(x_batch).unsqueeze(1)
        V = self.W_v(x_batch).unsqueeze(1)

        attn_out = self.fast_attn(Q, K, V).squeeze(1)

        return self.x_proj(x_batch) + self.attn_proj(attn_out)

class Transformer(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.W_q = nn.Linear(in_channels, hidden_channels)
        self.W_k = nn.Linear(in_channels, hidden_channels)
        self.W_v = nn.Linear(in_channels, hidden_channels)

        self.out_projection = nn.Sequential(
            nn.Linear(in_channels + hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward_batched(self, x_batch):

        Q = self.W_q(x_batch)  # (S, N, d)
        K = self.W_k(x_batch)  # (S, N, d)
        V = self.W_v(x_batch)  # (S, N, d)

        #Compute attention
        out = F.scaled_dot_product_attention(Q, K, V)

        combined = torch.cat([x_batch, out], dim=-1)
        return self.out_projection(combined)

    
class TensorGNAN(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers, hidden_channels=None, bias=True, dropout=0.0,
                 rho_per_feature=False, normalize_rho=False, is_graph_task=False, readout_n_layers=1):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.bias = bias
        self.dropout = dropout
        self.rho_per_feature = rho_per_feature
        self.normalize_rho = normalize_rho
        self.fs = nn.ModuleList()
        self.is_graph_task = is_graph_task

        self.out_projection = nn.Sequential(
            nn.Linear(2 * in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_channels, out_channels),
        )

        
        # Create n fully connected layers, one for each input feature.
        # This is one fully connected model for each input feature.
        # hidden_channels is the number of neurons in the "middle" layers. 
        # Output the amount of values to be returned.
        for _ in range(in_channels):
            if n_layers == 1:
                curr_f = [nn.Linear(1, in_channels, bias=bias)]
            else:
                curr_f = [nn.Linear(1, hidden_channels, bias=bias), nn.ReLU(), nn.Dropout(p=dropout)]
                for _ in range(1, n_layers - 1):
                    curr_f.append(nn.Linear(hidden_channels, hidden_channels, bias=bias))
                    curr_f.append(nn.ReLU())
                    curr_f.append(nn.Dropout(p=dropout))
                curr_f.append(nn.Linear(hidden_channels, out_channels, bias=bias))
            self.fs.append(nn.Sequential(*curr_f))

        rho_bias = True
        if is_graph_task:  rho_bias = False


        rho_per_feature = rho_per_feature if in_channels > 1 else False
        num_rhos = in_channels if rho_per_feature else 1
        self.rhos = nn.ModuleList()
        for _ in range(num_rhos):
            if n_layers == 1:
                rho_layers = [nn.Linear(2, 1, bias=rho_bias)]
            else:
                rho_layers = [nn.Linear(2, hidden_channels, bias=rho_bias), nn.ReLU()]
                for _ in range(1, n_layers - 1):
                    rho_layers.append(nn.Linear(hidden_channels, hidden_channels, bias=rho_bias))
                    rho_layers.append(nn.ReLU())
                rho_layers.append(nn.Linear(hidden_channels, 1, bias=rho_bias))
            self.rhos.append(nn.Sequential(*rho_layers))

        for name, param in self.named_parameters():
            if 'weight' in name:
                #changed from 0.01
                nn.init.xavier_normal_(param, gain=1.0)
            elif 'bias' in name:
                nn.init.constant_(param, 0)


    # def forward_batched(self, x_batch, dist_batch):
    #     """x_batch: (S, N, F), dist_batch: (N, N). Same graph, different features."""
    #     #Batch size, num nodes, num features
    #     S, N, F = x_batch.shape


    #     #Create an empty tensor of dimension (Batch*nodes, feature, out_channels)
    #     fx = torch.empty(S * N, F, self.out_channels, device=x_batch.device)

    #     #can it not just do it directly without unpacking?
    

    #     for feat_idx in range(F):
    #         #Take all feature values for the k'th feature, 
    #         #For all nodes across all batches.
    #         #Stack these columns into a column vector
    #         feat_col = x_batch[:, :, feat_idx].reshape(-1, 1)
    #         #Run the column vector through the MLP for the k'th feature
    #         #Add it to the respective column in the tensor fx
    #         fx[:, feat_idx, :] = self.fs[feat_idx](feat_col)
        
    #     #Reshape it back into (Batch, nodes, feature, out_channels)
    #     fx = fx.reshape(S, N, F, self.out_channels)

    #     # rho: computed once
    #     #Flatten the n x n distance matrix, and turn it into a column vector,
    #     #and run it through the rho nn, to get a vector of size (num_nodes*num_nodes, out_channels)
    #     dist_embed = self.rho(dist_batch.flatten().view(-1, 1))
    #     #View vs reshape, is just if it is contiguous in memory???
    #     #Ive just used them interchangeably, but maybe it matters.
    #     #Reshape it back into (num_nodes, num_nodes, out_channels)
    #     dist_embed = dist_embed.view(N, N, self.out_channels)

    #     # batched matmul
    #     fx_perm = fx.permute(0, 3, 1, 2)                    # (B, out_channels, N, F)
    #     m_dist = dist_embed.permute(2, 0, 1).unsqueeze(0)   # (1, out_channels, N, N)

    #     #Torch multiplication only cares about the last two dimensions,
    #     #So since m_dist has shape 1 in the first dimension, it will broadcast (repeat it B times https://docs.pytorch.org/docs/stable/notes/broadcasting.html)
    #     #Then for each batch, for each out_channel, take the weighte sum of the features of the neighboring nodes,
        
    #     #After this step mf, is a matrix of size batch, out_channels, num_nodes, features
    #     #The entry of the inner two dimensions means [i,j] is the sum over all neighbors of i on the j'th feature, weighted by the distance to the neighbor and the rho function.
    #     mf = torch.matmul(m_dist, fx_perm)                  # (B, out_channels, N, F)

    #     #We however expect a single output per out_channel, so we sum the features together
    #     mf = mf.sum(dim=3).permute(0, 2, 1)                 # (B, N, out_channels)
    
    #     return mf + self.self_mlp(x_batch)



    # def forward_batched(self, x_batch, dist_batch):
    #     """
    #     x_batch:    (S, N, F)
    #     dist_batch: (N, N, 2) 
    #     """
    #     S, N, F = x_batch.shape

    #     # --- per-feature MLPs ---
    #     fx = torch.empty(S * N, F, self.out_channels, device=x_batch.device)
    #     for feat_idx in range(F):
    #         feat_col = x_batch[:, :, feat_idx].reshape(-1, 1)
    #         fx[:, feat_idx, :] = self.fs[feat_idx](feat_col)
    #     fx = fx.reshape(S, N, F, self.out_channels)

    #     # Sum over features → (S, N, out_channels)
    #     f_sums = fx.sum(dim=2)

    #     # --- rho: (N, N, 2) → (N, N, 1) → (N, N) ---
    #     dist_weights = self.rho(dist_batch.reshape(-1, 2))  # (N*N, 1)
    #     dist_weights = dist_weights.view(N, N)              # (N, N)

    #     # --- aggregation ---
    #     # (1, N, N) @ (S, N, out) → (S, N, out)
    #     mf = torch.matmul(dist_weights.unsqueeze(0), f_sums)
    #     combined = torch.cat([x_batch, mf], dim=-1)
    #     return self.out_projection(combined)

    def forward_batched(self, x_batch, dist_batch):
        S, N, F = x_batch.shape

        # --- per-feature MLPs: (S, N, F) → (S, N, F) ---
        fx = torch.empty(S * N, F, device=x_batch.device)
        for feat_idx in range(F):
            feat_col = x_batch[:, :, feat_idx].reshape(-1, 1)
            fx[:, feat_idx] = self.fs[feat_idx](feat_col).squeeze(-1)
        fx = fx.reshape(S, N, F)

        rho_input = dist_batch.reshape(-1, 2)
        if self.rho_per_feature:
            mf = torch.empty(S, N, F, device=x_batch.device)
            for feat_idx in range(F):
                dist_weights = self.rhos[feat_idx](rho_input).view(N, N)
                mf[:, :, feat_idx] = torch.matmul(
                    dist_weights.unsqueeze(0), fx[:, :, feat_idx].unsqueeze(-1)
                ).squeeze(-1)

        # --- rho: (N, N, 2) → (N, N) ---
        else:
            dist_weights = self.rhos[0](rho_input).view(N, N)

        # --- aggregation: (1, N, N) @ (S, N, F) → (S, N, F) ---
            mf = torch.matmul(dist_weights.unsqueeze(0), fx)

        combined = torch.cat([x_batch, mf], dim=-1)  # (S, N, 2F)
        return self.out_projection(combined)           # (S, N, 1)

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
