import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from spd.models.gnan import TensorGNAN

# Load model
n_components = 1

gnan = TensorGNAN(in_channels=n_components, out_channels=1, n_layers=2, hidden_channels=16, device='cpu')
gnan.load_state_dict(torch.load(r"C:\\Users\\Knud\\uni\\spd\\spd\\experiments\\resid_mlp\\out\\nmasks1_stochrecon1.00e+00_stochreconlayer1.00e+00_p2.00e+00_impmin1.00e-05_C100_sd0_lr2.00e-03_bs2048_ft102_lay3_resid1000_mlp17_20260405_133756_929\\gnan.pth", map_location='cpu'))
gnan.eval()

n_layers = 6
max_distance = n_layers - 1

# Distances: raw and transformed
raw_distances = list(range(-max_distance, max_distance + 1))  # [-5, ..., 0, ..., 5]
transformed = [d / (1 + abs(d)) for d in raw_distances]       # what rho was trained on

# rho values
rho_y_values = np.zeros(len(transformed))
for i, val in enumerate(transformed):
    rho_dist = gnan.rho.forward(torch.tensor([val]).view(-1, 1)).detach()
    rho_y_values[i] = rho_dist

# Plot rho
sns.set_style("whitegrid")
plt.figure(figsize=(4, 2))
plt.plot(raw_distances, rho_y_values, marker='.', markersize=4)
plt.xlabel('Distance (signed)', size=9)
plt.ylabel('Distance function output', size=9)
plt.xticks(raw_distances, fontsize=8)
plt.yticks(fontsize=8)
plt.savefig("gnan_distance_function.png", dpi=150, bbox_inches="tight")
plt.close()

# f scores
f_scores = np.zeros(shape=(n_components,))
for i in range(n_components):
    f_scores[i] = gnan.fs[i].forward(torch.tensor([1.0]).view(-1, 1)).detach().flatten()[0]

# Plot f
plt.figure(figsize=(4, 2))
plt.bar([f"Feature {i}" for i in range(n_components)], f_scores)
plt.xlabel('Feature', size=9)
plt.ylabel('Feature function output', size=9)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.savefig("f_plot.png", dpi=150, bbox_inches="tight")
plt.close()

# Heatmap
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("custom", ["red", "white", "green"], N=100)

z = np.outer(f_scores, rho_y_values)
fig, ax = plt.subplots(figsize=(12, 3))
sns.heatmap(z, annot=True, fmt=".3f",
            xticklabels=[str(d) for d in raw_distances],
            yticklabels=[f"Feature {i}" for i in range(n_components)],
            cmap=cmap, center=0, annot_kws={"fontsize":10}, ax=ax)
plt.xlabel('Distance (signed)', size=12)
plt.ylabel('Feature', size=12)
plt.title('GNAN: f(activation) x rho(distance)', size=14)
plt.savefig("Outer product between f and rho.png", dpi=150, bbox_inches="tight")
plt.close()