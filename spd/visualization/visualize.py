import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from spd.models.gnan import TensorGNAN

# Load model
gnan = TensorGNAN(in_channels=1, out_channels=1, n_layers=2, hidden_channels=16, device='cpu')

gnan.load_state_dict(torch.load("spd\\experiments\\resid_mlp\\out\\gnan_version\\gnan.pth", map_location='cpu'))

#gnan.load_state_dict(torch.load("spd\\experiments\\tms\\out\\to_show_off\\40_10_gnan_latest_2_hidden\\gnan.pth", map_location='cpu'))
gnan.eval()

n_layers = 6  # number of layers in your model
max_distance = n_layers - 1

# rho values over all distances
y_input_values = torch.tensor([i for i in range(max_distance + 1)], dtype=torch.float32)
rho_y_values = np.zeros(shape=(y_input_values.size(0),))
for i, val in enumerate(y_input_values):
    rho_dist = gnan.rho.forward(val.view(-1, 1)).detach()
    rho_y_values[i] = rho_dist

# plot for distance function
sns.set_style("whitegrid")
plt.figure(figsize=(4, 2))

x_ticks = [i for i in range(max_distance+1)]

#remove the paddings in the beggining and end of the plot
plt.xlim(-0.5, max_distance-0.5)
plt.plot(x_ticks, rho_y_values, marker='.', markersize=4)
# plt.xticks(distance_ticks)
plt.xlabel('Distance', size=9)
plt.ylabel('Distance function output', size=9)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

plt.savefig("gnan_distance_function.png", dpi=150, bbox_inches="tight")
plt.close()

# f scores over all components
x_ticks = [i for i in range(max_distance + 1)]
n_components = 1  
f_scores = np.zeros(shape=(n_components,))
for i in range(n_components):
    f_scores[i] = gnan.fs[i].forward(torch.tensor([1.0]).view(-1, 1)).detach().flatten()[0]

# plot a bar plot of the f_scors with the feautre names for each bar
plt.figure(figsize=(4, 2))
plt.bar("only feature", f_scores)

plt.xlabel('Feature (atom)', size=9)
plt.ylabel('Feature function output', size=9)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.savefig("f_plot.png", dpi=150, bbox_inches="tight")
plt.close()


from matplotlib.colors import LinearSegmentedColormap
colors = ["red", "white", "green"]  # Red to white to green
n_bins = 100  # Number of bins in the colormap
cmap_name = "custom_colormap"
# Create the colormap
cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
plot_x = torch.tensor(np.arange(max_distance)).long()
plot_y = torch.zeros((len(f_scores), 1))

#plot heatmap
z = np.outer(f_scores, rho_y_values)
fig, ax= plt.subplots(figsize=(45, 15.5))
sns.heatmap(z, annot=True, fmt=".2f", xticklabels=x_ticks, yticklabels="only_feature", cmap=cmap,
            center=0, annot_kws={"fontsize":16}, cbar=False ,ax=ax)
cbar = plt.colorbar(ax.collections[0], ax=ax, use_gridspec=True, aspect=70)

cbar.ax.set_position([0.75, 0.1, 2, 0.755])
cbar.ax.tick_params(labelsize=20)
plt.xticks(fontsize=20, weight = 'bold')
plt.yticks(fontsize=20, weight = 'bold')
plt.xlabel('Distance', size=30)
plt.ylabel('Feature (atom)', size=30)
plt.title('Outer product between f and rho', size=20)
plt.savefig("Outer product between f and rho.png", dpi=150, bbox_inches="tight")


