import torch

checkpoint_path = r"C:\Users\Knud\uni\spd\spd\experiments\tms\out\tms_n-features5_n-hidden2_n-hidden-layers0_feat_prob0.05_seed0_20260304_172557_381\tms.pth"
# Load checkpoint (map to CPU if you don't have GPU)
checkpoint = torch.load(checkpoint_path, map_location="cpu")


W1 = checkpoint["linear1.weight"]
W2 = checkpoint["linear2.weight"]
b2 = checkpoint["linear2.bias"]

print(W1.shape)
print(W2.shape)
print(b2.shape)


for name, tensor in checkpoint.items():
    print(f"\n{name}:")
    print(tensor)


decomposition_path = r"C:\Users\Knud\uni\spd\spd\experiments\tms\out\nmasks1_stochrecon1.00e+00_stochreconlayer1.00e+00_p1.00e+00_impmin3.00e-03_C20_sd0_lr1.00e-03_bs4096_ft5_hid2hid-layers0_20260305_180359_787\model_40000.pth"
decomposition_checkpoint = torch.load(decomposition_path, map_location="cpu")

print("\nDecomposition checkpoint keys:")
print(decomposition_checkpoint.keys())

components_1a = decomposition_checkpoint['components.linear1.A']
components_2b = decomposition_checkpoint['components.linear1.B']

print("\nComponents for linear1:")
print("A shape:", components_1a.shape)
print("B shape:", components_2b.shape)

components_2a = decomposition_checkpoint['components.linear2.A']
components_2b = decomposition_checkpoint['components.linear2.B']
print("\nComponents for linear2:")
print("A shape:", components_2a.shape)
print("B shape:", components_2b.shape)

gate1_a = decomposition_checkpoint['gates.linear1.mlp_in']
print("\nGate for linear1:")
print("Shape:", gate1_a.shape)



# I think A_i @ B_i gives me one rank 1 component. Summing them gives the full matrix, check this!!