from transformers import ASTConfig, ASTModel

# Your mel spec produces 1102 patches (220 time × 5 freq with stride=10)
target_patches = 1102

print("Finding exact max_length for 1102 patches...\n")

# Binary search for the right max_length
for test_max_len in range(2100, 2300, 4):
    config = ASTConfig(num_mel_bins=64, patch_size=16, max_length=test_max_len)
    model = ASTModel(config)
    num_pos = model.embeddings.position_embeddings.shape[1]
    
    if num_pos == target_patches:
        print(f"✓✓✓ EXACT MATCH: max_length={test_max_len} -> {num_pos} position embeddings")
        break
    elif abs(num_pos - target_patches) <= 2:
        print(f"  Close: max_length={test_max_len} -> {num_pos} position embeddings (diff: {num_pos - target_patches})")
