import torch
import torch.nn as nn
from transformers import ASTConfig, ASTModel

class M2_AST_Model(nn.Module):
    def __init__(self, n_params=73):
        super().__init__()
        
        # 1. Paper Specific Config [cite: 510, 511]
        config = ASTConfig(
            num_mel_bins=64,       # Paper optimization: 64 bins (Standard AST is 128)
            patch_size=16,         # Standard patch size
            hidden_size=768,       # Hidden dimension (Base ViT)
            num_hidden_layers=12,  # 12 Encoder layers
            num_attention_heads=12,
            max_length=2208,       # Provides exactly 1102 position embeddings for mel spec (220 time × 5 freq)
            qkv_bias=True
        )
        
        # 2. Initialize with Random Weights (No ImageNet/AudioSet)
        self.ast = ASTModel(config)
        
        # 3. The 3-Layer MLP Head 
        # "insert a small 3-layer MLP of width 768"
        self.mlp_head = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, n_params) # Final regression output
        )

    def forward(self, spec):
        # spec shape: [Batch, Time, Freq] or [Batch, 1, Freq, Time]
        
        # Ensure correct shape for HF AST: [Batch, Time, Freq]
        if spec.dim() == 4: 
            spec = spec.squeeze(1).transpose(1, 2)
        elif spec.dim() == 3 and spec.shape[1] == 64: 
            spec = spec.transpose(1, 2)

        # Forward pass
        outputs = self.ast(input_values=spec)
        embedding = outputs.pooler_output 
        
        # Regression Head
        pred_params = self.mlp_head(embedding)
        
        return pred_params