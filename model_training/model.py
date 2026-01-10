import torch
import torch.nn as nn
from transformers import ASTConfig, ASTModel
import hdf5plugin

class Vimtopoeia_AST(nn.Module):
    def __init__(self, n_params=22, ast_model_path=None):
        """
        Args:
            n_params: Number of parameters to predict
            ast_model_path: Path to pretrained AST model directory
        """
        super().__init__()
        
        if ast_model_path is None:
            raise ValueError("ast_model_path must be provided")
        
        # Load Pretrained AST (AudioSet)
        # This is critical for convergence on small datasets
        self.ast = ASTModel.from_pretrained(
            ast_model_path,
            # attn_implementation="sdpa", # Enable if using torch >= 2.1.1 for speedup
        )
        
        # AST hidden size is 768
        # Input to FC: 768 (vocal) + 768 (ref) + 3 (osc) = 1539
        self.fc = nn.Sequential(
            nn.Linear(768 * 2 + 3, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, n_params)
        )

    def forward(self, spec_vocal, spec_ref, osc_one_hot):
        # spec_vocal: [B, 1024, 128]
        # AST expects: [B, 1024, 128]
        
        # Get pooled output (CLS token)
        # Note: ASTModel handles the patch embedding and positional encoding internally
        vocal_out = self.ast(spec_vocal).pooler_output # [B, 768]
        ref_out = self.ast(spec_ref).pooler_output     # [B, 768]
        
        # Concatenate
        combined = torch.cat([vocal_out, ref_out, osc_one_hot], dim=1)
        
        return self.fc(combined)
