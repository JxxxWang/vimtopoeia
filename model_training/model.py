import torch
import torch.nn as nn
from transformers import ASTModel

class Vimtopoeia_AST(nn.Module):
    def __init__(self, n_params=22, ast_model_path=None):
        super().__init__()
        
        if ast_model_path is None:
            raise ValueError("ast_model_path must be provided")
        
        # Load Pretrained AST
        self.ast = ASTModel.from_pretrained(ast_model_path)
        
        # --- FIX: INPUT SIZE IS 768 (One Audio), NOT 1539 (Two Audio + Osc) ---
        self.fc = nn.Sequential(
            nn.Linear(768, 512),  # Taking the embedding directly
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, n_params)
        )

    def forward(self, spec, return_embedding=False):
        """
        Args:
            spec: [Batch, 1024, 128] - The Mel Spectrogram
            return_embedding: If True, returns the 768-dim vector (The 'Thought')
                              If False, returns the parameters (The 'Answer')
        """
        # 1. Run through Transformer Layers
        outputs = self.ast(spec)
        embedding = outputs.pooler_output # Shape: [Batch, 768]
        
        # 2. Consistency Branch
        if return_embedding:
            return embedding
            
        # 3. Prediction Branch
        params = self.fc(embedding)
        return params