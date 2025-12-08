import torch
import torch.nn as nn


class SmallTransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim: int = 64,
        num_heads: int = 2,
        num_layers: int = 2,
        num_classes: int = 2,
        max_len: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        
        # Add embedding dropout to prevent overfitting
        self.embedding_dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Add layer normalization before classification
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Increase dropout before final layer
        self.dropout = nn.Dropout(dropout * 1.5)
        
        self.fc = nn.Linear(embed_dim, num_classes)
        self.max_len = max_len

    def forward(self, x):
        # x: (batch, seq_len)
        embeddings = self.embedding(x) + self.pos_embedding[:, : x.size(1), :]
        embeddings = self.embedding_dropout(embeddings)
        x = self.transformer(embeddings)
        x = self.layer_norm(x.mean(dim=1))
        x = self.dropout(x)
        return self.fc(x)
