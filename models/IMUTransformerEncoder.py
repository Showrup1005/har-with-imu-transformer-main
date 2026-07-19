"""
IMUTransformerEncoder model
"""

import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class IMUTransformerEncoder(nn.Module):

    def __init__(self, config):
        """
        config: (dict) configuration of the model
        """
        super().__init__()

        self.transformer_dim = config.get("transformer_dim")

        self.input_proj = nn.Sequential(
            nn.Conv1d(config.get("input_dim"), self.transformer_dim, 1), nn.GELU(),
            nn.Conv1d(self.transformer_dim, self.transformer_dim, 1), nn.GELU(),
            nn.Conv1d(self.transformer_dim, self.transformer_dim, 1), nn.GELU(),
            nn.Conv1d(self.transformer_dim, self.transformer_dim, 1), nn.GELU()
        )

        self.window_size = config.get("window_size")
        self.encode_position = config.get("encode_position")
        encoder_layer = TransformerEncoderLayer(
            d_model=self.transformer_dim,
            nhead=config.get("nhead"),
            dim_feedforward=config.get("dim_feedforward"),
            dropout=config.get("transformer_dropout"),
            activation=config.get("transformer_activation")
        )

        self.transformer_encoder = TransformerEncoder(
            encoder_layer,
            num_layers=config.get("num_encoder_layers"),
            norm=nn.LayerNorm(self.transformer_dim)
        )
        self.cls_token = nn.Parameter(torch.zeros((1, self.transformer_dim)), requires_grad=True)

        if self.encode_position:
            self.position_embed = nn.Parameter(torch.randn(self.window_size + 1, 1, self.transformer_dim))

        num_classes = config.get("num_classes")
        self.imu_head = nn.Sequential(
            nn.LayerNorm(self.transformer_dim),
            nn.Linear(self.transformer_dim, self.transformer_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.transformer_dim // 4, num_classes)
        )
        self.log_softmax = nn.LogSoftmax(dim=1)

        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_features(self, data):
        """Extract features before classification head (CLS token)"""
        src = data.get('imu')  # N x S x C
        src = self.input_proj(src.transpose(1, 2)).permute(2, 0, 1)

        # Prepend class token
        cls_token = self.cls_token.unsqueeze(1).repeat(1, src.shape[1], 1)
        src = torch.cat([cls_token, src])

        if self.encode_position:
            src += self.position_embed

        # Transformer Encoder pass
        transformer_output = self.transformer_encoder(src)
        features = transformer_output[0]  # CLS token features (batch_size x dim)

        return features

    def forward(self, data):
        src = data.get('imu')  # N x S x C
        src = self.input_proj(src.transpose(1, 2)).permute(2, 0, 1)

        cls_token = self.cls_token.unsqueeze(1).repeat(1, src.shape[1], 1)
        src = torch.cat([cls_token, src])

        if self.encode_position:
            src += self.position_embed

        target = self.transformer_encoder(src)[0]   # CLS token

        logits = self.imu_head(target)              # Raw logits
        return logits                                   # ← No log_softmax here if using CrossEntropyLoss


def get_activation(activation):
    if activation == "relu":
        return nn.ReLU(inplace=True)
    if activation == "gelu":
        return nn.GELU()
    raise RuntimeError("Activation {} not supported".format(activation))