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

        self.input_proj = nn.Sequential(nn.Conv1d(config.get("input_dim"), self.transformer_dim, 1), nn.GELU(),
                                        nn.Conv1d(self.transformer_dim, self.transformer_dim, 1), nn.GELU(),
                                        nn.Conv1d(self.transformer_dim, self.transformer_dim, 1), nn.GELU(),
                                        nn.Conv1d(self.transformer_dim, self.transformer_dim, 1), nn.GELU())

        self.window_size = config.get("window_size")
        self.encode_position = config.get("encode_position")
        encoder_layer = TransformerEncoderLayer(d_model = self.transformer_dim,
                                       nhead = config.get("nhead"),
                                       dim_feedforward = config.get("dim_feedforward"),
                                       dropout = config.get("transformer_dropout"),
                                       activation = config.get("transformer_activation"),
                                       batch_first=True, 
                                       norm_first=False)

        self.transformer_encoder = TransformerEncoder(encoder_layer,
                                              num_layers = config.get("num_encoder_layers"),
                                              norm = nn.LayerNorm(self.transformer_dim))
        self.cls_token = nn.Parameter(torch.zeros((1, self.transformer_dim)), requires_grad=True)
        

        if self.encode_position:
            self.position_embed = nn.Parameter(torch.randn(1, self.window_size + 1, self.transformer_dim))

        num_classes =  config.get("num_classes")
        self.imu_head = nn.Sequential(
            nn.LayerNorm(self.transformer_dim),
            nn.Linear(self.transformer_dim,  self.transformer_dim//4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.transformer_dim//4,  num_classes)
        )

        # init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, data):
        src = data.get('imu')                              # (N, S, C)
        src = self.input_proj(src.transpose(1, 2))          # (N, transformer_dim, S)
        src = src.transpose(1, 2)                           # (N, S, transformer_dim)  batch-first

        cls_token = self.cls_token.unsqueeze(0).expand(src.shape[0], -1, -1)  # (N, 1, C)
        src = torch.cat([cls_token, src], dim=1)            # (N, S+1, C)

        if self.encode_position:
            src = src + self.position_embed                 # (1, S+1, C) broadcasts over N

        target = self.transformer_encoder(src)               # (N, S+1, C)
        target = target[:, 0, :]                              # CLS token — batch dim0, seq dim1
        logits = self.imu_head(target)
        return logits

def get_activation(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU(inplace=True)
    if activation == "gelu":
        return nn.GELU()
    raise RuntimeError("Activation {} not supported".format(activation))
