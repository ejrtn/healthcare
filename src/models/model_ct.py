import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.optim import AdamW

class CTConvNeXtModel(nn.Module):
    """
    2.5D CT image processing model using ConvNeXt-Tiny as backbone,
    Transformer Encoder for slice-wise context, and Attention Pooling.
    Includes Gated Heads for organ-specific injury detection.
    """
    def __init__(self, model_name='convnext_tiny', num_slices=64, dim=768):
        super().__init__()
        # Backbone: Feature Extractor (ConvNeXt)
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        
        # Initially freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.dim = dim # tiny: 768
        self.num_slices = num_slices
        self.gated_norm = nn.LayerNorm(self.dim)

        # Position Encoding (Spatial index for 64 slices)
        self.position_embedding = nn.Parameter(torch.zeros(1, num_slices, self.dim))
        self.position_dropout = nn.Dropout(0.1)

        # Transformer Encoder: Slice-to-slice interaction
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dim,
            nhead=8,
            dim_feedforward=self.dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Attention Pooling: Weighing suspicious slices
        self.attention_net = nn.Sequential(
            nn.Linear(self.dim, 256),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

        # Suspicion Head: Global injury detection
        self.suspicion_head = nn.Sequential(
            nn.Linear(self.dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)
        )

        # Organ Heads: Specific organ injury classification
        self.organ_heads = nn.ModuleDict({
            'bowel': nn.Linear(self.dim, 2),
            'extravasation': nn.Linear(self.dim, 2),
            'liver': nn.Linear(self.dim, 3),
            'kidney': nn.Linear(self.dim, 3),
            'spleen': nn.Linear(self.dim, 3)
        })

    def forward(self, x):
        # x shape: (Batch, Slices, C, H, W)
        b, s, c, h, w = x.shape
        
        # Memory-efficient chunk processing for backbone
        chunk_size = 8 
        all_features = []
        for i in range(0, s, chunk_size):
            x_chunk = x[:, i : i + chunk_size]
            x_chunk = x_chunk.reshape(-1, c, h, w)
            feat_chunk = self.backbone(x_chunk)
            feat_chunk = feat_chunk.view(b, -1, self.dim)
            all_features.append(feat_chunk)

        features = torch.cat(all_features, dim=1) # (Batch, num_slices, dim)

        # Position Encoding
        features = features + self.position_embedding
        features = self.position_dropout(features)

        # Transformer Encoder
        features = self.transformer_encoder(features)

        # Attention Pooling
        att_scores = self.attention_net(features)
        att_weights = F.softmax(att_scores, dim=1)
        combined = torch.sum(features * att_weights, dim=1) # (Batch, dim)

        # Global Injury Prediction
        injury_logits = self.suspicion_head(combined)
        injury_prob = torch.softmax(injury_logits, dim=1)[:, 1:2]

        # Gated Multi-head classification
        # The organ heads are conditioned on the global injury probability
        gated_features = self.gated_norm(combined * injury_prob)
        
        out = {k: head(gated_features) for k, head in self.organ_heads.items()}
        out['any_injury'] = injury_logits

        return out

def get_optimizer_with_llrd(model, base_lr=1e-4, weight_decay=0.05, layer_decay=0.8):
    """
    Layer-wise Learning Rate Decay (LLRD) for ConvNeXt Backbone.
    """
    raw_model = model.module if hasattr(model, 'module') else model
    param_groups = []
    
    # 1. Head Group (Full Learning Rate)
    head_modules = [raw_model.transformer_encoder, raw_model.attention_net, 
                    raw_model.suspicion_head, raw_model.organ_heads, raw_model.gated_norm]
    head_params = []
    for m in head_modules:
        head_params.extend([p for p in m.parameters() if p.requires_grad])
    if raw_model.position_embedding.requires_grad:
        head_params.append(raw_model.position_embedding)

    param_groups.append({"params": head_params, "lr": base_lr, "weight_decay": weight_decay, "name": "head"})

    # 2. Backbone Stages (Decayed Learning Rate)
    backbone = raw_model.backbone
    stages = [backbone.stem, backbone.stages[0], backbone.stages[1], backbone.stages[2], backbone.stages[3]]
    stages.reverse() 

    for i, stage in enumerate(stages):
        ratio = layer_decay ** (i + 1)
        stage_params = [p for p in stage.parameters() if p.requires_grad]
        if len(stage_params) > 0:
            param_groups.append({
                "params": stage_params, 
                "lr": base_lr * ratio, 
                "weight_decay": weight_decay, 
                "name": f"backbone_layer_{4-i}"
            })

    return AdamW(param_groups)
