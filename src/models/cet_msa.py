import torch
import torch.nn as nn
import torch.nn.functional as F

from .pe_mhsa import PyramidalEfficientMultiHeadSelfAttention
from .tlce import TemporalLocalContextEnhancement


class CETMSA(nn.Module):
    """CET-MSA block: TLCE followed by PE-MHSA."""

    def __init__(
        self,
        d_model,
        d_k,
        d_v,
        heads,
        dropout=0.1,
        reduction_ratio=2,
        tlce_kernel_size=3,
        tlce_groups=5,
        apply_logit_smoothing=True,
    ):
        super().__init__()
        self.temporal_local_context_enhancement = TemporalLocalContextEnhancement(
            d_model=d_model,
            kernel_size=tlce_kernel_size,
            groups=tlce_groups,
        )
        self.pyramidal_efficient_mhsa = PyramidalEfficientMultiHeadSelfAttention(
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            heads=heads,
            dropout=dropout,
            reduction_ratio=reduction_ratio,
            apply_logit_smoothing=apply_logit_smoothing,
        )

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        enhanced_keys = self.temporal_local_context_enhancement(keys)
        enhanced_values = self.temporal_local_context_enhancement(values)
        return self.pyramidal_efficient_mhsa(
            queries=queries,
            keys=enhanced_keys,
            values=enhanced_values,
            attention_mask=attention_mask,
            attention_weights=attention_weights,
        )


class ChannelWiseTemporalReweighting(nn.Module):
    """Light channel-wise reweighting used after feature fusion when enabled."""

    def __init__(self, input_dim, reduction=16):
        super().__init__()
        hidden_dim = max(1, input_dim // reduction)
        self.global_average_pool = nn.AdaptiveAvgPool1d(1)
        self.down_projection = nn.Linear(input_dim, hidden_dim, bias=False)
        self.up_projection = nn.Linear(hidden_dim, input_dim, bias=False)
        self.gate = nn.Sigmoid()

    def forward(self, inputs):
        channel_summary = inputs.transpose(1, 2)
        channel_summary = self.global_average_pool(channel_summary).squeeze(-1)
        channel_summary = F.relu(self.down_projection(channel_summary))
        channel_weights = self.gate(self.up_projection(channel_summary)).unsqueeze(1)
        return inputs * channel_weights


class CETMSARULPredictor(nn.Module):
    """RUL predictor described as LSTM -> TLCE -> PE-MHSA -> regression head."""

    def __init__(
        self,
        input_dim=17,
        hidden_dim=50,
        window_length=30,
        heads=8,
        dropout=0.2,
        tlce_groups=5,
        pyramidal_ratio=2,
        handcrafted_dim=34,
    ):
        super().__init__()
        self.window_length = window_length
        self.temporal_encoder = nn.LSTM(batch_first=True, input_size=input_dim, hidden_size=hidden_dim, num_layers=1)
        self.cet_msa = CETMSA(
            d_model=hidden_dim,
            d_k=hidden_dim,
            d_v=hidden_dim,
            heads=heads,
            dropout=dropout,
            reduction_ratio=pyramidal_ratio,
            tlce_kernel_size=3,
            tlce_groups=tlce_groups,
            apply_logit_smoothing=True,
        )

        self.sequence_projection = nn.Sequential(
            nn.Linear(in_features=window_length * hidden_dim, out_features=hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_dim, out_features=10),
            nn.ReLU(inplace=True),
        )

        self.degradation_feature_branch = nn.Sequential(
            nn.Linear(in_features=handcrafted_dim, out_features=10),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        self.rul_head = nn.Linear(in_features=20, out_features=1)
        self.channel_wise_temporal_reweighting = ChannelWiseTemporalReweighting(input_dim=20, reduction=16)

    def forward(self, inputs, handcrafted_degradation_features):
        degradation_features = self.degradation_feature_branch(handcrafted_degradation_features)
        temporal_tokens, _ = self.temporal_encoder(inputs)
        temporal_tokens = self.cet_msa(temporal_tokens, temporal_tokens, temporal_tokens)
        sequence_features = temporal_tokens.reshape(temporal_tokens.size(0), -1)
        sequence_features = self.sequence_projection(sequence_features)
        fused_features = torch.cat((sequence_features, degradation_features), dim=1)
        return self.rul_head(fused_features)
