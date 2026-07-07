import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalLocalContextEnhancement(nn.Module):
    """TLCE: temporal local-context enhancement for key/value streams."""

    def __init__(self, d_model=50, kernel_size=3, groups=5):
        super().__init__()
        group_count = math.gcd(d_model, groups)
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.groups = group_count

        self.key_context = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2, groups=group_count, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
        )
        self.value_projection = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_model),
        )

        reduction_channels = max(1, d_model // 4)
        self.temporal_reweighting = nn.Sequential(
            nn.Conv1d(2 * d_model, reduction_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(reduction_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(reduction_channels, d_model, kernel_size=1),
        )

    def forward(self, tokens):
        tokens_by_channel = tokens.transpose(1, 2)
        local_keys = self.key_context(tokens_by_channel)
        projected_values = self.value_projection(tokens_by_channel)
        temporal_weights = self.temporal_reweighting(torch.cat([local_keys, tokens_by_channel], dim=1))
        temporal_weights = torch.softmax(temporal_weights, dim=-1)
        enhanced_tokens = local_keys + temporal_weights * projected_values
        return enhanced_tokens.transpose(1, 2)
