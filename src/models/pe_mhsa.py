import math

import torch
import torch.nn as nn
from torch.nn import init


class PyramidalEfficientMultiHeadSelfAttention(nn.Module):
    """PE-MHSA: pyramidal efficient temporal multi-head self-attention."""

    def __init__(
        self,
        d_model,
        d_k,
        d_v,
        heads,
        dropout=0.1,
        reduction_ratio=2,
        apply_logit_smoothing=True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.heads = heads
        self.reduction_ratio = reduction_ratio
        self.apply_logit_smoothing = apply_logit_smoothing and heads > 1

        self.query_projection = nn.Linear(d_model, heads * d_k)
        self.key_projection = nn.Linear(d_model, heads * d_k)
        self.value_projection = nn.Linear(d_model, heads * d_v)
        self.output_projection = nn.Linear(heads * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        if self.reduction_ratio > 1:
            self.pyramidal_reduction = nn.Conv1d(
                d_model,
                d_model,
                kernel_size=reduction_ratio + 1,
                stride=reduction_ratio,
                padding=reduction_ratio // 2,
                groups=d_model,
            )
            self.reduction_norm = nn.LayerNorm(d_model)

        if self.apply_logit_smoothing:
            self.logit_smoothing = nn.Sequential(
                nn.Conv2d(heads, heads, kernel_size=1),
                nn.Softmax(dim=-1),
                nn.InstanceNorm2d(heads),
            )

        self.initialize_pe_mhsa_parameters()

    def initialize_pe_mhsa_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                init.kaiming_normal_(module.weight, mode='fan_out')
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                init.normal_(module.weight, std=0.001)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def apply_pyramidal_key_value_reduction(self, tokens):
        if self.reduction_ratio <= 1:
            return tokens
        reduced = self.pyramidal_reduction(tokens.transpose(1, 2))
        reduced = reduced.transpose(1, 2).contiguous()
        return self.reduction_norm(reduced)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        batch_size, query_steps, _ = queries.shape
        reduced_keys = self.apply_pyramidal_key_value_reduction(keys)
        reduced_values = self.apply_pyramidal_key_value_reduction(values)

        q = self.query_projection(queries).view(batch_size, query_steps, self.heads, self.d_k).permute(0, 2, 1, 3)
        k = self.key_projection(reduced_keys).view(batch_size, -1, self.heads, self.d_k).permute(0, 2, 3, 1)
        v = self.value_projection(reduced_values).view(batch_size, -1, self.heads, self.d_v).permute(0, 2, 1, 3)

        attention_logits = torch.matmul(q, k) / math.sqrt(self.d_k)
        if attention_weights is not None:
            attention_logits = attention_logits * attention_weights
        if attention_mask is not None:
            attention_logits = attention_logits.masked_fill(attention_mask, -torch.inf)

        if self.apply_logit_smoothing:
            attention = self.logit_smoothing(attention_logits)
        else:
            attention = torch.softmax(attention_logits, dim=-1)
        attention = self.dropout(attention)

        attended_values = torch.matmul(attention, v).permute(0, 2, 1, 3).contiguous()
        attended_values = attended_values.view(batch_size, query_steps, self.heads * self.d_v)
        return self.output_projection(attended_values)
