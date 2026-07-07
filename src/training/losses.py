import torch


class TemporalSmoothnessRegularizer:
    """L2-SP style regularizer used during degradation-aware fine-tuning."""

    def __init__(self, model, regularization_weight=1e-5):
        self.regularization_weight = regularization_weight
        self.pretrained_parameters = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    def penalty(self, model):
        penalty = 0.0
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and name in self.pretrained_parameters:
                penalty = penalty + (parameter - self.pretrained_parameters[name]).pow(2).sum()
        return self.regularization_weight * penalty
