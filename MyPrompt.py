import torch
import torch.nn as nn


class MyPrompt(nn.Module):
    def __init__(self, input_dim):
        super(MyPrompt, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1, input_dim))
        self.trigger_weight = nn.Parameter(torch.Tensor(1, input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.trigger_weight)

    def forward(self, node_embeddings, is_attach_mask=None):
        if is_attach_mask is None:
            return node_embeddings * self.weight
        else:
            weight_expand = self.weight.expand_as(node_embeddings)
            trigger_expand = self.trigger_weight.expand_as(node_embeddings)
            return torch.where(is_attach_mask.unsqueeze(1), node_embeddings * trigger_expand,
                               node_embeddings * weight_expand)
