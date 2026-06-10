import torch
import torch.nn as nn


class DualPrompt(nn.Module):
    """Input-level dual prompt generator (Paper Section 5.3, Eq. 15-16).

    P = [P_soft; P_hard] in R^{2 x d}, where d is the input feature dimension.
    For each node v:  x_tilde_v = x_v * (zeta * pi_v^T @ P)
    pi_v = [1,0] for soft nodes, [0,1] for hard nodes.
    """

    def __init__(self, feature_dim):
        super().__init__()
        self.P_soft = nn.Parameter(torch.ones(feature_dim))
        self.P_hard = nn.Parameter(torch.ones(feature_dim))

    def init_from_data(self, x, soft_mask, hard_mask):
        """Initialize prompts as mean feature vectors of their node groups."""
        with torch.no_grad():
            if soft_mask.any():
                self.P_soft.copy_(x[soft_mask].mean(dim=0))
            if hard_mask.any():
                self.P_hard.copy_(x[hard_mask].mean(dim=0))

    def forward(self, x, hard_mask, zeta=1.0):
        """Apply input-level prompt modulation.

        Args:
            x: node features [N, d]
            hard_mask: bool tensor [N], True for hard nodes
            zeta: amplification coefficient (>1 during verification)
        Returns:
            x_tilde: prompt-modulated features [N, d]
        """
        prompt = torch.where(
            hard_mask.unsqueeze(1),
            zeta * self.P_hard.unsqueeze(0),
            self.P_soft.unsqueeze(0),
        )
        return x * prompt


class MyPrompt(DualPrompt):
    """Backward-compatible alias."""
    pass
