import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GraphAttentionLayer(nn.Module):
    """
    Dense Graph Attention Layer (GAT) operating on full adjacency matrices.
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1, alpha: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.W.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # h shape: (N, in_features), adj shape: (N, N)
        Wh = self.W(h)
        N = Wh.size(0)

        # Vectorized self-attention mechanism
        Wh1 = Wh.repeat(1, N).view(N * N, self.out_features)
        Wh2 = Wh.repeat(N, 1)
        all_combinations = torch.cat([Wh1, Wh2], dim=1)
        e = F.leaky_relu(torch.matmul(all_combinations, self.a).view(N, N), negative_slope=self.alpha)

        # Mask non-connected edges (-inf before softmax)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, Wh)
        return F.elu(h_prime)


class SpatioTemporalCausalGNN(nn.Module):
    """
    Spatio-Temporal Graph Neural Network with an integrated Causal Double ML head.
    """
    def __init__(self, in_features: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.spatial_gat = GraphAttentionLayer(in_features, hidden_dim, dropout=dropout)
        self.temporal_gru = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        
        # Supervised Risk Head: Predicts node-level default probability
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Causal Nuisance Head: Predicts continuous treatment propensity m(X)
        self.treatment_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x_seq: torch.Tensor, adj_seq: torch.Tensor):
        """
        Parameters:
            x_seq: (T, N, in_features) temporal sequence of balance sheet attributes.
            adj_seq: (T, N, N) temporal interbank liability networks.
        Returns:
            pred_risk: (N, 1) default probability vector.
            pred_treatment: (N, 1) predicted treatment intervention baseline m(X).
            embeddings: (N, hidden_dim) learned representations.
        """
        T, N, _ = x_seq.shape
        spatial_embeddings = []

        # 1. Spatial aggregation across time steps
        for t in range(T):
            h_t = self.spatial_gat(x_seq[t], adj_seq[t])  # (N, hidden_dim)
            spatial_embeddings.append(h_t.unsqueeze(0))

        # (T, N, hidden_dim) -> (N, T, hidden_dim)
        spatial_seq = torch.cat(spatial_embeddings, dim=0).permute(1, 0, 2)

        # 2. Temporal sequential modeling
        gru_out, _ = self.temporal_gru(spatial_seq)  # (N, T, hidden_dim)
        final_repr = gru_out[:, -1, :]  # Take terminal hidden state (N, hidden_dim)

        # 3. Multi-task output heads
        pred_risk = self.risk_head(final_repr)
        pred_treatment = self.treatment_head(final_repr)

        return pred_risk, pred_treatment, final_repr
