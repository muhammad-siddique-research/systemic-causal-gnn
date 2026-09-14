import torch
import torch.nn as nn
import numpy as np
from models.spatio_temporal_causal_gnn import SpatioTemporalCausalGNN
from sklearn.metrics import roc_auc_score, average_precision_score

def generate_synthetic_dynamic_macro_data(T: int = 8, N: int = 30, F_in: int = 6):
    """
    Generates dynamic financial time-series graph with exogenous policy interventions.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Dynamic node attributes: Leverage, Liquidity, Interbank Asset Size, Volatility
    x_seq = torch.randn(T, N, F_in)
    
    # 2. Dynamic interbank exposures with self-loops
    adj_seq = torch.zeros(T, N, N)
    for t in range(T):
        rand_adj = (torch.rand(N, N) > 0.70).float()
        adj_seq[t] = rand_adj + torch.eye(N)  # Add self-loops

    # 3. Ground truth policy intervention D and target default Y
    structural_confounder = x_seq[-1, :, 0] * 0.8 + x_seq[-1, :, 1] * 0.5
    treatment_d = structural_confounder + torch.randn(N) * 0.3
    
    # True causal parameter: theta = -1.65 (Liquidity injections reduce default probability)
    true_theta = -1.65
    latent_y = 1.2 * structural_confounder + true_theta * treatment_d + torch.randn(N) * 0.4
    target_y = (torch.sigmoid(latent_y) > 0.45).float().unsqueeze(1)
    treatment_d = treatment_d.unsqueeze(1)

    return x_seq, adj_seq, treatment_d, target_y, true_theta

def train_and_evaluate():
    T, N, F_in = 8, 50, 6
    x_seq, adj_seq, d, y, true_theta = generate_synthetic_dynamic_macro_data(T=T, N=N, F_in=F_in)

    model = SpatioTemporalCausalGNN(in_features=F_in, hidden_dim=32, num_layers=2, dropout=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()

    print("=" * 65)
    print(" 🌐 Spatio-Temporal Causal GNN: Empirical Training Pipeline")
    print(f" Structure: {N} Nodes | {T} Temporal Horizons | {F_in} Node Covariates")
    print("=" * 65)

    model.train()
    for epoch in range(1, 151):
        optimizer.zero_grad()
        pred_risk, pred_d, embeddings = model(x_seq, adj_seq)

        loss_risk = bce_loss(pred_risk, y)
        loss_treatment = mse_loss(pred_d, d)
        total_loss = loss_risk + 0.5 * loss_treatment

        total_loss.backward()
        optimizer.step()

        if epoch % 30 == 0:
            auc = roc_auc_score(y.detach().numpy(), pred_risk.detach().numpy())
            pr_auc = average_precision_score(y.detach().numpy(), pred_risk.detach().numpy())
            print(f"Epoch {epoch:03d} | Loss: {total_loss.item():.4f} | Risk AUC: {auc:.4f} | PR-AUC: {pr_auc:.4f}")

    # Neyman-Orthogonal Causal Effect Extraction
    model.eval()
    with torch.no_grad():
        pred_risk, pred_d, _ = model(x_seq, adj_seq)
        
        # Residualization
        v_res = (d - pred_d).numpy().flatten()
        u_res = (y - pred_risk).numpy().flatten()
        d_np = d.numpy().flatten()

        estimated_theta = np.dot(v_res, u_res) / np.dot(v_res, d_np)

    print("-" * 65)
    print(f"True Causal Policy Effect (Theta)       : {true_theta:.4f}")
    print(f"Orthogonalized Identified Effect        : {estimated_theta:.4f}")
    print(f"Absolute Policy Estimation Bias         : {abs(estimated_theta - true_theta):.4f}")
    print("=" * 65)

if __name__ == "__main__":
    train_and_evaluate()
