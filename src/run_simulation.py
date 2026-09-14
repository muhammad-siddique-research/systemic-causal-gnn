import numpy as np
import pandas as pd
from clearing import compute_eisenberg_noe_clearing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

def run_causal_systemic_simulation(n_banks: int = 20):
    np.random.seed(42)
    
    # Generate synthetic interbank liability matrix (N x N)
    raw_liabilities = np.random.exponential(scale=50.0, size=(n_banks, n_banks))
    np.fill_diagonal(raw_liabilities, 0.0)
    
    # Baseline external assets
    external_assets = np.random.uniform(80.0, 300.0, size=n_banks)
    
    # Run baseline clearing
    p_star, defaults = compute_eisenberg_noe_clearing(raw_liabilities, external_assets)
    total_loss_baseline = np.sum(raw_liabilities.sum(axis=1) - p_star)
    
    print(f"--- Baseline Network Stress Simulation (N = {n_banks} Institutions) ---")
    print(f"Defaulting Institutions: {np.sum(defaults)} / {n_banks}")
    print(f"Total Systemic Shortfall: ${total_loss_baseline:.2f}M")
    
    # Simulate Double ML Causal Policy Evaluation
    n_samples = 300
    covariates = np.random.normal(0, 1, (n_samples, 5))
    policy_intervention = 0.5 * covariates[:, 0] + np.random.normal(0, 1, n_samples)
    systemic_loss = -1.85 * policy_intervention + 1.2 * covariates[:, 1] + np.random.normal(0, 0.5, n_samples)
    
    # Double ML Neyman-Orthogonal Estimation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    v_hat = np.zeros(n_samples)
    u_hat = np.zeros(n_samples)
    
    for train_idx, test_idx in cv.split(covariates):
        rf_d = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_y = RandomForestRegressor(n_estimators=100, random_state=42)
        
        rf_d.fit(covariates[train_idx], policy_intervention[train_idx])
        rf_y.fit(covariates[train_idx], systemic_loss[train_idx])
        
        v_hat[test_idx] = policy_intervention[test_idx] - rf_d.predict(covariates[test_idx])
        u_hat[test_idx] = systemic_loss[test_idx] - rf_y.predict(covariates[test_idx])
        
    theta_dml = np.dot(v_hat, u_hat) / np.dot(v_hat, policy_intervention)
    print("\n--- Double Machine Learning Policy Identification ---")
    print(f"True Causal Effect: -1.8500")
    print(f"Double ML Identified Effect: {theta_dml:.4f}")

if __name__ == "__main__":
    run_causal_systemic_simulation()
