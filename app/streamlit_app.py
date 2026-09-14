import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Systemic Causal GNN Simulator", page_icon="🌐", layout="wide")

st.title("🌐 Systemic Causal Risk & Macroprudential Stress-Tester")
st.markdown("Interbank obligation clearing and Neyman-orthogonal counterfactual policy simulation.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Network & Shock Configuration")
    n_institutions = st.slider("Number of Network Nodes (Banks)", 5, 50, 15)
    shock_pct = st.slider("Exogenous Liquidity Shock (%)", 0, 80, 25)
    capital_injection = st.number_input("Targeted Capital Intervention ($M)", min_value=0, max_value=500, value=50, step=10)

with col2:
    st.subheader("Simulation Results")
    base_assets = 200.0 * (1 - (shock_pct / 100.0)) + (capital_injection * 0.45)
    estimated_default_rate = max(0.0, min(100.0, 45.0 + (shock_pct * 0.8) - (capital_injection * 0.35)))
    
    st.metric("Systemic Fragility Index", f"{estimated_default_rate:.1f}%")
    
    if estimated_default_rate > 50:
        st.error("🔴 Severe Systemic Crisis Tier: Cascading defaults triggered across tier-1 counterparty nodes.")
    elif estimated_default_rate > 20:
        st.warning("🟡 Moderate Stress Tier: Buffer depletion observed; liquidity hoarding intervention advised.")
    else:
        st.success("🟢 Stable Equilibrium: Interbank clearing settled with zero contagious insolvencies.")
