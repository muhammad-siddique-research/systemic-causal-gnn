"""Simulation-driven interactive network stress tester. No invented dashboards."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
from src.research.core import clear_network, draw_network

st.set_page_config(page_title="Network Fragility Lab", page_icon="📊", layout="wide")
st.title("Network Fragility Lab")
st.caption(
    "Mechanism-consistent Eisenberg–Noe simulations on fabricated financial "
    "networks. This is not observed interbank data, a forecast, or an "
    "identified policy intervention."
)

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Model specification")
    nodes = st.slider("Simulated institutions", 8, 55, 22)
    density = st.slider("Directed edge probability", .05, .65, .23, step=.02)
    shock = st.slider("External-asset shock", .0, .85, .36, step=.01)
    budget_pct = st.slider(
        "Capital injection budget (% mean interbank obligations)", 0, 100, 14
    )
    seed = st.number_input("Reproducibility seed", 1, 1_000_000, 42)

rng = np.random.default_rng(int(seed))
liabilities, assets, vulnerability = draw_network(rng, nodes, density)
external = assets * (1 - shock * vulnerability)
owed = liabilities.sum(axis=1)
capital_budget = budget_pct / 100 * owed.mean()
highest_debtor = int(owed.argmax())
targeted = external.copy()
targeted[highest_debtor] += capital_budget
random_recipient = int(rng.integers(nodes))
random_policy = external.copy()
random_policy[random_recipient] += capital_budget

baseline_res = clear_network(liabilities, external)
target_res = clear_network(liabilities, targeted)
random_res = clear_network(liabilities, random_policy)
with col2:
    st.subheader("Calculated clearing equilibria")
    m1, m2 = st.columns(2)
    m1.metric("Baseline default share", f"{baseline_res.defaults.mean():.1%}")
    m2.metric("Baseline unpaid obligations", f"{baseline_res.shortfall:,.1f}")
    st.metric(
        "Shortfall reduction: largest debtor",
        f"{baseline_res.shortfall-target_res.shortfall:,.2f}",
    )
    st.metric(
        "Shortfall reduction: random debtor",
        f"{baseline_res.shortfall-random_res.shortfall:,.2f}",
    )
    st.caption(
        "Unpaid-obligation units are arbitrary. The added injection is the "
        "same in both policy scenarios. Different allocation rules can be "
        "compared as simulation mechanisms, not real policy effects."
    )

st.subheader("Payments and shortfalls by simulated institution")
view = pd.DataFrame({
    "Bank": [f"Simulated bank {i+1}" for i in range(nodes)],
    "Obligations": owed,
    "Baseline payment": baseline_res.payments,
    "Targeted payment": target_res.payments,
    "Default baseline": baseline_res.defaults,
})
st.dataframe(view, hide_index=True, use_container_width=True)
st.bar_chart(
    view.set_index("Bank")[["Obligations", "Baseline payment", "Targeted payment"]]
)
st.info(
    "The simulator uses synthetic balance sheets and a specified clearing "
    "fixed point. Graph Machine Learning results must be validated separately "
    "on unseen networks; causal Double ML experiments are an independent "
    "synthetic exercise."
)
