# Data Provenance, Identification Boundaries and Validation Protocol

## Evidence ledger: what is real, what is simulated?

| Module | Source | Unit | Estimated quantity | Can infer real-world impacts? |
|---|---|---|---|---|
| \`research/core.py\` | Algorithmically generated directed liabilities and external assets | Independently drawn financial network, with nodes | Greatest clearing fixed point and unpaid obligations | **No** |
| Paired injection contrasts | Same fabricated network and identical shock | Network pair | Difference in modeled losses after equal-budget capital | **No**, mechanistic only |
| \`research/prediction.py\` | Defaults derived from simulated clearing | Bank nested within independent network draw | Held-out default classification within the simulator | **No** |
| \`research/dml.py\` | Separate synthetic partially linear DGP, known true coefficient | Independent synthetic record | Cross-fitted DML bias and CI coverage conditional on IID DGP | **No** |
| Original \`train_st_gnn.py\` | Synthetic GAT/GRU node sequence | Synthetic network node | Exploratory in-sample surrogate | No verified independent generalization |
| \`app/streamlit_app.py\` | Runs \`clear_network\` on seeded synthetic liabilities | Current synthetic network | Scenario-specific modeled shortfalls | **No** |

There is **no observed interbank bilateral obligation dataset** in this repository. Historical financial events, actual default ROC-AUC, real-policy treatment effects, empirical contagion thresholds, and healthcare-sector intervention claims are **not established**.

## Financial network source and simulator

For each network scenario, generate a directed Bernoulli adjacency mask \(M_{ij}\) with \(M_{ii}=0\). If a link exists, liability \(L_{ij}\) is a Gamma random draw; choose nonnegative baseline external assets with a positive component associated with each node's obligations plus an independent Gamma component. Each synthetic node receives a shock-vulnerability parameter. All magnitudes are in **arbitrary units**, not U.S. dollars.

The observed quantity within the model is the result of iterative, greatest-fixed-point Eisenberg–Noe clearing. The original article proves existence and uniqueness only under appropriate network regularity. The implementation asserts convergence, validates nonnegative inputs, and does not report a fictitious uniqueness theorem.

**Known abstraction limits:** bankruptcy costs, mark-to-market fire sales, regulatory capital risk-weighting, common creditors, real liquidity markets, endogenous price formation, and agent behavior are not included in this reference clearing model. No phase-transition theorem is demonstrated.

## Cross-fitted causal-estimator exercise

The DML Monte Carlo experiment generates \((X_i,D_i,Y_i)\) with a known parameter in a *synthetic IID* partially linear model. A nonlinear propensity \(m_0(X)\) confounds the continuous treatment, so naive unadjusted OLS can be biased. Five-fold honest predictions from training folds provide residualized exposures and outcomes, followed by estimation with an orthogonal score.

DML uses \(E[Y\mid X]\), not \(g_0(X)\), for the outcome nuisance estimate. Standard errors are an IID influence-function approximation. Network-based interference, shared systemic shocks, and real policy endogeneity are **not addressed**, so these results cannot be transferred to observed bank-level monetary policy without additional identification structure.

## Paired policy comparison

Within scenario \(b\), the liabilities, asset-shock exposure, injection budget, and initial resources are shared. One injects the budget into the institution with the largest **pre-shock** promised obligations; another into a uniformly chosen institution. Random comparison is random *within the model*, but not a real policy experiment. The bootstrap samples entire independent networks, not individual bank nodes.

Report the sign and 95% interval of the paired difference whether positive, negative, or indistinguishable from zero. There is no assumption that the largest debtor must be the optimal policy target.

## Predictive benchmark and leakage

Each synthetic network is one sampling cluster. A random node split would leak connected observations and correlated shocks. We use \`GroupShuffleSplit\` by **network_id** so no node from one fabricated network appears in both sets.

Models include regularized logistic regression, random forest and histogram gradient-boosted trees. Report ROC-AUC, average precision, Brier score, log loss, and calibration. Permutation importance describes predictive model reliance; it cannot identify causal drivers of default. Selecting the best-performing method on the same test split used to report performance introduces selection optimism. A separate development set and untouched test set would strengthen research credibility.

The existing \`SpatioTemporalCausalGNN\` architecture is a GAT + GRU with a default head and a treatment-regression head, but those heads are *not* a causal identification theorem. Do not interpret predictions from an in-sample network as externally validated default forecasts.

## Why a genuine dataset is harder than a public macroeconomic time series

Federal Reserve financial-condition indexes and public call reports can offer genuine **aggregate financial stress or institution-level accounting series**. They do **not** reveal the exact bilateral obligations \(L_{ij}\) needed to reconstruct this clearing network. Fabricating an adjacency graph from correlations and labeling it an observed liabilities network would invalidate the research design.

Potential empirical extensions must start from a source-aware measurement map: verify date and institution coverage, disclose accounting units, characterize link measurement error, identify the economic treatment and compare against independent external outcomes. Without observed bilateral obligations, the responsible deliverable remains a **simulator validated against its own known properties**.

## Robustness and reproducibility checklist

1. Report parameter values, random seeds and the simulated DGP in every run.
2. Assert clearing residual tolerance; require convergence and explain possible nonuniqueness.
3. Verify monotonic response to added external resources under the specified fixed-point solver.
4. Use paired network scenarios and independent-network bootstrap—not naive node resampling.
5. Split machine-learning train/test by network; avoid training-on-test graph leakage.
6. Separate synthetic DML known truth from real macroprudential identification.
7. Explicitly label stylized legacy assets and any code-generated plot.
8. Do not quote benchmark numbers as empirical without exact scripts and verified workflow.
9. Apply actual hospital/insurer relationship data before any Health Economics inference.
