# Network Fragility Under Stress: Mechanism-Consistent Clearing, Predictive Graph Benchmarks, and Synthetic Double Machine Learning

**Research stage:** Reproducible computational methods portfolio; not a peer-reviewed article.  
**Domain:** Applied Economics, Computational Finance, Network Econometrics, Graph Machine Learning.  
**Sample status:** Artificially generated interbank liability networks and a separate independent-data known-truth statistical experiment.

## Abstract

**Background.** Bilateral claims among financial intermediaries transmit financial shocks, but methods that predict defaults do not automatically quantify effects of macroprudential policies. Financial network models further present a methodological challenge: nodes within a network are dependent, while learned graph representations and post-treatment exposures can invalidate naive predictive validation and causal identification.

**Objective.** Develop an auditable, mechanism-consistent framework for testing simulated network repayment, comparing capital-allocation decision rules under identical shocks, estimating effects in a separate fully specified partially linear synthetic process, and benchmarking supervised default predictors on independent network draws.

**Methods.** We implement a greatest-fixed-point Eisenberg–Noe clearing solver with convergence assertions; a paired network Monte Carlo experiment comparing equal-budget allocations to the largest debtor and to a randomly selected node; percentile bootstrap uncertainty over independent network replicates; group-based train/test separation for Logistic Regression, Random Forests and Histogram Gradient-Boosted Trees; and five-fold cross-fitted orthogonal Double Machine Learning (DML) with a known structural coefficient in a separate IID experiment. Every random process has a recorded seed. Diagnostic outputs are published as tables, figures and a provenance JSON file.

**Contributions.** The research brings model-theoretic stress testing, ML data-splitting discipline, and causal-estimator validation under known synthetic truth into a single transparently executed portfolio. The code avoids previously unverified ROC-AUC values, empirical phase-transition claims and assertions that a GNN automatically provides causal inference.

**Scope.** Results concern specified simulation processes, **not real banks, regulatory interventions or U.S. healthcare systems**. A proposed Health Economics translation to hospital financing is described separately and requires actual data.

**Keywords:** Financial Contagion; Network Economics; Eisenberg–Noe; Double Machine Learning; Out-of-Network Validation; Econometric Simulation; Macroprudential Stress Testing; Health Economics Translation.

## 1. Research motivation and evidence gap

Interbank networks can absorb or amplify shocks; their structure and the magnitude of the shock jointly shape fragility (Acemoglu et al., 2015). Clearing models provide explicit repayment rules, while applied policy research seeks to estimate counterfactual outcomes and decide where limited resources should be deployed. Modern Graph Neural Networks (GNNs) can learn patterns in networks, but neither prediction accuracy nor neural embeddings automatically identify the effect of an intervention.

The research gap addressed here is **not the invention of network contagion theory, DML or graph attention**. Those methods are established. The practical gap is transparent and code-verifiable integration of distinct methodological tasks: a stable clearing calculation, paired simulation-based interventions, independent-network prediction evaluation, and a known-truth DML design with correctly formulated cross-fitting.

A broad literature review would still be needed before claiming the integration is novel in the peer-reviewed literature. The current deliverable is a research-engineering demonstration suitable for rigorous technical inspection.

### Research questions

**RQ1.** How do alternative capital allocation heuristics affect outstanding obligations and defaults under a *fixed simulated* financial network and shock?

**RQ2.** Do bank-level default classifiers generalize to independently simulated networks rather than merely previously observed nodes?

**RQ3.** Does five-fold cross-fitted orthogonal residualization recover a known treatment parameter more faithfully than naive regression under a prespecified IID synthetic confounding process?

**RQ4 (future).** Can a source-verified provider financing network be linked with valid healthcare access indicators and a credible policy shock? This is proposed, **not tested here**.

## 2. Theory: clearing and counterfactual policies

### 2.1 Directed obligations and the repayment operator

Let \(L_{ij}\) denote the nonnegative payment owed by institution \(i\) to institution \(j\). Total promised payments are \(\bar p_i=\sum_jL_{ij}\), and \(\Pi_{ij}=L_{ij}/\bar p_i\) where \(\bar p_i>0\); rows with no liabilities are set to zero. For nonnegative external assets \(e_i\), the payment vector satisfies:

$$
p^*=\min\{\bar p,\ \max[0,e+\Pi^\top p^*]\},
$$

where min and max apply elementwise. Starting at \(p^{(0)}=\bar p\), iteration toward a fixed point yields a **greatest clearing solution** for the modeled operator. Unique clearing depends on additional regularity conditions; the code checks convergence numerically instead of claiming general uniqueness (Eisenberg & Noe, 2001).

Compute the aggregate unpaid obligation:

$$
\mathcal{L}(L,e)=\sum_{i=1}^N(\bar p_i-p^*_i).
$$

This objective reflects unpaid simulated financial liabilities, not welfare, GDP, hospital output or household wellbeing. Bankruptcy liquidation costs and common-market fire sales are not modeled.

### 2.2 Synthetic network generator

Directed edge membership is independently drawn as \(M_{ij}\sim\mathrm{Bernoulli}(\rho)\) for \(i\ne j\). Positive obligations are Gamma distributed conditional on an edge; baseline external assets are a function of total obligations and an independent positive asset component. Institution-specific exposures modulate a common stress amplitude.

These choices are **arbitrary simulator parameters** and are not calibrated to real U.S. bank leverage, bilateral payment networks, FDIC failures, emergency lending or actual dollar units. Sensitivity checks should vary density, shock size and asset distributions.

### 2.3 Paired modeled interventions

For network replicate \(b\), let \(\mathcal{L}_b(0)\) denote loss after an identical prescribed shock and no injection; let \(\mathcal{L}_b(T)\) denote loss when a fixed budget is injected into the largest debtor by pre-shock promised obligations; let \(\mathcal{L}_b(R)\) denote loss with equal capital allocated to a random debtor.

Define the paired contrast:

$$
\Delta_b =
[\mathcal{L}_b(0)-\mathcal{L}_b(T)]
-
[\mathcal{L}_b(0)-\mathcal{L}_b(R)]
=\mathcal{L}_b(R)-\mathcal{L}_b(T).
$$

The mean \(\bar\Delta=B^{-1}\sum_b\Delta_b\) is a **simulator-conditioned policy-rule comparison**, not an observed treatment effect. We retain the sign whether it supports, rejects or is inconclusive for the targeted heuristic. Resample *entire independent networks* for a percentile bootstrap interval. Resampling individual nodes as IID observations would ignore network interference.

Monotonicity is expected for added external financial resources under the specified fixed clearing operator. The code asserts that neither equal-budget injection produces a greater modeled aggregate shortfall than the baseline.

## 3. Predictive learning and out-of-network validation

For each independently generated network, simulated node covariates include promised obligations, incoming receivables, before/after-shock external assets, funding buffers, link counts, shock vulnerability, graph density and prescribed shock strength. Defaults are created by solving the clearing model, **not by fabricating prediction labels**.

All nodes within one network share obligations and the global shock. To prevent optimistic estimates from sharing network information across the sample boundary, `GroupShuffleSplit` assigns whole networks to train or test. Baselines are regularized Logistic Regression, Random Forests and Histogram Gradient-Boosted Trees. The analysis exports ROC-AUC, Average Precision (PR-AUC), Brier score, negative log likelihood, and calibration bins.

Permutation reliance on the held-out set describes predictive feature dependence, subject to feature correlation. It does not show causality or identify a regulatory lever. Because the same holdout selects the “best” model and is used to assess its calibration, model-selection optimism remains; a three-way train/validation/test design is a future improvement.

**Graph caveat:** The original repo contains a GAT–GRU prototype, but **the default predictive benchmark does not train or validate that architecture**. Its scores must not be attributed to a GNN. The architecture can be evaluated on genuinely held-out independent graph draws in a separate experiment after implementing a comparable preprocessing and splitting protocol.

## 4. Econometric validation: known-truth Double Machine Learning

A separate data-generating process constructs independent observations with:

$$
D_i=m_0(X_i)+V_i,\qquad
Y_i=\theta_0D_i+g_0(X_i)+U_i,
$$

and known synthetic \(\theta_0=-1.5\). Both \(m_0\) and \(g_0\) include nonlinear covariate functions. Naive unadjusted OLS on \(Y\) and \(D\) is expected to suffer omitted-variable confounding in this design.

Correctly defined nuisance regression targets are:

$$
\ell_0(X_i)=E[Y_i\mid X_i],\qquad m_0(X_i)=E[D_i\mid X_i].
$$

Using out-of-fold estimates, solve:

$$
\hat\theta =
\frac{\sum_i (D_i-\hat m_{-k(i)}(X_i))
                (Y_i-\hat\ell_{-k(i)}(X_i))}
{\sum_i (D_i-\hat m_{-k(i)}(X_i))^2}.
$$

The numerator and denominator arise from a Neyman-orthogonal moment (Chernozhukov et al., 2018). A first-order IID influence-function approximation gives:

$$
\widehat{\mathrm{SE}}(\hat\theta)
=\sqrt{\frac{n^{-1}\sum_i\hat\psi_i^2}
             {n\{\mathbb E_n[(D_i-\hat m_i)^2]\}^2}},
$$

with \(\hat\psi_i=(D_i-\hat m_i)
[(Y_i-\hat\ell_i)-\hat\theta(D_i-\hat m_i)]\). The simulation records coefficient bias and nominal interval coverage over independent replications.

**Critical separation:** This known-truth DML exercise is not DML applied to the financial clearing network. Its IID observations lack network interference; nominal standard errors and orthogonality theory do **not** establish identification in macroprudential intervention data.

## 5. Research outputs and interpretation

The pipeline writes:
- An independent-network policy table, paired network-bootstrap CI and two network-clearing figures.
- A network-grouped predictor comparison, a held-out calibration table, feature reliance and two diagnostic figures.
- Replicated known-truth DML results and a comparison figure.
- A directed network schematic generated from the same financial clearing model.
- A code-generated research summary and explicit provenance/limitations metadata.

The exact numerical results must be read from [the generated research report](../reports/REPRODUCIBLE_RESULTS.md), not inserted from unsupported hand-coded benchmark values. Running the public pipeline and continuous integration is the operational verification step.

### Threats to internal and external validity

**Simulation calibration:** Parameter choices govern results; repeated simulations describe the selected DGP and not an external population.

**Graph structure:** Directed independent edge probabilities ignore common ownership, payment-system hierarchy, market-making, and endogenous link formation.

**Mechanism mismatch:** Standard Eisenberg–Noe does not include endogenous asset prices, fire sales, bankruptcy costs, liquidity hoarding or actual supervision rules.

**Machine learning:** Test results are synthetic; held-out **networks** improve evaluation validity within the simulator but do not test real-world generalization. A separate untouched dataset is needed for model selection without test contamination.

**Causal claims:** The bank policy simulation is a deterministic mechanism given injections and exogenous assets, not empirically identified treatment assignment; separately, DML identification applies only to the independent specified PLR DGP.

**Inference:** Bootstrap over independently simulated networks is appropriate for variation across generated scenarios, but not for clustered real banking data unless the real sampling process supports it. Finite Monte Carlo replications limit coverage conclusions.

## 6. Health Economics extension: a pre-registered roadmap

The economic mechanics can motivate a **new** Health Economics study about reimbursement delays, hospital financing distress and interruption of services. Such a project needs actual facility-level balance sheets, accurately measured provider/insurer interdependence, actual payer or regulatory shocks, and pre-specified service-access outcomes. Merely renaming “banks” as “hospitals” produces no evidence.

A future hospital–year panel might include facility and year fixed effects and then separate network-robust predictive models from any credible quasi-experimental identification. See [the extension plan](HEALTH_ECONOMICS_TRANSLATION.md) for possible federal administrative sources and limitations.

## 7. Conclusion

A scientifically credible computational portfolio requires methodological *separation*. Mechanism-consistent clearing permits numerical intervention thought experiments; out-of-network prediction tests default classification within a simulated DGP; cross-fitting provides an opportunity to test an estimator's recovery of synthetic truth under stated identification assumptions. **None is an observed policy impact.**

The project's value is **auditable integration, falsifiable software checks, and transparent uncertainty**, not previously unsupported model performance claims or novel theorems. Real financial networks and healthcare applications require the necessary source data and identification design before stronger conclusions can be justified.

## References (APA 7th edition)

Acemoglu, D., Ozdaglar, A., & Tahbaz-Salehi, A. (2015). Systemic risk and stability in financial networks. *American Economic Review, 105*(2), 564–608. https://doi.org/10.1257/aer.20130456

Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal, 21*(1), C1–C68. https://doi.org/10.1111/ectj.12097

Eisenberg, L., & Noe, T. H. (2001). Systemic risk in financial systems. *Management Science, 47*(2), 236–249. https://doi.org/10.1287/mnsc.47.2.236.9835

Glasserman, P., & Young, H. P. (2015). How likely is contagion in financial networks? *Journal of Banking & Finance, 50*, 383–399. https://doi.org/10.1016/j.jbankfin.2014.02.006

Glasserman, P., & Young, H. P. (2016). Contagion in financial networks. *Journal of Economic Literature, 54*(3), 779–831. https://doi.org/10.1257/jel.20151228

Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph attention networks. *International Conference on Learning Representations*. https://openreview.net/forum?id=rJXMpikCZ
