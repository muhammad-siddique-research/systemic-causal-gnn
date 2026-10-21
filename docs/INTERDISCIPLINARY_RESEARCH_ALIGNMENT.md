# Interdisciplinary Research Portfolio: Network Economics, Health Economics, and AI

## Research positioning

**Primary project:** *Network Fragility and Macroprudential Resilience: A Mechanism-Consistent Framework Integrating Financial Clearing, Graph Neural Networks, and Cross-Fitted Econometrics.*

This repository is primarily a **Computational Economics and Systemic Financial Risk** research portfolio. It implements Eisenberg–Noe clearing for deliberately simulated interbank networks, paired equal-budget capital-injection comparisons, synthetic orthogonal DML validation, graph neural network benchmarks, and leakage-aware out-of-network prediction. Its output does not identify a treatment effect on real banks.

### Connection with Healthcare Economics

The common intellectual theme is **resource-allocation integrity under uncertainty**. Bank capital-injection design and healthcare capacity/resilience programs both involve decisions across heterogeneous, connected institutions. However, substituting hospital labels for bank nodes would not produce empirical Health Economics evidence. Meaningful translation requires observed hospital financial panel data and economically interpreted provider referral/payment dependencies.

**Existing complementary independent repositories:**

- [The Price of Care: Real-Data Health Economics, Count Econometrics, and Predictive Machine Learning](https://github.com/muhammad-siddique-research/price-of-care-health-economics). Uses genuine *historical* RAND HIE observations but does not recover randomized participant-level treatment effects.
- [Medicare Claims Integrity Lab: Python Synthetic-Claims Research Engineering](https://github.com/muhammad-siddique-research/medicare-claims-python-research). Uses a CMS synthetic claims schema and invented test data; no real-patient conclusions.

Together, these independent projects demonstrate complementary strengths without pretending that their samples, outcomes or identification assumptions coincide.

## Defensible contribution statements

| Skill | Demonstrated technique | Evidentiary scope |
|---|---|---|
| Mechanism-based simulation | Clearing fixed points; monotonicity; paired equal-capital allocation | Synthetic networks only |
| Network uncertainty | Whole-network percentile bootstrap, shock × density stress surface | Simulated DGP, arbitrary financial units |
| Causal-method diagnostics | Five-fold cross-fitted orthogonal Double Machine Learning | Separate synthetic IID partially linear DGP, known truth |
| Graph AI | GAT–GRU with held-out independent synthetic graphs | Supervised risk prediction; NOT identified causation |
| Empirical time series | Real FRED source mapping, Newey–West projection, out-of-time predictive tests | Genuine source downloads pending execution; no real network edges |
| Methodological translation | Hospital-year fixed effects, spillover exposure mapping, prospective validation architecture | Research proposal pending restricted/validated provider records |

## Research questions for an eventual substantive paper

1. Do network density and externally specified shock severity modify the distribution of **simulated** unpaid-obligation reductions under equal-budget intervention rules?
2. How much does supervised default-risk forecasting degrade when all nodes of held-out networks are unseen during training?
3. Under what modeled conditions does cross-fitting reduce nuisance-estimation bias, and does finite-sample coverage approach nominal levels as the number of independent replications increases?
4. Using actual aggregate U.S. credit-spread data, do additional macro indicators improve one-month-ahead forecast accuracy over a persistence benchmark after time-series uncertainty is acknowledged?
5. With validated hospital panel data, how would provider financial distress predictions affect the probability of service loss, and could network features improve prediction beyond a hospital fixed-effects baseline? **This fifth question has not been tested.**

## Scholarly limitations

The interbank model does not represent bankruptcy costs, fire sales, confidential supervisory exposure edges, actual liquidity policies or hospital-to-insurer payments. Simulation comparison is mechanism-based, not a natural experiment. Existing DML evidence is based on a separate IID generator; network spillovers violate its independence assumptions. The proposed FRED extension analyzes macroeconomic time series and cannot stand in for institution-level transmission evidence.

## Publication-development strategy

A serious macroprudential paper needs observed interbank exposure or a demonstrably defensible proxy, externally credible scenario calibration, robust theory and econometric identification. A Health Economics manuscript requires real hospital/provider panel outcomes, matched facility identities, validated financial predictors, and temporal/spatial external validation. The current repository makes those **research gaps transparent** and supplies modular code to investigate them once data become available.

## References (APA 7)

Acemoglu, D., Ozdaglar, A., & Tahbaz-Salehi, A. (2015). Systemic risk and stability in financial networks. *American Economic Review, 105*(2), 564–608. https://doi.org/10.1257/aer.20130456

Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal, 21*(1), C1–C68. https://doi.org/10.1111/ectj.12097

Eisenberg, L., & Noe, T. H. (2001). Systemic risk in financial systems. *Management Science, 47*(2), 236–249. https://doi.org/10.1287/mnsc.47.2.236.9835

Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica, 55*(3), 703–708. https://doi.org/10.2307/1913610