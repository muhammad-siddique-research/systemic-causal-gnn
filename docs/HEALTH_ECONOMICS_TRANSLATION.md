# Health Economics Research Translation — Proposed, Not Tested

## Project concept

**Hospital Financial Resilience: Network Exposure, Reimbursement Shocks, and Explainable Risk Prediction in U.S. Healthcare Markets**

An extension of network Economic theory into Health Economics would study whether **financial stress transmitted through payer reimbursements, provider dependencies, and hospital referral networks** creates nonlinear constraints on access to care.

The existing repository does **not** contain real provider contracts, patient referral data, insurer reimbursement networks, or hospitalization outcomes. The model is a **research architecture** and candidate methods pipeline, not evidence that a contagion mechanism caused U.S. care shortages.

## Emerging research problem and hypotheses

Hospital financial vulnerability and payer-mix exposure can affect institutional liquidity. An attractive but *unvalidated* hypothesis is that hospital financial stress may transmit through supply contracts or referral dependencies in ways unmeasured by hospital-level accounting summaries.

**RQ1:** Do observable hospital financial indicators and local-market structure predict subsequent reductions in service access, after credible controls and out-of-time validation?

**RQ2:** Does a validated referral or payment-dependency network improve held-out hospital financial-distress prediction beyond panel Econometrics and tabular Machine Learning?

**RQ3:** If a credible exogenous insurer-payment or policy shock exists, do network-exposed hospitals exhibit different responses, consistent with a preregistered network spillover design?

None can be tested by simply renaming synthetic bank nodes as hospitals.

## Candidate legitimate U.S. data sources

| Source | Potential measurements | Does it supply bilateral financial network exposures? |
|---|---|---|
| CMS Healthcare Cost Report Information System (HCRIS) | Facility-level hospital costs, revenues and payer-related accounting measures | No; not a complete payment-obligation graph |
| CMS Care Compare and Provider of Services data | Hospital services, quality indicators and facility identities | No |
| AHRQ HCUP State Inpatient Databases | Hospital admissions, payer, patient-origin information subject to permission and fees | Not necessarily: sensitive use and data agreements apply |
| NPPES NPI Registry | Provider identifiers and locations | No; NPI co-occurrence is not validated financial exposure |
| State hospital discharge, network participation or insurer-provider contract data | Possible geographic or contractual relationships | Depends on state and access license |

Data are subject to availability, variable definitions, review/IRB as appropriate, privacy rules and licensing.

## Suggested Econometric design

Construct a hospital–year panel $`i,t`$ after validating mergers, system affiliation, accounting changes, provider IDs, deflators and payer mix. A baseline descriptive model:

```math
Y_{it} = \alpha_i + \gamma_t +
\beta S_{i,t-1} + X_{it}'\delta + \varepsilon_{it},
```
where $`Y_{it}`$ is a **verified access or operational capacity outcome** and $`S_{i,t-1}`$ is a *lagged, measured* financial-stress indicator. Hospital and year fixed effects remove time-invariant heterogeneity and national shocks, but do not make the stress coefficient causal.

If an exogenous payer policy change with defensible comparison groups exists, consider staggered event-study / differences-in-differences methods with correct timing, anticipation checks, heterogeneous treatment effects, and clustering at the relevant market/provider-system level. Network interference invalidates off-the-shelf SUTVA; define exposure mappings, partial-interference clusters, and transparent spillover estimands.

A temporal GNN might predict next-year distress using **actual, documented** affiliation/referral networks. Compare against Elastic Net, regularized panel models, gradient-boosted trees, and graph-ablated baselines on geographically and temporally separated holdouts. SHAP or permutation attribution is not automatically a causal mechanism.

## Policy relevance

A validated model could inform early-warning dashboards for provider payment disruptions and resilience investments, but it must be assessed on **care-access and clinical service continuity measures**, not only provider profitability. Communicate subgroup uncertainty, small hospital performance, geographic equity, and the consequences of false positives or missing high-risk facilities.

## What would justify a scholarly paper?

A verified dataset, precise graph link semantics, statistical identification, novelty search, privacy/ethics review, prospectively defined outcomes, meaningful inference and externally validated predictions. Until then this file is a **research development proposal**, separate from the financial-network results.

## Related literature

Eisenberg and Noe (2001) motivate payment network clearing. Acemoglu et al. (2015) analyze shock propagation and nonlinear interconnectedness. Chernozhukov et al. (2018) explain DML in designs meeting the identification assumptions. None establishes a hospital reimbursement exposure effect without the corresponding real data. The repository's README contains full APA references.
