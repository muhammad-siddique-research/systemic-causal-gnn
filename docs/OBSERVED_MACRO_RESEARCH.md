# Observed U.S. Macro-Financial Data Extension — Empirical Research Protocol

**Proposed title:** Forecasting U.S. Credit-Market Stress: HAC-Robust Econometrics and Out-of-Time Machine Learning as a Complement to Network Stress Simulation.

**Status:** Code and offline-generated fixture tests are implemented. The complete official FRED downloads have NOT been executed or validated within this project yet. No results in this document are claimed from actual FRED data.

## Emerging research question and contribution

Can simple persistence or linear regularized Econometrics forecast next-month corporate credit spreads as effectively as nonlinear Machine Learning when evaluation respects chronological order and autocorrelated loss uncertainty?

This tests observable aggregate credit-market behavior, not causal network contagion. It is an empirical complement to the repo's simulated Eisenberg–Noe clearing experiments, not a substitute for real bilateral bank exposures.

## Genuine public U.S. data sources

| FRED series | Measure | Source |
|---|---|---|
| BAA | Moody's Seasoned Baa Corporate Bond Yield (monthly) | https://fred.stlouisfed.org/series/BAA |
| GS10 | 10-Year Treasury Constant Maturity Rate (monthly) | https://fred.stlouisfed.org/series/GS10 |
| UNRATE | Civilian Unemployment Rate (monthly) | https://fred.stlouisfed.org/series/UNRATE |
| INDPRO | Industrial Production Index (monthly) | https://fred.stlouisfed.org/series/INDPRO |
| FEDFUNDS | Effective Federal Funds Rate (monthly) | https://fred.stlouisfed.org/series/FEDFUNDS |

Record each source's original URL, file SHA-256, calendar coverage, download date, variable units, and the series' vintage. Fingerprints establish local file identity, NOT provenance by themselves.

## Econometric methodology

Define an aggregate corporate credit-spread proxy S_t = BAA_t − GS10_t (percentage points). The target is S_(t+1). Fit on the development period:

```math
S_{t+1}=\beta_0+\beta_1 S_t+\beta_2 U_t+\beta_3 F_t+\beta_4\,100\Delta\log(IP_t)+\epsilon_{t+1},
```
where U_t is unemployment, F_t the federal funds rate, and IP_t industrial production. This is a **descriptive time-series linear projection**. Macroeconomic variables are potentially jointly determined with credit spreads, so fitted slopes are not identified monetary policy effects.

1. Validate monthly timestamps, reject duplicate months and unannounced interpolation; align the five official series.
2. Compute the spread and industrial production log growth; shift the next-month target and require adjacent calendar months.
3. Train on the earliest 75% of valid months; hold out the latest 25%, preserving temporal order.
4. Fit OLS with Newey–West/HAC standard errors (three lags) on development months.
5. Compare a persistence benchmark, standardized Ridge Regression (scaling trained on development only) and histogram gradient-boosted regression (fixed parameters).
6. Report out-of-time MAE and RMSE in **percentage points**, not percentages.
7. Evaluate candidate minus persistence squared-error differences using HAC uncertainty, emphasizing exploratory performance rather than statistically guaranteed supremacy.

For loss differential d_t = e²_candidate,t − e²_persistence,t, report:

```math
\bar d \pm 1.96\,\widehat{\mathrm{SE}}_{HAC}(\bar d).
```
Negative values favor the candidate model. Three HAC lags are a transparent initial choice, not an optimized bandwidth. Assess structural breaks, forecast origin dependence and alternate lag windows before scholarly submission.

## Crucial historical-vintage limitation

Current FRED downloads may include revised unemployment and industrial production observations, which were **not necessarily available at time t** to a forecaster predicting t+1. The pipeline is therefore a retrospective time-ordered comparison, **not a historical real-time trading or policymaking backtest**. For authentic real-time validation use ALFRED vintages, true release calendars, and expanding-window forecast origins.

These series are aggregate market indicators. They provide **no bilateral bank graph**, individual defaults, clinically observed hospital outcomes, or network treatment assignment. Therefore the observed series cannot validate the simulated Graph Neural Network or establish causal policy effects.

## Reproduction commands

```bash
python scripts/download_fred.py --output-dir data/fred
python -m src.research.observed_macro --data-dir data/fred --output-dir reports/observed_macro
python -m pytest -q tests/test_observed_macro.py
```

The public endpoint may be inaccessible in some computing environments; in that case download the five CSV series manually from FRED. The repository's existing CI tests an explicitly invented **offline CSV fixture**. Only an independently confirmed FRED run should be represented as genuine empirical findings.

## Expected output (no unexecuted results asserted)

- Source fingerprint manifest, period and split audit.
- HAC-robust OLS coefficients and nominal intervals.
- Three-model held-out forecast benchmark and per-month forecast CSV.
- PNG/SVG figure with actual-versus-predicted credit spread and RMSE comparison.
- HAC mean squared-loss-difference intervals relative to persistence.

## Research development agenda

Use original-release ALFRED snapshots and expansion-window refitting; study recession/financial-crisis subperiods, measurement revision, varying horizon, and prediction intervals. For causal financial-network identification, obtain actual liability edges or a legally and economically validated exposure proxy. This extension should **never** rename FRED aggregate observations as bank-level network evidence.

## Selected references (APA 7th edition)

Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics, 13*(3), 253–263. https://doi.org/10.1080/07350015.1995.10524599

Eisenberg, L., & Noe, T. H. (2001). Systemic risk in financial systems. *Management Science, 47*(2), 236–249. https://doi.org/10.1287/mnsc.47.2.236.9835

Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica, 55*(3), 703–708. https://doi.org/10.2307/1913610

Stock, J. H., & Watson, M. W. (2002). Forecasting using principal components from a large number of predictors. *Journal of the American Statistical Association, 97*(460), 1167–1179. https://doi.org/10.1198/016214502388618960

Federal Reserve Bank of St. Louis. (n.d.). *Federal Reserve Economic Data (FRED)* [Data sets]. https://fred.stlouisfed.org/