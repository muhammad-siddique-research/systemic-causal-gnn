# Reproduced computational results — synthetic benchmark only

**This is not an empirical bank dataset, a trial of real policies, or a validated causal GNN.**

- Independently generated networks: **140**
- Mean baseline unpaid obligations: **93.21 synthetic units**
- Mean loss reduction, injection to largest debtor: **5.86**
- Mean loss reduction, injection to random debtor: **1.80**
- Paired targeted-minus-random mean: **4.06** (bootstrap 95% interval 3.04 to 5.20)

## Out-of-network predictive models

| model                  |   n_train |   n_test |   train_networks |   test_networks |   test_event_rate |   roc_auc |   average_precision |   brier |   log_loss |
|:-----------------------|----------:|---------:|-----------------:|----------------:|------------------:|----------:|--------------------:|--------:|-----------:|
| Regularized Logistic   |      2100 |      700 |              105 |              35 |            0.1700 |    0.9979 |              0.9914 |  0.0171 |     0.0637 |
| Gradient-Boosted Trees |      2100 |      700 |              105 |              35 |            0.1700 |    0.9965 |              0.9848 |  0.0192 |     0.0680 |
| Random Forest          |      2100 |      700 |              105 |              35 |            0.1700 |    0.9931 |              0.9716 |  0.0369 |     0.1374 |

**Qualification:** Tabular supervised classifiers (logistic, RF, boosting), not a validated graph neural network. Model selection used test AUC; additional nested validation is needed.

## Cross-fitted DML known-truth exercise

- True synthetic coefficient: **-1.500**
- Mean naive OLS estimate: **-1.019**
- Mean cross-fit DML estimate: **-1.395**
- Mean absolute DML deviation from truth: **0.105**
- Fraction of IID DML 95% CIs covering synthetic truth: **70.00%** (10 simulated replications; not coverage validation).

## Research visuals

![Figure 1](figures/figure_1_policy_interventions.png)
![Figure 2](figures/figure_2_paired_clearing.png)
![Figure 3](figures/figure_3_prediction_validation.png)
![Figure 4](figures/figure_4_permutation_reliance.png)
![Figure 5](figures/figure_5_dml_known_truth.png)
![Figure 6](figures/figure_6_synthetic_network.png)

## Inference boundary

The paired bootstrap resamples entire independent simulated networks. DML variance relies on IID observations in a separately simulated partially linear model. Neither confidence interval transfers to real interconnected institutions or economic interventions.
