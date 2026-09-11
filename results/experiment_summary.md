# Experiment summary — Fase 4 (all numbers regenerable via 05_Code/run_all.sh)

Protocol: rolling origin, 3 origins (train 2015–2022→test 2023; →2024; →2025), one-step-ahead. All models see identical windows and identical inputs per feature set; ML target log1p, scored in levels (EUR). Seeds fixed (42). PPML converged in all 6 fits (11–16 iterations; contrast: Gopinath et al.'s PPML reference failed to converge).

## RQ1 — Predictive performance (Table 5.2, pooled over 3 origins)

| Model | RMSE (EUR) | MAE (EUR) | RMSE log | WAPE % | R² levels |
|---|---|---|---|---|---|
| **RF_lags (winner)** | **155,657** | **11,404** | 1.820 | **33.1** | **0.862** |
| LGBM_lags | 214,439 | 14,513 | **1.793** | 42.1 | 0.738 |
| MLP_lags | 236,295 | 18,610 | 1.801 | 54.0 | 0.681 |
| LGBM_nolags | 293,269 | 21,173 | 1.954 | 61.4 | 0.509 |
| PPML_B (FE prediction spec) | 334,620 | 36,662 | 5.716 | 106.3 | 0.361 |
| RF_nolags | 339,582 | 23,124 | 2.047 | 67.1 | 0.342 |
| PPML_A (interpretable spec) | 413,095 | 64,263 | 7.622 | 186.4 | 0.026 |
| MLP_nolags | 594,492 | 39,087 | 1.960 | 113.3 | −1.017 |

- Headline: **RF with lags cuts out-of-sample RMSE by 53.5% vs PPML_B** (paired bootstrap, B=2000: Δ=+178,963 EUR, CI95 [136,835; 223,587], p<0.001, n=75,816).
- **Metrics disagree (Sellami precedent):** LGBM_lags wins on RMSE-log; RF_lags on level metrics. Reported as a finding.
- **Without lags, PPML_B beats RF (334.6k vs 339.6k)** and only LGBM stays ahead — the ML advantage is largely persistence.
- Stable across origins: RF_lags beats PPML_B in O1/O2/O3 (126.5k/213.3k/105.9k vs 306.4k/375.3k/318.1k); R² 0.89/0.78/0.93.
- Segments: RF_lags errs less on true zeros (RMSE 1.5k vs 30.5k), on flows ≥100k EUR (823k vs 1,738k) and on Brazil (17.2k vs 153.6k). MAPE reported only on nonzero flows, with caveat.

## RQ2 — Coefficients vs importances (Table 5.3)

PPML spec A (O3, cluster-robust by pair): ln GDP_o **+0.700** (p<0.001), ln GDP_d **+0.835** (p<0.001), ln dist **−0.393** (p=0.010, weak — echoes Chen et al.), contig +1.155 (n.s.), comlang +0.536 (n.s.), comcol −1.402 (p<0.001), rta **−0.701** (p=0.002, counterintuitive sign — selection: the EU's biggest trimmings suppliers have no RTA; discussed honestly).

ML importances (permutation, RF winner, O3 test): **roll3_log 0.269 and l1_log 0.132 dominate** — persistence the gravity structure cannot see (the anticipated RQ2 divergence). Best economic variable: ln_gdp_o (0.027). SHAP directions computed on LGBM_lags (TreeExplainer on the large RF is computationally infeasible; runner-up model, same features — stated in the text): GDP_d positive, distance direction POSITIVE in the ML (vs negative PPML coefficient) — a real divergence to interpret (distance proxies large Asian suppliers once persistence is controlled).

## RQ3 — Brazil (Table 5.4, test years 2023–25 pooled)

- Benchmark = RF_lags: Brazil observed €12.65M vs predicted €11.00M (**+15.0%**). BUT the 10 similar-scale suppliers show median **+56.5%** (range +18.4% to +179.1%) — Brazil ranks **11th of 11**: last of its peer group.
- PPML_B benchmark: Brazil **−13.5%** (under-trading).
- Consistent reading across both benchmarks: **Brazil underperforms its peer group** — an indicative signal, not an entry recommendation (deviation mixes potential, model error and omitted factors).
- By heading: 5806 (98.7% of Brazil's EU trade) close to potential (+13.7%); tiny headings show huge relative deviations off near-zero bases (base effect, flagged).

## Robustness (Table 5.5, origin O3)

| Check | RMSE winner | RMSE PPML_B | Conclusion holds? |
|---|---|---|---|
| baseline O3 | 105,912 | 318,125 | Y |
| 1 without 5806 | 101,690 | 334,673 | Y |
| 2 HS4 aggregation | 295,855 | 868,614 | Y |
| 3 excl. 2020–21 | 102,331 | 317,730 | Y |
| 4 hyperparameter perturbation | 150,630 | 318,125 | Y |
| **5a ablation: no lags** | **321,841** | **318,125** | **N** |
| 5b ablation: no rta | 106,473 | 318,634 | Y |
| 6 EU as one bloc | 1,109,032 | 2,420,797 | Y |

The one failing check is the most informative: strip the lags and the RF loses to PPML — the ML gain comes from persistence, not from a better use of gravity covariates. (This feeds 5.5's closing sentence; all-confirming robustness would invite suspicion.)

## Real-world validation (5.1 P3, supervisor requirement)

Mirror Brazil→EU27: mean discrepancy +6.1% (CIF>FOB, expected direction), correlation 0.985. Pandemic dip: −20.1% in 2020, +18.1% rebound in 2021.

## Files

Tables: table_5_1_panel.csv, table_5_2_metrics.csv, table_5_2_segments.csv, table_5_3_rq2.csv, table_5_4_deviations.csv, table_5_5_robustness.csv. Figures: fig_5_1_headings.png, fig_5_2_suppliers.png, fig_5_3_shap.png, fig_5_4_brazil.png. Support: significance.json, rq3_summary.json, ppml_coefficients_specA.csv, ppml_fit_report.json, ml_tuning_report.json, perm_importance_blocks.csv, shap_summary.csv, predictions_*.csv.gz.

## Fase 5 — hybrid update (Sep 2026, supervisor feedback)

Hybrid feature sets added (PPML spec B prediction as an ML feature, train rows in-sample, test rows frozen-coefficient; no leakage). Winner: **RF_hyb_lags 155,370** vs RF_lags 155,657 — paired bootstrap Δ=288 EUR, CI95 [-1,243; +1,528], **p=0.635, a tie**. Headline vs PPML_B unchanged in substance: 53.57% RMSE reduction (p=0.0005). Without lags the gravity feature helps every family: RF 339,582→312,877; LGBM 293,269→287,840; MLP 500,479→494,421. MLP figures differ from Fase 4 because inputs moved to float32 (container memory fix; documented for 4.7). Robustness now 9 checks, all Y; new 5c (drop ppml feature from winner) gives 105,912 ≈ the pure RF_lags baseline, and 5a (drop lags from the hybrid) gives 291,452, still ahead of PPML because the gravity feature remains. RQ2: perm importance roll3 0.249, l1 0.135, **ppml_pred_log 0.044 in third**, ln_gdp_o 0.013. RQ3 with the hybrid benchmark: Brazil +21.5% vs peer median +72.0% (range 27.6 to 213.3), still 11th of 11; heading 5806 +20.2%; PPML_B cross-check −13.5% unchanged.

## Fase 6 — methodological fixes and re-run (11 Sep 2026, second round of supervisor feedback)

Four fixes, all re-run in one go (Colab, statsmodels 0.14.6, scikit-learn 1.8.0, LightGBM 4.7.0, SHAP 0.51.0): (1) roll3_log is now the rolling mean inside each exporter-destination-hs6 series (the previous window ran across series boundaries and contaminated the 2015 and 2016 rows, 12.4% of the panel); (2) the hybrid feature is the spec B prediction for year t from a fit on 2015..t-1, for training and test rows alike, starting in 2017 (2015 and 2016 get 0, like the second lag), instead of in-sample fitted values on the training rows; (3) the MLP epoch count is tuned on the temporal validation year (13 epochs without lags, 5 with) instead of sklearn's random early-stopping split; (4) the paired bootstrap resamples the 1,053 exporter-destination pairs instead of rows. A naive persistence forecast (last year's value) was added to Table 5.2.

Winner is now the pure **RF_lags, 156,560** (RF_hyb_lags 156,808; cluster bootstrap Δ=-247 EUR, CI95 [-1,440; +1,210], p=0.648, still a tie, now with the pure forest marginally ahead). Vs PPML_B: **53.2% RMSE reduction**, Δ=178,060, CI95 [98,502; 264,415] (wider than the row bootstrap's [136k; 225k]), p=0.0005. **Vs Naive_lag1 (161,420): 3.0% reduction, CI95 [-23,968; +28,446], p=0.814, not significant.** Per origin the naive forecast is ahead in 2023 (125,329 vs 129,165) and 2024 (194,602 vs 212,896) and behind only in 2025 (156,819 vs 107,354); the forest's edge over persistence is in the zeros (RMSE 1,342 vs 5,355 on true zeros) and the small flows, not in the large corridors. PPML results unchanged (the fixes did not touch the gravity branch). LGBM improved with the corrected lag (214,439→200,535 with lags); MLP moved a lot (296k–397k range) as expected from the epoch change. Without lags the gravity feature still helps every family: RF 339,307→318,298; LGBM 293,269→274,612; MLP 396,562→312,905.

RQ2 (RF_lags on O3): perm importance roll3 0.321, l1 0.147, ln_gdp_o 0.029, zero_streak 0.018, ln_gdp_d 0.008; SHAP directions from LGBM_lags: ln_dist +0.178, ln_gdp_d +0.071. RQ3 (benchmark RF_lags): Brazil +16.3% (12.65 vs 10.88 M EUR) against a peer median of +59.9% (range 17.0 to 174.1), 11th of 11; PPML_B cross-check -13.5% unchanged. Robustness (8 checks, 5c dropped because the winner is not a hybrid): all Y except 5a (no lags: 319,289 vs 318,125, N), i.e. the Fase 4 pattern is back.
