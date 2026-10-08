# XTrust-FL Final Study v1 — Frozen Protocol

**Status:** FROZEN BEFORE FINAL-STUDY OUTCOMES  
**Parent diagnostic:** C4-v7 (`6fcd3f936779c2bf0c98fbb6c849549df3fd1590`)  
**Purpose:** confirmatory evaluation of the mechanism selected after C4 diagnostics; no further C4 threshold tuning.

## 1. Confirmatory question
Does a **heterogeneity-calibrated explanation-aware prefilter followed by robust aggregation** improve robustness/utility under poisoning while preserving benign Non-IID clients, compared with standard and robust aggregation baselines?

The selected principal mechanism is fixed before final-study outcomes:

`round-conditioned clean calibration -> U+Ecal score -> clean Q0.05 prefilter -> coordinate-wise median`.

The 5th-percentile threshold is inherited from C4-v7 and is not tuned in the final study.

## 2. Data and core FL configuration
- Dataset: CICIoT2023 cleaned parquet used throughout XTrust-FL.
- Required SHA-256: `17e098c1cee713f5b1af79b918ae7ba59611c30a3bc609237d7afa7e9a97f8b1`.
- Clients: 50.
- FedProx mu: 0.01.
- Reference size: 128.
- Integrated-Gradients steps: 16.
- Per-client cap: 12000 unless a separately frozen resource study changes only the cap.
- Main Non-IID setting: Dirichlet alpha = 0.10.
- Clean calibration is server-side and round-conditioned.
- No client-provided explanation telemetry is trusted.

### Final-study initialization
The paper-final study must use a **clean federated warm-up**, not the development centralized pretraining shortcut used in C4. The warm-up procedure, number of rounds, optimizer settings, and model-selection rule must be frozen in the implementation commit before any final attacked outcomes are inspected. It must not use malicious labels or attacked validation outcomes.

## 3. Principal defense
For each evaluated round:
1. Build the clean shadow/reference calibration from the current canonical model.
2. Compute clean U+Ecal scores using the already-defined update anomaly and heterogeneity-calibrated explanation residual.
3. Set `tau = Q_0.05(clean U+Ecal scores)`.
4. On attacked client updates, retain `S = {i: score_i >= tau}`.
5. If fewer than 3 clients remain, retain the top 3 U+Ecal scores, ties by original client order.
6. Aggregate retained updates with coordinate-wise median.

No malicious label, beta value, attacked utility, or attacked validation metric may enter steps 1–6.

## 4. Baselines and ablations
Required aggregation baselines:
- FedAvg
- Coordinate Median
- Trimmed Mean, trim ratio 0.20
- Krum
- Multi-Krum
- FLTrust-style baseline, implemented according to its published trusted-root-update principle; implementation details must be documented and frozen before outcomes.

Required XTrust ablations:
- robust estimator alone: Coordinate Median
- U-only prefilter + Coordinate Median
- Ecal-only prefilter + Coordinate Median
- U+Ecal prefilter + Coordinate Median (**principal XTrust-FL**)
- U+Ecal prefilter + Trimmed Mean (secondary)

Historical hard/soft C4 arms may be reported as development evidence but are not required as principal final-study competitors.

## 5. Experimental grid
### Seeds
Primary confirmatory seeds are fixed as:
`[42, 77, 100, 999, 2026]`.

### Poisoning fraction
For the main supported poisoning attack, evaluate:
`beta in {0.10, 0.20, 0.30, 0.40}`.

### Clean control
For every seed, include `beta = 0.00` to measure benign utility and false exclusion under severe Non-IID heterogeneity.

### Heterogeneity
Primary final setting: `alpha = 0.10`.
A secondary heterogeneity sweep may use `alpha in {0.10, 0.30, 1.00}` but must be run only after the primary grid is complete and without changing the principal defense.

### Attack scope
The confirmatory implementation begins with the already-supported defensive benchmark attack used in C4 (`sign_flip`). Additional ordinary benchmark attacks already supported by the repository may be added only as separately identified robustness experiments, without tuning the principal mechanism to their outcomes. No sophisticated adaptive/colluding attack is introduced as part of this freeze.

## 6. Trajectory and pairing
Each method must have an **independent multi-round trajectory** in the final study. A method's next-round global model is its own previous-round model; do not use the XTrust trajectory as the shared base for competitors.

For fair pairing, within a seed/configuration, use the same deterministic client partition, malicious-client draw where applicable, and local stochastic seeds across methods whenever the method's own trajectory permits. Record all seeds and client IDs.

Cross-method metrics are paired by seed/configuration, not by reusing one method's global trajectory.

## 7. Outcomes
### IDS utility
Record at minimum:
- Macro-F1 (primary utility outcome)
- Accuracy
- Precision
- Recall
- AUROC
- AUPRC
- confusion matrix

### Poisoning detection / filtering
Record:
- TP, FP, TN, FN
- TPR
- FPR
- detection AUROC
- retained/excluded counts
- retrospective benign/malicious retained counts (analysis only)
- benign false-exclusion rate
- malicious exclusion rate

### Efficiency
Record:
- calibration time
- update scoring time
- explanation time
- aggregation time
- total round time
- peak memory if practical

## 8. Statistical analysis
For each fixed condition and method, report mean, standard deviation, and 95% confidence interval across the five frozen seeds.

For the principal comparison, use paired per-seed differences in Macro-F1 between principal XTrust-FL and each principal baseline. Report effect size/difference with 95% CI. Use a paired t-test only when its assumptions are reasonable; otherwise use a paired non-parametric test such as Wilcoxon and explicitly note the very small n=5 limitation. Do not equate non-significance with equivalence.

Detection/filtering results must be reported with descriptive uncertainty; avoid significance claims unsupported by the sample size.

## 9. Additional frozen analyses after the primary grid
Without retuning the principal mechanism:
- reference-size sensitivity: `D_ref in {64, 128, 256}`;
- detector ablation U vs Ecal vs U+Ecal;
- heterogeneity sweep described above;
- efficiency comparison;
- benign-only false-exclusion analysis.

These are sensitivity/ablation studies, not opportunities to select a better post-hoc principal threshold.

## 10. Reproducibility and provenance
Every final result artifact must contain:
- git execution commit
- git status
- Python/PyTorch/NumPy/scikit-learn versions
- device
- dataset path and verified SHA-256
- full CLI/config
- seed
- partition metadata
- malicious client IDs when beta > 0
- timing
- explicit `paper_final: true` only for runs conforming to this protocol and its pre-outcome implementation freeze.

Raw JSON results must be retained. Aggregated tables/figures must be generated from raw JSON by version-controlled scripts; no hand-entered paper numbers.

## 11. Stop rules / anti-tuning guardrails
After this protocol freeze:
- do not alter Q0.05 based on final attacked outcomes;
- do not change the principal U+Ecal prefilter+median arm after seeing results;
- do not tune on malicious labels;
- do not replace the frozen seeds because results are unfavorable;
- do not discard failed seeds unless a documented implementation/runtime failure is demonstrated and rerun under the identical commit/config;
- do not claim superiority from a single condition or seed;
- if XTrust-FL does not outperform a robust baseline, report that result and frame the contribution around calibrated filtering/detection or robustness trade-offs only if supported.

## 12. Paper claim gate
A paper claim of utility superiority is allowed only if the final multi-seed results support it with consistent effect direction and defensible uncertainty. Otherwise the manuscript must state that XTrust-FL is competitive or provides complementary explanation-aware filtering, according to the observed evidence.

This document freezes the scientific design. Implementation bugs may be corrected before final outcomes, but every correction must be committed and documented; scientific thresholds/arms/seeds may not be changed in response to outcomes.