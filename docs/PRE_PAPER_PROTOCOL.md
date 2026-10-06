# XTrust-FL Pre-Paper Validation Protocol

Status: **protocol freeze before final manuscript results**

This document defines the evaluation plan before final results are inspected. Changes after result inspection must be documented with a methodological reason.

## Central research question

Can server-computed explanation residuals, calibrated for legitimate client heterogeneity, provide complementary evidence for distinguishing stealthy malicious behavior from benign Non-IID client variation in federated IoT intrusion detection?

## Primary hypotheses

- **H1 — Calibration:** heterogeneity calibration reduces benign-client false positives relative to raw explanation anomaly under severe benign Non-IID.
- **H2 — Complementarity:** update-space + calibrated explanation-space detection improves malicious-client detection relative to update-space alone, especially for stealth/mimicry attacks.
- **H3 — Robust utility:** XTrust-FL preserves higher task utility under poisoning without materially degrading clean FL performance.
- **H4 — Backdoor resistance:** XTrust-FL lowers attack success rate under targeted/backdoor poisoning while preserving benign accuracy/F1.
- **H5 — Efficiency:** event-triggered XAI retains most detection benefit of always-on XAI at lower explanation/runtime overhead.

## Datasets

1. **CICIoT2023** — primary dataset.
2. **ToN-IoT** — independent replication/validation dataset.

Datasets are evaluated independently unless a defensible common feature space is constructed. No direct cross-dataset train/test claim is permitted merely because both are IoT IDS datasets.

## Leakage-control rules

1. Split train/validation/test before fitting preprocessing statistics.
2. Fit scaler/encoder on training data only.
3. Never use test data to tune thresholds, calibration, trust weights, early stopping, attack parameters, or D_ref.
4. Federated client partitions are constructed from the training split only.
5. Save split indices and client partition indices for every seed.
6. D_ref must be independent of the test set and its origin/size must be reported.

## Federation settings

- Debugging: N=20 clients.
- Final: N=50 clients.
- Initial participation fraction: C=0.5.
- Seeds: **42, 77, 100, 999, 2026**.
- Non-IID Dirichlet alpha: **{10, 1, 0.5, 0.1}**.
- Record realized heterogeneity (e.g., Jensen-Shannon divergence), not alpha alone.

## Threat model

The coordinator is trusted for defense computation (or honest-but-curious regarding training data) and receives individual client updates. A subset of clients may be Byzantine and may coordinate. Malicious fraction beta is evaluated at **{0.1, 0.2, 0.3, 0.4}** where computationally feasible.

XTrust-FL v1 is **not directly compatible with strict secure aggregation that hides individual updates from the server**. This is an explicit limitation, not an omitted assumption.

The adaptive attacker may know the defense family and attempts to remain statistically plausible in update space.

## Attacks

Required attack families:

1. Label flipping / data poisoning.
2. Sign flipping.
3. Scaling/model replacement style attack.
4. Targeted/backdoor poisoning; report ASR.
5. Adaptive stealth/mimicry poisoning constrained toward benign update norm/cosine regions.
6. Coordinated/colluding poisoning where multiple malicious clients cooperate.
7. Natural concept drift without an attacker as a false-positive stress test.

The stealth attack must not be reduced to a trivially small or obvious anomalous update. Its goal is to damage the task while remaining plausible under update-space statistics.

## Baselines

Minimum classical/reference baselines:

- FedAvg
- FedProx
- Coordinate-wise Median
- Trimmed Mean
- Krum / Multi-Krum
- FLTrust

Add at least one recent, reproducible defense relevant to poisoning + heterogeneous federated IDS (e.g., P4P or FedDBC) only when a faithful implementation/protocol can be justified. If exact reproduction is not feasible, discuss it in Related Work rather than presenting an unfair approximate baseline as equivalent.

## XTrust-FL components

### Update-space signal

Compute robust update anomaly features such as norm, reference/global cosine similarity, loss behavior, and robust distance features.

### Server-side explanation signal

For each received client update, construct/evaluate its candidate model server-side and compute Integrated Gradients on trusted D_ref. Byzantine clients do not supply trusted explanation telemetry.

Explanation fingerprint metrics include:

- Spearman/rank drift
- Top-K Jaccard drift
- normalized attribution L1 drift
- faithfulness where feasible

### Heterogeneity calibration

Let raw explanation anomaly be D^E_i and heterogeneity fingerprint/score be H_i. Estimate expected benign explanation drift from a trusted/warm-up or coarse-screened population:

R^E_i = max(0, D^E_i - E_hat[D^E_i | H_i]).

The final paper must state exactly which variables constitute H_i and how calibration contamination is controlled.

### Temporal trust

Combine update anomaly and calibrated explanation residual, then smooth trust over rounds. Report all coefficients and threshold-selection rules. No threshold may be selected using the test set.

### Aggregation

Use trust-weighted aggregation with robust/adaptive clipping. Compare rejection versus down-weighting where useful.

## Reference-set sensitivity

Evaluate D_ref size at **{100, 500, 1000, 5000}** when dataset size permits. Report performance and XAI runtime. This is required because trusted-server-data dependence is an important comparison point with FLTrust.

## Core 2x2 experiment

| Condition | Heterogeneity | Poisoning | Purpose |
|---|---|---|---|
| C1 | Mild/IID-like | None | clean reference |
| C2 | Severe Non-IID | None | benign false-positive stress test |
| C3 | Severe Non-IID | obvious poisoning | standard robustness |
| C4 | Severe Non-IID | adaptive stealth poisoning | primary novelty stress test |

Primary visualization: client-level **(A_update, R_explanation)** scatter/distribution, labeled only for evaluation. The desired separation is a hypothesis, not a guaranteed result.

## Ablation

Run at minimum:

1. Update-only.
2. Update + raw explanation anomaly.
3. Update + heterogeneity-calibrated explanation residual.
4. Full XTrust-FL with temporal history.
5. Full XTrust-FL without temporal history.
6. Always-XAI versus Event-Triggered-XAI.

This ablation must establish whether calibration itself adds value rather than XAI merely adding complexity.

## Metrics

### IDS task utility

- Accuracy
- Precision
- Recall
- Macro-F1
- Weighted-F1
- AUROC
- AUPRC
- Confusion matrix

### Robustness

- Delta F1 relative to clean
- Relative degradation
- Backdoor Attack Success Rate (ASR)

### Malicious-client detection

- AUROC_mal
- AUPRC_mal
- TPR_mal
- FPR_benign

FPR_benign under severe benign Non-IID (C2) is a primary metric.

### Explanation quality

- Spearman stability
- Top-K Jaccard
- normalized attribution L1 drift
- faithfulness/confidence-drop test where feasible

### Efficiency

- training time per round
- server aggregation time
- explanation time
- inference latency
- communication bytes per round
- peak memory when measurable
- overhead relative to FedAvg

## Statistics

For final claims, use all five fixed seeds. Report mean +/- SD and 95% confidence intervals where appropriate. Use paired tests because methods share seeds/partitions: paired t-test when assumptions are defensible; otherwise Wilcoxon signed-rank. Report effect size and exact p-values where appropriate. Do not call a result statistically significant before the test is actually run.

Multiple-comparison correction should be considered for families of pairwise comparisons (e.g., Holm correction).

## Predefined success criteria

XTrust-FL is considered supported by the evidence only if the following pattern is reproducible rather than seed-specific:

1. Calibration reduces FPR_benign versus raw-XAI in C2.
2. Calibrated explanation residual adds malicious-client detection information beyond update-only in C4.
3. The robustness gain does not come with an unacceptable clean-utility loss.
4. Backdoor ASR is reduced when backdoor attacks are evaluated.
5. The main effect persists across five seeds and at least one independent dataset/replication setting.

Failure of any criterion must be reported and may require revising the method before final manuscript claims are frozen.

## Result provenance and reproducibility

Every scientific result must have a saved machine-readable artifact containing at least:

- git commit SHA
- dataset identifier/hash where legally/technically feasible
- seed
- split/partition identifier
- malicious client IDs
- attack configuration
- defense configuration
- per-round metrics
- final metrics
- runtime/device information

Synthetic smoke-test results must never be presented as scientific dataset results.

## Current evidence rule

Previously discussed pilot numbers are **not manuscript-grade results unless independently recoverable from saved result files/logs and reproducible from the repository**. The final Results section will use only verified artifacts produced under this frozen protocol.

## Manuscript gate

Freeze final Results/Abstract/Conclusion only after:

- baseline implementations are verified,
- C2 and C4 are completed,
- five-seed ablation is complete,
- statistical analysis is complete,
- external replication is complete or its absence is explicitly justified,
- efficiency/overhead is measured,
- literature/novelty audit is updated immediately before submission.
