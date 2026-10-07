# C4 Frozen Protocol — Severe Non-IID + Poisoning

Status: **FROZEN BEFORE OUTCOME INSPECTION**

This document freezes the primary C4 evaluation protocol for XTrust-FL. The protocol must not be tuned in response to C4 confirmation outcomes. Any later sensitivity analysis must be explicitly labeled and reported separately.

## Scientific question

Does heterogeneity-calibrated explanation-aware scoring preserve low false-positive rates for benign Non-IID clients while detecting poisoned clients and preserving global IDS utility?

## Dataset and preprocessing

- Primary dataset: CICIoT2023 public cleaned parquet used by the existing reproducible pipeline.
- Expected SHA-256: `17e098c1cee713f5b1af79b918ae7ba59611c30a3bc609237d7afa7e9a97f8b1`.
- Binary target: `is_attack`.
- `Label`, `family`, and `is_attack` are excluded from predictors.
- Split before preprocessing.
- Imputation and standardization are fit on training data only.
- Test data are never used to fit preprocessing, thresholds, explanation references, or heterogeneity calibration.

## Federated setting

- 50 clients.
- Severe Non-IID partition: Dirichlet alpha = 0.1.
- FedProx local objective with fixed mu specified by the runner/config.
- Local epochs = 1 for the primary experiment.
- Client training cap = 12,000 examples per round/client.
- Server-side trusted reference set `D_ref`; primary pilot size = 128, with later preregistered sensitivity at {100, 500, 1000, 5000}.
- Integrated Gradients is computed server-side; clients do not supply explanation telemetry.
- Individual client updates are visible to the trusted/honest-but-curious coordinator; strict secure aggregation is outside the v1 threat model.

## Frozen XTrust-FL calibration

The C2-confirmed parsimonious candidate is retained without outcome-driven retuning:

`R_i^exp = max(0, D_i^exp - Dhat_i^exp(L_i^ref))`

where `L_i^ref` is the server-computed loss of client model i on trusted `D_ref`.

Calibration references must be fit without evaluation-client leakage. Explanation prototypes and robust feature location/scale are fixed before scoring held-out clients.

## Primary attack evaluation

Primary C4 evaluates malicious-client fractions beta in {0.1, 0.2, 0.3, 0.4} under supported poisoning transformations. Initial implementation validation may use the existing sign-flip and model-scaling transformations. Additional attacks (label flipping, backdoor, adaptive stealth, coordinated/colluding poisoning) are added only when implemented faithfully and are reported as separate attack conditions; unsupported attacks must not be simulated by relabeling another transformation.

Malicious client IDs are selected deterministically from the seed and frozen before round outcomes are observed.

## Baselines

All aggregators consume the exact same immutable set of client updates in each paired round:

- FedAvg
- FedProx training + FedAvg aggregation
- Coordinate-wise median
- Trimmed mean
- Krum
- Multi-Krum
- FLTrust-style baseline using the trusted server reference update
- XTrust-FL

The implementation must not describe the local FLTrust-style implementation as a fully faithful reproduction unless checked against the official method and its assumptions.

## Primary outcomes

Client-level defense metrics:

- Benign false-positive rate (FPR)
- Malicious true-positive rate (TPR / recall)
- Precision
- F1 score
- AUROC and AUPRC when continuous scores permit valid computation

Global IDS utility:

- Accuracy
- Macro-F1
- Precision
- Recall
- AUROC
- AUPRC
- Confusion matrix

Efficiency:

- training time per round
- explanation time
- total round time
- communication bytes per round (or an explicitly defined estimate)
- peak memory when measured reliably

## Paired evaluation and reproducibility

- Competing defenses receive the same client updates per round.
- Partitions, participating clients, malicious IDs, attack parameters, reference-set indices, and seeds are saved in result artifacts.
- Primary confirmation uses multiple fixed seeds; exploratory/development seeds are not silently mixed with confirmation results.
- Report mean ± SD and 95% confidence intervals where appropriate.
- Use paired tests across matched seeds/conditions when assumptions are reasonable; otherwise use a paired non-parametric test.
- Report effect sizes and apply Holm correction when testing multiple primary comparisons.
- A non-significant result must be reported as non-significant regardless of direction.

## Ablations after primary C4

Predefined ablations:

1. update-space only
2. update + raw explanation anomaly
3. update + calibrated explanation residual without temporal history
4. full XTrust-FL
5. no-temporal variant
6. Always-XAI vs event-triggered XAI

These ablations must not change the frozen primary C4 definition.

## Interpretation rule

C4 supports the central hypothesis only if the defense reduces or maintains benign FPR under severe Non-IID conditions while providing useful malicious-client detection and preserving global IDS utility. A low benign FPR alone is insufficient. Poisoning detection alone is also insufficient if benign heterogeneity is systematically rejected or global utility collapses.

## Known limitations to retain in the paper

- trusted server reference data are required;
- strict secure aggregation is incompatible with per-client scoring in the current design;
- server-side explanations introduce computation overhead;
- alpha = 0.1 is intentionally severe and creates extremely small client partitions in some seeds;
- the primary reference size 128 is a pilot setting and requires sensitivity analysis;
- operational deployment cannot assume known-benign client identities; controlled calibration experiments and operational trust bootstrapping must be distinguished.
