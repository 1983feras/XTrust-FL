# C4-v3 Diagnostic Protocol (Frozen Before Outcome Inspection)

Status: development diagnostic; **not paper-final evidence**.

## Motivation
C4-v2 (seed 31415, sign-flip, beta=0.2) produced zero attacker detections across three rounds. We therefore diagnose signal separation before any confirmation study. No threshold will be tuned against C4-v2 or C4-v3 attack labels.

## Fixed experimental setting
- CICIoT2023 public parquet with repository-pinned SHA-256.
- 50 clients, Dirichlet alpha=0.1, local cap=12,000, one local epoch, FedProx mu=0.01.
- Development seed 31415.
- Attack: existing defensive benchmark `sign_flip`; beta=0.2.
- Trusted server reference data from validation only; never test data.
- Same malicious client IDs, partitions, and reference indices as determined by the seed.
- Three attack rounds for diagnostic continuity with C4-v2.

## Clean-only fitting
Before any poisoned update is observed:
1. Build one clean client-update batch.
2. Fit a fixed update-anomaly reference from the clean updates.
3. Fit a fixed explanation-anomaly reference from clean server-side IG fingerprints.
4. Fit the heterogeneity calibration relationship on clean reference losses only.
5. Derive all normalization constants and trust/detection thresholds from clean scores only.

No reference, scaler, calibration model, or threshold may be refit on an attacked cohort.

## Predeclared diagnostic arms
Evaluate the same attacked RoundBatch under:
- U: fixed-reference update anomaly only.
- Eraw: fixed-reference raw explanation anomaly only.
- Ecal: heterogeneity-calibrated positive explanation residual only.
- U+Ecal: combined update and calibrated explanation signals without temporal smoothing.
- U+Ecal+T: same combination with the existing temporal smoothing rule.

The purpose is mechanistic diagnosis, not selecting a winner using attack labels.

## Required outputs
For every arm and round save:
- clean-derived threshold;
- per-client score/anomaly;
- TP, FP, TN, FN, TPR, FPR;
- benign and malicious mean/median distributions;
- separation diagnostics (AUROC where both classes exist);
- utility only where an arm is used for aggregation.

Also retain FedAvg, coordinate median, trimmed mean, Krum, and Multi-Krum paired utility metrics on the same immutable client updates.

## Interpretation rule
C4-v3 does **not** justify a paper claim and will not be used for significance testing. Its role is to determine whether the hypothesized signals contain defensible separation and whether temporal fusion masks it. Any subsequent design change must be documented and frozen before new confirmation seeds are inspected.

## Prohibited post-hoc actions
- No threshold search against malicious labels.
- No changing weights to maximize C4-v3 TPR/F1.
- No deleting difficult clients/seeds.
- No calling C4-v3 paper-final.
- No multi-seed confirmation until the mechanism is frozen.
