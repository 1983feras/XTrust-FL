# C4-v4 Frozen Development Protocol

## Status
Development diagnostic only. This protocol is frozen before any C4-v4 outcome inspection. It is not paper-final evidence.

## Motivation fixed from C4-v3
C4-v3 showed that clean-fixed update/explanation references avoid attacked-cohort contamination, but explanation separation degraded across rounds and XTrust utility was not competitive with robust baselines. C4-v4 tests one architectural hypothesis: reference staleness under model trajectory shift.

## Frozen experimental constants
- CICIoT2023 verified parquet and existing repository SHA check.
- Seed: 31415.
- 50 clients.
- Dirichlet alpha: 0.1.
- Per-client cap: 12000.
- Local epochs: 1.
- FedProx mu: 0.01.
- Attack: existing defensive benchmark `sign_flip`.
- Malicious fraction beta: 0.2.
- Three evaluated attack rounds.
- Validation-only trusted reference data; no test-data fitting.
- Reference size: 128.
- IG steps: 16.
- Same deterministic client partition, malicious IDs, and reference selection conventions as C4-v3.

## Single architectural change under test
At the start of every evaluated round, before constructing or scoring the attacked batch, fit round-conditioned anomaly references from a CLEAN shadow batch generated from the current canonical/base global model using the same benign client partitions and local-training configuration. This clean shadow batch is calibration-only and is never used as the attacked evaluation batch.

For each round, fit only from that round's clean shadow batch:
1. fixed update anomaly reference;
2. fixed explanation anomaly reference;
3. clean loss -> raw explanation anomaly heterogeneity calibration (RobustScaler + HuberRegressor);
4. normalization constants from clean q95 values;
5. trust thresholds from clean q05 values.

Then freeze all five items for the corresponding attacked round. Do not refit on, normalize from, or otherwise use the attacked cohort to estimate any reference, scale, regression, or threshold.

## Predeclared diagnostic arms
- U: update-only anomaly converted monotonically to trust.
- Eraw: raw explanation anomaly converted monotonically to trust.
- Ecal: positive heterogeneity-calibrated explanation residual converted monotonically to trust.
- U+Ecal: existing equal-weight scoring, no temporal smoothing.
- U+Ecal+T: existing temporal scoring. Previous-round trust is used only after the current round's U/Ecal components are normalized with that round's clean shadow calibration.

No arm weights, sigmoid bias, temporal gamma, percentile, or decision rule may be tuned from attacker labels.

## Detection evaluation
For every arm and round save client IDs, malicious ground-truth labels for evaluation only, per-client scores/signals, threshold, TP, FP, TN, FN, TPR, FPR, malicious/benign mean and median, and malicious-class AUROC (negative trust as attack score).

## Utility evaluation
Use the same immutable attacked RoundBatch within each round for paired mechanism comparisons. Retain FedAvg, coordinate median, trimmed mean, Krum, Multi-Krum, and XTrust aggregation. XTrust aggregation uses the predeclared non-temporal U+Ecal arm and its current round CLEAN-derived q05 trust threshold. Save accuracy, Macro-F1, precision, recall, AUROC and AUPRC where provided by the repository evaluator.

The canonical/base trajectory may continue through the XTrust candidate for this development diagnostic, matching the paired-mechanism scope. This is not a substitute for final independent defense trajectories.

## Required provenance and efficiency fields
Save round calibration constants and thresholds, clean-shadow calibration scores, malicious IDs, partition sizes, validation reference indices, dataset SHA, exact config, and paper_final=false. Measure at minimum clean calibration time, attacked batch/update time, explanation/scoring time, aggregation/evaluation time, and total round time.

## Success interpretation
C4-v4 is diagnostic. Evidence favoring the architectural hypothesis requires improved cross-round stability of calibrated explanation/fusion discrimination without a material FPR increase, while utility must be reported independently and may still expose an aggregation-policy problem. No single-seed development result establishes statistical significance or publication readiness.

## Prohibited post-hoc actions
- No threshold search using malicious labels.
- No changing fusion weights after seeing C4-v4 outcomes.
- No changing temporal gamma after seeing outcomes.
- No deleting clients, rounds, or unfavorable results.
- No calling C4-v4 paper-final.
- No multi-seed confirmation until the mechanism is frozen after this diagnostic.
- No claim that round-conditioned calibration is superior unless the recorded outcomes support it.
