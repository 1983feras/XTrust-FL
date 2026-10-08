# C4-v7 Frozen Protocol — XAI-Guided Prefilter + Robust Aggregation

**Status:** FROZEN BEFORE ATTACKED C4-v7 OUTCOMES  
**paper_final:** false  
**Parent diagnostic:** C4-v6 (`cdf1a546599eee553d6bf74c022b8545ffe9d5a5`)

## Scientific question
C4-v5/v6 showed that the round-conditioned U+Ecal detector contains poisoning signal, but scalar trust weighting did not convert that signal into competitive IDS utility. C4-v7 tests a different, pre-specified mechanism: **use explanation-aware trust only as a clean-calibrated client prefilter, then apply a standard robust aggregator to the retained updates**.

This is a mechanism diagnostic, not a paper-final experiment.

## Frozen configuration
Inherit C4-v6 without change:
- seed = 31415
- attack = sign_flip
- malicious fraction beta = 0.20
- rounds = 3
- clients = 50
- Dirichlet alpha = 0.10
- per-client cap = 12000
- pretraining epochs = 3 (development limitation)
- FedProx mu = 0.01
- reference size = 128
- IG steps = 16
- dataset SHA-256 = `17e098c1cee713f5b1af79b918ae7ba59611c30a3bc609237d7afa7e9a97f8b1`

## Detector freeze
The detector/calibration path is unchanged from C4-v6. The principal trust signal is `U+Ecal`, fitted from a clean shadow batch at the start of each evaluated round. No attacked labels enter calibration.

Let `s_i` be the attacked-round U+Ecal trust score and let

`tau = Q_0.05({s_j^clean})`

be the fifth percentile of clean-shadow U+Ecal scores for that same round.

The retained set is frozen as

`S = { i : s_i >= tau }`.

Thus the clean-derived detector threshold is used only for prefiltering; there is no soft trust multiplication in the principal C4-v7 arms.

### Safety/fallback rule
If fewer than 3 clients are retained, the prefilter arm falls back to the **three highest U+Ecal scores**, with ties resolved by original client order. This rule is fixed before attacked outcomes and uses no malicious labels.

## Frozen aggregation arms
All arms receive the exact same immutable attacked `RoundBatch` within each round.

Controls retained:
- FedAvg
- Coordinate Median
- Trimmed Mean (trim ratio 0.20)
- Krum
- Multi-Krum
- historical hard XTrust-FL
- C4-v6 relative-soft and relative-soft+clip controls

New C4-v7 arms:
1. **XTrust-Prefilter-Median (principal):** retain `S`, then coordinate-wise median over retained updates.
2. **XTrust-Prefilter-Trimmed (secondary):** retain `S`, then coordinate-wise trimmed mean with frozen trim ratio 0.20. If the retained set is too small for a non-empty 20% trim, use the coordinate-wise mean over the retained set; this deterministic fallback is recorded.

The robust stage does **not** use malicious labels, beta, or attacked-outcome threshold tuning.

## Canonical trajectory
The next-round base model is the **principal XTrust-Prefilter-Median** candidate. Therefore rounds 2–3 are on the C4-v7 principal trajectory. Within every round, all candidate arms remain paired on the same attacked updates. Cross-version rounds 2–3 are trajectory-dependent and must not be treated as paired replications of C4-v4/v5/v6.

## Required diagnostics
For every round record:
- unchanged detector TP/FP/TN/FN, TPR/FPR/AUROC for U, Eraw, Ecal, U+Ecal, U+Ecal+T;
- clean-derived `tau`;
- retained and excluded client IDs/counts;
- retrospective benign/malicious retained counts **for analysis only, never for decisions**;
- retained sample mass and update-norm summaries;
- utility metrics for every aggregation arm;
- round timing and provenance.

## Interpretation rule
C4-v7 is **promising** only if the principal prefilter+median arm preserves useful detector behavior and improves utility relative to historical hard/soft XTrust controls without merely duplicating the unfiltered Coordinate Median. A single seed cannot establish statistical superiority.

If the principal arm fails to improve meaningfully, stop mechanism tuning on this attacked seed and redesign the final study rather than searching thresholds on C4.

## Prohibited after freeze
- no post-hoc change to the 5th-percentile threshold;
- no malicious-label calibration;
- no tuning trim ratio on attacked outcomes;
- no changing the principal arm after seeing results;
- no extra seed until this diagnostic is inspected;
- no sophisticated/adaptive/colluding attack in this diagnostic;
- no paper-final or significance claim from C4-v7.
