# C4-v6 Frozen Diagnostic Protocol

Status: **FROZEN BEFORE C4-v6 ATTACKED OUTCOMES**  
Paper-final: **false**

## Question
Can the already-frozen C4-v4 detector signal be converted into useful aggregation influence by calibrating each attacked client trust score relative to the round-specific CLEAN benign score distribution, without changing the detector itself?

## Inherited configuration
Seed 31415; sign-flip; beta=0.20; 3 rounds; 50 clients; Dirichlet alpha=0.1; cap=12000; development centralized pretraining epochs=3; FedProx mu=0.01; reference size=128; IG steps=16. Dataset SHA-256: `17e098c1cee713f5b1af79b918ae7ba59611c30a3bc609237d7afa7e9a97f8b1`.

The attacked RoundBatch is immutable within a round and is reused across all aggregation arms.

## Detector freeze
No detector change is permitted. Round-conditioned CLEAN calibration, U, Eraw, Ecal, U+Ecal and U+Ecal+T are inherited unchanged. The principal aggregation score is U+Ecal. Malicious labels are forbidden from calibration and mapping; they are retrospective diagnostics only.

## C4-v6 mapping hypothesis
Absolute C4-v5 scores occupied a narrow high range, causing soft weights to be nearly uniform. C4-v6 therefore maps trust relative to the CLEAN U+Ecal distribution for the same round.

From CLEAN U+Ecal scores only, compute:

- `m = median(clean_scores)`
- `MAD = median(abs(clean_scores-m))`
- `sigma = max(1.4826*MAD, 1e-6)`

For attacked client score `s_i`, define standardized relative trust:

`z_i = (s_i - m) / sigma`

and frozen bounded influence:

`r_i = sigmoid(z_i) = 1/(1+exp(-clip(z_i,-60,60)))`

`q_i = q_min + (1-q_min)*r_i`, with **q_min=0.25** retained unchanged from C4-v5.

This mapping is fully determined by CLEAN scores plus the attacked client's own detector score. No attacked outcome, malicious identity, TPR/FPR, or utility metric enters it.

## Clipping
Two C4-v6 arms are evaluated: relative soft trust without clipping and relative soft trust with the same pre-existing label-free robust clipping rule used in C4-v5:

`C = median(update_norms) + 2.5*1.4826*MAD(update_norms)`.

No clipping parameter is tuned.

## Aggregation
`Delta_G = sum_i n_i q_i Delta_i* / sum_i n_i q_i`, where Delta_i* is original or clipped according to arm. Denominator must be finite and positive.

## Arms
Keep FedAvg, Coordinate Median, Trimmed Mean, Krum, Multi-Krum, historical hard XTrust, C4-v5 absolute soft and absolute soft+clip as controls. Add only `xtrust_relative_soft` and `xtrust_relative_soft_clip`.

## Required diagnostics
Record CLEAN median/MAD/sigma, attacked trust s_i, z_i, r_i, q_i, sample counts, update norms before/after clipping, clipping factors, total effective weight, retrospective benign/malicious weight mass, detector TP/FP/TN/FN/TPR/FPR/AUROC, utility metrics and timings.

## Integrity tests
Mapping must be finite and bounded; higher s must never produce lower q for fixed calibration; q must remain in [0.25,1]; zero MAD must be safe; equal scores give equal q; aggregation denominator positive; clipping non-increasing; no label argument in calibration/mapping API; deterministic replay; immutable RoundBatch.

## Interpretation and stop rule
This is one diagnostic seed, not confirmatory evidence. Success requires a meaningful utility improvement over C4-v5/hard controls together with non-degenerate differential weighting, without changing detector architecture. Mixed/failing evidence => structural review. Run seed 31415 once and STOP. No multi-seed, threshold search, q_min tuning, mapping slope tuning, detector tuning, or post-hoc arm selection is permitted.
