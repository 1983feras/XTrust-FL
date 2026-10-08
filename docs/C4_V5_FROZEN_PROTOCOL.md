# C4-v5 Frozen Diagnostic Protocol

Status: **FROZEN BEFORE ATTACKED C4-v5 OUTCOMES**  
Paper-final: **false**

## Question
Can the C4-v4 round-conditioned detector retain its detection signal while recovering IDS utility when trust is used as a bounded soft aggregation weight rather than a hard rejection gate?

## Fixed experimental configuration
C4-v5 inherits the C4-v4 diagnostic configuration: seed 31415, sign-flip attack, beta=0.20, 3 rounds, 50 clients, Dirichlet alpha=0.1, cap=12000, 3 centralized development pretraining epochs, FedProx mu=0.01, reference size=128, and IG steps=16. Dataset SHA-256 must equal `17e098c1cee713f5b1af79b918ae7ba59611c30a3bc609237d7afa7e9a97f8b1`.

The attacked RoundBatch is immutable within a round and is reused by every aggregation method.

## Detector freeze
The round-conditioned CLEAN calibration and detector arms are unchanged from C4-v4: U, Eraw, Ecal, U+Ecal, U+Ecal+T. The principal aggregation trust score is the C4-v4 U+Ecal score. Clean q05 thresholds remain detection diagnostics only; they are not used to reject clients in the primary C4-v5 soft arms. Malicious labels are used only after scoring to compute diagnostic TP/FP/TN/FN/TPR/FPR/AUROC.

## Frozen soft mapping
Let `s_i` be the already-bounded U+Ecal trust score. Freeze `q_min = 0.25` as a conservative nonzero floor, chosen a priori rather than from attacked outcomes:

`q_i = q_min + (1-q_min) * clip(s_i,0,1)`.

Thus every client retains nonzero influence while low-trust clients are downweighted. No attacked labels enter q_i.

## Frozen clipping
For the clipping arm only, use the repository's pre-existing robust threshold computed from the attacked update norms without labels:

`C = median(norms) + 2.5 * 1.4826 * MAD(norms)`.

Each update is transformed as

`Delta_i_clipped = Delta_i * min(1, C/(||Delta_i||_2 + 1e-12))`.

No-clipping arm uses the original update.

## Aggregation
For both soft arms:

`Delta_G = sum_i n_i q_i Delta_i* / sum_i n_i q_i`,

where Delta_i* is original for soft-no-clip and clipped for soft+clip. The denominator must be finite and strictly positive.

## Frozen arms
Baselines/control: FedAvg, Coordinate Median, Trimmed Mean (0.2), Krum, Multi-Krum, and the historical C4-v4 hard XTrust policy. New arms: `xtrust_soft` and `xtrust_soft_clip`.

## Diagnostics
For every detector arm: TP, FP, TN, FN, TPR, FPR, detection AUROC. For every aggregation method: accuracy, Macro-F1, precision, recall, AUROC, AUPRC, confusion matrix. For soft arms record client trust score, q_i, sample count, update norm before/after clipping, clipping factor, effective benign/malicious weight mass (labels used only for retrospective reporting), and total weight mass. Record calibration, batch, explanation, aggregation and round timing.

## Success interpretation
This single seed is diagnostic only. C4-v5 is promising if soft aggregation materially improves XTrust utility relative to the C4-v4 hard control while detector discrimination/FPR are preserved because the detector itself is frozen, and the soft policy still meaningfully downweights low-trust clients rather than degenerating to FedAvg. No significance claim is permitted. Failure or mixed evidence triggers structural review before any multi-seed confirmation.

## Prohibitions
No threshold search, malicious-label calibration, attacked-batch hyperparameter optimization, silent arm selection, post-hoc q_min change, detector tuning, sophisticated new attack, or automatic multi-seed run. Do not call C4-v5 paper-final.

## Stop rule
Run seed 31415 once under this frozen protocol, package protocol/code/tests/results/provenance, and STOP for independent inspection before any final freeze or multi-seed confirmation.
