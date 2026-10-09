# XTrust-FL Final Study v2 — Frozen Confirmatory Protocol

Status: **FROZEN BEFORE ANY v2 PAPER-FINAL OUTCOME**

## Why v2 exists
Final Study v1 was stopped at its first clean-control condition (seed 42, beta=0) because the unweighted three-round federated warm-up produced a degenerate majority-class decision rule. No attacked v1 condition was run. Two clean-only, non-XTrust diagnostics were then conducted under pre-specified protocols. The unweighted 20-round diagnostic did not pass the initialization gate. A deterministic class-balanced diagnostic passed the unchanged validation-only gate first at round 6. Therefore v2 incorporates only those clean-data-justified initialization corrections.

## Dataset
CICIoT2023 cleaned parquet, exact SHA-256:
`17e098c1cee713f5b1af79b918ae7ba59611c30a3bc609237d7afa7e9a97f8b1`

## Primary federated setting
- clients: 50
- primary Dirichlet alpha: 0.10
- per-client cap: 12000
- FedProx mu: 0.01
- local epochs: 1
- local batch size: 512
- local learning rate: 1e-3
- full client participation unless the inherited v1 protocol explicitly specifies otherwise

## Frozen v2 initialization correction
The clean global training split determines binary class counts `N_c`. The same deterministic global class weights are supplied to every client's CrossEntropyLoss:

`w_c = N / (2*N_c)`.

No validation/test labels are used to calculate weights. Weights are fixed for the run, not client-specific and not round-specific.

Every method starts from the same **six-round clean balanced federated warm-up**. Six is not selected for best utility: it is the earliest round that passed the previously frozen validation-only adequacy gate in the clean diagnostic.

## XTrust mechanism retained
The principal mechanism remains:
1. round-conditioned clean calibration;
2. update anomaly U plus heterogeneity-calibrated explanation residual Ecal;
3. clean-derived Q0.05 prefilter;
4. coordinate-median aggregation over retained clients.

No XTrust detector threshold or fusion parameter is retuned because of the initialization correction.

## Baselines retained
- FedAvg
- Coordinate Median
- Trimmed Mean (0.20)
- Krum
- Multi-Krum
- FLTrust-style baseline

## Ablations retained
- Coordinate Median alone
- U-only prefilter + Median
- Ecal-only prefilter + Median
- U+Ecal prefilter + Median (principal)
- U+Ecal prefilter + Trimmed Mean

## Confirmatory grid retained
- seeds: `[42, 77, 100, 999, 2026]`
- malicious fractions beta: `{0.00, 0.10, 0.20, 0.30, 0.40}`
- primary alpha: `0.10`
- confirmatory attack: sign flipping
- heterogeneity sweep only after the primary confirmatory grid according to the inherited v1 study plan

## Execution order and validity gates
1. Revalidate code/regression/smoke at the exact frozen execution commit.
2. Run only seed 42, beta=0.00 first.
3. The clean control must show a non-degenerate classifier after the six-round balanced warm-up: both predicted classes, TN>0, TP>0, validation Macro-F1 >=0.70, validation AUROC >=0.90.
4. If the clean control fails, STOP. Do not run beta>0.
5. If it passes, freeze/record the clean-control result, then execute the remaining predetermined grid without tuning.

## Pairing and trajectories
Each defense maintains its own independent global-model trajectory. Within a matched method/condition/round, RNG construction remains deterministic and paired where required by the inherited v1 design. Competing aggregators must not be allowed to share a model trajectory after aggregation.

## Metrics
Primary utility metric: Macro-F1. Also record accuracy, precision, recall, AUROC, AUPRC and confusion matrix. Detection reports TPR, FPR and AUROC where both benign and malicious clients exist. beta=0 detection AUROC is undefined/non-inferential and must not be presented as attack-detection evidence.

## Statistics
Across five seeds report mean, SD and 95% CI, plus paired comparisons where justified. With n=5, explicitly state inferential limitations. Do not claim statistical significance unless the frozen analysis supports it.

## Efficiency
Do not compare baseline wall-clock time if it includes XTrust calibration/XAI work. Report XTrust detector/explanation overhead separately or use timing scopes that isolate each method's actual computation.

## Anti-tuning
After the first v2 paper-final execution begins, do not change warm-up length, class-weight formula, Q0.05, fusion rule, seeds, beta values, alpha, attack, aggregation, or adequacy gate in response to outcomes. Any implementation defect must be documented and corrected under a new explicit revision before further confirmatory execution.

## Claim discipline
The clean diagnostics establish only initialization validity, not XTrust superiority. XTrust defense claims require the final multi-seed attacked results. Utility superiority, robustness superiority, and significance claims may be made only if supported by those final results.