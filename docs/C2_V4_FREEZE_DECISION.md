# C2-v4 Freeze Decision

## Status
Pre-registered freeze decision after development-only C2-v4 ablation and before confirmation seeds.

## Development seeds
314, 2718, 8086. These seeds are development-only and MUST NOT be reused as confirmation evidence.

## Observed development result
Across 75 held-out benign evaluation clients (25 per seed):

| Variant | Pooled FP | Pooled FPR |
|---|---:|---:|
| Raw explanation anomaly | 12/75 | 16.00% |
| Loss-only calibration | 4/75 | 5.33% |
| Loss + feature shift | 5/75 | 6.67% |
| Loss + feature shift + quantity | 4/75 | 5.33% |

Per-seed loss-only FPR: seed 314: 4% vs raw 28%; seed 2718: 0% vs raw 20%; seed 8086: 12% vs raw 0%.

These are descriptive development results only. They do not establish statistical significance or a general FPR-reduction claim.

## Frozen candidate
Use **reference-loss-only heterogeneity calibration** for confirmation:

R_i^exp = max(0, D_i^exp - Dhat_i^exp(L_i^ref)).

Rationale: it matches the best pooled development FPR while remaining parsimonious and server-computable on the trusted reference set. Feature-mean shift is excluded from the frozen deployable candidate because it adds no development benefit and requires information about local client data.

## Confirmation seeds (fixed before execution)
17, 271, 1618, 4099, 12345.

## Frozen C2 confirmation configuration
- dataset: exact public CICIoT2023 cleaned parquet with SHA-256 verification
- alpha: 0.1
- clients: 50
- pretrain epochs: 3
- local epochs: 1
- client cap: 12000
- reference size: 128 (pilot confirmation; reference-size sensitivity is a separate experiment)
- IG steps: 16
- calibration/evaluation: deterministic 25/25 split per seed
- explanation prototype and anomaly scaling: calibration clients only
- expected anomaly: clipped to non-negative values
- compared methods: Raw vs Loss-only only
- no parameter/model redesign after seeing confirmation outcomes

## Primary confirmation endpoint
Paired per-seed difference in benign held-out FPR:

DeltaFPR_s = FPR_loss-only,s - FPR_raw,s.

Report each seed, mean and SD of DeltaFPR, pooled FP/FPR as descriptive secondary summaries, and a 95% CI for the paired mean difference. With only five seeds, avoid strong distributional claims; report an exact paired sign/permutation-style result or Wilcoxon when informative, alongside the effect size. No claim of improvement is permitted solely from pooled client counts.

## Decision rule
- If the confirmation effect is consistently favorable and uncertainty supports a meaningful reduction, retain the FPR-reduction claim for C2.
- If mixed or uncertain, report C2 as evidence that calibration changes benign false-positive behavior but do not claim universal reduction.
- Do not tune a C2-v5 on confirmation seeds.
- Regardless of C2 outcome, proceed to the poisoning experiment (C4), because a useful defense must reduce benign confusion without suppressing detection of malicious clients.

## Next critical experiment
C4: severe Non-IID + poisoning. Evaluate malicious-client detection (TPR/recall, FPR, precision where meaningful), global IDS utility (Macro-F1/AUROC/AUPRC), and compare against robust FL baselines under identical partitions/seeds.
