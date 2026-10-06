# XTrust-FL

**Heterogeneity-Calibrated Explanation-Aware Defense against Stealthy Poisoning in Federated IoT Intrusion Detection**

XTrust-FL is a reproducible research codebase for studying whether server-side explanation inconsistency can provide a useful poisoning-detection signal in federated IoT intrusion detection, especially under severe non-IID heterogeneity and stealthy model-poisoning attacks.

## Research hypothesis

The central hypothesis is that a malicious client may remain close to benign clients in update space while still inducing abnormal feature-attribution behavior on a trusted server-side reference set. XTrust-FL therefore combines:

1. robust update-space anomaly signals,
2. server-computed explanation inconsistency,
3. heterogeneity calibration to reduce false positives for benign non-IID clients,
4. temporal trust smoothing,
5. trust-weighted clipped aggregation.

This repository does **not** claim that the hypothesis is proven. All paper claims must be supported by repeated experiments and statistical analysis.

## Planned evaluation

- Primary dataset: CICIoT2023
- Secondary validation: ToN-IoT
- Binary and multiclass intrusion-detection tasks
- 20 clients for debugging/ablation, 50 clients for final experiments
- Dirichlet non-IID: alpha in {10, 1.0, 0.5, 0.1}
- Malicious-client fractions: beta in {0, 0.1, 0.2, 0.3, 0.4}
- Five random seeds: 42, 77, 100, 999, 2026

## Baselines

- Centralized
- Local-only
- FedAvg
- FedProx
- Coordinate Median
- Trimmed Mean
- Krum / Multi-Krum
- FLTrust (planned as a fairness-critical trusted-reference baseline)

## Attacks

- Label flipping
- Sign flipping
- Model scaling
- Backdoor
- Stealth poisoning constrained to resemble benign update statistics

## Current repository stage

The first milestone is a minimum reproducible experiment (MRE) with a lightweight PyTorch implementation. Synthetic data may be used only for smoke testing. Synthetic smoke-test metrics must never be reported as scientific results.

## Project structure

```text
configs/
src/xtrust_fl/
scripts/
tests/
results/
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Smoke test

```bash
python scripts/smoke_test.py
```

## Scientific integrity

- No fabricated results.
- No normalization before train/validation/test splitting.
- Test data are never used for calibration or model selection.
- All final experiments must record configuration, seed, client partition, malicious-client identities, per-round metrics, environment details, and model checkpoints.
- Statistical significance is reported only after an actual statistical test.

## License

License not yet selected.
