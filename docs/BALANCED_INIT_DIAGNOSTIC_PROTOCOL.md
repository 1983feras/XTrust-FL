# XTrust-FL — Balanced Federated Initialization Diagnostic

Status: **FROZEN BEFORE BALANCED-DIAGNOSTIC OUTCOMES**

## Motivation
The previously frozen clean FedAvg initialization diagnostic (`final_study_init_diagnostic_seed42.json`) reached validation AUROC 0.9898 by round 20 but never passed the initialization adequacy gate; validation Macro-F1 remained about 0.54 because the argmax classifier was strongly biased toward the majority attack class. The previous diagnostic used the existing unweighted CrossEntropyLoss.

This follow-up tests one pre-specified initialization hypothesis only: whether deterministic global class-balanced cross entropy, derived exclusively from the clean training labels, corrects the decision-boundary collapse while preserving the same federated setting.

## Scientific isolation
- no poisoning;
- no XTrust filtering;
- no Q0.05 threshold;
- no malicious labels;
- no test-set tuning;
- no defense hyperparameter changes.

## Frozen configuration
- dataset SHA-256: `17e098c1cee713f5b1af79b918ae7ba59611c30a3bc609237d7afa7e9a97f8b1`
- seed: 42
- clients: 50
- Dirichlet alpha: 0.10
- cap: 12000
- FedProx mu: 0.01
- local epochs: 1
- batch size: 512
- learning rate: 1e-3
- optimizer: Adam (existing training behavior)
- aggregation: sample-count-weighted FedAvg
- maximum rounds: 20
- preprocessing and train/validation/test split: identical to Final Study v1

## Frozen balanced loss
Let `N_c` be the number of examples of class `c` in the **clean global training split before client capping/partition effects are used for optimization**. For binary classification with total `N`, the deterministic class weight is

`w_c = N / (2 * N_c)`.

The same two global weights are supplied to every benign client's CrossEntropyLoss. They are computed once from clean training labels only. They are not estimated from validation/test data and are not tuned per client or per round.

## Adequacy gate
The previous gate is retained unchanged. The first round satisfying all validation-only conditions is `first_adequate_round`:
1. predictions contain both classes;
2. validation confusion matrix has TN > 0 and TP > 0;
3. validation Macro-F1 >= 0.70;
4. validation AUROC >= 0.90.

The run continues through all 20 rounds. Test metrics are diagnostic records only and cannot determine the selected round.

## Decision rule
- If no round passes: do not run attacked Final Study; balanced CE alone is insufficient.
- If at least one round passes: only the earliest passing round may become the candidate warm-up length in a separately frozen revised Final Study protocol.
- A later round cannot be selected because of better validation/test performance.

## Anti-tuning
After observing this run, do not alter class-weight formula, adequacy gate, learning rate, optimizer, local epochs, alpha, cap, seed, or maximum rounds within this diagnostic. Any further initialization design requires a new pre-outcome protocol. XTrust scientific parameters remain untouched.