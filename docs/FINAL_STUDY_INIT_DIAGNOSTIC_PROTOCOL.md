# XTrust-FL — Clean Federated Initialization Diagnostic

Status: **FROZEN BEFORE DIAGNOSTIC OUTCOMES**

This diagnostic follows the invalid confirmatory clean-control execution at commit `a27bd330e314efc6a41b942f68cf6cef94461689`, where the clean federated warm-up produced a degenerate argmax classifier. No poisoned condition beyond beta=0 was executed.

## Purpose
Determine whether the failure is insufficient convergence of the clean federated warm-up, rather than an XTrust detector/aggregation effect.

## Scientific isolation
This diagnostic contains **no poisoning**, **no XTrust filtering**, **no Q0.05 threshold**, and **no attack labels**. It evaluates only benign FedAvg warm-up under the same primary Non-IID partition.

## Frozen data/configuration
- dataset SHA-256: `17e098c1cee713f5b1af79b918ae7ba59611c30a3bc609237d7afa7e9a97f8b1`
- seed: 42
- clients: 50
- Dirichlet alpha: 0.10
- per-client cap: 12000
- FedProx mu: 0.01
- local epochs: 1
- local batch size: 512
- local learning rate: 1e-3
- optimizer/loss: existing repository `local_train` (Adam + unweighted CrossEntropyLoss)
- aggregation: FedAvg weighted by client sample counts
- preprocessing/splits: identical to Final Study v1
- maximum diagnostic warm-up rounds: 20

The diagnostic must evaluate the same untouched validation and test sets after initialization and after every clean federated round. Test metrics are recorded for diagnosis only and MUST NOT determine the stopping round or any hyperparameter.

## Pre-specified adequacy gate
The first round satisfying **all** validation-only criteria is recorded as `first_adequate_round`:
1. validation predictions contain both classes;
2. validation confusion matrix has non-zero true negatives and non-zero true positives;
3. validation macro-F1 >= 0.70;
4. validation AUROC >= 0.90.

These thresholds are an initialization-validity gate, not a defense-performance target. The diagnostic continues through all 20 rounds so the first adequate round is not selected by test performance.

## Interpretation
- If no round passes the gate, do not run attacked Final Study conditions. Revisit initialization/training design in a new pre-outcome protocol.
- If a round passes, the earliest passing round is the only candidate warm-up length for a revised Final Study protocol. It must be frozen and revalidated before any attacked condition.

## Anti-tuning
Do not change the gate, optimizer, loss, learning rate, local epochs, client cap, alpha, seed, or aggregation after observing this diagnostic. Do not select a later round because it has better test Macro-F1. No XTrust scientific parameter may be changed from this diagnostic.