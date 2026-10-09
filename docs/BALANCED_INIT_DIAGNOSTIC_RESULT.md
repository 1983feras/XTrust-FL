# Balanced Federated Initialization Diagnostic — Frozen Outcome

This record documents the completed pre-outcome protocol in `docs/BALANCED_INIT_DIAGNOSTIC_PROTOCOL.md`.

## Result
- Dataset SHA-256: `17e098c1cee713f5b1af79b918ae7ba59611c30a3bc609237d7afa7e9a97f8b1`
- seed: 42
- clients: 50
- alpha: 0.10
- class-weight formula: `N/(2*N_c)`
- clean training counts: `[23910, 619224]`
- resulting weights: `[13.449058532714844, 0.5193064212799072]`
- **first_adequate_round: 6**

At round 6, validation Macro-F1 was `0.7233933331339387` and validation AUROC was `0.9750485729206485`; the confusion matrix was `[[1210,2206],[606,87855]]`. The unchanged gate was therefore satisfied. Test Macro-F1 at that round was `0.7212806365634388`; test results did not participate in selection.

Later rounds produced higher Macro-F1 values, but they are explicitly not eligible for warm-up selection because the frozen protocol requires the earliest passing round. Therefore **6 clean balanced federated warm-up rounds** is the only admissible candidate for a revised final-study protocol.

## Interpretation
The previous unweighted diagnostic failed to pass the gate within 20 rounds despite high AUROC. The balanced-loss diagnostic passed at round 6 under otherwise matched primary initialization settings. This supports the pre-specified hypothesis that severe class imbalance materially contributed to the decision-boundary collapse. It does not establish any XTrust poisoning-defense claim because no poisoning or XTrust filtering was used in this diagnostic.

## Next step
Freeze a revised Final Study protocol before further outcomes. The revision may incorporate only the predetermined global training-split class weights and the predetermined six-round clean federated warm-up. XTrust detector, clean-derived prefilter, Q0.05 rule, attack grid, seeds, alpha, baselines, ablations, and anti-tuning rules remain unchanged unless a separately justified pre-outcome protocol explicitly states otherwise.