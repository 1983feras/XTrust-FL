# XTrust-FL Final Study v1 — Execution Freeze

**Scientific protocol:** `docs/FINAL_STUDY_V1_FROZEN_PROTOCOL.md`  
**Validated execution commit:** `a27bd330e314efc6a41b942f68cf6cef94461689`  
**Dataset SHA-256:** `17e098c1cee713f5b1af79b918ae7ba59611c30a3bc609237d7afa7e9a97f8b1`

## Validation gate
The validated execution candidate was checked in the user execution environment before any paper-final scientific run. The reported gate passed:
- exact candidate commit;
- no tracked source-code modifications;
- exact dataset SHA-256;
- syntax checks;
- frozen-protocol guardrails;
- complete regression suite;
- final-study mechanics smoke test.

Local `data/` and `PROVENANCE.json` are permitted runtime artifacts and are not source-code changes.

## Freeze rule
Paper-final scientific conditions MUST execute the code at commit `a27bd330e314efc6a41b942f68cf6cef94461689` with the frozen protocol configuration. This documentation commit does not change executable code and MUST NOT replace the execution SHA recorded in result provenance.

Any future executable-code correction invalidates this execution freeze and requires a new validation gate before further paper-final runs. Scientific thresholds, seeds, beta grid, principal arm, and anti-tuning rules remain frozen regardless of outcomes.

## First confirmatory condition
The first paper-final condition is fixed as:
- seed: 42
- beta: 0.00
- alpha: 0.10
- rounds: 3
- warm-up rounds: 3
- clients: 50
- cap: 12000
- FedProx mu: 0.01
- reference size: 128
- IG steps: 16

This beta=0 condition is intentionally first because it measures benign Non-IID utility and false exclusion without poisoning. No threshold or method selection may be changed after observing it.