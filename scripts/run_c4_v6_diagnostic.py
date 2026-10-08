"""C4-v6: frozen C4-v4 detector, clean-relative trust mapping. Single diagnostic seed."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'src'))
# Reuse the frozen C4-v5 experimental pipeline by executing its main after monkey-patching
# aggregation injection is implemented below through a source-compatible wrapper.
import scripts.run_c4_v5_diagnostic as v5
from xtrust_fl.aggregate import soft_trust_aggregate, relative_trust_aggregate

_orig_soft=soft_trust_aggregate
_calls=0
_clean_scores_by_round=[]

# C4-v6 requires clean scores unavailable to v5 aggregate call directly; capture them by
# wrapping fit_round_clean, then add relative arms alongside absolute controls.
_orig_fit=v5.fit_round_clean
def _fit(*args,**kwargs):
    out=_orig_fit(*args,**kwargs)
    _clean_scores_by_round.append(np.asarray(out['arms']['U+Ecal'],float).copy())
    return out
v5.fit_round_clean=_fit

# Wrap candidate construction point by replacing the imported soft aggregator. v5 calls
# it twice per round (no-clip then clip). Return original result while storing relative
# result for a post-run paired diagnostic reconstruction is NOT sufficient for utility,
# so C4-v6 uses the dedicated implementation generated from v5 below when invoked.

if __name__=='__main__':
    raise SystemExit('C4-v6 runner guard: use scripts/run_c4_v6_full.py; this guard prevents accidental partial execution.')
