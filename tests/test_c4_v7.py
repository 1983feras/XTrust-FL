import numpy as np
import pytest
from xtrust_fl.aggregate import clean_threshold_prefilter


def test_prefilter_uses_clean_q05_only():
    clean=np.array([.50,.60,.70,.80,.90])
    attacked=np.array([.40,.55,.75,.95])
    keep,d=clean_threshold_prefilter(attacked,clean,quantile=.05,min_retained=2)
    assert d['threshold']==pytest.approx(np.quantile(clean,.05))
    assert keep.tolist()==[1,2,3]
    assert not d['fallback_used']


def test_prefilter_fallback_is_deterministic_top_scores():
    clean=np.array([.8,.85,.9,.95])
    attacked=np.array([.1,.4,.3,.2])
    keep,d=clean_threshold_prefilter(attacked,clean,quantile=.05,min_retained=3)
    assert d['fallback_used']
    assert keep.tolist()==[1,2,3]


def test_prefilter_stable_tie_break():
    clean=np.array([.9,.95])
    attacked=np.array([.2,.5,.5,.1])
    keep,d=clean_threshold_prefilter(attacked,clean,quantile=.05,min_retained=2)
    assert d['fallback_used']
    assert keep.tolist()==[1,2]


def test_prefilter_rejects_invalid_inputs():
    with pytest.raises(ValueError): clean_threshold_prefilter(np.array([1.,np.nan]),np.array([.8]),min_retained=1)
    with pytest.raises(ValueError): clean_threshold_prefilter(np.array([1.]),np.array([]),min_retained=1)
    with pytest.raises(ValueError): clean_threshold_prefilter(np.array([1.]),np.array([.8]),quantile=1.1,min_retained=1)
    with pytest.raises(ValueError): clean_threshold_prefilter(np.array([1.]),np.array([.8]),min_retained=2)
