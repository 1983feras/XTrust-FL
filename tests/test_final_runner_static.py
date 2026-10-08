from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUNNER=(ROOT/'scripts/run_final_study_v1.py').read_text(encoding='utf-8')
PROTO=(ROOT/'docs/FINAL_STUDY_V1_FROZEN_PROTOCOL.md').read_text(encoding='utf-8')

def test_runner_marks_only_protocol_conforming_runs_paper_final():
    assert "'paper_final':True" in RUNNER
    assert "FINAL_SEEDS" in RUNNER and "FINAL_BETAS" in RUNNER

def test_runner_uses_federated_warmup_not_centralized_pretrain():
    assert 'clean_federated_warmup' in RUNNER
    assert '--pretrain-epochs' not in RUNNER

def test_runner_has_independent_named_trajectories():
    for name in ('fedavg','coordinate_median','trimmed_mean','krum','multi_krum','fltrust','xtrust_u_median','xtrust_ecal_median','xtrust_uecal_median','xtrust_uecal_trimmed'):
        assert name in RUNNER
    assert 'models={k:deepcopy(warm) for k in METHODS}' in RUNNER

def test_principal_frozen_prefilter_constants_are_imported():
    assert 'PREFILTER_QUANTILE' in RUNNER
    assert 'MIN_RETAINED' in RUNNER
    assert 'clean_threshold_prefilter' in RUNNER

def test_protocol_contains_anti_tuning_guardrails():
    assert 'do not alter Q0.05' in PROTO
    assert 'do not replace the frozen seeds' in PROTO
