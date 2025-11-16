# Tests for phase2 hybrid pipeline
# ml/phase2_hybrid/tests/test_phase2_pipeline.py
from ml.phase2_hybrid.run_train_all import run_all

def test_run_small():
    # run orchestrator (it will early-exit if no events)
    run_all()
    assert True
