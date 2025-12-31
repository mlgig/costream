import numpy as np
from costream.evaluation.event_detection import evaluate_recording, iou

def test_iou():
    r1 = range(0, 10)
    r2 = range(5, 15)
    # Intersection 5,6,7,8,9 (5 items). Union 0..14 (15 items) -> 5/15 = 0.33
    assert np.isclose(iou(r1, r2), 1/3)
    
    r3 = range(20, 30)
    assert iou(r1, r3) == 0.0

def test_evaluate_recording_tp():
    # Signal len 1000. Event at 500.
    # Detection at 500.
    ts_len = 1000
    event = 500
    
    # Create fake confidence: 0 everywhere, 1 at 500
    conf = np.zeros(ts_len)
    conf[500] = 1.0
    
    cm, highs, delay = evaluate_recording(
        ts_len, event, conf,
        confidence_thresh=0.5,
        window_size=1.0,
        tolerance=1.0,
        freq=100
    )
    
    tn, fp, fn, tp = cm.ravel()
    
    assert tp == 1
    assert fn == 0
    assert fp == 0 # Should match the event
    assert highs is not None
    assert len(highs) == 1

def test_evaluate_recording_fn():
    # Event at 500. NO detection.
    ts_len = 1000
    event = 500
    conf = np.zeros(ts_len) # All zero
    
    cm, highs, delay = evaluate_recording(ts_len, event, conf)
    tn, fp, fn, tp = cm.ravel()
    
    assert tp == 0
    assert fn == 1
    assert fp == 0
    assert highs is None

def test_evaluate_recording_fp():
    # ADL file (event = -1). Detection at 500.
    ts_len = 1000
    event = -1
    conf = np.zeros(ts_len)
    conf[500] = 1.0
    
    cm, highs, delay = evaluate_recording(ts_len, event, conf)
    tn, fp, fn, tp = cm.ravel()
    
    assert tp == 0
    assert fn == 0 # Can't miss if no event
    assert fp == 1