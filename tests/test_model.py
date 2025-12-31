import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from costream.model import CostClassifierCV, cost_score

def test_cost_score():
    # TN=5, FP=1, FN=2, TP=2
    y_true = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 0, 0, 1, 0, 0, 1, 1]
    # FP=1, FN=2. Alpha=2. Gain = -(1 + 2*2) = -5
    gain = cost_score(y_true, y_pred, alpha=2)
    assert gain == -5

def test_cost_classifier_fit():
    # Generate simple imbalance data
    X, y = make_classification(n_samples=100, weights=[0.9, 0.1], random_state=42)
    
    clf = CostClassifierCV(
        base_estimators=[LogisticRegression()],
        alpha=2.0,
        n_dirichlet=5, # fast
        n_thresholds=10,
        cv=2,
        method="dirichlet",
        calibration=None # simplify
    )
    
    clf.fit(X, y)
    
    assert hasattr(clf, "weights_")
    assert hasattr(clf, "threshold_")
    assert hasattr(clf, "optimization_curve_")
    
    # Check predictions format
    preds = clf.predict(X)
    assert preds.shape == y.shape
    probs = clf.predict_proba(X)
    assert probs.shape == (100, 2)