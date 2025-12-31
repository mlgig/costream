import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.utils.validation import check_array, check_is_fitted

class PipingClassifier(BaseEstimator, ClassifierMixin):
    """
    A physics-aware PIPING classifier that extracts structural and energetic features
    from time series (specifically suited for Fall Detection magnitude data).
    
    Default Backend: ExtraTreesClassifier(n_estimators=150, random_state=42)
    """
    def __init__(self, scales=[5, 20], estimator=None):
        self.scales = scales
        self.estimator = estimator

    def _get_pip_indices(self, y, k):
        """Standard deterministic greedy PIP extraction."""
        n = len(y)
        if k >= n: return np.arange(n)
        
        pip_indices = [0, n-1]
        x = np.arange(n)
        
        for _ in range(k - 2):
            max_dist = -1; max_idx = -1
            pip_indices.sort()
            for i in range(len(pip_indices) - 1):
                start, end = pip_indices[i], pip_indices[i+1]
                if end - start > 1:
                    p1 = (start, y[start]); p2 = (end, y[end])
                    slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
                    c = p1[1] - slope * p1[0]
                    xs = x[start+1 : end]; ys = y[start+1 : end]
                    dists = np.abs(ys - (slope * xs + c))
                    curr_max = np.max(dists)
                    if curr_max > max_dist:
                        max_dist = curr_max; max_idx = xs[np.argmax(dists)]
            if max_idx != -1: pip_indices.append(max_idx)
        
        pip_indices.sort()
        return np.array(pip_indices)

    def _extract_physics_features(self, y_raw, indices):
        """
        Extracts PIPs plus Energy, Complexity (ArcLen), and Dynamics (Jerk).
        """
        y_pips = y_raw[indices]
        t_pips = indices
        
        # 1. Gradients (Angles) - Normalized
        dy = np.diff(y_pips)
        dt = np.diff(t_pips)
        with np.errstate(divide='ignore', invalid='ignore'):
            angles = np.arctan(dy / dt) / (np.pi/2)
            
        segment_energies = []
        segment_arclens = []
        segment_diffs = []
        
        for i in range(len(indices) - 1):
            start, end = indices[i], indices[i+1]
            
            if end - start <= 1:
                # Fallback for adjacent points
                segment_energies.append(np.abs(y_raw[start]))
                segment_arclens.append(0)
                segment_diffs.append(0)
                continue
                
            # Get raw segment data
            raw_seg = y_raw[start : end+1]
            
            # Feature A: Kinetic Energy (Mean Magnitude)
            # Differentiates a "hard" impact from a "soft" movement
            energy = np.mean(np.abs(raw_seg))
            segment_energies.append(energy)
            
            # Feature B: Arc Length (Frequency/Vibration proxy)
            # Differentiates a "thud" from a "clatter"
            d_y = np.diff(raw_seg)
            seg_len = np.sum(np.sqrt(1 + d_y**2))
            segment_arclens.append(seg_len / (end - start))
            
            # Feature C: Jerk (Volatility)
            # Differentiates controlled vs uncontrolled motion
            mean_jerk = np.mean(np.abs(np.diff(raw_seg)))
            segment_diffs.append(mean_jerk)
            
        return np.concatenate([
            y_pips, 
            angles, 
            np.array(segment_energies), 
            np.array(segment_arclens),
            np.array(segment_diffs)
        ])

    def _transform_row(self, row):
        # row is 1D array (Magnitude)
        all_scale_features = []
        
        for k in self.scales:
            indices = self._get_pip_indices(row, k)
            feats = self._extract_physics_features(row, indices)
            
            # Add context: Where in time did this happen?
            t_norm = indices / len(row)
            
            scale_feat = np.concatenate([feats, t_norm])
            all_scale_features.append(scale_feat)
            
        return np.concatenate(all_scale_features)

    def fit(self, X, y):
        X = check_array(X)
        X_feat = np.array([self._transform_row(row) for row in X])
        
        # Configure Default Estimator if None provided
        if self.estimator is None:
            self.estimator_ = ExtraTreesClassifier(
                n_estimators=150, 
                random_state=42,
                n_jobs=-1 # Use all cores for speed
            )
        else:
            self.estimator_ = clone(self.estimator)
            
        self.estimator_.fit(X_feat, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        check_is_fitted(self, ['estimator_'])
        X = check_array(X)
        X_feat = np.array([self._transform_row(row) for row in X])
        return self.estimator_.predict(X_feat)

    def predict_proba(self, X):
        """
        Returns probability estimates for the test data X.
        """
        check_is_fitted(self, ['estimator_'])
        X = check_array(X)
        X_feat = np.array([self._transform_row(row) for row in X])
        
        if hasattr(self.estimator_, "predict_proba"):
            return self.estimator_.predict_proba(X_feat)
        else:
            raise AttributeError(f"Underlying estimator {type(self.estimator_)} has no predict_proba method.")