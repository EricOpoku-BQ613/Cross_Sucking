#!/usr/bin/env python3
"""
Temperature Scaling for Post-Hoc Calibration
=============================================
Fits optimal temperature T on validation logits to minimize ECE.

Usage:
    python scripts/temperature_scaling.py \
        --predictions runs/ablation_E02/preds_val.csv \
        --output runs/ablation_E02/temperature.json
"""

import argparse
import json
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from pathlib import Path


class TemperatureScaler:
    """Post-hoc temperature scaling for neural network calibration."""
    
    def __init__(self, t_init=1.5, t_min=0.1, t_max=10.0):
        self.temperature = t_init
        self.t_min = t_min
        self.t_max = t_max
        self.fitted = False
        
    def fit(self, logits: np.ndarray, labels: np.ndarray, 
            method: str = 'nll', verbose: bool = True):
        """Fit optimal temperature on validation set."""
        self.logits = logits
        self.labels = labels
        
        objective = self._nll_objective if method == 'nll' else self._ece_objective
        
        if verbose:
            print(f"Fitting temperature using {method.upper()} objective...")
        
        # Grid search + refinement
        temps = np.linspace(self.t_min, self.t_max, 50)
        losses = [objective(t) for t in temps]
        best_idx = np.argmin(losses)
        
        result = minimize_scalar(
            objective,
            bounds=(max(self.t_min, temps[best_idx] - 0.5), 
                    min(self.t_max, temps[best_idx] + 0.5)),
            method='bounded'
        )
        
        self.temperature = result.x
        self.fitted = True
        
        if verbose:
            print(f"\n✅ Optimal temperature: {self.temperature:.4f}")
            print(f"Before - ECE: {self._compute_ece(logits):.4f}")
            print(f"After  - ECE: {self._compute_ece(logits / self.temperature):.4f}")
        
        return self
    
    def _nll_objective(self, temperature: float) -> float:
        scaled_logits = self.logits / temperature
        probs = softmax(scaled_logits, axis=1)
        log_probs = np.log(probs + 1e-10)
        nll = -np.mean(log_probs[np.arange(len(self.labels)), self.labels])
        return nll
    
    def _ece_objective(self, temperature: float) -> float:
        scaled_logits = self.logits / temperature
        return self._compute_ece(scaled_logits)
    
    def _compute_ece(self, logits: np.ndarray, n_bins: int = 15) -> float:
        probs = softmax(logits, axis=1)
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == self.labels).astype(float)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            prop_in_bin = in_bin.mean()
            if prop_in_bin > 0:
                avg_confidence = confidences[in_bin].mean()
                avg_accuracy = accuracies[in_bin].mean()
                ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
        return ece
    
    def transform(self, logits: np.ndarray) -> np.ndarray:
        return logits / self.temperature
    
    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        return softmax(self.transform(logits), axis=1)


def load_from_csv(csv_path: str):
    """Load logits and labels from prediction CSV."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    
    if 'logit_ear' in df.columns and 'logit_tail' in df.columns:
        logits = df[['logit_ear', 'logit_tail']].values
    else:
        raise ValueError("CSV must have logit_ear and logit_tail columns")
    
    label_map = {'ear': 0, 'tail': 1}
    labels = df['true_label'].map(label_map).values
    return logits, labels


def evaluate_calibration(logits, labels, temperature=1.0, n_bins=15):
    """Comprehensive calibration evaluation."""
    scaled_logits = logits / temperature
    probs = softmax(scaled_logits, axis=1)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    gaps = []
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
            gaps.append(abs(avg_accuracy - avg_confidence))
    
    mce = max(gaps) if gaps else 0
    
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1
    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
    
    return {'ece': ece, 'mce': mce, 'brier_score': brier}


def main():
    parser = argparse.ArgumentParser(description='Temperature Scaling')
    parser.add_argument('--predictions', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--method', choices=['nll', 'ece'], default='nll')
    args = parser.parse_args()
    
    print(f"Loading from: {args.predictions}")
    logits, labels = load_from_csv(args.predictions)
    print(f"Loaded {len(labels)} samples, class dist: {np.bincount(labels)}")
    
    before = evaluate_calibration(logits, labels, temperature=1.0)
    print(f"\nBEFORE: ECE={before['ece']:.4f}, Brier={before['brier_score']:.4f}")
    
    scaler = TemperatureScaler()
    scaler.fit(logits, labels, method=args.method)
    
    after = evaluate_calibration(logits, labels, temperature=scaler.temperature)
    print(f"AFTER:  ECE={after['ece']:.4f}, Brier={after['brier_score']:.4f}")
    
    result = {
        'temperature': float(scaler.temperature),
        'method': args.method,
        'before_ece': float(before['ece']),
        'after_ece': float(after['ece']),
        'ece_reduction': float(before['ece'] - after['ece'])
    }
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n Saved to {args.output}")
    print(f"Apply in inference: scaled_logits = logits / {scaler.temperature:.4f}")


if __name__ == '__main__':
    main()