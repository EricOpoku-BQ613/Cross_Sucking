"""
Evaluation Metrics for Cross-Sucking Detection
===============================================
Section 4.1: Supervised Baseline (Ear vs Tail)
Section 4.2: Open-Set Extension (Teat+Other as OOD)

Usage:
    python eval_metrics.py --predictions results/preds.csv --output results/metrics/
    
Expected CSV format:
    clip_id, true_label, pred_label, prob_ear, prob_tail, [prob_teat, prob_other]
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Metrics
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, confusion_matrix,
    precision_recall_curve, average_precision_score, roc_auc_score,
    roc_curve, precision_score, recall_score, matthews_corrcoef,
    classification_report
)

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# DATA CLASSES FOR STRUCTURED OUTPUT
# ============================================================================

@dataclass
class BinaryMetrics:
    """Section 4.1 metrics for Ear vs Tail classification"""
    macro_f1: float
    ear_f1: float
    tail_f1: float
    balanced_accuracy: float
    ear_recall: float
    tail_recall: float
    ear_precision: float
    tail_precision: float
    mcc: float
    tail_pr_auc: float
    tail_roc_auc: float
    n_ear: int
    n_tail: int
    
@dataclass 
class OODMetrics:
    """Section 4.2 metrics for Open-Set OOD detection"""
    auroc: float
    aupr_ood: float
    fpr_at_95tpr: float
    tpr_at_5fpr: float
    n_id: int
    n_ood: int
    
@dataclass
class SelectiveMetrics:
    """Selective prediction metrics at various coverage levels"""
    coverages: List[float]
    macro_f1s: List[float]
    risks: List[float]  # 1 - accuracy
    
@dataclass
class BootstrapCI:
    """Bootstrap confidence interval"""
    mean: float
    ci_lower: float
    ci_upper: float
    std: float

# ============================================================================
# SECTION 4.1: BINARY CLASSIFICATION METRICS (EAR VS TAIL)
# ============================================================================

def compute_binary_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_prob: np.ndarray,  # probability of positive class (tail)
    pos_label: int = 1   # tail = 1, ear = 0
) -> BinaryMetrics:
    """
    Compute all Section 4.1 metrics for binary ear vs tail classification.
    
    Args:
        y_true: Ground truth labels (0=ear, 1=tail)
        y_pred: Predicted labels
        y_prob: Predicted probability for tail class
    """
    # Core metrics
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    ear_f1 = f1_score(y_true, y_pred, pos_label=0)
    tail_f1 = f1_score(y_true, y_pred, pos_label=1)
    
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    ear_recall = recall_score(y_true, y_pred, pos_label=0)
    tail_recall = recall_score(y_true, y_pred, pos_label=1)
    
    ear_precision = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    tail_precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # PR-AUC and ROC-AUC for tail (minority class)
    tail_pr_auc = average_precision_score(y_true, y_prob)
    tail_roc_auc = roc_auc_score(y_true, y_prob)
    
    return BinaryMetrics(
        macro_f1=macro_f1,
        ear_f1=ear_f1,
        tail_f1=tail_f1,
        balanced_accuracy=balanced_acc,
        ear_recall=ear_recall,
        tail_recall=tail_recall,
        ear_precision=ear_precision,
        tail_precision=tail_precision,
        mcc=mcc,
        tail_pr_auc=tail_pr_auc,
        tail_roc_auc=tail_roc_auc,
        n_ear=int((y_true == 0).sum()),
        n_tail=int((y_true == 1).sum())
    )

def compute_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    labels: List[str] = ['ear', 'tail']
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns raw and normalized confusion matrices."""
    cm_raw = confusion_matrix(y_true, y_pred)
    cm_norm = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True)
    return cm_raw, cm_norm

def compute_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Expected Calibration Error (ECE) and reliability diagram data.
    
    Returns:
        ece: Expected Calibration Error
        bin_accuracies: Accuracy in each bin
        bin_confidences: Mean confidence in each bin  
        bin_counts: Number of samples in each bin
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    for i in range(n_bins):
        in_bin = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            bin_accuracies[i] = y_true[in_bin].mean()
            bin_confidences[i] = y_prob[in_bin].mean()
            bin_counts[i] = in_bin.sum()
    
    # ECE = weighted average of |accuracy - confidence| per bin
    ece = np.sum(bin_counts * np.abs(bin_accuracies - bin_confidences)) / y_true.shape[0]
    
    return ece, bin_accuracies, bin_confidences, bin_counts

# ============================================================================
# SECTION 4.2: OPEN-SET / OOD DETECTION METRICS
# ============================================================================

def compute_ood_scores(
    probs: np.ndarray,
    method: str = 'msp'
) -> np.ndarray:
    """
    Compute OOD scores from model outputs.
    Higher score = more likely to be ID (in-distribution).
    
    Args:
        probs: Softmax probabilities [N, num_classes]
        method: 'msp' (max softmax prob), 'entropy', 'energy'
    """
    if method == 'msp':
        # Maximum softmax probability - higher = more confident = more ID
        return probs.max(axis=1)
    
    elif method == 'entropy':
        # Negative entropy - higher = less uncertain = more ID
        entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1)
        return -entropy  # negate so higher = more ID
    
    elif method == 'energy':
        # Energy score (Logsumexp of logits)
        # Note: This requires logits, not probs. Approximate from probs.
        logits = np.log(probs + 1e-8)
        return np.log(np.sum(np.exp(logits), axis=1))
    
    else:
        raise ValueError(f"Unknown OOD method: {method}")

def compute_ood_metrics(
    id_scores: np.ndarray,
    ood_scores: np.ndarray
) -> OODMetrics:
    """
    Compute OOD detection metrics.
    
    Args:
        id_scores: OOD scores for in-distribution samples (ear/tail)
        ood_scores: OOD scores for out-of-distribution samples (teat/other)
    
    Convention: Higher score = more ID-like
    For OOD detection: we want to detect low scores as OOD
    """
    # Labels: ID=1, OOD=0 (we want to distinguish them)
    # But for AUROC where OOD is "positive", we flip
    y_true = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    scores = np.concatenate([id_scores, ood_scores])
    
    # AUROC: probability that ID sample has higher score than OOD sample
    auroc = roc_auc_score(y_true, scores)
    
    # For AUPR-OOD, we want OOD as positive class
    # So we negate scores and flip labels
    y_true_ood_pos = 1 - y_true  # OOD=1, ID=0
    scores_neg = -scores  # higher = more OOD
    aupr_ood = average_precision_score(y_true_ood_pos, scores_neg)
    
    # FPR@95%TPR: False positive rate when 95% of OOD is detected
    # TPR = OOD detection rate, FPR = ID incorrectly flagged as OOD
    fpr, tpr, thresholds = roc_curve(y_true_ood_pos, scores_neg)
    
    # Find threshold where TPR >= 0.95
    idx_95 = np.where(tpr >= 0.95)[0]
    fpr_at_95tpr = fpr[idx_95[0]] if len(idx_95) > 0 else 1.0
    
    # TPR@5%FPR
    idx_5 = np.where(fpr <= 0.05)[0]
    tpr_at_5fpr = tpr[idx_5[-1]] if len(idx_5) > 0 else 0.0
    
    return OODMetrics(
        auroc=auroc,
        aupr_ood=aupr_ood,
        fpr_at_95tpr=fpr_at_95tpr,
        tpr_at_5fpr=tpr_at_5fpr,
        n_id=len(id_scores),
        n_ood=len(ood_scores)
    )

def compute_selective_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    n_points: int = 20
) -> SelectiveMetrics:
    """
    Compute selective prediction metrics (coverage vs accuracy tradeoff).
    
    Only predict on samples where confidence > threshold.
    Sweep threshold to get coverage-accuracy curve.
    """
    thresholds = np.linspace(confidence.min(), confidence.max(), n_points)
    
    coverages = []
    macro_f1s = []
    risks = []
    
    for thresh in thresholds:
        mask = confidence >= thresh
        coverage = mask.mean()
        
        if mask.sum() > 1:
            # Need at least some samples
            y_t = y_true[mask]
            y_p = y_pred[mask]
            
            if len(np.unique(y_t)) > 1:
                mf1 = f1_score(y_t, y_p, average='macro')
            else:
                mf1 = (y_t == y_p).mean()  # fallback to accuracy
            
            acc = (y_t == y_p).mean()
            risk = 1 - acc
        else:
            mf1 = np.nan
            risk = np.nan
        
        coverages.append(coverage)
        macro_f1s.append(mf1)
        risks.append(risk)
    
    return SelectiveMetrics(
        coverages=coverages,
        macro_f1s=macro_f1s,
        risks=risks
    )

# ============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================================

def bootstrap_metric(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42
) -> BootstrapCI:
    """
    Compute bootstrap confidence interval for any metric.
    
    Args:
        y_true: Ground truth
        y_score: Predictions or scores
        metric_fn: Function(y_true, y_score) -> float
        n_bootstrap: Number of bootstrap iterations
        confidence: Confidence level (default 95%)
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        try:
            score = metric_fn(y_true[idx], y_score[idx])
            scores.append(score)
        except:
            continue
    
    scores = np.array(scores)
    alpha = (1 - confidence) / 2
    
    return BootstrapCI(
        mean=np.mean(scores),
        ci_lower=np.percentile(scores, alpha * 100),
        ci_upper=np.percentile(scores, (1 - alpha) * 100),
        std=np.std(scores)
    )

def bootstrap_ood_metrics(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Dict[str, BootstrapCI]:
    """Bootstrap CIs for all OOD metrics - essential for small OOD sets."""
    rng = np.random.RandomState(seed)
    
    aurocs, auprs, fprs = [], [], []
    
    for _ in range(n_bootstrap):
        # Resample both ID and OOD
        id_idx = rng.choice(len(id_scores), size=len(id_scores), replace=True)
        ood_idx = rng.choice(len(ood_scores), size=len(ood_scores), replace=True)
        
        try:
            metrics = compute_ood_metrics(id_scores[id_idx], ood_scores[ood_idx])
            aurocs.append(metrics.auroc)
            auprs.append(metrics.aupr_ood)
            fprs.append(metrics.fpr_at_95tpr)
        except:
            continue
    
    alpha = 0.025  # 95% CI
    
    def make_ci(arr):
        return BootstrapCI(
            mean=np.mean(arr),
            ci_lower=np.percentile(arr, 2.5),
            ci_upper=np.percentile(arr, 97.5),
            std=np.std(arr)
        )
    
    return {
        'auroc': make_ci(aurocs),
        'aupr_ood': make_ci(auprs),
        'fpr_at_95tpr': make_ci(fprs)
    }

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_confusion_matrix(
    cm_raw: np.ndarray,
    cm_norm: np.ndarray,
    labels: List[str],
    output_path: Path
):
    """Plot both raw and normalized confusion matrices."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Raw counts
    sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title('Confusion Matrix (Raw Counts)')
    
    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_title('Confusion Matrix (Normalized)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_reliability_diagram(
    bin_accuracies: np.ndarray,
    bin_confidences: np.ndarray,
    bin_counts: np.ndarray,
    ece: float,
    output_path: Path
):
    """Plot calibration/reliability diagram."""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    
    # Bar chart of accuracy vs confidence
    n_bins = len(bin_accuracies)
    width = 1.0 / n_bins
    positions = np.linspace(width/2, 1 - width/2, n_bins)
    
    # Filter out empty bins
    mask = bin_counts > 0
    ax.bar(positions[mask], bin_accuracies[mask], width=width * 0.8, 
           alpha=0.7, label='Accuracy')
    
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(f'Reliability Diagram (ECE = {ece:.3f})')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_pr_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    pr_auc: float,
    roc_auc: float,
    output_path: Path
):
    """Plot PR and ROC curves for tail class."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # PR curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    axes[0].plot(recall, precision, 'b-', lw=2)
    axes[0].fill_between(recall, precision, alpha=0.2)
    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('Precision')
    axes[0].set_title(f'Precision-Recall Curve (AUC = {pr_auc:.3f})')
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    axes[1].plot(fpr, tpr, 'b-', lw=2)
    axes[1].plot([0, 1], [0, 1], 'k--')
    axes[1].fill_between(fpr, tpr, alpha=0.2)
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title(f'ROC Curve (AUC = {roc_auc:.3f})')
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_coverage_curve(
    selective_metrics: SelectiveMetrics,
    output_path: Path
):
    """Plot selective prediction (coverage vs performance) curve."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    coverages = selective_metrics.coverages
    macro_f1s = selective_metrics.macro_f1s
    risks = selective_metrics.risks
    
    # Coverage vs Macro F1
    axes[0].plot(coverages, macro_f1s, 'b-o', lw=2, markersize=4)
    axes[0].set_xlabel('Coverage (fraction classified)')
    axes[0].set_ylabel('Macro F1')
    axes[0].set_title('Selective Prediction: Coverage vs Macro F1')
    axes[0].set_xlim(0, 1)
    axes[0].grid(True, alpha=0.3)
    
    # Risk-Coverage curve
    axes[1].plot(coverages, risks, 'r-o', lw=2, markersize=4)
    axes[1].set_xlabel('Coverage (fraction classified)')
    axes[1].set_ylabel('Risk (1 - Accuracy)')
    axes[1].set_title('Risk-Coverage Curve')
    axes[1].set_xlim(0, 1)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_ood_histograms(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    output_path: Path
):
    """Plot OOD score distributions for ID vs OOD samples."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.hist(id_scores, bins=30, alpha=0.6, label=f'ID (ear+tail, n={len(id_scores)})', 
            density=True, color='blue')
    ax.hist(ood_scores, bins=30, alpha=0.6, label=f'OOD (teat+other, n={len(ood_scores)})', 
            density=True, color='red')
    
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Density')
    ax.set_title('OOD Score Distribution')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================

def evaluate_all(
    df: pd.DataFrame,
    output_dir: Path,
    ood_method: str = 'msp',
    n_bootstrap: int = 1000
) -> Dict:
    """
    Run complete evaluation for both Section 4.1 and 4.2.
    
    Expected DataFrame columns:
        - clip_id: unique identifier
        - true_label: string label ('ear', 'tail', 'teat', 'other')
        - pred_label: predicted string label
        - prob_ear, prob_tail: class probabilities (required)
        - prob_teat, prob_other: optional, for 4-class models
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # =========================================================================
    # SECTION 4.1: Binary Classification (Ear vs Tail only)
    # =========================================================================
    print("\n" + "="*60)
    print("SECTION 4.1: Binary Classification (Ear vs Tail)")
    print("="*60)
    
    # Filter to ear/tail only
    df_binary = df[df['true_label'].isin(['ear', 'tail'])].copy()
    
    # Encode labels
    y_true_binary = (df_binary['true_label'] == 'tail').astype(int).values
    y_pred_binary = (df_binary['pred_label'] == 'tail').astype(int).values
    y_prob_tail = df_binary['prob_tail'].values
    
    # Compute metrics
    binary_metrics = compute_binary_metrics(y_true_binary, y_pred_binary, y_prob_tail)
    results['section_4_1'] = asdict(binary_metrics)
    
    # Print results
    print(f"\nDataset: {binary_metrics.n_ear} ear, {binary_metrics.n_tail} tail")
    print(f"\n--- Core Metrics ---")
    print(f"Macro F1:          {binary_metrics.macro_f1:.4f}")
    print(f"Ear F1:            {binary_metrics.ear_f1:.4f}")
    print(f"Tail F1:           {binary_metrics.tail_f1:.4f}")
    print(f"Balanced Accuracy: {binary_metrics.balanced_accuracy:.4f}")
    print(f"\n--- Per-class ---")
    print(f"Ear  Precision/Recall: {binary_metrics.ear_precision:.4f} / {binary_metrics.ear_recall:.4f}")
    print(f"Tail Precision/Recall: {binary_metrics.tail_precision:.4f} / {binary_metrics.tail_recall:.4f}")
    print(f"\n--- Additional ---")
    print(f"MCC:               {binary_metrics.mcc:.4f}")
    print(f"Tail PR-AUC:       {binary_metrics.tail_pr_auc:.4f}")
    print(f"Tail ROC-AUC:      {binary_metrics.tail_roc_auc:.4f}")
    
    # Confusion matrix
    cm_raw, cm_norm = compute_confusion_matrix(y_true_binary, y_pred_binary)
    results['section_4_1']['confusion_matrix_raw'] = cm_raw.tolist()
    results['section_4_1']['confusion_matrix_norm'] = cm_norm.tolist()
    
    print(f"\n--- Confusion Matrix ---")
    print(f"             Pred Ear  Pred Tail")
    print(f"True Ear     {cm_raw[0,0]:8d}  {cm_raw[0,1]:8d}")
    print(f"True Tail    {cm_raw[1,0]:8d}  {cm_raw[1,1]:8d}")
    
    # Calibration
    ece, bin_acc, bin_conf, bin_counts = compute_calibration(y_true_binary, y_prob_tail)
    results['section_4_1']['ece'] = ece
    print(f"\nECE (Expected Calibration Error): {ece:.4f}")
    
    # Bootstrap CI for Tail F1 (minority class)
    tail_f1_ci = bootstrap_metric(
        y_true_binary, y_pred_binary,
        lambda y, p: f1_score(y, p, pos_label=1),
        n_bootstrap=n_bootstrap
    )
    results['section_4_1']['tail_f1_ci'] = asdict(tail_f1_ci)
    print(f"\nTail F1 95% CI: {tail_f1_ci.ci_lower:.4f} - {tail_f1_ci.ci_upper:.4f}")
    
    # Plots
    plot_confusion_matrix(cm_raw, cm_norm, ['ear', 'tail'], 
                         output_dir / 'confusion_matrix.png')
    plot_reliability_diagram(bin_acc, bin_conf, bin_counts, ece,
                            output_dir / 'reliability_diagram.png')
    plot_pr_roc_curves(y_true_binary, y_prob_tail, 
                      binary_metrics.tail_pr_auc, binary_metrics.tail_roc_auc,
                      output_dir / 'pr_roc_curves.png')
    
    # =========================================================================
    # SECTION 4.2: Open-Set OOD Detection (Teat+Other as OOD)
    # =========================================================================
    
    # Check if we have OOD samples
    df_ood = df[df['true_label'].isin(['teat', 'other'])]
    
    if len(df_ood) > 0:
        print("\n" + "="*60)
        print("SECTION 4.2: Open-Set OOD Detection")
        print("="*60)
        
        # Get probabilities for all samples
        if 'prob_teat' in df.columns and 'prob_other' in df.columns:
            probs = df[['prob_ear', 'prob_tail', 'prob_teat', 'prob_other']].values
        else:
            probs = df[['prob_ear', 'prob_tail']].values
        
        # Compute OOD scores
        all_scores = compute_ood_scores(probs, method=ood_method)
        
        # Split into ID and OOD
        id_mask = df['true_label'].isin(['ear', 'tail'])
        ood_mask = df['true_label'].isin(['teat', 'other'])
        
        id_scores = all_scores[id_mask]
        ood_scores = all_scores[ood_mask]
        
        print(f"\nID samples (ear+tail):   {len(id_scores)}")
        print(f"OOD samples (teat+other): {len(ood_scores)}")
        print(f"OOD method: {ood_method}")
        
        # Compute OOD metrics
        ood_metrics = compute_ood_metrics(id_scores, ood_scores)
        results['section_4_2'] = asdict(ood_metrics)
        
        print(f"\n--- OOD Detection Metrics ---")
        print(f"AUROC:        {ood_metrics.auroc:.4f}")
        print(f"AUPR-OOD:     {ood_metrics.aupr_ood:.4f}")
        print(f"FPR@95%TPR:   {ood_metrics.fpr_at_95tpr:.4f}")
        print(f"TPR@5%FPR:    {ood_metrics.tpr_at_5fpr:.4f}")
        
        # Bootstrap CIs (critical for small OOD sets like teat)
        ood_cis = bootstrap_ood_metrics(id_scores, ood_scores, n_bootstrap=n_bootstrap)
        results['section_4_2']['bootstrap_ci'] = {k: asdict(v) for k, v in ood_cis.items()}
        
        print(f"\n--- Bootstrap 95% CIs ---")
        print(f"AUROC:    {ood_cis['auroc'].ci_lower:.4f} - {ood_cis['auroc'].ci_upper:.4f}")
        print(f"AUPR-OOD: {ood_cis['aupr_ood'].ci_lower:.4f} - {ood_cis['aupr_ood'].ci_upper:.4f}")
        print(f"FPR@95%:  {ood_cis['fpr_at_95tpr'].ci_lower:.4f} - {ood_cis['fpr_at_95tpr'].ci_upper:.4f}")
        
        # Selective prediction on ID samples
        selective = compute_selective_metrics(
            y_true_binary, y_pred_binary, 
            compute_ood_scores(df_binary[['prob_ear', 'prob_tail']].values, method=ood_method)
        )
        results['section_4_2']['selective'] = asdict(selective)
        
        # Plots
        plot_ood_histograms(id_scores, ood_scores, output_dir / 'ood_histograms.png')
        plot_coverage_curve(selective, output_dir / 'coverage_curve.png')
        
        # OOD breakdown by class
        teat_mask = df['true_label'] == 'teat'
        other_mask = df['true_label'] == 'other'
        
        if teat_mask.sum() > 0:
            teat_scores = all_scores[teat_mask]
            print(f"\n--- OOD Breakdown ---")
            print(f"Teat scores:  mean={teat_scores.mean():.4f}, std={teat_scores.std():.4f}, n={len(teat_scores)}")
        if other_mask.sum() > 0:
            other_scores = all_scores[other_mask]
            print(f"Other scores: mean={other_scores.mean():.4f}, std={other_scores.std():.4f}, n={len(other_scores)}")
    
    else:
        print("\n[INFO] No teat/other samples found - skipping Section 4.2")
        results['section_4_2'] = None
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    
    # Save JSON
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n" + "="*60)
    print(f"Results saved to: {output_dir}")
    print(f"  - metrics.json")
    print(f"  - confusion_matrix.png")
    print(f"  - reliability_diagram.png")
    print(f"  - pr_roc_curves.png")
    if results.get('section_4_2'):
        print(f"  - ood_histograms.png")
        print(f"  - coverage_curve.png")
    print("="*60)
    
    return results

# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluation metrics for cross-sucking detection')
    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to predictions CSV')
    parser.add_argument('--output', type=str, default='results/metrics/',
                       help='Output directory for metrics and plots')
    parser.add_argument('--ood-method', type=str, default='msp',
                       choices=['msp', 'entropy', 'energy'],
                       help='OOD scoring method')
    parser.add_argument('--n-bootstrap', type=int, default=1000,
                       help='Number of bootstrap iterations')
    
    args = parser.parse_args()
    
    # Load predictions
    df = pd.read_csv(args.predictions)
    
    # Validate columns
    required = ['clip_id', 'true_label', 'pred_label', 'prob_ear', 'prob_tail']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Run evaluation
    evaluate_all(
        df=df,
        output_dir=args.output,
        ood_method=args.ood_method,
        n_bootstrap=args.n_bootstrap
    )

if __name__ == '__main__':
    main()