#!/usr/bin/env python3
"""
Ablation Results Aggregator
===========================

Collects results from all ablation experiments and creates comparison table.

Usage:
    python scripts/aggregate_ablation_results.py --runs-dir runs/ --output results/ablation_summary.csv
"""

import argparse
import json
from pathlib import Path
import pandas as pd
from datetime import datetime


def find_experiment_dirs(runs_dir: Path) -> list[Path]:
    """Find all ablation experiment directories."""
    pattern = "ablation_*"
    return sorted(runs_dir.glob(pattern))


def load_experiment_results(exp_dir: Path) -> dict:
    """Load results from a single experiment."""
    results = {
        'experiment': exp_dir.name,
        'path': str(exp_dir),
    }
    
    # Try to load eval results
    eval_file = exp_dir / "eval_results.json"
    if eval_file.exists():
        with open(eval_file) as f:
            eval_data = json.load(f)
            # Flatten nested dict
            if 'section_4_1' in eval_data:
                for k, v in eval_data['section_4_1'].items():
                    if isinstance(v, (int, float)):
                        results[k] = v
    
    # Try to load training history
    history_file = exp_dir / "training_history.json"
    if history_file.exists():
        with open(history_file) as f:
            history = json.load(f)
            if 'val_macro_f1' in history:
                results['best_val_macro_f1'] = max(history['val_macro_f1'])
            if 'val_tail_f1' in history:
                results['best_val_tail_f1'] = max(history['val_tail_f1'])
            if 'train_macro_f1' in history:
                results['final_train_macro_f1'] = history['train_macro_f1'][-1]
                results['train_val_gap'] = results['final_train_macro_f1'] - results.get('best_val_macro_f1', 0)
    
    # Try to load config
    config_file = exp_dir / "config.yaml"
    if config_file.exists():
        import yaml
        with open(config_file) as f:
            config = yaml.safe_load(f)
            results['model'] = config.get('model', {}).get('backbone', 'unknown')
            results['pretrained'] = config.get('model', {}).get('pretrained', True)
            results['batch_size'] = config.get('data', {}).get('batch_size', 'unknown')
            results['lr'] = config.get('optim', {}).get('lr', 'unknown')
    
    return results


def create_comparison_table(results: list[dict]) -> pd.DataFrame:
    """Create comparison table from results."""
    df = pd.DataFrame(results)
    
    # Reorder columns for readability
    priority_cols = [
        'experiment', 
        'tail_f1', 
        'tail_recall', 
        'macro_f1',
        'tail_roc_auc',
        'best_val_tail_f1',
        'train_val_gap',
        'model',
        'pretrained',
    ]
    
    # Get columns that exist
    cols = [c for c in priority_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in cols]
    
    df = df[cols + other_cols]
    
    # Sort by tail_f1 descending
    if 'tail_f1' in df.columns:
        df = df.sort_values('tail_f1', ascending=False)
    
    return df


def print_summary(df: pd.DataFrame):
    """Print formatted summary."""
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    
    # Key metrics
    key_cols = ['experiment', 'tail_f1', 'tail_recall', 'macro_f1', 'train_val_gap']
    key_cols = [c for c in key_cols if c in df.columns]
    
    print("\nKey Metrics:")
    print(df[key_cols].to_string(index=False))
    
    # Best experiment
    if 'tail_f1' in df.columns:
        best = df.loc[df['tail_f1'].idxmax()]
        print(f"\n🏆 Best Experiment: {best['experiment']}")
        print(f"   Tail F1: {best['tail_f1']:.3f}")
        if 'tail_recall' in best:
            print(f"   Tail Recall: {best['tail_recall']:.3f}")
        if 'macro_f1' in best:
            print(f"   Macro F1: {best['macro_f1']:.3f}")
    
    # Baseline comparison
    baseline = df[df['experiment'].str.contains('E01|baseline', case=False)]
    if len(baseline) > 0:
        baseline_tail_f1 = baseline.iloc[0].get('tail_f1', 0)
        print(f"\n📊 Baseline (E01) Tail F1: {baseline_tail_f1:.3f}")
        
        for _, row in df.iterrows():
            if 'E01' not in row['experiment']:
                delta = row.get('tail_f1', 0) - baseline_tail_f1
                sign = '+' if delta >= 0 else ''
                print(f"   {row['experiment']}: {sign}{delta:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Aggregate ablation results')
    parser.add_argument('--runs-dir', type=str, default='runs/',
                       help='Directory containing experiment runs')
    parser.add_argument('--output', type=str, default='results/ablation_summary.csv',
                       help='Output CSV file')
    
    args = parser.parse_args()
    
    runs_dir = Path(args.runs_dir)
    
    # Find experiments
    exp_dirs = find_experiment_dirs(runs_dir)
    print(f"Found {len(exp_dirs)} ablation experiments")
    
    if len(exp_dirs) == 0:
        print("No experiments found. Run some experiments first!")
        return
    
    # Load results
    results = []
    for exp_dir in exp_dirs:
        try:
            result = load_experiment_results(exp_dir)
            results.append(result)
            print(f"  ✓ Loaded: {exp_dir.name}")
        except Exception as e:
            print(f"  ✗ Failed: {exp_dir.name} - {e}")
    
    # Create comparison table
    df = create_comparison_table(results)
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    
    # Print summary
    print_summary(df)
    
    # Save markdown version
    md_path = output_path.with_suffix('.md')
    with open(md_path, 'w') as f:
        f.write(f"# Ablation Study Results\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(df.to_markdown(index=False))
    print(f"Saved markdown: {md_path}")


if __name__ == '__main__':
    main()