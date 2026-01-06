#!/usr/bin/env python3
"""
Clean and Normalize Annotations
===============================

Applies mapping rules to normalize labels and flags QA issues.

Usage:
    python -m src.cli.clean_annotations --input data/annotations/interactions.xlsx
    cs-clean --input data/annotations/interactions.xlsx
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import typer
import pandas as pd
from omegaconf import OmegaConf
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Clean and normalize annotation data")
console = Console()


def load_mapping_rules(rules_path: str) -> Dict:
    """Load mapping rules from YAML."""
    return OmegaConf.to_container(OmegaConf.load(rules_path))


def normalize_behavior(value: str, mapping: Dict) -> str:
    """Normalize behavior label."""
    if pd.isna(value):
        return "unknown"
    
    value_lower = str(value).lower().strip()
    behavior_map = mapping.get('behavior', {})
    
    # Check exact match first
    if value_lower in behavior_map:
        return behavior_map[value_lower]
    
    # Check original case
    if value in behavior_map:
        return behavior_map[value]
    
    return value_lower


def normalize_ended_by(value: str, mapping: Dict) -> str:
    """Normalize ended_by label to initiator/receiver/other."""
    if pd.isna(value):
        return "unknown"
    
    value_lower = str(value).lower().strip()
    patterns = mapping.get('ended_by', {}).get('patterns', {})
    
    for role, role_patterns in patterns.items():
        for pattern in role_patterns:
            if pattern.lower() in value_lower:
                return role
    
    return mapping.get('ended_by', {}).get('default', 'unknown')


def normalize_pen_location(value: str, mapping: Dict) -> str:
    """Normalize pen location."""
    if pd.isna(value):
        return "unknown"
    
    value_str = str(value).lower().strip()
    location_map = mapping.get('pen_location', {})
    
    if value_str in location_map:
        return location_map[value_str]
    
    return value_str


def check_duration(duration: float, rules: Dict) -> Optional[str]:
    """Check if duration is valid, return flag if not."""
    if pd.isna(duration):
        return "missing_duration"
    
    duration_rules = rules.get('duration', {})
    min_valid = duration_rules.get('min_valid', 0)
    max_valid = duration_rules.get('max_valid', 300)
    flag_threshold = duration_rules.get('flag_threshold', 120)
    
    if duration < min_valid:
        return f"duration_below_min_{min_valid}"
    if duration > max_valid:
        return f"duration_above_max_{max_valid}"
    if duration > flag_threshold:
        return f"duration_above_threshold_{flag_threshold}"
    
    return None


def check_calf_id(calf_id, rules: Dict) -> Optional[str]:
    """Check if calf ID is valid."""
    if pd.isna(calf_id):
        return "missing_calf_id"
    
    try:
        calf_id = int(calf_id)
    except (ValueError, TypeError):
        return f"invalid_calf_id_format_{calf_id}"
    
    valid_ranges = rules.get('calf_ids', {}).get('valid_ranges', [])
    
    for min_id, max_id in valid_ranges:
        if min_id <= calf_id <= max_id:
            return None
    
    if rules.get('calf_ids', {}).get('flag_if_outside', False):
        return f"calf_id_outside_range_{calf_id}"
    
    return None


def clean_annotations(
    df: pd.DataFrame,
    rules: Dict
) -> tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Clean and normalize annotations.
    
    Returns:
        - Cleaned DataFrame
        - QA flags DataFrame
        - Audit log Dict
    """
    df = df.copy()
    qa_flags = []
    audit = {
        'timestamp': datetime.now().isoformat(),
        'original_rows': len(df),
        'changes': [],
        'distributions': {}
    }
    
    # Track original distributions
    if 'Behavior' in df.columns:
        audit['distributions']['behavior_original'] = df['Behavior'].value_counts().to_dict()
    
    # Normalize behavior
    if 'Behavior' in df.columns:
        df['behavior_normalized'] = df['Behavior'].apply(
            lambda x: normalize_behavior(x, rules)
        )
        changes = (df['Behavior'].astype(str).str.lower() != df['behavior_normalized']).sum()
        audit['changes'].append({'field': 'behavior', 'changes': int(changes)})
        audit['distributions']['behavior_normalized'] = df['behavior_normalized'].value_counts().to_dict()
    
    # Normalize ended_by
    if 'Ended.by.initiator.or.receiver.' in df.columns:
        df['ended_by_normalized'] = df['Ended.by.initiator.or.receiver.'].apply(
            lambda x: normalize_ended_by(x, rules)
        )
        audit['distributions']['ended_by_normalized'] = df['ended_by_normalized'].value_counts().to_dict()
    
    # Normalize pen location
    if 'Pen.location' in df.columns:
        df['pen_location_normalized'] = df['Pen.location'].apply(
            lambda x: normalize_pen_location(x, rules)
        )
    
    # QA checks
    for idx, row in df.iterrows():
        flags = []
        
        # Duration check
        if 'Duration' in df.columns:
            duration_flag = check_duration(row.get('Duration'), rules)
            if duration_flag:
                flags.append(duration_flag)
        
        # Calf ID checks
        if 'ID.initiator' in df.columns:
            init_flag = check_calf_id(row.get('ID.initiator'), rules)
            if init_flag:
                flags.append(f"initiator_{init_flag}")
        
        if 'ID.receiver' in df.columns:
            recv_flag = check_calf_id(row.get('ID.receiver'), rules)
            if recv_flag:
                flags.append(f"receiver_{recv_flag}")
        
        if flags:
            qa_flags.append({
                'event_idx': idx,
                'group': row.get('Group'),
                'day': row.get('Day'),
                'start_time': row.get('Start.time'),
                'flags': '|'.join(flags)
            })
    
    qa_df = pd.DataFrame(qa_flags)
    audit['qa_flags_count'] = len(qa_flags)
    
    return df, qa_df, audit


def print_summary(audit: Dict, qa_df: pd.DataFrame):
    """Print cleaning summary."""
    console.print("\n[bold]📊 CLEANING SUMMARY[/bold]")
    
    # Changes
    table = Table(title="Label Changes")
    table.add_column("Field", style="cyan")
    table.add_column("Changes Made", justify="right")
    
    for change in audit.get('changes', []):
        table.add_row(change['field'], str(change['changes']))
    
    console.print(table)
    
    # Distributions
    if 'behavior_normalized' in audit.get('distributions', {}):
        console.print("\n[bold]Behavior Distribution (Normalized):[/bold]")
        for behavior, count in audit['distributions']['behavior_normalized'].items():
            pct = count / audit['original_rows'] * 100
            console.print(f"  {behavior}: {count} ({pct:.1f}%)")
    
    if 'ended_by_normalized' in audit.get('distributions', {}):
        console.print("\n[bold]Role Distribution (Who Ended):[/bold]")
        for role, count in audit['distributions']['ended_by_normalized'].items():
            pct = count / audit['original_rows'] * 100
            console.print(f"  {role}: {count} ({pct:.1f}%)")
    
    # QA flags
    if len(qa_df) > 0:
        console.print(f"\n[bold yellow]⚠️ QA Flags: {len(qa_df)} events flagged[/bold yellow]")
        flag_counts = qa_df['flags'].str.split('|').explode().value_counts()
        for flag, count in flag_counts.head(10).items():
            console.print(f"  • {flag}: {count}")


@app.command()
def clean(
    input: str = typer.Argument(..., help="Input Excel file"),
    rules: str = typer.Option("data/annotations/mapping_rules.yaml", help="Mapping rules file"),
    output_dir: str = typer.Option("data/manifests", help="Output directory"),
):
    """Clean and normalize annotation data."""
    console.print("[bold]🧹 Cleaning Annotations[/bold]\n")
    
    # Load data
    console.print(f"Loading: {input}")
    df = pd.read_csv(input)
    console.print(f"Loaded {len(df)} events")
    
    # Drop junk columns
    junk_cols = [c for c in df.columns if isinstance(c, str) and c.strip().lower().startswith("enter a 0")]
    if junk_cols:
        df = df.drop(columns=junk_cols)
        console.print(f"Dropped junk columns: {junk_cols}")
    
    # Load rules
    console.print(f"Loading rules: {rules}")
    mapping_rules = load_mapping_rules(rules)
    
    # Clean
    console.print("Applying normalization rules...")
    cleaned_df, qa_df, audit = clean_annotations(df, mapping_rules)
    
    # Print summary
    print_summary(audit, qa_df)
    
    # Save outputs
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save cleaned data
    cleaned_file = output_path / "labeled_events.csv"
    cleaned_df.to_csv(cleaned_file, index=False)
    console.print(f"\n[green]✅ Cleaned data saved to: {cleaned_file}[/green]")
    
    # Save QA flags
    if len(qa_df) > 0:
        qa_file = Path("data/annotations/qa_flags.csv")
        qa_df.to_csv(qa_file, index=False)
        console.print(f"[yellow]⚠️ QA flags saved to: {qa_file}[/yellow]")
    
    # Save audit log
    audit_file = output_path / "cleaning_audit.json"
    with open(audit_file, 'w') as f:
        json.dump(audit, f, indent=2, default=str)
    console.print(f"[green]✅ Audit log saved to: {audit_file}[/green]")


@app.command()
def stats(
    input: str = typer.Argument(..., help="Cleaned events CSV"),
):
    """Show statistics for cleaned annotations."""
    console.print("[bold]📊 Annotation Statistics[/bold]\n")
    
    df = pd.read_csv(input)
    
    # Basic stats
    console.print(f"[bold]Total Events:[/bold] {len(df)}")
    
    # By group/day
    if 'Group' in df.columns and 'Day' in df.columns:
        console.print("\n[bold]Events by Group & Day:[/bold]")
        cross = pd.crosstab(df['Group'], df['Day'], margins=True)
        console.print(cross.to_string())
    
    # Behavior
    if 'behavior_normalized' in df.columns:
        console.print("\n[bold]Behavior Distribution:[/bold]")
        for behavior in df['behavior_normalized'].value_counts().items():
            console.print(f"  {behavior[0]}: {behavior[1]}")
    
    # Duration
    if 'Duration' in df.columns:
        console.print("\n[bold]Duration Statistics:[/bold]")
        console.print(f"  Mean: {df['Duration'].mean():.1f} sec")
        console.print(f"  Median: {df['Duration'].median():.1f} sec")
        console.print(f"  Max: {df['Duration'].max():.1f} sec")


if __name__ == "__main__":
    app()
