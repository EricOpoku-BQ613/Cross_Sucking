#!/usr/bin/env python3
"""
Build Data Manifests
====================

Creates canonical manifests for labeled events, unlabeled clips, and all videos.

Usage:
    python -m src.cli.make_manifests --config configs/paths.yaml
    cs-manifest --config configs/paths.yaml
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import typer
import pandas as pd
from omegaconf import OmegaConf
from rich.console import Console
from rich.table import Table
from rich.progress import track

app = typer.Typer(help="Build data manifests for cross-sucking pipeline")
console = Console()


def parse_video_filename(filename: str, pattern: str) -> Optional[Dict]:
    """Parse video filename to extract metadata."""
    match = re.search(pattern, filename)
    if match:
        try:
            start_dt = datetime.strptime(match.group(2), '%Y%m%d%H%M%S')
            end_dt = datetime.strptime(match.group(3), '%Y%m%d%H%M%S')
            duration = (end_dt - start_dt).total_seconds()
            return {
                'channel': int(match.group(1)),
                'start_time': start_dt,
                'end_time': end_dt,
                'duration_sec': duration,
                'date': start_dt.strftime('%Y-%m-%d')
            }
        except (ValueError, IndexError):
            return None
    return None


def scan_video_folder(
    folder_path: Path,
    pattern: str,
    expected_cameras: List[int] = None
) -> Dict:
    """Scan a folder for video files and extract metadata."""
    result = {
        'path': str(folder_path),
        'cameras': {},
        'total_videos': 0,
        'total_hours': 0,
        'issues': []
    }
    
    if not folder_path.exists():
        result['issues'].append(f"Folder does not exist: {folder_path}")
        return result
    
    # Check for camera subfolders
    camera_folders = [f for f in folder_path.iterdir() if f.is_dir() and 'Cam' in f.name]
    
    if camera_folders:
        for cam_folder in camera_folders:
            cam_match = re.search(r'Cam\s*(\d+)', cam_folder.name)
            if cam_match:
                cam_id = int(cam_match.group(1))
                videos = list(cam_folder.glob('*.mp4')) + list(cam_folder.glob('*.MP4'))
                
                dates = set()
                total_hours = 0
                video_list = []
                
                for video in videos:
                    parsed = parse_video_filename(video.name, pattern)
                    if parsed:
                        dates.add(parsed['date'])
                        total_hours += parsed['duration_sec'] / 3600
                        video_list.append({
                            'filename': video.name,
                            'path': str(video),
                            'start': parsed['start_time'].isoformat(),
                            'end': parsed['end_time'].isoformat(),
                            'duration_sec': parsed['duration_sec']
                        })
                
                result['cameras'][cam_id] = {
                    'path': str(cam_folder),
                    'video_count': len(videos),
                    'total_hours': round(total_hours, 2),
                    'date_range': sorted(list(dates)),
                    'videos': video_list
                }
                result['total_videos'] += len(videos)
                result['total_hours'] += total_hours
                
                # Check if camera was expected
                if expected_cameras and cam_id not in expected_cameras:
                    result['issues'].append(f"Unexpected camera {cam_id}, expected {expected_cameras}")
    
    result['total_hours'] = round(result['total_hours'], 2)
    return result


def scan_labeled_data(config: Dict) -> Dict:
    """Scan all labeled data folders."""
    labeled = config.get('labeled', {})
    pattern = config.get('video_pattern', {}).get('regex', r'N884A6_ch(\d+)_main_(\d{14})_(\d{14})')
    
    results = {}
    for key, info in labeled.items():
        console.print(f"  Scanning {key}...")
        folder_path = Path(info['path'])
        expected_cameras = info.get('cameras', [])
        
        scan_result = scan_video_folder(folder_path, pattern, expected_cameras)
        scan_result['group_id'] = info.get('group_id')
        scan_result['day'] = info.get('day')
        results[key] = scan_result
    
    return results


def scan_unlabeled_data(config: Dict) -> Dict:
    """Scan all unlabeled data folders."""
    unlabeled = config.get('unlabeled', {})
    pattern = config.get('video_pattern', {}).get('regex', r'N884A6_ch(\d+)_main_(\d{14})_(\d{14})')
    
    results = {}
    groups = unlabeled.get('groups', {})
    
    for key, info in groups.items():
        console.print(f"  Scanning {key}...")
        folder_path = Path(info['path'])
        expected_cameras = info.get('cameras', [])
        
        scan_result = scan_video_folder(folder_path, pattern, expected_cameras)
        scan_result['group_id'] = info.get('group_id')
        results[key] = scan_result
    
    return results


def check_duplicates(manifest: Dict) -> List[Dict]:
    """Check for duplicate videos across folders."""
    seen_files = {}
    duplicates = []
    
    for dataset_type in ['labeled', 'unlabeled']:
        for folder_key, folder_data in manifest.get(dataset_type, {}).items():
            for cam_id, cam_data in folder_data.get('cameras', {}).items():
                for video in cam_data.get('videos', []):
                    filename = video['filename']
                    if filename in seen_files:
                        duplicates.append({
                            'filename': filename,
                            'location1': seen_files[filename],
                            'location2': f"{folder_key}/Cam{cam_id}"
                        })
                    else:
                        seen_files[filename] = f"{folder_key}/Cam{cam_id}"
    
    return duplicates


def print_summary(manifest: Dict):
    """Print manifest summary."""
    console.print("\n[bold]📊 DATA MANIFEST SUMMARY[/bold]")
    
    # Labeled data table
    table = Table(title="Labeled Data")
    table.add_column("Folder", style="cyan")
    table.add_column("Group", justify="center")
    table.add_column("Day", justify="center")
    table.add_column("Cameras", justify="center")
    table.add_column("Videos", justify="right")
    table.add_column("Hours", justify="right")
    
    total_labeled_videos = 0
    total_labeled_hours = 0
    
    for key, data in manifest.get('labeled', {}).items():
        cameras = list(data.get('cameras', {}).keys())
        table.add_row(
            key,
            str(data.get('group_id', '?')),
            str(data.get('day', '?')),
            str(cameras),
            str(data.get('total_videos', 0)),
            f"{data.get('total_hours', 0):.1f}"
        )
        total_labeled_videos += data.get('total_videos', 0)
        total_labeled_hours += data.get('total_hours', 0)
    
    console.print(table)
    console.print(f"  [bold]Total:[/bold] {total_labeled_videos} videos, {total_labeled_hours:.1f} hours\n")
    
    # Unlabeled data table
    table = Table(title="Unlabeled Data (SSL Pretraining)")
    table.add_column("Folder", style="cyan")
    table.add_column("Group", justify="center")
    table.add_column("Cameras", justify="center")
    table.add_column("Videos", justify="right")
    table.add_column("Hours", justify="right")
    
    total_unlabeled_videos = 0
    total_unlabeled_hours = 0
    
    for key, data in manifest.get('unlabeled', {}).items():
        cameras = list(data.get('cameras', {}).keys())
        table.add_row(
            key,
            str(data.get('group_id', '?')),
            str(cameras),
            str(data.get('total_videos', 0)),
            f"{data.get('total_hours', 0):.1f}"
        )
        total_unlabeled_videos += data.get('total_videos', 0)
        total_unlabeled_hours += data.get('total_hours', 0)
    
    console.print(table)
    console.print(f"  [bold]Total:[/bold] {total_unlabeled_videos} videos, {total_unlabeled_hours:.1f} hours\n")
    
    # Issues
    all_issues = []
    for dataset_type in ['labeled', 'unlabeled']:
        for folder_key, folder_data in manifest.get(dataset_type, {}).items():
            for issue in folder_data.get('issues', []):
                all_issues.append(f"{folder_key}: {issue}")
    
    if all_issues:
        console.print("[bold yellow]⚠️ Issues Found:[/bold yellow]")
        for issue in all_issues:
            console.print(f"  • {issue}")
    
    # Duplicates
    duplicates = manifest.get('duplicates', [])
    if duplicates:
        console.print(f"\n[bold yellow]⚠️ Duplicate Videos Found: {len(duplicates)}[/bold yellow]")
        for dup in duplicates[:5]:
            console.print(f"  • {dup['filename']}: {dup['location1']} vs {dup['location2']}")


@app.command()
def build(
    config: str = typer.Option("configs/paths.yaml", help="Path to paths config"),
    output: str = typer.Option("data/manifests", help="Output directory"),
    check_dups: bool = typer.Option(True, help="Check for duplicate videos")
):
    """Build complete data manifest."""
    console.print("[bold]🔍 Building Data Manifest[/bold]\n")
    
    # Load config
    cfg = OmegaConf.load(config)
    cfg = OmegaConf.to_container(cfg)
    
    manifest = {
        'created_at': datetime.now().isoformat(),
        'config_file': config,
        'labeled': {},
        'unlabeled': {},
        'summary': {}
    }
    
    # Scan labeled data
    console.print("[bold]📁 Scanning Labeled Data...[/bold]")
    manifest['labeled'] = scan_labeled_data(cfg)
    
    # Scan unlabeled data
    console.print("\n[bold]📁 Scanning Unlabeled Data...[/bold]")
    manifest['unlabeled'] = scan_unlabeled_data(cfg)
    
    # Check for duplicates
    if check_dups:
        console.print("\n[bold]🔎 Checking for Duplicates...[/bold]")
        manifest['duplicates'] = check_duplicates(manifest)
    
    # Calculate summary
    manifest['summary'] = {
        'labeled_folders': len(manifest['labeled']),
        'unlabeled_folders': len(manifest['unlabeled']),
        'labeled_videos': sum(d.get('total_videos', 0) for d in manifest['labeled'].values()),
        'labeled_hours': round(sum(d.get('total_hours', 0) for d in manifest['labeled'].values()), 1),
        'unlabeled_videos': sum(d.get('total_videos', 0) for d in manifest['unlabeled'].values()),
        'unlabeled_hours': round(sum(d.get('total_hours', 0) for d in manifest['unlabeled'].values()), 1),
        'duplicate_count': len(manifest.get('duplicates', []))
    }
    
    # Print summary
    print_summary(manifest)
    
    # Save manifest
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    manifest_file = output_path / "video_manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    
    console.print(f"\n[bold green]✅ Manifest saved to: {manifest_file}[/bold green]")


@app.command()
def verify(
    manifest: str = typer.Option("data/manifests/video_manifest.json", help="Manifest file"),
    sample: int = typer.Option(5, help="Number of videos to verify per folder")
):
    """Verify manifest by checking video file existence."""
    console.print("[bold]🔍 Verifying Manifest[/bold]\n")
    
    with open(manifest, 'r') as f:
        data = json.load(f)
    
    issues = []
    verified = 0
    
    for dataset_type in ['labeled', 'unlabeled']:
        for folder_key, folder_data in data.get(dataset_type, {}).items():
            for cam_id, cam_data in folder_data.get('cameras', {}).items():
                videos = cam_data.get('videos', [])[:sample]
                for video in videos:
                    path = Path(video['path'])
                    if path.exists():
                        verified += 1
                    else:
                        issues.append(f"Missing: {video['path']}")
    
    console.print(f"[green]✅ Verified: {verified} videos exist[/green]")
    if issues:
        console.print(f"[red]❌ Missing: {len(issues)} videos[/red]")
        for issue in issues[:10]:
            console.print(f"  • {issue}")


if __name__ == "__main__":
    app()
