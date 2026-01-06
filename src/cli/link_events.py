#!/usr/bin/env python3
"""
Link Labeled Events to Video Files
===================================

Maps each event from labeled_events.csv to its corresponding video file
based on Group, Day, and timestamp.

Usage:
    python -m src.cli.link_events link
"""

import json
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import typer
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import track

app = typer.Typer(help="Link labeled events to video files")
console = Console()


def parse_time(time_val) -> Optional[datetime]:
    """Parse time value to datetime.time object."""
    if pd.isna(time_val):
        return None
    
    if isinstance(time_val, str):
        for fmt in ['%H:%M:%S', '%H:%M', '%I:%M:%S %p', '%I:%M %p']:
            try:
                return datetime.strptime(time_val, fmt)
            except ValueError:
                continue
        return None
    elif hasattr(time_val, 'hour'):
        return datetime.combine(datetime.today(), time_val)
    return None


def parse_video_times(filename: str) -> Optional[Tuple[datetime, datetime]]:
    """Extract start and end times from video filename."""
    pattern = r'N884A6_ch(\d+)_main_(\d{14})_(\d{14})'
    match = re.search(pattern, filename)
    
    if match:
        try:
            start = datetime.strptime(match.group(2), '%Y%m%d%H%M%S')
            end = datetime.strptime(match.group(3), '%Y%m%d%H%M%S')
            return start, end
        except ValueError:
            return None
    return None


def get_event_date(row: pd.Series) -> Optional[datetime]:
    """Extract date from event row."""
    if 'Date' in row and pd.notna(row['Date']):
        date_val = row['Date']
        if isinstance(date_val, str):
            for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y']:
                try:
                    return datetime.strptime(date_val, fmt)
                except ValueError:
                    continue
        elif hasattr(date_val, 'year'):
            return datetime(date_val.year, date_val.month, date_val.day)
    return None


def find_video_for_event(
    event_time: datetime,
    event_date: Optional[datetime],
    videos: List[Dict],
    tolerance_sec: int = 60
) -> Optional[Dict]:
    """Find the video file containing an event."""
    event_time_only = event_time.time()
    
    for video in videos:
        video_times = parse_video_times(video['filename'])
        if not video_times:
            continue
            
        video_start, video_end = video_times
        
        if event_date:
            if video_start.date() != event_date.date():
                continue
        
        event_datetime = datetime.combine(video_start.date(), event_time_only)
        
        start_with_tolerance = video_start - timedelta(seconds=tolerance_sec)
        end_with_tolerance = video_end + timedelta(seconds=tolerance_sec)
        
        if start_with_tolerance <= event_datetime <= end_with_tolerance:
            offset = (event_datetime - video_start).total_seconds()
            
            return {
                'video_filename': video['filename'],
                'video_path': video['path'],
                'video_start': video_start.isoformat(),
                'video_end': video_end.isoformat(),
                'event_offset_sec': max(0, offset),
                'video_duration_sec': video.get('duration_sec', (video_end - video_start).total_seconds())
            }
    
    return None


def link_events_to_videos(events_df: pd.DataFrame, manifest: Dict) -> Tuple[pd.DataFrame, List, Dict]:
    """Link each event to its video file."""
    results = []
    not_found = []
    camera_stats = {}
    
    video_lookup = {}
    
    for folder_key, folder_data in manifest.get('labeled', {}).items():
        group_id = folder_data.get('group_id')
        day = folder_data.get('day')
        
        if group_id is None:
            continue
            
        key = (group_id, day)
        if key not in video_lookup:
            video_lookup[key] = {}
        
        for cam_id, cam_data in folder_data.get('cameras', {}).items():
            cam_id = int(cam_id)
            video_lookup[key][cam_id] = cam_data.get('videos', [])
    
    console.print(f"\n[bold]Linking {len(events_df)} events to videos...[/bold]\n")
    
    for idx, row in track(events_df.iterrows(), total=len(events_df), description="Linking"):
        group = row.get('Group')
        day = row.get('Day')
        
        start_time_col = 'Start.time' if 'Start.time' in row else 'Start_time'
        start_time = parse_time(row.get(start_time_col))
        
        if start_time is None:
            not_found.append({'idx': idx, 'reason': 'Could not parse start time'})
            continue
        
        event_date = get_event_date(row)
        
        key = (group, day)
        if key not in video_lookup:
            not_found.append({'idx': idx, 'reason': f'No videos for Group {group} Day {day}'})
            continue
        
        cameras = video_lookup[key]
        
        video_found = None
        found_camera = None
        
        for cam_id, videos in cameras.items():
            video_info = find_video_for_event(start_time, event_date, videos)
            if video_info:
                video_found = video_info
                found_camera = cam_id
                break
        
        result = {
            'event_idx': idx,
            'group': group,
            'day': day,
            'date': event_date.strftime('%Y-%m-%d') if event_date else None,
            'start_time': str(start_time.time()) if start_time else None,
            'end_time': str(parse_time(row.get('End.time', row.get('End_time'))).time()) if parse_time(row.get('End.time', row.get('End_time'))) else None,
            'duration_sec': row.get('Duration'),
            'behavior': row.get('behavior_normalized', row.get('Behavior')),
            'initiator_id': row.get('ID.initiator', row.get('ID_initiator')),
            'receiver_id': row.get('ID.receiver', row.get('ID_receiver')),
            'ended_by': row.get('ended_by_normalized', row.get('Ended.by.initiator.or.receiver.')),
            'pen_location': row.get('pen_location_normalized', row.get('Pen.location')),
        }
        
        if video_found:
            result.update({
                'linked': True,
                'camera_id': found_camera,
                'video_filename': video_found['video_filename'],
                'video_path': video_found['video_path'],
                'event_offset_sec': video_found['event_offset_sec'],
            })
            
            if found_camera not in camera_stats:
                camera_stats[found_camera] = 0
            camera_stats[found_camera] += 1
        else:
            result.update({
                'linked': False,
                'camera_id': None,
                'video_filename': None,
                'video_path': None,
                'event_offset_sec': None,
            })
            not_found.append({'idx': idx, 'reason': 'No matching video found'})
        
        results.append(result)
    
    return pd.DataFrame(results), not_found, camera_stats


def print_summary(df: pd.DataFrame, not_found: List, camera_stats: Dict):
    """Print linking summary."""
    console.print("\n" + "=" * 60)
    console.print("[bold]EVENT-VIDEO LINKING SUMMARY[/bold]")
    console.print("=" * 60)
    
    linked = df['linked'].sum()
    total = len(df)
    
    console.print(f"\n✅ Successfully linked: [green]{linked}[/green] / {total} ({linked/total*100:.1f}%)")
    console.print(f"❌ Not linked: [red]{len(not_found)}[/red]")
    
    if camera_stats:
        console.print("\n[bold]Events by Camera:[/bold]")
        table = Table()
        table.add_column("Camera", style="cyan")
        table.add_column("Events", justify="right")
        table.add_column("%", justify="right")
        
        for cam_id in sorted(camera_stats.keys()):
            count = camera_stats[cam_id]
            pct = count / linked * 100 if linked > 0 else 0
            table.add_row(f"Cam {cam_id}", str(count), f"{pct:.1f}%")
        
        console.print(table)
    
    console.print("\n[bold]Linked Events by Group & Day:[/bold]")
    linked_df = df[df['linked'] == True]
    if len(linked_df) > 0:
        cross = pd.crosstab(linked_df['group'], linked_df['day'], margins=True)
        console.print(cross.to_string())
    
    if not_found:
        console.print(f"\n[yellow]Sample of unlinked events:[/yellow]")
        for item in not_found[:5]:
            console.print(f"  • Event {item['idx']}: {item['reason']}")


@app.command()
def link(
    events: str = typer.Option("data/manifests/labeled_events.csv", help="Cleaned events CSV"),
    manifest: str = typer.Option("data/manifests/video_manifest.json", help="Video manifest JSON"),
    output: str = typer.Option("data/manifests/linked_events.csv", help="Output CSV path"),
):
    """Link labeled events to their video files."""
    console.print("[bold]🔗 Linking Events to Videos[/bold]\n")
    
    console.print(f"Loading events: {events}")
    events_df = pd.read_csv(events)
    console.print(f"  Loaded {len(events_df)} events")
    
    console.print(f"Loading manifest: {manifest}")
    with open(manifest, 'r') as f:
        manifest_data = json.load(f)
    
    linked_df, not_found, camera_stats = link_events_to_videos(events_df, manifest_data)
    
    print_summary(linked_df, not_found, camera_stats)
    
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    linked_df.to_csv(output_path, index=False)
    console.print(f"\n[green]✅ Linked events saved to: {output_path}[/green]")
    
    json_path = output_path.with_suffix('.json')
    linked_df.to_json(json_path, orient='records', indent=2)
    console.print(f"[green]✅ JSON version saved to: {json_path}[/green]")


if __name__ == "__main__":
    app()