#!/usr/bin/env python3
"""
Audit Event Extraction - Find ALL issues with event clips
=========================================================

This script checks every event and reports:
1. Video file exists?
2. Offset within video duration?
3. Duration > 0?
4. macOS hidden file?
5. Can actually extract frames?

Usage:
    python scripts/audit_event_extraction.py \
        --manifest data/manifests/MASTER_FINAL_CLEAN_BOUNDARYFIX.csv \
        --output audit_results.csv
"""

import argparse
import csv
import re
from pathlib import Path
from datetime import datetime, timedelta
import cv2
from tqdm import tqdm
from collections import defaultdict


def parse_video_timestamps(filename):
    """Extract start/end timestamps from video filename."""
    pattern = r'(\d{14})_(\d{14})'
    match = re.search(pattern, str(filename))
    if match:
        start_str, end_str = match.groups()
        try:
            video_start = datetime.strptime(start_str, '%Y%m%d%H%M%S')
            video_end = datetime.strptime(end_str, '%Y%m%d%H%M%S')
            duration = (video_end - video_start).total_seconds()
            return video_start, video_end, duration
        except:
            return None, None, None
    return None, None, None


def parse_time_string(time_str):
    """Parse HH:MM:SS to timedelta."""
    try:
        parts = str(time_str).split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(float(parts[2])) if len(parts) > 2 else 0
        return timedelta(hours=hours, minutes=minutes, seconds=seconds)
    except:
        return None


def calculate_correct_offset(event_start_time, video_start_dt):
    """Calculate the correct offset from video start to event start."""
    event_td = parse_time_string(event_start_time)
    if event_td is None or video_start_dt is None:
        return None
    
    video_td = timedelta(
        hours=video_start_dt.hour,
        minutes=video_start_dt.minute,
        seconds=video_start_dt.second
    )
    
    offset = (event_td - video_td).total_seconds()
    
    # Handle day boundary
    if offset < 0:
        offset += 24 * 3600
    
    return offset


def get_video_info(video_path):
    """Get actual video duration and FPS using OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None, None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    return duration, fps, frame_count


def can_extract_frame(video_path, offset_sec):
    """Test if we can actually extract a frame at this offset."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, "Cannot open video"
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return False, "Invalid FPS"
    
    frame_num = int(offset_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return False, "Frame read failed"
    if frame is None or frame.size == 0:
        return False, "Empty frame"
    
    return True, "OK"


def audit_manifest(manifest_path, output_path=None, test_extraction=False, limit=None):
    """Audit all events in manifest."""
    
    # Read manifest
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        events = list(reader)
    
    if limit:
        events = events[:limit]
    
    print(f"Auditing {len(events)} events...")
    print("=" * 70)
    
    results = []
    issues = defaultdict(list)
    
    for event in tqdm(events):
        event_idx = event.get('event_idx', 'unknown')
        video_filename = event.get('video_filename', '')
        video_path = event.get('video_path', '')
        start_time = event.get('start_time', '')
        end_time = event.get('end_time', '')
        duration_sec = float(event.get('duration_sec', 0) or 0)
        event_offset_sec = event.get('event_offset_sec', '')
        behavior = event.get('behavior', event.get('_primary_label', 'unknown'))
        
        result = {
            'event_idx': event_idx,
            'behavior': behavior,
            'video_filename': video_filename,
            'start_time': start_time,
            'end_time': end_time,
            'duration_sec': duration_sec,
            'stored_offset': event_offset_sec,
            'issue': [],
            'status': 'OK'
        }
        
        # Check 1: macOS hidden file
        if video_filename.startswith('._'):
            result['issue'].append('MACOS_HIDDEN_FILE')
            result['status'] = 'FAIL'
            issues['macos_hidden'].append(event_idx)
        
        # Check 2: Zero duration
        if duration_sec <= 0:
            result['issue'].append('ZERO_DURATION')
            issues['zero_duration'].append(event_idx)
        
        # Check 3: Video file exists
        video_exists = Path(video_path).exists() if video_path else False
        if not video_exists:
            result['issue'].append('VIDEO_NOT_FOUND')
            result['status'] = 'FAIL'
            issues['video_not_found'].append(event_idx)
            results.append(result)
            continue
        
        # Check 4: Parse video timestamps from filename
        video_start_dt, video_end_dt, video_duration_from_filename = parse_video_timestamps(video_filename)
        
        if video_start_dt is None:
            result['issue'].append('CANNOT_PARSE_FILENAME')
            issues['parse_error'].append(event_idx)
        
        # Check 5: Calculate correct offset
        calc_offset = None
        if video_start_dt and start_time:
            calc_offset = calculate_correct_offset(start_time, video_start_dt)
            result['calculated_offset'] = calc_offset
            
            # Compare with stored offset
            if event_offset_sec:
                try:
                    stored = float(event_offset_sec)
                    if calc_offset and abs(calc_offset - stored) > 5:
                        result['issue'].append(f'OFFSET_MISMATCH(stored={stored:.0f},calc={calc_offset:.0f})')
                        issues['offset_mismatch'].append(event_idx)
                except:
                    pass
        
        # Check 6: Get actual video duration
        if video_exists:
            actual_duration, fps, frame_count = get_video_info(video_path)
            result['video_duration'] = actual_duration
            result['video_fps'] = fps
            
            # Check if offset exceeds video duration
            offset_to_check = calc_offset if calc_offset else (float(event_offset_sec) if event_offset_sec else 0)
            
            if actual_duration and offset_to_check > actual_duration:
                result['issue'].append(f'OFFSET_EXCEEDS_DURATION(offset={offset_to_check:.0f},video={actual_duration:.0f})')
                result['status'] = 'FAIL'
                issues['offset_exceeds'].append(event_idx)
            
            # Check 7: Actually try to extract frame (optional, slow)
            if test_extraction and result['status'] != 'FAIL':
                can_extract, extract_msg = can_extract_frame(video_path, offset_to_check)
                result['extraction_test'] = extract_msg
                if not can_extract:
                    result['issue'].append(f'EXTRACTION_FAILED({extract_msg})')
                    result['status'] = 'FAIL'
                    issues['extraction_failed'].append(event_idx)
        
        # Final status
        if result['issue'] and result['status'] == 'OK':
            result['status'] = 'WARN'
        
        result['issue'] = '; '.join(result['issue']) if result['issue'] else 'None'
        results.append(result)
    
    # Summary
    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)
    
    total = len(results)
    ok_count = sum(1 for r in results if r['status'] == 'OK')
    warn_count = sum(1 for r in results if r['status'] == 'WARN')
    fail_count = sum(1 for r in results if r['status'] == 'FAIL')
    
    print(f"Total events:        {total}")
    print(f"  OK:                {ok_count} ({100*ok_count/total:.1f}%)")
    print(f"  WARN:              {warn_count} ({100*warn_count/total:.1f}%)")
    print(f"  FAIL:              {fail_count} ({100*fail_count/total:.1f}%)")
    
    print(f"\nIssue Breakdown:")
    print(f"  macOS hidden:      {len(issues['macos_hidden'])}")
    print(f"  Zero duration:     {len(issues['zero_duration'])}")
    print(f"  Video not found:   {len(issues['video_not_found'])}")
    print(f"  Parse error:       {len(issues['parse_error'])}")
    print(f"  Offset mismatch:   {len(issues['offset_mismatch'])}")
    print(f"  Offset exceeds:    {len(issues['offset_exceeds'])}")
    if test_extraction:
        print(f"  Extraction failed: {len(issues['extraction_failed'])}")
    
    # Class distribution of failures
    print(f"\nFailures by class:")
    fail_by_class = defaultdict(int)
    for r in results:
        if r['status'] == 'FAIL':
            fail_by_class[r['behavior']] += 1
    for cls, count in sorted(fail_by_class.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {count}")
    
    # Save to CSV
    if output_path:
        fieldnames = ['event_idx', 'behavior', 'status', 'issue', 'video_filename', 
                      'start_time', 'duration_sec', 'stored_offset', 'calculated_offset',
                      'video_duration']
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\nResults saved to: {output_path}")
        
        # Also save failures only
        fail_path = Path(output_path).stem + '_FAILURES.csv'
        failures = [r for r in results if r['status'] == 'FAIL']
        with open(fail_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(failures)
        print(f"Failures saved to: {fail_path}")
    
    return results, issues


def main():
    parser = argparse.ArgumentParser(description='Audit event extraction')
    parser.add_argument('--manifest', required=True, help='Manifest CSV path')
    parser.add_argument('--output', default='audit_results.csv', help='Output CSV path')
    parser.add_argument('--test-extraction', action='store_true', 
                        help='Actually test frame extraction (slower)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of events to audit')
    args = parser.parse_args()
    
    audit_manifest(
        args.manifest,
        args.output,
        test_extraction=args.test_extraction,
        limit=args.limit
    )


if __name__ == '__main__':
    main()