"""
Video Utilities
===============

Functions for video loading, frame extraction, and manipulation.
"""

import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


def get_video_info(video_path: str) -> Optional[Dict]:
    """
    Get video metadata using ffprobe.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dict with duration, fps, width, height, or None if failed
    """
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        str(video_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        video_stream = None
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_stream = stream
                break
        
        if video_stream:
            fps_str = video_stream.get('r_frame_rate', '0/1')
            if '/' in fps_str:
                num, den = map(int, fps_str.split('/'))
                fps = num / den if den != 0 else 0
            else:
                fps = float(fps_str)
            
            return {
                'duration': float(data.get('format', {}).get('duration', 0)),
                'fps': fps,
                'width': int(video_stream.get('width', 0)),
                'height': int(video_stream.get('height', 0)),
                'codec': video_stream.get('codec_name', ''),
                'total_frames': int(video_stream.get('nb_frames', 0)) if video_stream.get('nb_frames') else None
            }
    except (subprocess.CalledProcessError, json.JSONDecodeError, ValueError, KeyError):
        pass
    
    return None


def extract_frames(
    video_path: str,
    output_dir: str,
    fps: Optional[float] = None,
    start_sec: float = 0,
    duration_sec: Optional[float] = None,
    size: Optional[Tuple[int, int]] = None,
    format: str = 'jpg',
    quality: int = 2
) -> List[str]:
    """
    Extract frames from a video using ffmpeg.
    
    Args:
        video_path: Path to source video
        output_dir: Output directory for frames
        fps: Output frame rate (None = use video fps)
        start_sec: Start time in seconds
        duration_sec: Duration in seconds (None = entire video)
        size: Output size (width, height) or None for original
        format: Output format ('jpg' or 'png')
        quality: JPEG quality (1-31, lower is better)
        
    Returns:
        List of extracted frame paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_pattern = str(output_dir / f"frame_%06d.{format}")
    
    cmd = ['ffmpeg', '-y', '-ss', str(start_sec), '-i', str(video_path)]
    
    if duration_sec:
        cmd.extend(['-t', str(duration_sec)])
    
    # Build filter string
    filters = []
    if fps:
        filters.append(f'fps={fps}')
    if size:
        filters.append(f'scale={size[0]}:{size[1]}')
    
    if filters:
        cmd.extend(['-vf', ','.join(filters)])
    
    cmd.extend(['-q:v', str(quality), output_pattern])
    
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        frames = sorted(output_dir.glob(f"frame_*.{format}"))
        return [str(f) for f in frames]
    except subprocess.CalledProcessError as e:
        print(f"Error extracting frames: {e.stderr.decode() if e.stderr else 'Unknown error'}")
        return []


def extract_clip(
    video_path: str,
    output_path: str,
    start_sec: float,
    duration_sec: float,
    size: Optional[Tuple[int, int]] = None,
    fps: Optional[float] = None
) -> bool:
    """
    Extract a clip from a video.
    
    Args:
        video_path: Path to source video
        output_path: Output clip path
        start_sec: Start time in seconds
        duration_sec: Duration in seconds
        size: Output size (width, height) or None for original
        fps: Output fps or None for original
        
    Returns:
        True if successful
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start_sec),
        '-i', str(video_path),
        '-t', str(duration_sec)
    ]
    
    filters = []
    if size:
        filters.append(f'scale={size[0]}:{size[1]}')
    if fps:
        filters.append(f'fps={fps}')
    
    if filters:
        cmd.extend(['-vf', ','.join(filters)])
    
    cmd.extend([
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-an',
        str(output_path)
    ])
    
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def load_video_frames(
    video_path: str,
    num_frames: int = 16,
    size: Optional[Tuple[int, int]] = None
) -> Optional[np.ndarray]:
    """
    Load evenly-spaced frames from a video.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample
        size: Output size (height, width) or None for original
        
    Returns:
        Array of shape (num_frames, H, W, 3) or None if failed
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV required: pip install opencv-python")
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return None
    
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    
    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if size:
                frame = cv2.resize(frame, (size[1], size[0]))
            frames.append(frame)
    
    cap.release()
    
    if len(frames) == num_frames:
        return np.stack(frames)
    return None
