# scripts/inference_aggregated.py
"""
Multi-Clip Inference Aggregation for Event Classification
==========================================================

From FERAL paper: "Overlapping predictions are ensembled to produce 
stable frame-level probabilities."

For event classification, we sample multiple clips from each event
(different temporal offsets) and aggregate logits for more robust predictions.

This typically improves tail recall without destroying ear precision.

Usage:
    python scripts/inference_aggregated.py \
        --checkpoint runs/sup_binary_stable/best.ckpt \
        --manifest data/manifests/test.csv \
        --output runs/sup_binary_stable/predictions_aggregated.csv \
        --config configs/train_binary_stable.yaml \
        --clips-per-event 5 \
        --aggregation mean
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    import yaml
except ImportError:
    raise ImportError("Please install pyyaml: pip install pyyaml")

# Local imports - adjust path as needed
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.backbone import VideoBackbone
from src.models.classifier import LinearHead, MLPHead, VideoClassifier


class MultiClipEventDataset(Dataset):
    """
    Dataset that samples multiple clips per event for aggregated inference.
    
    For each event, samples N clips at different temporal offsets:
    - Clip 0: Start from frame 0
    - Clip 1: Start from frame offset
    - Clip 2: Start from frame 2*offset
    - etc.
    
    Also supports overlapping windows (like FERAL's 50% overlap).
    """
    
    def __init__(
        self,
        manifest_csv: str,
        clip_len: int = 16,
        fps: int = 12,
        clips_per_event: int = 5,
        overlap: float = 0.5,  # 0.5 = 50% overlap like FERAL
        transform=None,
    ):
        self.df = pd.read_csv(manifest_csv)
        self.clip_len = clip_len
        self.fps = fps
        self.clips_per_event = clips_per_event
        self.overlap = overlap
        self.transform = transform
        
        # Build index: (event_idx, clip_idx) -> row
        self.samples = []
        for idx, row in self.df.iterrows():
            event_id = row.get('event_id', row.get('clip_id', f'event_{idx}'))
            video_path = row['video_path']
            behavior = row['behavior']
            
            # Get video duration to compute valid offsets
            duration_sec = row.get('duration_sec', None)
            if duration_sec is None:
                # Estimate from clip_len
                duration_sec = clip_len / fps * 2  # Assume at least 2x clip length
            
            # Compute clip offsets
            clip_duration = clip_len / fps
            stride = clip_duration * (1 - overlap)
            
            # Generate clip start times
            max_start = max(0, duration_sec - clip_duration)
            
            if max_start == 0:
                # Short event - just use one clip at start
                offsets = [0.0]
            else:
                # Generate evenly spaced offsets
                offsets = np.linspace(0, max_start, clips_per_event).tolist()
            
            for clip_idx, start_sec in enumerate(offsets[:clips_per_event]):
                self.samples.append({
                    'event_id': event_id,
                    'event_idx': idx,
                    'clip_idx': clip_idx,
                    'video_path': video_path,
                    'behavior': behavior,
                    'start_sec': start_sec,
                })
        
        print(f"[MultiClipDataset] {len(self.df)} events -> {len(self.samples)} clips")
        print(f"[MultiClipDataset] clips_per_event={clips_per_event}, overlap={overlap}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load video clip starting at start_sec
        video_path = sample['video_path']
        start_sec = sample['start_sec']
        
        # Use your existing video loading logic
        # This is a simplified version - adapt to your actual loader
        try:
            import decord
            decord.bridge.set_bridge('torch')
            
            vr = decord.VideoReader(video_path)
            total_frames = len(vr)
            video_fps = vr.get_avg_fps()
            
            # Compute frame indices
            start_frame = int(start_sec * video_fps)
            
            # Sample clip_len frames at target fps
            frame_step = max(1, int(video_fps / self.fps))
            frame_indices = []
            
            for i in range(self.clip_len):
                frame_idx = start_frame + i * frame_step
                frame_idx = min(frame_idx, total_frames - 1)  # Clamp
                frame_indices.append(frame_idx)
            
            frames = vr.get_batch(frame_indices)  # (T, H, W, C)
            frames = frames.permute(3, 0, 1, 2).float() / 255.0  # (C, T, H, W)
            
            # Resize if needed
            if frames.shape[2] != 112 or frames.shape[3] != 112:
                frames = F.interpolate(
                    frames.permute(1, 0, 2, 3),  # (T, C, H, W)
                    size=(112, 112),
                    mode='bilinear',
                    align_corners=False
                ).permute(1, 0, 2, 3)  # Back to (C, T, H, W)
            
        except Exception as e:
            # Fallback: return zeros (will be filtered)
            print(f"[WARNING] Failed to load {video_path}: {e}")
            frames = torch.zeros(3, self.clip_len, 112, 112)
        
        if self.transform:
            frames = self.transform(frames)
        
        return {
            'video': frames,
            'event_id': sample['event_id'],
            'event_idx': sample['event_idx'],
            'clip_idx': sample['clip_idx'],
            'behavior': sample['behavior'],
        }


def aggregate_predictions(
    clip_logits: Dict[str, List[np.ndarray]],
    clip_labels: Dict[str, str],
    method: str = 'mean'
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, str]]:
    """
    Aggregate logits across multiple clips per event.
    
    Args:
        clip_logits: {event_id: [logits_clip0, logits_clip1, ...]}
        clip_labels: {event_id: behavior_label}
        method: 'mean', 'max', 'softmax_mean'
    
    Returns:
        event_logits: {event_id: aggregated_logits}
        event_probs: {event_id: softmax(aggregated_logits)}
        event_labels: {event_id: behavior_label}
    """
    event_logits = {}
    event_probs = {}
    event_labels = {}
    
    for event_id, logits_list in clip_logits.items():
        logits_array = np.stack(logits_list)  # (N_clips, N_classes)
        
        if method == 'mean':
            # Average logits, then softmax
            agg_logits = logits_array.mean(axis=0)
            
        elif method == 'max':
            # Max logits per class, then softmax
            agg_logits = logits_array.max(axis=0)
            
        elif method == 'softmax_mean':
            # Softmax each clip, then average probabilities
            probs_array = np.exp(logits_array) / np.exp(logits_array).sum(axis=1, keepdims=True)
            agg_probs = probs_array.mean(axis=0)
            # Convert back to logits for consistency
            agg_logits = np.log(agg_probs + 1e-8)
            
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        # Compute final probabilities
        agg_probs = np.exp(agg_logits) / np.exp(agg_logits).sum()
        
        event_logits[event_id] = agg_logits
        event_probs[event_id] = agg_probs
        event_labels[event_id] = clip_labels[event_id]
    
    return event_logits, event_probs, event_labels


def run_inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    label_map: Dict[int, str],
) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, str]]:
    """
    Run inference on all clips, collecting logits per event.
    """
    model.eval()
    
    clip_logits = defaultdict(list)
    clip_labels = {}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running inference"):
            videos = batch['video'].to(device)
            event_ids = batch['event_id']
            behaviors = batch['behavior']
            
            # Forward pass
            out = model(videos)  # (B, num_classes)
            logits = out.logits 
            logits_np = logits.cpu().numpy()
            
            # Collect per event
            for i, event_id in enumerate(event_ids):
                clip_logits[event_id].append(logits_np[i])
                clip_labels[event_id] = behaviors[i]
    
    return dict(clip_logits), clip_labels


def load_model(checkpoint_path: str, config: dict, device: torch.device):
    """Load model from checkpoint."""
    
    num_classes = config['labels']['num_classes']
    
    backbone = VideoBackbone(
        name=config['model']['backbone'],
        pretrained=False,  # Will load from checkpoint
        dropout=float(config['model'].get('backbone_dropout', 0.0)),
    )
    
    head_name = config['model'].get('head', 'linear')
    if head_name == 'linear':
        head = LinearHead(
            backbone.feat_dim, 
            num_classes=num_classes,
            dropout=float(config['model'].get('head_dropout', 0.2))
        )
    elif head_name == 'mlp':
        head = MLPHead(
            backbone.feat_dim,
            num_classes=num_classes,
            hidden_dim=int(config['model'].get('head_hidden_dim', 512)),
            dropout=float(config['model'].get('head_dropout', 0.3)),
        )
    else:
        raise ValueError(f"Unknown head: {head_name}")
    
    model = VideoClassifier(backbone, head)
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    
    model = model.to(device)
    model.eval()
    
    print(f"[Model] Loaded from {checkpoint_path}")
    return model


def main():
    parser = argparse.ArgumentParser(description='Multi-clip aggregated inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--manifest', type=str, required=True,
                       help='Path to test/val manifest CSV')
    parser.add_argument('--output', type=str, required=True,
                       help='Output predictions CSV')
    parser.add_argument('--config', type=str, required=True,
                       help='Training config YAML')
    parser.add_argument('--clips-per-event', type=int, default=5,
                       help='Number of clips to sample per event')
    parser.add_argument('--overlap', type=float, default=0.5,
                       help='Overlap between clips (0.5 = 50%% like FERAL)')
    parser.add_argument('--aggregation', type=str, default='mean',
                       choices=['mean', 'max', 'softmax_mean'],
                       help='Aggregation method for logits')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")
    
    # Build label map
    class_names = config['labels']['class_names']
    label_map = {i: name for i, name in enumerate(class_names)}
    reverse_label_map = {name.lower(): i for i, name in enumerate(class_names)}
    print(f"[Labels] {label_map}")
    
    # Load model
    model = load_model(args.checkpoint, config, device)
    
    # Create multi-clip dataset
    dataset = MultiClipEventDataset(
        manifest_csv=args.manifest,
        clip_len=config['clip']['clip_len'],
        fps=config['clip']['fps'],
        clips_per_event=args.clips_per_event,
        overlap=args.overlap,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # Run inference
    clip_logits, clip_labels = run_inference(model, dataloader, device, label_map)
    
    print(f"\n[Inference] Processed {len(clip_logits)} events")
    
    # Aggregate predictions
    event_logits, event_probs, event_labels = aggregate_predictions(
        clip_logits, clip_labels, method=args.aggregation
    )
    
    print(f"[Aggregation] Method: {args.aggregation}")
    
    # Build output DataFrame
    records = []
    for event_id in event_logits:
        probs = event_probs[event_id]
        true_label = event_labels[event_id].lower()
        pred_idx = probs.argmax()
        pred_label = label_map[pred_idx]
        
        record = {
            'clip_id': event_id,
            'true_label': true_label,
            'pred_label': pred_label,
        }
        
        # Add probability columns
        for i, name in label_map.items():
            record[f'prob_{name}'] = probs[i]
        
        # Add aggregation metadata
        record['n_clips'] = len(clip_logits[event_id])
        record['aggregation'] = args.aggregation
        
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"AGGREGATED INFERENCE RESULTS")
    print(f"{'='*60}")
    print(f"Events: {len(df)}")
    print(f"Clips per event: {args.clips_per_event}")
    print(f"Aggregation: {args.aggregation}")
    print(f"\nClass distribution (true):")
    print(df['true_label'].value_counts())
    print(f"\nClass distribution (predicted):")
    print(df['pred_label'].value_counts())
    
    # Quick accuracy
    correct = (df['true_label'] == df['pred_label']).sum()
    print(f"\nAccuracy: {correct}/{len(df)} = {correct/len(df):.4f}")
    
    # Per-class recall
    for label in class_names:
        label_lower = label.lower()
        mask = df['true_label'] == label_lower
        if mask.sum() > 0:
            recall = (df.loc[mask, 'pred_label'] == label_lower).mean()
            print(f"{label} recall: {recall:.4f} (n={mask.sum()})")
    
    print(f"\nSaved to: {output_path}")
    print(f"{'='*60}")
    
    return df


if __name__ == '__main__':
    main()