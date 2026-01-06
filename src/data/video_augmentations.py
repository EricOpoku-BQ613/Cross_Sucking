# src/data/video_augmentations.py
"""
Video augmentation module inspired by FERAL paper.
Applies consistent augmentations across all frames in a clip.
"""

import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
from torchvision.transforms import functional as F
import random
import numpy as np
from typing import Optional, Tuple


class VideoTrivialAugment(nn.Module):
    """
    TrivialAugment for video clips.
    Applies the SAME random augmentation to ALL frames in a clip,
    preserving temporal coherence (critical for action recognition).
    
    From FERAL paper: "The same augmentation was applied consistently 
    across all frames within a video, preserving temporal coherence 
    while introducing diversity across video samples."
    """
    
    def __init__(self, num_magnitude_bins: int = 31):
        super().__init__()
        self.num_magnitude_bins = num_magnitude_bins
        
        # Available augmentations with their magnitude ranges
        # Format: (name, min_val, max_val, negate_possible)
        self.augmentations = [
            ("identity", 0, 0, False),
            ("brightness", 0.0, 0.5, True),
            ("contrast", 0.0, 0.5, True),
            ("saturation", 0.0, 0.5, True),
            ("sharpness", 0.0, 0.9, True),
            ("rotate", 0.0, 15.0, True),
            ("translate_x", 0.0, 0.1, True),
            ("translate_y", 0.0, 0.1, True),
            ("scale", 0.0, 0.2, True),
            ("hflip", 0, 0, False),
        ]
    
    def _get_magnitude(self, name: str, min_val: float, max_val: float) -> float:
        """Sample a random magnitude for the augmentation."""
        if min_val == max_val:
            return min_val
        magnitude_bin = random.randint(0, self.num_magnitude_bins - 1)
        return min_val + (max_val - min_val) * magnitude_bin / (self.num_magnitude_bins - 1)
    
    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to video clip.
        
        Args:
            clip: Tensor of shape (C, T, H, W) or (T, C, H, W)
        
        Returns:
            Augmented clip with same shape
        """
        # Handle different input formats
        if clip.ndim == 4:
            if clip.shape[0] in [1, 3]:  # (C, T, H, W)
                c_first = True
                C, T, H, W = clip.shape
            else:  # (T, C, H, W)
                c_first = False
                T, C, H, W = clip.shape
                clip = clip.permute(1, 0, 2, 3)  # -> (C, T, H, W)
        else:
            return clip  # Don't augment unexpected shapes
        
        # Randomly select one augmentation
        aug_name, min_val, max_val, can_negate = random.choice(self.augmentations)
        magnitude = self._get_magnitude(aug_name, min_val, max_val)
        
        if can_negate and random.random() < 0.5:
            magnitude = -magnitude
        
        # Apply the same augmentation to all frames
        augmented_frames = []
        for t in range(T):
            frame = clip[:, t, :, :]  # (C, H, W)
            frame = self._apply_augmentation(frame, aug_name, magnitude, H, W)
            augmented_frames.append(frame)
        
        clip = torch.stack(augmented_frames, dim=1)  # (C, T, H, W)
        
        # Restore original format if needed
        if not c_first:
            clip = clip.permute(1, 0, 2, 3)  # -> (T, C, H, W)
        
        return clip
    
    def _apply_augmentation(self, frame: torch.Tensor, name: str, 
                           magnitude: float, H: int, W: int) -> torch.Tensor:
        """Apply a single augmentation to one frame."""
        
        if name == "identity":
            return frame
        
        elif name == "brightness":
            return F.adjust_brightness(frame, 1.0 + magnitude)
        
        elif name == "contrast":
            return F.adjust_contrast(frame, 1.0 + magnitude)
        
        elif name == "saturation":
            return F.adjust_saturation(frame, 1.0 + magnitude)
        
        elif name == "sharpness":
            return F.adjust_sharpness(frame, 1.0 + magnitude)
        
        elif name == "rotate":
            return F.rotate(frame, magnitude)
        
        elif name == "translate_x":
            pixels = int(magnitude * W)
            return F.affine(frame, angle=0, translate=[pixels, 0], 
                          scale=1.0, shear=0)
        
        elif name == "translate_y":
            pixels = int(magnitude * H)
            return F.affine(frame, angle=0, translate=[0, pixels], 
                          scale=1.0, shear=0)
        
        elif name == "scale":
            return F.affine(frame, angle=0, translate=[0, 0], 
                          scale=1.0 + magnitude, shear=0)
        
        elif name == "hflip":
            return F.hflip(frame)
        
        return frame


class VideoMixUp(nn.Module):
    """
    MixUp regularization for video batches.
    
    From FERAL paper: "Each augmented sample was formed as a convex 
    combination of two videos and their corresponding label sequences."
    """
    
    def __init__(self, alpha: float = 0.2, p: float = 0.5):
        """
        Args:
            alpha: Beta distribution parameter (higher = more mixing)
            p: Probability of applying mixup
        """
        super().__init__()
        self.alpha = alpha
        self.p = p
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply MixUp to a batch.
        
        Args:
            x: Batch of clips (B, C, T, H, W) or (B, T, C, H, W)
            y: Batch of labels (B,)
        
        Returns:
            mixed_x, y_a, y_b, lam
        """
        if random.random() > self.p:
            return x, y, y, 1.0
        
        batch_size = x.size(0)
        
        # Sample mixing coefficient
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Random permutation for mixing partners
        index = torch.randperm(batch_size, device=x.device)
        
        # Mix inputs
        mixed_x = lam * x + (1 - lam) * x[index]
        
        # Return both label sets for loss computation
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Compute MixUp loss.
    
    Args:
        criterion: Loss function
        pred: Model predictions
        y_a: First set of labels
        y_b: Second set of labels (shuffled)
        lam: Mixing coefficient
    
    Returns:
        Mixed loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# Convenience function for creating augmentation pipeline
def get_train_augmentations(use_trivial: bool = True, 
                            use_mixup: bool = False,
                            mixup_alpha: float = 0.2) -> dict:
    """
    Get augmentation transforms for training.
    
    Args:
        use_trivial: Whether to use TrivialAugment
        use_mixup: Whether to use MixUp
        mixup_alpha: MixUp alpha parameter
    
    Returns:
        Dictionary with 'clip_transform' and optionally 'mixup'
    """
    transforms = {}
    
    if use_trivial:
        transforms['clip_transform'] = VideoTrivialAugment()
    else:
        transforms['clip_transform'] = None
    
    if use_mixup:
        transforms['mixup'] = VideoMixUp(alpha=mixup_alpha)
    else:
        transforms['mixup'] = None
    
    return transforms


if __name__ == "__main__":
    # Test the augmentations
    print("Testing VideoTrivialAugment...")
    
    # Create dummy video clip (C, T, H, W)
    clip = torch.rand(3, 16, 112, 112)
    
    aug = VideoTrivialAugment()
    augmented = aug(clip)
    
    print(f"Input shape: {clip.shape}")
    print(f"Output shape: {augmented.shape}")
    print(f"Values changed: {not torch.allclose(clip, augmented)}")
    
    # Test MixUp
    print("\nTesting VideoMixUp...")
    batch = torch.rand(4, 3, 16, 112, 112)
    labels = torch.tensor([0, 1, 0, 1])
    
    mixup = VideoMixUp(alpha=0.2, p=1.0)
    mixed, y_a, y_b, lam = mixup(batch, labels)
    
    print(f"Lambda: {lam:.3f}")
    print(f"y_a: {y_a}")
    print(f"y_b: {y_b}")