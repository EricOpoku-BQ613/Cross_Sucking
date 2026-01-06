# src/data/clip_transforms.py
"""
Temporally Consistent Clip Transforms for Calf Behavior Recognition
====================================================================

Merges:
- Your domain-tuned augmentations (barn cameras, night vision, calves)
- FERAL's temporal consistency (same params for all frames in a clip)

Key principles:
1. CONSERVATIVE geometric transforms (avoid distorting behavior)
2. AGGRESSIVE color/lighting transforms (handle night/day variation)
3. TEMPORAL CONSISTENCY (same random params for all frames)
"""

from __future__ import annotations

import random
from typing import Optional, Callable, Literal, Tuple

import numpy as np
import torch

try:
    import torchvision.transforms.functional as TF
    from PIL import Image, ImageFilter, ImageOps
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
    TF = None
    Image = None


Mode = Literal["train", "eval"]


class TemporallyConsistentTransform:
    """
    Your augmentations + temporal consistency from FERAL.
    
    Designed for barn camera footage with:
    - Night/low-light conditions
    - Distance/scale variation  
    - Line artifacts from compression
    - Multiple camera viewpoints
    
    All random parameters are sampled ONCE and applied to ALL frames.
    """
    
    def __init__(
        self,
        mode: Mode = "train",
        size: int = 112,
        
        # === GEOMETRIC (Conservative - don't distort behavior) ===
        # Crop: handles distance variation
        crop_scale: Tuple[float, float] = (0.75, 1.0),  # Your setting
        crop_ratio: Tuple[float, float] = (0.9, 1.1),   # Your setting
        
        # Affine: very mild shifts (your settings)
        max_rotation: float = 3.0,        # degrees (yours was 3)
        max_translate: float = 0.02,      # fraction (yours was 0.02)
        scale_range: Tuple[float, float] = (0.92, 1.06),  # Your setting
        
        p_hflip: float = 0.5,
        
        # === COLOR/LIGHTING (Aggressive - handle night/day) ===
        p_color_jitter: float = 0.8,
        brightness: float = 0.45,    # Your setting
        contrast: float = 0.45,      # Your setting
        saturation: float = 0.15,    # Your setting (conservative)
        hue: float = 0.02,           # Your setting (conservative)
        
        p_autocontrast: float = 0.2,  # Your setting - important for night
        p_grayscale: float = 0.05,    # Your setting - camera differences
        
        # === ARTIFACT ROBUSTNESS ===
        p_blur: float = 0.3,
        blur_sigma: Tuple[float, float] = (0.1, 1.2),  # Your setting
        
        # === OCCLUSION (Mild - other calves/clutter) ===
        p_erasing: float = 0.15,      # Your setting
        erasing_scale: Tuple[float, float] = (0.02, 0.08),  # Your setting (small)
    ):
        self.mode = mode
        self.size = size
        self.is_train = (mode == "train")
        
        # Geometric (conservative)
        self.crop_scale = crop_scale
        self.crop_ratio = crop_ratio
        self.max_rotation = max_rotation
        self.max_translate = max_translate
        self.scale_range = scale_range
        self.p_hflip = p_hflip
        
        # Color (can be aggressive)
        self.p_color_jitter = p_color_jitter
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p_autocontrast = p_autocontrast
        self.p_grayscale = p_grayscale
        
        # Artifacts
        self.p_blur = p_blur
        self.blur_sigma = blur_sigma
        
        # Occlusion
        self.p_erasing = p_erasing
        self.erasing_scale = erasing_scale
    
    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        """
        Apply temporally consistent transforms.
        
        Args:
            clip: (C, T, H, W) float tensor in [0, 1]
        
        Returns:
            Transformed clip (C, T, H, W) in [0, 1]
        """
        if not HAS_TORCHVISION:
            return self._resize_clip(clip)
        
        C, T, H, W = clip.shape
        
        # Eval: just resize, no augmentation
        if not self.is_train:
            return self._resize_clip(clip)
        
        # =============================================
        # SAMPLE ALL RANDOM PARAMETERS ONCE
        # (Applied consistently to all frames)
        # =============================================
        
        # Geometric decisions
        do_hflip = random.random() < self.p_hflip
        crop_params = self._sample_crop_params(H, W)
        affine_params = self._sample_affine_params()
        
        # Color decisions
        do_color_jitter = random.random() < self.p_color_jitter
        do_autocontrast = random.random() < self.p_autocontrast
        do_grayscale = random.random() < self.p_grayscale
        
        # Sample color jitter factors once
        if do_color_jitter:
            brightness_factor = random.uniform(
                max(0, 1 - self.brightness), 
                1 + self.brightness
            )
            contrast_factor = random.uniform(
                max(0, 1 - self.contrast), 
                1 + self.contrast
            )
            saturation_factor = random.uniform(
                max(0, 1 - self.saturation), 
                1 + self.saturation
            )
            hue_factor = random.uniform(-self.hue, self.hue)
        
        # Artifact decisions
        do_blur = random.random() < self.p_blur
        if do_blur:
            blur_sigma = random.uniform(self.blur_sigma[0], self.blur_sigma[1])
        
        # Erasing decisions (sample region once)
        do_erasing = random.random() < self.p_erasing
        if do_erasing:
            erase_params = self._sample_erase_params(self.size, self.size)
        
        # =============================================
        # APPLY SAME TRANSFORMS TO ALL FRAMES
        # =============================================
        
        frames_out = []
        
        for t in range(T):
            frame = clip[:, t, :, :]  # (C, H, W)
            
            # Convert to PIL
            frame_np = (frame.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(frame_np)
            
            # --- GEOMETRIC (Conservative) ---
            
            # 1. Resized crop (handles distance variation)
            img = TF.resized_crop(
                img,
                top=crop_params['top'],
                left=crop_params['left'],
                height=crop_params['height'],
                width=crop_params['width'],
                size=(self.size, self.size),
            )
            
            # 2. Affine (very mild rotation/translation/scale)
            if affine_params['apply']:
                img = TF.affine(
                    img,
                    angle=affine_params['angle'],
                    translate=[affine_params['tx'], affine_params['ty']],
                    scale=affine_params['scale'],
                    shear=0,
                    fill=0,
                )
            
            # 3. Horizontal flip
            if do_hflip:
                img = TF.hflip(img)
            
            # --- COLOR/LIGHTING (Can be aggressive) ---
            
            # 4. Autocontrast (critical for night footage)
            if do_autocontrast:
                img = ImageOps.autocontrast(img)
            
            # 5. Color jitter (same factors for all frames)
            if do_color_jitter:
                img = TF.adjust_brightness(img, brightness_factor)
                img = TF.adjust_contrast(img, contrast_factor)
                img = TF.adjust_saturation(img, saturation_factor)
                img = TF.adjust_hue(img, hue_factor)
            
            # 6. Grayscale (camera differences)
            if do_grayscale:
                img = TF.rgb_to_grayscale(img, num_output_channels=3)
            
            # --- ARTIFACT ROBUSTNESS ---
            
            # 7. Gaussian blur (line artifacts, compression)
            if do_blur:
                img = img.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
            
            # Convert to tensor
            frame_out = TF.to_tensor(img)  # (C, H, W) in [0, 1]
            
            # --- OCCLUSION ---
            
            # 8. Random erasing (other calves, clutter)
            # Applied after tensor conversion
            if do_erasing and erase_params is not None:
                frame_out = self._apply_erasing(frame_out, erase_params)
            
            frames_out.append(frame_out)
        
        # Stack back to (C, T, H, W)
        clip_out = torch.stack(frames_out, dim=1)
        
        return clip_out
    
    def _sample_crop_params(self, H: int, W: int) -> dict:
        """Sample random crop parameters (conservative)."""
        area = H * W
        
        for _ in range(10):
            target_area = random.uniform(self.crop_scale[0], self.crop_scale[1]) * area
            aspect_ratio = random.uniform(self.crop_ratio[0], self.crop_ratio[1])
            
            w = int(round(np.sqrt(target_area * aspect_ratio)))
            h = int(round(np.sqrt(target_area / aspect_ratio)))
            
            if 0 < w <= W and 0 < h <= H:
                top = random.randint(0, H - h)
                left = random.randint(0, W - w)
                return {'top': top, 'left': left, 'height': h, 'width': w}
        
        # Fallback: center crop at minimum scale
        scale = self.crop_scale[0]
        h = int(H * np.sqrt(scale))
        w = int(W * np.sqrt(scale))
        top = (H - h) // 2
        left = (W - w) // 2
        return {'top': top, 'left': left, 'height': h, 'width': w}
    
    def _sample_affine_params(self) -> dict:
        """Sample affine parameters (very mild)."""
        # Only apply affine sometimes to avoid over-augmenting
        if random.random() > 0.5:
            return {'apply': False}
        
        angle = random.uniform(-self.max_rotation, self.max_rotation)
        tx = int(random.uniform(-self.max_translate, self.max_translate) * self.size)
        ty = int(random.uniform(-self.max_translate, self.max_translate) * self.size)
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        
        return {
            'apply': True,
            'angle': angle,
            'tx': tx,
            'ty': ty,
            'scale': scale,
        }
    
    def _sample_erase_params(self, H: int, W: int) -> Optional[dict]:
        """Sample erasing region (small, to avoid destroying behavior cues)."""
        area = H * W
        
        for _ in range(10):
            target_area = random.uniform(self.erasing_scale[0], self.erasing_scale[1]) * area
            aspect_ratio = random.uniform(0.3, 3.3)
            
            h = int(round(np.sqrt(target_area / aspect_ratio)))
            w = int(round(np.sqrt(target_area * aspect_ratio)))
            
            if h < H and w < W:
                top = random.randint(0, H - h)
                left = random.randint(0, W - w)
                return {'top': top, 'left': left, 'height': h, 'width': w}
        
        return None
    
    def _apply_erasing(self, frame: torch.Tensor, params: dict) -> torch.Tensor:
        """Apply random erasing to frame."""
        frame = frame.clone()
        frame[
            :,
            params['top']:params['top'] + params['height'],
            params['left']:params['left'] + params['width']
        ] = 0
        return frame
    
    def _resize_clip(self, clip: torch.Tensor) -> torch.Tensor:
        """Resize all frames (for eval or fallback)."""
        C, T, H, W = clip.shape
        
        if H == self.size and W == self.size:
            return clip
        
        clip_reshaped = clip.permute(1, 0, 2, 3)  # (T, C, H, W)
        clip_resized = torch.nn.functional.interpolate(
            clip_reshaped,
            size=(self.size, self.size),
            mode='bilinear',
            align_corners=False,
        )
        return clip_resized.permute(1, 0, 2, 3)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_clip_transform(
    mode: Mode = "train",
    size: int = 112,
) -> TemporallyConsistentTransform:
    """
    Get the recommended clip transform for calf behavior recognition.
    
    Uses your domain-tuned augmentation parameters with temporal consistency.
    """
    return TemporallyConsistentTransform(mode=mode, size=size)


def get_clip_transform_light(
    mode: Mode = "train",
    size: int = 112,
) -> TemporallyConsistentTransform:
    """
    Lighter augmentation variant (if training is unstable).
    
    Reduces all probabilities and strengths.
    """
    return TemporallyConsistentTransform(
        mode=mode,
        size=size,
        # Reduced geometric
        crop_scale=(0.85, 1.0),
        max_rotation=2.0,
        max_translate=0.01,
        scale_range=(0.95, 1.05),
        # Reduced color
        p_color_jitter=0.5,
        brightness=0.3,
        contrast=0.3,
        saturation=0.1,
        p_autocontrast=0.1,
        p_grayscale=0.02,
        # Reduced artifacts
        p_blur=0.2,
        p_erasing=0.1,
    )


def get_clip_transform_strong(
    mode: Mode = "train",
    size: int = 112,
) -> TemporallyConsistentTransform:
    """
    Stronger augmentation variant (for SSL or if overfitting).
    
    More aggressive color/lighting, same conservative geometry.
    """
    return TemporallyConsistentTransform(
        mode=mode,
        size=size,
        # Same conservative geometric
        crop_scale=(0.70, 1.0),  # Slightly more aggressive crop
        # More aggressive color
        p_color_jitter=0.9,
        brightness=0.55,
        contrast=0.55,
        saturation=0.25,
        hue=0.04,
        p_autocontrast=0.3,
        p_grayscale=0.1,
        # More blur
        p_blur=0.4,
        blur_sigma=(0.1, 1.5),
        # Same mild erasing
        p_erasing=0.2,
    )


# =============================================================================
# INTEGRATION HELPER
# =============================================================================

class ClipTransformWrapper:
    """
    Wrapper to integrate with your existing dataset.
    
    Usage in LabeledEventDataset:
        self.transform = ClipTransformWrapper(mode=self.mode)
        
        def __getitem__(self, idx):
            clip = self._load_clip(...)  # Returns (C, T, H, W) or (T, H, W, C)
            clip = self.transform(clip)
            return clip, label
    """
    
    def __init__(self, mode: Mode = "train", size: int = 112, variant: str = "default"):
        if variant == "light":
            self.transform = get_clip_transform_light(mode, size)
        elif variant == "strong":
            self.transform = get_clip_transform_strong(mode, size)
        else:
            self.transform = get_clip_transform(mode, size)
        
        self.size = size
    
    def __call__(self, clip):
        """
        Handle various input formats and apply transform.
        
        Accepts:
            - torch.Tensor (C, T, H, W) in [0, 1]
            - torch.Tensor (T, H, W, C) in [0, 255]
            - np.ndarray (T, H, W, C) uint8
        
        Returns:
            torch.Tensor (C, T, H, W) in [0, 1]
        """
        # Convert to tensor if needed
        if isinstance(clip, np.ndarray):
            clip = torch.from_numpy(clip)
        
        # Handle (T, H, W, C) format
        if clip.ndim == 4 and clip.shape[-1] in (1, 3):
            clip = clip.permute(3, 0, 1, 2)  # -> (C, T, H, W)
        
        # Handle grayscale
        if clip.shape[0] == 1:
            clip = clip.repeat(3, 1, 1, 1)
        
        # Scale to [0, 1] if needed
        if clip.dtype == torch.uint8:
            clip = clip.float() / 255.0
        elif clip.max() > 1.5:
            clip = clip.float() / 255.0
        
        # Apply transform
        return self.transform(clip)