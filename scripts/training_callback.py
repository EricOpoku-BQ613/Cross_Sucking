"""
Training Callbacks for Cross-Sucking Detection
===============================================
Includes:
- OverfittingDetector: Monitor train/val gap
- CalibrationMonitor: Track ECE during training
- EMACallback: Exponential moving average of weights

Usage:
    from training_callbacks import OverfittingDetector, CalibrationMonitor
    
    # In your training loop:
    detector = OverfittingDetector(patience=5, loss_threshold=0.3)
    calibration = CalibrationMonitor()
    
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(...)
        val_loss, val_acc = validate(...)
        
        # Check overfitting
        status = detector.update(train_loss, val_loss, train_acc, val_acc)
        if status['is_overfitting']:
            print(f"⚠️ Overfitting detected: {status['message']}")
        
        # Track calibration
        calibration.update(val_logits, val_labels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from copy import deepcopy
import warnings


class OverfittingDetector:
    """
    Monitor training for overfitting signals.
    
    Signals monitored:
    1. Train loss << Val loss (generalization gap)
    2. Val loss increasing while train loss decreasing
    3. Train accuracy >> Val accuracy
    """
    
    def __init__(
        self,
        patience: int = 5,
        loss_threshold: float = 0.3,
        acc_threshold: float = 0.15,
        min_epochs: int = 5
    ):
        """
        Args:
            patience: Number of epochs to wait before flagging overfit
            loss_threshold: Max acceptable train-val loss gap
            acc_threshold: Max acceptable train-val accuracy gap
            min_epochs: Don't check overfitting before this epoch
        """
        self.patience = patience
        self.loss_threshold = loss_threshold
        self.acc_threshold = acc_threshold
        self.min_epochs = min_epochs
        
        # History
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        # Tracking
        self.best_val_loss = float('inf')
        self.epochs_since_improvement = 0
        self.overfit_counter = 0
    
    def update(
        self,
        train_loss: float,
        val_loss: float,
        train_acc: float = None,
        val_acc: float = None,
        epoch: int = None
    ) -> dict:
        """
        Update with new epoch metrics.
        
        Returns:
            dict with overfitting status and recommendations
        """
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        if train_acc is not None:
            self.train_accs.append(train_acc)
        if val_acc is not None:
            self.val_accs.append(val_acc)
        
        epoch = epoch or len(self.train_losses)
        
        # Initialize result
        result = {
            'epoch': epoch,
            'is_overfitting': False,
            'severity': 'none',
            'message': '',
            'metrics': {
                'loss_gap': train_loss - val_loss,
                'acc_gap': (train_acc - val_acc) if train_acc and val_acc else None,
            }
        }
        
        # Skip early epochs
        if epoch < self.min_epochs:
            result['message'] = f"Warmup period (epoch {epoch}/{self.min_epochs})"
            return result
        
        # Check for improvement
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.epochs_since_improvement = 0
        else:
            self.epochs_since_improvement += 1
        
        # Overfitting checks
        issues = []
        
        # 1. Loss gap check (train << val means overfitting)
        loss_gap = val_loss - train_loss
        if loss_gap > self.loss_threshold:
            issues.append(f"Loss gap {loss_gap:.3f} > {self.loss_threshold}")
        
        # 2. Val loss increasing check
        if len(self.val_losses) >= 3:
            recent_val = self.val_losses[-3:]
            if all(recent_val[i] < recent_val[i+1] for i in range(len(recent_val)-1)):
                issues.append("Val loss increasing for 3+ epochs")
        
        # 3. Accuracy gap check
        if train_acc is not None and val_acc is not None:
            acc_gap = train_acc - val_acc
            if acc_gap > self.acc_threshold:
                issues.append(f"Acc gap {acc_gap:.3f} > {self.acc_threshold}")
        
        # 4. No improvement check
        if self.epochs_since_improvement >= self.patience:
            issues.append(f"No improvement for {self.epochs_since_improvement} epochs")
        
        # Determine severity
        if len(issues) >= 3:
            result['severity'] = 'severe'
            result['is_overfitting'] = True
            self.overfit_counter += 1
        elif len(issues) >= 2:
            result['severity'] = 'moderate'
            result['is_overfitting'] = True
            self.overfit_counter += 1
        elif len(issues) >= 1:
            result['severity'] = 'mild'
            self.overfit_counter = max(0, self.overfit_counter - 1)
        else:
            result['severity'] = 'none'
            self.overfit_counter = max(0, self.overfit_counter - 1)
        
        result['message'] = '; '.join(issues) if issues else "Training normally"
        result['overfit_count'] = self.overfit_counter
        result['epochs_since_improvement'] = self.epochs_since_improvement
        
        return result
    
    def get_summary(self) -> dict:
        """Get summary of training history."""
        if len(self.train_losses) == 0:
            return {'status': 'no_data'}
        
        return {
            'total_epochs': len(self.train_losses),
            'best_val_loss': self.best_val_loss,
            'final_train_loss': self.train_losses[-1],
            'final_val_loss': self.val_losses[-1],
            'final_loss_gap': self.val_losses[-1] - self.train_losses[-1],
            'final_acc_gap': (self.train_accs[-1] - self.val_accs[-1]) if self.train_accs and self.val_accs else None,
            'overfit_epochs': self.overfit_counter,
            'verdict': 'overfitting' if self.overfit_counter > 3 else 'normal'
        }


class CalibrationMonitor:
    """
    Monitor model calibration (ECE) during training.
    """
    
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.ece_history = []
        self.mce_history = []
    
    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> dict:
        """Compute calibration metrics for this epoch."""
        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            
            if isinstance(probs, torch.Tensor):
                probs = probs.cpu().numpy()
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()
            
            ece = self._compute_ece(probs, labels)
            mce = self._compute_mce(probs, labels)
            
            self.ece_history.append(ece)
            self.mce_history.append(mce)
            
            return {
                'ece': ece,
                'mce': mce,
                'calibration_status': self._interpret_ece(ece)
            }
    
    def _compute_ece(self, probs, labels):
        """Expected Calibration Error."""
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels).astype(float)
        
        ece = 0.0
        for i in range(self.n_bins):
            lower = i / self.n_bins
            upper = (i + 1) / self.n_bins
            in_bin = (confidences > lower) & (confidences <= upper)
            
            if np.sum(in_bin) > 0:
                prop_in_bin = np.mean(in_bin)
                avg_conf = np.mean(confidences[in_bin])
                avg_acc = np.mean(accuracies[in_bin])
                ece += np.abs(avg_conf - avg_acc) * prop_in_bin
        
        return ece
    
    def _compute_mce(self, probs, labels):
        """Maximum Calibration Error."""
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels).astype(float)
        
        mce = 0.0
        for i in range(self.n_bins):
            lower = i / self.n_bins
            upper = (i + 1) / self.n_bins
            in_bin = (confidences > lower) & (confidences <= upper)
            
            if np.sum(in_bin) > 0:
                avg_conf = np.mean(confidences[in_bin])
                avg_acc = np.mean(accuracies[in_bin])
                mce = max(mce, np.abs(avg_conf - avg_acc))
        
        return mce
    
    def _interpret_ece(self, ece):
        if ece < 0.05:
            return "excellent"
        elif ece < 0.10:
            return "good"
        elif ece < 0.15:
            return "moderate"
        else:
            return "poor - consider temperature scaling"


class EMAModel:
    """
    Exponential Moving Average of model weights.
    
    Provides more stable predictions by averaging weights over training.
    Particularly helpful for calibration.
    
    Usage:
        ema = EMAModel(model, decay=0.999)
        
        for batch in dataloader:
            loss = train_step(model, batch)
            ema.update()  # Update EMA after each step
        
        # For evaluation, use EMA model
        ema_preds = ema.ema_model(inputs)
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        """
        Args:
            model: The model to track
            decay: EMA decay rate (0.999 = slow update, 0.99 = faster)
        """
        self.model = model
        self.decay = decay
        self.ema_model = deepcopy(model)
        self.ema_model.eval()
        
        # Disable gradients for EMA model
        for param in self.ema_model.parameters():
            param.requires_grad_(False)
    
    @torch.no_grad()
    def update(self):
        """Update EMA weights."""
        for ema_param, model_param in zip(
            self.ema_model.parameters(),
            self.model.parameters()
        ):
            ema_param.data.mul_(self.decay).add_(
                model_param.data, alpha=1 - self.decay
            )
    
    def state_dict(self):
        """Get EMA model state dict."""
        return self.ema_model.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load EMA model state dict."""
        self.ema_model.load_state_dict(state_dict)


class TrainingMonitor:
    """
    Comprehensive training monitor that combines all callbacks.
    """
    
    def __init__(
        self,
        model: nn.Module = None,
        use_ema: bool = True,
        ema_decay: float = 0.999,
        overfit_patience: int = 5,
        overfit_loss_threshold: float = 0.3,
        log_calibration: bool = True
    ):
        self.overfit_detector = OverfittingDetector(
            patience=overfit_patience,
            loss_threshold=overfit_loss_threshold
        )
        
        self.calibration_monitor = CalibrationMonitor() if log_calibration else None
        
        self.ema = EMAModel(model, decay=ema_decay) if use_ema and model else None
        
        self.epoch_logs = []
    
    def on_epoch_end(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_acc: float = None,
        val_acc: float = None,
        val_logits: torch.Tensor = None,
        val_labels: torch.Tensor = None
    ) -> dict:
        """
        Call at end of each epoch.
        
        Returns comprehensive status dict.
        """
        # Update EMA
        if self.ema:
            self.ema.update()
        
        # Check overfitting
        overfit_status = self.overfit_detector.update(
            train_loss, val_loss, train_acc, val_acc, epoch
        )
        
        # Check calibration
        calibration_status = None
        if self.calibration_monitor and val_logits is not None:
            calibration_status = self.calibration_monitor.update(val_logits, val_labels)
        
        # Combine
        log = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'overfitting': overfit_status,
            'calibration': calibration_status
        }
        
        self.epoch_logs.append(log)
        
        return log
    
    def get_summary(self) -> dict:
        """Get full training summary."""
        return {
            'total_epochs': len(self.epoch_logs),
            'overfitting_summary': self.overfit_detector.get_summary(),
            'final_ece': self.calibration_monitor.ece_history[-1] if self.calibration_monitor and self.calibration_monitor.ece_history else None,
            'ece_trend': self.calibration_monitor.ece_history if self.calibration_monitor else None
        }
    
    def print_epoch_summary(self, log: dict):
        """Print formatted epoch summary."""
        print(f"\n{'='*60}")
        print(f"Epoch {log['epoch']} Summary")
        print(f"{'='*60}")
        print(f"  Train Loss: {log['train_loss']:.4f}  |  Val Loss: {log['val_loss']:.4f}")
        
        if log['train_acc'] is not None:
            print(f"  Train Acc:  {log['train_acc']:.4f}  |  Val Acc:  {log['val_acc']:.4f}")
        
        if log['calibration']:
            print(f"  ECE: {log['calibration']['ece']:.4f} ({log['calibration']['calibration_status']})")
        
        if log['overfitting']['is_overfitting']:
            print(f"  ⚠️ OVERFITTING: {log['overfitting']['message']}")
        else:
            print(f"  ✅ Status: {log['overfitting']['message']}")


# Example integration with training loop
def example_training_integration():
    """
    Example of how to integrate these callbacks into your training loop.
    """
    example_code = '''
    from training_callbacks import TrainingMonitor, TemperatureScaler
    
    # Initialize
    model = YourModel()
    monitor = TrainingMonitor(
        model=model,
        use_ema=True,
        ema_decay=0.999,
        overfit_patience=5,
        log_calibration=True
    )
    
    # Training loop
    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader)
        
        # Validate (save logits for calibration monitoring)
        val_loss, val_acc, val_logits, val_labels = validate(model, val_loader, return_logits=True)
        
        # Monitor
        status = monitor.on_epoch_end(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_acc=train_acc,
            val_acc=val_acc,
            val_logits=val_logits,
            val_labels=val_labels
        )
        
        monitor.print_epoch_summary(status)
        
        # Early stopping on overfitting
        if status['overfitting']['overfit_count'] > 5:
            print("Stopping due to persistent overfitting")
            break
    
    # Post-training: Apply temperature scaling
    print("\\nApplying temperature scaling...")
    scaler = TemperatureScaler()
    scaler.fit(val_logits, val_labels)
    
    # Save calibrated model
    torch.save({
        'model_state_dict': model.state_dict(),
        'ema_state_dict': monitor.ema.state_dict() if monitor.ema else None,
        'temperature': scaler.temperature.item()
    }, 'calibrated_model.pt')
    '''
    print(example_code)


if __name__ == '__main__':
    print("Training Callbacks Module")
    print("="*60)
    print("\nExample integration:")
    example_training_integration()