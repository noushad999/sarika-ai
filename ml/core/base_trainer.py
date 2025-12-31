"""
Sarika AI - Base Trainer
Handles model loading, training loops, distillation
Windows-compatible version (no quantization dependency)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
from typing import Optional, Dict, List
import os
from pathlib import Path
from datetime import datetime
import json

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from ml.config import (
    DEVICE, GPU_MEMORY_GB, HF_TOKEN, CHECKPOINT_DIR, 
    TrainingConfig, SpaceConfig, LogConfig, PROJECT_ROOT
)


class BaseTrainer:
    """Base class for all training stages"""
    
    def __init__(
        self,
        stage_name: str,
        output_dir: Optional[Path] = None
    ):
        self.stage_name = stage_name
        self.output_dir = output_dir or CHECKPOINT_DIR / stage_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = DEVICE
        self.tokenizer = None
        
        # Training metrics
        self.training_start_time = None
        self.training_history = []
        
        print(f"\n{'='*60}")
        print(f"Initializing: {stage_name}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*60}\n")
    
    def load_model_4bit(
        self, 
        model_id: str,
        trust_remote_code: bool = False
    ) -> AutoModelForCausalLM:
        """
        Load model - Windows compatible
        Falls back to FP16 if quantization fails
        """
        print(f"📥 Loading model: {model_id}")
        
        # First try: Load with FP16 (no quantization)
        try:
            print(f"   Mode: FP16 (full precision)")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=trust_remote_code,
                token=os.getenv("HF_TOKEN"),
                low_cpu_mem_usage=True
            )
            
            print("✓ Model loaded in FP16 mode")
            
        except Exception as e:
            print(f"⚠️ FP16 failed: {str(e)[:100]}...")
            print("   Falling back to CPU mode...")
            
            # Fallback: Load on CPU
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                trust_remote_code=trust_remote_code,
                token=os.getenv("HF_TOKEN"),
                low_cpu_mem_usage=True
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                print("   Moving model to GPU...")
                model = model.to('cuda')
            
            print("✓ Model loaded (CPU→GPU mode)")
        
        # Load tokenizer
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
                token=os.getenv("HF_TOKEN")
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
        
        # Print memory usage
        if torch.cuda.is_available():
            vram_used = torch.cuda.memory_allocated() / 1024**3
            print(f"✓ VRAM used: {vram_used:.2f}GB / {GPU_MEMORY_GB}GB")
        else:
            print(f"✓ Running on CPU")
        
        return model
    
    def apply_lora(
        self,
        model: AutoModelForCausalLM,
        r: int = None,
        lora_alpha: int = None,
        lora_dropout: float = None
    ) -> AutoModelForCausalLM:
        """
        Apply LoRA for efficient fine-tuning
        Works with device_map="auto" models
        """
        print("🎯 Applying LoRA...")
        
        # Use config defaults if not specified
        r = r or TrainingConfig.LORA_R
        lora_alpha = lora_alpha or TrainingConfig.LORA_ALPHA
        lora_dropout = lora_dropout or TrainingConfig.LORA_DROPOUT
        
        # Enable gradient checkpointing to save memory
        try:
            model.gradient_checkpointing_enable()
            print("   ✓ Gradient checkpointing enabled")
        except Exception as e:
            print(f"   ⚠️ Gradient checkpointing failed: {str(e)[:50]}")
        
        # LoRA config
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=TrainingConfig.LORA_TARGET_MODULES,
            bias="none",
            # Important: Don't move base model
            modules_to_save=None
        )
        
        # Apply LoRA directly without moving model
        print("   Applying LoRA adapters...")
        try:
            model = get_peft_model(model, lora_config)
            print("   ✓ LoRA applied successfully")
        except Exception as e:
            print(f"   ⚠️ LoRA application failed: {str(e)[:100]}")
            print("   Continuing without LoRA (full fine-tuning)")
            return model
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_pct = 100 * trainable_params / total_params
        
        print(f"✓ LoRA applied:")
        print(f"   Trainable params: {trainable_params:,} ({trainable_pct:.2f}%)")
        print(f"   Total params: {total_params:,}")
        
        return model
    
    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        alpha: float = None,
        temperature: float = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute knowledge distillation loss
        
        Loss = α * KL_divergence(student, teacher) + (1-α) * CrossEntropy(student, labels)
        """
        alpha = alpha or TrainingConfig.DISTILLATION_ALPHA
        temperature = temperature or TrainingConfig.TEMPERATURE
        
        # Soft targets from teacher
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        
        # Student predictions
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        
        # KL divergence loss (knowledge distillation)
        kl_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (temperature ** 2)
        
        # Cross-entropy loss (task loss)
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1),
            ignore_index=self.tokenizer.pad_token_id
        )
        
        # Combined loss
        total_loss = alpha * kl_loss + (1 - alpha) * ce_loss
        
        return {
            "loss": total_loss,
            "kl_loss": kl_loss,
            "ce_loss": ce_loss
        }
    
    def save_checkpoint(
        self,
        model: AutoModelForCausalLM,
        checkpoint_name: str,
        metadata: Optional[Dict] = None
    ):
        """Save model checkpoint"""
        checkpoint_path = self.output_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 Saving checkpoint: {checkpoint_name}")
        
        # Save model
        model.save_pretrained(checkpoint_path)
        
        # Save tokenizer
        if self.tokenizer:
            self.tokenizer.save_pretrained(checkpoint_path)
        
        # Save metadata
        if metadata:
            metadata_path = checkpoint_path / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Check saved size
        total_size = sum(
            f.stat().st_size for f in checkpoint_path.rglob('*') if f.is_file()
        ) / 1024**3
        
        print(f"✓ Checkpoint saved: {total_size:.2f}GB")
        print(f"   Path: {checkpoint_path}")
        
        return checkpoint_path
    
    def cleanup_memory(self):
        """Free GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def log_metrics(self, metrics: Dict):
        """Log training metrics"""
        self.training_history.append({
            "timestamp": datetime.now().isoformat(),
            **metrics
        })
        
        # Print
        print(f"\n📊 Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
    
    def get_memory_stats(self) -> Dict:
        """Get GPU memory statistics"""
        if not torch.cuda.is_available():
            return {}
        
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "max_allocated_gb": max_allocated,
            "free_gb": GPU_MEMORY_GB - allocated
        }
    
    def print_memory_stats(self):
        """Print GPU memory usage"""
        stats = self.get_memory_stats()
        if stats:
            print(f"\n💻 GPU Memory:")
            print(f"   Allocated: {stats['allocated_gb']:.2f}GB")
            print(f"   Free: {stats['free_gb']:.2f}GB / {GPU_MEMORY_GB}GB")


class SequentialDistillationTrainer(BaseTrainer):
    """
    Sequential distillation trainer
    Loads one teacher at a time to save memory
    """
    
    def __init__(self, stage_name: str):
        super().__init__(stage_name)
        self.student = None
    
    def load_student(
        self,
        model_id: str,
        apply_lora: bool = True
    ):
        """Load student model"""
        print(f"\n🎓 Loading student model...")
        
        self.student = self.load_model_4bit(model_id)
        
        if apply_lora:
            self.student = self.apply_lora(self.student)
        
        self.student.train()
        
        return self.student
    
    def train_with_teachers(
        self,
        teacher_ids: List[str],
        dataset,
        epochs_per_teacher: int = 1
    ):
        """
        Train student with multiple teachers sequentially
        One teacher at a time to save memory
        """
        total_teachers = len(teacher_ids)
        
        print(f"\n{'='*60}")
        print(f"Sequential Training: {total_teachers} teachers")
        print(f"{'='*60}\n")
        
        for idx, teacher_id in enumerate(teacher_ids, 1):
            print(f"\n{'─'*60}")
            print(f"Teacher {idx}/{total_teachers}: {teacher_id}")
            print(f"{'─'*60}")
            
            # Load teacher
            teacher = self.load_model_4bit(teacher_id)
            teacher.eval()
            
            # Train with this teacher
            self._train_with_single_teacher(
                teacher=teacher,
                teacher_name=f"teacher_{idx}",
                dataset=dataset,
                epochs=epochs_per_teacher
            )
            
            # Save checkpoint after each teacher
            self.save_checkpoint(
                self.student,
                checkpoint_name=f"after_teacher_{idx}",
                metadata={
                    "teacher_id": teacher_id,
                    "teacher_num": idx,
                    "total_teachers": total_teachers,
                    "stage": self.stage_name
                }
            )
            
            # Delete teacher and free memory
            del teacher
            self.cleanup_memory()
            
            self.print_memory_stats()
        
        print(f"\n{'='*60}")
        print(f"✅ Sequential training complete!")
        print(f"{'='*60}\n")
    
    def _train_with_single_teacher(
        self,
        teacher,
        teacher_name: str,
        dataset,
        epochs: int
    ):
        """Train student with single teacher"""
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=TrainingConfig.LEARNING_RATE,
            weight_decay=TrainingConfig.WEIGHT_DECAY
        )
        
        # Training loop
        for epoch in range(epochs):
            print(f"\n📚 Epoch {epoch+1}/{epochs}")
            
            epoch_loss = 0.0
            epoch_kl = 0.0
            epoch_ce = 0.0
            num_batches = 0
            
            # Batch loop
            for batch_idx, batch in enumerate(dataset):
                # Tokenize inputs
                inputs = self.tokenizer(
                    batch["text"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=TrainingConfig.MAX_SEQ_LENGTH
                ).to(self.device)
                
                # Student forward pass
                student_outputs = self.student(**inputs)
                student_logits = student_outputs.logits
                
                # Teacher forward pass (no gradient)
                with torch.no_grad():
                    teacher_outputs = teacher(**inputs)
                    teacher_logits = teacher_outputs.logits
                
                # Compute distillation loss
                losses = self.compute_distillation_loss(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    labels=inputs["input_ids"]
                )
                
                # Backward pass
                optimizer.zero_grad()
                losses["loss"].backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.student.parameters(),
                    TrainingConfig.MAX_GRAD_NORM
                )
                
                optimizer.step()
                
                # Accumulate metrics
                epoch_loss += losses["loss"].item()
                epoch_kl += losses["kl_loss"].item()
                epoch_ce += losses["ce_loss"].item()
                num_batches += 1
                
                # Log every N batches
                if (batch_idx + 1) % LogConfig.LOG_STEPS == 0:
                    avg_loss = epoch_loss / num_batches
                    print(f"   Batch {batch_idx+1}: Loss = {avg_loss:.4f}")
            
            # Epoch metrics
            self.log_metrics({
                "teacher": teacher_name,
                "epoch": epoch + 1,
                "loss": epoch_loss / num_batches,
                "kl_loss": epoch_kl / num_batches,
                "ce_loss": epoch_ce / num_batches
            })


class SpaceEfficientTrainer:
    """
    Space-efficient trainer that manages disk space
    """
    
    def __init__(self):
        self.max_space_gb = SpaceConfig.MAX_TOTAL_USAGE
        self.cleanup_threshold = SpaceConfig.CLEANUP_THRESHOLD
    
    def check_space(self) -> Dict:
        """Check available disk space"""
        import shutil
        
        total, used, free = shutil.disk_usage(PROJECT_ROOT)
        
        used_gb = used / 1024**3
        free_gb = free / 1024**3
        total_gb = total / 1024**3
        used_pct = (used / total) * 100
        
        return {
            "total_gb": total_gb,
            "used_gb": used_gb,
            "free_gb": free_gb,
            "used_pct": used_pct
        }
    
    def auto_cleanup(self):
        """Automatically cleanup if space is low"""
        space = self.check_space()
        
        if space["used_pct"] > self.cleanup_threshold:
            print(f"\n⚠️ Disk usage: {space['used_pct']:.1f}% > {self.cleanup_threshold}%")
            print("   Running auto-cleanup...")
            
            # Delete old checkpoints
            self._cleanup_old_checkpoints()
            
            # Clear cache
            self._clear_cache()
            
            # Check again
            space_after = self.check_space()
            freed = space["used_gb"] - space_after["used_gb"]
            print(f"✓ Freed {freed:.2f}GB")
    
    def _cleanup_old_checkpoints(self):
        """Keep only recent checkpoints"""
        import glob
        
        checkpoints = sorted(
            CHECKPOINT_DIR.glob("**/checkpoint-*"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        # Keep only MAX_CHECKPOINTS
        for ckpt in checkpoints[SpaceConfig.MAX_CHECKPOINTS:]:
            import shutil
            shutil.rmtree(ckpt)
            print(f"   Deleted: {ckpt.name}")
    
    def _clear_cache(self):
        """Clear HuggingFace cache"""
        from ml.config import HF_HOME
        cache_dir = Path(HF_HOME)
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            print("   Cleared HF cache")


# Export
__all__ = [
    "BaseTrainer",
    "SequentialDistillationTrainer", 
    "SpaceEfficientTrainer"
]
