"""
Sarika AI - Main Configuration
5→1→6→1→6→1 Progressive Hierarchical Cascade Distillation
Optimized for RTX 5060 Ti 16GB + 200GB SSD
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================
# PROJECT PATHS
# ============================================
PROJECT_ROOT = Path(__file__).parent.parent
ML_DIR = PROJECT_ROOT / "ml"
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = ML_DIR / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"

# Create directories if not exist
for dir_path in [DATA_DIR, MODEL_DIR, CHECKPOINT_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================
# HARDWARE CONFIGURATION
# ============================================
DEVICE = os.getenv("DEVICE", "cuda")
GPU_MEMORY_GB = int(os.getenv("GPU_MEMORY_GB", 16))
SSD_SPACE_GB = int(os.getenv("SSD_SPACE_GB", 200))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", 4))
MIXED_PRECISION = os.getenv("MIXED_PRECISION", "fp16")

# ============================================
# HUGGING FACE
# ============================================
HF_TOKEN = os.getenv("HF_TOKEN")
HF_HOME = os.getenv("HF_HOME", str(MODEL_DIR / "cache"))

# ============================================
# STAGE 1: GIANT TEACHERS (Cloud - Google Colab)
# ============================================
@dataclass
class GiantTeacher:
    model_id: str
    size: str
    vram_full: str
    vram_4bit: str
    strength: str
    company: str
    
STAGE_1_GIANTS = {
    "llama_70b": GiantTeacher(
        model_id="meta-llama/Llama-3.1-70B-Instruct",
        size="70B",
        vram_full="140GB",
        vram_4bit="35GB",
        strength="Best instruction following, conversation, general intelligence",
        company="Meta"
    ),
    
    "qwen_72b": GiantTeacher(
        model_id="Qwen/Qwen2.5-72B-Instruct",
        size="72B",
        vram_full="144GB",
        vram_4bit="36GB",
        strength="Multilingual champion, Asian languages, Bengali context",
        company="Alibaba Cloud"
    ),
    
    "mistral_large": GiantTeacher(
        model_id="mistralai/Mistral-Large-Instruct-2411",
        size="123B",
        vram_full="246GB",
        vram_4bit="62GB",
        strength="Advanced reasoning, coding, problem solving",
        company="Mistral AI"
    ),
    
    "gemma_27b": GiantTeacher(
        model_id="google/gemma-2-27b-it",
        size="27B",
        vram_full="54GB",
        vram_4bit="14GB",
        strength="Safety, responsible AI, ethics, careful responses",
        company="Google"
    ),
    
    "phi_moe": GiantTeacher(
        model_id="microsoft/Phi-3.5-MoE-instruct",
        size="42B",
        vram_full="84GB",
        vram_4bit="21GB",
        strength="Efficient MoE, diverse knowledge, creative tasks",
        company="Microsoft"
    )
}

# ============================================
# STAGE 2: INTERMEDIATE TEACHER (Local GPU)
# ============================================
INTERMEDIATE_TEACHER_1 = {
    "model_id": "meta-llama/Llama-3.1-8B-Instruct",
    "size": "8B",
    "vram_full": "16GB",
    "vram_4bit": "5GB",
    "role": "Receives knowledge from 5 giant teachers, combines their strengths"
}

# ============================================
# STAGE 3: CONTEXT TEACHERS (Local GPU)
# ============================================
@dataclass
class ContextTeacher:
    model_id: str
    size: str
    vram_4bit: str
    specialty: str
    personality: str

STAGE_2_CONTEXT_TEACHERS = {
    "bengali_culture": ContextTeacher(
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        size="7B",
        vram_4bit="4.5GB",
        specialty="Bengali culture, idioms, traditions, festivals",
        personality="Sanskriti - Cultural Expert"
    ),
    
    "emotional_intelligence": ContextTeacher(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        size="7B",
        vram_4bit="4.5GB",
        specialty="Reading emotions, empathy, mood detection",
        personality="Bhab - Emotional Intelligence"
    ),
    
    "conversation_flow": ContextTeacher(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        size="8B",
        vram_4bit="5GB",
        specialty="Maintaining engaging conversations, follow-ups",
        personality="Kotha - Conversation Expert"
    ),
    
    "humor": ContextTeacher(
        model_id="google/gemma-2-9b-it",
        size="9B",
        vram_4bit="5.5GB",
        specialty="Bengali humor, wordplay, witty responses",
        personality="Hashi - Humor Specialist"
    ),
    
    "deep_conversations": ContextTeacher(
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        size="7B",
        vram_4bit="4.5GB",
        specialty="Philosophy, existential talks, deep meaning",
        personality="Gobhir - Deep Thinker"
    ),
    
    "crisis_support": ContextTeacher(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        size="7B",
        vram_4bit="4.5GB",
        specialty="Mental health support, crisis detection",
        personality="Shohay - Crisis Helper"
    )
}

# ============================================
# STAGE 4: INTEGRATION TEACHER (Local GPU)
# ============================================
INTEGRATION_TEACHER = {
    "model_id": "meta-llama/Llama-3.1-8B-Instruct",
    "size": "8B",
    "vram_4bit": "5GB",
    "role": "Fuses knowledge from 6 context teachers"
}

# ============================================
# STAGE 5: DOMAIN SPECIALISTS (Local GPU)
# ============================================
@dataclass
class DomainSpecialist:
    model_id: str
    size: str
    vram_4bit: str
    domain: str
    personality: str

STAGE_4_SPECIALISTS = {
    "romance": DomainSpecialist(
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        size="3B",
        vram_4bit="2GB",
        domain="Romantic conversations, love advice, poetry",
        personality="Priya - The Romantic"
    ),
    
    "entertainment": DomainSpecialist(
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        size="3B",
        vram_4bit="2GB",
        domain="Movies, music, books, recommendations",
        personality="Manoranjan - Entertainment Guide"
    ),
    
    "life_coaching": DomainSpecialist(
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        size="3B",
        vram_4bit="2GB",
        domain="Motivation, goal setting, personal growth",
        personality="Uddipok - Life Coach"
    ),
    
    "tech_help": DomainSpecialist(
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        size="3B",
        vram_4bit="2GB",
        domain="Tech questions, coding help, troubleshooting",
        personality="Projukti - Tech Helper"
    ),
    
    "lifestyle": DomainSpecialist(
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        size="3B",
        vram_4bit="2GB",
        domain="Health, fitness, food, daily routines",
        personality="Jibon - Lifestyle Guide"
    ),
    
    "literature": DomainSpecialist(
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        size="3B",
        vram_4bit="2GB",
        domain="Bengali literature, poetry, books",
        personality="Sahitya - Literature Expert"
    )
}

# ============================================
# FINAL: STUDENT MODEL (Sarika AI)
# ============================================
STUDENT_MODEL = {
    "model_id": "meta-llama/Llama-3.2-1B-Instruct",
    "size": "1B",
    "vram_full": "2GB",
    "vram_4bit": "800MB",
    "inference_speed": "50+ tokens/sec on RTX 5060 Ti",
    "final_output": "Sarika AI - Bengali AI Companion"
}

# ============================================
# TRAINING HYPERPARAMETERS
# ============================================
class TrainingConfig:
    # Batch & Gradient
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 2))
    GRADIENT_ACCUMULATION = int(os.getenv("GRADIENT_ACCUMULATION", 8))
    EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION  # = 16
    
    # Learning Rate
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", 2e-4))
    WARMUP_STEPS = int(os.getenv("WARMUP_STEPS", 100))
    WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", 0.01))
    MAX_GRAD_NORM = float(os.getenv("MAX_GRAD_NORM", 1.0))
    
    # Epochs & Length
    NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 3))
    MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", 512))
    
    # LoRA Configuration
    LORA_R = int(os.getenv("LORA_R", 16))
    LORA_ALPHA = int(os.getenv("LORA_ALPHA", 32))
    LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", 0.05))
    LORA_TARGET_MODULES = os.getenv(
        "LORA_TARGET_MODULES", 
        "q_proj,v_proj,k_proj,o_proj"
    ).split(",")
    
    # Distillation
    DISTILLATION_ALPHA = float(os.getenv("DISTILLATION_ALPHA", 0.5))
    TEMPERATURE = float(os.getenv("TEMPERATURE", 2.0))
    
    # Optimization
    OPTIMIZER = "paged_adamw_8bit"  # Memory efficient
    SCHEDULER = "cosine"
    
    # Mixed Precision
    FP16 = MIXED_PRECISION == "fp16"
    BF16 = MIXED_PRECISION == "bf16"

# ============================================
# SPACE MANAGEMENT
# ============================================
class SpaceConfig:
    AUTO_CLEANUP = os.getenv("AUTO_CLEANUP", "true").lower() == "true"
    MAX_CHECKPOINTS = int(os.getenv("MAX_CHECKPOINTS", 2))
    CLEANUP_THRESHOLD = int(os.getenv("CLEANUP_THRESHOLD", 85))  # %
    
    # Space budget (GB)
    MAX_TOTAL_USAGE = 180  # Out of 200GB
    RESERVED_BUFFER = 20   # Keep 20GB free

# ============================================
# LOGGING & MONITORING
# ============================================
class LogConfig:
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # Weights & Biases
    WANDB_ENABLED = os.getenv("WANDB_ENABLED", "false").lower() == "true"
    WANDB_PROJECT = os.getenv("WANDB_PROJECT", "sarika-ai")
    WANDB_ENTITY = os.getenv("WANDB_ENTITY", None)
    
    # TensorBoard
    TENSORBOARD_ENABLED = os.getenv("TENSORBOARD_ENABLED", "true").lower() == "true"
    TENSORBOARD_DIR = LOG_DIR / "tensorboard"
    
    # Logging intervals
    LOG_STEPS = 10
    SAVE_STEPS = 500
    EVAL_STEPS = 500

# ============================================
# DATASET CONFIGURATION
# ============================================
class DataConfig:
    DATASET_NAME = os.getenv("DATASET_NAME", "custom_bengali")
    TRAIN_SPLIT = os.getenv("TRAIN_SPLIT", "train")
    VAL_SPLIT = os.getenv("VAL_SPLIT", "validation")
    TEST_SPLIT = os.getenv("TEST_SPLIT", "test")
    
    # Data paths
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
    
    # Data loading
    NUM_WORKERS = int(os.getenv("NUM_WORKERS", 4))
    PREFETCH_FACTOR = 2

# ============================================
# MODEL SAVE PATHS
# ============================================
class ModelPaths:
    # Teachers
    GIANTS_DIR = MODEL_DIR / "giants"
    INTERMEDIATE_DIR = MODEL_DIR / "intermediate"
    CONTEXT_DIR = MODEL_DIR / "context"
    INTEGRATION_DIR = MODEL_DIR / "integration"
    SPECIALISTS_DIR = MODEL_DIR / "specialists"
    
    # Final
    STUDENT_DIR = MODEL_DIR / "student"
    FINAL_DIR = MODEL_DIR / "final"
    
    # Checkpoints
    CHECKPOINT_DIR = CHECKPOINT_DIR

# ============================================
# INFERENCE CONFIGURATION
# ============================================
class InferenceConfig:
    MAX_NEW_TOKENS = 512
    TEMPERATURE = 0.7
    TOP_P = 0.9
    TOP_K = 50
    REPETITION_PENALTY = 1.1
    DO_SAMPLE = True

# ============================================
# UTILITY FUNCTIONS
# ============================================
def print_config():
    """Print configuration summary"""
    print("=" * 60)
    print("         SARIKA AI - Configuration Summary")
    print("=" * 60)
    print(f"\n🖥️  Hardware:")
    print(f"   Device: {DEVICE}")
    print(f"   GPU Memory: {GPU_MEMORY_GB}GB")
    print(f"   SSD Space: {SSD_SPACE_GB}GB")
    print(f"   Mixed Precision: {MIXED_PRECISION}")
    
    print(f"\n📊 Training:")
    print(f"   Batch Size: {TrainingConfig.BATCH_SIZE}")
    print(f"   Gradient Accumulation: {TrainingConfig.GRADIENT_ACCUMULATION}")
    print(f"   Effective Batch: {TrainingConfig.EFFECTIVE_BATCH_SIZE}")
    print(f"   Learning Rate: {TrainingConfig.LEARNING_RATE}")
    print(f"   Epochs: {TrainingConfig.NUM_EPOCHS}")
    
    print(f"\n🎯 Architecture:")
    print(f"   Stage 1: 5 Giant Teachers (70B-123B)")
    print(f"   Stage 2: 1 Intermediate Teacher (8B)")
    print(f"   Stage 3: 6 Context Teachers (7B-9B)")
    print(f"   Stage 4: 1 Integration Teacher (8B)")
    print(f"   Stage 5: 6 Domain Specialists (3B)")
    print(f"   Final: Sarika AI Student (1B)")
    
    print(f"\n💾 Storage:")
    print(f"   Auto Cleanup: {SpaceConfig.AUTO_CLEANUP}")
    print(f"   Max Checkpoints: {SpaceConfig.MAX_CHECKPOINTS}")
    print(f"   Cleanup Threshold: {SpaceConfig.CLEANUP_THRESHOLD}%")
    
    print(f"\n📁 Paths:")
    print(f"   Project Root: {PROJECT_ROOT}")
    print(f"   Data Dir: {DATA_DIR}")
    print(f"   Model Dir: {MODEL_DIR}")
    print(f"   Checkpoint Dir: {CHECKPOINT_DIR}")
    
    print("=" * 60)

def validate_config():
    """Validate configuration"""
    errors = []
    
    # Check HF token
    if not HF_TOKEN or "xxxxx" in HF_TOKEN:
        errors.append("❌ HuggingFace token not set in .env")
    
    # Check disk space
    import shutil
    free_space_gb = shutil.disk_usage(PROJECT_ROOT).free / (1024**3)
    if free_space_gb < 100:
        errors.append(f"⚠️ Low disk space: {free_space_gb:.1f}GB (100GB+ recommended)")
    
    # Check CUDA
    if DEVICE == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                errors.append("⚠️ CUDA not available, will use CPU (slow)")
        except ImportError:
            errors.append("❌ PyTorch not installed")
    
    if errors:
        print("\n⚠️  Configuration Issues:")
        for error in errors:
            print(f"  {error}")
        print()
    else:
        print("\n✅ Configuration validated successfully!\n")
    
    return len(errors) == 0

# ============================================
# EXPORT ALL
# ============================================
__all__ = [
    # Paths
    "PROJECT_ROOT", "ML_DIR", "DATA_DIR", "MODEL_DIR", 
    "CHECKPOINT_DIR", "LOG_DIR",
    
    # Hardware
    "DEVICE", "GPU_MEMORY_GB", "SSD_SPACE_GB",
    
    # Models
    "STAGE_1_GIANTS", "INTERMEDIATE_TEACHER_1",
    "STAGE_2_CONTEXT_TEACHERS", "INTEGRATION_TEACHER",
    "STAGE_4_SPECIALISTS", "STUDENT_MODEL",
    
    # Configs
    "TrainingConfig", "SpaceConfig", "LogConfig",
    "DataConfig", "ModelPaths", "InferenceConfig",
    
    # Functions
    "print_config", "validate_config"
]

if __name__ == "__main__":
    print_config()
    validate_config()
