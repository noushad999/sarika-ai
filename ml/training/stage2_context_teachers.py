"""
Sarika AI - Stage 2 Training
Context Teachers (6 teachers: Bengali culture, emotion, conversation, humor, deep, crisis)
Runs on local GPU (RTX 5060 Ti)
"""

import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from ml.config import *
from ml.core.base_trainer import SequentialDistillationTrainer, SpaceEfficientTrainer
from typing import Dict 
from ml.data.dataset_handler import create_training_dataset
from ml.prompts.personalities import *


class Stage2ContextTrainer:
    """Stage 2: Train with 6 context teachers"""
    
    def __init__(self):
        self.stage_name = "stage2_context_teachers"
        self.output_dir = ModelPaths.CONTEXT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Space manager
        self.space_manager = SpaceEfficientTrainer()
        
        print("\n" + "="*60)
        print("STAGE 2: CONTEXT TEACHERS TRAINING")
        print("="*60)
        print(f"Teachers: 6 (Bengali, Emotion, Conversation, Humor, Deep, Crisis)")
        print(f"Student: Intermediate Teacher from Stage 1")
        print(f"Output: {self.output_dir}")
        print("="*60 + "\n")
    
    def run(self):
        """Execute Stage 2 training"""
        
        # Check space
        self.space_manager.auto_cleanup()
        
        # Step 1: Load dataset
        print("\n📊 Step 1: Loading dataset...")
        dataset = create_training_dataset(size='medium')  # 500 samples
        print(f"✓ Train: {len(dataset['train'])} samples")
        print(f"✓ Val: {len(dataset['validation'])} samples")
        
        # Step 2: Initialize trainer
        print("\n🎯 Step 2: Initializing trainer...")
        trainer = SequentialDistillationTrainer(self.stage_name)
        
        # Step 3: Load student (intermediate teacher from Stage 1)
        # For demo, we'll use base model. In production, load Stage 1 output
        print("\n🎓 Step 3: Loading student model...")
        student_model_id = INTERMEDIATE_TEACHER_1["model_id"]
        trainer.load_student(student_model_id, apply_lora=True)
        
        # Step 4: Get context teacher models
        print("\n👥 Step 4: Preparing context teachers...")
        teacher_models = self._get_context_teachers()
        
        for idx, (name, model_id) in enumerate(teacher_models.items(), 1):
            print(f"   {idx}. {name}: {model_id}")
        
        # Step 5: Sequential training
        print("\n🚀 Step 5: Starting sequential training...")
        print("   (Each teacher trained one at a time)")
        
        teacher_ids = list(teacher_models.values())
        
        trainer.train_with_teachers(
            teacher_ids=teacher_ids,
            dataset=dataset['train'],
            epochs_per_teacher=2  # 2 epochs per teacher
        )
        
        # Step 6: Save final model
        print("\n💾 Step 6: Saving final Stage 2 model...")
        final_checkpoint = trainer.save_checkpoint(
            model=trainer.student,
            checkpoint_name="stage2_final",
            metadata={
                "stage": "stage2_context_teachers",
                "num_teachers": len(teacher_ids),
                "teachers": list(teacher_models.keys()),
                "dataset_size": len(dataset['train']),
                "epochs_per_teacher": 2
            }
        )
        
        print(f"\n{'='*60}")
        print("✅ STAGE 2 COMPLETE!")
        print(f"{'='*60}")
        print(f"Output: {final_checkpoint}")
        print(f"Next: Run Stage 3 (Integration Teacher)")
        print("="*60 + "\n")
    
    def _get_context_teachers(self) -> Dict[str, str]:
        """Get context teacher model IDs"""
        return {
            "bengali_culture": STAGE_2_CONTEXT_TEACHERS["bengali_culture"].model_id,
            "emotional_intelligence": STAGE_2_CONTEXT_TEACHERS["emotional_intelligence"].model_id,
            "conversation_flow": STAGE_2_CONTEXT_TEACHERS["conversation_flow"].model_id,
            "humor": STAGE_2_CONTEXT_TEACHERS["humor"].model_id,
            "deep_conversations": STAGE_2_CONTEXT_TEACHERS["deep_conversations"].model_id,
            "crisis_support": STAGE_2_CONTEXT_TEACHERS["crisis_support"].model_id
        }


def main():
    """Main entry point"""
    trainer = Stage2ContextTrainer()
    trainer.run()


if __name__ == "__main__":
    main()
