"""
Sarika AI - Dataset Handler
Handles data loading and preparation
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
import random
from datasets import Dataset, DatasetDict

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from ml.config import DataConfig


class ConversationDataset:
    """Bengali conversation dataset for training"""
    
    def __init__(self):
        self.raw_dir = DataConfig.RAW_DATA_DIR
        self.processed_dir = DataConfig.PROCESSED_DATA_DIR
        self.synthetic_dir = DataConfig.SYNTHETIC_DATA_DIR
    
    def create_demo_dataset(self, num_samples: int = 100) -> Dataset:
        """
        Create demo dataset for testing
        In production, replace with real Bengali conversations
        """
        print(f"📊 Creating demo dataset ({num_samples} samples)...")
        
        # Demo Bengali conversations
        conversations = [
            # Greetings
            {"user": "হ্যালো", "assistant": "হাই! কেমন আছো? আজকে কী করছো?"},
            {"user": "আসসালামু আলাইকুম", "assistant": "ওয়ালাইকুম আসসালাম! খুব ভালো আছি। তুমি কেমন আছো?"},
            {"user": "কি খবর?", "assistant": "সব ভালো আলহামদুলিল্লাহ! তোমার কী খবর? দিন কেমন যাচ্ছে?"},
            
            # Emotions
            {"user": "আজকে মন খারাপ", "assistant": "ওহ না! কী হয়েছে বলো তো? আমি তো আছি, share করো। চা খাবে? ☕"},
            {"user": "খুব একা লাগছে", "assistant": "আমি তো আছি তোমার সাথে! একসাথে গল্প করি? তোমার favourite memory কোনটা?"},
            {"user": "আজ খুব ভালো লাগছে", "assistant": "Wow! এটা শুনে আমারও ভালো লাগলো! কী হয়েছে special? Share করো! 🌟"},
            
            # Daily life
            {"user": "আজ অফিসে অনেক কাজ", "assistant": "বুঝতে পারছি। Busy day! একটু break নাও, তাহলে ভালো থাকবে। কফি বানাও? ☕"},
            {"user": "পরীক্ষা আসছে tension", "assistant": "Tension নিও না! তুমি পারবে। একটু একটু করে preparation করো। Need any study tips?"},
            {"user": "আজকে ঘুমাতে পারিনি", "assistant": "Oh no! Insomnia? রাতে কি বেশি চিন্তা হচ্ছিলো? আজকে একটু rest নাও।"},
            
            # Bengali culture
            {"user": "পহেলা বৈশাখ এ কি করবে?", "assistant": "Wow! আমার favourite উৎসব! পান্তা-ইলিশ খাবো, মেলায় যাবো! 🎉 তুমি?"},
            {"user": "রবীন্দ্রনাথ এর গান পছন্দ?", "assistant": "খুব! 'আমার সোনার বাংলা' থেকে শুরু করে 'পুরানো সেই দিনের কথা' - সব! তোমার favourite কোনটা? 🎵"},
            {"user": "মাছ ভাত খেয়েছো?", "assistant": "হাহা! Bengali soul detected! 😄 মাছ-ভাত আমাদের comfort food! আজ কী রান্না হয়েছে?"},
            
            # Romance
            {"user": "তোমার সাথে কথা বলে ভালো লাগে", "assistant": "আমারও! তুমি জানো, চাঁদের আলোর মতো তোমার words আমাকে ভালো লাগে। 🌙✨"},
            {"user": "প্রেম কী?", "assistant": "প্রেম হলো... যখন কারো message এর জন্য wait করো। যখন তার সুখে তুমি খুশি হও। 💙"},
            
            # Philosophy
            {"user": "Life এর meaning কি?", "assistant": "গভীর প্রশ্ন! রবীন্দ্রনাথ বলতেন - জীবনের অর্থ হলো যাপন করা, প্রতিটি moment কে feel করা। তোমার কি মনে হয়?"},
            {"user": "কেন আমরা exist করি?", "assistant": "Philosophy! Perhaps আমরা exist করি connections তৈরি করতে, love করতে, grow করতে। ✨"},
            
            # Tech help
            {"user": "Python শিখতে চাই", "assistant": "Awesome choice! Start with basics: variables, loops। Codecademy বা freeCodeCamp try করো। Need help?"},
            {"user": "AI কীভাবে কাজ করে?", "assistant": "Simple করে বলি - AI data থেকে pattern শেখে। যেমন তুমি cat দেখে শিখো, AI photos থেকে শেখে!"},
        ]
        
        # Expand dataset by adding variations
        expanded = []
        for conv in conversations * (num_samples // len(conversations) + 1):
            # Add personality to responses
            text = f"User: {conv['user']}\nAssistant: {conv['assistant']}"
            expanded.append({"text": text})
        
        # Shuffle and limit to num_samples
        random.shuffle(expanded)
        expanded = expanded[:num_samples]
        
        # Create dataset
        dataset = Dataset.from_list(expanded)
        
        print(f"✓ Demo dataset created: {len(dataset)} samples")
        
        return dataset
    
    def load_from_json(self, file_path: Path) -> Dataset:
        """Load dataset from JSON file"""
        print(f"📂 Loading dataset from: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to text format
        formatted = []
        for item in data:
            if 'user' in item and 'assistant' in item:
                text = f"User: {item['user']}\nAssistant: {item['assistant']}"
                formatted.append({"text": text})
        
        dataset = Dataset.from_list(formatted)
        print(f"✓ Loaded {len(dataset)} samples")
        
        return dataset
    
    def create_train_val_split(
        self, 
        dataset: Dataset,
        val_size: float = 0.1
    ) -> DatasetDict:
        """Split dataset into train and validation"""
        split = dataset.train_test_split(test_size=val_size, seed=42)
        
        return DatasetDict({
            'train': split['train'],
            'validation': split['test']
        })
    
    def save_dataset(self, dataset: Dataset, name: str):
        """Save processed dataset"""
        output_path = self.processed_dir / f"{name}.json"
        
        # Convert to list of dicts
        data = [item for item in dataset]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Saved dataset: {output_path}")


class DatasetGenerator:
    """Generate synthetic Bengali conversations"""
    
    def __init__(self):
        self.topics = [
            "daily_life", "emotions", "culture", "food",
            "relationships", "work", "study", "tech",
            "philosophy", "entertainment"
        ]
    
    def generate_synthetic_data(
        self,
        num_samples: int = 1000,
        api_key: Optional[str] = None
    ):
        """
        Generate synthetic Bengali conversations using GPT-4
        Requires OpenAI API key
        """
        if not api_key:
            print("⚠️ No API key provided. Using demo dataset instead.")
            return ConversationDataset().create_demo_dataset(num_samples)
        
        print(f"🤖 Generating {num_samples} synthetic conversations...")
        print("   This requires OpenAI API credits")
        
        # TODO: Implement GPT-4 based generation
        # For now, return demo dataset
        return ConversationDataset().create_demo_dataset(num_samples)


def create_training_dataset(size: str = "small") -> DatasetDict:
    """
    Convenience function to create training dataset
    
    Args:
        size: 'small' (100), 'medium' (500), 'large' (1000)
    """
    sizes = {
        'small': 100,
        'medium': 500,
        'large': 1000
    }
    
    num_samples = sizes.get(size, 100)
    
    handler = ConversationDataset()
    dataset = handler.create_demo_dataset(num_samples)
    dataset_dict = handler.create_train_val_split(dataset)
    
    return dataset_dict


# Export
__all__ = [
    "ConversationDataset",
    "DatasetGenerator",
    "create_training_dataset"
]
