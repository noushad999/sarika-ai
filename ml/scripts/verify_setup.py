"""
Setup Verification Script
Checks if all dependencies and hardware are properly configured
"""

import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"🐍 Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 11:
        print("   ✅ Python version OK")
        return True
    else:
        print("   ❌ Python 3.11+ required")
        return False

def check_torch():
    """Check PyTorch installation"""
    try:
        import torch
        print(f"\n🔥 PyTorch Version: {torch.__version__}")
        print(f"   CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            print("   ✅ PyTorch + CUDA OK")
        else:
            print("   ⚠️  CUDA not available (CPU mode)")
        
        return True
    except ImportError:
        print("   ❌ PyTorch not installed")
        return False

def check_transformers():
    """Check Transformers installation"""
    try:
        import transformers
        print(f"\n🤗 Transformers Version: {transformers.__version__}")
        print("   ✅ Transformers OK")
        return True
    except ImportError:
        print("   ❌ Transformers not installed")
        return False

def check_peft():
    """Check PEFT installation"""
    try:
        import peft
        print(f"\n🎯 PEFT Version: {peft.__version__}")
        print("   ✅ PEFT OK (LoRA support)")
        return True
    except ImportError:
        print("   ❌ PEFT not installed")
        return False

def check_disk_space():
    """Check available disk space"""
    import shutil
    
    total, used, free = shutil.disk_usage(Path.cwd())
    
    free_gb = free // (2**30)
    total_gb = total // (2**30)
    
    print(f"\n💾 Disk Space:")
    print(f"   Total: {total_gb} GB")
    print(f"   Free: {free_gb} GB")
    
    if free_gb >= 100:
        print("   ✅ Sufficient space")
        return True
    else:
        print("   ⚠️  Low disk space (100GB+ recommended)")
        return False

def check_env_file():
    """Check .env file"""
    env_file = Path.cwd() / ".env"
    
    print(f"\n⚙️  Configuration:")
    if env_file.exists():
        print("   ✅ .env file exists")
        
        with open(env_file) as f:
            content = f.read()
            if "hf_xxxxx" in content or "your-" in content:
                print("   ⚠️  Please update .env with real values")
            else:
                print("   ✅ .env configured")
        return True
    else:
        print("   ❌ .env file missing")
        return False

def check_folders():
    """Check folder structure"""
    required_folders = [
        "ml/checkpoints",
        "ml/datasets",
        "models/teachers",
        "models/student",
        "models/final",
        "data/raw",
        "data/processed",
        "logs"
    ]
    
    print(f"\n📁 Folder Structure:")
    all_exist = True
    for folder in required_folders:
        path = Path.cwd() / folder
        if path.exists():
            print(f"   ✅ {folder}")
        else:
            print(f"   ❌ {folder} (creating...)")
            path.mkdir(parents=True, exist_ok=True)
            all_exist = False
    
    return all_exist

def main():
    """Run all checks"""
    print("="*60)
    print("       SARIKA AI - Setup Verification")
    print("="*60)
    
    checks = [
        check_python_version(),
        check_torch(),
        check_transformers(),
        check_peft(),
        check_disk_space(),
        check_env_file(),
        check_folders()
    ]
    
    print("\n" + "="*60)
    if all(checks):
        print("✅ All checks passed! Ready to train.")
    else:
        print("⚠️  Some checks failed. Please fix above issues.")
    print("="*60)

if __name__ == "__main__":
    main()
