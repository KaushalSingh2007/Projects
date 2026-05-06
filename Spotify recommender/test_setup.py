"""
Script to test the optimized Moodify setup.

This script verifies that all required dependencies are installed
and that the project structure is correct.

Author: Moodify Team
"""

import os
import sys

def check_dependencies():
    """Check if all required dependencies are installed."""
    dependencies = [
        'streamlit',
        'cv2',  # opencv-python
        'tensorflow',
        'spotipy',
        'dotenv',  # python-dotenv
        'numpy'
    ]
    
    missing_deps = []
    
    for dep in dependencies:
        try:
            if dep == 'cv2':
                import cv2
            elif dep == 'dotenv':
                from dotenv import load_dotenv
            else:
                __import__(dep)
            print(f"✅ {dep} is installed")
        except ImportError:
            print(f"❌ {dep} is not installed")
            missing_deps.append(dep)
    
    return len(missing_deps) == 0

def check_project_structure():
    """Check if the project structure is correct."""
    required_files = [
        'app.py',
        'train.py',
        'requirements.txt',
        'README.md',
        '.gitignore',
        '.env.example'
    ]
    
    required_dirs = [
        'utils',
        'data'
    ]
    
    missing_files = []
    missing_dirs = []
    
    # Check files
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ Missing file: {file}")
            missing_files.append(file)
        else:
            print(f"✅ Found file: {file}")
    
    # Check directories
    for directory in required_dirs:
        if not os.path.exists(directory):
            print(f"❌ Missing directory: {directory}")
            missing_dirs.append(directory)
        else:
            print(f"✅ Found directory: {directory}")
    
    return len(missing_files) == 0 and len(missing_dirs) == 0

def check_utils_modules():
    """Check if utility modules can be imported."""
    utils_modules = [
        'utils.emotion_predictor',
        'utils.spotify_utils'
    ]
    
    missing_modules = []
    
    for module in utils_modules:
        try:
            __import__(module)
            print(f"✅ {module} can be imported")
        except ImportError as e:
            print(f"❌ {module} cannot be imported: {e}")
            missing_modules.append(module)
    
    return len(missing_modules) == 0

def main():
    """Main function to run all checks."""
    print("=== Moodify Optimized Setup Test ===\n")
    
    print("1. Checking dependencies...")
    deps_ok = check_dependencies()
    
    print("\n2. Checking project structure...")
    structure_ok = check_project_structure()
    
    print("\n3. Checking utility modules...")
    utils_ok = check_utils_modules()
    
    print("\n=== Summary ===")
    if deps_ok and structure_ok and utils_ok:
        print("✅ All checks passed! The optimized setup is ready.")
        print("\nTo run the application:")
        print("  streamlit run app.py")
        print("\nTo train the model:")
        print("  python train.py --train_dir data/train --val_dir data/test --epochs 30")
    else:
        print("❌ Some checks failed:")
        if not deps_ok:
            print("   - Install missing dependencies: pip install -r requirements.txt")
        if not structure_ok:
            print("   - Check the project structure")
        if not utils_ok:
            print("   - Check the utility modules")

if __name__ == "__main__":
    main()