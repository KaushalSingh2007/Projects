"""
Script to run the Moodify application.

This script provides a simple way to start the Moodify Streamlit application
with proper error handling and environment checking.

Author: Moodify Team
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import streamlit
        import tensorflow
        import cv2
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def check_model():
    """Check if the model file exists."""
    model_files = ["emotion_model.keras", "emotion_model.h5"]
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"✅ Found model file: {model_file}")
            return True
    
    print("❌ No model file found.")
    print("Please train the model first with: python train.py --train_dir data/train --val_dir data/test")
    return False

def check_env_file():
    """Check if .env file exists, if not suggest creating one."""
    if os.path.exists(".env"):
        print("✅ Environment file found")
        return True
    else:
        print("⚠️  No .env file found. Using .env.example as template.")
        print("Please create a .env file with your Spotify credentials.")
        return True  # Not critical for running the app

def main():
    """Main function to run the Moodify application."""
    print("=== Moodify - AI Mood-Based Music Recommender ===\n")
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    # Check model
    print("Checking model...")
    if not check_model():
        sys.exit(1)
    
    # Check environment file
    print("Checking environment...")
    if not check_env_file():
        sys.exit(1)
    
    # Run the Streamlit app
    print("\n🚀 Starting Moodify application...")
    print("The app will be available at http://localhost:8501")
    print("Press CTRL+C to stop the application.\n")
    
    try:
        subprocess.run(["streamlit", "run", "app.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running the application: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()