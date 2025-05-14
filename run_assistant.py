#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Smart Virtual Assistant Runner

This script runs the Smart Virtual Assistant application.
Q
Author: AI Assistant
Date: April 2025
"""

# GPU optimization settings - must be before other imports
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Try to set memory growth to avoid taking all GPU memory
try:
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"Found {len(gpus)} GPU(s)")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU {gpu} set to memory growth mode")
except Exception as e:
    print(f"GPU configuration error: {e}")

import sys
import argparse
from smart_assistant.main import SmartAssistant

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Smart Virtual Assistant")
    parser.add_argument("--calibrate", action="store_true", help="Run camera calibration")
    parser.add_argument("--use-gpu", action="store_true", default=True, help="Use GPU acceleration if available")
    args = parser.parse_args()
    
    print("Starting Smart Virtual Assistant...")
    
    # Create data directories if they don't exist
    os.makedirs(os.path.join("smart_assistant", "data"), exist_ok=True)
    
    # Create the Smart Assistant instance and run it
    assistant = SmartAssistant(use_gpu=args.use_gpu)
    
    if args.calibrate:
        print("Running camera calibration mode...")
        print("TIP: If you don't have a physical chessboard, the program will use sample")
        print("     images from the 'chessboard_images' directory.")
        calibration_success = assistant.camera_calibration.calibrate(assistant.cap)
        if calibration_success:
            print("Camera calibration completed successfully!")
        else:
            print("Camera calibration failed or was cancelled.")
    else:
        assistant.run()