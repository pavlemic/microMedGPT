#!/usr/bin/env python3
"""
Simple script to run the complete training pipeline
Equivalent to the original monolithic script
"""

# Install required packages
# !pip install tokenizers tensorboard

from train import main

if __name__ == "__main__":
    print("Starting MicroMedGPT training...")
    model, data_loader = main()
    print("\nTraining completed! You can now:")
    print("1. Run 'python generate.py interactive' for interactive generation")
    print("2. Run 'python generate.py batch' for batch examples")
    print("3. Run 'tensorboard --logdir runs' to view training logs")
