#!/usr/bin/env python3

"""
Training script for the localization environment using PyMARL.
Usage: python train_localization.py --config=qmix --env-config=localization
"""

import sys
import os

# Add pymarl source to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pymarl_src'))

# Change directory to pymarl_src so relative imports work correctly
original_dir = os.getcwd()
os.chdir(os.path.join(os.path.dirname(__file__), 'pymarl_src'))

try:
    # Execute main.py directly
    exec(open('main.py').read())
finally:
    # Restore original directory
    os.chdir(original_dir)