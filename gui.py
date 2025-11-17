#!/usr/bin/env python3
"""
GUI launcher for the attentiveness detection system.
Uses the modern professional GUI.
"""

import sys
import os
import warnings
import logging

# Suppress all warnings at startup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)

from src.gui.main_window import run_gui

def main():
    """Main function."""
    run_gui()

if __name__ == "__main__":
    main()
