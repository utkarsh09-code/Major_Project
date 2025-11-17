#!/usr/bin/env python3
"""
Setup script for the Real-time User Attentiveness Detection System.
"""

import os
import sys
import subprocess
from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop


def read_requirements(filename):
    """Read requirements from file."""
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]


def check_system_requirements():
    """Check if system meets requirements."""
    print("Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 9):
        print("ERROR: Python 3.9 or higher is required")
        return False
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Check for CUDA (optional)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA not available - will use CPU")
    except ImportError:
        print("⚠ PyTorch not installed - will install during setup")
    
    # Check for OpenCV
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError:
        print("⚠ OpenCV not installed - will install during setup")
    
    return True


def install_requirements():
    """Install required packages."""
    print("Installing required packages...")
    
    try:
        # Install requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install requirements: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    print("Creating directories...")
    
    directories = [
        "data",
        "data/models",
        "data/configs",
        "logs",
        "screenshots"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def download_models():
    """Download required models (placeholder for future implementation)."""
    print("Checking for required models...")
    
    # This would download pre-trained models in a real implementation
    # For now, just create placeholder files
    model_files = [
        "data/models/shape_predictor_68_face_landmarks.dat",
        "data/models/face_recognition_model.dat"
    ]
    
    for model_file in model_files:
        if not os.path.exists(model_file):
            # Create placeholder file
            os.makedirs(os.path.dirname(model_file), exist_ok=True)
            with open(model_file, 'w') as f:
                f.write("# Placeholder model file\n")
            print(f"⚠ Created placeholder: {model_file}")
        else:
            print(f"✓ Model exists: {model_file}")


class CustomInstall(install):
    """Custom install command."""
    
    def run(self):
        """Run custom installation."""
        if not check_system_requirements():
            sys.exit(1)
        
        # Run standard install
        install.run(self)
        
        # Post-installation tasks
        create_directories()
        download_models()
        
        print("\n" + "="*60)
        print("Installation completed successfully!")
        print("="*60)
        print("\nTo run the application:")
        print("  python main.py          # Command line interface")
        print("  python gui.py           # GUI interface")
        print("\nTo run tests:")
        print("  python tests/test_basic_functionality.py")
        print("\nFor more information, see README.md")


class CustomDevelop(develop):
    """Custom develop command."""
    
    def run(self):
        """Run custom development installation."""
        if not check_system_requirements():
            sys.exit(1)
        
        # Run standard develop
        develop.run(self)
        
        # Post-installation tasks
        create_directories()
        download_models()
        
        print("\n" + "="*60)
        print("Development installation completed successfully!")
        print("="*60)


# Read README for long description
def read_readme():
    """Read README file."""
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Real-time User Attentiveness Detection System"


setup(
    name="attentiveness-detection",
    version="1.0.0",
    author="Attentiveness Detection Team",
    author_email="contact@attentiveness-detection.com",
    description="Real-time user attentiveness detection using deep learning and computer vision",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/attentiveness-detection/attentiveness-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": [
            "pytest>=7.4.2",
            "black>=23.9.1",
            "flake8>=6.1.0",
        ],
        "gui": [
            "PySide6>=6.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "attentiveness-detection=main:main",
            "attentiveness-gui=gui:run_gui",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.json"],
    },
    cmdclass={
        "install": CustomInstall,
        "develop": CustomDevelop,
    },
    keywords=[
        "attentiveness",
        "attention",
        "computer-vision",
        "deep-learning",
        "eye-tracking",
        "face-detection",
        "gaze-estimation",
        "pose-estimation",
        "behavioral-analysis",
        "real-time",
        "monitoring",
        "education",
        "research",
    ],
    project_urls={
        "Bug Reports": "https://github.com/attentiveness-detection/attentiveness-detection/issues",
        "Source": "https://github.com/attentiveness-detection/attentiveness-detection",
        "Documentation": "https://github.com/attentiveness-detection/attentiveness-detection/blob/main/README.md",
    },
)


if __name__ == "__main__":
    # If running directly, perform installation
    if len(sys.argv) == 1:
        print("Installing Attentiveness Detection System...")
        
        if not check_system_requirements():
            sys.exit(1)
        
        if not install_requirements():
            sys.exit(1)
        
        create_directories()
        download_models()
        
        print("\n" + "="*60)
        print("Installation completed successfully!")
        print("="*60)
        print("\nTo run the application:")
        print("  python main.py          # Command line interface")
        print("  python gui.py           # GUI interface")
        print("\nTo run tests:")
        print("  python tests/test_basic_functionality.py")
        print("\nFor more information, see README.md")
    else:
        # Run setup normally
        setup() 