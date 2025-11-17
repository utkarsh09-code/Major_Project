#!/usr/bin/env python3
"""
Comprehensive Setup Script
Downloads all dependencies and models required for the project.
"""

import os
import sys
import subprocess
import urllib.request
import urllib.error
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Model URLs
DLIB_SHAPE_PREDICTOR_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"


def check_python_version():
    """Check if Python version is compatible."""
    logger.info("Checking Python version...")
    if sys.version_info < (3, 9):
        logger.error("ERROR: Python 3.9 or higher is required")
        logger.error(f"Current version: {sys.version}")
        return False
    logger.info(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True


def create_directories():
    """Create necessary directories."""
    logger.info("\nCreating directories...")
    directories = [
        "data",
        "data/models",
        "data/configs",
        "data/anonymized",
        "logs",
        "screenshots"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"✓ Created/verified: {directory}")


def install_requirements():
    """Install all Python dependencies."""
    logger.info("\n" + "="*60)
    logger.info("Installing Python dependencies...")
    logger.info("="*60)
    
    try:
        # Upgrade pip first
        logger.info("\nUpgrading pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info("✓ pip upgraded")
        
        # Install requirements one by one for better error handling
        logger.info("\nInstalling requirements from requirements.txt...")
        with open("requirements.txt", "r") as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        
        failed_packages = []
        for req in requirements:
            try:
                logger.info(f"Installing {req}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", req], 
                                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                logger.info(f"✓ {req}")
            except subprocess.CalledProcessError as e:
                logger.warning(f"⚠ Failed to install {req}: {e}")
                failed_packages.append(req)
        
        if failed_packages:
            logger.warning(f"\n⚠ Some packages failed to install: {', '.join(failed_packages)}")
            logger.warning("  These may be optional or may require different installation methods.")
            logger.warning("  The system may still work with the successfully installed packages.")
        
        logger.info("\n✓ Requirements installation completed")
        return True
    except FileNotFoundError:
        logger.error("✗ ERROR: requirements.txt not found")
        return False
    except Exception as e:
        logger.error(f"✗ ERROR: Unexpected error during installation: {e}")
        return False


def download_dlib_model():
    """Download dlib shape predictor model."""
    logger.info("\n" + "="*60)
    logger.info("Downloading dlib models...")
    logger.info("="*60)
    
    model_dir = Path("data/models")
    model_file = model_dir / "shape_predictor_68_face_landmarks.dat"
    compressed_file = model_dir / "shape_predictor_68_face_landmarks.dat.bz2"
    
    # Check if already downloaded
    if model_file.exists():
        logger.info(f"✓ Dlib model already exists: {model_file}")
        return True
    
    try:
        logger.info(f"Downloading dlib shape predictor from: {DLIB_SHAPE_PREDICTOR_URL}")
        logger.info("This may take a few minutes (file size ~100MB)...")
        
        # Download compressed file
        urllib.request.urlretrieve(DLIB_SHAPE_PREDICTOR_URL, compressed_file)
        logger.info("✓ Downloaded compressed model file")
        
        # Check if bz2 module is available for decompression
        try:
            import bz2
            logger.info("Decompressing model file...")
            with bz2.open(compressed_file, 'rb') as f_in:
                with open(model_file, 'wb') as f_out:
                    f_out.write(f_in.read())
            logger.info("✓ Model decompressed successfully")
            
            # Remove compressed file
            compressed_file.unlink()
            logger.info(f"✓ Dlib model saved to: {model_file}")
            return True
        except ImportError:
            logger.warning("⚠ bz2 module not available. Please manually decompress:")
            logger.warning(f"  {compressed_file}")
            logger.warning("  You can use: bzip2 -d shape_predictor_68_face_landmarks.dat.bz2")
            return False
            
    except urllib.error.URLError as e:
        logger.warning(f"⚠ Could not download dlib model: {e}")
        logger.warning("  Note: dlib is optional. The system works with MediaPipe and OpenCV.")
        return False
    except Exception as e:
        logger.warning(f"⚠ Error downloading dlib model: {e}")
        logger.warning("  Note: dlib is optional. The system works with MediaPipe and OpenCV.")
        return False


def initialize_models():
    """Initialize models to trigger automatic downloads (MediaPipe, DeepFace)."""
    logger.info("\n" + "="*60)
    logger.info("Initializing models (triggering automatic downloads)...")
    logger.info("="*60)
    
    try:
        # Initialize MediaPipe (downloads models automatically)
        logger.info("\nInitializing MediaPipe...")
        import mediapipe as mp
        mp_face_detection = mp.solutions.face_detection
        mp_face_mesh = mp.solutions.face_mesh
        
        # Create instances to trigger model download
        face_detection = mp_face_detection.FaceDetection(model_selection=1)
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
        logger.info("✓ MediaPipe models initialized (models downloaded automatically)")
        
    except Exception as e:
        logger.warning(f"⚠ Error initializing MediaPipe: {e}")
    
    try:
        # Initialize DeepFace (downloads models automatically on first use)
        logger.info("\nInitializing DeepFace...")
        from deepface import DeepFace
        logger.info("✓ DeepFace imported (models will download on first use)")
        logger.info("  Note: DeepFace models are large and download on first actual use")
        
    except ImportError:
        logger.warning("⚠ DeepFace not available (optional dependency)")
    except Exception as e:
        logger.warning(f"⚠ Error initializing DeepFace: {e}")
    
    try:
        # Check OpenCV
        logger.info("\nChecking OpenCV...")
        import cv2
        # Check if Haar cascades are available
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if os.path.exists(cascade_path):
            logger.info("✓ OpenCV Haar cascades available")
        else:
            logger.warning("⚠ OpenCV Haar cascades not found")
            
    except Exception as e:
        logger.warning(f"⚠ Error checking OpenCV: {e}")


def verify_installation():
    """Verify that all critical components are installed."""
    logger.info("\n" + "="*60)
    logger.info("Verifying installation...")
    logger.info("="*60)
    
    critical_modules = [
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("mediapipe", "MediaPipe"),
        ("torch", "PyTorch"),
        ("PIL", "Pillow"),
        ("sklearn", "scikit-learn"),
    ]
    
    optional_modules = [
        ("dlib", "Dlib"),
        ("deepface", "DeepFace"),
        ("PySide6", "PySide6"),
    ]
    
    all_ok = True
    
    logger.info("\nCritical modules:")
    for module_name, display_name in critical_modules:
        try:
            __import__(module_name)
            logger.info(f"✓ {display_name}")
        except ImportError:
            logger.error(f"✗ {display_name} - NOT INSTALLED")
            all_ok = False
    
    logger.info("\nOptional modules:")
    for module_name, display_name in optional_modules:
        try:
            __import__(module_name)
            logger.info(f"✓ {display_name}")
        except ImportError:
            logger.warning(f"⚠ {display_name} - Not installed (optional)")
    
    return all_ok


def main():
    """Main setup function."""
    logger.info("\n" + "="*60)
    logger.info("Comprehensive Setup for Attentiveness Detection System")
    logger.info("="*60)
    logger.info("\nThis script will:")
    logger.info("  1. Install all Python dependencies")
    logger.info("  2. Create necessary directories")
    logger.info("  3. Download required model files")
    logger.info("  4. Initialize models (trigger automatic downloads)")
    logger.info("  5. Verify installation")
    logger.info("\n" + "="*60 + "\n")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        logger.error("\n✗ Installation failed. Please check the errors above.")
        sys.exit(1)
    
    # Download dlib model (optional)
    download_dlib_model()
    
    # Initialize models (triggers automatic downloads)
    initialize_models()
    
    # Verify installation
    if not verify_installation():
        logger.error("\n✗ Some critical modules are missing. Please check the errors above.")
        sys.exit(1)
    
    logger.info("\n" + "="*60)
    logger.info("✓ Setup completed successfully!")
    logger.info("="*60)
    logger.info("\nNext steps:")
    logger.info("  • Run tests: python test_system.py")
    logger.info("  • Run demo: python demo.py")
    logger.info("  • Run GUI: python gui.py")
    logger.info("  • Run CLI: python main.py")
    logger.info("\nNote: DeepFace models will download automatically on first use.")
    logger.info("      This may take several minutes depending on your internet connection.")
    logger.info("\n")


if __name__ == "__main__":
    main()

