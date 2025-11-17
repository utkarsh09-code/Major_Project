# Real-time User Attentiveness Detection System

A comprehensive, robust, and efficient desktop application for real-time user attentiveness detection during video playback. This system continuously monitors and quantifies a user's focus and engagement by integrating multiple non-verbal cues, including face detection, eye movement, head pose, and other relevant behavioral indicators.

## 🚀 Current Status

✅ **System is fully functional and working!**

The system has been successfully set up and tested with the following components:
- **Face Detection**: MediaPipe (primary), OpenCV (fallback)
- **Gaze Estimation**: MediaPipe face mesh
- **Pose Estimation**: MediaPipe face mesh
- **Camera Access**: Working
- **Real-time Processing**: Functional

## 🎯 Key Features

### Core Functionality
- **Real-time Face Detection**: Multi-backend support with MediaPipe and OpenCV
- **Eye Gaze Estimation**: Tracks where the user is looking
- **Head Pose Analysis**: Monitors head orientation (pitch, yaw, roll)
- **Behavioral Cues Detection**: Blinking, yawning, and movement patterns
- **Attention Scoring**: Quantifiable attention level calculation
- **Privacy-Conscious**: Optional data anonymization and minimal permissions

### Technical Features
- **Multi-backend Architecture**: Fallback mechanisms for reliability
- **GPU Acceleration**: Optional CUDA support for improved performance
- **Real-time Processing**: Optimized for low latency
- **Modular Design**: Easy to extend and customize
- **Comprehensive Logging**: Detailed performance and error tracking

## 📋 System Requirements

### Minimum Requirements
- **Python**: 3.9+ (tested with Python 3.12)
- **RAM**: 4GB
- **Storage**: 2GB free space
- **Camera**: Webcam or USB camera
- **OS**: Windows 10/11, macOS, or Linux

### Recommended Requirements
- **Python**: 3.11+
- **RAM**: 8GB+
- **GPU**: NVIDIA GPU with CUDA support (optional)
- **Camera**: HD webcam (720p or higher)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Major_Project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Note**: The system works without `dlib` (which requires CMake compilation). All core functionality uses MediaPipe and OpenCV.

### 3. Verify Installation
```bash
python test_system.py
```

You should see output indicating all components are working:
```
🎉 All tests passed! System is working without dlib.
```

## 🚀 Quick Start

### Basic Demo
Run the real-time attentiveness detection demo:
```bash
python demo.py
```

**Controls:**
- Press `q` to quit
- Press `s` to save a screenshot

### Command Line Interface
```bash
python main.py
```

### GUI Application
```bash
python gui.py
```

## 📁 Project Structure

```
Major_Project/
├── src/
│   ├── core/                 # Core detection modules
│   │   ├── face_detection.py
│   │   ├── gaze_estimation.py
│   │   ├── pose_estimation.py
│   │   ├── behavioral_cues.py
│   │   ├── multimodal_fusion.py
│   │   └── attentiveness_scoring.py
│   ├── gui/                  # GUI components
│   │   ├── main_window.py
│   │   ├── camera_widget.py
│   │   └── permission_handler.py
│   └── utils/                # Utilities
│       ├── config.py
│       ├── logger.py
│       └── privacy.py
├── tests/                    # Test files
├── requirements.txt          # Python dependencies
├── demo.py                  # Real-time demo
├── main.py                  # CLI interface
├── gui.py                   # GUI launcher
└── README.md               # This file
```

## 🔧 Configuration

The system uses a flexible configuration system. Key settings can be modified in `src/utils/config.py`:

### Camera Settings
- Resolution and frame rate
- Device selection
- GPU acceleration options

### Model Settings
- Detection confidence thresholds
- Backend selection (MediaPipe, OpenCV)
- Performance optimization options

### Privacy Settings
- Data retention policies
- Anonymization options
- Permission management

## 📊 Usage Examples

### Real-time Monitoring
```python
from src.core.face_detection import FaceDetector
from src.core.gaze_estimation import GazeEstimator
from src.core.pose_estimation import PoseEstimator

# Initialize components
face_detector = FaceDetector(backend="mediapipe")
gaze_estimator = GazeEstimator(backend="mediapipe")
pose_estimator = PoseEstimator(backend="mediapipe")

# Process frame
faces = face_detector.detect_faces(frame)
if faces:
    gaze_result = gaze_estimator.estimate_gaze(frame)
    pose_result = pose_estimator.estimate_pose(frame)
    # Calculate attention score...
```

### Attention Scoring
```python
from src.core.attentiveness_scoring import AttentivenessScorer

scorer = AttentivenessScorer()
attention_score = scorer.calculate_attention_score(
    gaze_result, pose_result, behavioral_cues
)
```

## 🔍 Troubleshooting

### Common Issues

**1. Camera not accessible**
- Check camera permissions
- Ensure no other application is using the camera
- Try different camera index: `cv2.VideoCapture(1)`

**2. Low performance**
- Reduce frame resolution
- Disable GPU acceleration if causing issues
- Close other applications

**3. No faces detected**
- Ensure good lighting
- Position face clearly in camera view
- Check camera focus

**4. Import errors**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version compatibility

### Performance Optimization

**For better performance:**
1. Use GPU acceleration if available
2. Reduce frame resolution
3. Lower detection confidence thresholds
4. Close unnecessary applications

**For better accuracy:**
1. Ensure good lighting
2. Position face clearly in view
3. Minimize background movement
4. Use HD camera if available

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **MediaPipe**: For excellent face detection and mesh capabilities
- **OpenCV**: For computer vision foundations
- **TensorFlow**: For deep learning support
- **PySide6**: For modern GUI framework

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review existing issues
3. Create a new issue with detailed information

---

**Note**: This system is designed to work without `dlib` for easier installation. All core functionality uses MediaPipe and OpenCV, providing excellent performance and reliability. 