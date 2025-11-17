# Robust Attentiveness Detection System

A production-grade attentiveness detection system that reliably computes attention scores from head pose, gaze stability, and movement magnitude with temporal smoothing and per-user calibration.

## Features

### Correctness & Reliability
- **Robust scoring**: Computes attentiveness from head pose (yaw/pitch/roll), gaze stability, and movement magnitude
- **Temporal smoothing**: Uses EWMA (Exponential Weighted Moving Average) filters to prevent sudden jumps
- **No brittle thresholds**: Smooth, continuous scoring without arbitrary barriers

### Human-First Design
- **Handles real-world conditions**: Works with glasses, beards, headscarves, partial occlusions, and lighting changes
- **Per-user calibration**: Learns individual baseline movement patterns during 5-10 second calibration phase
- **Tolerates micro-movements**: Doesn't penalize normal human micro-movements and natural glance-away moments
- **Graceful degradation**: Reduces reliance on gaze when signal quality is low (e.g., glasses glare)

### Deterministic Output
- **Score range**: [0, 100] per second
- **Binary label**: `attentive: true/false` with hysteresis to prevent flickering
- **Low latency**: <150ms per frame on CPU
- **Stable output**: Temporal smoothing ensures stable, realistic scores

### Robustness
- **Missing detections**: Handles no-face scenarios with exponential decay (20% per second)
- **Off-screen movement**: Tracks movement using optical flow
- **Quick head turns**: Applies refractory window to prevent harsh score drops
- **Low light**: Adjusts signal quality and reduces reliance on gaze

### Explainability
Returns intermediate signals for debugging:
- `face_present`: Boolean
- `yaw/pitch/roll`: Head pose angles in degrees
- `gaze_off_axis`: Gaze deviation from screen center in degrees
- `movement_px_per_sec`: Movement magnitude in pixels per second
- `blink_rate`: Blinks per minute (if available)
- `signal_quality`: [0, 1] quality metric

## Architecture

### Core Components

1. **RobustAttentivenessCalculator** (`src/core/robust_attentiveness.py`)
   - Main calculator class
   - Handles calibration, temporal smoothing, and scoring

2. **AttentivenessState**
   - State container for all measurements and temporal buffers
   - Manages calibration samples and hysteresis state

### Scoring Formula

The attentiveness score is computed as:

```
pose_penalty = clamp01((abs(yaw) + abs(pitch)) / 50°)
gaze_penalty = clamp01(gaze_off_axis / 30°)
movement_penalty = clamp01((movement - M_base) / (M_hi - M_base))

combined_penalty = 0.4 * pose_penalty + 0.4 * gaze_penalty + 0.2 * movement_penalty
raw_score = 1.0 - combined_penalty + blink_bonus

# Apply signal quality weighting and temporal smoothing
final_score = smooth(raw_score) * signal_quality_factor
```

Where:
- `M_base`: Per-user movement baseline (learned during calibration)
- `M_hi`: Per-user high movement threshold (90th percentile)
- `blink_bonus`: +0.05 if blink rate is 8-25/min

### Temporal Smoothing

- **EWMA filters**: Applied to pose, gaze, and movement measurements
  - Alpha values: 0.3 (pose/gaze), 0.4 (movement)
  - Higher alpha = more responsive, lower = more stable

- **Face absence decay**: When face is not detected
  - Exponential decay: 20% per second
  - After 2 seconds: `face_present = false`, `attentive = false`

### Binary Label Hysteresis

To prevent label flickering:
- **Become attentive**: Score > 65 for 2 seconds
- **Become inattentive**: Score < 55 for 2 seconds
- **Hysteresis zone**: Between 55-65, maintain current state

## Calibration

### Calibration Phase (5-10 seconds)

During calibration, the system learns:
1. **Movement baseline** (`M_base`): Median movement during normal attention
2. **Movement high threshold** (`M_hi`): 90th percentile movement
3. **Gaze jitter baseline**: Standard deviation of gaze offsets
4. **Pose variance baseline**: Standard deviation of pose angles

### Calibration Process

1. User looks at screen naturally for 5-10 seconds
2. System collects samples of pose, gaze, and movement
3. Computes per-user baselines
4. Calibration completes automatically

**Note**: During calibration, scores may be less accurate. After calibration, the system uses learned baselines for more personalized scoring.

## Usage

### Basic Example

```python
from src.core.robust_attentiveness import RobustAttentivenessCalculator
from src.core.face_detection import FaceDetector
from src.core.pose_estimation import PoseEstimator
from src.core.gaze_estimation import GazeEstimator

# Initialize
calc = RobustAttentivenessCalculator(calibration_duration_sec=8.0)
face_detector = FaceDetector()
pose_estimator = PoseEstimator()
gaze_estimator = GazeEstimator()

# Process frame
faces = face_detector.detect_faces(frame)
face_detected = len(faces) > 0

pose_result = None
gaze_result = None
if face_detected:
    bbox = faces[0]['bbox']
    pose_result = pose_estimator.estimate_pose(frame, bbox)
    gaze_result = gaze_estimator.estimate_gaze(frame, bbox)

# Calculate attentiveness
result = calc.calculate_attentiveness(
    frame=frame,
    face_detected=face_detected,
    pose_result=pose_result,
    gaze_result=gaze_result,
    behavioral_result=None
)

print(f"Score: {result['attentiveness_score']}/100")
print(f"Attentive: {result['attentive']}")
```

### Running the Example Script

```bash
python example_robust_attentiveness.py
```

**Controls**:
- Press `q` to quit
- Press `r` to reset calibration
- JSON output printed every second

### Output Format

```json
{
  "attentiveness_score": 75.3,
  "attentive": true,
  "face_present": true,
  "yaw": 2.1,
  "pitch": -1.5,
  "roll": 0.3,
  "gaze_off_axis": 5.2,
  "movement_px_per_sec": 8.5,
  "blink_rate": 15.0,
  "signal_quality": 0.85,
  "is_calibrating": false,
  "latency_ms": 45.2
}
```

## Edge Cases

### No Face Detected

- **< 2 seconds**: Score decays exponentially (20% per second)
- **> 2 seconds**: `face_present = false`, `attentive = false`
- Uses last known good state with decay

### Sudden Head Turns

- Applies short refractory window (≤1 second)
- Temporal smoothing prevents immediate harsh score drops
- EWMA filters smooth out sudden changes

### Low Light Conditions

- Signal quality automatically reduced
- Less reliance on gaze estimation
- More weight on pose and movement
- `signal_quality` reflects lighting conditions

### Partial Occlusions

- System continues to work with reduced confidence
- Signal quality reflects occlusion level
- Scores adjusted based on available information

### Glasses/Reflections

- Gaze confidence reduced automatically
- System relies more on head pose
- Signal quality reflects reduced gaze reliability

## Thresholds & Parameters

### Reference Thresholds

- **Pose max angle** (`theta_max`): 50° (yaw + pitch)
- **Gaze max angle**: 30° off-axis
- **Attentive threshold**: 65/100 (with 2s hysteresis)
- **Inattentive threshold**: 55/100 (with 2s hysteresis)

### Calibration Defaults

- **Calibration duration**: 8 seconds (configurable)
- **Movement baseline default**: 5 px/sec (learned per-user)
- **Movement high default**: 50 px/sec (learned per-user)

### Temporal Parameters

- **EWMA alpha (pose/gaze)**: 0.3
- **EWMA alpha (movement)**: 0.4
- **Hysteresis time**: 2.0 seconds
- **Face absence decay rate**: 0.2 (20% per second)

## Testing

Run unit tests:

```bash
python -m pytest tests/test_robust_attentiveness.py -v
```

Test coverage includes:
- Initialization
- No face detection
- Pose/gaze/movement penalties
- Temporal smoothing
- Face absence decay
- Binary label hysteresis
- Calibration
- Signal quality
- Reset functionality

## Performance

### Latency

- **Target**: <150ms per frame on CPU
- **Typical**: 30-80ms per frame
- Measured via `latency_ms` in output

### Optimization Tips

1. **Reduce frame resolution**: Downsample to 480p or 360p
2. **Lower FPS**: 15-20 FPS is often sufficient
3. **Skip frames**: Process every 2nd or 3rd frame
4. **Reduce optical flow features**: Fewer corners = faster

## Privacy & Fairness

- **No recording by default**: Frames are not stored
- **Opt-in logging**: Users can enable session logging
- **No demographics**: Thresholds are behavior-based, not appearance-based
- **Per-user calibration**: Adapts to individual movement patterns

## Common Pitfalls Avoided

✅ **No single-frame thresholds**: All measurements use temporal smoothing  
✅ **No harsh penalties**: Normal micro-movements are tolerated  
✅ **No hard resets**: Exponential decay instead of instant zero  
✅ **No perfect stillness assumption**: Handles natural human movement  
✅ **No confident low-light outputs**: Signal quality reflects conditions  

## Dependencies

- `numpy`: Numerical computations
- `opencv-python`: Computer vision and optical flow
- `mediapipe`: Face detection and pose estimation (optional, falls back to OpenCV)

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review unit tests for usage examples
3. Check signal quality values - low values indicate detection issues

