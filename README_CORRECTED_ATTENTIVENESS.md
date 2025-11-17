# Corrected Robust Attentiveness Detection System

Production-ready attentiveness calculator with corrected formulas, robust baselines, and human-friendly thresholds. All corrections from the specification have been implemented.

## Key Corrections Implemented

### 1. Clamp Helpers & NaN Safety
- `clamp01(x)`: Clamps to [0, 1] with NaN safety
- All divisions use safe denominators (+1e-6)
- Prevents crashes from invalid inputs

### 2. Pose Penalty (Human-Scaled)
**Before**: `(abs(yaw) + abs(pitch)) / 50°`  
**After**: `max(abs(yaw) / 25.0, abs(pitch) / 20.0)`

- More tolerant of pitch (humans look down naturally)
- Uses max instead of sum (more accurate)
- Roll tracked but not penalized directly

### 3. Gaze Penalty (Dwell-Aware)
**Before**: Instantaneous penalty  
**After**: Dwell-aware with tolerance for micro-saccades

- Instantaneous: `clamp01(gaze_off_axis / 30.0)`
- Dwell gate: Full penalty only if mean off-axis >15° for ≥1.2s
- Otherwise: Scale by 0.5 (tolerates quick glances)

### 4. Movement Penalty (Robust Baselines)
- Safe denominator: `max(M_hi - M_base, 1e-6)`
- IQR sanity checks during calibration
- Auto-adjustment: If user is fidgety (narrow baseline), down-weight movement to 10%

### 5. Blink Bonus (Fixed Units)
**Before**: +5 "points" (incorrect)  
**After**: +0.05 in 0-1 space (correct)

- Only applies if 8 ≤ blink_rate ≤ 25 per minute
- Returns None if blink detection is unreliable (no guessing)

### 6. Signal Quality Mixer (Trust Mapping)
**Before**: Simple interpolation  
**After**: Neutral prior with quality-derived trust

```python
trust = clamp01((signal_quality - 0.3) / 0.7)  # 0 at 0.3, 1 at 1.0
raw_score = (1.0 - trust) * 0.5 + trust * raw_score
```

- Prevents overly generous 50s at very low quality
- Avoids harsh penalties when signal is poor
- Neutral prior: 0.5 (middle ground)

### 7. Face Absence Decay
**Before**: Linear decay  
**After**: Exponential decay `0.8^seconds_absent`

- 20% per second (more realistic)
- After 2 seconds: `face_present = false`, `attentive = false`

### 8. Sudden Turn Refractory
- Detects sudden turns: |Δyaw| > 40° in < 300ms
- Applies ≤1.0s refractory window
- Prevents score from crashing during quick orienting movements

### 9. Temporal Smoothing (Tuned EWMA)
- Pose: α = 0.25
- Gaze: α = 0.35
- Movement: α = 0.30
- Score: α = 0.30 (second-level smoothing)

### 10. Hysteresis (Timer-Based)
**Before**: Frame-based  
**After**: Timer-based (seconds, not frames)

- Requires continuous ≥2s beyond thresholds to flip
- FPS-independent behavior
- Prevents label flickering

## Scoring Formula (Corrected)

```python
# Individual penalties
pose_penalty = clamp01(max(abs(yaw) / 25.0, abs(pitch) / 20.0))
gaze_penalty_inst = clamp01(gaze_off_axis / 30.0)
gaze_penalty = 0.5 * gaze_penalty_inst if dwell_time < 1.2s else gaze_penalty_inst
movement_penalty = clamp01((movement - M_base) / max(M_hi - M_base, 1e-6))

# Combined (weights adjust for fidgety users)
combined_penalty = 0.4 * pose_penalty + 0.4 * gaze_penalty + 0.2 * movement_penalty

# Raw score
blink_bonus = 0.05 if (8 <= blink_rate <= 25 and blink_rate is not None) else 0.0
raw_score = clamp01(1.0 - combined_penalty + blink_bonus)

# Signal quality trust mapping
trust = clamp01((signal_quality - 0.3) / 0.7)
raw_score = (1.0 - trust) * 0.5 + trust * raw_score

# Face absence decay
if not face_present:
    raw_score = last_good_score * (0.8 ** seconds_absent)

# Temporal smoothing (EWMA)
smoothed_score = 0.3 * raw_score + 0.7 * last_smoothed_score

# Final score
score = int(round(100.0 * clamp01(smoothed_score)))
```

## Output Schema (Per Second)

```json
{
  "score": 75,
  "attentive": true,
  "face_present": true,
  "yaw": 2.1,
  "pitch": -1.5,
  "roll": 0.3,
  "gaze_off_axis": 5.2,
  "movement": 8.5,
  "blink_rate": 15.0,
  "signal_quality": 0.85
}
```

## Usage

### Basic Example

```python
from src.core.robust_attentiveness import RobustAttentivenessCalculator

calc = RobustAttentivenessCalculator(calibration_duration_sec=8.0)

result = calc.calculate_attentiveness(
    frame=frame,
    face_detected=True,
    pose_result={'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 'confidence': 0.9},
    gaze_result={'gaze_direction': (0.0, 0.0), 'confidence': 0.8},
    behavioral_result={'blink_rate': 15.0}
)

print(f"Score: {result['score']}/100")
print(f"Attentive: {result['attentive']}")
```

### Running the Example

```bash
python example_robust_attentiveness_corrected.py
```

**Controls**:
- Press `q` to quit
- Press `r` to reset calibration
- JSON output printed every second

### Running Tests

```bash
python -m pytest tests/test_robust_attentiveness_corrected.py -v
```

## Calibration

### Process (5-10 seconds)

1. User looks at screen naturally
2. System collects samples of:
   - Movement (median = M_base, 90th percentile = M_hi, IQR for sanity)
   - Gaze jitter (std of off-axis angles)
   - Pose variance (std of pose angles)
3. Computes per-user baselines
4. Auto-adjusts weights if user is naturally fidgety

### Calibration Output

```
Calibration complete: M_base=5.2, M_hi=18.5, IQR=8.3
```

## Performance

- **Target latency**: <150ms per frame on CPU
- **Typical latency**: 30-80ms per frame
- **FPS**: 15-30 FPS depending on resolution
- **Memory**: Minimal (no frame storage by default)

## Edge Cases Handled

1. **No face detected**: Exponential decay (0.8^seconds), after 2s: `attentive = false`
2. **Sudden head turns**: Refractory window (≤1s grace period)
3. **Low light**: Signal quality reduced, trust mapping prevents harsh penalties
4. **Glasses/reflections**: Gaze confidence reduced, system relies more on pose
5. **Fidgety users**: Movement weight auto-adjusted to 10%
6. **Narrow baselines**: Safe denominators prevent division by zero

## Validation Checklist

✅ **Clamp helpers**: `clamp01` with NaN safety  
✅ **Pose penalty**: `max(|yaw|/25, |pitch|/20)`  
✅ **Gaze penalty**: Dwell-aware (≥1.2s sustained >15°)  
✅ **Movement penalty**: Safe denominators, IQR checks  
✅ **Blink bonus**: 0.05 in 0-1 space (not +5)  
✅ **Signal quality**: Trust mapping with neutral prior  
✅ **Face absence**: 0.8^seconds exponential decay  
✅ **Sudden turns**: Refractory window detection  
✅ **Temporal smoothing**: Tuned EWMA alphas  
✅ **Hysteresis**: Timer-based (seconds, not frames)  

## Limitations

1. **Calibration required**: First 5-10 seconds needed for baseline learning
2. **Gaze accuracy**: Depends on lighting and face visibility
3. **Movement tracking**: Optical flow may be noisy in low-texture scenes
4. **Blink detection**: Optional, may be unreliable in some conditions

## Privacy

- **No frame storage**: Frames processed in-memory only
- **No recording**: By default, no video/photo capture
- **Opt-in logging**: Users can enable session logging if needed

## Dependencies

- `numpy`: Numerical computations
- `opencv-python`: Computer vision and optical flow
- `mediapipe`: Face detection and pose estimation (optional)

## License

MIT License

