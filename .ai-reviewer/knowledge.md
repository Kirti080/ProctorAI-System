# ProctorAI-System reviewer notes

## Architecture

This is a computer-vision proctoring application with two execution paths: `dashboard.py` provides the Streamlit web UI, while `webcam_engine.py` provides a standalone OpenCV desktop runner. Both paths combine Ultralytics YOLO for phone detection, MediaPipe Face Mesh for gaze/blink checks, MediaPipe Hands for raised-hand requests, and `pyttsx3` for spoken alerts. The dashboard keeps session statistics and logs in `st.session_state`, and runs the camera-processing loop directly when the “SYSTEM ACTIVE” toggle is enabled.

## Conventions

- Detection models are initialized locally in the execution path rather than at module import:
  - `dashboard.py` initializes YOLO, Face Mesh, and Hands inside the `if run_proctoring:` block.
  - `webcam_engine.py` initializes models inside `start_smart_proctor()`.
- Phone detection uses COCO class `67`, confidence `0.5`, and deliberately skips frames for performance:
  - Every sixth frame in `dashboard.py`.
  - Every fifth frame in `webcam_engine.py`.
- Alerts use `ProctorVoice.say()` and are rate-limited with a cooldown. Speech runs in a daemon thread so camera/UI processing is not blocked (`dashboard.py`, `webcam_engine.py`).
- MediaPipe input is converted from OpenCV BGR to RGB before processing (`dashboard.py`, `webcam_engine.py`).
- Landmark-based rules use fixed thresholds:
  - Eye/blink threshold: vertical landmark distance `< 0.012`.
  - Gaze/depth tilt: approximately `0.030–0.035`.
  - Liveness failure after 10 seconds in `dashboard.py`, 15 seconds in `webcam_engine.py`.
- Dashboard state is initialized through `st.session_state` keys:
  - `voice`, `stats`, and `logs` in `dashboard.py`.
  - Violation categories are `Phone`, `Distracted`, `Spoofing`, and `Requests`.
- Dashboard event logs are newest-first, formatted as pipe-delimited timestamped strings by `record_event()` (`dashboard.py`).
- UI-facing dashboard alerts use Streamlit placeholders and inline HTML/CSS; the standalone engine uses `cv2.putText()` and an OpenCV window instead (`dashboard.py`, `webcam_engine.py`).

## Intentional non-standard choices

- Broad bare `except:` blocks in both `ProctorVoice._speak()` methods intentionally suppress platform/audio-driver failures so speech cannot terminate proctoring (`dashboard.py`, `webcam_engine.py`).
- Audio is disabled on Streamlit Cloud using `STREAMLIT_RUNTIME_ENV == "cloud"` in `dashboard.py`, avoiding unavailable audio drivers.
- YOLO is intentionally run less frequently than face/hand tracking to reduce CPU usage; do not treat the differing five- versus six-frame cadence as accidental.
- `dashboard.py` uses a long-running synchronous camera loop because the Streamlit page is designed as a live monitoring session, not a request/response endpoint.

## Watch out for

- Keep dashboard and standalone detection behavior aligned when changing thresholds, landmark indices, class IDs, or cooldown logic; the two implementations currently differ.
- Changes to violation categories must update both `record_event()` callers and the initial `st.session_state.stats` mapping, or dashboard execution can fail with a missing key.
- Ensure camera resources are released on errors or interruptions; both loops currently release `cap` only after normal loop exit.
- Avoid loading YOLO or MediaPipe repeatedly on Streamlit reruns unless intentional; model construction currently occurs whenever the active branch is entered.
- Check for duplicate event counting: detections and liveness alerts can be recorded on every qualifying frame, rather than once per incident.
- Preserve RGB conversion before MediaPipe and BGR handling for OpenCV drawing; mixing these formats will produce incorrect detection or display colors.