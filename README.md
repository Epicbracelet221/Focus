# FocusFlow

FocusFlow is a real-time computer vision media controller that uses eye tracking and hand gestures to control video playback and system volume. It works with browser videos and external media players by sending OS-level media key commands.

## Project Description
FocusFlow improves hands-free media interaction during study/work sessions.

- Eye-based state control:
  - Eyes closed for a short stable duration -> pause media
  - Eyes open for a short stable duration -> resume media
  - Eyes closed for a long duration -> shutdown FocusFlow
- Hand gesture volume control:
  - Short thumb-index distance -> volume down
  - Large thumb-index distance -> volume up
  - Mid distance -> no volume change
- Global exit support:
  - `Ctrl+Shift+X` when `pynput` is installed
  - Fallback: `q` or `Esc` in the FocusFlow window

## Working of the Project
### 1. Initialization
- Opens webcam using OpenCV.
- Initializes MediaPipe Face Mesh and Hands pipelines.
- Starts optional global hotkey listener (`Ctrl+Shift+X`).
- Loads runtime thresholds and cooldown timers.

### 2. Frame Processing Loop
For each camera frame:
- Convert BGR -> RGB and run:
  - Face mesh detection every frame
  - Hand detection every `N` frames (`HAND_PROCESS_EVERY_N_FRAMES`) for efficiency
- Compute Eye Aspect Ratio (EAR) from eye landmarks.
- Smooth EAR with a moving average window (`EAR_SMOOTHING_WINDOW`) to reduce fluctuation.
- Update eye-open/eye-closed counters.

### 3. Pause/Play Logic
- If eye-closed counter exceeds `PAUSE_PLAY_STABILITY_FRAMES` and state is playing -> send `playpause`.
- If eye-open counter exceeds `PAUSE_PLAY_STABILITY_FRAMES` and state is paused -> send `playpause`.
- `COOLDOWN` prevents rapid toggling.

### 4. Exit Logic
- If eye-closed counter exceeds `SHUTDOWN_EYE_CLOSED_FRAMES` -> graceful shutdown.
- If global event `Ctrl+Shift+X` is detected -> graceful shutdown.
- If `q` or `Esc` is pressed in display window -> graceful shutdown.

### 5. Volume Logic
- Thumb-index Euclidean distance controls volume commands:
  - `< 50` -> `volumedown`
  - `> 180` -> `volumeup`
  - otherwise stable
- `VOLUME_COOLDOWN` throttles repeated volume key presses.

### 6. Cleanup
- Releases camera and destroys OpenCV windows.
- Stops global keyboard listener if active.

## Control Flow Graph (CFG)
```mermaid
flowchart TD
    A[Start Program] --> B[Init Camera + MediaPipe + Config + Optional Hotkey Listener]
    B --> C{Camera Opened?}
    C -- No --> Z[Exit with Error]
    C -- Yes --> D[Read Frame]
    D --> E{Frame Read Success?}
    E -- No --> D
    E -- Yes --> F[Process Face]
    F --> G[Compute + Smooth EAR]
    G --> H[Update Eye Counters]
    H --> I{Global Exit / Long Eye Closure / Key Exit?}
    I -- Yes --> Y[Cleanup + Exit]
    I -- No --> J{Pause/Play Conditions Met?}
    J -- Yes --> K[Send playpause]
    J -- No --> L[Skip Toggle]
    K --> M[Process Hand (every N frames)]
    L --> M
    M --> N{Hand Detected?}
    N -- Yes --> O[Compute Finger Distance]
    O --> P{Distance Zone}
    P -- Low --> Q[Send volumedown]
    P -- High --> R[Send volumeup]
    P -- Mid --> S[No Volume Command]
    N -- No --> T[No Volume Action]
    Q --> U[Render Overlay]
    R --> U
    S --> U
    T --> U
    U --> D
    Y --> End[Program End]
```

## Configuration (Key Parameters)
Defined in `focus_flow.py`:

- `EYE_CLOSED_THRESHOLD`: EAR threshold to classify eye closure
- `PAUSE_PLAY_STABILITY_FRAMES`: stable frame count before pause/play action
- `SHUTDOWN_EYE_CLOSED_FRAMES`: closed-eye frame count to exit
- `COOLDOWN`: pause/play action cooldown
- `VOLUME_COOLDOWN`: volume action cooldown
- `EAR_SMOOTHING_WINDOW`: moving average window for EAR
- `HAND_PROCESS_EVERY_N_FRAMES`: hand inference interval for performance

## Installation
Use the project virtual environment (recommended):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional for global exit hotkey:

```powershell
pip install pynput
```

## Run
```powershell
.\venv\Scripts\python.exe .\focus_flow.py
```

## Controls Summary
- Close eyes briefly (stable) -> Pause
- Open eyes briefly (stable) -> Play
- Close eyes for ~3s -> Exit
- `Ctrl+Shift+X` -> Global exit (if `pynput` installed)
- `q` / `Esc` -> Exit from FocusFlow window

## Notes
- Media keys control the active media session, so behavior is consistent for browser and external players.
- If you see `MediaPipe version incompatibility`, run using the project `venv` Python.
