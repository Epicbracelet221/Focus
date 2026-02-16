import cv2
import time
import math
import numpy as np
import pyautogui
import mediapipe as mp
from collections import deque
from threading import Event

try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except Exception:
    PYNPUT_AVAILABLE = False

# CONFIGURATION
CAM_WIDTH, CAM_HEIGHT = 640, 480
EYE_CLOSED_THRESHOLD = 0.20
PAUSE_PLAY_STABILITY_FRAMES = 10
SHUTDOWN_EYE_CLOSED_FRAMES = 90
COOLDOWN = 1.0
VOLUME_COOLDOWN = 0.25
EAR_SMOOTHING_WINDOW = 5
HAND_PROCESS_EVERY_N_FRAMES = 2

# Mediapipe face mesh landmarks for EAR (Eye Aspect Ratio)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
EYE_DRAW_IDX = LEFT_EYE_IDX + RIGHT_EYE_IDX

# SETUP MEDIA PIPE
try:
    mp_face_mesh = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    print("Error: MediaPipe version incompatibility.")
    raise SystemExit(1)


def press_key_safe(key: str):
    try:
        pyautogui.press(key)
    except Exception as e:
        print(f"Key press failed for '{key}': {e}")


def get_point(face_landmarks, idx: int, width: int, height: int):
    lm = face_landmarks.landmark[idx]
    return np.array([lm.x * width, lm.y * height], dtype=np.float64)


def eye_aspect_ratio(face_landmarks, eye_indices, width: int, height: int):
    p1 = get_point(face_landmarks, eye_indices[0], width, height)
    p2 = get_point(face_landmarks, eye_indices[1], width, height)
    p3 = get_point(face_landmarks, eye_indices[2], width, height)
    p4 = get_point(face_landmarks, eye_indices[3], width, height)
    p5 = get_point(face_landmarks, eye_indices[4], width, height)
    p6 = get_point(face_landmarks, eye_indices[5], width, height)

    horizontal = np.linalg.norm(p1 - p4)
    if horizontal < 1e-6:
        return 1.0
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    return vertical / (2.0 * horizontal)


def start_global_exit_listener(exit_event: Event):
    if not PYNPUT_AVAILABLE:
        return None

    pressed = set()

    def normalize(key):
        if isinstance(key, keyboard.KeyCode) and key.char:
            return key.char.lower()
        return key

    def on_press(key):
        norm = normalize(key)
        pressed.add(norm)
        ctrl_pressed = keyboard.Key.ctrl_l in pressed or keyboard.Key.ctrl_r in pressed
        shift_pressed = keyboard.Key.shift_l in pressed or keyboard.Key.shift_r in pressed
        if ctrl_pressed and shift_pressed and "x" in pressed:
            exit_event.set()

    def on_release(key):
        norm = normalize(key)
        if norm in pressed:
            pressed.remove(norm)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.daemon = True
    listener.start()
    return listener


# MAIN EXECUTION
cap = cv2.VideoCapture(0)
cap.set(3, CAM_WIDTH)
cap.set(4, CAM_HEIGHT)
if not cap.isOpened():
    print("Error: Could not open camera.")
    raise SystemExit(1)

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

is_paused = False
last_state_change_time = 0.0
closed_eye_counter = 0
open_eye_counter = 0
last_volume_change_time = 0.0
should_exit = False
frame_count = 0
ear_history = deque(maxlen=EAR_SMOOTHING_WINDOW)
global_exit_event = Event()
global_exit_listener = start_global_exit_listener(global_exit_event)

print("FocusFlow is Active. Controls work on active media sessions (browser/VLC/players).")
if PYNPUT_AVAILABLE:
    print("Commands: close eyes to pause, open eyes to play, hold eyes closed ~3s to exit, or press Ctrl+Shift+X globally.")
else:
    print("Commands: close eyes to pause, open eyes to play, hold eyes closed ~3s to exit, or press 'q'/'Esc' on FocusFlow window.")
    print("Tip: Install 'pynput' for global exit hotkey (Ctrl+Shift+X).")

with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh, \
     mp_hands.Hands(model_complexity=0, max_num_hands=1) as hands:

    while cap.isOpened():
        if global_exit_event.is_set():
            should_exit = True

        success, image = cap.read()
        if not success:
            continue
        frame_count += 1

        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        face_results = face_mesh.process(image_rgb)
        hand_results = hands.process(image_rgb) if (frame_count % HAND_PROCESS_EVERY_N_FRAMES == 0) else None

        image.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        h, w, _ = image.shape

        # FEATURE 1: EYE TRACKING FOR PAUSE/PLAY + SHUTDOWN
        eye_status_text = "Face Not Found"
        eyes_closed = False
        avg_ear = 0.0

        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            left_ear = eye_aspect_ratio(face_landmarks, LEFT_EYE_IDX, w, h)
            right_ear = eye_aspect_ratio(face_landmarks, RIGHT_EYE_IDX, w, h)
            avg_ear = (left_ear + right_ear) / 2.0
            ear_history.append(avg_ear)
            smoothed_ear = sum(ear_history) / len(ear_history)

            eyes_closed = smoothed_ear < EYE_CLOSED_THRESHOLD
            eye_status_text = "Eyes Closed" if eyes_closed else "Eyes Open"

            for idx in EYE_DRAW_IDX:
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

            cv2.putText(image, f"EAR: {smoothed_ear:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if eyes_closed:
            closed_eye_counter += 1
            open_eye_counter = 0
        else:
            open_eye_counter += 1
            closed_eye_counter = 0

        if closed_eye_counter >= SHUTDOWN_EYE_CLOSED_FRAMES:
            print("Action: SHUTDOWN")
            should_exit = True

        current_time = time.time()
        if (current_time - last_state_change_time) > COOLDOWN and not should_exit:
            if closed_eye_counter > PAUSE_PLAY_STABILITY_FRAMES and not is_paused:
                press_key_safe("playpause")
                is_paused = True
                last_state_change_time = current_time
                print("Action: PAUSED")
            elif open_eye_counter > PAUSE_PLAY_STABILITY_FRAMES and is_paused:
                press_key_safe("playpause")
                is_paused = False
                last_state_change_time = current_time
                print("Action: PLAYING")

        # FEATURE 2: VOLUME CONTROL
        if hand_results and hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                thumb = hand_landmarks.landmark[4]
                index = hand_landmarks.landmark[8]
                x1, y1 = int(thumb.x * w), int(thumb.y * h)
                x2, y2 = int(index.x * w), int(index.y * h)

                cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 3)
                length = math.hypot(x2 - x1, y2 - y1)

                if length < 50 and (current_time - last_volume_change_time) > VOLUME_COOLDOWN:
                    cv2.putText(image, "VOL DOWN", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    press_key_safe("volumedown")
                    last_volume_change_time = current_time
                elif length > 180 and (current_time - last_volume_change_time) > VOLUME_COOLDOWN:
                    cv2.putText(image, "VOL UP", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    press_key_safe("volumeup")
                    last_volume_change_time = current_time
                else:
                    cv2.putText(image, "VOL STABLE", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Show Output
        cv2.putText(image, f"Eyes: {eye_status_text}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        cv2.putText(image, f"State: {'PAUSED' if is_paused else 'PLAYING'}", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        if closed_eye_counter > PAUSE_PLAY_STABILITY_FRAMES:
            shutdown_left = max(0, SHUTDOWN_EYE_CLOSED_FRAMES - closed_eye_counter)
            cv2.putText(image, f"Hold to exit: {shutdown_left}", (20, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

        cv2.imshow("FocusFlow", image)

        key = cv2.waitKey(5) & 0xFF
        if key == 27 or key == ord('q') or should_exit:
            break

cap.release()
cv2.destroyAllWindows()
if global_exit_listener is not None:
    global_exit_listener.stop()
