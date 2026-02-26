import cv2
import dlib
import numpy as np
import time
from imutils import face_utils
from scipy.spatial import distance
import pygame
import os

# ============================================================
# EAR calculation
# ============================================================
def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# ============================================================
# Fatigue event log (2-minute sliding window)
# ============================================================
class FatigueEventLog:
    WINDOW_SECONDS = 120

    def __init__(self):
        self.events = []

    def add_event(self, label=""):
        self.events.append((time.time(), label))
        self._purge()

    def _purge(self):
        cutoff = time.time() - self.WINDOW_SECONDS
        self.events = [(t, l) for t, l in self.events if t >= cutoff]

    def count_recent(self):
        self._purge()
        return len(self.events)

# ============================================================
# ROBUST ALARM SYSTEM
# ============================================================
ALARM_FILE = "Alarm.wav"   

pygame.mixer.init()
alarm_sound = None

if os.path.exists(ALARM_FILE):
    alarm_sound = pygame.mixer.Sound(ALARM_FILE)
    alarm_sound.set_volume(1.0)
else:
    print(f"WARNING: {ALARM_FILE} not found!")

def play_alarm():
    global alarm_sound
    if alarm_sound is not None:
        if not pygame.mixer.get_busy():
            alarm_sound.play(loops=-1) 

def stop_alarm():
    pygame.mixer.stop()

# ============================================================
# Load dlib
# ============================================================
print("Loading face predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# ============================================================
# Webcam
# ============================================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not accessible")
    exit()

# ============================================================
# Calibration (5 seconds)
# ============================================================
print("Calibration: Sit upright, eyes OPEN for 5 seconds...")
calibration_ear = []
cal_start = time.time()

while time.time() - cal_start < 5:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector(rgb)

    for face in faces:
        shape = predictor(rgb, face)
        shape = face_utils.shape_to_np(shape)
        leftEAR = calculate_EAR(shape[lStart:lEnd])
        rightEAR = calculate_EAR(shape[rStart:rEnd])
        calibration_ear.append((leftEAR + rightEAR) / 2.0)

    remaining = max(0, int(5 - (time.time() - cal_start)) + 1)
    cv2.putText(frame, f"Calibrating... {remaining}s (Eyes Open)",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.imshow("Calibration", frame)
    cv2.waitKey(1)

cv2.destroyWindow("Calibration")

baseline_EAR = np.mean(calibration_ear) if calibration_ear else 0.30
EAR_THRESHOLD = baseline_EAR * 0.85

# ============================================================
# Thresholds & Counters
# ============================================================
EYE_FRAME_THRESHOLD = 25    # Threshold to log an "event"
ALARM_EVENT_COUNT = 5       # Max events in 2 mins before persistent alarm

fatigue_log = FatigueEventLog()
eye_counter = 0
face_missing_counter = 0
face_was_seen = False
alarm_active = False

print(f"Baseline EAR: {baseline_EAR:.3f} | Threshold: {EAR_THRESHOLD:.3f}")
print("Detection started. Press ESC to quit.\n")

# ============================================================
# Detection Loop
# ============================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector(rgb)
    
    # Track alarm conditions for this frame
    current_alarm_condition = False
    recent_events = fatigue_log.count_recent()

    # --------------------------------------------------------
    # FACE DETECTED
    # --------------------------------------------------------
    if faces:
        # Check if face just returned after being missing (Head Down event)
        if face_was_seen and face_missing_counter >= 20:
            fatigue_log.add_event("head-down")
            print(f"[{time.strftime('%H:%M:%S')}] HEAD-DOWN event logged.")
            
        face_was_seen = True
        face_missing_counter = 0

        for face in faces:
            shape = predictor(rgb, face)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            ear = (calculate_EAR(leftEye) + calculate_EAR(rightEye)) / 2.0

            cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0,255,0), 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0,255,0), 1)

            if ear < EAR_THRESHOLD:
                eye_counter += 1
            else:
                # If eyes just opened after a long closure, log an event
                if eye_counter >= EYE_FRAME_THRESHOLD:
                    fatigue_log.add_event("eye-drowsy")
                    print(f"[{time.strftime('%H:%M:%S')}] EYE-DROWSY event logged.")
                eye_counter = 0

            eyes_closed_secs = eye_counter / 30.0

            # HUD: Eyes Closed Warning (Orange)
            if eye_counter >= EYE_FRAME_THRESHOLD:
                cv2.putText(frame, f"EYES CLOSED! ({eyes_closed_secs:.1f}s)",
                            (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)

            # Immediate Alarm Condition: Eyes closed 2+ seconds
            if eyes_closed_secs >= 2.0:
                current_alarm_condition = True
                cv2.putText(frame, "** DROWSINESS ALARM (EYES) **",
                            (10, frame.shape[0]-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

            cv2.putText(frame, f"EAR: {ear:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # --------------------------------------------------------
    # FACE NOT DETECTED (HEAD DOWN)
    # --------------------------------------------------------
    else:
        if face_was_seen:
            face_missing_counter += 1
            secs_missing = face_missing_counter / 30.0

            cv2.putText(frame, f"HEAD DOWN / FACE LOST ({secs_missing:.1f}s)",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            if secs_missing >= 2.0:
                current_alarm_condition = True
                cv2.putText(frame, "** DROWSINESS ALARM (HEAD) **",
                            (10, frame.shape[0]-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
        else:
            cv2.putText(frame, "Waiting for face...",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,100,100), 2)

    # --------------------------------------------------------
    # AGGREGATE EVENT ALARM
    # --------------------------------------------------------
    if recent_events >= ALARM_EVENT_COUNT:
        current_alarm_condition = True
        cv2.putText(frame, "** HIGH FATIGUE DETECTED **",
                    (10, frame.shape[0]-60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

    # Display Event Count HUD
    cv2.putText(frame, f"Events(2min): {recent_events}",
                (10, 60) if faces else (10, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 255), 2)

    # --------------------------------------------------------
    # ALARM CONTROL
    # --------------------------------------------------------
    if current_alarm_condition:
        if not alarm_active:
            play_alarm()
            alarm_active = True
    else:
        if alarm_active:
            stop_alarm()
            alarm_active = False

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
stop_alarm()
pygame.mixer.quit()