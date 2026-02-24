import cv2
import dlib
import numpy as np
import time
from imutils import face_utils
from scipy.spatial import distance
import pygame
import os

# ============================================================
# WHY FACE DISAPPEARANCE = HEAD DOWN:
# dlib's frontal face detector is trained on upright faces.
# When a driver faints or droops their head forward/down,
# the face rotates out of the frontal plane and dlib LOSES it.
# We treat "face was visible but suddenly vanished for N frames"
# as a reliable head-down / fainting signal.
# ============================================================

# -------------------------
# EAR calculation
# -------------------------
def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# -------------------------
# Fatigue event log — 2-minute sliding window
# -------------------------
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

# -------------------------
# Alarm helpers (pygame)
# -------------------------
ALARM_FILE = "Alarm.WAV"
pygame.mixer.init()

def play_alarm():
    if not pygame.mixer.music.get_busy():
        if os.path.exists(ALARM_FILE):
            pygame.mixer.music.load(ALARM_FILE)
            pygame.mixer.music.play()
        else:
            print(f"WARNING: {ALARM_FILE} not found — alarm skipped.")

def stop_alarm():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()

# -------------------------
# Load dlib
# -------------------------
print("Loading face predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# -------------------------
# Webcam
# -------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not accessible")
    exit()

# -------------------------
# Calibration (5 seconds — eyes open, head upright)
# -------------------------
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
        leftEAR  = calculate_EAR(shape[lStart:lEnd])
        rightEAR = calculate_EAR(shape[rStart:rEnd])
        calibration_ear.append((leftEAR + rightEAR) / 2.0)

    remaining = max(0, int(5 - (time.time() - cal_start)) + 1)
    cv2.putText(frame, f"Calibrating... {remaining}s  (sit upright, eyes open)",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.imshow("Calibration", frame)
    cv2.waitKey(1)

cv2.destroyWindow("Calibration")

baseline_EAR  = np.mean(calibration_ear) if calibration_ear else 0.30
EAR_THRESHOLD = baseline_EAR * 0.85

# -------------------------
# Thresholds & counters
# -------------------------
# Consecutive low-EAR frames before confirming eye-drowsiness
EYE_FRAME_THRESHOLD = 25    # ~0.8 s at 30 fps

# Consecutive MISSING-FACE frames before confirming head-down.
# 20 frames ≈ 0.67 s — ignores brief detection blips but catches real droops.
HEAD_FRAME_THRESHOLD = 20

ALARM_EVENT_COUNT = 5       # events within 2 min to trigger alarm

fatigue_log = FatigueEventLog()

eye_counter       = 0
eye_event_active  = False

face_missing_counter = 0
head_event_active    = False
face_was_seen        = False  # prevents false alarm before first detection

alarm_triggered = False

print(f"Baseline EAR: {baseline_EAR:.3f}  |  EAR Threshold: {EAR_THRESHOLD:.3f}")
print("Detection started. Press ESC to quit.\n")

# -------------------------
# Detection loop
# -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector(rgb)

    # ══════════════════════════════════════════════
    # BRANCH A — Face detected normally
    # ══════════════════════════════════════════════
    if faces:
        face_was_seen = True

        # Face just came back after being absent —
        # if it was gone long enough, log a head-down event
        if face_missing_counter >= HEAD_FRAME_THRESHOLD and head_event_active:
            fatigue_log.add_event("head-down")
            print(f"[{time.strftime('%H:%M:%S')}] HEAD-DOWN event logged "
                  f"(face absent {face_missing_counter} frames). "
                  f"Events in 2 min: {fatigue_log.count_recent()}")

        face_missing_counter = 0
        head_event_active    = False
        # If alarm was triggered by sustained head-down, stop it now that face is back
        if alarm_triggered and fatigue_log.count_recent() < ALARM_EVENT_COUNT:
            stop_alarm()
            alarm_triggered = False

        for face in faces:
            shape = predictor(rgb, face)
            shape = face_utils.shape_to_np(shape)

            leftEye  = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            ear = (calculate_EAR(leftEye) + calculate_EAR(rightEye)) / 2.0

            # Draw eye contours
            cv2.drawContours(frame, [cv2.convexHull(leftEye)],  -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

            # ---- EAR / eye-drowsiness check ----
            if ear < EAR_THRESHOLD:
                eye_counter += 1
            else:
                # Eyes reopened — log event if drowsiness was sustained
                if eye_counter >= EYE_FRAME_THRESHOLD and eye_event_active:
                    fatigue_log.add_event("eye-drowsy")
                    print(f"[{time.strftime('%H:%M:%S')}] EYE-DROWSY event logged. "
                          f"Events in 2 min: {fatigue_log.count_recent()}")
                eye_counter      = 0
                eye_event_active = False

            if eye_counter >= EYE_FRAME_THRESHOLD:
                eye_event_active = True

            # ---- HUD ----
            cv2.putText(frame, f"EAR: {ear:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Events(2min): {fatigue_log.count_recent()}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 255), 2)

            if eye_counter >= EYE_FRAME_THRESHOLD:
                cv2.putText(frame, "EYES CLOSED!",
                            (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)

    # ══════════════════════════════════════════════
    # BRANCH B — Face NOT detected
    # ══════════════════════════════════════════════
    else:
        if face_was_seen:
            # Count how long the face has been missing
            face_missing_counter += 1
            head_event_active     = True

            secs_missing = face_missing_counter / 30.0   # approx at ~30 fps
            cv2.putText(frame,
                        f"HEAD DOWN / FACE LOST  ({secs_missing:.1f}s)",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"Events(2min): {fatigue_log.count_recent()}",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 255), 2)

            # Head down 3+ seconds → alarm immediately, no event count needed
            if secs_missing >= 1.0:
                cv2.putText(frame, "** DROWSINESS ALARM **",
                            (10, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
                if not alarm_triggered:
                    play_alarm()
                    alarm_triggered = True
                    print(f"[{time.strftime('%H:%M:%S')}] *** ALARM TRIGGERED (head down {secs_missing:.1f}s) ***")
        else:
            # Haven't seen a face yet at all — just waiting
            cv2.putText(frame, "Waiting for face...",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

    # ══════════════════════════════════════════════
    # ALARM — fires when 2+ events occur within 2 min
    # ══════════════════════════════════════════════
    if fatigue_log.count_recent() >= ALARM_EVENT_COUNT:
        cv2.putText(frame, "** DROWSINESS ALARM **",
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
        if not alarm_triggered:
            play_alarm()
            alarm_triggered = True
            print(f"[{time.strftime('%H:%M:%S')}] *** ALARM TRIGGERED ***")
    else:
        if alarm_triggered:
            stop_alarm()
            alarm_triggered = False

    cv2.imshow("Driver Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
stop_alarm()
pygame.mixer.quit()
# import cv2
# import dlib
# import numpy as np
# import time
# from imutils import face_utils
# from scipy.spatial import distance
# import pygame
# import os

# # ============================================================
# # WHY FACE DISAPPEARANCE = HEAD DOWN:
# # dlib's frontal face detector is trained on upright faces.
# # When a driver faints or droops their head forward/down,
# # the face rotates out of the frontal plane and dlib LOSES it.
# # We treat "face was visible but suddenly vanished for N frames"
# # as a reliable head-down / fainting signal.
# # ============================================================

# # -------------------------
# # EAR calculation
# # -------------------------
# def calculate_EAR(eye):
#     A = distance.euclidean(eye[1], eye[5])
#     B = distance.euclidean(eye[2], eye[4])
#     C = distance.euclidean(eye[0], eye[3])
#     return (A + B) / (2.0 * C)

# # -------------------------
# # Fatigue event log — 2-minute sliding window
# # -------------------------
# class FatigueEventLog:
#     WINDOW_SECONDS = 120

#     def __init__(self):
#         self.events = []

#     def add_event(self, label=""):
#         self.events.append((time.time(), label))
#         self._purge()

#     def _purge(self):
#         cutoff = time.time() - self.WINDOW_SECONDS
#         self.events = [(t, l) for t, l in self.events if t >= cutoff]

#     def count_recent(self):
#         self._purge()
#         return len(self.events)

# # -------------------------
# # Alarm helpers (pygame)
# # -------------------------
# ALARM_FILE = "Alarm.WAV"
# pygame.mixer.init()

# def play_alarm():
#     if not pygame.mixer.music.get_busy():
#         if os.path.exists(ALARM_FILE):
#             pygame.mixer.music.load(ALARM_FILE)
#             pygame.mixer.music.play()
#         else:
#             print(f"WARNING: {ALARM_FILE} not found — alarm skipped.")

# def stop_alarm():
#     if pygame.mixer.music.get_busy():
#         pygame.mixer.music.stop()

# # -------------------------
# # Load dlib
# # -------------------------
# print("Loading face predictor...")
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
# (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# # -------------------------
# # Webcam
# # -------------------------
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Camera not accessible")
#     exit()

# # -------------------------
# # Calibration (5 seconds — eyes open, head upright)
# # -------------------------
# print("Calibration: Sit upright, eyes OPEN for 5 seconds...")
# calibration_ear = []
# cal_start = time.time()

# while time.time() - cal_start < 5:
#     ret, frame = cap.read()
#     if not ret:
#         continue

#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     faces = detector(rgb)

#     for face in faces:
#         shape = predictor(rgb, face)
#         shape = face_utils.shape_to_np(shape)
#         leftEAR  = calculate_EAR(shape[lStart:lEnd])
#         rightEAR = calculate_EAR(shape[rStart:rEnd])
#         calibration_ear.append((leftEAR + rightEAR) / 2.0)

#     remaining = max(0, int(5 - (time.time() - cal_start)) + 1)
#     cv2.putText(frame, f"Calibrating... {remaining}s  (sit upright, eyes open)",
#                 (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
#     cv2.imshow("Calibration", frame)
#     cv2.waitKey(1)

# cv2.destroyWindow("Calibration")

# baseline_EAR  = np.mean(calibration_ear) if calibration_ear else 0.30
# EAR_THRESHOLD = baseline_EAR * 0.85

# # -------------------------
# # Thresholds & counters
# # -------------------------
# # Consecutive low-EAR frames before confirming eye-drowsiness
# EYE_FRAME_THRESHOLD = 12    # ~0.4 s at 30 fps

# # Consecutive MISSING-FACE frames before confirming head-down.
# # 20 frames ≈ 0.67 s — ignores brief detection blips but catches real droops.
# HEAD_FRAME_THRESHOLD = 20

# ALARM_EVENT_COUNT = 4      # events within 2 min to trigger alarm

# fatigue_log = FatigueEventLog()

# eye_counter       = 0
# eye_event_active  = False

# face_missing_counter = 0
# head_event_active    = False
# face_was_seen        = False  # prevents false alarm before first detection

# alarm_triggered = False

# print(f"Baseline EAR: {baseline_EAR:.3f}  |  EAR Threshold: {EAR_THRESHOLD:.3f}")
# print("Detection started. Press ESC to quit.\n")

# # -------------------------
# # Detection loop
# # -------------------------
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     faces = detector(rgb)

#     # ══════════════════════════════════════════════
#     # BRANCH A — Face detected normally
#     # ══════════════════════════════════════════════
#     if faces:
#         face_was_seen = True

#         # Face just came back after being absent —
#         # if it was gone long enough, log a head-down event
#         if face_missing_counter >= HEAD_FRAME_THRESHOLD and head_event_active:
#             fatigue_log.add_event("head-down")
#             print(f"[{time.strftime('%H:%M:%S')}] HEAD-DOWN event logged "
#                   f"(face absent {face_missing_counter} frames). "
#                   f"Events in 2 min: {fatigue_log.count_recent()}")

#         face_missing_counter = 0
#         head_event_active    = False

#         for face in faces:
#             shape = predictor(rgb, face)
#             shape = face_utils.shape_to_np(shape)

#             leftEye  = shape[lStart:lEnd]
#             rightEye = shape[rStart:rEnd]
#             ear = (calculate_EAR(leftEye) + calculate_EAR(rightEye)) / 2.0

#             # Draw eye contours
#             cv2.drawContours(frame, [cv2.convexHull(leftEye)],  -1, (0, 255, 0), 1)
#             cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

#             # ---- EAR / eye-drowsiness check ----
#             if ear < EAR_THRESHOLD:
#                 eye_counter += 1
#             else:
#                 # Eyes reopened — log event if drowsiness was sustained
#                 if eye_counter >= EYE_FRAME_THRESHOLD and eye_event_active:
#                     fatigue_log.add_event("eye-drowsy")
#                     print(f"[{time.strftime('%H:%M:%S')}] EYE-DROWSY event logged. "
#                           f"Events in 2 min: {fatigue_log.count_recent()}")
#                 eye_counter      = 0
#                 eye_event_active = False

#             if eye_counter >= EYE_FRAME_THRESHOLD:
#                 eye_event_active = True

#             # ---- HUD ----
#             cv2.putText(frame, f"EAR: {ear:.2f}",
#                         (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#             cv2.putText(frame, f"Events(2min): {fatigue_log.count_recent()}",
#                         (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 255), 2)

#             if eye_counter >= EYE_FRAME_THRESHOLD:
#                 cv2.putText(frame, "EYES CLOSED!",
#                             (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)

#     # ══════════════════════════════════════════════
#     # BRANCH B — Face NOT detected
#     # ══════════════════════════════════════════════
#     else:
#         if face_was_seen:
#             # Count how long the face has been missing
#             face_missing_counter += 1
#             head_event_active     = True

#             secs_missing = face_missing_counter / 30.0   # approx at ~30 fps
#             cv2.putText(frame,
#                         f"HEAD DOWN / FACE LOST  ({secs_missing:.1f}s)",
#                         (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
#             cv2.putText(frame, f"Events(2min): {fatigue_log.count_recent()}",
#                         (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 255), 2)
#         else:
#             # Haven't seen a face yet at all — just waiting
#             cv2.putText(frame, "Waiting for face...",
#                         (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

#     # ══════════════════════════════════════════════
#     # ALARM — fires when 2+ events occur within 2 min
#     # ══════════════════════════════════════════════
#     if fatigue_log.count_recent() >= ALARM_EVENT_COUNT:
#         cv2.putText(frame, "** DROWSINESS ALARM **",
#                     (10, frame.shape[0] - 20),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
#         if not alarm_triggered:
#             play_alarm()
#             alarm_triggered = True
#             print(f"[{time.strftime('%H:%M:%S')}] *** ALARM TRIGGERED ***")
#     else:
#         if alarm_triggered:
#             stop_alarm()
#             alarm_triggered = False

#     cv2.imshow("Driver Drowsiness Detection", frame)
#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()
# stop_alarm()
# pygame.mixer.quit()
