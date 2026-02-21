import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance

# -------------------------
# EAR calculation function
# -------------------------
def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# -------------------------
# Thresholds
# -------------------------
EAR_THRESHOLD = 0.25
FRAME_THRESHOLD = 20
counter = 0

# -------------------------
# Initialize dlib
# -------------------------
print("Loading predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# -------------------------
# Start webcam
# -------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not accessible")
    exit()

print("Camera started...")

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Error: Failed to grab frame")
        break

    # Convert BGR to RGB (dlib prefers RGB)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = detector(rgb)

    for face in faces:
        shape = predictor(rgb, face)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = calculate_EAR(leftEye)
        rightEAR = calculate_EAR(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Draw eye contours
        leftHull = cv2.convexHull(leftEye)
        rightHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightHull], -1, (0, 255, 0), 1)

        # Display EAR value
        cv2.putText(frame, f"EAR: {ear:.2f}",
                    (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2)

        # Drowsiness detection
        if ear < EAR_THRESHOLD:
            counter += 1
            if counter >= FRAME_THRESHOLD:
                cv2.putText(frame, "DROWSINESS ALERT!",
                            (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            (0, 0, 255),
                            3)
        else:
            counter = 0

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()