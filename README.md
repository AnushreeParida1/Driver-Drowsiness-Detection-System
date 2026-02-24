This project is a Real-Time Driver Drowsiness Detection System built using Python and Computer Vision.

The system monitors a driver through a webcam and detects signs of fatigue such as:
  Prolonged eye closure
  Head dropping / face disappearing (possible fainting or sleeping)

When repeated fatigue events occur within a short time window, the system triggers an audio alarm to alert the driver and prevent accidents.

🧠 How It Works
1️⃣ Eye Detection (EAR Method)
  Uses facial landmarks from dlib.
  Calculates Eye Aspect Ratio (EAR).
If EAR stays below a threshold for multiple frames → counts as a drowsy event.

2️⃣ Head-Down Detection
  If the face disappears for several frames (due to head dropping forward), it is treated as a fatigue event.
  This helps detect fainting or nodding off.

3️⃣ Smart 2-Minute Validation
  Fatigue events are stored in a 2-minute sliding window.
  Alarm triggers only if multiple events occur within those 2 minutes.
  Prevents false alarms from isolated blinks.
