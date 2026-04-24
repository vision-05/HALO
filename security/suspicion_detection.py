import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

model_path = "face_landmarker.task"

# set up MediaPipe
BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE
)

cap = cv2.VideoCapture(0)

# set up OpenCV
# load the standard eye detector, which is built-in
eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
mouth_cascade_path = 'haarcascade_mcs_mouth.xml'
nose_cascade_path = 'haarcascade_mcs_nose.xml'

# Load the mouth and nose detectors
mouth_detector = cv2.CascadeClassifier(mouth_cascade_path)
nose_detector = cv2.CascadeClassifier(nose_cascade_path)

# center points from MediaPipe
LEFT_EYE = 159
RIGHT_EYE = 386
NOSE = 1
MOUTH = 14


def safe_crop(frame, center_x, center_y, box_size=80):
    """safely crops a square from the frame."""
    h, w, _ = frame.shape
    half = box_size // 2

    # ensures square is not outside the video frame
    y1 = max(0, center_y - half)
    y2 = min(h, center_y + half)
    x1 = max(0, center_x - half)
    x2 = min(w, center_x + half)

    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def is_feature_visible(crop, detector, feature_type):
    """uses a cascade classifier to verify if the feature actually exists in the crop."""
    if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
        return False

    # turns the cropped image into grayscale as Haar Cascades only works in black and white
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # run the specific detector
    if feature_type == "eye":
        detections = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    elif feature_type == "mouth":
        detections = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    elif feature_type == "nose":
        detections = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    else:
        return False

    # if the list is greater than 0, it found the feature
    return len(detections) > 0


suspicion_start_time = None
REQUIRED_TIME = 3.0

with FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # flips camera so that it is mirrored
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        covered_features = []
        is_suspicious = False
        trigger_timer = False

        if result.face_landmarks:
            face = result.face_landmarks[0]

            # map the coordinates
            features = {
                "Left Eye": (int(face[LEFT_EYE].x * w), int(face[LEFT_EYE].y * h), "eye"),
                "Right Eye": (int(face[RIGHT_EYE].x * w), int(face[RIGHT_EYE].y * h), "eye"),
                "Nose": (int(face[NOSE].x * w), int(face[NOSE].y * h), "nose"),
                "Mouth": (int(face[MOUTH].x * w), int(face[MOUTH].y * h), "mouth")
            }

            # check each feature
            for name, (cx, cy, f_type) in features.items():

                # crop an 80x80 box
                crop, (x1, y1, x2, y2) = safe_crop(frame, cx, cy, box_size=80)

                # assign the correct detector
                if f_type == "eye":
                    detector_to_use = eye_detector
                elif f_type == "nose":
                    detector_to_use = nose_detector
                else:
                    detector_to_use = mouth_detector

                # verify with Cascade
                if not is_feature_visible(crop, detector_to_use, f_type):
                    covered_features.append(name)
                    box_color = (0, 0, 255)  # red = cascade couldn't find it
                    status = "COVERED"
                else:
                    box_color = (0, 255, 0)  # green = cascade found it
                    status = "VISIBLE"

                # draw box and text
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, f"{name}: {status}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)

            # suspect if 2 or more features are covered
            if len(covered_features) >= 2:
                trigger_timer = True # starts the timer
                cv2.putText(frame, f"Covered: {', '.join(covered_features)}",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        else:
            # face entirely lost
            trigger_timer = True
            cv2.putText(frame, "No Face Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if trigger_timer:
            if suspicion_start_time is None:
                suspicion_start_time = time.time()

            # calculate how many seconds have passed since we started recording
            elapsed_time = time.time() - suspicion_start_time

            # draw the timer on the screen
            cv2.putText(frame, f"Timer: {elapsed_time:.1f}s / {REQUIRED_TIME}s",
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            # if 3 seconds have passed, trigger the master alarm
            if elapsed_time >= REQUIRED_TIME:
                is_suspicious = True
        else:
            # if face becomes visible again, instantly reset the timer to zero
            suspicion_start_time = None
            is_suspicious = False

        # final alert display
        if is_suspicious:
            cv2.putText(frame, "SUSPICIOUS", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.imshow("Cascade Verification", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()