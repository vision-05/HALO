import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from pathlib import Path
from discovery.src.base_agent import BaseAgent
from loguru import logger
import asyncio
import json

BASE_DIR = Path(__file__).resolve().parent
model_path = str(BASE_DIR / "face_landmarker.task")

class SuspicionDetectionAgent(BaseAgent):
    def __init__(self, name="SuspicionDetection", role="Aggregator"):
        super().__init__(name=name, role=role)
        self.desc = "Detects solely if a suspicious person is outside and triggers message sent. Cannot provide more info. Do not query or poll the state, a message will be sent if a suspicious person is detected outside automatically. "

        # set up MediaPipe
        BaseOptions = python.BaseOptions
        self.FaceLandmarker = vision.FaceLandmarker
        FaceLandmarkerOptions = vision.FaceLandmarkerOptions
        VisionRunningMode = vision.RunningMode

        self.options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE
        )

        self.cap = cv2.VideoCapture("http://192.168.1.119:4747/video")

        # set up OpenCV
        # load the standard eye detector, which is built-in
        self.eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        mouth_cascade_path = str(BASE_DIR / 'haarcascade_mcs_mouth.xml')
        nose_cascade_path = str(BASE_DIR / 'haarcascade_mcs_nose.xml')

        # Load the mouth and nose detectors
        self.mouth_detector = cv2.CascadeClassifier(mouth_cascade_path)
        self.nose_detector = cv2.CascadeClassifier(nose_cascade_path)

        # center points from MediaPipe
        self.LEFT_EYE = 159
        self.RIGHT_EYE = 386
        self.NOSE = 1
        self.MOUTH = 14


    def get_dynamic_box_size(self, face, frame_width, frame_height, min_size=48, max_size=200):
            xs = [lm.x * frame_width for lm in face]
            ys = [lm.y * frame_height for lm in face]

            face_width = max(xs) - min(xs)
            face_height = max(ys) - min(ys)

            estimated_size = int(max(face_width, face_height) * 0.3)

            return max(min_size, min(max_size, estimated_size))


    def safe_crop(self, frame, center_x, center_y, box_size=80):
        """safely crops a square from the frame."""
        h, w, _ = frame.shape
        box_size = max(1, min(box_size, w, h))
        half = box_size // 2

        # build the crop at the requested size, then shift it back inside the frame
        # so the box does not shrink just because the face is near an edge.
        x1 = int(center_x - half)
        y1 = int(center_y - half)
        x2 = x1 + box_size
        y2 = y1 + box_size

        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x2 > w:
            shift = x2 - w
            x1 = max(0, x1 - shift)
            x2 = w
        if y2 > h:
            shift = y2 - h
            y1 = max(0, y1 - shift)
            y2 = h

        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


    def is_feature_visible(self, crop, detector, feature_type):
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

    async def loop(self):
        suspicion_start_time = None
        REQUIRED_TIME = 3.0
        msg_sent = False
        with self.FaceLandmarker.create_from_options(self.options) as landmarker:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
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
                    box_size = self.get_dynamic_box_size(face, w, h)

                    # map the coordinates
                    features = {
                        "Left Eye": (int(face[self.LEFT_EYE].x * w), int(face[self.LEFT_EYE].y * h), "eye"),
                        "Right Eye": (int(face[self.RIGHT_EYE].x * w), int(face[self.RIGHT_EYE].y * h), "eye"),
                        "Nose": (int(face[self.NOSE].x * w), int(face[self.NOSE].y * h), "nose"),
                        "Mouth": (int(face[self.MOUTH].x * w), int(face[self.MOUTH].y * h), "mouth")
                    }

                    # check each feature
                    for name, (cx, cy, f_type) in features.items():

                        # crop a box sized according to how close the face is to the camera
                        crop, (x1, y1, x2, y2) = self.safe_crop(frame, cx, cy, box_size=box_size)

                        # assign the correct detector
                        if f_type == "eye":
                            detector_to_use = self.eye_detector
                        elif f_type == "nose":
                            detector_to_use = self.nose_detector
                        else:
                            detector_to_use = self.mouth_detector

                        # verify with Cascade
                        if not self.is_feature_visible(crop, detector_to_use, f_type):
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
                        
                        logger.debug(f"{name}: {status}")

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
                        if msg_sent != True:
                            self.state["suspicious_person_outside"] = True
                            await self.send_msg("LLM", json.dumps({"action": "self_prompt", "source": self.name, "target": "Claude", "params": {"prompt": "Suspicious person detected outside by security camera. Take appropriate action to keep the house safe. "}})) #hardcoded remove this
                            msg_sent = True
                else:
                    # if face becomes visible again, instantly reset the timer to zero
                    suspicion_start_time = None
                    is_suspicious = False
                    self.state["suspicious_person_outside"] = False
                    msg_sent = False

                # final alert display
                if is_suspicious:
                    cv2.putText(frame, "SUSPICIOUS", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                #cv2.imshow("Cascade Verification", frame)

                #if cv2.waitKey(1) & 0xFF == ord('q'):
                #    break
                await asyncio.sleep(0.5)

        self.cap.release()
        cv2.destroyAllWindows()


async def main():
    a = SuspicionDetectionAgent()
    asyncio.create_task(a.loop())
    await a.run()

asyncio.run(main())