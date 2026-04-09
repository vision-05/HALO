import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = "face_landmarker.task"

# Specific landmark indices for each feature
LEFT_EYE = [33, 133, 160, 159, 158, 157, 173]
RIGHT_EYE = [362, 263, 387, 386, 385, 384, 398]
MOUTH = [13, 14, 78, 308]
NOSE = [1, 2, 4, 5, 195, 197]  # Added nose points (tip and bridge)

BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE
)

cap = cv2.VideoCapture(0)


# ---------- Helper Function ----------
def get_visibility_score(face, indices):
    """Calculates the average visibility for a specific group of facial landmarks."""
    visibilities = []

    for i in indices:
        # Fetch the visibility value
        val = getattr(face[i], 'visibility', None)

        # If MediaPipe explicitly returns None, default to 1.0 to prevent math crashes
        if val is None:
            val = 1.0

        visibilities.append(val)

    if not visibilities:
        return 1.0

    return sum(visibilities) / len(visibilities)


# --------------------------------------

with FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        # 👆 RAISED THRESHOLD: Now requires 75% confidence to be considered "visible"
        VISIBILITY_THRESHOLD = 0.75

        covered_features = []
        is_suspicious = False

        if result.face_landmarks:
            face = result.face_landmarks[0]

            # Get exact scores
            left_eye_vis = get_visibility_score(face, LEFT_EYE)
            right_eye_vis = get_visibility_score(face, RIGHT_EYE)
            mouth_vis = get_visibility_score(face, MOUTH)
            nose_vis = get_visibility_score(face, NOSE)

            # Check against threshold
            if left_eye_vis < VISIBILITY_THRESHOLD: covered_features.append("Left Eye")
            if right_eye_vis < VISIBILITY_THRESHOLD: covered_features.append("Right Eye")
            if mouth_vis < VISIBILITY_THRESHOLD: covered_features.append("Mouth")
            if nose_vis < VISIBILITY_THRESHOLD: covered_features.append("Nose")

            # --- DEBUG HUD: See exactly what the AI is scoring ---
            cv2.putText(frame, f"L-Eye: {left_eye_vis:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                        1)
            cv2.putText(frame, f"R-Eye: {right_eye_vis:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                        1)
            cv2.putText(frame, f"Mouth: {mouth_vis:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Nose:  {nose_vis:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Draw the points
            for lm in face:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            # If 2 or more features drop below the threshold
            if len(covered_features) >= 2:
                is_suspicious = True
                cv2.putText(frame, f"Covered: {', '.join(covered_features)}",
                            (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        else:
            # If the face vanishes entirely because too much is covered
            is_suspicious = True
            cv2.putText(frame, "No Face Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # -------- FINAL DECISION --------
        if is_suspicious:
            cv2.putText(frame, "🚨 SUSPICIOUS", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.imshow("Occlusion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()