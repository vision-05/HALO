from __future__ import annotations

import cv2
import face_recognition
import numpy as np

from config import CAMERA_INDEX, FRAME_RESIZE, TOLERANCE
from face_db import load_face_database


def _best_match_name(known_names, known_encodings, face_encoding) -> tuple[str, float]:
    distances = face_recognition.face_distance(known_encodings, face_encoding)
    best_index = int(np.argmin(distances))
    best_distance = float(distances[best_index])
    if best_distance <= TOLERANCE:
        return known_names[best_index], best_distance
    return "UNKNOWN", best_distance


def main() -> None:
    database = load_face_database()
    known_names = database["names"]
    known_encodings = database["encodings"]

    if not known_encodings:
        raise RuntimeError("Encoding database is empty. Run encode_faces.py with valid images.")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Live recognition started. Press 'q' to quit.")

    process_this_frame = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        if process_this_frame:
            small_frame = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE, fy=FRAME_RESIZE)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

            labels = []
            for face_encoding in face_encodings:
                name, distance = _best_match_name(known_names, known_encodings, face_encoding)
                labels.append(f"{name} ({distance:.2f})")
        process_this_frame = not process_this_frame

        if 'face_locations' in locals() and 'labels' in locals():
            for (top, right, bottom, left), label in zip(face_locations, labels):
                top = int(top / FRAME_RESIZE)
                right = int(right / FRAME_RESIZE)
                bottom = int(bottom / FRAME_RESIZE)
                left = int(left / FRAME_RESIZE)

                color = (0, 255, 0) if not label.startswith("UNKNOWN") else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 28), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, label, (left + 6, bottom - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Household Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

