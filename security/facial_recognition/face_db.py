from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import face_recognition
import numpy as np
from PIL import Image

from config import ENCODINGS_PATH, KNOWN_FACES_DIR, SUPPORTED_EXTENSIONS


def _iter_person_images(known_faces_dir: Path) -> List[Tuple[str, Path]]:
    """Iterate through the known_faces directory to find valid images."""
    image_paths: List[Tuple[str, Path]] = []
    if not known_faces_dir.exists():
        return image_paths

    for person_dir in sorted(known_faces_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        person_name = person_dir.name
        for image_path in sorted(person_dir.iterdir()):
            if image_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                image_paths.append((person_name, image_path))

    return image_paths


def _ensure_dlib_compatible(image: np.ndarray) -> np.ndarray:
    """Return writable, C-contiguous uint8 RGB array for face_recognition/dlib."""
    if image.size == 0:
        raise ValueError("empty image")

    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            max_value = float(np.max(image))
            if max_value <= 1.0:
                image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3:
        channel_count = image.shape[2]
        if channel_count == 1:
            image = np.repeat(image, 3, axis=2)
        elif channel_count >= 3:
            image = image[:, :, :3]
        else:
            raise ValueError(f"unsupported channel count: {channel_count}")
    else:
        raise ValueError(f"unsupported shape: {image.shape}")

    if image.shape[2] != 3:
        raise ValueError(f"expected RGB image, got shape: {image.shape}")

    # Enforce C-contiguous memory layout for dlib C++ bindings
    return np.require(image, dtype=np.uint8, requirements=["C", "W", "O"])


def _load_image_rgb(image_path: Path) -> np.ndarray:
    """Load image robustly as RGB using Pillow, dropping alpha/gray channels."""
    try:
        with Image.open(image_path) as img:
            # Force conversion to 3-channel RGB to prevent alpha-channel crashes
            image_array = np.array(img.convert("RGB"), dtype=np.uint8, copy=True)
        return _ensure_dlib_compatible(image_array)
    except Exception:
        # Fallback to OpenCV if Pillow fails
        bgr_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr_image is None:
            raise ValueError(f"failed to decode image: {image_path}")
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        return _ensure_dlib_compatible(rgb_image)


def _largest_face(face_locations: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    """Select the largest bounding box from detected faces."""
    return max(
        face_locations,
        key=lambda box: (box[2] - box[0]) * (box[1] - box[3]),
    )


def build_face_database(
        known_faces_dir: Path = KNOWN_FACES_DIR,
        output_path: Path = ENCODINGS_PATH,
) -> Dict[str, List[np.ndarray]]:
    """Scan the directory, detect faces, encode them, and save to a pickle file."""
    names: List[str] = []
    encodings: List[np.ndarray] = []

    for person_name, image_path in _iter_person_images(known_faces_dir):
        try:
            image = _load_image_rgb(image_path)
            face_locations = face_recognition.face_locations(image, model="hog")
        except (ValueError, RuntimeError) as error:
            print(f"[WARN] Unsupported image in {image_path.name}: {error}; skipping")
            continue

        if not face_locations:
            print(f"[WARN] No face found in {image_path.name}; skipping")
            continue

        selected_location = [_largest_face(face_locations)]

        try:
            face_enc = face_recognition.face_encodings(image, known_face_locations=selected_location)
        except RuntimeError as error:
            print(f"[WARN] Could not encode face in {image_path.name}: {error}; skipping")
            continue

        if not face_enc:
            print(f"[WARN] Could not encode face in {image_path.name}; skipping")
            continue

        names.append(person_name)
        encodings.append(face_enc[0])

    database = {"names": names, "encodings": encodings}

    # Ensure the parent directory for the output path exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Safely dump the database
    with output_path.open("wb") as file:
        pickle.dump(database, file)

    return database


def load_face_database(path: Path = ENCODINGS_PATH) -> Dict[str, List[np.ndarray]]:
    """Load the pre-computed face encodings from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"Encoding file not found: {path}. Run encode_faces.py first."
        )

    with path.open("rb") as file:
        database = pickle.load(file)

    if "names" not in database or "encodings" not in database:
        raise ValueError("Invalid encoding database format.")

    return database