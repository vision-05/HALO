from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
KNOWN_FACES_DIR = BASE_DIR / "known_faces"
ENCODINGS_PATH = BASE_DIR / "encodings.pkl"

CAMERA_INDEX = 0
TOLERANCE = 0.47
FRAME_RESIZE = 0.25
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

