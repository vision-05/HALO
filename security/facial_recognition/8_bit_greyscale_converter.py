from pathlib import Path
from typing import Optional

from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_IMAGE = BASE_DIR / "known_faces" / "lakshaya" / "lakshaya1.jpg"


def convert_to_8bit_grayscale(input_path: Path, output_path: Optional[Path] = None) -> Path:
    """Convert an image to 8-bit grayscale (Pillow mode 'L')."""
    resolved_input = input_path.expanduser().resolve()
    if not resolved_input.exists():
        raise FileNotFoundError(f"Input file not found: {resolved_input}")
    if not resolved_input.is_file():
        raise ValueError(f"Input path is not a file: {resolved_input}")

    if output_path is None:
        output_path = resolved_input.with_name(f"{resolved_input.stem}_grayscale.jpg")

    resolved_output = output_path.expanduser().resolve()
    resolved_output.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(resolved_input) as image:
        gray_image = image.convert("L")
        gray_image.save(resolved_output)

    return resolved_output


def main() -> int:
    input_path = DEFAULT_INPUT_IMAGE
    output_path = None

    try:
        saved_path = convert_to_8bit_grayscale(input_path, output_path)
        print(f"Grayscale image saved: {saved_path}")
        return 0
    except FileNotFoundError as error:
        print(error)
        return 1
    except Exception as error:
        print(f"Error converting image: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

