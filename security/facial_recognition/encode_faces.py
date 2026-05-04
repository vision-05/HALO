from collections import Counter

from face_db import build_face_database


def main() -> None:
    database = build_face_database()
    names = database["names"]

    if not names:
        print("No faces were encoded. Add images under known_faces/<person_name>/ and rerun.")
        return

    counts = Counter(names)
    print("\nEncoding complete.")
    print(f"Total encodings: {len(names)}")
    for person, count in sorted(counts.items()):
        print(f"- {person}: {count}")


if __name__ == "__main__":
    main()

