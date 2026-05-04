# Facial Recognition (Household Users)

This module lets you:
1. Encode known household members from labeled photos.
2. Run live webcam recognition to classify faces as known users or `UNKNOWN`.

## Folder layout

Create this structure before encoding:

```
known_faces/
  Alice/
    alice1.jpg
    alice2.jpg
  Bob/
    bob1.jpg
```

## Install

```bash
python3 -m pip install face-recognition opencv-python numpy
```

## Encode known users

```bash
python3 encode_faces.py
```

This creates `encodings.pkl` in the same folder.


## Run live recognition

```bash
python3 live_recognition.py
```

- Green box: known household user.
- Red box: unknown person.
- Press `q` to quit.

## Notes

- Matching threshold is controlled by `TOLERANCE` in `config.py`.
- Lower `TOLERANCE` reduces false positives but can increase unknown results.

