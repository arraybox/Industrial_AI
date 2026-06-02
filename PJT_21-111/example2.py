import os

import cv2
import numpy as np


script_dir = os.path.dirname(os.path.abspath(__file__))


def find_video_source():
    for filename in ("faces.mp4", "Faces.mp4"):
        video_path = os.path.join(script_dir, filename)
        if os.path.exists(video_path):
            return video_path
    return 0


def detect_faces(video_file, detector, win_title):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video source: {video_file}")

    while True:
        status_cap, frame = cap.read()
        if not status_cap:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector.detectMultiScale(gray, 1.3, 5)

        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            text_size, _ = cv2.getTextSize("Face", cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(
                frame,
                (x, y - text_size[1]),
                (x + text_size[0], y),
                (255, 255, 255),
                cv2.FILLED,
            )
            cv2.putText(
                frame,
                "Face",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2,
            )

        cv2.imshow(win_title, frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def load_cascade(local_filename, fallback_filename=None, required=True):
    local_path = os.path.join(script_dir, local_filename)
    cascade = cv2.CascadeClassifier(local_path)

    if cascade.empty() and fallback_filename:
        fallback_path = os.path.join(cv2.data.haarcascades, fallback_filename)
        cascade = cv2.CascadeClassifier(fallback_path)

    if cascade.empty() and required:
        raise FileNotFoundError(
            f"Cannot load cascade file. Place {local_filename} in {script_dir}."
        )

    if cascade.empty():
        return None

    return cascade


video_source = find_video_source()

haar_face_cascade = load_cascade(
    "haarcascade_frontalface_default.xml",
    "haarcascade_frontalface_default.xml",
)
detect_faces(video_source, haar_face_cascade, "Haar cascade face detector")

lbp_face_cascade = load_cascade("lbpcascade_frontalface.xml", required=False)
if lbp_face_cascade is not None:
    detect_faces(video_source, lbp_face_cascade, "LBP cascade face detector")
else:
    print("LBP cascade file not found. Skipping LBP cascade face detector.")
