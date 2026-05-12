import os

import cv2
import numpy as np


script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(script_dir, "traffic.mp4")

video = cv2.VideoCapture(video_path)
if not video.isOpened():
    raise FileNotFoundError(
        "Video file not found or cannot be opened. "
        f"Expected path: {video_path}"
    )

prev_pts = None
prev_gray_frame = None
tracks = None

lk_params = dict(
    winSize=(15, 15),
    maxLevel=5,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

feature_params = dict(
    maxCorners=500,
    qualityLevel=0.05,
    minDistance=10,
)

while True:
    retval, frame = video.read()

    if not retval:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_pts is not None:
        pts, status, errors = cv2.calcOpticalFlowPyrLK(
            prev_gray_frame,
            gray_frame,
            prev_pts,
            None,
            **lk_params,
        )

        good_pts = pts[status == 1]

        if tracks is None:
            tracks = good_pts
        else:
            tracks = np.vstack((tracks, good_pts))

        for p in tracks:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1)
    else:
        pts = cv2.goodFeaturesToTrack(gray_frame, **feature_params)
        if pts is None:
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            continue

        pts = pts.reshape(-1, 1, 2)

    prev_pts = pts
    prev_gray_frame = gray_frame

    cv2.imshow("frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

    if key == ord("c"):
        tracks = None
        prev_pts = None

video.release()
cv2.destroyAllWindows()
