import cv2
import numpy as np
import sys
from scipy import signal


def new_seg(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    op_kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, op_kernel)

    mean_kernel = np.ones((5, 5), np.uint8) / np.mean(opening)
    mean = cv2.filter2D(opening, -1, mean_kernel)

    output = mean
    return output


cap = cv2.VideoCapture("sampel.mp4")

idx = 0
idx_mot = []
frames = []
mot_frames = []
mot_limit = 10

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     seg_curr = segmentation(frame, use_canny=False)
    seg_curr = new_seg(frame)
    frames.append(seg_curr)
    seg_curr
    S = 0.5

    if len(frames) > 0:
        abs_diff = cv2.absdiff(frames[idx], frames[idx - 1])
        motion = np.average(abs_diff) > S
        if motion:
            idx_mot.append(idx)
            print(idx_mot)
            if len(mot_frames) < 10:
                mot_frames.append(abs_diff)
            else:
                mot_frames = []

        print('\r' + str(motion), end='')

    #     seg = new_seg(frame)
    cv2.imshow('frame', frame)
    cv2.imshow('seg', abs_diff)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    idx = idx + 1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()