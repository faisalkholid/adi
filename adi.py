import cv2
import numpy as np
import sys
from scipy import signal
from typing import *
import os


class MotionADI(object):
    # limit = batas ADI yang di proses || fdn = batas frame untuk mereset put-text deteksi
    def __init__(self, thresh = 0.5, limit = 10, fdn = 20, adi_path = "result", **kwargs):
        self.fdn = fdn
        self.thresh = thresh
        self.limit = limit
        self.frames: List = []
        self.idx = 0
        self.motion_idx = []
        self.motion_frames = []
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.is_detected = False
        self.detected_id = 0
        self.adi_path = adi_path
        self.adi_id = 0

    def _segmentation(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        op_kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, op_kernel)

        mean_kernel = np.ones((5, 5), np.uint8) / np.mean(opening)
        mean = cv2.filter2D(opening, -1, mean_kernel)

        output = mean
        return output

    # Mengkombinsai setiap frame yang ada gerakan per 20 frame
    def combine_motion_frame(self):
        fshape = self.motion_frames[-1].shape
        image = np.zeros(fshape)
        for frame in self.motion_frames:
            image = np.add(image, frame)
        return image

    def filter(self, frame):
        # Menghitung jumlah frame
        seg = self._segmentation(frame)
        self.frames.append(seg)
        self._put_text(frame, f'Frame Number: {str(self.idx)}', loc=(10, 40))

        if len(self.frames) > 0:
            abs_diff = cv2.absdiff(self.frames[self.idx - 1], self.frames[self.idx])
            motion = np.mean(abs_diff) > self.thresh
            if motion:
                # Looping id jika ada gerakan
                self.motion_idx.append(self.idx)
                if len(self.motion_frames) < self.limit:
                    # Mengumpulkan Frame
                    self.motion_frames.append(abs_diff)
                    return False, abs_diff
                else:
                    # Meyimpan hasil 20 frame ketika ada gerakan
                    image_adi = self.combine_motion_frame()
                    path = os.path.join(self.adi_path, f'frame_{self.adi_id}.jpg')
                    cv2.imwrite(path, image_adi)
                    self.adi_id += 1

                    self.motion_frames = []
                    self.motion_idx = []
                    self.is_detected = True
                    self.detected_id = self.idx

                    return True, image_adi
            else:
                return False, abs_diff
        else:
            return False, np.zeros(frame.shape)

    # Penambahan/Looping frame id
    def increment_id(self):
        self.idx = self.idx + 1

    def show_detection(self, frame):
        # Jika terdeteksi munculin text
        if self.is_detected:
            self._put_text(frame, "Motion: Detected")
            # Jika sudah lebih dari fdn maka akan kembali ke tidak terdeteksi jika tidak ada gerakan
            if self.idx - self.detected_id >= self.fdn:
                self.is_detected = False
                self.detected_id = 0
        else:
        # Jika tidak terdeteksi munculin text
            self._put_text(frame, "Motion: Undetected")

    def _put_text(self, frame, text, loc=(10, 20)):
        cv2.putText(frame, text, loc, self.font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)


cap = cv2.VideoCapture(0)
madi = MotionADI(thresh = 0.5, fdn = 20)

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    mot_detect, output = madi.filter(frame)
    madi.show_detection(frame)
    madi.increment_id()

    cv2.imshow('motion', output)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()