import cv2
import numpy as np
from typing import *
import os

class MotionADI(object):
    # limit = batas ADI yang di proses || fd = batas frame untuk mereset put-text deteksi
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
        self.human_detector = HumanDetector()

    def _segmentation(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # r = 500.0 / gray.shape[1]
        # dim = (500, int(gray.shape[0] * r))
        # gray = cv2.resize(gray, dim, interpolation=cv2.INTER_AREA)
        # cv2.imshow('grayscale',gray)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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
            abs_diff = cv2.absdiff(self.frames[self.idx],self.frames[self.idx - 1])
            motion = np.mean(abs_diff) > self.thresh
            # mengecek jika ada gerakan dan itu gerakan manusia maka sistem mengeluarkan "Human Detection"
            if motion and self.human_detector.detect(frame) :
                # Looping id jika ada gerakan
                self.motion_idx.append(self.idx)
                if len(self.motion_frames) < self.limit:
                    # Mengumpulkan Frame
                    self.motion_frames.append(abs_diff)
                    return False, abs_diff
                else :
                    # Meyimpan hasil 10 frame ketika ada gerakan
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
        if self.is_detected and self.human_detector.detect(frame):
            self._put_text(frame, "Human Motion: Detected")
            # Jika sudah lebih dari fdn maka akan kembali ke tidak terdeteksi jika tidak ada gerakan
            if self.idx - self.detected_id >= self.fdn:
                self.is_detected = False
                self.detected_id = 0
        else:
        # Jika tidak terdeteksi munculin text
            self._put_text(frame, "Human Motion: Undetected")

    def _put_text(self, frame, text, loc=(10, 20)):
        cv2.putText(frame, text, loc, self.font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)


class HumanDetector():
    def __init__(self, upper_detect = False, face_detect = False):
        self.upper_detect = upper_detect
        self.face_detect = face_detect
        self.build()

    def build(self):
        self.person_cascade = cv2.CascadeClassifier(os.path.join('data/haarcascade_fullbody.xml'))
        if self.upper_detect:
            self.upper_cascade = cv2.CascadeClassifier(os.path.join('data/haarcascade_upperbody.xml'))

        if self.face_detect:
            self.face_cascade = cv2.CascadeClassifier(os.path.join('data/haarcascade_frontalface_default.xml'))

    def detect(self, frame):
        # mengecek dengan haar cascade clasifier apaka itu manusia atau bukan
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        human = self.person_cascade.detectMultiScale(gray)

        if len(human) > 0:
            if self.upper_detect:
                upper_body = self.upper_cascade.detectMultiScale(gray)
                if len(upper_body) > 0:
                    return True
            if self.face_detect:
                face = self.face_cascade.detectMultiScale(gray)
                if len(face) > 0:
                    return True
            if self.face_detect and self.upper_detect:
                upper_body = self.upper_cascade.detectMultiScale(gray)
                face = self.face_cascade.detectMultiScale(gray)
                if len(face) > 0 and len(upper_body) > 0:
                    return True
            return True
        else:
            return False

# person_cascade = cv2.CascadeClassifier(os.path.join('data/haarcascade_fullbody.xml'))
# upper_body_cascade = cv2.CascadeClassifier(os.path.join('data/haarcascade_upperbody.xml'))
# lower_body_cascade = cv2.CascadeClassifier(os.path.join('data/haarcascade_lowerbody.xml'))
# face_cascade = cv2.CascadeClassifier(os.path.join('data/haarcascade_frontalface_default.xml'))

# def tester(frame):
#
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     human = face_cascade.detectMultiScale(gray)
#
#     for (x, y, w, h) in human:
#         # for whole body detetction
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
#         roi_gray = gray[y:y + h, x:x + w]
#         roi_color = frame[y:y + h, x:x + w]
#
#         # for upper body detection
#         # upper_body = upper_body_cascade.detectMultiScale(roi_gray)
#         # for (ux, uy, uw, uh) in upper_body:
#         #     cv2.rectangle(roi_color, (ux, uy), (ux + uw, uy + uh), (0, 0, 255), 2)
#
#         # for lower body detection
#         # lower_body = lower_body_cascade.detectMultiScale(roi_gray)
#         # for (lx, ly, lw, lh) in lower_body:
#         #     cv2.rectangle(roi_color, (lx, ly), (lx + lw, ly + lh), (255, 0, 0), 2)
#
#         # for face detection
#         face = face_cascade.detectMultiScale(roi_gray)
#         for (fx, fy, fw, fh) in face:
#             cv2.rectangle(roi_color, (fx, fy), (fx + fw, fy + fh), (120, 230, 0), 4)

cap = cv2.VideoCapture("video/1.mp4")
madi = MotionADI(thresh = 0.3, fdn = 10)

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    mot_detect, output = madi.filter(frame)
    madi.show_detection(frame)
    madi.increment_id()
    # tester(frame)

    r = 800.0 / frame.shape[1]
    dim = (800, int(frame.shape[0] * r))
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    r = 500.0 / output.shape[1]
    dim = (500, int(output.shape[0] * r))
    output = cv2.resize(output, dim, interpolation=cv2.INTER_AREA)

    cv2.imshow('motion', output)
    cv2.imshow('frame', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()