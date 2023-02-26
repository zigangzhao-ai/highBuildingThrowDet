import numpy as np
import cv2

import time


class knnDetector:
    def __init__(self, history, dist2Threshold, minArea):
        self.minArea = minArea 
        self.detector = cv2.createBackgroundSubtractorKNN(history, dist2Threshold, False)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detectOneFrame(self, frame):
        if frame is None:
            return None
        start = time.time()
        mask = self.detector.apply(frame)
        stop = time.time()
        print("detect cast {} ms".format(stop - start))

        start = time.time()
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, self.kernel)
        stop = time.time()
        print("open contours cast {} ms".format(stop - start))

        start = time.time()
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        stop = time.time()
        print("find contours cast {} ms".format(stop - start))
        i = 0
        bboxs = []
        start = time.time()
        for c in contours:
            i += 1
            if cv2.contourArea(c) < self.minArea:
                continue

            bboxs.append(cv2.boundingRect(c))
        stop = time.time()
        print("select cast {} ms".format(stop - start))

        return mask, bboxs
