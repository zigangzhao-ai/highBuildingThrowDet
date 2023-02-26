#!/usr/bin/python
# -*- coding:utf8 -*-

# import the necessary packages
import numpy as np
import cv2
import time


class Adjuster:
    def __init__(self, start_image, edge=(60, 20)):
        # determine if we are using OpenCV v3.X
        self.start_image = cv2.resize(start_image, (int(start_image.shape[1]/1), int(start_image.shape[0]/1)))
        self.edge = edge
        self.descriptor = cv2.ORB_create()
        self.matcher = cv2.DescriptorMatcher_create("BruteForce")
        (self.kps, self.features) = self.detectAndDescribe(self.start_image)

    def debouncing(self, image, ratio=0.7, reprojThresh=4.0, showMatches=False):
        image = cv2.resize(image, (int(image.shape[1]/1), int(image.shape[0]/1)))
        start = time.time()
        (kps, features) = self.detectAndDescribe(image)
        print(f"take {time.time() - start} s")
        M = self.matchKeypoints(kps, self.kps, features, self.features, ratio, reprojThresh)

        if M is None:
            return None

        (matches, H, status) = M
        result = cv2.warpPerspective(image, H, (image.shape[1] + image.shape[1], image.shape[0] + image.shape[0]))

        result = result[int(self.edge[1]):int(image.shape[0] - self.edge[1]),
                 int(self.edge[0]):int(image.shape[1] - self.edge[0])]

        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)

        start_img = self.start_image[int(self.edge[1]):int(image.shape[0] - self.edge[1]),
                    int(self.edge[0]):int(image.shape[1] - self.edge[0])]

        sub_img = cv2.absdiff(result, start_img)

        cv2.namedWindow("start_img", cv2.WINDOW_NORMAL)
        cv2.imshow("start_img", start_img)
        cv2.namedWindow("sub_img", cv2.WINDOW_NORMAL)
        cv2.imshow("sub_img", sub_img)

        return result

    def detectAndDescribe(self, image):
        # convert the image to grayscale
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # detect and extract features from the image

        (kps, features) = self.descriptor.detectAndCompute(image, None)

        # convert the keypoints from KeyPoint objects to NumPy
        # arrays
        kps = np.float32([kp.pt for kp in kps])

        # return a tuple of keypoints and features
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
                       ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual
        # matches
        rawMatches = self.matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # compute the homography between the two sets of points
            # ptsA = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
            # ptsB = np.array([[50, 50], [200, 50], [200, 200], [50, 200]])
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                             reprojThresh)

            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)

        # otherwise, no homograpy could be computed
        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # loop over the matches
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully
            # matched
            if s == 1:
                # draw the match
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 2)

        # return the visualization
        return vis
