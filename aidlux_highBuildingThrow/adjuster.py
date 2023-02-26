#!/usr/bin/python
# -*- coding:utf8 -*-

# import the necessary packages
import numpy as np
import cv2
import time
import os

class Adjuster:
    def __init__(self, start_image, edge=(60, 20)):
        # determine if we are using OpenCV v3.X
        self.start_image = cv2.resize(start_image, (int(start_image.shape[1]/1), int(start_image.shape[0]/1))) # 
        self.edge = edge
        self.descriptor = cv2.ORB_create() # 定位关键点 ，找到描述符
        self.matcher = cv2.DescriptorMatcher_create("BruteForce") # 描述符的匹配 
        (self.kps, self.features) = self.detectAndDescribe(self.start_image) # 将图片灰度化，并基于ORB算法，定位到关键点

    def debouncing(self, image, index, ratio=0.7, reprojThresh=4.0, showMatches=False):
        image = cv2.resize(image, (int(image.shape[1]/1), int(image.shape[0]/1))) # 截帧图片的长宽的量化
        start = time.time() 
        (kps, features) = self.detectAndDescribe(image) # 对图片灰度化，并基于ORB算法，定位到关键点
        print(f"take {time.time() - start} s") 
        M = self.matchKeypoints(kps, self.kps, features, self.features, ratio, reprojThresh) # 基于每一帧与背景帧，完成关键点的匹配,输出匹配变换矩阵、

        if M is None:
            return None

        (matches, H, status) = M

        """
        将图像按照变换映射M执行后返回变换后的图像result。
        参数: 
        * src input image.
        * dst output image that has the size dsize and the same type as src .
        * M  $ 3\cdot 3 $ transformation matrix.
        * dsize size of the output image.
        * flags  combination of interpolation methods (INTER_LINEAR or INTER_NEAREST) and the optional flag WARP_INVERSE_MAP, that sets M as the inverse transformation ( $\text{dst}\to \text{src}$ ).
        * borderMode  pixel extrapolation method (BORDER_CONSTANT or BORDER_REPLICATE).
        * borderValue  value used in case of a constant border; by default, it equals 0.
        """
        result = cv2.warpPerspective(image, H, (image.shape[1] + image.shape[1], image.shape[0] + image.shape[0])) 
        
        # 填充图片 opencv显示结果
        result = result[int(self.edge[1]):int(image.shape[0] - self.edge[1]),
                 int(self.edge[0]):int(image.shape[1] - self.edge[0])]

        # cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        # cv2.imshow("result", result)
        start_img = self.start_image[int(self.edge[1]):int(image.shape[0] - self.edge[1]),
                    int(self.edge[0]):int(image.shape[1] - self.edge[0])]

        # 获取两张图的差分图
        sub_img = cv2.absdiff(result, start_img)
        # if index% 10 == 0 :
        #    cv2.imwrite(os.path.join(r"C:\Users\shime\Desktop\highthrow(1)\images", "ori_{index}.jpg".format(index=index)), image)
        #    cv2.imwrite(os.path.join(r"C:\Users\shime\Desktop\highthrow(1)\images", "jiaozheng_{index}.jpg".format(index=index)), result)
        #    cv2.imwrite(os.path.join(r"C:\Users\shime\Desktop\highthrow(1)\images", "chafeng_{index}.jpg".format(index=index)), sub_img)


        # cv2.namedWindow("start_img", cv2.WINDOW_NORMAL)
        # cv2.imshow("start_img", start_img)
        # cv2.namedWindow("sub_img", cv2.WINDOW_NORMAL)
        # cv2.imshow("sub_img", sub_img)

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
        # 使用knn算法匹配关键点 K 表示按knn匹配规则输出的最优的K个结果， 在该算法中，输出的结果是：featureA（检测图像）中每个点与被匹配对象featureB（样本图像）中特征向量进行运算的匹配结果。
        """
        rawMatches中共有featureA条记录，每一条有最优K个匹配结果。
        每个结果中包含三个非常重要的数据分别是queryIdx，trainIdx，distance

        - queryIdx：特征向量的特征点描述符的下标（第几个特征点描述符），同时也是描述符对应特征点的下标
        - trainIdx：样本特征向量的特征点描述符下标,同时也是描述符对应特征点的下标
        - distance：代表匹配的特征点描述符的欧式距离，数值越小也就说明俩个特征点越相近
        """
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
            ptsA = np.float32([kpsA[i] for (_, i) in matches]) # 匹配到的样本图像特征点
            ptsB = np.float32([kpsB[i] for (i, _) in matches]) # 匹配到的检测图像的特征点

            # compute the homography between the two sets of points
            # ptsA = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
            # ptsB = np.array([[50, 50], [200, 50], [200, 200], [50, 200]])
            """
            计算多个二维点对之间的最优单映射变换矩阵H（3*3），status为可选的输出掩码，0/1 表示在变换映射中无效/有效
            * method（0， RANSAC, LMEDS, RHO）
            * ransacReprojThreshold 最大允许冲投影错误阈值（限方法RANSAC和RHO）
            * mask可选输出掩码矩阵，通常由鲁棒算法（RANSAC或LMEDS）设置，是不需要设置的
            * maxIters为RANSAC算法的最大迭代次数，默认值为2000
            * confidence可信度值，取值范围为0到1.
            """
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

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
