import numpy as np
import cv2
from cvs import *
import os
import time

class mogDetector:
    def __init__(self, history, varThreshold, minArea):
        self.minArea = minArea 
        """
        背景重建方法:MOG2
            history:用于训练背景的帧数，默认帧数为500帧，如果不动手设置learingRate,history就被用于计算当前的learningRate, 此时history越大，learningRate越小，背景更新越慢
            varThreshold:方差阈值，用于判断当前像素是前景还是背景。一般默认为16，如果光照变化明显，如阳光下的水面，建议设为25，值越大灵敏度越低。
            deteShadows:是否检测影子，设为true为检测，false为不检测，检测影子会增加程序时间复杂度，一般设置为false
        """
        self.detector = cv2.createBackgroundSubtractorMOG2(history, varThreshold, False) # 背景建模

        """
        # 得到一个结构元素（卷积核）。主要用于后续的腐蚀、膨胀、开、闭等运算。
          因为这些运算都是依赖于卷积核的，不同的卷积核（形状、大小）对图形的腐蚀、膨胀操作效果不一样

        输入参数：
            a设定卷积核的形状、b设定卷积核的大小、c表示描点的位置，一般 c = 1, 表示描点位于中心。
        返回值：
            返回指定形状和尺寸的结构元素（一般是返回一个矩形）、也就是腐蚀/膨胀用的核的大小。
        """
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1)) # 5

    def detectOneFrame(self, frame, index):

        if frame is None:
            return None
        start = time.time()
        mask = self.detector.apply(frame) # 背景重建，提取前景 
        # if index% 10 == 0 :
        #    cv2.imwrite(os.path.join(r"C:\Users\shime\Desktop\highthrow(1)\images", "mask_unprocess_{index}.jpg".format(index=index)), mask)
        stop = time.time()
        print("detect cast {} ms".format(stop - start))
        # cv2.namedWindow("mask_unprocess", cv2.WINDOW_NORMAL)
        # cv2.imshow("mask_unprocess", mask)

        start = time.time()
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel) # 做开运算 先腐蚀，再膨胀
        # if index% 10 == 0 :
        #    cv2.imwrite(os.path.join(r"C:\Users\shime\Desktop\highthrow(1)\images", "mask_process_open_{index}.jpg".format(index=index)), mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, self.kernel) # 再膨胀
        # if index% 10 == 0 :
        #    cv2.imwrite(os.path.join(r"C:\Users\shime\Desktop\highthrow(1)\images", "mask_process_dilate_{index}.jpg".format(index=index)), mask)
        stop = time.time()
        print("open contours cast {} ms".format(stop - start))

        start = time.time()
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 基于mask提取轮廓
        stop = time.time()
        print("find contours cast {} ms".format(stop - start))
        i = 0
        bboxs = []
        start = time.time()
        for c in contours:
            i += 1
            if cv2.contourArea(c) < self.minArea: # 过滤
                continue

            bboxs.append(cv2.boundingRect(c)) # 基于轮廓 寻找外接矩形 
        stop = time.time()
        print("select cast {} ms".format(stop - start))

        return mask, bboxs
