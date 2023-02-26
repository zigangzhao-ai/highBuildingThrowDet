import cv2
import numpy as np
from knnDetector import knnDetector
from sort import Sort
import time
import adjuster

def start_detect():
    path = "IMG_4550.MOV"
    capture = cv2.VideoCapture(path) # 获取视频 
    capture.set(cv2.CAP_PROP_POS_FRAMES, 200) # 设置视频从第几帧开始读取
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # 设置写入视频的编码类型
    fps = capture.get(cv2.CAP_PROP_FPS) # 读取视频的fps
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))) # 读取视频图谱的宽和长
    out = cv2.VideoWriter('highThrow-closer.mp4', fourcc, 25, size) # 写入视频的格式设置

    detector = knnDetector(500, 400, 10) # 背景重建
    cv2.destroyAllWindows() # 删除所有窗口
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL) # 创建三个窗口
    cv2.namedWindow("history", cv2.WINDOW_NORMAL)
    cv2.namedWindow("mask", cv2.WINDOW_NORMAL)

    flag = False

    # 不能忍受漏检，需要预测成功十次才返回预测框，IOU最少0.1

    # max_age=1, min_hits=3, iou_threshold=0.3
    """
    max_age:
    目标所对应的轨迹停止更新状态的最大帧数 
    如果一个当前帧的目标在下一帧中没被检测出来，
    那么该目标的kalman滤波的先验状态预测将会失去与下一帧目标检测值的匹配的机会（因为下一帧这个目标没被检测出来），
    此时轨迹的kalman滤波器状态不会更新，将先验预测作为下一帧该目标的状态，然后重点来了！如果此时max_age设置为1，
    则在下下一帧中，如果该目标得轨迹还是没有得到匹配和更新，则它就会被删除，在后续的帧中，就不会跟踪到该目标，跟踪器认为该目标已经走出了帧外，或者该目标被遮挡。
    
    min_hits:
    代表持续多少帧检测到，生成trackers
    上一帧和下一帧目标之间的iou阈值，大于iou阈值认为是一个目标

    iou_threshold 

    """
    sort = Sort(3, 5, 0.1)

    # 读取视频帧
    ret, frame = capture.read()

    # adjuster 
    
    adjust = adjuster.Adjuster(frame, (120, 60))

    index = 0
    while True:
        ret, frame = capture.read()
        if frame is None:
            break

        frame_start = time.time()
        frame = adjust.debouncing(frame)
        print(f"debouncing image take {time.time() - frame_start} s")

        start = time.time()
        mask, bboxs = detector.detectOneFrame(frame)
        print(f"detectOneFrame image take {time.time() - start} s")

        start = time.time()
        if bboxs != []:
            bboxs = np.array(bboxs)
            bboxs[:, 2:4] += bboxs[:, 0:2]
            # test
            # for bbox in bboxs:
            #     cv2.rectangle(mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 1)
            #     cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 2)
            trackBox = sort.update(bboxs)
        else:
            # test
            trackBox = sort.update()

        print(f"track image take {time.time() - start} s")

        # test
        for bbox in trackBox:
            bbox = [int(bbox[i]) for i in range(5)]
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 6)
            cv2.putText(frame, str(bbox[4]), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

        # # out.write(frame)
        #
        cv2.imshow("mask", mask)
        cv2.imshow("frame", frame)
        if flag:
            if cv2.waitKey(0) == 27:
                flag = False
        else:
            if cv2.waitKey(1) == 27:
                flag = True
        end = time.time()
        print("one frame coast : ", end - frame_start)
        print(index)
        index += 1
    out.release()
    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    start_detect()