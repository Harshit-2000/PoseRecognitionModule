import cv2 as cv
import mediapipe as mp
import time


class PoseDetector:
    def __init__(self, mode = False, model_complexity = 1, smooth_landmarks = True,
                 detection_conf = 0.5, tracking_conf = 0.5):

        self.mode = mode
        self.model_complexity = model_complexity
        self.smooth_landmarks =  smooth_landmarks
        self.detection_conf = detection_conf
        self.tracking_conf = tracking_conf

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.model_complexity, self.smooth_landmarks,
                                     self.detection_conf, self.tracking_conf)
        self.mpDraw = mp.solutions.drawing_utils

    def detectBody(self, img, draw = True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        # print(results.pose_landmarks)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw = False):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(w * lm.x), int(h * lm.y)
                lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 6, (255, 255, 255), -1)
        return lmList

def main():
    capture = cv.VideoCapture(0)
    pTime = 0

    detector = PoseDetector()
    while True:
        isTrue, img = capture.read()
        img = detector.detectBody(img)
        lmList = detector.findPosition(img)

        if len(lmList) != 0:
            print(lmList[4])
            
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_ITALIC, 2, (0, 0, 255), 3)
        cv.imshow('webCam', img)
        cv.waitKey(1)

if __name__ == "__main__":
    main()