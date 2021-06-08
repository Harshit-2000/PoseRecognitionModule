import cv2 as cv
import mediapipe as mp
import time

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

capture = cv.VideoCapture(0)
pTime = 0
while True:
    isTrue, img = capture.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h , w, c = img.shape
            cx, cy = int(w * lm.x), int(h * lm.y)

            print(id, cx, cy)
            # cv.circle(img, (cx, cy), 6, (255, 255, 255), -1)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv.putText(img, str(int(fps)), (10, 70), cv.FONT_ITALIC, 2, (0, 0, 255), 3)
    cv.imshow('webCam', img)
    cv.waitKey(1)