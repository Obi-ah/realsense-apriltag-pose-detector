import cv2
from apriltag import Detector, DetectorOptions
import pyrealsense2 as rs
import numpy as np


options = DetectorOptions()
detector = Detector(options)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    img = frame

    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = detector.detect(imggray)

    # Mark boundaries and center
    if results:
        tag = results[0]
        ptA, ptB, ptC, ptD  = tag.corners
        
        ptA = (int(ptA[0]), int(ptA[1]))
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))

        cv2.line(img, ptA, ptB, (0, 255, 0), 4)
        cv2.line(img, ptB, ptC, (0, 255, 0), 4)
        cv2.line(img, ptC, ptD, (0, 255, 0), 4)
        cv2.line(img, ptD, ptA, (0, 255, 0), 4)

        center = (int(tag.center[0]), int(tag.center[1]))
        cv2.circle(img, center, 5, (0, 0, 255), -1)



    cv2.namedWindow('image')
    cv2.resizeWindow('image', 400, 400)
    color_image_flipped = cv2.flip(img, cv2.ROTATE_180)
    cv2.imshow("image", color_image_flipped)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
