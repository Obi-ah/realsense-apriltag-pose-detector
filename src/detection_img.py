import cv2
from apriltag import Detector, DetectorOptions, Detection

img = cv2.imread("src/Object detection/tags/tag_36h11.png")
imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


options = DetectorOptions()
detector = Detector(options)
print(detector.families)
results = detector.detect(imggray)
print(len(results))

# x= Detection()
# x.
if len(results) > 0:
    # print('hey')
    
    for tag in results:
        # print(tag.homography)
        ptA, ptB, ptC, ptD  = tag.corners

        ptA = (int(ptA[0]), int(ptA[1]))
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))

        cv2.line(img, ptA, ptB, (0, 255, 0), 4)
        cv2.line(img, ptB, ptC, (0, 255, 0), 4)
        cv2.line(img, ptC, ptD, (0, 255, 0), 4)
        cv2.line(img, ptD, ptA, (0, 255, 0), 4)

        # print(ptA)
        cv2.circle(img, (int(tag.center[0]), int(tag.center[1])), 5, (0, 0, 255), -1)


cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()