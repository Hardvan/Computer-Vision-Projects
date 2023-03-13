import cv2
from cvzone_HandTrackingModule_Improved import HandDetector
import cvzone
import numpy as np

cap = cv2.VideoCapture(0)

# Width
cap.set(3, 1280)

# Height
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8, maxHands=1)

colorR = (255, 0, 255)

cx, cy, w, h = 100, 100, 200, 200


class DragRect():

    def __init__(self, posCenter, size=[200, 200], colorR=(255, 0, 255)):
        self.posCenter = posCenter
        self.size = size
        self.colorR = colorR

    def update(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size

        # If index finger in rectangle region
        if cx-w//2 < cursor[0] < cx+w//2 and cy-h//2 < cursor[1] < cy+h//2:
            self.colorR = (0, 255, 0)
            # print(cursor)
            self.posCenter[0], self.posCenter[1] = cursor[0], cursor[1]
        # else:
            self.colorR = (255, 0, 255)


rectList = []
for x in range(3):
    rectList.append(DragRect([x*250 + 150, 150]))

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    allHands, img = detector.findHands(img)
    lmList = []
    if allHands:
        lmList = allHands[0]["lmList"]

    if lmList:

        l, _, _ = detector.findDistance(lmList[8], lmList[12], img, draw=False)
        print(l)
        if l < 50:
            cursor = lmList[8]  # Index finger tip
            # call update here
            for rect in rectList:
                rect.update(cursor)

    # # Drawing Rectangle
    # for rect in rectList:
    #     cx, cy = rect.posCenter
    #     w, h = rect.size
    #     cv2.rectangle(img, (cx-w//2, cy-h//2), (cx+w//2, cy+h//2), colorR, cv2.FILLED)
    #     cvzone.cornerRect(img, (cx-w//2, cy-h//2, w, h), 20, rt=0)

    # Draw Transparency
    imgNew = np.zeros_like(img, np.uint8)
    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size
        cv2.rectangle(imgNew, (cx-w//2, cy-h//2),
                      (cx+w//2, cy+h//2), rect.colorR, cv2.FILLED)
        cvzone.cornerRect(imgNew, (cx-w//2, cy-h//2, w, h), 20, rt=0)
    out = img.copy()
    alpha = 0.0
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1-alpha, 0)[mask]

    cv2.imshow("Image", out)
    cv2.waitKey(1)
