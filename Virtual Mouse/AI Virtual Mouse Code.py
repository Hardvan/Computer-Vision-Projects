import cv2
import numpy as np
import mediapipe as mp
import time
import autopy
import cvzone_HandTrackingModule_Improved as htm

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.HandDetector(maxHands=1)

wScreen, hScreen = autopy.screen.size()
# print(wScreen, hScreen)

while True:

    success, img = cap.read()

    # 1) Find hand landmarks
    allHands, img = detector.findHands(img)
    hand1 = allHands[0]
    lmList1 = hand1["lmList"]

    # 2) Get the tip of the index and middle fingers
    if len(lmList1) != 0:
        x1, y1 = lmList1[8][:2]  # Tip of index finger
        x2, y2 = lmList1[12][:2]  # Tip of middle finger
        # print(x1, y1, x2, y2)

    # 3) Check which fingers are up
    fingers = detector.fingersUp(hand1)
    # print(fingers)

    # 4) Only Index Finger : Moving Mode
    x3, y3 = 0, 0
    if fingers[1] == 1 and fingers[2] == 0:
        # 5) Convert Coordinates
        x3 = np.interp(x1, (0, wCam), (0, wScreen))
        y3 = np.interp(y1, (0, hCam), (0, hScreen))

    # 6) Smoothen Values
    # 7) Move Mouse
    autopy.mouse.move(x3, y3)
    cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
    # 8) Both Index and middle fingers are up : Clicking Mode
    # 9) Find distance between fingers
    # 10) Click mouse if distance short

    # 11) Frame Rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)

    # 12) Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)
