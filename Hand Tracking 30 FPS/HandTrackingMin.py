import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Setting Frame Rate
prev_time = 0
cur_time = 0

while True:
    success, img = cap.read()
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Converting image to RGB
    results = hands.process(imgRGB)
    
    # print(results.multi_hand_landmarks)
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                height, width, chance = img.shape
                center_x, center_y = int(lm.x * width), int(lm.y * height)
                print(id, center_x, center_y)
                if id == 0:
                    cv2.circle(img, (center_x, center_y), 25, (255,0,255), cv2.FILLED)
                
            
            # Drawing Hand Dots only
            # mpDraw.draw_landmarks(img, handLms)

            # Drawing Hand Connections
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    
    # Adjusting Frame Rate
    cur_time = time.time()
    fps = 1/(cur_time - prev_time)
    prev_time = cur_time
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN,
                3, (255,0,255), 3)
    
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    









































