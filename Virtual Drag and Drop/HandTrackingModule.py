import cv2
import mediapipe as mp
import time

class handDetector():
    
    def __init__(self, mode=False, maxHands=1, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, 
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
    
    def findHands(self, img, draw=True):
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Converting image to RGB
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        
        return img
    
    def findPosition(self, img, handNo=0, draw=True):
        
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                height, width, chance = img.shape
                center_x, center_y = int(lm.x * width), int(lm.y * height)
                # print(id, center_x, center_y)
                lmList.append([id, center_x, center_y])
                if draw:
                    cv2.circle(img, (center_x, center_y), 7, (255,0,255), cv2.FILLED)
        
        return lmList

    
def main():
    
    prev_time = 0
    cur_time = 0
    
    cap = cv2.VideoCapture(0)
    
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList)!=0:
            print(lmList[4])
    
        cur_time = time.time()
        fps = 1/(cur_time - prev_time)
        prev_time = cur_time
        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN,
                3, (255,0,255), 3)
    
        cv2.imshow("Image", img)
        cv2.waitKey(1)
    
    

if __name__ == "__main__":
    main()








































