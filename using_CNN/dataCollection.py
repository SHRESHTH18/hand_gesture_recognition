import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

folder = "Data2/Right"
counter = 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img)
# To adjust the bounding box so that no part gets cropped out at the extremes
    if hands:
        hand = hands[0]
        
        x, y, w, h = hand['bbox']
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)

        imgCrop = img[y1:y2, x1:x2]
        
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255 # to generate an all-white fixed sized image
        
        aspectRatio = h / w # to cover the maximum amount of the area in the white image and ensure minimal space gets wasted, for better training
        if(aspectRatio > 1):
            k = imgSize / h
            wCalculated = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCalculated, imgSize))
            wGap = math.ceil((imgSize-wCalculated)/2) # to centralize the image
            imgWhite[:, wGap:wCalculated + wGap] = imgResize
        else:
            k = imgSize / w
            hCalculated = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCalculated))
            hGap = math.ceil((imgSize-hCalculated)/2)
            imgWhite[hGap:hCalculated + hGap, :] = imgResize           
        
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImgWhite", imgWhite)
    
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)