import cv2
import numpy as np
import tensorflow as tf
import math
from cvzone.HandTrackingModule import HandDetector
from keras import models

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=3)

offset = 20
imgSize = 300

labelsR = ["Backward", "Down", "Forward", "Idle", "Left", "Right", "Start", "Stop", "Up"]
labelsL = ["Backward", "Down", "Forward", "Idle", "Right", "Left", "Start", "Stop", "Up"]

model = models.load_model("Model4/kaggle_model4.h5")

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        for hand in hands:

            x, y, w, h = hand['bbox']
            handType = hand['type']
            y1 = max(0, y - offset)
            y2 = min(img.shape[0], y + h + offset)
            x1 = max(0, x - offset)
            x2 = min(img.shape[1], x + w + offset)

            imgCrop = img[y1:y2, x1:x2] # To adjust the bounding box so that no part gets cropped out at the extremes
            if handType == "Left":
                imgCrop = cv2.flip(imgCrop, 1)

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # to generate an all-white fixed sized image
            aspectRatio = h / w  # to cover the maximum amount of the area in the white image and ensure minimal space gets wasted, for better training
            if aspectRatio > 1:
                k = imgSize / h
                wCalculated = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCalculated, imgSize))
                wGap = math.ceil((imgSize - wCalculated) / 2)  # to centralize the image
                imgWhite[:, wGap:wCalculated + wGap] = imgResize
            else:
                k = imgSize / w
                hCalculated = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCalculated))
                hGap = math.ceil((imgSize - hCalculated) / 2)
                imgWhite[hGap:hCalculated + hGap, :] = imgResize

            imgWhiteReshaped = np.expand_dims(imgWhite, axis=0)
            prediction = model.predict(imgWhiteReshaped)
            score = tf.nn.softmax(prediction)
            # print(labelsR[np.argmax(score)]) if handType == "Right" else print(labelsL[np.argmax(score)])
            
            if(np.argmax(score) > 0.9):
                cv2.putText(imgOutput, f"{labelsR[np.argmax(score)] if handType == "Right" else labelsL[np.argmax(score)]} ({handType} Hand)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.rectangle(imgOutput, (x1, y1), (x2, y2), (255, 255, 0), 2)
            # cv2.imshow("ImageCrop", imgCrop)
            # cv2.imshow("ImgWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
