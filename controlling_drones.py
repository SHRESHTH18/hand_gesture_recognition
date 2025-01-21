
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('model.h5')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

labels = ['back', 'forward', 'down', 'left', 'right', 'up']

screen_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
screen_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    drone_1_text = "Drone 1: Hold"
    drone_2_text = "Drone 2: Hold"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark
            wrist_x = landmarks[0].x * screen_width
            data = np.array([[landmark.x, landmark.y, landmark.z] for landmark in landmarks]).flatten()

            prediction = model.predict(np.expand_dims(data, axis=0))

            gesture = np.argmax(prediction)

            if wrist_x < screen_width / 2:
                drone_1_text = f"Drone 1: {labels[gesture]}"
            else:
                drone_2_text = f"Drone 2: {labels[gesture]}"

    cv2.putText(frame, drone_1_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, drone_2_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.line(frame,(screen_width//2,0),(screen_width//2,screen_height),(255,0,0),thickness=1)
    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()