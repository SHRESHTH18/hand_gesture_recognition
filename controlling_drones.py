import cv2
import mediapipe as mp
import numpy as np
import os
from tensorflow.keras.models import load_model
import speech_recognition as sr
from prompting import response

model = load_model('model.h5')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

if not os.path.exists("photos"):
    os.makedirs("photos")

cap = cv2.VideoCapture(0)

labels = ['back', 'forward', 'down', 'left', 'right', 'up']

recognizer = sr.Recognizer()

screen_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
screen_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

transcribed_text = ""
output_text = ""

def listen_for_speech():
    global transcribed_text
    with sr.Microphone() as source:
        print("Listening for speech...")
        audio = recognizer.listen(source)
        try:
            transcribed_text = recognizer.recognize_google(audio)
            print(f"Recognized: {transcribed_text}")
        except sr.UnknownValueError:
            transcribed_text = "Sorry, I didn't understand."
        except sr.RequestError:
            transcribed_text = "Speech recognition service error."

listening_for_speech = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('m'):
        listening_for_speech = True

    if listening_for_speech:
        listen_for_speech()
        output_text = response(transcribed_text)

        if "selfie" in transcribed_text.lower():
            selfie_path = f"photos/selfie_{len(os.listdir('photos')) + 1}.jpg"
            cv2.imwrite(selfie_path, frame)
            print(f"Selfie saved at: {selfie_path}")
            output_text="selfie clicked"

        listening_for_speech = False

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
    cv2.putText(frame, f"Transcribed Text: {transcribed_text}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Response: {output_text}", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.line(frame, (screen_width // 2, 0), (screen_width // 2, screen_height), (255, 0, 0), thickness=1)

    cv2.imshow('Hand Gesture Recognition with Speech', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

