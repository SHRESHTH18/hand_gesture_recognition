import cv2
import mediapipe as mp
import os
import time

# Initialize Mediapipe Hands and Drawing modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def capture_gestures(label, save_dir="dataset", interval=0.5, max_images=10):
    """
    Captures images of a hand gesture and saves them to a labeled folder.

    Args:
        label (str): The label of the hand gesture (e.g., "thumbs_up").
        save_dir (str): The directory to save images.
        interval (float): Time interval between captures (in seconds).
        max_images (int): Maximum number of images to capture.
    """
    # Create a directory for the gesture label
    gesture_dir = os.path.join(save_dir, label)
    os.makedirs(gesture_dir, exist_ok=True)

    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    time.sleep(5)
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
        captured_count = 0
        last_capture_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image. Exiting.")
                break

            # Flip the image horizontally for a mirror view
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            # Draw hand landmarks on the frame
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Display the frame
            cv2.putText(frame, f"Label: {label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"Images: {captured_count}/{max_images}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Hand Gesture Capture", frame)

            # Capture image at the specified interval
            if time.time() - last_capture_time >= interval and results.multi_hand_landmarks:
                image_path = os.path.join(gesture_dir, f"{captured_count + 1}.jpg")
                cv2.imwrite(image_path, frame)
                captured_count += 1
                last_capture_time = time.time()

            # Break when 'q' is pressed or max_images are captured
            if cv2.waitKey(1) & 0xFF == ord('q') or captured_count >= max_images:
                break

    # Release the webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {captured_count} images for label: {label}")


# Call the function to capture gestures
label = input("Enter the gesture label (e.g., thumbs_up): ")
capture_gestures(label)
