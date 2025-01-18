import cv2
import mediapipe as mp
import os
import time
import csv


# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def capture_gestures_to_csv(label, save_dir="dataset_csv", interval=0.5, max_images=10):
    """
    Captures hand gesture data and saves the landmark coordinates in a CSV file.

    Args:
        label (str): The label of the hand gesture (e.g., "thumbs_up").
        save_dir (str): The directory to save the CSV file.
        interval (float): Time interval between captures (in seconds).
        max_images (int): Maximum number of images to capture.
    """
    # Create a directory for the gesture label
    os.makedirs(save_dir, exist_ok=True)
    csv_file = os.path.join(save_dir, f"{label}.csv")

    # Initialize the CSV file and write the header
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        header = ["Label"] + [f"{axis}_{i}" for i in range(21) for axis in ["x", "y", "z"]]
        writer.writerow(header)

        # Initialize the webcam
        cap = cv2.VideoCapture(0)
        time.sleep(2)  # Allow the camera to initialize
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
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        # Check the time interval for capturing data
                        current_time = time.time()
                        if current_time - last_capture_time >= interval:
                            # Extract landmarks and save to CSV
                            row = [label]  # Start with the label
                            for landmark in hand_landmarks.landmark:
                                row.extend([landmark.x, landmark.y, landmark.z])
                            writer.writerow(row)
                            captured_count += 1
                            last_capture_time = current_time  # Update the last capture time

                            print(f"Captured {captured_count}/{max_images} for label: {label}")

                        # Break if max_images reached
                        if captured_count >= max_images:
                            break

                # Display the frame
                cv2.putText(frame, f"Label: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, f"Captured: {captured_count}/{max_images}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Hand Gesture Capture", frame)

                # Break on 'q' or when max_images are captured
                if cv2.waitKey(1) & 0xFF == ord('q') or captured_count >= max_images:
                    break

    # Release the webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {captured_count} entries for label: {label} and saved to {csv_file}")


# Call the function to capture gestures
label = input("Enter the gesture label (e.g., thumbs_up): ")
capture_gestures_to_csv(label)