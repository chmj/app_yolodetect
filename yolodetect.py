# ==============================================================================
#
# Real-Time Gesture Recognition Application
#
# Author: [Charles Majola/github:chmj]
# Version: 1.0.0
# License: MIT
#
# Description:
# This application combines YOLOv8, MediaPipe, and Keras to perform real-time
# gesture recognition from a webcam feed.
#
# Workflow:
# 1. YOLOv8: Detects a person in the frame to identify a Region of Interest (ROI).
#    This is a robust method to narrow down the search area for a hand.
# 2. MediaPipe: Accurately extracts hand landmarks from within the detected ROI.
# 3. Keras: Uses the extracted landmarks to classify a custom gesture.
#
# ==============================================================================

import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import torch
import pickle

# --- 1. Model Definition & Setup ---------------------------------------------

def create_gesture_model():
    """
    Creates a simple Keras model for gesture recognition.

    This model is a placeholder. For a real-world application, this model
    should be trained on a dataset of hand landmarks and their corresponding
    gestures. The input shape is 63, which corresponds to 21 landmarks,
    each with 3 coordinates (x, y, z).

    Returns:
        tensorflow.keras.Model: A compiled, untrained Keras model.
    """
    model = Sequential([
        Dense(128, activation='relu', input_shape=(63,)),
        Dense(64, activation='relu'),
        Dense(4, activation='softmax')  # Output layer for 4 gestures
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# --- 2. Application Setup ----------------------------------------------------

# Robustly load the YOLOv8 model, handling potential PyTorch UnpicklingErrors.
yolo_model = None

# Monkey-patch torch.load to temporarily force weights_only=False.
# This is the most reliable way to handle the PyTorch 2.x security update
# that causes UnpicklingError when loading older model formats.
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    """A patched version of torch.load that forces weights_only=False."""
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

try:
    # Temporarily replace the original torch.load with our patched version
    torch.load = patched_torch_load
    yolo_model = YOLO('yolov8n.pt')
finally:
    # IMPORTANT: Always restore the original torch.load function to avoid
    # potential side effects in other parts of the application or libraries.
    torch.load = original_torch_load

if yolo_model is None:
    print("CRITICAL: YOLOv8 model could not be loaded. Exiting.")
    exit()

# Initialize MediaPipe Hands solution.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,      # Process a video stream, not static images.
    max_num_hands=1,              # Detect only one hand for simplicity.
    min_detection_confidence=0.5, # Minimum confidence for a hand to be detected.
    min_tracking_confidence=0.5   # Minimum confidence for tracking to be successful.
)
mp_drawing = mp.solutions.drawing_utils

# Create the custom Keras gesture model.
gesture_model = create_gesture_model()
# To use your own trained model, uncomment the following line and provide the path:
# gesture_model.load_weights('path/to/your/gesture_model_weights.h5')

# Define the gesture labels that correspond to the Keras model's output.
GESTURE_LABELS = ['Fist', 'Open Hand', 'Peace', 'Unknown']

# --- 3. Main Application Logic -----------------------------------------------

def main():
    """
    The main function that captures video, runs the detection pipeline,
    and displays the results.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Warning: Ignoring empty camera frame.")
            continue

        # Flip the frame horizontally for a natural, selfie-style view.
        frame = cv2.flip(frame, 1)

        # Use YOLOv8 to detect a 'person' (class 0) to find the ROI.
        # verbose=False prevents YOLO from printing its own logs to the console.
        yolo_results = yolo_model(frame, classes=[0], verbose=False)

        hand_detected = False
        for result in yolo_results:
            # Check if any person was detected.
            if len(result.boxes) > 0:
                hand_detected = True
                # Get the bounding box of the first detected person.
                box = result.boxes[0].xyxy[0].int().tolist()
                x1, y1, x2, y2 = box

                # Draw the YOLO bounding box for visualization.
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, 'Hand Region', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

                # Crop the frame to the detected ROI for MediaPipe processing.
                hand_roi = frame[y1:y2, x1:x2]
                if hand_roi.size == 0:
                    continue

                # Convert the ROI to RGB, as MediaPipe requires RGB input.
                rgb_hand_roi = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2RGB)
                mediapipe_results = hands.process(rgb_hand_roi)

                # If landmarks are found, process and predict the gesture.
                if mediapipe_results.multi_hand_landmarks:
                    for hand_landmarks in mediapipe_results.multi_hand_landmarks:
                        # Draw the landmarks and connections on the original frame.
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS
                        )

                        # Extract landmarks and flatten them for the Keras model.
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            landmarks.extend([lm.x, lm.y, lm.z])
                        landmarks = np.array(landmarks).reshape(1, -1)

                        # Predict the gesture using the Keras model.
                        prediction = gesture_model.predict(landmarks, verbose=0)
                        gesture_id = np.argmax(prediction)
                        gesture_name = GESTURE_LABELS[gesture_id]

                        # Display the final gesture prediction on the screen.
                        cv2.putText(frame, f'Gesture: {gesture_name}', (x1, y2 + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Process only the first detected person to avoid clutter.
                break

        if not hand_detected:
            cv2.putText(frame, "No hand detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the final, annotated frame.
        cv2.imshow('Real-Time Gesture Recognition', frame)

        # Exit the loop if the 'ESC' key is pressed.
        if cv2.waitKey(5) & 0xFF == 27:
            break

    # Release resources.
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
