# ==============================================================================
#
# Real-Time Gesture Recognition Streamlit Web App
#
# Author: Charles Majola
# Version: 1.1.0
#
# Description:
# This web application uses Streamlit and Streamlit-WebRTC to provide a real-time
# gesture recognition interface directly in the browser. It uses a robust
# pipeline of YOLOv8 for hand region detection, MediaPipe for landmark
# extraction, and a Keras model for gesture classification.
#
# ==============================================================================

import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import torch
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# --- 1. Page Configuration and Title ---
st.set_page_config(
    page_title="Real-Time Gesture Recognition",
    page_icon="👋",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("👋 Real-Time Gesture Recognition")
st.write(
    "This app uses YOLOv8, MediaPipe, and a Keras model to recognize hand gestures "
    "in real-time from your webcam. Click 'START' below to begin."
)

# --- 2. Model Loading (Cached for performance) ---

@st.cache_resource
def load_yolo_model():
    """
    Loads the YOLOv8 model with a monkey-patch to handle PyTorch UnpicklingError.
    The @st.cache_resource decorator ensures the model is loaded only once.
    """
    # Monkey-patch torch.load to temporarily force weights_only=False
    original_torch_load = torch.load
    def patched_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_torch_load(*args, **kwargs)

    yolo_model = None
    try:
        torch.load = patched_torch_load
        yolo_model = YOLO('yolov8n.pt')
    finally:
        # Restore the original torch.load function
        torch.load = original_torch_load

    if yolo_model is None:
        st.error("CRITICAL: YOLOv8 model could not be loaded.")
    return yolo_model

@st.cache_resource
def load_gesture_model():
    """
    Creates and returns the untrained Keras gesture model.
    The @st.cache_resource decorator ensures the model is loaded only once.
    """
    model = Sequential([
        Dense(128, activation='relu', input_shape=(63,)),
        Dense(64, activation='relu'),
        Dense(4, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # To use your own trained model, uncomment the following line:
    # model.load_weights('path/to/your/gesture_model_weights.h5')
    return model

@st.cache_resource
def load_mediapipe_hands():
    """
    Initializes and returns the MediaPipe Hands solution.
    The @st.cache_resource decorator ensures the model is loaded only once.
    """
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7, # Increased confidence for better stability
        min_tracking_confidence=0.5
    )
    return hands, mp_hands.HAND_CONNECTIONS

# Load all the models and necessary components
yolo_model = load_yolo_model()
gesture_model = load_gesture_model()
hands, mp_hand_connections = load_mediapipe_hands()
mp_drawing = mp.solutions.drawing_utils
GESTURE_LABELS = ['Fist', 'Open Hand', 'Peace', 'Unknown']

# --- 3. Frame Processing Logic ---

def process(frame: av.VideoFrame) -> av.VideoFrame:
    """
    Callback function to process each frame from the WebRTC stream.
    This function contains the core computer vision pipeline.
    """
    img = frame.to_ndarray(format="bgr24")

    # Flip the frame horizontally for a natural, selfie-style view.
    img = cv2.flip(img, 1)

    # Use YOLOv8 to detect a 'person' (class 0) to find the ROI.
    yolo_results = yolo_model(img, classes=[0], verbose=False)

    hand_detected = False
    for result in yolo_results:
        if len(result.boxes) > 0:
            hand_detected = True
            # Get the bounding box of the first detected person.
            box = result.boxes[0].xyxy[0].int().tolist()
            x1, y1, x2, y2 = box

            # Draw the YOLO bounding box for visualization.
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Crop ROI and process with MediaPipe
            hand_roi = img[y1:y2, x1:x2]
            if hand_roi.size > 0:
                # Convert the ROI to RGB, as MediaPipe requires RGB input.
                rgb_hand_roi = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2RGB)
                mediapipe_results = hands.process(rgb_hand_roi)

                if mediapipe_results.multi_hand_landmarks:
                    for hand_landmarks in mediapipe_results.multi_hand_landmarks:
                        # Draw landmarks and connections on the original frame.
                        mp_drawing.draw_landmarks(img, hand_landmarks, mp_hand_connections)

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
                        cv2.putText(img, f'Gesture: {gesture_name}', (x1, y2 + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            # Process only the first detected person.
            break

    if not hand_detected:
        cv2.putText(img, "No hand detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 4. WebRTC Streaming ---

st.header("Webcam Feed")
st.write("Click 'START' to begin gesture recognition. You will need to grant webcam access when prompted by your browser.")

# RTCConfiguration is needed for deployment on platforms like Streamlit Cloud
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="gesture-recognition",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=process,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# --- 5. Sidebar Information ---
st.sidebar.header("About This Project")
st.sidebar.markdown(
    "This application demonstrates a multi-stage computer vision pipeline for "
    "real-time hand gesture recognition."
)
st.sidebar.markdown("**Author**: Charles Majola")
st.sidebar.markdown(
    "**GitHub Repository**: "
    "[chmj/app_yolodetect](https://github.com/chmj/app_yolodetect)"
)
