# Real-Time Gesture Recognition App (Desktop & Web)

A real-time computer vision application that detects and classifies hand gestures from a webcam feed using a powerful combination of deep learning and machine learning models. This repository contains both a local desktop version and a web-based Streamlit application.

This project uses:
- **YOLOv8** for initial, robust hand region detection.
- **MediaPipe** for accurate, high-fidelity hand landmark extraction.
- **TensorFlow/Keras** for custom gesture classification.
- **OpenCV** for local video capture and rendering.
- **Streamlit** and **Streamlit-WebRTC** for the interactive web interface.

---

## How It Works

The application follows a multi-stage pipeline to achieve efficient and accurate gesture recognition:

1.  **Video Input**: The app can use a local OpenCV window (`gesture_app_cleaned.py`) or a browser's webcam feed via Streamlit-WebRTC (`app_streamlit.py`).
2.  **Region of Interest (ROI) Detection**: Instead of scanning the entire frame, the app first uses a pre-trained YOLOv8 model to detect a `person`. This quickly and reliably identifies the main area where a hand is likely to be.
3.  **Landmark Extraction**: The detected ROI is cropped and passed to the MediaPipe Hands model to extract 21 detailed 3D landmarks for the hand.
4.  **Gesture Classification**: The 3D coordinates of the landmarks are flattened and fed into a custom Keras neural network, which classifies the gesture into predefined categories.

This pipeline approach is highly efficient, as the heavy-duty landmark extraction is only performed on a small, relevant section of the video frame.

---

## Features

- **Dual Versions**: Run the app locally in a desktop window or as an interactive web application in your browser.
- **Real-Time Performance**: Optimized pipeline runs smoothly on a standard webcam.
- **Robust Detection**: Uses YOLOv8 to reliably find the hand's location.
- **Extensible**: Easily train the Keras model to recognize your own custom gestures.
- **Well-Documented Code**: Scripts are cleaned and commented for easy understanding and modification.

---

## Setup and Installation

### Prerequisites

- Python 3.9+
- A webcam
- A modern web browser (for the Streamlit version)

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/chmj/app_yolodetect.git
    cd app_yolodetect
    ```

2.  **Create and activate a virtual environment:**
    - **macOS/Linux:**
      ```bash
      python3 -m venv .venv
      source .venv/bin/activate
      ```
    - **Windows:**
      ```bash
      python -m venv .venv
      .\.venv\Scripts\activate
      ```

3.  **Install the required dependencies:**
    The `requirements.txt` file contains all dependencies for both the desktop and web versions.
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

You can run either the local desktop version or the Streamlit web app.

### Option 1: Running the Desktop App

This will open a standard OpenCV window on your desktop to display the webcam feed.

```bash
python3 yolodetect.py
```

Press the `ESC` key to close the application window.

### Option 2: Running the Streamlit Web App

This will launch a local web server and open the application in your browser.

```bash
streamlit run app_streamlit.py
```

Your browser will open a new tab. Click the **"START"** button and grant webcam permissions when prompted.

---

## Customization: Training Your Own Gestures

The included Keras model is a placeholder and is not trained. To recognize your own gestures:

1.  **Collect Landmark Data**: Modify either `yolodetect.py` or `app_streamlit.py` to save the flattened landmark vectors (`landmarks` variable in the code) to a CSV file. Create separate files or use labels for each gesture you want to train.

2.  **Train the Keras Model**: Create a separate Python script to:
    - Load the data from your CSV files.
    - Build the `create_gesture_model()`.
    - Train the model on your landmark data.
    - Save the trained model's weights: `model.save_weights('my_gesture_model.h5')`.

3.  **Load Your Trained Model**: In `yolodetect.py` or `app_streamlit.py`, uncomment the following line and update the path to your saved weights file:
    ```python
    # gesture_model.load_weights('my_gesture_model.h5')
    ```

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Author

- **Charles Majola**
- GitHub: [@chmj](https://github.com/chmj)
