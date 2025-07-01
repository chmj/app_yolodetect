# Real-Time Gesture Recognition App


A real-time computer vision application that detects and classifies hand gestures from a webcam feed using a powerful combination of deep learning and machine learning models.

This project uses:
- **YOLOv8** for initial, robust hand region detection.
- **MediaPipe** for accurate, high-fidelity hand landmark extraction.
- **TensorFlow/Keras** for custom gesture classification based on landmark data.
- **OpenCV** for video capture and rendering.

---

## How It Works

The application follows a multi-stage pipeline to achieve efficient and accurate gesture recognition:

1.  **Region of Interest (ROI) Detection**: Instead of scanning the entire frame for a hand, the app first uses a pre-trained YOLOv8 model to detect a `person`. This quickly and reliably identifies the main area where a hand is likely to be.
2.  **Landmark Extraction**: The detected person ROI is cropped and passed to the MediaPipe Hands model. MediaPipe, optimized for this task, processes the smaller image to extract 21 detailed 3D landmarks for the hand.
3.  **Gesture Classification**: The 3D coordinates of the 21 landmarks are flattened into a single vector (63 data points) and fed into a custom Keras neural network. This model then classifies the gesture into predefined categories.

This pipeline approach is highly efficient, as the heavy-duty landmark extraction is only performed on a small, relevant section of the video frame.

---

## Features

- **Real-Time Performance**: Optimized pipeline runs smoothly on a standard webcam.
- **Robust Detection**: Uses YOLOv8 to reliably find the hand's location.
- **High-Fidelity Landmarks**: Leverages Google's MediaPipe for precise keypoint detection.
- **Extensible**: Easily train the Keras model to recognize your own custom gestures.
- **Well-Documented Code**: The script is cleaned and commented for easy understanding and modification.

---

## Setup and Installation

### Prerequisites

- Python 3.9+
- A webcam

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/chmj/app_yolodetect.git](https://github.com/chmj/app_yolodetect.git)
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
    A `requirements.txt` file is provided for easy installation.
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

To run the application, simply execute the main Python script from your terminal:

```bash
python3 yolodetect.py
```

A window will open showing your webcam feed. When you show your hand, a blue box should identify the region, and the detected gesture will be displayed on the screen.

Press the `ESC` key to close the application.

## Customization: Training Your Own Gestures

The included Keras model is a placeholder and is not trained. To recognize your own gestures, you will need to:

1.  **Collect Landmark Data**: Modify the script to save the flattened landmark vectors (`landmarks` variable in the code) for each frame to a CSV file. Create separate files or use labels for each gesture you want to train (e.g., 'thumbs_up.csv', 'fist.csv').

2.  **Train the Keras Model**: Create a separate Python script to:
    - Load the data from your CSV files.
    - Build the `create_gesture_model()`.
    - Train the model on your landmark data.
    - Save the trained model's weights: `model.save_weights('my_gesture_model.h5')`.

3.  **Load Your Trained Model**: In `yolodetect.py`, uncomment the following line and update the path to your saved weights file:
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
