# License Plate Detection System

A multi-model ensemble system for detecting and classifying license plates in vehicle images. This project uses three independent models (YOLOv8, ResNet50, and MobileNetV2) to provide a robust identification system based on a majority-vote verdict.

## Features
- **YOLOv8 Object Detection**: Precisely locates the license plate and draws bounding boxes.
- **ResNet50 & MobileNetV2**: Transfer learning models used for redundant classification and feature extraction.
- **Ensemble Verdict**: Combines results from all three models for higher-accuracy detection.
- **Web Interface**: Clean, dark-themed UI for easy image uploads and real-time inference visualization.

## Prerequisites
- Python 3.10 or higher
- [Optional] CUDA-enabled GPU for faster inference

## Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Adhithya-Kanaksabai/License-Plate-Detection.git
    cd License-Plate-Detection
    ```

2.  **Create a Virtual Environment (Recommended)**
    ```bash
    python -m venv .venv
    ```

3.  **Activate the Virtual Environment**
    - **Windows**:
      ```bash
      .venv\Scripts\activate
      ```
    - **Mac/Linux**:
      ```bash
      source .venv/bin/activate
      ```

4.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

1.  Start the Flask server:
    ```bash
    python app.py
    ```

2.  Open your browser and navigate to:
    ```
    http://localhost:5000
    ```

3.  Upload a clear image of a vehicle to see the detection and classification results.

## Project Structure
- `app.py`: Flask backend and model inference logic.
- `models/`: Contains pre-trained weights (e.g., `license_plate_detector.pt`).
- `templates/`: HTML frontend.
- `uploads/`: Temporary storage for uploaded and annotated images.
- `requirements.txt`: Python package dependencies.

## Technical Details
The system utilizes a **Majority Vote Ensemble**:
- If at least 2 out of 3 models detect a license plate, the final verdict is "Detected."
- YOLOv8 provides the visual bounding box, while ResNet and MobileNet act as independent verifiers.

## Acknowledgments
- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8.
- [PyTorch](https://pytorch.org/) for the classification and transfer learning frameworks.
