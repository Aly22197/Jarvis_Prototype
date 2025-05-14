# Smart Virtual Assistant with Vision-Based Controls

A computer vision-based desktop assistant that can perform tasks using hand gestures, facial recognition, and optical character recognition (OCR).

## Features

- **User Authentication:** Identify users via facial recognition using DeepFace
- **Gesture Recognition:** Control your system with hand gestures using MediaPipe Hands
- **OCR Integration:** Extract text from physical documents using Tesseract OCR
- **Camera Calibration:** Improve detection accuracy with OpenCV camera calibration
- **Voice Feedback:** Text-to-speech capabilities using pyttsx3
- **Drawing Mode:** Control a digital canvas with finger gestures
- **Multi-User Support:** Personalized settings and attendance logging

## Supported Gestures

| Gesture | Action | Description |
|---------|--------|-------------|
| Point | Move Cursor | Point index finger to control cursor position |
| Thumbs Up | Volume Up | Only thumb extended upward increases volume |
| Thumbs Down | Volume Down | Only thumb extended downward decreases volume |
| Closed Fist | Play Media | Closed fist (no fingers extended) to play media |
| Open Hand | Stop Media | All fingers extended (open hand) to stop media |
| Pinky Only | Next Slide/Track | Only pinky finger extended to go to next item |
| Spider-Man | Previous Slide/Track | Index and pinky fingers extended to go to previous item |
| Pinch | Click | Thumb and index finger pinched to perform mouse click |
| OCR | OCR Mode | Thumb, index, and middle fingers extended to toggle OCR mode |
| Draw | Drawing Mode | Index and middle fingers extended close together to toggle drawing mode |

## Requirements

### Software Requirements

- Python 3.8 or higher
- OpenCV 4.5 or higher
- Tesseract OCR
- MediaPipe
- DeepFace
- PyAutoGUI
- pyttsx3
- NumPy

### Hardware Requirements

- Webcam
- Microphone (optional, for voice commands)
- 8 GB RAM or higher (recommended)
- 2 GHz dual-core processor or better

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/smart-virtual-assistant.git
   cd smart-virtual-assistant
   ```

2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

3. Install Tesseract OCR:
   - **Windows**: Download and install from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - **Mac**: `brew install tesseract`
   - **Linux**: `sudo apt install tesseract-ocr`

4. Update the Tesseract path in `ocr_module.py` if necessary.

## Usage

1. Run the main application:
   ```
   python main.py
   ```

2. First-time setup:
   - The system will ask you to register your face for authentication
   - Follow the on-screen instructions to complete the setup

3. Camera Calibration:
   - Press 'c' to enter calibration mode
   - Hold a chessboard pattern in different orientations
   - Press 'c' to capture calibration frames (at least 10 recommended)
   - Press 'q' to finish calibration

4. Using the assistant:
   - After successful authentication, you can use hand gestures to control your system
   - The recognized gesture and current status will be displayed on screen
   - Use the supported gestures to control various functions

5. OCR Mode:
   - Make the OCR gesture (thumb, index, and middle fingers extended)
   - Place a document in the highlighted region
   - The recognized text will be displayed on screen
   - Press 's' to save the OCR results

6. Drawing Mode:
   - Make the Draw gesture (index and middle fingers extended)
   - Use your index finger to draw on the canvas
   - Make the Draw gesture again to exit drawing mode

## Keyboard Shortcuts

- `q`: Quit the application
- `c`: Start camera calibration (or clear canvas in Drawing mode)
- `s`: Save OCR text (when in OCR mode)
- `a`: Add current user to authorized users
- `h`: Open gesture guide
- `g`: Cycle through gestures in guide mode

## Folder Structure

```
smart_assistant/
│
├── main.py               # Main application entry point
├── requirements.txt      # Python dependencies
├── README.md             # This documentation
│
├── utils/                # Utility modules
│   ├── face_recognition.py     # Face recognition module using DeepFace
│   ├── gesture_recognition.py  # Gesture recognition using MediaPipe
│   ├── ocr_module.py           # OCR module using Tesseract
│   ├── camera_calibration.py   # Camera calibration using OpenCV
│   ├── voice_feedback.py       # Text-to-speech module
│   └── application_control.py  # System control functions
│
├── models/               # Model files (if any)
│
└── data/                 # Data storage
    ├── faces/            # Stored face images
    ├── calibration/      # Camera calibration data
    ├── ocr_results/      # Saved OCR results
    └── voice_logs/       # Voice feedback logs
```

## Troubleshooting

- **Face recognition issues:** Ensure good lighting conditions and try recalibrating your face
- **Gesture recognition problems:** Make sure your hand is clearly visible and within frame
- **OCR accuracy issues:** Improve lighting, ensure document is flat and well-positioned
- **Performance issues:** Try reducing the camera resolution in main.py

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV team for computer vision capabilities
- Google MediaPipe for hand tracking
- Tesseract OCR for text recognition
- DeepFace for facial recognition 