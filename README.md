# Voice & Gesture Controlled Python Code Editor

This application provides a hands-free Python coding experience, allowing users to write, edit, and run Python code using voice commands and hand gestures. It features a simple GUI with a code editor, a camera feed to display gesture input, and a status bar for feedback.

## Features

*   **Voice Control:**
    *   Dictate Python code directly.
    *   Execute common commands like "run code", "clear editor", "save file".
    *   Insert code snippets for "print", "if statement", "for loop", "while loop", "function", "variable".
    *   Control cursor: "next line", "previous line".
    *   Manage indentation: "indent", "unindent", "tab", "untab".
    *   Delete lines: "delete line".
*   **Gesture Control:**
    *   **Scroll Up:** One finger up.
    *   **Scroll Down:** Peace sign (index and middle finger).
    *   **Indent:** Thorns gesture (all fingers up except thumb).
    *   **Unindent:** Closed fist.
*   **Real-time Feedback:**
    *   Camera view shows live gesture input.
    *   Status bar displays recognized commands and application state.
*   **Code Editor:**
    *   Basic Python syntax highlighting.
    *   Auto-indentation for control statements.
    *   Cursor highlight for better visibility.
    *   Includes a default Python code template.

## Prerequisites

*   Python 3.7+
*   A microphone for voice input.
*   A webcam for gesture recognition.

## Setup

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.  **Install required Python packages:**
    It's highly recommended to use a virtual environment.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On macOS/Linux
    # venv\\\\Scripts\\\\activate    # On Windows
    ```
    Then install the packages:
    ```bash
    pip install opencv-python mediapipe PyQt5 SpeechRecognition
    ```
    *   `opencv-python`: For camera access and image processing.
    *   `mediapipe`: For hand gesture recognition.
    *   `PyQt5`: For the graphical user interface.
    *   `SpeechRecognition`: For voice command processing (uses Google Web Speech API by default, requires internet).

3.  **PortAudio (if not already installed, for PyAudio which SpeechRecognition might use):**
    *   **On macOS:**
        ```bash
        brew install portaudio
        pip install pyaudio
        ```
    *   **On Ubuntu/Debian:**
        ```bash
        sudo apt-get install portaudio19-dev python3-pyaudio
        pip install pyaudio
        ```
    *   **On Windows:** `pip install pyaudio` might work directly. If not, you may need to find a precompiled wheel (.whl) file for your Python version and architecture.

## Running the Application

1.  Ensure your microphone and webcam are connected and enabled.
2.  Navigate to the project directory in your terminal.
3.  If you created a virtual environment, activate it:
    ```bash
    source venv/bin/activate # macOS/Linux
    # venv\\\\Scripts\\\\activate   # Windows
    ```
4.  Run the main script:
    ```bash
    python main.py
    ```

## How to Use

*   **Voice Commands:** Speak clearly into your microphone. The application will transcribe your speech and attempt to match it with predefined commands or insert it as code.
    *   To type code: Simply say the code you want to write, e.g., "print hello world", "x equals 5".
    *   For commands: "run code", "clear editor", "indent", etc.
*   **Gestures:** Perform gestures in view of the webcam.
    *   **Scroll Up/Down:** Use one finger or a peace sign.
    *   **Indent/Unindent:** Use the thorns gesture or a fist.
*   The editor starts with a minimal Python template.
*   The application will attempt to auto-indent after control statements like `for`, `if`, `while`, `def`.

## Known Issues & Limitations

*   Gesture recognition can be sensitive to lighting and background.
*   Voice recognition accuracy depends on microphone quality and ambient noise. It uses an online service (Google Web Speech API) by default.
*   The set of recognized voice commands and gestures is currently fixed.
*   Error handling for code execution is basic.

## Future Enhancements (Ideas)

*   Offline voice recognition.
*   Customizable voice commands and gesture mappings.
*   More advanced code editing features (e.g., autocomplete, debugging).
*   Support for different programming languages. 