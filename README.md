# Audio Capture for WhisperLive

This is a simple program that captures audio from your microphone and transcribes it in real-time using WhisperLive.

## Prerequisites

- Python 3.7+
- WhisperLive installed
- PyAudio installed (required for audio capture)

## Setup

1. Make sure you have WhisperLive set up correctly:
   ```bash
   pip install whisper-live
   ```

2. You may need to install PyAudio:
   ```bash
   # On macOS
   brew install portaudio
   pip install pyaudio
   
   # On Ubuntu/Debian
   sudo apt-get install python3-pyaudio
   
   # On Windows
   pip install pyaudio
   ```

3. Install numpy:
   ```bash
   pip install numpy
   ```

## Running the WhisperLive Server

Before running this client, you need to start the WhisperLive server:

```bash
# From the WhisperLive directory
python run_server.py --port 9090 --backend faster_whisper
```

## Running the Audio Capture Client

After the server is running, run this program:

```bash
python main.py
```

## Configuration

You can modify the client configuration in `main.py` to change:

- The server host and port
- The language for transcription
- The model size (tiny, base, small, medium, large)
- Whether to use Voice Activity Detection (VAD)
- And more

## Output

The transcription will be displayed in real-time in the console, and an SRT file with the transcription will be saved in the current directory as `output.srt`.

## Stopping the Capture

Press `Ctrl+C` to stop the recording and exit the program. 