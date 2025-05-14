#!/usr/bin/env python3
import sys
import os
import time
import logging
import cv2
import mediapipe as mp
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QTextEdit, QFrame, 
                            QMessageBox, QGridLayout, QSplitter, QInputDialog, QDialog, 
                            QButtonGroup, QRadioButton, QLineEdit, QDialogButtonBox)
from PyQt5.QtCore import (Qt, QThread, pyqtSignal, pyqtSlot, QTimer, QSize, 
                          QPoint, QMetaObject, Q_ARG, QObject)
from PyQt5.QtGui import (QImage, QPixmap, QFont, QColor, QTextCursor, 
                        QTextCharFormat, QTextFormat, QSyntaxHighlighter)
import threading
import speech_recognition as sr
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Changed level to INFO as DEBUG is verbose for console
    format='%(asctime)s - %(levelname)s - %(message)s', # Simplified format for console
    handlers=[
        logging.StreamHandler() # Keep only console output
        # logging.FileHandler('app_debug.log') # Removed file handler
    ]
)
logger = logging.getLogger(__name__)

# Program blocks that can be added through gestures
PROGRAM_BLOCKS = {
    "print": 'print("")',
    "variable": 'var = 0',
    "if": 'if condition == True:\n    pass',
    "for": 'for i in range(10):\n    ',  # Add space for indentation
    "while": 'while condition:\n    ',   # Add space for indentation
    "function": 'def function_name():\n    ',  # Add space for indentation
    "increment": 'var += 1'
}

# Control statements that require indentation
CONTROL_STATEMENTS = ["for", "if", "while", "def", "function"]

class Signals(QObject):
    """Signal definitions for thread-safe UI updates."""
    update_frame = pyqtSignal(QImage)
    gesture_detected = pyqtSignal(str)
    speech_detected = pyqtSignal(str)
    status_updated = pyqtSignal(str)
    highlight_current_line = pyqtSignal() # Add signal for highlighting current line

class GestureRecognitionThread(QThread):
    """Thread for performing gesture recognition."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # Signals must be defined as class variables
        self.update_frame = pyqtSignal(QImage)
        self.gesture_detected = pyqtSignal(str)
        
        # Gesture tracking state
        self.running = True
        self.gestures = { # Defines mapping from simple finger counts/states to command names
            "two_fingers": "if",
            "three_fingers": "for",
            "fist": "execute",
            "hand_swipe": "indent",
            "hand_swipe_left": "unindent"
        }
        self.last_gesture = None
        self.gesture_cooldown = 1.0  # Prevents rapid firing of the same gesture
        self.last_gesture_time = 0
        
        # MediaPipe setup for hand tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False, # Process video stream
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils # For drawing landmarks
        
    def stop(self):
        """Stop the thread execution gracefully."""
        self.running = False
        self.wait() # Wait for the run loop to finish
    
    def detect_gesture(self, hand_landmarks):
        """Detects specific gestures based on finger extension and applies a cooldown."""
        # Get fingertip y-positions relative to base
        fingertips = [hand_landmarks.landmark[i] for i in [4, 8, 12, 16, 20]]  # Thumb, index, middle, ring, pinky
        finger_bases = [hand_landmarks.landmark[i] for i in [2, 5, 9, 13, 17]]  # Base of each finger
        
        # Check if fingers are extended
        fingers_up = []
        for i in range(5):
            if i == 0:  # Thumb has different logic
                fingers_up.append(fingertips[i].x < finger_bases[i].x)
            else:
                fingers_up.append(fingertips[i].y < finger_bases[i].y)
        
        # Detect specific gestures
        current_time = time.time()
        
        # Simple gesture detection based on fingers up
        if current_time - self.last_gesture_time > self.gesture_cooldown: # Check if cooldown has passed
            # One finger up - move up
            if sum(fingers_up) == 1 and fingers_up[1]: # Index finger
                self.last_gesture_time = current_time
                return "scroll_up"
            
            # Peace sign (index and middle) - move down
            elif sum(fingers_up) == 2 and fingers_up[1] and fingers_up[2] and not fingers_up[0] and not fingers_up[3] and not fingers_up[4]:
                self.last_gesture_time = current_time
                return "scroll_down"
            
            # Thorns (all fingers except thumb) - indent
            elif sum(fingers_up) == 4 and fingers_up[1] and fingers_up[2] and fingers_up[3] and fingers_up[4] and not fingers_up[0]:
                self.last_gesture_time = current_time
                return "thorns"
            
            # Closed fist - unindent
            elif sum(fingers_up) == 0: # No fingers extended
                self.last_gesture_time = current_time
                return "fist"
        
        return None # No gesture detected or cooldown active

    def run(self):
        """Main loop for capturing video, processing hand landmarks, and emitting gesture signals."""
        cap = cv2.VideoCapture(0) # Initialize video capture
        
        # Set camera resolution to balance performance and accuracy
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while self.running:
            try:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Flip frame horizontally for a more intuitive mirror-like interaction
                frame = cv2.flip(frame, 1)
                
                # Convert to RGB for MediaPipe processing
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame) # Process frame for hand landmarks
                
                # Draw hand landmarks on the frame for visual feedback
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(
                            frame, 
                            hand_landmarks, 
                            self.mp_hands.HAND_CONNECTIONS
                        )
                        
                        # Detect gesture from landmarks
                        gesture = self.detect_gesture(hand_landmarks)
                        if gesture:
                            # Emit signal if a gesture is recognized
                            if hasattr(self.parent(), 'signals'):
                                self.parent().signals.gesture_detected.emit(gesture)
                
                # Convert the OpenCV frame (BGR) to QImage (RGB) for display in PyQt
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                
                # Emit the processed frame for UI update
                if hasattr(self.parent(), 'signals'):
                    self.parent().signals.update_frame.emit(qt_image)
                
                # Brief sleep to reduce CPU load from constant processing
                time.sleep(0.01)
                
            except Exception as e:
                pass
        
        # Release resources
        cap.release()

class SpeechRecognitionThread(QThread):
    """Thread for performing speech recognition."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = True
        
        # Defines a mapping of spoken phrases to internal command identifiers
        self.commands = {
            "execute": "execute",
            "clear": "clear",
            "print": "print",
            "if statement": "if",
            "for loop": "for",
            "while loop": "while",
            "function": "function",
            "indent": "indent",
            "unindent": "unindent",
            "variable": "variable", 
            "next line": "next_line",
            "move down": "next_line",
            "move up": "previous_line",
            "delete line": "delete_line",
            "tab": "indent",
            "untab": "unindent",
            "increment": "increment",
            "save": "save_file",
            "save file": "save_file"
        }
    
    def stop(self):
        """Stop the thread execution gracefully."""
        self.running = False
        self.wait() # Wait for the run loop to finish
    
    def run(self):
        """Main loop for listening to microphone input, recognizing speech, and emitting command signals."""
        recognizer = sr.Recognizer()
        
        while self.running:
            try:
                with sr.Microphone() as source:
                    # Adjust for ambient noise to improve recognition accuracy
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    # Listen for speech input with timeouts to prevent blocking
                    audio = recognizer.listen(source, timeout=2, phrase_time_limit=5)
                
                try:
                    # Use Google Web Speech API for recognition
                    text = recognizer.recognize_google(audio).lower()
                    
                    # Custom handling for "variable" command to extract name and value
                    if "variable" in text:
                        words = text.split()
                        if len(words) >= 2:
                            var_index = words.index("variable")
                            if var_index + 1 < len(words):
                                var_name = words[var_index + 1]
                                var_value = "0"
                                
                                # Look for a number after the variable name
                                for i in range(var_index + 2, len(words)):
                                    if words[i].isdigit():
                                        var_value = words[i]
                                        break
                                
                                # Send custom variable command in format "variable:name:value"
                                if hasattr(self.parent(), 'signals'):
                                    self.parent().signals.speech_detected.emit(f"variable:{var_name}:{var_value}")
                                continue # Skip further command checks
                    
                    # Custom handling for "print" command to extract content to print
                    elif "print" in text and text != "print":
                        content = text.replace("print", "", 1).strip()
                        if content:
                            # Send custom print command in format "print:content"
                            if hasattr(self.parent(), 'signals'):
                                self.parent().signals.speech_detected.emit(f"print:{content}")
                            continue # Skip further command checks
                    
                    # Check for other predefined commands
                    command_found = False
                    for command, action in self.commands.items():
                        if command in text:
                            # Emit signal through parent
                            if hasattr(self.parent(), 'signals'):
                                self.parent().signals.speech_detected.emit(action)
                                command_found = True
                            break
                
                except sr.UnknownValueError:
                    pass
                except sr.RequestError as e:
                    pass
                
                # Sleep to reduce CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                pass
        
        logger.debug("Speech recognition thread stopped")

class CodeBlock(QFrame):
    """A draggable code block widget."""
    
    def __init__(self, code_type, code_text, parent=None):
        super().__init__(parent)
        self.code_type = code_type
        self.code_text = code_text
        
        self.setStyleSheet("""
            QFrame {
                background-color: #2D2D2D;
                border: 2px solid #3E3E3E;
                border-radius: 5px;
                padding: 5px;
            }
            QLabel {
                color: #D4D4D4;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 14px;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # Title label with block type
        title = QLabel(code_type.title())
        title.setStyleSheet("font-weight: bold; color: #569CD6;")
        layout.addWidget(title)
        
        # Code content
        code_label = QLabel(code_text)
        code_label.setStyleSheet("color: #CE9178;")
        layout.addWidget(code_label)
        
        self.setLayout(layout)
        self.setFixedSize(QSize(200, 100))

class CodeEditor(QTextEdit):
    """Custom code editor with basic syntax highlighting."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Set up a monospaced font suitable for coding
        font = QFont()
        font.setFamily("Courier")
        font.setFixedPitch(True)
        font.setPointSize(11)
        self.setFont(font)
        
        # Set up the custom Python syntax highlighter
        self.highlighter = PythonHighlighter(self.document())
        
        # Set tab width to a standard 4 spaces for Python
        self.setTabStopDistance(40) # 4 spaces * 10 (typical point size)
        
        # Connect cursor position changes to the highlighting function
        self.cursorPositionChanged.connect(self.highlight_current_line)
        
        # Current variable for increment
        self.current_variable = "var"
        
    def highlight_current_line(self):
        """Highlights the entire line where the text cursor is currently located."""
        extraSelections = []
        
        if not self.isReadOnly(): # Only highlight if editable
            selection = QTextEdit.ExtraSelection()
            
            # Define the background color for the current line highlight
            lineColor = QColor(60, 80, 100)
            selection.format.setBackground(lineColor)
            selection.format.setProperty(QTextFormat.FullWidthSelection, True) # Ensure full line width
            
            selection.cursor = self.textCursor()
            selection.cursor.clearSelection() # Important to avoid interfering with user selections
            
            extraSelections.append(selection)
        
        self.setExtraSelections(extraSelections) # Apply the highlight
    
    def move_cursor_up(self):
        """Move cursor up one line and update highlighting."""
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.Up)
        self.setTextCursor(cursor)
        self.highlight_current_line() # Re-apply highlight after cursor move
    
    def move_cursor_down(self):
        """Move cursor down one line and update highlighting."""
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.Down)
        self.setTextCursor(cursor)
        self.highlight_current_line() # Re-apply highlight after cursor move
    
    def move_cursor_to_start(self):
        """Move cursor to the start of the document and update highlighting."""
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.Start)
        self.setTextCursor(cursor)
        self.highlight_current_line() # Re-apply highlight after cursor move
    
    def move_cursor_to_end(self):
        """Move cursor to the end of the document and update highlighting."""
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.setTextCursor(cursor)
        self.highlight_current_line() # Re-apply highlight after cursor move

class PythonHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for Python code using regular expressions."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.highlighting_rules = [] # List to store (pattern, format) tuples
        
        # Keyword format (e.g., def, for, if)
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor("#569CD6"))  # Blue
        keyword_format.setFontWeight(QFont.Bold)
        keywords = [
            "and", "as", "assert", "break", "class", "continue", "def",
            "del", "elif", "else", "except", "False", "finally", "for",
            "from", "global", "if", "import", "in", "is", "lambda", "None",
            "nonlocal", "not", "or", "pass", "raise", "return", "True",
            "try", "while", "with", "yield"
        ]
        
        for word in keywords:
            pattern = f"\\b{word}\\b"
            self.highlighting_rules.append((re.compile(pattern), keyword_format))
        
        # String format (e.g., "hello", 'world')
        string_format = QTextCharFormat()
        string_format.setForeground(QColor("#CE9178"))  # Orange
        self.highlighting_rules.append((re.compile('".*?"'), string_format)) # Double quotes
        self.highlighting_rules.append((re.compile("'.*?'"), string_format)) # Single quotes
        
        # Comment format (e.g., # This is a comment)
        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor("#6A9955"))  # Green
        self.highlighting_rules.append((re.compile("#.*"), comment_format))
        
        # Function call/definition format (e.g., my_function(), def my_func)
        function_format = QTextCharFormat()
        function_format.setForeground(QColor("#DCDCAA"))  # Yellow
        self.highlighting_rules.append((re.compile("\\b[A-Za-z0-9_]+(?=\\()"), function_format)) # Matches names followed by (
        
        # Number format (e.g., 123, 0.5)
        number_format = QTextCharFormat()
        number_format.setForeground(QColor("#B5CEA8"))  # Light green
        self.highlighting_rules.append((re.compile("\\b\\d+\\b"), number_format))
    
    def highlightBlock(self, text):
        """Apply highlighting rules to the given block of text (typically one line)."""
        for pattern, format_rule in self.highlighting_rules:
            # Iterate over all matches for the current rule's pattern in the text
            for match in pattern.finditer(text):
                start = match.start()
                length = match.end() - start
                self.setFormat(start, length, format_rule) # Apply the format

class GestureProgrammingApp(QMainWindow):
    """Main application window, orchestrating UI, threads, and logic."""
    
    def __init__(self):
        super().__init__()
        
        # Initialize signals for thread-safe UI updates from background threads
        self.signals = Signals()
        
        # Setup the main UI components
        self.init_ui()
        
        # Initialize state for managing code blocks and indentation
        self.code_blocks = []
        self.current_indent = 0
        
        # Connect signals from background threads and UI elements to handler methods
        self.connect_signals()
        
        # Start the gesture and speech recognition threads
        self.start_background_threads()
        
        # Display a welcome message and code template after a short delay
        QTimer.singleShot(1000, self.add_welcome_message)
        
        # Stores the name of the last variable created, for the "increment" command
        self.last_variable = "var"
    
    def init_ui(self):
        """Initialize the main user interface layout and components."""
        self.setWindowTitle("Gesture Programming Environment")
        self.setGeometry(100, 100, 1200, 800)
        
        # Apply a dark theme stylesheet for the application
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1E1E1E;
            }
            QWidget {
                font-family: 'Consolas', 'Monaco', monospace;
                color: #D4D4D4;
            }
            QLabel {
                color: #D4D4D4;
                font-size: 14px;
            }
            QTextEdit {
                background-color: #1E1E1E;
                color: #D4D4D4;
                border: 1px solid #3E3E3E;
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
                line-height: 1.5;
                selection-background-color: #264F78;
            }
            QPushButton {
                background-color: #2D2D2D;
                color: #D4D4D4;
                border: 2px solid #3E3E3E;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #3E3E3E;
                border-color: #569CD6;
            }
            QSplitter::handle {
                background-color: #3E3E3E;
            }
            QSplitter::handle:horizontal {
                width: 2px;
            }
        """)
        
        # Create central widget and layouts
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Top bar layout for title and status label
        top_bar = QHBoxLayout()
        title = QLabel("Gesture Programming Environment")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        top_bar.addWidget(title)
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #808080; font-style: italic;")
        top_bar.addStretch()
        top_bar.addWidget(self.status_label)
        
        main_layout.addLayout(top_bar)
        
        # Create a QSplitter to allow resizable sections for camera/instructions and editor/output
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel: Camera feed and instructions
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Camera view label, displays processed frames from GestureRecognitionThread
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("background-color: #000000; border: 1px solid #3E3E3E;")
        left_layout.addWidget(self.camera_label)
        
        # Instructions panel displaying available gestures and voice commands
        instructions = QTextEdit()
        instructions.setReadOnly(True)
        instructions.setHtml("""
            <h3 style="color: #569CD6;">Gesture Controls:</h3>
            <ul>
                <li><b>One finger up</b> - Move cursor up one line</li>
                <li><b>Peace sign</b> - Move cursor down one line</li>
                <li><b>Thorns (all fingers except thumb)</b> - Indent code</li>
                <li><b>Fist</b> - Unindent code</li>
            </ul>
            
            <h3 style="color: #569CD6;">Voice Commands (for code snippets):</h3>
            <ul>
                <li><b>"Execute"</b> - Run the code</li>
                <li><b>"Clear"</b> - Clear the editor and restore template</li>
                <li><b>"Save" or "Save file"</b> - Save to file</li>
                <li><b>"Print"</b> - Add print statement</li>
                <li><b>"Variable [name] [value]"</b> - Create a variable</li>
                <li><b>"If statement"</b> - Add if statement</li>
                <li><b>"For loop"</b> - Add for loop</li>
                <li><b>"While loop"</b> - Add while loop</li>
                <li><b>"Function"</b> - Add function definition</li>
                <li><b>"Increment"</b> - Add increment statement</li>
                <li><b>"Next line" or "Move down"</b> - Move cursor down</li>
                <li><b>"Move up"</b> - Move cursor up</li>
                <li><b>"Delete line"</b> - Delete current line</li>
                <li><b>"Tab" or "Indent"</b> - Increase indentation</li>
                <li><b>"Untab" or "Unindent"</b> - Decrease indentation</li>
            </ul>
        """)
        instructions.setMaximumHeight(200)
        left_layout.addWidget(instructions)
        
        left_widget.setLayout(left_layout)
        splitter.addWidget(left_widget)
        
        # Right panel: Code editor, control buttons, and output console
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Layout for control buttons (Execute, Clear, Save)
        control_layout = QHBoxLayout()
        
        # Execute button
        execute_button = QPushButton("Execute")
        execute_button.clicked.connect(self.execute_code)
        control_layout.addWidget(execute_button)
        
        # Clear button
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self.clear_editor)
        control_layout.addWidget(clear_button)
        
        # Save button
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.file_save)
        control_layout.addWidget(save_button)
        
        right_layout.addLayout(control_layout)
        
        # Code editor widget where users write/dictate Python code
        self.code_editor = CodeEditor()
        self.code_editor.setStyleSheet("""
            QTextEdit {
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 16px;
                line-height: 1.5;
                background-color: #1E1E1E;
                color: #D4D4D4;
            }
        """)
        self.code_editor.setPlaceholderText("Your code will appear here...")
        right_layout.addWidget(self.code_editor)
        
        # Output console to display results of code execution or errors
        self.output_console = QTextEdit()
        self.output_console.setReadOnly(True)
        self.output_console.setStyleSheet("""
            QTextEdit {
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 14px;
                background-color: #0F0F0F;
                color: #8FE687;
                border-top: 2px solid #3E3E3E;
            }
        """)
        self.output_console.setPlaceholderText("Output will appear here...")
        self.output_console.setMaximumHeight(200)
        right_layout.addWidget(self.output_console)
        
        right_widget.setLayout(right_layout)
        splitter.addWidget(right_widget)
        
        # Set initial relative sizes for the splitter panels
        splitter.setSizes([400, 800])
        
        main_layout.addWidget(splitter)
    
    def connect_signals(self):
        """Connect signals from various components to their respective slot handlers."""
        # Connect gesture thread signals for UI updates
        self.signals.update_frame.connect(self.update_camera_view, Qt.QueuedConnection)
        self.signals.gesture_detected.connect(self.handle_gesture, Qt.QueuedConnection)
        # Connect speech thread signal
        self.signals.speech_detected.connect(self.handle_speech, Qt.QueuedConnection)
        # Connect status update signal (can be emitted from various parts)
        self.signals.status_updated.connect(self.update_status, Qt.QueuedConnection)
    
    def start_background_threads(self):
        """Initialize and start the gesture and speech recognition background threads."""
        # Gesture recognition thread
        self.gesture_thread = GestureRecognitionThread(self)
        self.gesture_thread.start()
        
        # Speech recognition thread
        self.speech_thread = SpeechRecognitionThread(self)
        self.speech_thread.start()
    
    @pyqtSlot(QImage)
    def update_camera_view(self, image):
        """Update the camera view QLabel with the new QImage frame from the gesture thread."""
        pixmap = QPixmap.fromImage(image)
        # Scale pixmap to fit the label while maintaining aspect ratio
        self.camera_label.setPixmap(pixmap.scaled(
            self.camera_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))
    
    @pyqtSlot(str)
    def handle_gesture(self, gesture):
        """Process a recognized gesture string from the gesture thread."""
        try:
            self.process_gesture(gesture) # Internal method to map gesture to action
            self.signals.status_updated.emit(f"Gesture detected: {gesture}")
        except Exception as e:
            logger.error(f"Error handling gesture '{gesture}': {e}") # Log errors
    
    @pyqtSlot(str)
    def handle_speech(self, command):
        """Process a recognized speech command string from the speech thread."""
        try:
            # Handle commands that include parameters (e.g., "variable:name:value", "print:content")
            if ":" in command:
                parts = command.split(":", 1)
                cmd_type = parts[0]
                cmd_value = parts[1]
                
                if cmd_type == "variable":
                    # Handles "variable:var_name:var_value" command
                    var_parts = cmd_value.split(":", 1)
                    var_name = var_parts[0]
                    var_value = var_parts[1] if len(var_parts) > 1 else "0" # Default value if not provided
                    code = f"{var_name} = {var_value}"
                    
                    cursor = self.code_editor.textCursor()
                    if cursor.positionInBlock() > 0: # Add newline if not at start of line
                        cursor.insertText("\n")
                    indentation = "    " * self.current_indent
                    cursor.insertText(f"{indentation}{code}")
                    # Store the variable name for potential use with "increment" command
                    self.last_variable = var_name
                    return # Command processed
                
                elif cmd_type == "print":
                    # Handles "print:content" command
                    content = cmd_value.strip()
                    # If content is a valid Python identifier, print it as a variable, otherwise as a string
                    if content.isidentifier():
                        code = f"print({content})"
                    else:
                        code = f"print(\"{content}\")"
                    cursor = self.code_editor.textCursor()
                    if cursor.positionInBlock() > 0: # Add newline if not at start of line
                        cursor.insertText("\n")
                    indentation = "    " * self.current_indent
                    cursor.insertText(f"{indentation}{code}")
                    return # Command processed
            
            # Handle simple commands without parameters
            if command == "execute":
                self.execute_code()
            elif command == "clear":
                self.clear_editor()
            elif command == "save_file":
                self.file_save()
            elif command == "next_line":
                cursor = self.code_editor.textCursor()
                cursor.movePosition(QTextCursor.Down)
                self.code_editor.setTextCursor(cursor)
            elif command == "previous_line":
                cursor = self.code_editor.textCursor()
                cursor.movePosition(QTextCursor.Up)
                self.code_editor.setTextCursor(cursor)
            elif command == "delete_line":
                cursor = self.code_editor.textCursor()
                cursor.select(QTextCursor.LineUnderCursor)
                cursor.removeSelectedText()
                cursor.deleteChar()  # Also remove the newline character after deleting the line content
                self.code_editor.setTextCursor(cursor)
            elif command == "indent":
                self.change_indent(1)
            elif command == "unindent":
                self.change_indent(-1)
            elif command == "increment":
                # Inserts an increment operation for the last defined variable
                cursor = self.code_editor.textCursor()
                if cursor.positionInBlock() > 0:
                    cursor.insertText("\n")
                indentation = "    " * self.current_indent
                cursor.insertText(f"{indentation}{self.last_variable} += 1")
                self.signals.status_updated.emit(f"Incremented {self.last_variable}")
            else:
                # For control statements (if, for, while, def), use add_control_block for auto-indent
                if command in CONTROL_STATEMENTS:
                    self.add_control_block(command)
                # For other predefined code snippets, use add_code_block
                elif command in PROGRAM_BLOCKS:
                    self.add_code_block(command)
            
            self.signals.status_updated.emit(f"Voice command detected: {command}")
        except Exception as e:
            logger.error(f"Error handling speech command '{command}': {e}") # Log errors
    
    def add_code_block(self, block_type):
        """Add a predefined code block (snippet) to the editor."""
        try:
            if block_type in PROGRAM_BLOCKS:
                code = PROGRAM_BLOCKS[block_type]
                
                # Special handling for "if" statements to allow custom condition input via dialog
                if block_type == "if":
                    dialog = QDialog(self)
                    dialog.setWindowTitle("If Statement Condition")
                    dialog_layout = QVBoxLayout()
                    
                    # Condition options
                    label = QLabel("Choose a condition type or enter custom:")
                    dialog_layout.addWidget(label)
                    
                    # Option buttons
                    condition_options = {
                        "Equality": "value1 == value2",
                        "Greater Than": "value1 > value2",
                        "Less Than": "value1 < value2",
                        "Not Equal": "value1 != value2",
                        "Custom": ""
                    }
                    
                    button_group = QButtonGroup(dialog)
                    selected_condition = condition_options["Equality"]  # Default
                    
                    for i, (name, condition) in enumerate(condition_options.items()):
                        radio = QRadioButton(name)
                        if i == 0:  # Default selection
                            radio.setChecked(True)
                        dialog_layout.addWidget(radio)
                        button_group.addButton(radio, i)
                    
                    # Custom condition input
                    custom_input = QLineEdit()
                    custom_input.setPlaceholderText("Enter custom condition")
                    dialog_layout.addWidget(custom_input)
                    
                    # Button box
                    button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
                    button_box.accepted.connect(dialog.accept)
                    button_box.rejected.connect(dialog.reject)
                    dialog_layout.addWidget(button_box)
                    
                    dialog.setLayout(dialog_layout)
                    
                    # Show dialog and get result; if accepted, construct "if" statement
                    if dialog.exec_() == QDialog.Accepted:
                        selected_id = button_group.checkedId()
                        # Use custom input if "Custom" radio button selected
                        if selected_id == 4: 
                            condition = custom_input.text() if custom_input.text() else "condition == True"
                        else:
                            condition = list(condition_options.values())[selected_id]
                        
                        # "if" is a control statement, so use add_control_block for proper indentation
                        # Note: The original code snippet PROGRAM_BLOCKS["if"] is used by add_control_block,
                        # but the condition part is dynamic here.
                        # We modify the 'code' variable for 'if' before calling add_control_block.
                        # A small refactor might be to pass the condition to add_control_block directly for 'if'.
                        # For now, we'll set PROGRAM_BLOCKS["if"] temporarily if needed, or handle it carefully.
                        
                        # Create the code with the chosen condition
                        # The add_control_block function will then add the 'if' structure with this.
                        # This ensures the : and subsequent auto-indent logic in add_control_block is triggered.
                        PROGRAM_BLOCKS["if"] = f'if {condition}:\\n    ' # Temporarily update for add_control_block
                        self.add_control_block("if")
                        PROGRAM_BLOCKS["if"] = 'if condition == True:\\n    pass' # Restore original template
                        return # Action complete
                    else:
                        return  # User cancelled dialog
                
                # Apply current indentation level to the code block
                if self.current_indent > 0:
                    indented_code = ""
                    for line in code.split("\n"):
                        indented_code += "    " * self.current_indent + line + "\n"
                    code = indented_code.rstrip()
                
                # Get cursor position
                cursor = self.code_editor.textCursor()
                
                # If not at beginning of line, add a newline
                if cursor.positionInBlock() > 0:
                    cursor.insertText("\n")
                
                # Insert code
                cursor.insertText(code)
                
                # Move to the end and add a newline
                if not code.endswith("\n"):
                    cursor.insertText("\n")
                
                # Update cursor
                self.code_editor.setTextCursor(cursor)
                
            else:
                logger.warning(f"Attempted to add unknown code block type: {block_type}")
        except Exception as e:
            logger.error(f"Error adding code block '{block_type}': {e}")
    
    def add_control_block(self, block_type):
        """Add a control flow statement (e.g., for, if, while, def) with auto-indentation for the next line."""
        try:
            if block_type in PROGRAM_BLOCKS:
                code = PROGRAM_BLOCKS[block_type] # Get the code template
                
                cursor = self.code_editor.textCursor()
                
                # If not at beginning of line, add a newline
                if cursor.positionInBlock() > 0:
                    cursor.insertText("\n")
                
                # Apply current indentation to the control statement itself
                indentation = "    " * self.current_indent
                
                # Handle multi-line templates correctly with current indentation
                lines = code.split("\n")
                indented_code_lines = []
                for i, line_content in enumerate(lines):
                    # Apply current indent to all lines of the template.
                    # The template itself usually has a placeholder for the next level of indent (e.g. "    pass")
                    # or ends with a colon expecting the next line to be indented.
                    indented_code_lines.append(indentation + line_content)
                
                final_code = "\n".join(indented_code_lines)
                
                # Insert the control statement
                cursor.insertText(final_code)
                
                # Increase current_indent for subsequent lines automatically after a control statement
                self.current_indent += 1
                
                # Move cursor to the end of the inserted text
                self.code_editor.setTextCursor(cursor)
            
        except Exception as e:
            logger.error(f"Error adding control block '{block_type}': {e}")
    
    def change_indent(self, amount):
        """Change the current indentation level for subsequently added code blocks."""
        self.current_indent = max(0, self.current_indent + amount) # Ensure indent level doesn't go below 0
        self.signals.status_updated.emit(f"Indentation level: {self.current_indent}")
    
    def execute_code(self):
        """Execute the Python code present in the code editor and display output/errors."""
        try:
            code = self.code_editor.toPlainText()
            if not code.strip(): # Check if there's any actual code
                self.output_console.setPlainText("No code to execute")
                return
            
            # Redirect standard output to capture print() statements from the executed code
            import io
            from contextlib import redirect_stdout
            
            output_buffer = io.StringIO()
            try:
                with redirect_stdout(output_buffer): # Capture stdout
                    # Attempt to compile first to catch syntax errors early
                    try:
                        compiled_code = compile(code, "<string>", "exec")
                        exec(compiled_code) # Execute the compiled code
                    except IndentationError as ie:
                        # Handle indentation errors specifically: try to fix and inform user
                        line_number = ie.lineno
                        error_msg = str(ie)
                        fixed_code = self.fix_indentation(code, line_number) # Attempt auto-fix
                        if fixed_code != code: # If fix was made
                            self.code_editor.setPlainText(fixed_code)
                            # Highlight the problematic line
                            cursor = self.code_editor.textCursor()
                            cursor.movePosition(QTextCursor.Start)
                            for _ in range(line_number - 1):
                                cursor.movePosition(QTextCursor.Down)
                            self.code_editor.setTextCursor(cursor)
                            self.code_editor.highlight_current_line()
                            
                            self.output_console.setPlainText(f"Attempted to fix indentation at line {line_number}. Please review and run again.")
                            return
                        else:
                            raise # Re-raise if auto-fix didn't change anything
                    except Exception as e:
                        # Re-raise other compilation/runtime exceptions
                        raise
                
                output = output_buffer.getvalue() # Get captured output
                
                if output:
                    self.output_console.setPlainText(output)
                else:
                    self.output_console.setPlainText("Code executed without output") # Provide feedback if no output
                
            except Exception as e:
                # Handle errors during execution (including syntax errors not caught by compile if any)
                error_message = str(e)
                # If error has line number info, try to highlight the line in editor
                if "line" in error_message and hasattr(e, 'lineno'):
                    line_number = e.lineno
                    cursor = self.code_editor.textCursor()
                    cursor.movePosition(QTextCursor.Start)
                    for _ in range(line_number - 1):
                        if not cursor.movePosition(QTextCursor.Down):
                            break 
                    self.code_editor.setTextCursor(cursor)
                    self.code_editor.highlight_current_line()
                
                self.output_console.setPlainText(f"Error: {error_message}")
                
            self.signals.status_updated.emit("Code executed")
        except Exception as e: # Catch-all for unexpected system errors during execution logic
            self.output_console.setPlainText(f"System Error during execution: {str(e)}")
            logger.error(f"System error executing code: {e}")
    
    def fix_indentation(self, code, error_line_num):
        """Attempts to automatically fix common Python indentation errors."""
        lines = code.split('\n')
        
        if not (0 < error_line_num <= len(lines)): # Validate error line number
            return code

        current_error_line_index = error_line_num - 1
        prev_line_index = current_error_line_index - 1

        # Get indentation of the error line and the line before it
        error_line_text = lines[current_error_line_index]
        leading_spaces_error_line = len(error_line_text) - len(error_line_text.lstrip(' '))
        
        prev_line_text = lines[prev_line_index] if prev_line_index >= 0 else ""
        leading_spaces_prev_line = len(prev_line_text) - len(prev_line_text.lstrip(' '))

        # Case 1: Unexpected indent (current line is more indented than previous, but previous doesn't start a block)
        if leading_spaces_error_line > leading_spaces_prev_line and not prev_line_text.strip().endswith(':'):
            lines[current_error_line_index] = ' ' * leading_spaces_prev_line + error_line_text.lstrip(' ')
        
        # Case 2: Expected indent (previous line ends with ':', current line is not more indented)
        elif prev_line_text.strip().endswith(':'):
            expected_indent = leading_spaces_prev_line + 4
            if not error_line_text.strip(): # If the error line is blank
                lines[current_error_line_index] = ' ' * expected_indent + 'pass' # Add pass
            elif leading_spaces_error_line < expected_indent : # If line has content but not enough indent
                 lines[current_error_line_index] = ' ' * expected_indent + error_line_text.lstrip(' ')

        # General pass: Ensure all non-empty, non-comment lines have indentation that's a multiple of 4
        # This is a heuristic and might not always be correct for complex cases or mixed tabs/spaces.
        for i in range(len(lines)):
            line = lines[i]
            if line.strip() and not line.strip().startswith('#'):
                current_indent = len(line) - len(line.lstrip(' '))
                if current_indent % 4 != 0:
                    # Adjust to the nearest multiple of 4 (could be up or down)
                    # This part might be too aggressive or not desired, depending on coding style.
                    # A simpler approach might be to only fix if it's *less* than expected after a colon.
                    # For now, let's stick to the rounding logic.
                    new_indent = round(current_indent / 4.0) * 4
                    lines[i] = ' ' * new_indent + line.lstrip(' ')
        
        return '\n'.join(lines)

    
    def file_save(self):
        """Save the current content of the code editor to a default file."""
        try:
            filename = "gesture_program.py" # Default filename
            
            with open(filename, "w") as f:
                f.write(self.code_editor.toPlainText()) # Write editor content to file
            
            self.signals.status_updated.emit(f"Program saved to {filename}")
        except Exception as e:
            self.signals.status_updated.emit(f"Error saving program: {str(e)}")
            logger.error(f"Error saving file: {e}")
    
    @pyqtSlot(str)
    def update_status(self, text):
        """Update the text of the status label in the UI."""
        self.status_label.setText(text)
    
    def add_welcome_message(self):
        """Add a default Python code template to the editor when the app starts or editor is cleared."""
        template = """# Main function definition
def main():
    '''Main program entry point'''
    # Your code here
  

# Standard Python idiom for main execution
if __name__ == "__main__":
    main()
"""
        self.code_editor.setPlainText(template)
        
        # Find the "# Your code here" line to position the cursor
        found_cursor_pos = document.find("# Your code here")
        if not found_cursor_pos.isNull(): # Check if the find operation was successful
            cursor.setPosition(found_cursor_pos.position()) # Move to the found position
            cursor.movePosition(QTextCursor.EndOfLine) # Go to the end of that line
            cursor.insertText("\n    ")  # Insert a newline and indent for the user to start typing
            self.code_editor.setTextCursor(cursor)
        
        self.signals.status_updated.emit("Ready") # Update status
        self.code_editor.highlight_current_line() # Ensure current line (cursor position) is highlighted
    
    def keyPressEvent(self, event):
        """Handle specific key press events, e.g., 'Q' to quit."""
        if event.key() == Qt.Key_Q: # Allow quitting with 'Q' key
            self.close()
        else:
            super().keyPressEvent(event) # Pass other key events to base class
    
    def closeEvent(self, event):
        """Handle the window close event to gracefully shut down background threads."""
        try:
            # Stop gesture recognition thread if it exists and is running
            if hasattr(self, 'gesture_thread') and self.gesture_thread.isRunning():
                self.gesture_thread.stop()
            # Stop speech recognition thread if it exists and is running
            if hasattr(self, 'speech_thread') and self.speech_thread.isRunning():
                self.speech_thread.stop()
            
            event.accept() # Accept the close event
        except Exception as e:
            logger.error(f"Error during closeEvent: {e}")
            event.accept() # Still accept event to ensure window closes

    def clear_editor(self):
        """Clear all content from the code editor and restore the default welcome template."""
        self.code_editor.clear()
        self.add_welcome_message() # Restore template
        self.signals.status_updated.emit("Code editor cleared and template restored")
    
    def process_gesture(self, gesture):
        """Maps recognized gestures to specific editor control actions."""
        self.status_label.setText(f"Gesture: {gesture}")
        
        # Defines direct mappings from gesture names to editor actions
        gesture_map = {
            "scroll_down": "cursor_down", 
            "scroll_up": "cursor_up",
            "thorns": "indent", # Mapped to increase current_indent level
            "fist": "unindent", # Mapped to decrease current_indent level
        }
        
        action = gesture_map.get(gesture) # Get corresponding action for the gesture
        if action:
            if action == "cursor_up":
                self.code_editor.move_cursor_up()
            elif action == "cursor_down":
                self.code_editor.move_cursor_down()
            elif action == "delete_line":
                cursor = self.code_editor.textCursor()
                cursor.select(QTextCursor.LineUnderCursor)
                cursor.removeSelectedText()
                cursor.deleteChar() 
                self.code_editor.setTextCursor(cursor)
            elif action == "indent":
                self.change_indent(1) # Increase indent level for next code block
                self.signals.status_updated.emit("Indented (next block)")
            elif action == "unindent":
                self.change_indent(-1) # Decrease indent level for next code block
                self.signals.status_updated.emit("Unindented (next block)")

def main():
    """Main entry point: Initializes and runs the PyQt5 application."""
    app = QApplication(sys.argv)
    window = GestureProgrammingApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    # Standard Python construct to run main() when the script is executed
    main()