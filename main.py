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
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app_debug.log')
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
        self.gestures = {
            "two_fingers": "if",      # Two fingers extended - add if statement
            "three_fingers": "for",   # Three fingers extended - add for loop
            "fist": "execute",        # Closed fist - execute code
            "hand_swipe": "indent",   # Hand swipe right - indent code
            "hand_swipe_left": "unindent"  # Hand swipe left - unindent code
        }
        self.last_gesture = None
        self.gesture_cooldown = 1.0  # Cooldown to prevent rapid repeated gestures
        self.last_gesture_time = 0
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def stop(self):
        """Stop the thread execution."""
        self.running = False
        self.wait()
    
    def detect_gesture(self, hand_landmarks):
        """Detect which gesture is being performed."""
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
        if current_time - self.last_gesture_time > self.gesture_cooldown:
            # One finger up - move up
            if sum(fingers_up) == 1 and fingers_up[1]:
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
            elif sum(fingers_up) == 0:
                self.last_gesture_time = current_time
                return "fist"
        
        return None

    def run(self):
        """Main thread loop."""
        cap = cv2.VideoCapture(0)
        
        # Set camera resolution for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while self.running:
            try:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Flip frame horizontally for more intuitive interaction
                frame = cv2.flip(frame, 1)
                
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                # Draw hand landmarks
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(
                            frame, 
                            hand_landmarks, 
                            self.mp_hands.HAND_CONNECTIONS
                        )
                        
                        # Detect gesture
                        gesture = self.detect_gesture(hand_landmarks)
                        if gesture:
                            # Emit signal through parent's signals object
                            if hasattr(self.parent(), 'signals'):
                                self.parent().signals.gesture_detected.emit(gesture)
                
                # Convert the frame to QImage for display
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                
                # Emit the frame through parent's signals object
                if hasattr(self.parent(), 'signals'):
                    self.parent().signals.update_frame.emit(qt_image)
                
                # Sleep to reduce CPU usage
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
        
        # Commands that can be recognized
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
        """Stop the thread execution."""
        self.running = False
        self.wait()
    
    def run(self):
        """Main thread loop."""
        recognizer = sr.Recognizer()
        
        while self.running:
            try:
                with sr.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = recognizer.listen(source, timeout=2, phrase_time_limit=5)
                
                try:
                    text = recognizer.recognize_google(audio).lower()
                    
                    # Check for custom variable command
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
                                
                                # Send custom variable command
                                if hasattr(self.parent(), 'signals'):
                                    self.parent().signals.speech_detected.emit(f"variable:{var_name}:{var_value}")
                                continue
                    
                    # Check for print command with content
                    elif "print" in text and text != "print":
                        content = text.replace("print", "", 1).strip()
                        if content:
                            if hasattr(self.parent(), 'signals'):
                                self.parent().signals.speech_detected.emit(f"print:{content}")
                            continue
                    
                    # Check for regular commands
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
        
        # Set up a monospaced font
        font = QFont()
        font.setFamily("Courier")
        font.setFixedPitch(True)
        font.setPointSize(11)
        self.setFont(font)
        
        # Set up the highlighter
        self.highlighter = PythonHighlighter(self.document())
        
        # Set tab width to 4 spaces
        self.setTabStopDistance(40)
        
        # Set up cursor highlighting
        self.cursorPositionChanged.connect(self.highlight_current_line)
        
        # Current variable for increment
        self.current_variable = "var"
        
    def highlight_current_line(self):
        """Highlight the line where the cursor is positioned."""
        extraSelections = []
        
        if not self.isReadOnly():
            selection = QTextEdit.ExtraSelection()
            
            # Use a more visible background color with better contrast
            lineColor = QColor(60, 80, 100)  # Blue-gray background for better visibility
            selection.format.setBackground(lineColor)
            selection.format.setProperty(QTextFormat.FullWidthSelection, True)
            
            selection.cursor = self.textCursor()
            selection.cursor.clearSelection()
            
            extraSelections.append(selection)
        
        self.setExtraSelections(extraSelections)
    
    def move_cursor_up(self):
        """Move cursor up one line."""
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.Up)
        self.setTextCursor(cursor)
        self.highlight_current_line()
    
    def move_cursor_down(self):
        """Move cursor down one line."""
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.Down)
        self.setTextCursor(cursor)
        self.highlight_current_line()
    
    def move_cursor_to_start(self):
        """Move cursor to the start of the document."""
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.Start)
        self.setTextCursor(cursor)
        self.highlight_current_line()
    
    def move_cursor_to_end(self):
        """Move cursor to the end of the document."""
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.setTextCursor(cursor)
        self.highlight_current_line()

class PythonHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for Python code."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.highlighting_rules = []
        
        # Keyword format
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
        
        # String format
        string_format = QTextCharFormat()
        string_format.setForeground(QColor("#CE9178"))  # Orange
        self.highlighting_rules.append((re.compile('".*?"'), string_format))
        self.highlighting_rules.append((re.compile("'.*?'"), string_format))
        
        # Comment format
        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor("#6A9955"))  # Green
        self.highlighting_rules.append((re.compile("#.*"), comment_format))
        
        # Function format
        function_format = QTextCharFormat()
        function_format.setForeground(QColor("#DCDCAA"))  # Yellow
        self.highlighting_rules.append((re.compile("\\b[A-Za-z0-9_]+(?=\\()"), function_format))
        
        # Number format
        number_format = QTextCharFormat()
        number_format.setForeground(QColor("#B5CEA8"))  # Light green
        self.highlighting_rules.append((re.compile("\\b\\d+\\b"), number_format))
    
    def highlightBlock(self, text):
        """Apply highlighting to the given block of text."""
        for pattern, format in self.highlighting_rules:
            for match in pattern.finditer(text):
                start = match.start()
                length = match.end() - start
                self.setFormat(start, length, format)

class GestureProgrammingApp(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        # Initialize signals for thread-safe UI updates
        self.signals = Signals()
        
        # Initialize UI
        self.init_ui()
        
        # Initialize code blocks
        self.code_blocks = []
        self.current_indent = 0
        
        # Connect signals
        self.connect_signals()
        
        # Start background threads
        self.start_background_threads()
        
        # Add a test message after a short delay
        QTimer.singleShot(1000, self.add_welcome_message)
        
        # Stored variable for increment functionality
        self.last_variable = "var"
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Gesture Programming Environment")
        self.setGeometry(100, 100, 1200, 800)
        
        # Set application style with dark theme colors
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
        
        # Title and status area
        top_bar = QHBoxLayout()
        title = QLabel("Gesture Programming Environment")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        top_bar.addWidget(title)
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #808080; font-style: italic;")
        top_bar.addStretch()
        top_bar.addWidget(self.status_label)
        
        main_layout.addLayout(top_bar)
        
        # Create splitter for resizable sections
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side - Camera view and instructions
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Camera view
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("background-color: #000000; border: 1px solid #3E3E3E;")
        left_layout.addWidget(self.camera_label)
        
        # Gesture instructions
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
        
        # Right side - Code editor and output console
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Control buttons
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
        
        # Code editor
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
        
        # Output console
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
        
        # Set initial splitter sizes
        splitter.setSizes([400, 800])
        
        main_layout.addWidget(splitter)
    
    def connect_signals(self):
        """Connect all signals to their handlers."""
        self.signals.update_frame.connect(self.update_camera_view, Qt.QueuedConnection)
        self.signals.gesture_detected.connect(self.handle_gesture, Qt.QueuedConnection)
        self.signals.speech_detected.connect(self.handle_speech, Qt.QueuedConnection)
        self.signals.status_updated.connect(self.update_status, Qt.QueuedConnection)
    
    def start_background_threads(self):
        """Start background threads for gesture and speech recognition."""
        # Initialize and start gesture recognition thread
        self.gesture_thread = GestureRecognitionThread(self)
        self.gesture_thread.start()
        
        # Initialize and start speech recognition thread
        self.speech_thread = SpeechRecognitionThread(self)
        self.speech_thread.start()
    
    @pyqtSlot(QImage)
    def update_camera_view(self, image):
        """Update camera view with the latest frame."""
        pixmap = QPixmap.fromImage(image)
        self.camera_label.setPixmap(pixmap.scaled(
            self.camera_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))
    
    @pyqtSlot(str)
    def handle_gesture(self, gesture):
        """Handle detected gestures."""
        try:
            self.process_gesture(gesture)
            self.signals.status_updated.emit(f"Gesture detected: {gesture}")
        except Exception as e:
            pass
    
    @pyqtSlot(str)
    def handle_speech(self, command):
        """Handle speech commands."""
        try:
            # Handle custom commands with parameters
            if ":" in command:
                parts = command.split(":", 1)
                cmd_type = parts[0]
                cmd_value = parts[1]
                
                if cmd_type == "variable":
                    # Format: variable:name:value
                    var_parts = cmd_value.split(":", 1)
                    var_name = var_parts[0]
                    var_value = var_parts[1] if len(var_parts) > 1 else "0"
                    code = f"{var_name} = {var_value}"
                    
                    cursor = self.code_editor.textCursor()
                    if cursor.positionInBlock() > 0:
                        cursor.insertText("\n")
                    indentation = "    " * self.current_indent
                    cursor.insertText(f"{indentation}{code}")
                    # Store variable name for increment command
                    self.last_variable = var_name
                    return
                
                elif cmd_type == "print":
                    # Format: print:content
                    # If the content is a valid variable name, print without quotes
                    content = cmd_value.strip()
                    if content.isidentifier():
                        code = f"print({content})"
                    else:
                        code = f"print(\"{content}\")"
                    cursor = self.code_editor.textCursor()
                    if cursor.positionInBlock() > 0:
                        cursor.insertText("\n")
                    indentation = "    " * self.current_indent
                    cursor.insertText(f"{indentation}{code}")
                    return
            
            # Handle regular commands
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
                cursor.deleteChar()  # Delete the newline
                self.code_editor.setTextCursor(cursor)
            elif command == "indent":
                self.change_indent(1)
            elif command == "unindent":
                self.change_indent(-1)
            elif command == "increment":
                cursor = self.code_editor.textCursor()
                if cursor.positionInBlock() > 0:
                    cursor.insertText("\n")
                indentation = "    " * self.current_indent
                cursor.insertText(f"{indentation}{self.last_variable} += 1")
                self.signals.status_updated.emit(f"Incremented {self.last_variable}")
            else:
                # For control statements that need auto-indentation
                if command in CONTROL_STATEMENTS:
                    self.add_control_block(command)
                # For other commands, add corresponding code block
                elif command in PROGRAM_BLOCKS:
                    self.add_code_block(command)
            
            self.signals.status_updated.emit(f"Voice command detected: {command}")
        except Exception as e:
            pass
    
    def add_code_block(self, block_type):
        """Add a code block to the editor."""
        try:
            if block_type in PROGRAM_BLOCKS:
                code = PROGRAM_BLOCKS[block_type]
                
                # Special handling for if statements to allow custom condition
                if block_type == "if":
                    # Create a dialog with condition options
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
                    
                    # Show dialog and get result
                    if dialog.exec_() == QDialog.Accepted:
                        selected_id = button_group.checkedId()
                        if selected_id == 4:  # Custom option
                            condition = custom_input.text() if custom_input.text() else "condition == True"
                        else:
                            condition = list(condition_options.values())[selected_id]
                        
                        # Check if it's a control statement that needs auto-indentation
                        if block_type in CONTROL_STATEMENTS:
                            code = f'if {condition}:\n    '
                            return self.add_control_block("if")
                        else:
                            code = f'if {condition}:\n    pass'
                    else:
                        return  # If canceled
                
                # Add indentation
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
                pass
        except Exception as e:
            pass
    
    def add_control_block(self, block_type):
        """Add a control block with auto-indentation."""
        try:
            if block_type in PROGRAM_BLOCKS:
                code = PROGRAM_BLOCKS[block_type]
                
                # Get cursor position
                cursor = self.code_editor.textCursor()
                
                # If not at beginning of line, add a newline
                if cursor.positionInBlock() > 0:
                    cursor.insertText("\n")
                
                # Add indentation to the block
                indentation = "    " * self.current_indent
                
                # Split by newlines to handle multi-line blocks
                lines = code.split("\n")
                indented_code = []
                for i, line in enumerate(lines):
                    # First line gets current indentation
                    if i == 0:
                        indented_code.append(indentation + line)
                    else:
                        # Other lines get additional indentation
                        indented_code.append(indentation + line)
                
                # Join back with newlines
                final_code = "\n".join(indented_code)
                
                # Insert code
                cursor.insertText(final_code)
                
                # Increase indentation for next lines
                self.current_indent += 1
                
                # Update cursor at the end of the insertion
                self.code_editor.setTextCursor(cursor)
            
        except Exception as e:
            pass
    
    def change_indent(self, amount):
        """Change the current indentation level."""
        self.current_indent = max(0, self.current_indent + amount)
        self.signals.status_updated.emit(f"Indentation level: {self.current_indent}")
    
    def execute_code(self):
        """Execute the code in the editor."""
        try:
            code = self.code_editor.toPlainText()
            if not code.strip():
                self.output_console.setPlainText("No code to execute")
                return
            
            # Redirect stdout to capture print output
            import io
            from contextlib import redirect_stdout
            
            output_buffer = io.StringIO()
            try:
                with redirect_stdout(output_buffer):
                    # First try to compile the code to check for syntax errors
                    try:
                        compiled_code = compile(code, "<string>", "exec")
                        exec(compiled_code)
                    except IndentationError as ie:
                        line_number = ie.lineno
                        error_msg = str(ie)
                        # Try to fix indentation automatically
                        fixed_code = self.fix_indentation(code, line_number)
                        if fixed_code != code:
                            self.code_editor.setPlainText(fixed_code)
                            # Get the error line and highlight it
                            cursor = self.code_editor.textCursor()
                            cursor.movePosition(QTextCursor.Start)
                            for _ in range(line_number - 1):
                                cursor.movePosition(QTextCursor.Down)
                            self.code_editor.setTextCursor(cursor)
                            self.code_editor.highlight_current_line()
                            
                            if "unexpected indent" in error_msg:
                                self.output_console.setPlainText(f"Fixed unexpected indentation at line {line_number}. Please run again.")
                            else:
                                self.output_console.setPlainText(f"Fixed indentation at line {line_number}. Please run again.")
                            return
                        else:
                            raise  # Re-raise if we couldn't fix it
                    except Exception as e:
                        # Re-raise other exceptions
                        raise
                
                output = output_buffer.getvalue()
                
                if output:
                    self.output_console.setPlainText(output)
                else:
                    self.output_console.setPlainText("Code executed without output")
                
            except Exception as e:
                error_message = str(e)
                # Extract line number from syntax errors
                if "line" in error_message and hasattr(e, 'lineno'):
                    line_number = e.lineno
                    # Move cursor to the error line
                    cursor = self.code_editor.textCursor()
                    cursor.movePosition(QTextCursor.Start)
                    for _ in range(line_number - 1):
                        if not cursor.movePosition(QTextCursor.Down):
                            break  # Break if we can't move down anymore
                    self.code_editor.setTextCursor(cursor)
                    self.code_editor.highlight_current_line()
                
                self.output_console.setPlainText(f"Error: {error_message}")
                
            self.signals.status_updated.emit("Code executed")
        except Exception as e:
            self.output_console.setPlainText(f"System Error: {str(e)}")
    
    def fix_indentation(self, code, error_line):
        """Try to fix indentation errors automatically."""
        lines = code.split('\n')
        
        # Make sure we're not out of bounds
        if error_line <= 0 or error_line > len(lines):
            return code
        
        # Check the line before the error (usually a control statement)
        prev_line = lines[error_line - 2] if error_line > 1 else ""
        error_line_text = lines[error_line - 1]
        
        error_indent = len(error_line_text) - len(error_line_text.lstrip())
        prev_indent = len(prev_line) - len(prev_line.lstrip())
        
        # For unexpected indent errors (too much indentation)
        if error_indent > prev_indent and not prev_line.strip().endswith(':'):
            # Reduce indentation to match previous line
            lines[error_line - 1] = ' ' * prev_indent + error_line_text.lstrip()
        
        # For expected indent after a colon
        elif prev_line.strip().endswith(':'):
            # The error line should be indented at least 4 more spaces
            if not error_line_text.strip():
                # Empty line, add proper indentation + pass statement
                lines[error_line - 1] = ' ' * (prev_indent + 4) + 'pass'
            else:
                # Line has content, check indentation
                if error_indent <= prev_indent:
                    # Need to add more indentation
                    lines[error_line - 1] = ' ' * (prev_indent + 4) + error_line_text.lstrip()
        
        # Check all lines for consistent indentation
        indentation_levels = []
        for i, line in enumerate(lines):
            if line.strip() and not line.strip().startswith('#'):
                current_indent = len(line) - len(line.lstrip())
                
                # If indentation is not a multiple of 4, fix it
                if current_indent % 4 != 0:
                    # Round to nearest multiple of 4
                    new_indent = round(current_indent / 4) * 4
                    lines[i] = ' ' * new_indent + line.lstrip()
        
        return '\n'.join(lines)
    
    def file_save(self):
        """Save the code to a single default file."""
        try:
            # Use a standard filename
            filename = "gesture_program.py"
            
            # Write code to file
            with open(filename, "w") as f:
                f.write(self.code_editor.toPlainText())
            
            self.signals.status_updated.emit(f"Program saved to {filename}")
        except Exception as e:
            self.signals.status_updated.emit(f"Error saving program: {str(e)}")
    
    @pyqtSlot(str)
    def update_status(self, text):
        """Update the status label."""
        self.status_label.setText(text)
    
    def add_welcome_message(self):
        """Add a Python code template to the editor."""
        template = """# Main function definition
def main():
    '''Main program entry point'''
    # Your code here
  

# Standard Python idiom for main execution
if __name__ == "__main__":
    main()
"""
        self.code_editor.setPlainText(template)
        
        # Set cursor position after the "Your code here" comment
        cursor = self.code_editor.textCursor()
        document = self.code_editor.document()
        
        # Find the "Your code here" line
        found = document.find("# Your code here")
        # Check if text was found (position is not -1)
        if not found.isNull():
            cursor.setPosition(found.position())
            cursor.movePosition(QTextCursor.EndOfLine)
            cursor.insertText("\n    ")  # Add a new indented line
            self.code_editor.setTextCursor(cursor)
        
        self.signals.status_updated.emit("Ready")
        self.code_editor.highlight_current_line()
    
    def keyPressEvent(self, event):
        """Handle keyboard events."""
        if event.key() == Qt.Key_Q:
            self.close()
        else:
            super().keyPressEvent(event)
    
    def closeEvent(self, event):
        """Handle window close event."""
        try:
            # Stop background threads
            if hasattr(self, 'gesture_thread'):
                self.gesture_thread.stop()
            if hasattr(self, 'speech_thread'):
                self.speech_thread.stop()
            
            event.accept()
        except Exception as e:
            event.accept()

    def clear_editor(self):
        """Clear all content from the code editor and restore the template."""
        self.code_editor.clear()
        # Restore the template after clearing
        self.add_welcome_message()
        self.signals.status_updated.emit("Code editor cleared and template restored")
    
    def process_gesture(self, gesture):
        """Process recognized gesture and execute corresponding command."""
        self.status_label.setText(f"Gesture: {gesture}")
        
        # Map gestures to editor control actions only
        gesture_map = {
            "scroll_down": "cursor_down",     # Move cursor down
            "scroll_up": "cursor_up",         # Move cursor up
            "thorns": "indent",               # Indent code
            "fist": "unindent",               # Unindent code
        }
        
        action = gesture_map.get(gesture)
        if action:
            if action == "cursor_up":
                self.code_editor.move_cursor_up()
            elif action == "cursor_down":
                self.code_editor.move_cursor_down()
            elif action == "delete_line":
                cursor = self.code_editor.textCursor()
                cursor.select(QTextCursor.LineUnderCursor)
                cursor.removeSelectedText()
                cursor.deleteChar()  # Delete the newline
                self.code_editor.setTextCursor(cursor)
            elif action == "new_line":
                cursor = self.code_editor.textCursor()
                cursor.insertText("\n")
                self.code_editor.setTextCursor(cursor)
            elif action == "indent":
                self.change_indent(1)
                self.signals.status_updated.emit("Indented code")
            elif action == "unindent":
                self.change_indent(-1)
                self.signals.status_updated.emit("Unindented code")

def main():
    """Main entry point of the application."""
    app = QApplication(sys.argv)
    window = GestureProgrammingApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()