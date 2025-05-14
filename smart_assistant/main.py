#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Smart Virtual Assistant with Vision-Based Controls

This is the main entry point for the smart assistant application. It initializes all modules
and runs the main application loop.

Author: AI Assistant
Date: April 2025
"""

import os
import sys
import time
import cv2
import numpy as np
import threading
from datetime import datetime

# Import custom modules - fix relative imports
try:
    # When running as a module (with run_assistant.py)
    from smart_assistant.utils.face_recognition import FaceRecognitionModule
    from smart_assistant.utils.gesture_recognition import GestureRecognitionModule
    from smart_assistant.utils.ocr_module import OCRModule
    from smart_assistant.utils.camera_calibration import CameraCalibrationModule
    from smart_assistant.utils.voice_feedback import VoiceFeedbackModule
    from smart_assistant.utils.application_control import ApplicationControlModule
except ImportError:
    # When running directly from within the package
    from utils.face_recognition import FaceRecognitionModule
    from utils.gesture_recognition import GestureRecognitionModule
    from utils.ocr_module import OCRModule
    from utils.camera_calibration import CameraCalibrationModule
    from utils.voice_feedback import VoiceFeedbackModule
    from utils.application_control import ApplicationControlModule

class SmartAssistant:
    """
    Main class for the Smart Virtual Assistant with Vision-Based Controls.
    This class orchestrates all the modules and runs the main application loop.
    """
    
    def __init__(self, use_gpu=False):
        """
        Initialize the Smart Assistant and all its modules.
        
        Args:
            use_gpu: Boolean indicating whether to use GPU acceleration
        """
        print("Initializing Smart Virtual Assistant...")
        print(f"GPU acceleration is {'enabled' if use_gpu else 'disabled'}")
        
        # Set GPU optimization for OpenCV if available
        if use_gpu:
            try:
                # Try to use OpenCV with CUDA if available
                cv2.setUseOptimized(True)
                if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    print(f"OpenCV CUDA support enabled with {cv2.cuda.getCudaEnabledDeviceCount()} devices")
            except Exception as e:
                print(f"OpenCV GPU optimization error: {e}")
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Initialize modules with GPU preference
        self.face_recognition = FaceRecognitionModule(use_gpu=use_gpu)
        self.gesture_recognition = GestureRecognitionModule(use_gpu=use_gpu)
        self.ocr_module = OCRModule(use_gpu=use_gpu)
        self.camera_calibration = CameraCalibrationModule()
        self.voice_feedback = VoiceFeedbackModule()
        self.app_control = ApplicationControlModule()
        
        # State variables
        self.authorized_user = False
        self.current_user = None
        self.current_gesture = None
        self.ocr_mode = False
        self.ocr_text = ""
        self.drawing_mode = False
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Canvas for drawing
        self.canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.prev_point = None  # Track previous point for continuous drawing lines
        
        # Add gesture cooldown tracking
        self.last_gesture = None
        self.last_gesture_time = time.time()
        self.gesture_cooldown = 1.5  # 1.5 seconds cooldown between regular gestures
        
        # Special gesture tracking (to fix toggling)
        self.special_gestures = {
            "Draw": {"active": False, "last_toggled": time.time(), "hold_time": 0, "detected_frames": 0, "consecutive_frames": 0},
            "OCR": {"active": False, "last_toggled": time.time(), "hold_time": 0, "detected_frames": 0, "consecutive_frames": 0},
        }
        self.special_gesture_cooldown = 1.5  # Reduced cooldown for more responsive toggling
        self.special_gesture_hold_frames = 10  # Reduced frames required (easier activation)
        self.last_detected_special_gesture = None
        
        # Add gesture guide mode
        self.show_gesture_guide = False
        self.guide_gesture = None
        
        print("Smart Assistant initialized successfully!")
        self.voice_feedback.speak("Smart assistant is ready.")

    def process_frame(self, frame):
        """
        Process a single frame from the camera.
        
        Args:
            frame: The camera frame to process
            
        Returns:
            processed_frame: The frame with overlays and visualizations
        """
        # Apply camera calibration if available
        if self.camera_calibration.is_calibrated:
            frame = self.camera_calibration.undistort_frame(frame)
        
        # Make a copy of the frame for display
        display_frame = frame.copy()
        
        # User authentication
        face_results = self.face_recognition.process_frame(frame)
        if face_results["faces_detected"]:
            self.current_user = face_results["recognized_user"]
            self.authorized_user = face_results["is_authorized"]
            
            # Display face recognition results
            for (x, y, w, h), name, confidence in face_results["face_info"]:
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                text = f"{name} ({confidence:.2f})"
                cv2.putText(display_frame, text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Process gestures if user is authorized
        if self.authorized_user:
            gesture_results = self.gesture_recognition.process_frame(frame)
            new_gesture = gesture_results["gesture"]
            
            # Debug display for detected gesture
            raw_gesture_text = f"Raw Gesture: {new_gesture}"
            cv2.putText(display_frame, raw_gesture_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            
            # Debug display for finger states
            if "finger_states" in gesture_results:
                finger_states = gesture_results["finger_states"]
                finger_debug_text = f"Fingers: T:{int(finger_states['thumb'])} I:{int(finger_states['index'])} M:{int(finger_states['middle'])} R:{int(finger_states['ring'])} P:{int(finger_states['pinky'])}"
                cv2.putText(display_frame, finger_debug_text, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            
            # Reset all consecutive frame counters for special gestures
            for gesture_key in self.special_gestures.keys():
                if gesture_key != new_gesture:
                    self.special_gestures[gesture_key]["consecutive_frames"] = 0
            
            # Handle special gestures that require holding (Draw and OCR)
            if new_gesture in self.special_gestures:
                special_data = self.special_gestures[new_gesture]
                special_data["consecutive_frames"] += 1
                
                # Only start counting detected frames when the gesture has been consistently detected
                if special_data["consecutive_frames"] >= 3:
                    special_data["detected_frames"] += 1
                else:
                    special_data["detected_frames"] = 0
                
                # Update hold status on display
                hold_percent = min(100, (special_data["detected_frames"] / self.special_gesture_hold_frames) * 100)
                
                # Show hold progress
                hold_color = (0, 255, 255) if new_gesture == "Draw" else (255, 0, 0)
                cv2.putText(display_frame, f"Hold {new_gesture}: {hold_percent:.0f}%", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, hold_color, 2)
                
                # Print debugging info for Draw gesture
                if new_gesture == "Draw":
                    finger_debug = f"Draw detection: frames={special_data['detected_frames']}/{self.special_gesture_hold_frames}, consecutive={special_data['consecutive_frames']}"
                    cv2.putText(display_frame, finger_debug, (10, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                    
                    # Add extra-visible activation indicator if close to activation
                    if hold_percent > 50:
                        radius = int(hold_percent / 2)
                        cv2.circle(display_frame, (display_frame.shape[1] - 100, 100), radius, (0, 255, 255), -1)
                        cv2.putText(display_frame, "ACTIVATING", (display_frame.shape[1] - 180, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Check if the gesture is held long enough to toggle (for Draw and OCR)
                if special_data["detected_frames"] >= self.special_gesture_hold_frames:
                    current_time = time.time()
                    # Check cooldown period
                    if (current_time - special_data["last_toggled"]) > self.special_gesture_cooldown:
                        # Toggle the special gesture mode
                        special_data["active"] = not special_data["active"]
                        special_data["last_toggled"] = current_time
                        special_data["detected_frames"] = 0  # Reset frame counter
                        self.last_detected_special_gesture = new_gesture
                        
                        # Update application state based on special gesture
                        if new_gesture == "Draw":
                            self.drawing_mode = special_data["active"]
                            print(f"Drawing mode {'activated' if self.drawing_mode else 'deactivated'}")
                            if self.drawing_mode:
                                # Turn off OCR mode if it's on
                                if self.ocr_mode:
                                    self.ocr_mode = False
                                    self.special_gestures["OCR"]["active"] = False
                                
                                # Clear canvas when entering drawing mode
                                self.canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
                                self.voice_feedback.speak("Drawing mode activated.")
                            else:
                                self.prev_point = None  # Reset previous point
                                self.voice_feedback.speak("Drawing mode deactivated.")
                        
                        elif new_gesture == "OCR":
                            self.ocr_mode = special_data["active"]
                            print(f"OCR mode {'activated' if self.ocr_mode else 'deactivated'}")
                            if self.ocr_mode:
                                # Turn off drawing mode if it's on
                                if self.drawing_mode:
                                    self.drawing_mode = False
                                    self.special_gestures["Draw"]["active"] = False
                                
                                self.voice_feedback.speak("OCR mode activated. Show a document to the camera.")
                            else:
                                self.voice_feedback.speak("OCR mode deactivated.")
            
            # Display current gesture name
            if new_gesture and new_gesture != "None":
                gesture_text = f"Gesture: {new_gesture}"
                cv2.putText(display_frame, gesture_text, (10, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2)
            
            # Execute regular commands (for non-special gestures)
            if new_gesture and new_gesture != "None" and new_gesture not in self.special_gestures:
                current_time = time.time()
                gesture_changed = new_gesture != self.last_gesture
                cooldown_passed = (current_time - self.last_gesture_time) > self.gesture_cooldown
                
                if gesture_changed or cooldown_passed:
                    self.current_gesture = new_gesture
                    self.last_gesture = new_gesture
                    self.last_gesture_time = current_time
                    
                    # Execute commands for regular gestures
                    if new_gesture in self.app_control.gesture_commands:
                        # Pass finger tip position for cursor movement if available
                        kwargs = {}
                        if gesture_results["finger_tip"] and new_gesture == "Point":
                            kwargs["finger_tip"] = gesture_results["finger_tip"]
                        
                        print(f"Executing command for {new_gesture}")
                        self.app_control.execute_command(new_gesture, **kwargs)
            
            # Display gesture information
            gesture_text = f"Gesture: {self.current_gesture if self.current_gesture else new_gesture}"
            cv2.putText(display_frame, gesture_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Mode status display
            mode_status_y = 60
            cv2.putText(display_frame, f"Status:", (1050, mode_status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            mode_status_y += 30
            
            # Draw mode status with color coding
            draw_status = "ON" if self.drawing_mode else "OFF"
            draw_color = (0, 255, 255) if self.drawing_mode else (100, 100, 100)
            cv2.putText(display_frame, f"Drawing: {draw_status}", (1050, mode_status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_color, 1)
            mode_status_y += 30
            
            # OCR mode status with color coding
            ocr_status = "ON" if self.ocr_mode else "OFF"
            ocr_color = (255, 0, 0) if self.ocr_mode else (100, 100, 100)
            cv2.putText(display_frame, f"OCR: {ocr_status}", (1050, mode_status_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, ocr_color, 1)
            
            # Draw hand landmarks on the frame
            display_frame = self.gesture_recognition.draw_landmarks(display_frame)
            
            # Handle drawing mode
            if self.drawing_mode and gesture_results["finger_tip"]:
                # Visual indicator that drawing mode is active (frame border)
                cv2.rectangle(display_frame, (0, 0), (1279, 719), (0, 255, 255), 8)
                cv2.putText(display_frame, "DRAWING MODE ACTIVE", (display_frame.shape[1]//2 - 150, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Use only index finger for drawing (point gesture)
                index_extended = gesture_results.get("finger_states", {}).get("index", False)
                middle_extended = gesture_results.get("finger_states", {}).get("middle", False)
                
                # Draw with index finger (when middle finger is not extended)
                if index_extended and not middle_extended:
                    # Get current finger tip position
                    x, y = gesture_results["finger_tip"]
                    finger_tip = (x, y)
                    
                    # Add visual feedback for drawing point
                    cv2.circle(display_frame, finger_tip, 10, (0, 255, 255), -1)
                    
                    # Draw dot at current position
                    cv2.circle(self.canvas, finger_tip, 5, (0, 255, 255), -1)
                    
                    # Draw line from previous position if available
                    if self.prev_point is not None:
                        cv2.line(self.canvas, self.prev_point, finger_tip, (0, 255, 255), 5)
                    
                    # Update previous point
                    self.prev_point = finger_tip
                elif index_extended and middle_extended:
                    # Both index and middle fingers extended - eraser mode
                    x, y = gesture_results["finger_tip"]
                    eraser_size = 20
                    cv2.rectangle(display_frame, (x-eraser_size, y-eraser_size), 
                                 (x+eraser_size, y+eraser_size), (255, 0, 0), 2)
                    cv2.putText(display_frame, "Eraser", (x+eraser_size+5, y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
                    
                    # Erase area on canvas
                    cv2.rectangle(self.canvas, (x-eraser_size, y-eraser_size), 
                                 (x+eraser_size, y+eraser_size), (0, 0, 0), -1)
                    
                    # Reset previous point to avoid connecting lines after erasing
                    self.prev_point = None
                else:
                    # Reset previous point if index finger is not extended
                    # This allows lifting finger to create separate lines
                    self.prev_point = None
                
                # Blend canvas with display frame
                display_frame = cv2.addWeighted(display_frame, 1, self.canvas, 0.7, 0)
                
                # Display drawing mode status and instructions
                status_y = 640
                
                # Add Clear Canvas button
                clear_btn_x, clear_btn_y = 100, 670
                clear_btn_w, clear_btn_h = 150, 40
                
                cv2.rectangle(display_frame, (clear_btn_x, clear_btn_y), 
                             (clear_btn_x + clear_btn_w, clear_btn_y + clear_btn_h), 
                             (0, 0, 200), -1)
                cv2.putText(display_frame, "Clear Canvas (c)", (clear_btn_x + 10, clear_btn_y + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display instructions for drawing mode
                cv2.putText(display_frame, "Drawing Mode: Move index finger to draw", (300, status_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(display_frame, "Point index finger only to draw, index+middle to erase", (300, status_y + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(display_frame, "Hold Draw gesture to exit", (300, status_y + 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            elif not self.drawing_mode:
                self.prev_point = None  # Reset previous point when not in drawing mode
        
        # Process OCR if in OCR mode
        if self.ocr_mode:
            cv2.putText(display_frame, "OCR Mode Active", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Define ROI rectangle
            roi_x, roi_y, roi_w, roi_h = 300, 200, 680, 320
            
            # Draw highlighted OCR capture area
            overlay = display_frame.copy()
            # Draw semi-transparent overlay on entire frame
            cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], display_frame.shape[0]), (0, 0, 0), -1)
            # Blend with original frame
            alpha = 0.5  # Transparency factor
            display_frame = cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0)
            
            # Draw ROI rectangle (the clear area for document)
            cv2.rectangle(display_frame, (roi_x, roi_y), (roi_x+roi_w, roi_y+roi_h), (255, 0, 0), 2)
            
            # Add corner markers to make ROI more visible
            corner_length = 40
            # Top-left
            cv2.line(display_frame, (roi_x, roi_y), (roi_x + corner_length, roi_y), (255, 0, 0), 3)
            cv2.line(display_frame, (roi_x, roi_y), (roi_x, roi_y + corner_length), (255, 0, 0), 3)
            # Top-right
            cv2.line(display_frame, (roi_x + roi_w, roi_y), (roi_x + roi_w - corner_length, roi_y), (255, 0, 0), 3)
            cv2.line(display_frame, (roi_x + roi_w, roi_y), (roi_x + roi_w, roi_y + corner_length), (255, 0, 0), 3)
            # Bottom-left
            cv2.line(display_frame, (roi_x, roi_y + roi_h), (roi_x + corner_length, roi_y + roi_h), (255, 0, 0), 3)
            cv2.line(display_frame, (roi_x, roi_y + roi_h), (roi_x, roi_y + roi_h - corner_length), (255, 0, 0), 3)
            # Bottom-right
            cv2.line(display_frame, (roi_x + roi_w, roi_y + roi_h), (roi_x + roi_w - corner_length, roi_y + roi_h), (255, 0, 0), 3)
            cv2.line(display_frame, (roi_x + roi_w, roi_y + roi_h), (roi_x + roi_w, roi_y + roi_h - corner_length), (255, 0, 0), 3)
            
            # Add guide text
            cv2.putText(display_frame, "Place document here", (roi_x + 10, roi_y + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Process OCR on the ROI
            roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            self.ocr_text = self.ocr_module.extract_text(roi)
            
            # Create a text display area
            text_box_x, text_box_y = 10, 190
            text_box_w, text_box_h = 280, 380
            
            # Draw text display box
            cv2.rectangle(display_frame, (text_box_x, text_box_y), 
                         (text_box_x + text_box_w, text_box_y + text_box_h), (255, 0, 0), 2)
            cv2.rectangle(display_frame, (text_box_x, text_box_y - 30), 
                         (text_box_x + text_box_w, text_box_y), (255, 0, 0), -1)
            cv2.putText(display_frame, "OCR Results:", (text_box_x + 10, text_box_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display OCR text in the box
            lines = self.ocr_text.strip().split('\n')
            max_lines = 12  # Maximum number of lines to display
            line_height = 30
            
            # Display OCR text (limited to max_lines)
            for i, line in enumerate(lines[:max_lines]):
                # Truncate long lines
                if len(line) > 30:
                    line = line[:27] + "..."
                cv2.putText(display_frame, line, (text_box_x + 10, text_box_y + 20 + i*line_height), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show number of lines not displayed if there are more lines
            if len(lines) > max_lines:
                cv2.putText(display_frame, f"... +{len(lines) - max_lines} more lines", 
                           (text_box_x + 10, text_box_y + text_box_h - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Display instructions for OCR mode
            instruction_y = 620
            cv2.putText(display_frame, "Hold OCR gesture to exit", (10, instruction_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            instruction_y -= 30
            cv2.putText(display_frame, "Press 's' to save OCR text", (10, instruction_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Calculate and display FPS
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 1:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
        
        cv2.putText(display_frame, f"FPS: {self.fps:.1f}", (10, 710), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Display authorization status
        auth_status = "Authorized" if self.authorized_user else "Unauthorized"
        auth_color = (0, 255, 0) if self.authorized_user else (0, 0, 255)
        cv2.putText(display_frame, f"Status: {auth_status}", (1050, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, auth_color, 2)
        
        # Display help option
        cv2.putText(display_frame, "Press 'h' for gesture guide", (10, 550), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Show gesture guide if enabled
        if self.show_gesture_guide:
            display_frame = self._draw_gesture_guide(display_frame)
        
        return display_frame

    def _draw_gesture_guide(self, frame):
        """Draw a visual guide for how to perform gestures correctly"""
        # Create a semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        
        # Set transparency
        alpha = 0.7
        result_frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)
        
        # Title
        cv2.putText(result_frame, "Gesture Guide", (result_frame.shape[1]//2 - 100, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display guide for cycling through gestures
        cv2.putText(result_frame, "Press 'g' to cycle through gestures", (result_frame.shape[1]//2 - 150, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(result_frame, "Press 'h' to exit guide", (result_frame.shape[1]//2 - 100, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Define gesture guides
        gesture_guides = {
            "Draw": {
                "description": "Index and middle fingers extended together (like a pencil)",
                "tips": [
                    "Keep fingers close and parallel",
                    "Keep thumb, ring and pinky fingers down",
                    "Hold gesture for activation/deactivation"
                ],
                "color": (0, 255, 255)
            },
            "OCR": {
                "description": "Thumb, index and middle fingers extended (like number 3)",
                "tips": [
                    "Extend thumb clearly away from palm",
                    "Keep ring and pinky fingers down",
                    "Hold gesture for activation/deactivation"
                ],
                "color": (255, 0, 0)
            },
            "Point": {
                "description": "Only index finger extended (pointing)",
                "tips": [
                    "Keep only index finger extended",
                    "Keep all other fingers down",
                    "Use to move cursor"
                ],
                "color": (255, 255, 0)
            },
            "Pinch": {
                "description": "Thumb and index finger touching",
                "tips": [
                    "Touch thumb and index finger tips",
                    "Other fingers can be relaxed",
                    "Use to click"
                ],
                "color": (255, 0, 255)
            },
            "Volume Up": {
                "description": "Only thumb extended upward (thumbs up)",
                "tips": [
                    "Keep only thumb extended upward",
                    "All other fingers closed",
                    "Use to increase volume"
                ],
                "color": (0, 255, 128)
            },
            "Volume Down": {
                "description": "Only thumb extended downward (thumbs down)",
                "tips": [
                    "Keep only thumb extended downward",
                    "All other fingers closed",
                    "Use to decrease volume"
                ],
                "color": (0, 255, 128)
            },
            "Play": {
                "description": "Closed fist (no fingers extended)",
                "tips": [
                    "Make a fist with all fingers closed",
                    "Keep fingers curled in",
                    "Use to play media"
                ],
                "color": (0, 165, 255)
            },
            "Stop": {
                "description": "Open hand (all fingers extended)",
                "tips": [
                    "Extend all fingers outward",
                    "Spread fingers slightly",
                    "Use to stop media"
                ],
                "color": (0, 165, 255)
            },
            "Next Slide": {
                "description": "Only pinky finger extended",
                "tips": [
                    "Extend only the pinky finger",
                    "Keep all other fingers closed",
                    "Use to go to next slide"
                ],
                "color": (128, 0, 255)
            },
            "Previous Slide": {
                "description": "Index and pinky fingers extended (SpiderMan)",
                "tips": [
                    "Extend only index and pinky fingers",
                    "Keep thumb, middle and ring fingers closed",
                    "Use to go to previous slide"
                ],
                "color": (128, 0, 255)
            }
        }
        
        # If no specific gesture is selected, show list of all gestures
        if not self.guide_gesture:
            y_pos = 150
            spacing = 30
            
            cv2.putText(result_frame, "Available Gestures:", (50, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            y_pos += spacing
            
            for i, (gesture, guide) in enumerate(gesture_guides.items()):
                cv2.putText(result_frame, f"{i+1}. {gesture}: {guide['description']}", (50, y_pos + i*spacing),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, guide['color'], 1)
        else:
            # Show detailed guide for selected gesture
            guide = gesture_guides.get(self.guide_gesture)
            if guide:
                # Title
                y_pos = 150
                spacing = 40
                
                cv2.putText(result_frame, f"Gesture: {self.guide_gesture}", (50, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, guide['color'], 2)
                y_pos += spacing
                
                # Description
                cv2.putText(result_frame, f"Description: {guide['description']}", (50, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                y_pos += spacing + 20
                
                # Tips
                cv2.putText(result_frame, "Tips:", (50, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                y_pos += spacing - 10
                
                for i, tip in enumerate(guide['tips']):
                    cv2.putText(result_frame, f"• {tip}", (70, y_pos + i*spacing),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                
                # Visual diagram placeholder (future enhancement)
                cv2.putText(result_frame, "Try to match this hand position:", (50, y_pos + (len(guide['tips']) + 1)*spacing),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, guide['color'], 1)
                
                # Draw gesture-specific diagrams
                if self.guide_gesture == "Draw":
                    # Draw a simple diagram of index and middle finger extended
                    start_x, start_y = 200, y_pos + (len(guide['tips']) + 3)*spacing
                    self._draw_hand_diagram_draw(result_frame, start_x, start_y, guide['color'])
                elif self.guide_gesture == "OCR":
                    start_x, start_y = 200, y_pos + (len(guide['tips']) + 3)*spacing
                    self._draw_hand_diagram_ocr(result_frame, start_x, start_y, guide['color'])
                elif self.guide_gesture == "Volume Up" or self.guide_gesture == "Volume Down":
                    start_x, start_y = 200, y_pos + (len(guide['tips']) + 3)*spacing
                    self._draw_hand_diagram_thumb(result_frame, start_x, start_y, guide['color'], 
                                               self.guide_gesture == "Volume Up")
                elif self.guide_gesture == "Play":
                    start_x, start_y = 200, y_pos + (len(guide['tips']) + 3)*spacing
                    self._draw_hand_diagram_fist(result_frame, start_x, start_y, guide['color'])
                elif self.guide_gesture == "Stop":
                    start_x, start_y = 200, y_pos + (len(guide['tips']) + 3)*spacing
                    self._draw_hand_diagram_open_hand(result_frame, start_x, start_y, guide['color'])
        
        return result_frame

    def _draw_hand_diagram_draw(self, frame, x, y, color):
        """Draw a simple diagram of the Draw gesture"""
        # Palm
        cv2.rectangle(frame, (x, y), (x+100, y+150), (100, 100, 100), -1)
        
        # Index and middle fingers extended together
        cv2.rectangle(frame, (x+30, y-100), (x+70, y), color, -1)
        
        # Label
        cv2.putText(frame, "Index and middle", (x-50, y-120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        cv2.putText(frame, "fingers together", (x-40, y-90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    
    def _draw_hand_diagram_ocr(self, frame, x, y, color):
        """Draw a simple diagram of the OCR gesture"""
        # Palm
        cv2.rectangle(frame, (x, y), (x+100, y+150), (100, 100, 100), -1)
        
        # Thumb
        cv2.rectangle(frame, (x-40, y+20), (x, y+60), color, -1)
        
        # Index finger
        cv2.rectangle(frame, (x+20, y-100), (x+40, y), color, -1)
        
        # Middle finger
        cv2.rectangle(frame, (x+60, y-100), (x+80, y), color, -1)
        
        # Label
        cv2.putText(frame, "Thumb, index, and", (x-70, y-120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        cv2.putText(frame, "middle extended", (x-40, y-90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    def _draw_hand_diagram_thumb(self, frame, x, y, color, thumb_up=True):
        """Draw a simple diagram of thumb up/down gesture"""
        # Palm
        cv2.rectangle(frame, (x, y), (x+100, y+150), (100, 100, 100), -1)
        
        # Thumb
        if thumb_up:
            cv2.rectangle(frame, (x-40, y-60), (x, y+20), color, -1)
            label = "Thumb pointing up"
        else:
            cv2.rectangle(frame, (x-40, y+130), (x, y+210), color, -1)
            label = "Thumb pointing down"
        
        # Label
        cv2.putText(frame, label, (x-60, y-90 if thumb_up else y+240),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        cv2.putText(frame, "Other fingers closed", (x-80, y-60 if thumb_up else y+270),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    
    def _draw_hand_diagram_fist(self, frame, x, y, color):
        """Draw a simple diagram of closed fist gesture"""
        # Fist
        cv2.rectangle(frame, (x, y), (x+120, y+150), (100, 100, 100), -1)
        cv2.ellipse(frame, (x+60, y+75), (60, 75), 0, 0, 360, color, 2)
        
        # Label
        cv2.putText(frame, "Closed fist", (x, y-30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        cv2.putText(frame, "No fingers extended", (x-40, y-60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    
    def _draw_hand_diagram_open_hand(self, frame, x, y, color):
        """Draw a simple diagram of open hand gesture"""
        # Palm
        cv2.rectangle(frame, (x, y), (x+100, y+150), (100, 100, 100), -1)
        
        # Fingers
        # Thumb
        cv2.rectangle(frame, (x-40, y+20), (x, y+60), color, -1)
        # Index
        cv2.rectangle(frame, (x+20, y-100), (x+40, y), color, -1)
        # Middle
        cv2.rectangle(frame, (x+45, y-110), (x+65, y), color, -1)
        # Ring
        cv2.rectangle(frame, (x+70, y-100), (x+90, y), color, -1)
        # Pinky
        cv2.rectangle(frame, (x+95, y-80), (x+115, y), color, -1)
        
        # Label
        cv2.putText(frame, "All fingers extended", (x-40, y-120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    def run(self):
        """Run the main application loop."""
        print("Starting Smart Assistant...")
        print("Press 'a' to add a new user, 'c' to calibrate the camera, 'h' for gesture guide, 'q' to quit.")
        
        # Check if we have any authorized users
        if not self.face_recognition.authorized_users:
            print("No authorized users found. You will need to add a user first.")
            print("Position your face in front of the camera and press 'a' to add yourself.")
            self.voice_feedback.speak("No authorized users found. Please add a user.")
        
        try:
            while True:
                # Capture frame from camera
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Failed to capture frame.")
                    break
                
                # Process the frame
                processed_frame = self.process_frame(frame)
                
                # Display the processed frame
                cv2.imshow("Smart Virtual Assistant", processed_frame)
                
                # Check for key presses
                key = cv2.waitKey(1) & 0xFF
                
                # Press 'q' to quit
                if key == ord('q'):
                    break
                # Press 'c' to calibrate camera or clear canvas in drawing mode
                elif key == ord('c'):
                    if self.drawing_mode:
                        self.canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
                        print("Canvas cleared")
                    else:
                        print("Starting camera calibration...")
                        self.voice_feedback.speak("Starting camera calibration.")
                        self.camera_calibration.calibrate(self.cap)
                        if self.camera_calibration.is_calibrated:
                            print("Camera calibration completed successfully!")
                            self.voice_feedback.speak("Camera calibration completed successfully.")
                        else:
                            print("Camera calibration failed or was cancelled.")
                            self.voice_feedback.speak("Camera calibration failed or was cancelled.")
                # Press 'Shift+C' to force camera calibration even in drawing mode
                elif key == ord('C'):  # Capital C (Shift+C)
                    print("Starting camera calibration (forced)...")
                    self.voice_feedback.speak("Starting camera calibration.")
                    self.camera_calibration.calibrate(self.cap)
                    if self.camera_calibration.is_calibrated:
                        print("Camera calibration completed successfully!")
                        self.voice_feedback.speak("Camera calibration completed successfully.")
                    else:
                        print("Camera calibration failed or was cancelled.")
                        self.voice_feedback.speak("Camera calibration failed or was cancelled.")
                # Press 's' to save current OCR text
                elif key == ord('s') and self.ocr_text:
                    filename = f"ocr_text_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    data_dir = os.path.join("smart_assistant", "data")
                    os.makedirs(data_dir, exist_ok=True)
                    with open(os.path.join(data_dir, filename), "w") as f:
                        f.write(self.ocr_text)
                    print(f"OCR text saved to {filename}")
                    self.voice_feedback.speak("OCR text saved.")
                # Press 'h' to toggle gesture guide
                elif key == ord('h'):
                    self.show_gesture_guide = not self.show_gesture_guide
                    if not self.show_gesture_guide:
                        self.guide_gesture = None
                # Press 'g' to cycle through gestures in the guide
                elif key == ord('g') and self.show_gesture_guide:
                    gesture_list = ["Draw", "OCR", "Point", "Pinch", 
                                   "Volume Up", "Volume Down", "Play", "Stop",
                                   "Next Slide", "Previous Slide"]
                    if not self.guide_gesture:
                        self.guide_gesture = gesture_list[0]
                    else:
                        curr_idx = gesture_list.index(self.guide_gesture)
                        next_idx = (curr_idx + 1) % len(gesture_list)
                        self.guide_gesture = gesture_list[next_idx]
                # Press 'a' to add current user to authorized users
                elif key == ord('a'):
                    if self.current_user and self.current_user.lower() != "unknown":
                        success = self.face_recognition.add_authorized_user(self.current_user)
                        if success:
                            print(f"Added {self.current_user} to authorized users.")
                            self.voice_feedback.speak(f"Added {self.current_user} to authorized users.")
                    else:
                        # Prompt for a username
                        print("Face not recognized. Please enter a username:")
                        username = input("Enter username: ")
                        if username and username.lower() != "unknown":
                            # Save current frame with the face
                            success = self.face_recognition.add_face(frame, username)
                            if success:
                                self.face_recognition.add_authorized_user(username)
                                print(f"Added {username} to authorized users.")
                                self.voice_feedback.speak(f"Added {username} to authorized users.")
                            else:
                                print("Failed to add face. Please try again with your face clearly visible.")
                                self.voice_feedback.speak("Failed to add face. Please try again.")
                        else:
                            print("Invalid username. User not added.")
                
        except KeyboardInterrupt:
            print("Smart Assistant interrupted by user.")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up
            self.cap.release()
            cv2.destroyAllWindows()
            print("Smart Assistant terminated.")

if __name__ == "__main__":
    assistant = SmartAssistant()
    assistant.run() 