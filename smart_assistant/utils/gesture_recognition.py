#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gesture Recognition Module

This module handles hand gesture recognition using MediaPipe Hands.
It detects hand landmarks and classifies gestures based on hand positions.

Author: AI Assistant
Date: April 2025
"""

import cv2
import time
import math
import numpy as np
import mediapipe as mp
from collections import deque

class GestureRecognitionModule:
    """
    Gesture Recognition Module using MediaPipe Hands
    """
    
    def __init__(self, use_gpu=False):
        """
        Initialize the Gesture Recognition module
        
        Args:
            use_gpu: Boolean indicating whether to use GPU acceleration
        """
        print("Initializing Gesture Recognition Module...")
        
        # Store GPU preference
        self.use_gpu = use_gpu
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize Hands object with custom settings
        # MediaPipe automatically uses GPU acceleration when available through TensorFlow
        # For better performance, set higher confidence thresholds on GPU
        min_detection_confidence = 0.6 if use_gpu else 0.5
        min_tracking_confidence = 0.6 if use_gpu else 0.5
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        if use_gpu:
            print("MediaPipe Hands configured to use GPU acceleration if available")
        
        # Gesture definitions and thresholds
        self.THUMB_TIP = 4
        self.INDEX_TIP = 8
        self.MIDDLE_TIP = 12
        self.RING_TIP = 16
        self.PINKY_TIP = 20
        
        self.THUMB_IP = 3
        self.INDEX_PIP = 6
        self.MIDDLE_PIP = 10
        self.RING_PIP = 14
        self.PINKY_PIP = 18
        
        self.WRIST = 0
        
        # List of supported gestures
        self.GESTURES = [
            "None",      # No gesture detected
            "Point",     # Index finger pointing (for controlling cursor)
            "FistUp",    # Fist pointed upward (volume up)
            "FistDown",  # Fist pointed downward (volume down)
            "ThumbsUp",  # Thumbs up (approve/like)
            "ThumbsDown",# Thumbs down (disapprove/dislike)
            "Victory",   # Victory/Peace sign (play/pause)
            "SpiderMan", # Index and pinky fingers extended (next track/slide)
            "OpenHand",  # All fingers extended (stop)
            "Pinch",     # Thumb and index finger pinched (select)
            "OCR",       # Thumb, index, and middle fingers extended (activate OCR)
            "Draw"       # Index and middle fingers extended (activate drawing)
        ]
        
        # Gesture history for more stability
        self.gesture_history = deque(maxlen=9)  # Increased for even more stability
        self.last_gesture = "None"
        
        # Finger positions tracking and smoothing
        self.finger_tips = []
        self.finger_tip_history = deque(maxlen=5)  # Track last 5 positions for smoothing
        
        # Debug information
        self.finger_states = {
            "thumb": False,
            "index": False,
            "middle": False,
            "ring": False,
            "pinky": False
        }
        
        print("Gesture Recognition Module initialized successfully!")

    def _calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

    def _get_angle(self, point1, point2, point3):
        """Calculate angle between three points"""
        a = np.array([point1.x, point1.y])
        b = np.array([point2.x, point2.y])
        c = np.array([point3.x, point3.y])
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Calculate dot product
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        
        # Ensure the value is within valid range for arccos
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        
        # Get angle in radians and convert to degrees
        angle = np.arccos(cosine_angle) * 180 / np.pi
        
        return angle

    def _is_finger_extended(self, hand_landmarks, finger_tip, finger_pip):
        """Check if a finger is extended based on its position relative to the PIP joint"""
        # The finger is extended if the tip is further from the wrist than the PIP joint
        wrist = hand_landmarks.landmark[self.WRIST]
        tip = hand_landmarks.landmark[finger_tip]
        pip = hand_landmarks.landmark[finger_pip]
        
        # Get distance from wrist to tip and wrist to pip
        wrist_to_tip = self._calculate_distance(wrist, tip)
        wrist_to_pip = self._calculate_distance(wrist, pip)
        
        # The finger is extended if the tip is further from the wrist than the pip joint
        # Add a small threshold for more reliable detection (5% buffer)
        return wrist_to_tip > (wrist_to_pip * 1.05)

    def _is_thumb_extended(self, hand_landmarks):
        """Special check for thumb extension which is different from other fingers"""
        wrist = hand_landmarks.landmark[self.WRIST]
        thumb_tip = hand_landmarks.landmark[self.THUMB_TIP]
        thumb_ip = hand_landmarks.landmark[self.THUMB_IP]
        index_pip = hand_landmarks.landmark[self.INDEX_PIP]
        
        # Calculate the angle formed by wrist, thumb IP, and thumb tip
        angle = self._get_angle(wrist, thumb_ip, thumb_tip)
        
        # Calculate distance between thumb tip and index PIP
        thumb_to_index = self._calculate_distance(thumb_tip, index_pip)
        
        # Calculate horizontal distance component (thumb needs to be extended away from palm)
        thumb_horizontal_distance = abs(thumb_tip.x - wrist.x)
        
        # The thumb is extended if:
        # 1. The angle is large enough
        # 2. The thumb is not close to the index finger
        # 3. The thumb is horizontally away from the wrist
        return angle > 35 and thumb_to_index > 0.1 and thumb_horizontal_distance > 0.1

    def _recognize_gesture(self, hand_landmarks):
        """
        Recognize the gesture based on hand landmarks
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            
        Returns:
            gesture: The recognized gesture string
        """
        # Check finger states
        thumb_extended = self._is_thumb_extended(hand_landmarks)
        index_extended = self._is_finger_extended(hand_landmarks, self.INDEX_TIP, self.INDEX_PIP)
        middle_extended = self._is_finger_extended(hand_landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP)
        ring_extended = self._is_finger_extended(hand_landmarks, self.RING_TIP, self.RING_PIP)
        pinky_extended = self._is_finger_extended(hand_landmarks, self.PINKY_TIP, self.PINKY_PIP)
        
        # Store finger states for debugging
        self.finger_states = {
            "thumb": thumb_extended,
            "index": index_extended,
            "middle": middle_extended,
            "ring": ring_extended,
            "pinky": pinky_extended
        }
        
        # Get landmark positions
        thumb_tip = hand_landmarks.landmark[self.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.INDEX_TIP]
        middle_tip = hand_landmarks.landmark[self.MIDDLE_TIP]
        
        # Calculate distances between fingertips
        thumb_index_distance = self._calculate_distance(thumb_tip, index_tip)
        index_middle_distance = self._calculate_distance(index_tip, middle_tip)
        
        # Get wrist and finger positions for directional gestures
        wrist = hand_landmarks.landmark[self.WRIST]
        middle_pip = hand_landmarks.landmark[self.MIDDLE_PIP]
        
        # Find hand orientation (up or down)
        hand_up = middle_pip.y > wrist.y
        
        # Recognize gestures based on finger positions
        
        # Draw gesture: index and middle fingers extended together, others not
        # Very specific detection to avoid confusion
        if not thumb_extended and index_extended and middle_extended and not ring_extended and not pinky_extended:
            # Calculate the angle between index and middle fingers
            index_middle_angle = self._get_angle(index_tip, hand_landmarks.landmark[self.INDEX_PIP], middle_tip)
            
            # Draw gesture: index and middle fingers close together/parallel
            if index_middle_angle < 25 and index_middle_distance < 0.08:
                return "Draw"
        
        # OCR gesture: thumb, index, and middle fingers extended (like number 3)
        if thumb_extended and index_extended and middle_extended and not ring_extended and not pinky_extended:
            return "OCR"
        
        # Point gesture: only index finger is extended
        if not thumb_extended and index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return "Point"
            
        # Pinch gesture: thumb and index are close, other fingers can be in any position
        if thumb_index_distance < 0.07:
            return "Pinch"
            
        # Next Slide: only pinky finger extended
        if not thumb_extended and not index_extended and not middle_extended and not ring_extended and pinky_extended:
            return "Next Slide"  # Fixed to match UI expectations
            
        # Previous Slide: SpiderMan gesture (index and pinky extended)
        if not thumb_extended and index_extended and not middle_extended and not ring_extended and pinky_extended:
            return "Previous Slide"  # Fixed to match UI expectations
            
        # Play: closed fist (no fingers extended)
        if not thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return "Play"  # Fixed to match UI expectations
            
        # Stop: all fingers extended (open hand)
        if index_extended and middle_extended and ring_extended and pinky_extended:
            # Thumb can be either extended or not for open hand
            return "Stop"  # Fixed to match UI expectations
            
        # ThumbsUp/Volume Up: only thumb extended upward
        if thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            # Calculate thumb direction relative to wrist
            thumb_direction = thumb_tip.y - wrist.y
            if thumb_direction < 0:  # Thumb is pointing up
                return "Volume Up"  # Fixed to match UI expectations
            else:  # Thumb is pointing down
                return "Volume Down"  # Fixed to match UI expectations
        
        # No recognized gesture
        return "None"

    def _get_smoothed_gesture(self, current_gesture):
        """
        Get smoothed gesture to prevent flickering between gestures
        
        Args:
            current_gesture: The currently detected gesture
            
        Returns:
            smoothed_gesture: The smoothed gesture
        """
        # Add current gesture to history
        self.gesture_history.append(current_gesture)
        
        # Count occurrences of each gesture in history
        gesture_counts = {}
        for gesture in self.gesture_history:
            if gesture not in gesture_counts:
                gesture_counts[gesture] = 0
            gesture_counts[gesture] += 1
        
        # Find the most common gesture
        max_count = 0
        most_common_gesture = "None"
        
        for gesture, count in gesture_counts.items():
            if count > max_count:
                max_count = count
                most_common_gesture = gesture
        
        # Only change the gesture if it's consistent enough (reduce required % for faster response)
        required_count = len(self.gesture_history) * 0.6  # Increased threshold for better stability
        if max_count >= required_count:
            smoothed_gesture = most_common_gesture
            self.last_gesture = smoothed_gesture
        else:
            # Otherwise, keep the last stable gesture
            smoothed_gesture = self.last_gesture
            
        return smoothed_gesture

    def draw_landmarks(self, frame):
        """
        Draw hand landmarks on the frame
        
        Args:
            frame: The frame to draw on
            
        Returns:
            frame: The frame with hand landmarks drawn
        """
        # Make a copy of the frame to avoid modifying the original
        annotated_frame = frame.copy()
        
        # Check if we have landmarks to draw
        landmarks = self.results.multi_hand_landmarks if hasattr(self, 'results') and self.results.multi_hand_landmarks else []
        
        # Draw each hand's landmarks
        for hand_landmarks in landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        
        return annotated_frame

    def process_frame(self, frame):
        """
        Process a frame to detect hand landmarks and recognize gestures
        
        Args:
            frame: The frame to process
            
        Returns:
            result: A dictionary containing gesture recognition results
        """
        # Initialize results
        result = {
            "gesture": "None",
            "landmarks": None,
            "finger_tip": None,
            "finger_states": self.finger_states  # Include finger states for debugging
        }
        
        # Convert frame to RGB (MediaPipe requires RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Hands
        rgb_frame.flags.writeable = False
        hand_results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True
        
        # Store the results for drawing
        self.results = hand_results
        
        # Check if any hands were detected
        if hand_results.multi_hand_landmarks:
            # Get the first hand detected
            hand_landmarks = hand_results.multi_hand_landmarks[0]
            
            # Store landmarks for drawing
            result["landmarks"] = hand_landmarks
            
            # Recognize gesture
            current_gesture = self._recognize_gesture(hand_landmarks)
            
            # Update finger states in result
            result["finger_states"] = self.finger_states
            
            # Apply smoothing to gesture
            smoothed_gesture = self._get_smoothed_gesture(current_gesture)
            result["gesture"] = smoothed_gesture
            
            # Get finger tip position (index finger)
            h, w, _ = frame.shape
            index_finger = hand_landmarks.landmark[self.INDEX_TIP]
            x, y = int(index_finger.x * w), int(index_finger.y * h)
            
            # Add to finger tip history for smoothing
            self.finger_tip_history.append((x, y))
            
            # Smooth finger tip position (average of history)
            if len(self.finger_tip_history) > 0:
                x_sum = sum(x for x, y in self.finger_tip_history)
                y_sum = sum(y for x, y in self.finger_tip_history)
                smooth_x = int(x_sum / len(self.finger_tip_history))
                smooth_y = int(y_sum / len(self.finger_tip_history))
                result["finger_tip"] = (smooth_x, smooth_y)
        else:
            # Reset gesture history if no hand detected
            self.gesture_history.clear()
            self.finger_tip_history.clear()
            self.last_gesture = "None"
        
        return result 