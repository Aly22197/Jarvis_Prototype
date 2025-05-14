#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Face Recognition Module

This module handles user authentication using facial recognition.
It uses DeepFace for face recognition and OpenCV for face detection.

Author: AI Assistant
Date: April 2025
"""

import os
import cv2
import pickle
import numpy as np
from datetime import datetime
import threading
from deepface import DeepFace

class FaceRecognitionModule:
    """
    Face Recognition Module for user authentication using DeepFace
    """
    
    def __init__(self, use_gpu=False):
        """
        Initialize the Face Recognition module
        
        Args:
            use_gpu: Boolean indicating whether to use GPU acceleration
        """
        print("Initializing Face Recognition Module...")
        
        # Store GPU preference
        self.use_gpu = use_gpu
        
        # Define paths
        self.data_dir = os.path.join("smart_assistant", "data", "faces")
        self.authorized_users_file = os.path.join("smart_assistant", "data", "authorized_users.pkl")
        self.attendance_log = os.path.join("smart_assistant", "data", "attendance_log.txt")
        
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load the face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Set DeepFace model parameters
        self.model_name = "VGG-Face"  # Options: VGG-Face, Facenet, OpenFace, DeepFace, DeepID, ArcFace, Dlib
        
        # Use retinaface for better accuracy (it uses GPU if available)
        self.detector_backend = "retinaface"  # Options: opencv, ssd, mtcnn, retinaface, mediapipe
        
        # For better performance with GPU
        if self.use_gpu:
            # TensorFlow will automatically use GPU if available
            print("DeepFace configured to use GPU acceleration if available")
        
        self.distance_metric = "cosine"  # Options: cosine, euclidean, euclidean_l2
        self.recognition_threshold = 0.3  # Lower is more strict
        
        # Initialize authorized users
        self.authorized_users = self._load_authorized_users()
        
        # Tracking variables
        self.current_users = {}  # {name: timestamp}
        self.recognition_lock = threading.Lock()
        
        # Initialize face database
        self.face_database = self._build_face_database()
        
        print(f"Face Recognition Module initialized with {len(self.face_database)} faces")

    def _load_authorized_users(self):
        """Load authorized users from pickle file"""
        if os.path.exists(self.authorized_users_file):
            try:
                with open(self.authorized_users_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading authorized users: {e}")
                return []
        else:
            return []

    def _save_authorized_users(self):
        """Save authorized users to pickle file"""
        try:
            with open(self.authorized_users_file, 'wb') as f:
                pickle.dump(self.authorized_users, f)
            print(f"Saved {len(self.authorized_users)} authorized users")
        except Exception as e:
            print(f"Error saving authorized users: {e}")

    def add_authorized_user(self, username):
        """Add a user to the authorized users list"""
        # Don't allow adding "Unknown" users
        if username and username.lower() == "unknown":
            print("Cannot add 'Unknown' as an authorized user. Please provide a valid username.")
            return False
            
        if username and username not in self.authorized_users and username != "Unknown":
            # Prompt for username if needed
            if username.lower() == "unknown":
                print("Please provide a name for this user:")
                username = input("Username: ")
                if not username or username.lower() == "unknown":
                    print("Invalid username. User not added.")
                    return False
                
            self.authorized_users.append(username)
            self._save_authorized_users()
            self._log_event(f"Added user {username} to authorized users")
            print(f"Successfully added {username} to authorized users.")
            return True
        return False

    def remove_authorized_user(self, username):
        """Remove a user from the authorized users list"""
        if username in self.authorized_users:
            self.authorized_users.remove(username)
            self._save_authorized_users()
            self._log_event(f"Removed user {username} from authorized users")
            return True
        return False

    def _build_face_database(self):
        """Build a database of face representations from saved face images"""
        face_database = {}
        
        # Get all user directories
        user_dirs = [d for d in os.listdir(self.data_dir) 
                     if os.path.isdir(os.path.join(self.data_dir, d)) and d != "temp"]
        
        for user in user_dirs:
            user_dir = os.path.join(self.data_dir, user)
            face_images = [f for f in os.listdir(user_dir) 
                          if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            if not face_images:
                continue
                
            # Use multiple images to create representations
            embeddings = []
            for image_file in face_images:
                img_path = os.path.join(user_dir, image_file)
                try:
                    # First verify the image can be read correctly
                    img = cv2.imread(img_path)
                    if img is None or img.size == 0:
                        print(f"Could not read image file {image_file} for user {user}")
                        # Move problematic file to temp directory
                        temp_dir = os.path.join(self.data_dir, "temp")
                        os.makedirs(temp_dir, exist_ok=True)
                        temp_path = os.path.join(temp_dir, image_file)
                        try:
                            os.rename(img_path, temp_path)
                            print(f"Moved problematic image {image_file} to temp directory")
                        except Exception as e:
                            print(f"Could not move problematic file: {e}")
                        continue
                    
                    embedding = DeepFace.represent(
                        img_path=img_path,
                        model_name=self.model_name,
                        detector_backend=self.detector_backend,
                        enforce_detection=False
                    )
                    
                    # Verify embedding is valid
                    if (isinstance(embedding, list) and 
                        len(embedding) > 0 and 
                        "embedding" in embedding[0]):
                        # Try to convert embedding to numpy array to validate it
                        try:
                            embedding_array = np.array(embedding[0]["embedding"])
                            if not hasattr(embedding_array, "shape") or embedding_array.size == 0:
                                raise ValueError("Invalid embedding array")
                            
                            embeddings.append(embedding)
                            print(f"Added embedding for {user} from {image_file}")
                        except Exception as e:
                            print(f"Invalid embedding data for {user} in {image_file}: {e}")
                    else:
                        print(f"Invalid embedding format for {user} from {image_file}")
                        
                except Exception as e:
                    print(f"Error processing face for user {user} in {image_file}: {e}")
                    # Move problematic file to temp directory
                    temp_dir = os.path.join(self.data_dir, "temp")
                    os.makedirs(temp_dir, exist_ok=True)
                    temp_path = os.path.join(temp_dir, image_file)
                    try:
                        os.rename(img_path, temp_path)
                        print(f"Moved problematic image {image_file} to temp directory")
                    except Exception as move_error:
                        print(f"Could not move problematic file: {move_error}")
            
            if embeddings:
                face_database[user] = embeddings
        
        return face_database

    def _log_event(self, event_text):
        """Log an event with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {event_text}\n"
        
        try:
            with open(self.attendance_log, 'a') as f:
                f.write(log_entry)
        except Exception as e:
            print(f"Error logging event: {e}")

    def _log_attendance(self, username):
        """Log user attendance with timestamp"""
        if username != "Unknown":
            timestamp = datetime.now()
            current_time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            
            # If user was not previously detected or was detected more than 5 minutes ago
            if (username not in self.current_users or 
                (timestamp - self.current_users.get(username, timestamp)).total_seconds() > 300):
                
                self.current_users[username] = timestamp
                self._log_event(f"User {username} detected at {current_time_str}")

    def add_face(self, frame, username):
        """
        Add a new face to the database
        
        Args:
            frame: The image frame containing the face
            username: The name of the user to add
            
        Returns:
            success: Boolean indicating if the face was successfully added
        """
        if not username:
            return False
        
        # Create user directory if it doesn't exist
        user_dir = os.path.join(self.data_dir, username)
        os.makedirs(user_dir, exist_ok=True)
        
        # Detect face in the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            print("No face detected")
            return False
        
        if len(faces) > 1:
            print("Multiple faces detected, using the largest face")
            # Find the largest face
            largest_area = 0
            largest_face = None
            for face in faces:
                x, y, w, h = face
                area = w * h
                if area > largest_area:
                    largest_area = area
                    largest_face = face
            faces = [largest_face]
        
        # Extract face ROI
        x, y, w, h = faces[0]
        face_roi = frame[y:y+h, x:x+w]
        
        # Save face image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        face_filename = f"{username}_{timestamp}.jpg"
        face_path = os.path.join(user_dir, face_filename)
        cv2.imwrite(face_path, face_roi)
        
        # Verify that DeepFace can process this face
        try:
            # Use represent to get face embedding
            DeepFace.represent(
                img_path=face_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False
            )
            
            # Rebuild face database to include the new face
            self.face_database = self._build_face_database()
            
            print(f"Successfully added face for user {username}")
            return True
        except Exception as e:
            print(f"Error adding face: {e}")
            # If error, remove the face image
            if os.path.exists(face_path):
                os.remove(face_path)
            return False

    def recognize_face(self, face_img):
        """
        Recognize a face in an image
        
        Args:
            face_img: The image containing the face
            
        Returns:
            name: The recognized name
            confidence: The confidence score (0-1)
        """
        # Default result
        result = {"name": "Unknown", "confidence": 0.0}
        
        # Validate input image
        if face_img is None or face_img.size == 0:
            print("Invalid face image provided")
            return result
            
        # Save the face image to a temporary file
        temp_dir = os.path.join("smart_assistant", "data", "temp")
        os.makedirs(temp_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_face_path = os.path.join(temp_dir, f"temp_face_{timestamp}.jpg")
        cv2.imwrite(temp_face_path, face_img)
        
        try:
            # Get embeddings for the detected face
            # Modified to set enforce_detection=False
            embedding = DeepFace.represent(
                img_path=temp_face_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False
            )
            
            # Validate embedding structure
            if not isinstance(embedding, list) or len(embedding) == 0 or "embedding" not in embedding[0]:
                print(f"Invalid embedding structure: {type(embedding)}")
                return result
                
            # Validate embedding data
            try:
                embedding_array = np.array(embedding[0]["embedding"])
                if not hasattr(embedding_array, "shape") or embedding_array.size == 0:
                    print("Invalid embedding array shape")
                    return result
            except Exception as e:
                print(f"Error validating embedding: {e}")
                return result
                
            # Compare with the database
            min_distance = float('inf')
            recognized_name = "Unknown"
            distances_by_user = {}
            
            # Compare with each face in the database
            for name, db_embeddings in self.face_database.items():
                user_distances = []
                
                # Compare with each embedding for this user
                for db_embedding in db_embeddings:
                    try:
                        # Ensure both embeddings are properly formatted
                        if not isinstance(db_embedding, list) or len(db_embedding) == 0 or "embedding" not in db_embedding[0]:
                            continue
                        
                        try:
                            v1 = np.array(embedding[0]["embedding"])
                            v2 = np.array(db_embedding[0]["embedding"])
                            
                            # Validate vectors shape
                            if not hasattr(v1, "shape") or not hasattr(v2, "shape"):
                                continue
                                
                            # Ensure vectors have proper dimensions
                            if v1.size == 0 or v2.size == 0:
                                continue
                                
                            # Normalize vectors
                            v1_norm = np.linalg.norm(v1)
                            v2_norm = np.linalg.norm(v2)
                            
                            # Skip if either norm is zero
                            if v1_norm == 0 or v2_norm == 0:
                                continue
                            
                            # Compute cosine similarity
                            cosine_sim = np.dot(v1, v2) / (v1_norm * v2_norm)
                            
                            # Convert to cosine distance (1 - similarity)
                            distance = 1 - cosine_sim
                            user_distances.append(distance)
                        except Exception as e:
                            print(f"Error comparing embeddings: {e}")
                            continue
                    except Exception as e:
                        print(f"Error processing database embedding: {e}")
                        continue
                
                # Use the minimum distance for this user (best match)
                if user_distances:
                    distances_by_user[name] = min(user_distances)
                    if distances_by_user[name] < min_distance:
                        min_distance = distances_by_user[name]
                        recognized_name = name
            
            # Clean up temp file
            if os.path.exists(temp_face_path):
                os.remove(temp_face_path)
            
            # No matches found in database
            if min_distance == float('inf'):
                return result
                
            # Convert distance to confidence (1 - distance)
            confidence = 1.0 - min_distance
            
            # Check if the confidence is high enough
            if min_distance <= self.recognition_threshold:
                # Add secondary validation for unknown faces
                # If the best match is just barely under the threshold, mark as unknown
                validation_threshold = self.recognition_threshold * 0.9  # 90% of threshold
                
                if min_distance > validation_threshold:
                    # This is a borderline match, require more validation
                    # Count how many faces are close to the best match
                    close_matches = sum(1 for dist in distances_by_user.values() 
                                       if dist <= self.recognition_threshold * 1.1)
                    
                    # If there are multiple close matches, mark as unknown
                    if close_matches > 1:
                        print(f"Multiple possible matches detected ({close_matches}), marking as Unknown")
                        result["name"] = "Unknown"
                        result["confidence"] = confidence
                    else:
                        # Log attendance for recognized user
                        self._log_attendance(recognized_name)
                        result["name"] = recognized_name
                        result["confidence"] = confidence
                        result["is_authorized"] = recognized_name in self.authorized_users
                else:
                    # Strong match, proceed with confidence
                    self._log_attendance(recognized_name)
                    result["name"] = recognized_name
                    result["confidence"] = confidence
                    result["is_authorized"] = recognized_name in self.authorized_users
            else:
                result["name"] = "Unknown"
                result["confidence"] = confidence
            
            return result
                
        except Exception as e:
            print(f"Error in face recognition: {e}")
            return result

    def process_frame(self, frame):
        """
        Process a frame to detect and recognize faces
        
        Args:
            frame: The video frame to process
            
        Returns:
            results: Dictionary containing face detection and recognition results
        """
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize results
        results = {
            "faces_detected": False,
            "face_info": [],
            "recognized_user": None,
            "is_authorized": False
        }
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Process detected faces
        if len(faces) > 0:
            results["faces_detected"] = True
            print(f"Face detection: Found {len(faces)} faces")
            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Extract face ROI
                face_roi = frame[y:y+h, x:x+w]
                
                # Recognize face
                face_result = self.recognize_face(face_roi)
                name = face_result["name"]
                confidence = face_result["confidence"]
                
                # Add to results
                results["face_info"].append(((x, y, w, h), name, confidence))
                
                # If this is the most confident result, use it as the main recognition
                if results["recognized_user"] is None or confidence > results["face_info"][0][2]:
                    results["recognized_user"] = name
                    results["is_authorized"] = name in self.authorized_users
                
                print(f"Face recognition: Detected {name} with confidence {confidence:.2f}")
        else:
            print("Face detection: Found 0 faces")
        
        return results 