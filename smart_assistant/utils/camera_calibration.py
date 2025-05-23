#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Camera Calibration Module

This module handles camera calibration using OpenCV.
It uses a chessboard pattern to calibrate the camera and correct distortion.

Author: AI Assistant
Date: April 2025
"""

import os
import cv2
import numpy as np
import pickle
from datetime import datetime

class CameraCalibrationModule:
    """
    Camera Calibration Module using OpenCV
    """
    
    def __init__(self):
        """Initialize the Camera Calibration module"""
        print("Initializing Camera Calibration Module...")
        
        # Define paths
        self.data_dir = os.path.join("smart_assistant", "data", "calibration")
        self.calibration_file = os.path.join(self.data_dir, "camera_calibration.pkl")
        
        # Create directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Calibration parameters - try multiple chessboard sizes as fallbacks
        self.chessboard_size_options = [(9, 6), (8, 6), (7, 6), (6, 6)]  # Common chessboard sizes
        self.chessboard_size = self.chessboard_size_options[0]  # Start with 9x6
        self.square_size = 0.024  # Size of a square in meters (typical printed chessboards)
        
        # Calibration state
        self.is_calibrated = False
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        
        # Performance optimization
        self.undistort_maps = None
        self.cached_frame_size = None
        
        # Load calibration if available
        self._load_calibration()
        
        print("Camera Calibration Module initialized successfully!")

    def _load_calibration(self):
        """Load camera calibration parameters from file"""
        if os.path.exists(self.calibration_file):
            try:
                with open(self.calibration_file, 'rb') as f:
                    calibration_data = pickle.load(f)
                
                self.camera_matrix = calibration_data['camera_matrix']
                self.dist_coeffs = calibration_data['dist_coeffs']
                self.rvecs = calibration_data['rvecs']
                self.tvecs = calibration_data['tvecs']
                self.is_calibrated = True
                
                print(f"Camera calibration loaded from {self.calibration_file}")
            except Exception as e:
                print(f"Error loading calibration: {e}")
                self.is_calibrated = False
        else:
            print("No camera calibration file found")
            self.is_calibrated = False

    def _save_calibration(self):
        """Save camera calibration parameters to file"""
        if self.is_calibrated:
            try:
                calibration_data = {
                    'camera_matrix': self.camera_matrix,
                    'dist_coeffs': self.dist_coeffs,
                    'rvecs': self.rvecs,
                    'tvecs': self.tvecs
                }
                
                with open(self.calibration_file, 'wb') as f:
                    pickle.dump(calibration_data, f)
                
                print(f"Camera calibration saved to {self.calibration_file}")
                return True
            except Exception as e:
                print(f"Error saving calibration: {e}")
                return False
        else:
            print("Cannot save calibration: Camera is not calibrated")
            return False

    def _find_chessboard_corners(self, frame):
        """
        Find chessboard corners in a frame
        
        Args:
            frame: The input frame
            
        Returns:
            ret: Boolean indicating if corners were found
            corners: The detected corners
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Try each chessboard size option
        for size in self.chessboard_size_options:
            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(
                gray, size, 
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
            )
            
            if ret:
                # Update the current chessboard size
                self.chessboard_size = size
                print(f"Found chessboard with size {size}")
                
                # Refine corner positions
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                return ret, corners
        
        # If no pattern was found with any size
        return False, None

    def undistort_frame(self, frame):
        """
        Undistort a frame using the calibration parameters
        
        Args:
            frame: The input frame
            
        Returns:
            undistorted_frame: The undistorted frame
        """
        if not self.is_calibrated:
            return frame
        
        try:
            h, w = frame.shape[:2]
            
            # Check if we need to compute undistortion maps
            if self.undistort_maps is None or self.cached_frame_size != (w, h):
                print("Computing undistortion maps for size:", (w, h))
                # Compute optimal new camera matrix
                # Use alpha=0 to keep full image frame (not zoomed in or cropped)
                new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                    self.camera_matrix, self.dist_coeffs, (w, h), 0, (w, h)
                )
                
                # Compute undistortion maps once
                mapx, mapy = cv2.initUndistortRectifyMap(
                    self.camera_matrix, self.dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_32FC1
                )
                
                self.undistort_maps = (mapx, mapy, roi)
                self.cached_frame_size = (w, h)
            
            # Use precomputed maps for faster undistortion (much faster than cv2.undistort)
            mapx, mapy, roi = self.undistort_maps
            undistorted = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
            
            # Skip ROI cropping to avoid zoomed in effect
            # x, y, w, h = roi
            # if all(v > 0 for v in [x, y, w, h]):
            #     undistorted = undistorted[y:y+h, x:x+w]
            
            return undistorted
        except Exception as e:
            print(f"Error undistorting frame: {e}")
            return frame

    def calibrate(self, cap, use_sample_images=True):
        """
        Calibrate the camera using a live video feed
        
        Args:
            cap: OpenCV VideoCapture object
            use_sample_images: Boolean to indicate whether to use sample images
            
        Returns:
            success: Boolean indicating if calibration was successful
        """
        print("Starting camera calibration...")
        
        # Arrays to store object points and image points
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane
        
        # Remember original frame size
        frame_size = None
        
        # Counter for captured frames
        captured_frames = 0
        
        # Remember which chessboard size was used for each frame
        used_chessboard_sizes = []
        
        # Check for sample chessboard images
        if use_sample_images:
            # First try to use sample images from chessboard_images directory
            chessboard_dir = "chessboard_images"
            sample_images = []
            
            if os.path.exists(chessboard_dir):
                sample_images = [f for f in os.listdir(chessboard_dir) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                if sample_images:
                    print(f"Found {len(sample_images)} sample chessboard images. Processing...")
                    
                    for img_file in sample_images:
                        img_path = os.path.join(chessboard_dir, img_file)
                        frame = cv2.imread(img_path)
                        
                        if frame is None:
                            print(f"Could not read image: {img_path}")
                            continue
                            
                        # Store frame size
                        if frame_size is None:
                            frame_size = (frame.shape[1], frame.shape[0])
                            
                        # Find chessboard corners
                        ret, corners = self._find_chessboard_corners(frame)
                        
                        if ret:
                            # Create object points for the current chessboard size
                            current_size = self.chessboard_size
                            objp = np.zeros((current_size[0] * current_size[1], 3), np.float32)
                            objp[:, :2] = np.mgrid[0:current_size[0], 0:current_size[1]].T.reshape(-1, 2)
                            objp = objp * self.square_size
                            
                            objpoints.append(objp)
                            imgpoints.append(corners)
                            used_chessboard_sizes.append(current_size)
                            captured_frames += 1
                            print(f"Found corners in {img_file}. Total: {captured_frames}")
                        else:
                            print(f"No corners found in {img_file}")
                            
                        # Display the image with corners
                        display_frame = frame.copy()
                        if ret:
                            cv2.drawChessboardCorners(display_frame, self.chessboard_size, corners, ret)
                            
                        cv2.imshow("Chessboard Sample", display_frame)
                        cv2.waitKey(500)  # Show each image for 500ms
                        
                    cv2.destroyWindow("Chessboard Sample")
                    
                    if captured_frames >= 3:
                        print(f"Successfully processed {captured_frames} sample images.")
                        
                        # If we already have enough sample images, we can proceed to calibration
                        if captured_frames >= 5:
                            print("Proceeding to calibration using sample images...")
                            goto_calibration = True
                            
                            # Skip the live camera calibration
                            if goto_calibration:
                                # Proceed with calibration
                                print(f"Calculating calibration using {captured_frames} frames...")
                                
                                # Perform camera calibration
                                ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                                    objpoints, imgpoints, frame_size, None, None
                                )
                                
                                if ret:
                                    # Store calibration results
                                    self.camera_matrix = camera_matrix
                                    self.dist_coeffs = dist_coeffs
                                    self.rvecs = rvecs
                                    self.tvecs = tvecs
                                    self.is_calibrated = True
                                    
                                    # Calculate reprojection error
                                    mean_error = 0
                                    for i in range(len(objpoints)):
                                        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
                                        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                                        mean_error += error
                                    
                                    mean_error = mean_error / len(objpoints)
                                    print(f"Calibration complete. Reprojection error: {mean_error}")
                                    
                                    # Save calibration
                                    self._save_calibration()
                                    
                                    return True
                                else:
                                    print("Calibration failed with sample images. Continuing with live calibration...")
                    else:
                        print("Not enough sample images had valid chessboard corners. Continuing with live calibration...")
                else:
                    print("No sample images found. Continuing with live calibration...")
            else:
                print(f"Sample directory {chessboard_dir} not found. Continuing with live calibration...")
                
        # Live calibration with camera
        print("Please show a chessboard pattern to the camera.")
        print("Press 'c' to capture frames for calibration.")
        print("Press 'q' to finish calibration.")
        
        # Feedback window
        cv2.namedWindow("Camera Calibration", cv2.WINDOW_NORMAL)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame")
                    break
                
                # Store frame size
                if frame_size is None:
                    frame_size = (frame.shape[1], frame.shape[0])
                
                # Make a copy of the frame to draw on
                display_frame = frame.copy()
                
                # Find chessboard corners
                ret, corners = self._find_chessboard_corners(frame)
                
                # Draw corners on the display frame
                if ret:
                    cv2.drawChessboardCorners(display_frame, self.chessboard_size, corners, ret)
                    status_text = "Chessboard detected. Press 'c' to capture."
                else:
                    status_text = "No chessboard detected. Please show the chessboard pattern."
                
                # Display status and captured frames count
                cv2.putText(display_frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(display_frame, f"Captured frames: {captured_frames}", (20, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show frame
                cv2.imshow("Camera Calibration", display_frame)
                
                # Check for key presses
                key = cv2.waitKey(1) & 0xFF
                
                # Press 'c' to capture frame
                if key == ord('c') and ret:
                    # Create object points for the current chessboard size
                    current_size = self.chessboard_size
                    objp = np.zeros((current_size[0] * current_size[1], 3), np.float32)
                    objp[:, :2] = np.mgrid[0:current_size[0], 0:current_size[1]].T.reshape(-1, 2)
                    objp = objp * self.square_size
                    
                    objpoints.append(objp)
                    imgpoints.append(corners)
                    used_chessboard_sizes.append(current_size)
                    captured_frames += 1
                    print(f"Frame captured. Total: {captured_frames}, chessboard size: {current_size}")
                    
                    # Save the calibration frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    frame_path = os.path.join(self.data_dir, f"calib_frame_{timestamp}.jpg")
                    cv2.imwrite(frame_path, frame)
                
                # Press 'q' to quit
                elif key == ord('q'):
                    break
                
                # Require at least 10 frames for calibration
                if captured_frames >= 10:
                    cv2.putText(display_frame, "Enough frames captured. Press 'q' to proceed with calibration.", 
                               (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Close the calibration window
            cv2.destroyWindow("Camera Calibration")
            
            # Check if we have enough frames
            if captured_frames < 5:
                print("Not enough frames captured for calibration.")
                return False
            
            print(f"Calculating calibration using {captured_frames} frames...")
            
            # Perform camera calibration
            try:
                ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                    objpoints, imgpoints, frame_size, None, None
                )
                
                if ret:
                    # Store calibration results
                    self.camera_matrix = camera_matrix
                    self.dist_coeffs = dist_coeffs
                    self.rvecs = rvecs
                    self.tvecs = tvecs
                    self.is_calibrated = True
                    
                    # Calculate reprojection error
                    mean_error = 0
                    for i in range(len(objpoints)):
                        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
                        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                        mean_error += error
                    
                    mean_error = mean_error / len(objpoints)
                    print(f"Calibration complete. Reprojection error: {mean_error}")
                    
                    # Save calibration
                    self._save_calibration()
                    
                    return True
                else:
                    print("Calibration failed.")
                    return False
            except Exception as e:
                print(f"Error during calibration calculation: {e}")
                
                # If calibration failed, try a single chessboard size
                if len(set(used_chessboard_sizes)) > 1:
                    print("Trying calibration with only one chessboard size...")
                    # Find the most common chessboard size
                    from collections import Counter
                    most_common_size = Counter(used_chessboard_sizes).most_common(1)[0][0]
                    print(f"Using most common chessboard size: {most_common_size}")
                    
                    # Filter data points to only use the most common size
                    filtered_objpoints = []
                    filtered_imgpoints = []
                    
                    for i in range(len(used_chessboard_sizes)):
                        if used_chessboard_sizes[i] == most_common_size:
                            filtered_objpoints.append(objpoints[i])
                            filtered_imgpoints.append(imgpoints[i])
                    
                    print(f"Using {len(filtered_objpoints)} frames with consistent chessboard size.")
                    
                    try:
                        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                            filtered_objpoints, filtered_imgpoints, frame_size, None, None
                        )
                        
                        if ret:
                            # Store calibration results
                            self.camera_matrix = camera_matrix
                            self.dist_coeffs = dist_coeffs
                            self.rvecs = rvecs
                            self.tvecs = tvecs
                            self.is_calibrated = True
                            
                            print("Calibration completed successfully with filtered data.")
                            self._save_calibration()
                            return True
                        else:
                            print("Calibration failed with filtered data.")
                            return False
                    except Exception as e2:
                        print(f"Error during filtered calibration: {e2}")
                        return False
                
                return False
                
        except Exception as e:
            print(f"Error during calibration: {e}")
            return False
        finally:
            cv2.destroyAllWindows()

    def get_calibration_info(self):
        """
        Get information about the current calibration
        
        Returns:
            info: Dictionary with calibration information
        """
        if not self.is_calibrated:
            return {"status": "Not calibrated"}
        
        # Calculate focal length
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        
        # Calculate principal point
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        return {
            "status": "Calibrated",
            "focal_length_x": fx,
            "focal_length_y": fy,
            "principal_point_x": cx,
            "principal_point_y": cy,
            "distortion_coefficients": self.dist_coeffs.flatten().tolist()
        } 