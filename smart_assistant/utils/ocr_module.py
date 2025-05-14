#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OCR Module

This module handles optical character recognition using Tesseract.
It extracts text from images and documents captured from the camera.

Author: AI Assistant
Date: April 2025
"""

import os
import cv2
import numpy as np
import pytesseract
from datetime import datetime

class OCRModule:
    """
    OCR Module for text recognition from images using Tesseract
    """
    
    def __init__(self, use_gpu=False):
        """
        Initialize the OCR module
        
        Args:
            use_gpu: Boolean indicating whether to use GPU acceleration
        """
        print("Initializing OCR Module...")
        
        # Store GPU preference
        self.use_gpu = use_gpu
        
        # OCR availability flag
        self.ocr_available = True
        
        # On Windows, you may need to set the tesseract command path
        if os.name == 'nt':
            # Try to find Tesseract in common locations
            possible_paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            ]
            
            tesseract_found = False
            for path in possible_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    tesseract_found = True
                    break
            
            if not tesseract_found:
                print("Warning: Tesseract OCR not found. OCR functionality will be limited.")
                print("To enable OCR, please install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
                self.ocr_available = False
        
        # Check if pytesseract can actually run
        if self.ocr_available:
            try:
                # Try a simple OCR test
                test_img = np.zeros((50, 100), dtype=np.uint8)
                test_img.fill(255)  # White image
                pytesseract.image_to_string(test_img)
            except Exception as e:
                print(f"OCR test failed: {e}")
                print("OCR functionality will be disabled.")
                self.ocr_available = False
        
        # OCR settings - Select optimal OCR engine mode based on GPU availability
        # OEM values: 0=Legacy engine only, 1=Neural nets LSTM only, 2=Legacy + LSTM, 3=Default
        oem_mode = "1" if self.use_gpu else "3"  # Use LSTM only with GPU for better speed
        self.config = f'--oem {oem_mode} --psm 6'
        
        if self.use_gpu:
            print("OCR configured for optimal performance with GPU")
        
        # Define path for saving OCR results
        self.output_dir = os.path.join("smart_assistant", "data", "ocr_results")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Processing parameters
        self.last_text = ""
        self.last_processed_time = datetime.now()
        # Shorter interval for GPU processing since it's faster
        self.processing_interval = 0.5 if self.use_gpu else 1.0  # seconds
        
        print(f"OCR Module initialized {'successfully' if self.ocr_available else 'in limited mode (Tesseract not available)'}!")

    def _preprocess_image(self, image):
        """
        Preprocess the image for better OCR results
        
        Args:
            image: The input image
            
        Returns:
            processed_image: The processed image ready for OCR
        """
        # Check for CUDA availability and use GPU if available and enabled
        try:
            if self.use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                # Use CUDA-accelerated functions if available
                cuda_stream = cv2.cuda_Stream()
                
                # Upload image to GPU
                gpu_image = cv2.cuda_GpuMat()
                gpu_image.upload(image)
                
                # Convert to grayscale on GPU
                gpu_gray = cv2.cuda.cvtColor(gpu_image, cv2.COLOR_BGR2GRAY)
                
                # Apply Gaussian blur on GPU
                gpu_blurred = cv2.cuda.GaussianBlur(gpu_gray, (5, 5), 0)
                
                # Download result back to CPU
                gray = gpu_blurred.download()
                
                print("Using CUDA-accelerated image preprocessing for OCR")
                return gray
        except Exception as e:
            print(f"CUDA acceleration not available or error: {e}, falling back to CPU")
            # Fall back to CPU processing
            pass
        
        # Standard CPU processing
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to get black and white image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply adaptive thresholding for varying illumination
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 100, 200)
        
        # Dilate the image to connect broken text
        kernel = np.ones((1, 1), np.uint8)
        dilated = cv2.dilate(gray, kernel, iterations=1)
        
        # Return the original grayscale image as the primary processed image
        # Other processed images could be used depending on the context
        return gray

    def extract_text(self, image):
        """
        Extract text from an image using Tesseract OCR
        
        Args:
            image: The input image
            
        Returns:
            text: The extracted text
        """
        # Check if the image is valid
        if image is None or image.size == 0:
            return "No valid image for OCR"
        
        # Check if Tesseract is available
        if not self.ocr_available:
            return "OCR unavailable - Tesseract not installed. Please install Tesseract to enable OCR."
        
        # Check if enough time has passed since last processing
        current_time = datetime.now()
        time_diff = (current_time - self.last_processed_time).total_seconds()
        
        if time_diff < self.processing_interval:
            return self.last_text
        
        # Resize large images for faster processing
        h, w = image.shape[:2]
        max_dimension = 1000  # Maximum dimension for OCR processing
        
        if max(h, w) > max_dimension:
            scale = max_dimension / max(h, w)
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            
        # Preprocess the image
        processed_image = self._preprocess_image(image)
        
        try:
            # Use optimized OCR parameters
            optimized_config = self.config + ' --oem 1'  # LSTM only - faster
            
            # Perform OCR with timeout to prevent hanging on complex images
            import threading
            import queue
            
            q = queue.Queue()
            
            def worker():
                try:
                    result = pytesseract.image_to_string(processed_image, config=optimized_config)
                    q.put(result)
                except Exception as e:
                    q.put(f"OCR Error: {e}")
            
            # Start OCR in separate thread
            t = threading.Thread(target=worker)
            t.daemon = True
            t.start()
            
            # Wait for result with timeout
            t.join(timeout=5.0)  # 5 second timeout
            
            if t.is_alive():
                text = "OCR processing timeout - text too complex, try with simpler text"
            else:
                if not q.empty():
                    text = q.get()
                else:
                    text = "OCR failed to return results"
            
            # Clean up the text
            text = text.strip()
            
            # Update last processed time and text
            self.last_processed_time = current_time
            self.last_text = text
            
            return text
        except Exception as e:
            error_msg = f"OCR Error: {e}"
            print(error_msg)
            return error_msg

    def save_ocr_result(self, image, text):
        """
        Save the OCR results (image and extracted text)
        
        Args:
            image: The image that was processed
            text: The extracted text
            
        Returns:
            success: Boolean indicating if the save was successful
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Save the image
            image_path = os.path.join(self.output_dir, f"ocr_image_{timestamp}.jpg")
            cv2.imwrite(image_path, image)
            
            # Save the text
            text_path = os.path.join(self.output_dir, f"ocr_text_{timestamp}.txt")
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            print(f"OCR result saved: {text_path}")
            return True
        except Exception as e:
            print(f"Error saving OCR result: {e}")
            return False

    def detect_text_regions(self, image):
        """
        Detect regions in the image that may contain text
        
        Args:
            image: The input image
            
        Returns:
            regions: List of bounding boxes for potential text regions
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on size
        min_area = 100  # Minimum contour area to consider
        text_regions = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                text_regions.append((x, y, w, h))
        
        return text_regions

    def highlight_text_regions(self, image):
        """
        Highlight potential text regions in the image
        
        Args:
            image: The input image
            
        Returns:
            highlighted_image: Image with text regions highlighted
        """
        # Make a copy of the input image
        highlighted = image.copy()
        
        # Detect text regions
        regions = self.detect_text_regions(image)
        
        # Draw rectangles around text regions
        for (x, y, w, h) in regions:
            cv2.rectangle(highlighted, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        return highlighted

    def recognize_specific_text(self, image, text_type=None):
        """
        Recognize specific types of text (e.g., numbers, dates)
        
        Args:
            image: The input image
            text_type: The type of text to recognize ('digits', 'date', etc.)
            
        Returns:
            text: The recognized text
        """
        # Preprocess the image
        processed_image = self._preprocess_image(image)
        
        if text_type == 'digits':
            # Configure Tesseract to only look for digits
            config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
            return pytesseract.image_to_string(processed_image, config=config)
        
        elif text_type == 'date':
            # First try to find dates with standard OCR
            text = pytesseract.image_to_string(processed_image, config=self.config)
            
            # TODO: Implement date extraction from the OCR text
            # This would involve regex patterns for common date formats
            
            return text
        
        else:
            # Use default OCR for all other cases
            return pytesseract.image_to_string(processed_image, config=self.config) 