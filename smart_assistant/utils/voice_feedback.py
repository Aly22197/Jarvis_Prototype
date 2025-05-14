#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Voice Feedback Module

This module handles text-to-speech functionality for the smart assistant.
It uses pyttsx3 for offline text-to-speech capability.

Author: AI Assistant
Date: April 2025
"""

import os
import threading
import queue
import time
import pyttsx3
from datetime import datetime

class VoiceFeedbackModule:
    """
    Voice Feedback Module for text-to-speech functionality
    """
    
    def __init__(self):
        """Initialize the Voice Feedback module"""
        print("Initializing Voice Feedback Module...")
        
        # TTS engine initialization
        self.engine = pyttsx3.init()
        
        # Configure voice properties
        self.engine.setProperty('rate', 150)  # Speed of speech
        self.engine.setProperty('volume', 0.8)  # Volume (0.0 to 1.0)
        
        # Get available voices
        voices = self.engine.getProperty('voices')
        if voices:
            # Try to set a female voice if available
            female_voices = [v for v in voices if 'female' in v.name.lower()]
            if female_voices:
                self.engine.setProperty('voice', female_voices[0].id)
            else:
                self.engine.setProperty('voice', voices[0].id)
        
        # Create a message queue
        self.speech_queue = queue.Queue()
        
        # Create a thread for speech processing
        self.speech_thread = threading.Thread(target=self._process_speech_queue, daemon=True)
        self.speech_thread.start()
        
        # Log file
        self.log_dir = os.path.join("smart_assistant", "data", "voice_logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, f"voice_log_{datetime.now().strftime('%Y%m%d')}.txt")
        
        # Speaking state
        self.is_speaking = False
        
        print("Voice Feedback Module initialized successfully!")
        self.speak("Voice feedback system ready.")

    def _log_speech(self, text):
        """Log spoken text to a file"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] {text}\n"
            
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
        except Exception as e:
            print(f"Error logging speech: {e}")

    def _process_speech_queue(self):
        """Background thread to process the speech queue"""
        while True:
            try:
                # Get text from queue (timeout to allow checking if program is exiting)
                text = self.speech_queue.get(timeout=0.5)
                
                # Log the speech
                self._log_speech(text)
                
                # Set speaking flag
                self.is_speaking = True
                
                # Speak the text
                self.engine.say(text)
                self.engine.runAndWait()
                
                # Reset speaking flag
                self.is_speaking = False
                
                # Mark task as done
                self.speech_queue.task_done()
                
                # Small delay to prevent 100% CPU usage
                time.sleep(0.1)
                
            except queue.Empty:
                # Queue is empty, just continue
                continue
            except Exception as e:
                print(f"Error in speech thread: {e}")
                # Reset speaking flag in case of error
                self.is_speaking = False
                time.sleep(0.5)

    def speak(self, text):
        """
        Add text to the speech queue
        
        Args:
            text: The text to speak
            
        Returns:
            None
        """
        if not text:
            return
        
        try:
            # Add to queue
            self.speech_queue.put(text)
        except Exception as e:
            print(f"Error adding to speech queue: {e}")

    def speak_priority(self, text):
        """
        Speak text immediately, interrupting current speech
        
        Args:
            text: The text to speak with priority
            
        Returns:
            None
        """
        if not text:
            return
        
        # Create a separate thread to avoid blocking
        def _priority_speak():
            try:
                # Stop current speech
                self.engine.stop()
                
                # Log the priority speech
                self._log_speech(f"[PRIORITY] {text}")
                
                # Set speaking flag
                self.is_speaking = True
                
                # Speak the priority text
                self.engine.say(text)
                self.engine.runAndWait()
                
                # Reset speaking flag
                self.is_speaking = False
            except Exception as e:
                print(f"Error in priority speech: {e}")
                self.is_speaking = False
        
        # Start the priority speech thread
        threading.Thread(target=_priority_speak, daemon=True).start()

    def change_voice(self, gender=None, index=None):
        """
        Change the voice used for speech
        
        Args:
            gender: 'male' or 'female'
            index: Index of the voice to use
            
        Returns:
            success: Boolean indicating if the voice was successfully changed
        """
        try:
            voices = self.engine.getProperty('voices')
            
            if not voices:
                print("No voices available")
                return False
            
            # Select by gender
            if gender:
                gender_voices = [v for v in voices if gender.lower() in v.name.lower()]
                if gender_voices:
                    self.engine.setProperty('voice', gender_voices[0].id)
                    print(f"Changed to {gender} voice: {gender_voices[0].name}")
                    return True
            
            # Select by index
            if index is not None and 0 <= index < len(voices):
                self.engine.setProperty('voice', voices[index].id)
                print(f"Changed to voice: {voices[index].name}")
                return True
            
            # If no selection was made, return False
            return False
        except Exception as e:
            print(f"Error changing voice: {e}")
            return False

    def set_speaking_rate(self, rate):
        """
        Set the speaking rate
        
        Args:
            rate: Speaking rate (words per minute, typically 100-200)
            
        Returns:
            None
        """
        try:
            self.engine.setProperty('rate', rate)
        except Exception as e:
            print(f"Error setting speaking rate: {e}")

    def set_volume(self, volume):
        """
        Set the speaking volume
        
        Args:
            volume: Volume level (0.0 to 1.0)
            
        Returns:
            None
        """
        try:
            # Ensure volume is within valid range
            volume = max(0.0, min(1.0, volume))
            self.engine.setProperty('volume', volume)
        except Exception as e:
            print(f"Error setting volume: {e}")

    def get_available_voices(self):
        """
        Get a list of available voices
        
        Returns:
            voices_info: List of dictionaries with voice information
        """
        try:
            voices = self.engine.getProperty('voices')
            voices_info = []
            
            for i, voice in enumerate(voices):
                voices_info.append({
                    'index': i,
                    'id': voice.id,
                    'name': voice.name,
                    'languages': voice.languages,
                    'gender': 'female' if 'female' in voice.name.lower() else 'male'
                })
            
            return voices_info
        except Exception as e:
            print(f"Error getting available voices: {e}")
            return [] 