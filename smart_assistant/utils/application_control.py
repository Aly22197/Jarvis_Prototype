#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Application Control Module

This module handles system control actions based on recognized gestures.
It controls media playback, volume, presentations, and other system functions.

Author: AI Assistant
Date: April 2025
"""

import os
import sys
import threading
import time
import subprocess
import pyautogui
from datetime import datetime

# Set up pyautogui safely
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1

class ApplicationControlModule:
    """
    Application Control Module for executing system commands based on gestures
    """
    
    def __init__(self):
        """Initialize the Application Control module"""
        print("Initializing Application Control Module...")
        
        # Log directory
        self.log_dir = os.path.join("smart_assistant", "data", "command_logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, f"command_log_{datetime.now().strftime('%Y%m%d')}.txt")
        
        # Command execution lock
        self.command_lock = threading.Lock()
        
        # Cooldown system to prevent rapid execution of the same command
        self.last_command = None
        self.last_command_time = datetime.now()
        self.command_cooldown = 1.0  # seconds
        
        # Volume control
        self.current_volume = 50  # Assume 50% as default
        
        # System info
        self.is_windows = sys.platform.startswith('win')
        self.is_mac = sys.platform.startswith('darwin')
        self.is_linux = sys.platform.startswith('linux')
        
        # Gesture to command mapping
        self.gesture_commands = {
            "Point": self.move_cursor,
            "Volume Up": self.volume_up,
            "Volume Down": self.volume_down,
            "Play": self.media_playpause,
            "Stop": self.media_stop,
            "Next Slide": self.next_slide,
            "Previous Slide": self.previous_slide,
            "Pinch": self.mouse_click,
            # "OCR" and "Draw" are handled in the main class
        }
        
        print("Application Control Module initialized successfully!")

    def _log_command(self, command, details=None):
        """Log a command execution with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] Command: {command}"
        
        if details:
            log_entry += f" - Details: {details}"
        
        log_entry += "\n"
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
        except Exception as e:
            print(f"Error logging command: {e}")

    def execute_command(self, gesture, **kwargs):
        """
        Execute a command based on the recognized gesture
        
        Args:
            gesture: The recognized gesture
            **kwargs: Additional parameters for the command
            
        Returns:
            success: Boolean indicating if the command was successfully executed
        """
        # Check cooldown period for the same command
        current_time = datetime.now()
        if gesture == self.last_command:
            time_diff = (current_time - self.last_command_time).total_seconds()
            if time_diff < self.command_cooldown:
                return False
        
        # Update last command info
        self.last_command = gesture
        self.last_command_time = current_time
        
        # Check if the gesture has a mapped command
        if gesture in self.gesture_commands:
            # Execute the command with a lock to prevent concurrent execution
            with self.command_lock:
                try:
                    command_func = self.gesture_commands[gesture]
                    result = command_func(**kwargs)
                    self._log_command(gesture, details=str(kwargs) if kwargs else None)
                    return result
                except Exception as e:
                    print(f"Error executing command {gesture}: {e}")
                    return False
        else:
            print(f"No command mapped for gesture: {gesture}")
            return False

    def move_cursor(self, position=None, **kwargs):
        """
        Move the cursor to a specified position
        
        Args:
            position: (x, y) tuple for absolute position, or None to use finger_tip from kwargs
            
        Returns:
            success: Boolean indicating if the cursor was moved
        """
        try:
            # Try to get position from kwargs if not provided
            if position is None and 'finger_tip' in kwargs:
                position = kwargs['finger_tip']
            
            if position:
                pyautogui.moveTo(position[0], position[1], duration=0.1)
                return True
            return False
        except Exception as e:
            print(f"Error moving cursor: {e}")
            return False

    def mouse_click(self, button='left', **kwargs):
        """
        Perform a mouse click
        
        Args:
            button: Mouse button to click ('left', 'right', 'middle')
            
        Returns:
            success: Boolean indicating if the click was performed
        """
        try:
            pyautogui.click(button=button)
            return True
        except Exception as e:
            print(f"Error clicking mouse: {e}")
            return False

    def volume_up(self, increment=5, **kwargs):
        """
        Increase system volume
        
        Args:
            increment: Volume increment percentage (0-100)
            
        Returns:
            success: Boolean indicating if the volume was increased
        """
        try:
            # Adjust current volume tracker
            self.current_volume = min(100, self.current_volume + increment)
            
            # Increase volume based on OS
            if self.is_windows:
                pyautogui.press('volumeup', presses=increment//5)
            elif self.is_mac:
                for _ in range(increment//5):
                    pyautogui.keyDown('fn')
                    pyautogui.press('f12')
                    pyautogui.keyUp('fn')
            elif self.is_linux:
                subprocess.run(['amixer', '-D', 'pulse', 'sset', 'Master', f'{self.current_volume}%'])
            
            return True
        except Exception as e:
            print(f"Error increasing volume: {e}")
            return False

    def volume_down(self, decrement=5, **kwargs):
        """
        Decrease system volume
        
        Args:
            decrement: Volume decrement percentage (0-100)
            
        Returns:
            success: Boolean indicating if the volume was decreased
        """
        try:
            # Adjust current volume tracker
            self.current_volume = max(0, self.current_volume - decrement)
            
            # Decrease volume based on OS
            if self.is_windows:
                pyautogui.press('volumedown', presses=decrement//5)
            elif self.is_mac:
                for _ in range(decrement//5):
                    pyautogui.keyDown('fn')
                    pyautogui.press('f11')
                    pyautogui.keyUp('fn')
            elif self.is_linux:
                subprocess.run(['amixer', '-D', 'pulse', 'sset', 'Master', f'{self.current_volume}%'])
            
            return True
        except Exception as e:
            print(f"Error decreasing volume: {e}")
            return False

    def media_play(self, **kwargs):
        """
        Play media
        
        Returns:
            success: Boolean indicating if the play command was sent
        """
        try:
            pyautogui.press('playpause')
            return True
        except Exception as e:
            print(f"Error playing media: {e}")
            return False

    def media_stop(self, **kwargs):
        """
        Stop media
        
        Returns:
            success: Boolean indicating if the stop command was sent
        """
        try:
            pyautogui.press('stop')
            return True
        except Exception as e:
            print(f"Error stopping media: {e}")
            return False

    def media_playpause(self, **kwargs):
        """
        Toggle media play/pause
        
        Returns:
            success: Boolean indicating if the play/pause command was sent
        """
        try:
            pyautogui.press('playpause')
            return True
        except Exception as e:
            print(f"Error toggling media play/pause: {e}")
            return False

    def next_slide(self, **kwargs):
        """
        Go to next slide in presentation or next track in media
        
        Returns:
            success: Boolean indicating if the next command was sent
        """
        try:
            # Try both common presentation shortcuts and media next
            pyautogui.press('right')  # For presentations
            time.sleep(0.1)
            pyautogui.press('nexttrack')  # For media
            return True
        except Exception as e:
            print(f"Error going to next slide/track: {e}")
            return False

    def previous_slide(self, **kwargs):
        """
        Go to previous slide in presentation or previous track in media
        
        Returns:
            success: Boolean indicating if the previous command was sent
        """
        try:
            # Try both common presentation shortcuts and media previous
            pyautogui.press('left')  # For presentations
            time.sleep(0.1)
            pyautogui.press('prevtrack')  # For media
            return True
        except Exception as e:
            print(f"Error going to previous slide/track: {e}")
            return False

    def scroll_up(self, lines=3, **kwargs):
        """
        Scroll up
        
        Args:
            lines: Number of lines to scroll
            
        Returns:
            success: Boolean indicating if the scroll was performed
        """
        try:
            pyautogui.scroll(lines)
            return True
        except Exception as e:
            print(f"Error scrolling up: {e}")
            return False

    def scroll_down(self, lines=3, **kwargs):
        """
        Scroll down
        
        Args:
            lines: Number of lines to scroll
            
        Returns:
            success: Boolean indicating if the scroll was performed
        """
        try:
            pyautogui.scroll(-lines)
            return True
        except Exception as e:
            print(f"Error scrolling down: {e}")
            return False

    def type_text(self, text, **kwargs):
        """
        Type text
        
        Args:
            text: Text to type
            
        Returns:
            success: Boolean indicating if the text was typed
        """
        try:
            pyautogui.write(text)
            return True
        except Exception as e:
            print(f"Error typing text: {e}")
            return False

    def press_key(self, key, **kwargs):
        """
        Press a specific key
        
        Args:
            key: Key to press
            
        Returns:
            success: Boolean indicating if the key was pressed
        """
        try:
            pyautogui.press(key)
            return True
        except Exception as e:
            print(f"Error pressing key: {e}")
            return False 