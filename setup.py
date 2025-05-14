#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="smart_assistant",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.5.0",
        "numpy>=1.20.0",
        "mediapipe>=0.8.9",
        "pytesseract>=0.3.8",
        "deepface>=0.0.75",
        "pyautogui>=0.9.53",
        "pyttsx3>=2.90",
        "pillow>=8.0.0"
    ],
    author="AI Assistant",
    author_email="ai@example.com",
    description="A smart virtual assistant with vision-based controls",
    keywords="computer vision, face recognition, gesture recognition, OCR",
    python_requires=">=3.7",
) 