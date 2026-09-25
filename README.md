# Face Recognition Attendance System 📸

A Python-based attendance system that uses face recognition through a webcam to identify known faces and automatically record attendance.

## Technologies Used

- Python
- OpenCV
- face_recognition
- NumPy
- CSV

## Features

- Detects faces using a webcam
- Recognizes known faces
- Loads face images from a local `images` folder
- Automatically records attendance
- Stores the name, date, and time of attendance
- Prevents the same person from being marked multiple times during one run

## How It Works

1. The program loads the known face images.
2. It generates face encodings for the available images.
3. The webcam captures live video.
4. Detected faces are compared with the known face encodings.
5. When a known person is recognized, their attendance is recorded with the current date and time.
6. Press `q` to exit the application.

## Project Structure

```text
face-recognition-attendance-system/
│
├── attendance.py
├── .gitignore
└── README.md

## Note

This project is for educational and practice purposes.
