import cv2
import os
import face_recognition
import numpy as np
from datetime import datetime
import csv

# Path to attendance file
attendance_file = "Attendance.csv"

# Create CSV if not exists
if not os.path.exists(attendance_file):
    with open(attendance_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time"])

# Store marked students
marked_students = set()

def markAttendance(name):
    if name not in marked_students:
        with open(attendance_file, "a", newline="") as f:
            writer = csv.writer(f)
            now = datetime.now()
            date_today = now.strftime("%Y-%m-%d")
            time_now = now.strftime("%H:%M:%S")
            writer.writerow([name, date_today, time_now])

        marked_students.add(name)
        print(f"📌 Attendance marked for {name} on {date_today} at {time_now}")

# Load known faces
known_face_encodings = []
known_face_names = []

path = "images"

for file_name in os.listdir(path):
    img_path = os.path.join(path, file_name)

    try:
        img = cv2.imread(img_path)

        if img is None:
            raise Exception("Image not loaded properly")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.ascontiguousarray(img, dtype=np.uint8)

        encodings = face_recognition.face_encodings(img)

        if len(encodings) > 0:
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(file_name)[0])
            print(f"✅ Loaded {file_name}")
        else:
            print(f"⚠️ No face found in {file_name}")

    except Exception as e:
        print(f"❌ Skipping {file_name} due to error: {e}")

print(f"Total known faces loaded: {len(known_face_encodings)}")

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    rgb_small_frame = np.ascontiguousarray(rgb_small_frame, dtype=np.uint8)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):

        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.65)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        name = "Unknown"

        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        # Scale back face location
        top, right, bottom, left = [v * 4 for v in face_location]

        # Draw rectangle
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Show name
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Mark attendance ONLY ONCE per person
        if name != "Unknown" and name not in marked_students:
            markAttendance(name)

        # Show already marked
        if name in marked_students:
            cv2.putText(frame, "Already Marked", (left, bottom + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Display window
    cv2.imshow("Attendance System", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()