import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import messagebox, simpledialog
import json

# Directory to store face data
FACE_DATA_DIR = "face_data"
if not os.path.exists(FACE_DATA_DIR):
    os.makedirs(FACE_DATA_DIR)

# File to store the mapping of names to IDs
LABELS_FILE = os.path.join(FACE_DATA_DIR, "labels.json")

# Load or initialize the name-to-ID mapping
if os.path.exists(LABELS_FILE):
    with open(LABELS_FILE, "r") as f:
        name_to_id = json.load(f)
else:
    name_to_id = {}

# Reverse mapping for ID-to-name lookup
id_to_name = {v: k for k, v in name_to_id.items()}

# Check if the cv2.face module is available
try:
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
except AttributeError:
    messagebox.showerror("Error", "cv2.face module is not available. Please install 'opencv-contrib-python'.")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Train the recognizer with existing data
def train_recognizer():
    faces, labels = [], []
    for filename in os.listdir(FACE_DATA_DIR):
        if filename.endswith(".npy"):
            label = filename.split(".")[0]
            if label in name_to_id:
                face_data = np.load(os.path.join(FACE_DATA_DIR, filename))
                for face in face_data:
                    faces.append(face)
                    labels.append(name_to_id[label])
    if faces:
        face_recognizer.train(faces, np.array(labels, dtype=np.int32))

# Add a new face to the database
def add_face(name):
    if name not in name_to_id:
        name_to_id[name] = len(name_to_id) + 1
        with open(LABELS_FILE, "w") as f:
            json.dump(name_to_id, f)
    cap = cv2.VideoCapture(0)
    face_samples = []
    count = 0
    while count < 20:  # Capture 20 samples
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            # Resize the face to a consistent shape (100x100 pixels)
            resized_face = cv2.resize(face, (100, 100))
            face_samples.append(resized_face)
            count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"Capturing {count}/20", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow("Add Face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    if face_samples:
        # Stack face samples into a single 3D array
        face_samples = np.stack(face_samples, axis=0)
        np.save(os.path.join(FACE_DATA_DIR, f"{name}.npy"), face_samples)
        train_recognizer()
        messagebox.showinfo("Success", f"Face for '{name}' added successfully!")

# Detect faces and recognize them
def start_detection():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            resized_face = cv2.resize(face, (100, 100))
            label, confidence = face_recognizer.predict(resized_face)
            name = id_to_name.get(label, "Unknown") if confidence < 100 else "Unknown"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# GUI to manage the system
def open_gui():
    def add_face_gui():
        name = simpledialog.askstring("Add Face", "Enter the name of the person:")
        if name:
            add_face(name)

    root = tk.Tk()
    root.title("Face Detection System")

    tk.Button(root, text="Add Face", command=add_face_gui).grid(row=0, column=0, padx=10, pady=10)
    tk.Button(root, text="Start Detection", command=start_detection).grid(row=1, column=0, padx=10, pady=10)

    root.mainloop()

if __name__ == "__main__":
    train_recognizer()
    open_gui()
