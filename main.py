import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox

# Predefined HSV ranges for common colors
COLOR_RANGES = {
    "red": (np.array([0, 120, 70]), np.array([10, 255, 255])),
    "blue": (np.array([94, 80, 2]), np.array([126, 255, 255])),
    "green": (np.array([35, 52, 72]), np.array([89, 255, 255])),
    "yellow": (np.array([22, 93, 0]), np.array([45, 255, 255])),
    "orange": (np.array([10, 100, 20]), np.array([25, 255, 255])),
    "purple": (np.array([129, 50, 70]), np.array([158, 255, 255])),
}

# Function to detect items by color
def detect_color(frame, lower_bound, upper_bound):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return mask, contours

# Function to start the camera and detect color
def start_detection(color_name, lower_bound, upper_bound):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        mask, contours = detect_color(frame, lower_bound, upper_bound)

        # Draw bounding boxes around detected items
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small objects
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{color_name.capitalize()} Item", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Detected Items", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# GUI to manage the system
def open_gui():
    def start_system():
        color_name = color_entry.get().lower()
        if color_name in COLOR_RANGES:
            lower_bound, upper_bound = COLOR_RANGES[color_name]
            start_detection(color_name, lower_bound, upper_bound)
        else:
            messagebox.showerror("Input Error", f"Color '{color_name}' is not supported.")

    root = tk.Tk()
    root.title("Color Detection System")

    tk.Label(root, text="Enter Color Name:").grid(row=0, column=0)
    color_entry = tk.Entry(root)
    color_entry.grid(row=0, column=1)

    tk.Button(root, text="Start Detection", command=start_system).grid(row=1, column=0, columnspan=2)

    root.mainloop()

if __name__ == "__main__":
    open_gui()
