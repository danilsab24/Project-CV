import cv2
import os
import time

# Directory to save images
save_dir = "C:/Users/danie/OneDrive - uniroma1.it/Desktop/camera_test/output"  # Change this to your desired save folder
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Open the video capture
cap = cv2.VideoCapture(0)

saving = False
frame_count = 0

def countdown(seconds):
    for i in range(seconds, 0, -1):
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, str(i), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.imshow('Camera', frame)
        cv2.waitKey(1000)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Camera', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        if not saving:
            countdown(3)
            saving = True
        else:
            saving = False

    if saving:
        frame_path = os.path.join(save_dir, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    if key == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
