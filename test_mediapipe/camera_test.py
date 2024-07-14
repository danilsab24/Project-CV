import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from model_CNN import HandGestureCNN
import torchvision.transforms as transforms

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

num_classes = 29
model_path = "C:/Users/danie/OneDrive - uniroma1.it/Desktop/hand_gesture_cnn_with_metrics.pth"
model = HandGestureCNN(num_classes)

# Load the checkpoint and extract the model state_dict
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Setup video capture
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
          'del', 'nothing', 'space']

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Get bounding box coordinates
            x_ = [landmark.x for landmark in hand_landmarks.landmark]
            y_ = [landmark.y for landmark in hand_landmarks.landmark]

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            hand_img = frame_rgb[y1:y2, x1:x2]

            if hand_img.size == 0:  # Check if the hand_img is empty
                continue

            # Preprocess the hand image
            hand_img = transform(hand_img)
            hand_img = hand_img.unsqueeze(0)  # Add batch dimension

            # Predict the hand gesture
            with torch.no_grad():
                output = model(hand_img)
                _, predicted = torch.max(output, 1)
                predicted_character = labels[predicted.item()]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()