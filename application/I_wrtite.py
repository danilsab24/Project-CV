import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
from graphnet import PointNet

# Load the pre-trained model
num_classes = 29  # Total number of labels
model_path = "C:\\Users\\danie\\OneDrive - uniroma1.it\\Desktop\\point_net_model.pth"
model = PointNet(num_classes)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Define the labels
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
          'del', 'nothing', 'space']

# Initialize webcam
cap = cv2.VideoCapture(0)

def standardize_landmarks(landmarks):
    min_x = np.min(landmarks[:, 0])
    max_x = np.max(landmarks[:, 0])
    min_y = np.min(landmarks[:, 1])
    max_y = np.max(landmarks[:, 1])
    landmarks[:, 0] = (landmarks[:, 0] - min_x) / (max_x - min_x)
    landmarks[:, 1] = (landmarks[:, 1] - min_y) / (max_y - min_y)
    return landmarks

predicted_text = ""
max_text_width = 300  # Maximum width of the predicted text box

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    
    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        landmarks = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])
        landmarks = standardize_landmarks(landmarks)
        
        landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = model(landmarks_tensor)
            predicted_label = labels[torch.argmax(output).item()]
        
        # Draw landmarks and prediction on the frame
        for lm in hand_landmarks.landmark:
            x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
        # If the predicted label is one of the special cases, do not display it in the box
        if predicted_label not in ['nothing', 'space', 'del']:
            cv2.putText(frame, f"Prediction: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        predicted_label = ""
    
    # Create a larger white space next to the webcam feed
    height, width, _ = frame.shape
    white_space = np.ones((height, 400, 3), dtype=np.uint8) * 255
    combined_frame = np.hstack((frame, white_space))
    
    # Draw the predicted letter box if it's not a special case
    if predicted_label not in ['nothing', 'space', 'del']:
        cv2.rectangle(combined_frame, (width + 50, 50), (width + 200, 150), (0, 0, 0), 2)
        cv2.putText(combined_frame, predicted_label, (width + 100, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)
    
    # Calculate the width of the predicted text
    text_size = cv2.getTextSize(predicted_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_width = text_size[0]
    
    # Display the predicted text higher up and check if it exceeds the max width
    if text_width < max_text_width:
        cv2.putText(combined_frame, predicted_text, (width + 10, height - 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        block_input = False
    else:
        # Draw a red "X" if the text exceeds the maximum width
        cv2.putText(combined_frame, "X", (width + 150, height - 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
        block_input = True
    
    # Display the legend
    legend = "c: close application\nspace: add space\nd: delete letter"
    y0, dy = height - 100, 30
    for i, line in enumerate(legend.split('\n')):
        y = y0 + i * dy
        cv2.putText(combined_frame, line, (width + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('Hand Gesture Recognition', combined_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        break
    elif key == ord('k') and predicted_label not in ['nothing', 'space', 'del'] and not block_input:
        predicted_text += predicted_label
    elif key == ord(' ') and not block_input:
        predicted_text += ' '
    elif key == ord('d'):  # D key for delete
        predicted_text = predicted_text[:-1]

cap.release()
cv2.destroyAllWindows()
