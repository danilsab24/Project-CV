import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
import time

# Define the PointNet model class as given
class PointNet(nn.Module):
    def __init__(self, num_classes):
        super(PointNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc_layers = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.mlp(x)
        x = self.global_max_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# Load the pre-trained model
num_classes = 29
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

def is_text_too_long(text, max_width, font_scale, thickness):
    img = np.zeros((100, max_width, 3), dtype=np.uint8)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    return text_size[0] > max_width - 20

predicted_text = ""
last_predicted_label = ""
prediction_start_time = 0
max_text_width = 400  # Width of the white space

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    
    text_too_long = is_text_too_long(predicted_text, max_text_width, 1, 2)
    
    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        landmarks = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])
        landmarks = standardize_landmarks(landmarks)
        
        landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = model(landmarks_tensor)
            predicted_label = labels[torch.argmax(output).item()]
        
        if predicted_label == last_predicted_label:
            if time.time() - prediction_start_time >= 2 and predicted_label not in ['nothing', 'del'] and not text_too_long:
                if predicted_label == 'space':
                    predicted_text += ' '
                else:
                    predicted_text += predicted_label
                last_predicted_label = ""
                prediction_start_time = 0
        else:
            last_predicted_label = predicted_label
            prediction_start_time = time.time()
        
        # Draw landmarks and prediction on the frame
        for lm in hand_landmarks.landmark:
            x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
    else:
        predicted_label = ""
    
    # Create a larger white space next to the webcam feed
    height, width, _ = frame.shape
    white_space = np.ones((height, max_text_width, 3), dtype=np.uint8) * 255
    combined_frame = np.hstack((frame, white_space))
    
    # Determine box color based on prediction time
    box_color = (0, 255, 0) if time.time() - prediction_start_time >= 2 else (0, 0, 255)
    
    # Draw the predicted letter box if it's not a special case
    if predicted_label not in ['nothing', 'del']:
        box_width = 270 if predicted_label == 'space' else 150
        cv2.rectangle(combined_frame, (width + 50, 50), (width + 50 + box_width, 150), box_color, 2)
        if text_too_long:
            cv2.putText(combined_frame, "X", (width + 100, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
        else:
            cv2.putText(combined_frame, predicted_label, (width + 100, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)
    
    # Display the predicted text higher up
    cv2.putText(combined_frame, predicted_text, (width + 10, height - 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    
    # Display the legend
    legend = "c: close application\nd: delete letter"
    y0, dy = height - 100, 30
    for i, line in enumerate(legend.split('\n')):
        y = y0 + i * dy
        cv2.putText(combined_frame, line, (width + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('Hand Gesture Recognition', combined_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        break
    elif key == ord('d'):  # D key for delete
        predicted_text = predicted_text[:-1]

cap.release()
cv2.destroyAllWindows()
