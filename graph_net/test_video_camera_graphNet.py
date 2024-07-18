import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp

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
        # Using AdaptiveMaxPool1d to adapt the output size to 1
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

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
          'del', 'nothing', 'space']

cap = cv2.VideoCapture(0)

def standardize_landmarks(landmarks):
    min_x = np.min(landmarks[:, 0])
    max_x = np.max(landmarks[:, 0])
    min_y = np.min(landmarks[:, 1])
    max_y = np.max(landmarks[:, 1])
    landmarks[:, 0] = (landmarks[:, 0] - min_x) / (max_x - min_x)
    landmarks[:, 1] = (landmarks[:, 1] - min_y) / (max_y - min_y)
    return landmarks

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
        
        cv2.putText(frame, f"Prediction: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    cv2.imshow('Hand Gesture Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
