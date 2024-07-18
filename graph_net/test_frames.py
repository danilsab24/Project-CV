import os
import torch
import torch.nn as nn
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

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
num_classes = 29  # Total number of labels
model_path = "C:\\Users\\danie\\OneDrive - uniroma1.it\\Desktop\\point_net_model.pth"
model = PointNet(num_classes)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Define the labels
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
          'del', 'nothing', 'space']

# Prepare the input images directory
input_dir = "C:\\Users\\danie\\OneDrive - uniroma1.it\\Desktop\\camera_test\\input"
output_dir = "C:\\Users\\danie\\OneDrive - uniroma1.it\\Desktop\\camera_test\\npy"
os.makedirs(output_dir, exist_ok=True)

image_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith('.jpg') or file.endswith('.png')]

# Function to process images and get landmarks
def get_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        landmarks = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])

        # Standardize landmarks
        min_x = np.min(landmarks[:, 0])
        max_x = np.max(landmarks[:, 0])
        min_y = np.min(landmarks[:, 1])
        max_y = np.max(landmarks[:, 1])
        landmarks[:, 0] = (landmarks[:, 0] - min_x) / (max_x - min_x)
        landmarks[:, 1] = (landmarks[:, 1] - min_y) / (max_y - min_y)

        return landmarks
    return None

# Extract landmarks from images and save to .npy files
for image_file in image_files:
    landmarks = get_landmarks(image_file)
    if landmarks is not None:
        npy_file = os.path.join(output_dir, os.path.basename(image_file).replace('.jpg', '.npy').replace('.png', '.npy'))
        np.save(npy_file, landmarks)

# Load .npy files and make predictions
npy_files = [os.path.join(output_dir, file) for file in os.listdir(output_dir) if file.endswith('.npy')]

fig, axs = plt.subplots(len(npy_files), 1, figsize=(10, len(npy_files) * 5))
if len(npy_files) == 1:
    axs = [axs]  # Make it iterable

for i, npy_file in enumerate(npy_files):
    landmarks = np.load(npy_file)
    if landmarks is not None:
        landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = model(landmarks_tensor)
            predicted_label = labels[torch.argmax(output).item()]

        # Plot the image with prediction
        image_file = os.path.join(input_dir, os.path.basename(npy_file).replace('.npy', '.jpg'))
        if not os.path.exists(image_file):
            image_file = os.path.join(input_dir, os.path.basename(npy_file).replace('.npy', '.png'))
        image = cv2.imread(image_file)
        if image is None:
            print(f"Error loading image for plotting: {image_file}")
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axs[i].imshow(image_rgb)
        axs[i].scatter(landmarks[:, 0] * image.shape[1], landmarks[:, 1] * image.shape[0], c='r', marker='o')
        axs[i].set_title(f"Prediction: {predicted_label}")
        axs[i].axis('off')
    else:
        axs[i].text(0.5, 0.5, "No landmarks detected", fontsize=12, ha='center')
        axs[i].axis('off')

plt.tight_layout()
plt.show()
