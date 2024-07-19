import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2
import mediapipe as mp
import numpy as np
import torchvision.models as models
from VGG16_hande_gesture_detection_model import HandGestureVGG16

def load_model(model_path, num_classes):
    model = HandGestureVGG16(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# Inizialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function for remove backgrounf and remains landmarks
def remove_background_and_keep_landmarks(image_path):
    
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Find  landmarks of hand
    results = hands.process(image_rgb)
    
    if not results.multi_hand_landmarks:
        return None

    landmarks_image = np.zeros_like(image)
    
    for hand_landmarks in results.multi_hand_landmarks:
        
        mp_drawing.draw_landmarks(landmarks_image, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2))

    gray_landmarks_image = cv2.cvtColor(landmarks_image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_landmarks_image, 1, 255, cv2.THRESH_BINARY)

    result = cv2.bitwise_and(image, image, mask=mask)

    white_background = np.ones_like(image, dtype=np.uint8) * 255
    result_with_white_bg = np.where(result==0, white_background, result)
    
    return result_with_white_bg

transform = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def load_images_from_folder(folder_path, transform):
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        image_processed = remove_background_and_keep_landmarks(img_path)
        if image_processed is not None:
            image = Image.fromarray(cv2.cvtColor(image_processed, cv2.COLOR_BGR2RGB))
            image = transform(image)
            images.append((image, filename))
    return images

# Result of label prediction
def predict_and_plot(model, images, labels):
    if not images:
        print("Nessuna immagine processata correttamente.")
        return

    fig, axs = plt.subplots(len(images), 1, figsize=(10, 5 * len(images)))
    
    if len(images) == 1:
        axs = [axs]
    
    for i, (image, filename) in enumerate(images):
        image = image.unsqueeze(0)  
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        label = labels[predicted.item()]

        axs[i].imshow(image.squeeze().permute(1, 2, 0).numpy())
        axs[i].set_title(f'Predicted: {label}, File: {filename}')
        axs[i].axis('off')

    plt.show()

# Definisci i percorsi e le etichette
model_path = "C:\\Users\\danie\\OneDrive - uniroma1.it\\Desktop\\2_VGG16_only_landmark_hand_gesture_cnn_with_metrics.pth"
input_folder = "C:\\Users\\danie\\OneDrive - uniroma1.it\\Desktop\\camera_test\\input"
labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
]

num_classes = len(labels)
model = load_model(model_path, num_classes)

images = load_images_from_folder(input_folder, transform)

predict_and_plot(model, images, labels)
