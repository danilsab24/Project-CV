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

class HandGestureVGG16(nn.Module):
    def __init__(self, num_classes=29):
        super(HandGestureVGG16, self).__init__()
        # Carica il modello VGG16 pre-addestrato
        self.vgg16 = models.vgg16(weights='IMAGENET1K_V1')

        # Sostituisci l'ultimo layer completamente connesso
        self.vgg16.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.vgg16(x)

# Funzione per caricare il modello
def load_model(model_path, num_classes):
    model = HandGestureVGG16(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

# Inizializza Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Funzione per rimuovere lo sfondo e mantenere solo i landmarks e le linee che li collegano
def remove_background_and_keep_landmarks(image_path):
    # Leggi l'immagine
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Rileva i landmarks della mano
    results = hands.process(image_rgb)
    
    if not results.multi_hand_landmarks:
        return None

    # Crea un'immagine vuota a colori
    landmarks_image = np.zeros_like(image)
    
    for hand_landmarks in results.multi_hand_landmarks:
        # Disegna i landmarks e le linee che li collegano sull'immagine dei landmarks
        mp_drawing.draw_landmarks(landmarks_image, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2))

    # Crea una maschera binaria basata sull'immagine dei landmarks
    gray_landmarks_image = cv2.cvtColor(landmarks_image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_landmarks_image, 1, 255, cv2.THRESH_BINARY)

    # Applica la maschera all'immagine originale
    result = cv2.bitwise_and(image, image, mask=mask)

    # Converte lo sfondo in bianco (opzionale)
    white_background = np.ones_like(image, dtype=np.uint8) * 255
    result_with_white_bg = np.where(result==0, white_background, result)
    
    return result_with_white_bg

# Trasformazioni delle immagini
transform = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor(),
])

# Caricamento delle immagini da una cartella
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

# Predizione e visualizzazione delle immagini con etichette predette
def predict_and_plot(model, images, labels):
    if not images:
        print("Nessuna immagine processata correttamente.")
        return

    fig, axs = plt.subplots(len(images), 1, figsize=(10, 5 * len(images)))
    
    if len(images) == 1:
        axs = [axs]
    
    for i, (image, filename) in enumerate(images):
        image = image.unsqueeze(0)  # Aggiungi batch dimension
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        label = labels[predicted.item()]

        axs[i].imshow(image.squeeze().permute(1, 2, 0).numpy())
        axs[i].set_title(f'Predicted: {label}, File: {filename}')
        axs[i].axis('off')

    plt.show()

# Definisci i percorsi e le etichette
model_path = "C:\\Users\\danie\\OneDrive - uniroma1.it\\Desktop\\VGG16_only_landmark_hand_gesture_cnn_with_metrics.pth"
input_folder = "C:\\Users\\danie\\OneDrive - uniroma1.it\\Desktop\\camera_test\\input"
labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
]

# Carica il modello
num_classes = len(labels)
model = load_model(model_path, num_classes)

# Carica le immagini
images = load_images_from_folder(input_folder, transform)

# Fai predizioni e visualizza i risultati
predict_and_plot(model, images, labels)
