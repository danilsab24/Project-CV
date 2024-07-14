import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from model_CNN import HandGestureCNN

num_classes = 29
model_path = "C:/Users/danie/OneDrive - uniroma1.it/Desktop/LandMarks_hand_gesture_cnn_with_metrics.pth"
model = HandGestureCNN(num_classes)

# Carica il checkpoint ed estrai lo state_dict del modello
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Configura la cattura video
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
          'del', 'nothing', 'space']

# Trasformazione dell'immagine per il modello
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((640, 480)),  # Assicurati che l'immagine sia ridimensionata correttamente per il modello
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

            # Ottieni le coordinate della bounding box
            x_ = [landmark.x for landmark in hand_landmarks.landmark]
            y_ = [landmark.y for landmark in hand_landmarks.landmark]

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            # Aggiungi controlli per assicurarti che i bounding box non superino i limiti dell'immagine
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W, x2)
            y2 = min(H, y2)

            hand_img = frame_rgb[y1:y2, x1:x2]

            if hand_img.size == 0:  # Controlla se l'immagine della mano è vuota
                continue

            # Preprocessa l'immagine della mano
            hand_img = transform(hand_img)
            hand_img = hand_img.unsqueeze(0)  # Aggiungi la dimensione del batch

            # Predici il gesto della mano
            with torch.no_grad():
                output = model(hand_img)
                _, predicted = torch.max(output, 1)
                predicted_character = labels[predicted.item()]

            # Disegna la bounding box e l'etichetta predetta sull'immagine
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Mostra il frame con le predizioni
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
