import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from model_CNN import HandGestureCNN

model_path = "C:/Users/danie/OneDrive - uniroma1.it/Desktop/hand_gesture_cnn_with_metrics (1).pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(model_path, map_location=device)

num_classes = 29
model = HandGestureCNN(num_classes=num_classes)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to(device)

# Dimensioni delle immagini che il modello si aspetta
img_height, img_width = 480, 480  # Cambiato a 480x480

# Mappa delle classi (assumendo che il tuo modello restituisca un indice che corrisponde a una lettera)
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                'del', 'nothing', 'space']  # Lista di tutte le lettere che il modello può riconoscere

# Trasformazioni delle immagini
transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Apri la videocamera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Ridimensiona il frame
    resized_frame = cv2.resize(frame, (img_width, img_height))
    
    # Pre-processamento dell'immagine
    pil_frame = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
    input_frame = transform(pil_frame).unsqueeze(0).to(device)

    # Predizione con il modello
    with torch.no_grad():
        predictions = model(input_frame)
        predicted_class = torch.argmax(predictions, dim=1).item()
        predicted_label = class_labels[predicted_class]

    # Mostra il risultato
    cv2.putText(frame, predicted_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Sign Language Recognition', frame)

    # Esci premendo 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Rilascia le risorse
cap.release()
cv2.destroyAllWindows()
