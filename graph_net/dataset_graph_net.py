import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

# Percorso del dataset
dataset_path = r"C:\\Users\\danie\\OneDrive - uniroma1.it\\Desktop\\DATA\\dataset"
folders = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
]

# Inizializzare MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Funzione per estrarre i landmarks
def extract_landmarks(image):
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        return [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
    return []

# Creare un dataframe per memorizzare i dati
data = []

# Contare il numero totale di immagini
total_images = sum(len([f for f in os.listdir(os.path.join(dataset_path, folder)) if f.endswith('.jpg') or f.endswith('.png')]) for folder in folders)

# Estrarre i landmarks dalle immagini in ciascuna cartella
with tqdm(total=total_images, desc="Processing images") as pbar:
    for folder in folders:
        folder_path = os.path.join(dataset_path, folder)
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):  # Assicurarsi che siano immagini
                image_path = os.path.join(folder_path, filename)
                image = cv2.imread(image_path)
                landmarks = extract_landmarks(image)
                if landmarks:
                    # Aggiungere le informazioni al dataframe
                    data.append([folder, filename] + [coord for lm in landmarks for coord in lm])
                pbar.update(1)

# Convertire i dati in un DataFrame pandas
columns = ['label', 'filename'] + [f'{axis}{i+1}' for i in range(21) for axis in ['x', 'y', 'z']]
df = pd.DataFrame(data, columns=columns)

# Salvare il dataframe in un file CSV
output_csv = r"C:\\Users\\danie\\OneDrive - uniroma1.it\\Desktop\\DATA\\landmarks_output.csv"
df.to_csv(output_csv, index=False)

print(f"Dati salvati in {output_csv}")
