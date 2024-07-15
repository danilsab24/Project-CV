import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm

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

# Percorsi delle cartelle
main_folder = "C:\\Users\\danie\\OneDrive - uniroma1.it\\Desktop\\camera_test\\input"
output_folder = "C:\\Users\\danie\\OneDrive - uniroma1.it\\Desktop\\camera_test\\output"

# Elenco delle sottocartelle
subfolders = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
]

# Creazione delle cartelle di output
for subfolder in subfolders:
    os.makedirs(os.path.join(output_folder, subfolder), exist_ok=True)

# Elaborazione delle immagini
for subfolder in subfolders:
    input_path = os.path.join(main_folder, subfolder)
    output_path = os.path.join(output_folder, subfolder)
    
    # Lista delle immagini nella sottocartella
    image_files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]
    
    for image_file in tqdm(image_files, desc=f'Processing {subfolder}'):
        input_image_path = os.path.join(input_path, image_file)
        result_image = remove_background_and_keep_landmarks(input_image_path)
        
        if result_image is not None:
            output_image_path = os.path.join(output_path, image_file)
            cv2.imwrite(output_image_path, result_image)
