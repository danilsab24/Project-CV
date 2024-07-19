import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm

# Inizialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Function for removing background and remain only landmarks
def remove_background_and_keep_landmarks(image_path):
    
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = hands.process(image_rgb)
    
    if not results.multi_hand_landmarks:
        return None

    # create an empty image
    landmarks_image = np.zeros_like(image)
    
    for hand_landmarks in results.multi_hand_landmarks:
        # Draw landmarks and lines that connect landmarks
        mp_drawing.draw_landmarks(landmarks_image, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2))

    gray_landmarks_image = cv2.cvtColor(landmarks_image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_landmarks_image, 1, 255, cv2.THRESH_BINARY)

    result = cv2.bitwise_and(image, image, mask=mask)

    white_background = np.ones_like(image, dtype=np.uint8) * 255
    result_with_white_bg = np.where(result==0, white_background, result)
    
    return result_with_white_bg

main_folder = "C:\\Users\\danie\\OneDrive - uniroma1.it\\Desktop\\camera_test\\input"
output_folder = "C:\\Users\\danie\\OneDrive - uniroma1.it\\Desktop\\camera_test\\output"

subfolders = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
]

for subfolder in subfolders:
    os.makedirs(os.path.join(output_folder, subfolder), exist_ok=True)

for subfolder in subfolders:
    input_path = os.path.join(main_folder, subfolder)
    output_path = os.path.join(output_folder, subfolder)
    
    image_files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]
    
    for image_file in tqdm(image_files, desc=f'Processing {subfolder}'):
        input_image_path = os.path.join(input_path, image_file)
        result_image = remove_background_and_keep_landmarks(input_image_path)
        
        if result_image is not None:
            output_image_path = os.path.join(output_path, image_file)
            cv2.imwrite(output_image_path, result_image)
