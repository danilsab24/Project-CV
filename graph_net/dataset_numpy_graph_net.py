import os
import random
import cv2
import glob
from tqdm import tqdm
from transformation_3D import Scale, Rotate, GaussianNoise
import mediapipe as mp
import numpy as np
from utils import get_hand_points

# Inizializza Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def create_augmentations(points_raw, angles, scales, rot_axes, scale_axes, augmentations, dest_path, label, name):
    count = 0
    # Applicazione delle trasformazioni
    if 'rot' in augmentations:
        for _ in range(9):
            axis = random.choice(rot_axes)
            angle = random.choice(angles)
            angles.remove(angle)
            rotate_3d = Rotate(axis=axis, angle=angle, prob=1)
            rotated_points = rotate_3d(points_raw.copy())
            save_path = os.path.join(dest_path, label, f"{name.split('.')[0]}_Rot_{angle}_Axis_{axis}.npy")
            np.save(save_path, rotated_points)
            count += 1

    if 'scale' in augmentations:
        for _ in range(4):
            axis = random.choice(scale_axes)
            factor = random.choice(scales)
            scales.remove(factor)
            scale = Scale(axis=axis, factor=factor, prob=1)
            scaled_points = scale(points_raw.copy())
            save_path = os.path.join(dest_path, label, f"{name.split('.')[0]}_Scale_{factor}_Axis_{axis}.npy")
            np.save(save_path, scaled_points)
            count += 1

    if 'noise' in augmentations:
        for _ in range(4):
            amount = 1 / random.randint(1000, 2000)
            noise = GaussianNoise(amount=amount)
            noised_points = noise(points_raw.copy())
            save_path = os.path.join(dest_path, label, f"{name.split('.')[0]}_Noise_{amount}.npy")
            np.save(save_path, noised_points)
            count += 1

    return count

def create(name, root, label, dest_path, augmentations):
    # Crea directory se non esistono
    os.makedirs(dest_path, exist_ok=True)
    os.makedirs(os.path.join(dest_path, label), exist_ok=True)

    # Legge e prepara l'immagine
    img_path = os.path.join(root, name)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img.flags.writeable = False
    
    # Estrae punti della mano
    points_raw = get_hand_points(img)
    if points_raw is not None:
        # Normalizza i punti della mano
        min_x, max_x = np.min(points_raw[:, 0]), np.max(points_raw[:, 0])
        min_y, max_y = np.min(points_raw[:, 1]), np.max(points_raw[:, 1])
        points_raw[:, 0] = (points_raw[:, 0] - min_x) / (max_x - min_x)
        points_raw[:, 1] = (points_raw[:, 1] - min_y) / (max_y - min_y)

        # Salva i punti originali
        save_path = os.path.join(dest_path, label, f"{name.split('.')[0]}.npy")
        np.save(save_path, points_raw)
        count = 1

        # Prepara parametri per le trasformazioni
        angles = list(range(-15, 15))
        scales = list(np.arange(1.0, 0.7, -0.05))
        rot_axes = ['y', 'z']
        scale_axes = ['y', 'x']

        # Crea le augmentations
        count += create_augmentations(points_raw, angles, scales, rot_axes, scale_axes, augmentations, dest_path, label, name)

        return count
    return 0

def main():
    # Percorso della cartella principale
    main_folder = r"C:\\Users\\danie\\OneDrive - uniroma1.it\\Desktop\\DATA\\dataset"
    
    # Impostazioni
    destination_path = os.path.join(main_folder, 'npy_dataset')
    augmentations = ['rot', 'scale', 'noise']  # Specifica qui le augmentazioni desiderate
    to_discard = []  # Aggiungi qui le etichette da scartare, se necessario

    # Ottieni tutti i file immagine
    image_files = glob.glob(os.path.join(main_folder, '**/*.jp*'), recursive=True) + glob.glob(os.path.join(main_folder, '**/*.png'), recursive=True)

    # Barra di progresso
    progress_bar = tqdm(total=len(image_files), desc="Creating dataset")

    count = 0
    # Scorri tutti i file immagine
    for root, _, files in os.walk(main_folder):
        for name in files:
            label = os.path.basename(root)
            if label in to_discard:
                continue
            count += create(name, root, label, destination_path, augmentations)
            progress_bar.update(1)

    print(f'Created {count} files.')

if __name__ == '__main__':
    main()
