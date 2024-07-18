import cv2
import mediapipe as mp
import numpy as np
import os
import shutil

# Inizializza Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Dizionario per la mappatura dei caratteri agli interi
char2int = {
    "a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7, "i": 8, "j": 9, "k": 10, "l": 11, "m": 12,
    "n": 13, "o": 14, "p": 15, "q": 16, "r": 17, "s": 18, "t": 19, "u": 20, "v": 21, "w": 22, "x": 23, "y": 24, "z": 25,
    "nothing": 26, "del": 27, "space": 28
}

# Funzione per ottenere i punti della mano
def get_hand_points(img):
    results = hands.process(img)
    points = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(21):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                z = hand_landmarks.landmark[i].z
                points.append([x, y, z])
    else:
        border_size = 100
        img = cv2.copyMakeBorder(
            img,
            top=border_size,
            bottom=border_size,
            left=border_size,
            right=border_size,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )
        results = hands.process(img)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(21):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    z = hand_landmarks.landmark[i].z
                    points.append([x, y, z])
        else:
            points = None

    if points is not None:
        if len(points) > 21:
            points = points[:21]
        elif len(points) < 21:
            dif = 21 - len(points)
            for i in range(dif):
                points.append([0, 0, 0])

        points = np.array(points)

    return points

# Funzione per cancellare una cartella
def clean_folder(folder):
    shutil.rmtree(folder)

# Funzione per controllare il dataset
def check_dataset(root):
    width = 640
    height = 480
    for root, dirs, files in os.walk(root):
        for file in files:
            img = np.zeros([height, width, 3], dtype=np.uint8)
            img.fill(255)
            print(file)
            points = np.load(os.path.join(root, file))
            for pp in mp_hands.HAND_CONNECTIONS:
                cv2.line(img, (int((points[pp[0]][0]) * width), int((points[pp[0]][1]) * height)),
                         (int((points[pp[1]][0]) * width), int((points[pp[1]][1]) * height)), (0, 0, 255), 4)
            cv2.imshow('', img)
            cv2.waitKey(0)

# Funzione principale
def main():
    main_folder = r"C:\\Users\\danie\\OneDrive - uniroma1.it\\Desktop\\DATA\\dataset"
    subfolders = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        'del', 'nothing', 'space'
    ]

    # Esegui operazioni su ciascuna sottocartella
    for subfolder in subfolders:
        folder_path = os.path.join(main_folder, subfolder)
        if not os.path.exists(folder_path):
            continue

        # Esempio: Stampa i punti della mano per ogni immagine nella sottocartella
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                hand_points = get_hand_points(img)
                print(f"Hand points for {filename} in {subfolder}: {hand_points}")

if __name__ == '__main__':
    main()
