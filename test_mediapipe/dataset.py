import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define data directory
DATA_DIR = 'C:/Users/danie/OneDrive - uniroma1.it/Desktop/DATA/dataset'

data = []
labels = []

# Get list of subfolders
subfolders = [dir_ for dir_ in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, dir_))]

# Count total images for the progress bar
total_images = sum([len(files) for r, d, files in os.walk(DATA_DIR)])

# Process images with progress bar
with tqdm(total=total_images, desc='Processing images', unit='img') as pbar:
    for dir_ in subfolders:
        dir_path = os.path.join(DATA_DIR, dir_)
        for img_path in os.listdir(dir_path):
            data_aux = []
            x_ = []
            y_ = []

            img = cv2.imread(os.path.join(dir_path, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    if len(hand_landmarks.landmark) == 21:
                        for landmark in hand_landmarks.landmark:
                            x_.append(landmark.x)
                            y_.append(landmark.y)

                        for landmark in hand_landmarks.landmark:
                            data_aux.append(landmark.x - min(x_))
                            data_aux.append(landmark.y - min(y_))

                        # Add padding if necessary
                        while len(data_aux) < 42:
                            data_aux.append(0.0)

                        data.append(data_aux)
                        labels.append(dir_)

            pbar.update(1)

# Save data and labels to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

# Load data and labels
data_dict = pickle.load(open("C:\\Users\\danie\\OneDrive - uniroma1.it\\Desktop\\Project-CV\\data.pickle", 'rb'))

data = np.array([np.array(seq) if len(seq) == 42 else np.pad(seq, (0, 42 - len(seq)), 'constant') for seq in data_dict['data']], dtype=float)
labels = np.array(data_dict['labels'])

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict and evaluate the model
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
