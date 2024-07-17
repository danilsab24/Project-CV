import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import mediapipe as mp
from tqdm import tqdm

# Disabilitare XNNPACK in TensorFlow Lite
os.environ['TF_DISABLE_MLIR'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_XNNPACK_DELEGATE_NO'] = '1'

def detect_hand_and_save(image_path, label, output_dir):
    # Initialize Mediapipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image {image_path}")
        hands.close()
        return

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and find hands
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get bounding box coordinates
            x_min, y_min = float('inf'), float('inf')
            x_max, y_max = float('-inf'), float('-inf')
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                if x < x_min: x_min = x
                if y < y_min: y_min = y
                if x > x_max: x_max = x
                if y > y_max: y_max = y

            # Draw bounding box and label
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Optionally, draw landmarks
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Save the result image
    os.makedirs(output_dir, exist_ok=True)
    result_image_name = os.path.splitext(os.path.basename(image_path))[0] + "_label.jpg"
    result_image_path = os.path.join(output_dir, result_image_name)
    
    # Skip saving if the result image already exists
    if os.path.exists(result_image_path):
        print(f"Image {result_image_path} already exists. Skipping.")
        hands.close()
        return

    cv2.imwrite(result_image_path, image)
    hands.close()

def process_directory(input_dir, output_base_dir):
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
              'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
              'del', 'nothing', 'space']

    for label in labels:
        input_label_dir = os.path.join(input_dir, label)
        output_label_dir = os.path.join(output_base_dir, f"{label}_label")
        if os.path.exists(input_label_dir):
            for root, _, files in os.walk(input_label_dir):
                for image_name in tqdm(files, desc=f"Processing {label}", unit="image"):
                    image_path = os.path.join(root, image_name)
                    detect_hand_and_save(image_path, label, output_label_dir)

if __name__ == "__main__":
    input_dir = "C:/Users/danie/OneDrive - uniroma1.it/Desktop/camera_test/input"
    output_base_dir = "C:/Users/danie/OneDrive - uniroma1.it/Desktop/camera_test/output"
    process_directory(input_dir, output_base_dir)
