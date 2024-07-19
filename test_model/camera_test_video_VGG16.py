import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from hand_gesture_CNN_model import HandGestureCNN
from VGG16_hande_gesture_detection_model import HandGestureVGG16

num_classes = 29
model_path = "C:\\Users\\danie\\OneDrive - uniroma1.it\\Desktop\\2_VGG16_only_landmark_hand_gesture_cnn_with_metrics.pth"
model = HandGestureVGG16(num_classes)

# load of trained model
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
          'del', 'nothing', 'space']

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((480, 640)),  
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def remove_background_and_keep_landmarks(frame, results):
    H, W, _ = frame.shape
    landmarks_image = np.zeros_like(frame)

    for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            landmarks_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
        )

    gray_landmarks_image = cv2.cvtColor(landmarks_image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_landmarks_image, 1, 255, cv2.THRESH_BINARY)

    result = cv2.bitwise_and(frame, frame, mask=mask)

    white_background = np.ones_like(frame, dtype=np.uint8) * 255
    result_with_white_bg = np.where(result == 0, white_background, result)
    
    return result_with_white_bg

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (640, 480))

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        processed_frame = remove_background_and_keep_landmarks(frame, results)

        for hand_landmarks in results.multi_hand_landmarks:
            # compute bounding box
            x_ = [landmark.x for landmark in hand_landmarks.landmark]
            y_ = [landmark.y for landmark in hand_landmarks.landmark]

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W, x2)
            y2 = min(H, y2)

            hand_img = processed_frame[y1:y2, x1:x2]

            if hand_img.size == 0:  
                continue

            hand_img = transform(hand_img)
            hand_img = hand_img.unsqueeze(0)  

            with torch.no_grad():
                output = model(hand_img)
                _, predicted = torch.max(output, 1)
                predicted_character = labels[predicted.item()]

            # draw bounding box with label prediction on the output image
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

            cv2.putText(processed_frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('Original Frame', frame)


    if results.multi_hand_landmarks:
        cv2.imshow('Landmarks Frame', processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
