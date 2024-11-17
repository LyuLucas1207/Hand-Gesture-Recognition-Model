# How to run: python -m interface.pytorch_classifier2

import cv2
import numpy as np
import torch
from torch.nn.functional import softmax
from enums.labels_dict import labels_dict
import utils.graphtools as gt
from utils.modeltools import pytorch_load_full_model

# 加载 PyTorch 模型
num_classes = len(labels_dict)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = pytorch_load_full_model("./models/cnn_model_pytorch.pth", device)
model.eval()

# 初始化相机
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera initialization failed.")
    exit()

# 初始化 MediaPipe Hands
hands, mp_hands, mp_drawing, mp_drawing_styles = gt.init_hands(
    static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5
)

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to capture frame. Exiting...")
        break

    H, W, _ = frame.shape

    # Convert BGR to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Collect hand landmark coordinates
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            # Calculate bounding box
            x_min = int(min(x_coords) * W)
            y_min = int(min(y_coords) * H)
            x_max = int(max(x_coords) * W)
            y_max = int(max(y_coords) * H)

            # Add margin to the bounding box
            margin = 10
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(W, x_max + margin)
            y_max = min(H, y_max + margin)

            # Crop the hand region
            hand_img = frame[y_min:y_max, x_min:x_max]

            # Resize to model input size (128x128)
            hand_img_resized = cv2.resize(hand_img, (128, 128))

            # Convert to grayscale and normalize
            # hand_img_gray = cv2.cvtColor(hand_img_resized, cv2.COLOR_BGR2GRAY)
            # hand_img_normalized = hand_img_gray / 255.0
            # hand_img_tensor = torch.tensor(hand_img_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            # # Predict using the PyTorch model
            # with torch.no_grad():
            #     outputs = model(hand_img_tensor)
            #     probabilities = softmax(outputs, dim=1).cpu().numpy()
            #     predicted_index = np.argmax(probabilities)
            #     predicted_character = labels_dict[predicted_index]

            hand_img_rgb = cv2.cvtColor(hand_img_resized, cv2.COLOR_BGR2RGB)
            hand_img_normalized = hand_img_rgb / 255.0
            hand_img_tensor = torch.tensor(hand_img_normalized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

            # Predict using the PyTorch model
            with torch.no_grad():
                outputs = model(hand_img_tensor)
                probabilities = softmax(outputs, dim=1).cpu().numpy()
                predicted_index = np.argmax(probabilities)
                predicted_character = labels_dict[predicted_index]

            # Draw bounding box and prediction label
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(
                frame,
                predicted_character,
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    # Display the frame
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
