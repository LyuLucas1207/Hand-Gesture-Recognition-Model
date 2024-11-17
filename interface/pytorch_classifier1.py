# How to run: python -m interface.pytorch_classifier

import cv2
import numpy as np
import torch
from torch.nn.functional import softmax
from torch import nn
from torch.utils.data import DataLoader
import enums.labels_dict as ld
import utils.graphtools as gt
from environments.cnn_pytorch_trainer1 import CNNModel

# 初始化模型并加载权重
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# num_classes = len(ld.labels_dict)  # 获取类别数量
# model = CNNModel(num_classes=num_classes).to(device)
# model.load_state_dict(torch.load("./models/cnn_advanced.pth"))
# model.eval()

# 加载完整模型
model = torch.load("./models/cnn_advanced_complete.pth")
model.eval()  # 切换到评估模式


# 初始化相机
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera initialization failed.")
    exit()

# 初始化 MediaPipe Hands
hands, mp_hands, mp_drawing, mp_drawing_styles = gt.init_hands(
    static_image_mode=True, min_detection_confidence=0.3
)

# 加载标签字典
labels_dict = ld.labels_dict

while True:
    data_aux = []
    x_ = []
    y_ = []

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
            # Draw landmarks for each hand
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

            # Collect landmark data for this hand
            x_, y_ = [], []
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            # Compute normalized features for CNN prediction
            data_aux = []
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            # Reshape and normalize features
            data_aux = np.asarray(data_aux).reshape(1, 1, 6, 7).astype("float32")
            data_aux = torch.tensor(data_aux, device=device)

            # Predict the gesture using CNN
            with torch.no_grad():
                prediction = model(data_aux)
                prediction = softmax(prediction, dim=1)
                predicted_character = labels_dict[torch.argmax(prediction).item()]

            # Calculate bounding box for this hand
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                predicted_character,
                (x1, y1 - 10),
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

# 释放资源
cap.release()
cv2.destroyAllWindows()
