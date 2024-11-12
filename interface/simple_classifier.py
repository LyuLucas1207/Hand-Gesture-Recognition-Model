#How to run: python -m ui.image_process

import cv2
import numpy as np
import enums.labels_dict as ld
import utils.modeltools as mt
import utils.graphtools as gt
import enums.labels_dict as ld

SVM = ld.models_dict['svm']
RANDOM_FOREST = ld.models_dict['random_forest']

# Load the trained model
# model, _ = mt.load_model(RANDOM_FOREST)
model, _ = mt.load_model(SVM)

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
hands, mp_hands, mp_drawing, mp_drawing_styles = gt.init_hands(static_image_mode=True, min_detection_confidence=0.3)

# Load label dictionary
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
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Collect landmark data for this hand
            x_, y_ = [], []
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            # Compute normalized features for prediction
            data_aux = []
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            # Calculate bounding box for this hand
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            # Predict the gesture
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
                        cv2.LINE_AA)

    # Display the frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
