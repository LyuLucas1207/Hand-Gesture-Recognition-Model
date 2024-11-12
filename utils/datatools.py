# Hand Gesture Recognition Model - Data Utility Functions
#
# List of Functions:
#! 1. save_hand_data(dir_path, min_dir, max_dir, dir_all, min_image, max_image, image_all, output_file):
#     - Extract hand landmark data from images and save it to a file.
#


import cv2
import pickle
import mediapipe as mp
import utils.filetools as fts


def save_hand_data(
    dir_path="./images",
    min_dir=0,
    max_dir=1,
    dir_all=False,
    min_image=0,
    max_image=1,
    image_all=False,
    output_file="./data/data.pickle",
):
    """
    Extract hand landmark data from images and save it to a file.

    Parameters:
    dir_path (str): Base directory containing image folders.
    min_dir (int): The minimum directory index to process.
    max_dir (int): The maximum directory index to process (non-inclusive).
    dir_all (bool): If True, process all directories regardless of min_dir and max_dir.
    min_image (int): The minimum image index to process.
    max_image (int): The maximum image index to process (non-inclusive).
    image_all (bool): If True, process all images in each directory.
    output_file (str): The name of the file to save the processed data.

    Returns:
    None
    """
    mp_hands = mp.solutions.hands  # Import the Mediapipe Hands module

    # Initialize Mediapipe Hands model
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    data = []
    labels = []

    # Get directories and images to process
    processed_data = fts.process_directories(
        dir_path, min_dir, max_dir, dir_all, min_image, max_image, image_all
    )

    for dir_, image_paths in processed_data:
        for img_path in image_paths:
            data_aux = []
            x_ = []
            y_ = []

            # Read and process image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to read image: {img_path}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                data.append(data_aux)
                labels.append(dir_)

    # Save data to file
    with open(output_file, "wb") as f:
        pickle.dump({"data": data, "labels": labels}, f)

    print(f"Data saved to {output_file}")
    hands.close()
