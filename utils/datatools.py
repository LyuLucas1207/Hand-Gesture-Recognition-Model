# Hand Gesture Recognition Model - Data Utility Functions
#
# List of Functions:
#! 1. save_hand_data(dir_path, min_dir, max_dir, dir_all, min_image, max_image, image_all, output_file):
#     - Extract hand landmark data from images and save it to a file.
#
#! 2. crop_hand_images(input_dir, output_dir, margin, crop_size):
#     - Recursively process images from input_dir, crop hand regions using MediaPipe, to get speicific hand region and save processed images to output_dir


import cv2, os
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

def crop_hand_images(input_dir, output_dir, margin=0.1, greyscale=False, target_size=128):
    """
    Recursively process images from input_dir, crop hand regions using MediaPipe,
    and save processed images to output_dir while retaining the directory structure.

    Parameters:
    input_dir (str): Path to the input directory containing original images.
    output_dir (str): Path to the output directory for saving cropped images.
    margin (float): Percentage of additional margin around the hand bounding box (e.g., 0.1 for 10%).

    Returns:
    None
    """
    mp_hands = mp.solutions.hands  # Initialize MediaPipe Hands module
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    # Traverse the input directory recursively
    for root, dirs, files in os.walk(input_dir):
        # Recreate the directory structure in the output directory
        relative_path = os.path.relpath(root, input_dir)
        target_dir = os.path.join(output_dir, relative_path)
        fts.create_dir(target_dir)  # Ensure target directory exists

        for file in files:
            file_path = os.path.join(root, file)
            img = cv2.imread(file_path)

            if img is None:
                print(f"Failed to read image: {file_path}")
                continue

            # Convert image to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            # Check if hand landmarks were detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Calculate the bounding box of the hand
                    h, w, _ = img.shape
                    x_min = min([lm.x for lm in hand_landmarks.landmark]) * w
                    y_min = min([lm.y for lm in hand_landmarks.landmark]) * h
                    x_max = max([lm.x for lm in hand_landmarks.landmark]) * w
                    y_max = max([lm.y for lm in hand_landmarks.landmark]) * h

                    # Apply margin to the bounding box
                    box_width = x_max - x_min
                    box_height = y_max - y_min
                    box_size = max(box_width, box_height)  # Use the largest side for a square box

                    x_center = (x_min + x_max) / 2
                    y_center = (y_min + y_max) / 2

                    # Adjust box coordinates to make it square
                    x_min = max(0, int(x_center - box_size / 2 - margin * box_size))
                    y_min = max(0, int(y_center - box_size / 2 - margin * box_size))
                    x_max = min(w, int(x_center + box_size / 2 + margin * box_size))
                    y_max = min(h, int(y_center + box_size / 2 + margin * box_size))

                    # Crop the hand region
                    cropped_img = img[y_min:y_max, x_min:x_max]

                    # Convert the image to greyscale if required
                    if greyscale:
                        # Convert cropped image to grayscale
                        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

                    # Resize the cropped image to the target size
                    cropped_img = cv2.resize(cropped_img, (target_size, target_size))
                    
                    # Save the cropped image to the target directory
                    output_path = os.path.join(target_dir, file)
                    cv2.imwrite(output_path, cropped_img)
                    print(f"Processed and saved: {output_path}")

    hands.close()
    print(f"All images processed and saved to {output_dir}")
