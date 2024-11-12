# Hand Gesture Recognition Model - Image Collection and Depiction Utilities
#
# This module provides utility functions for:
# - Collecting images for different gesture classes from a video feed.
# - Visualizing hand landmarks in real-time using the MediaPipe library.
#
# List of Functions:
#
# 1. collect_imgs(number_of_classes, dataset_size, data_dir, video_source):
#    - Collects images for specified gesture classes using a video feed.
#    - Saves the images into class-specific folders within a base directory.
#
# 2. hand_depiction(static_image_mode, min_detection_confidence):
#    - Captures real-time video from a camera and detects hand landmarks.
#    - Draws hand landmarks and connections on the video feed in real-time.
#
# Prerequisites:
# - OpenCV for capturing and displaying video.
# - MediaPipe for hand landmark detection.
# - Utility functions from `utils.filetools` and `utils.graphtools` for file handling and landmark processing.
#
# How to Use:
# - Run `collect_imgs()` to collect gesture class images.
# - Run `hand_depiction()` to visualize hand landmarks.
#
# How to Execute:
# - Save this module in your project and call the functions in your script.

import os
import cv2

import utils.filetools as fts
import utils.graphtools as gts


def collect_imgs(number_of_classes=[0, 1, 2], dataset_size=100, data_dir='./images', video_source=0):
    """
    Collects images for specified classes using a video feed.

    Parameters:
    number_of_classes (list): List of integers representing class labels. Each class will have its own folder for storing images.
    dataset_size (int): The number of images to collect for each class.
    data_dir (str): The base directory where images will be saved.
    video_source (int): The index of the video capture device (e.g., 0 for default camera).

    Returns:
    None
    """
    # Create the base data directory if it doesn't exist
    DATA_DIR = fts.create_dir(data_dir)

    # Create a VideoCapture object to access the video feed
    cap = cv2.VideoCapture(video_source)

    # Iterate over the specified class labels
    for folder in number_of_classes:
        # Create a subfolder for each class
        IMAGE_DIR = fts.create_dir(os.path.join(DATA_DIR, str(folder)))
        print('Collecting data for class {}'.format(folder))

        # Display instructions to start collecting images
        content = 'Press "S" to start collecting data for class {}'.format(folder)
        gts.show_text(cap, content)

        # Get the maximum image number in the folder to avoid overwriting
        max_images, _ = fts.get_max_list_image(IMAGE_DIR)
        print('Max number: {}'.format(max_images))

        # Capture and save images for the current class
        fts.write_images(cap, max_images + 1, dataset_size, DATA_DIR, folder)

    # Release the video capture object
    cap.release()
    # Close all OpenCV windows
    cv2.destroyAllWindows()


def hand_depiction(static_image_mode=True, min_detection_confidence=0.5):
    """
    Captures real-time video from a camera and detects hand landmarks.

    Parameters:
    static_image_mode (bool): Whether to treat input as a batch of static images or a video stream.
    min_detection_confidence (float): Minimum confidence value (0.0 to 1.0) for hand detection.

    Returns:
    None
    """
    # Create a VideoCapture object to access the video feed
    cap = cv2.VideoCapture(0)

    # Initialize MediaPipe Hands and related utilities
    hands, mp_hands, mp_drawing, mp_drawing_styles = gts.init_hands(static_image_mode, min_detection_confidence)

    while True:
        # Read a frame from the video feed
        ret, frame = cap.read()
        if not ret:
            print("Camera not found")
            break

        # Correct color conversion: BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect and draw hand landmarks
        frame_rgb = gts.draw_landmarks(frame_rgb, hands, mp_hands, mp_drawing, mp_drawing_styles)

        # Convert back to BGR for OpenCV display
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Display the processed frame
        cv2.imshow('frame', frame_bgr)

        # Exit loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object
    cap.release()
    # Close all OpenCV windows
    cv2.destroyAllWindows()
