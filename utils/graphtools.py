# Hand Gesture Recognition Model - Graph Utility Functions
#
#! 1. show_text(cap, content, start, font_type, font_scale, font_color, font_thickness, line_type):
#     - Displays text on a video feed until the user presses the 'S' key.
#! 2. show_image(dir_path, min_dir, max_dir, dir_all, min_image, max_image, image_all):
#     - Displays images from the specified directories and image numbers.
#! 3. show_hand(dir_path, min_dir, max_dir, dir_all, min_image, max_image, image_all):
#     - Displays hand landmarks on images from specified directories.
#! 4. display_image(img_rgb, title):
#     - Display an image using Matplotlib.
#! 5. init_hands(static_image_mode, min_detection_confidence):
#     - Initialize MediaPipe hand detection modules.
#! 6. draw_landmarks(img_rgb, hands, mp_hands, mp_drawing, mp_drawing_styles):
#     - Draw hand landmarks on an image.
#

import os, cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import utils.filetools as fts


def show_text(
    cap,
    content='Press "S" to start collecting data for class {}'.format,
    start=(50, 50),
    font_type=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=1.0,
    font_color=(0, 0, 0),
    font_thickness=3,
    line_type=cv2.LINE_AA,
):
    while True:
        ret, frame = cap.read()
        if ret == False:
            print("Camera not found")
            break
        cv2.putText(
            frame,
            content,
            start,
            font_type,
            font_scale,
            font_color,
            font_thickness,
            line_type,
        )
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("s"):
            break


def show_image(
    dir_path="./images",
    min_dir=0,
    max_dir=1,
    dir_all=False,
    min_image=0,
    max_image=1,
    image_all=False,
):
    """
    Display images from the specified directories and image numbers.
    """
    data = fts.process_directories(
        dir_path, min_dir, max_dir, dir_all, min_image, max_image, image_all
    )

    for dir_, image_paths in data:
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to read image: {img_path}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            display_image(
                img_rgb, f"Directory: {dir_}, Image: {os.path.basename(img_path)}"
            )

    plt.show()


def show_hand(
    dir_path="./images",
    min_dir=0,
    max_dir=1,
    dir_all=False,
    min_image=0,
    max_image=1,
    image_all=False,
):
    """
    Display hand landmarks on images from specified directories and image ranges.
    """
    # Initialize Mediapipe hand detection modules
    hands, mp_hands, mp_drawing, mp_drawing_styles = init_hands(True, 0.5)

    data = fts.process_directories(
        dir_path, min_dir, max_dir, dir_all, min_image, max_image, image_all
    )

    for dir_, image_paths in data:
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to read image: {img_path}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb = draw_landmarks(
                img_rgb, hands, mp_hands, mp_drawing, mp_drawing_styles
            )

            display_image(
                img_rgb, f"Directory: {dir_}, Image: {os.path.basename(img_path)}"
            )

    hands.close()
    plt.show()


def display_image(img_rgb, title):
    """
    Display an image using Matplotlib.

    Parameters:
    img_rgb (np.array): Image in RGB format.
    title (str): Title of the image.

    Returns:
    None
    """
    plt.figure(figsize=[10, 10])
    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis("off")


def init_hands(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=True, 
        min_detection_confidence=0.5,
        min_tracking_confidence=min_tracking_confidence
        )

    return hands, mp_hands, mp_drawing, mp_drawing_styles


def draw_landmarks(img_rgb, hands, mp_hands, mp_drawing, mp_drawing_styles):
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img_rgb,  # Image to draw landmarks on
                hand_landmarks,  # Hand landmarks to draw, (x, y, z)
                mp_hands.HAND_CONNECTIONS,  # Connections between the landmarks
                mp_drawing_styles.get_default_hand_landmarks_style(),  # Drawing style for landmarks
                mp_drawing_styles.get_default_hand_connections_style(),  # Drawing style for connections
            )
            # from mediapipe.python.solutions.drawing_utils import DrawingSpec
            # mp_drawing.draw_landmarks(
            #     img_rgb,
            #     hand_landmarks,
            #     mp_hands.HAND_CONNECTIONS,
            #     DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),  # self-defined landmark style
            #     DrawingSpec(color=(255, 0, 0), thickness=3)  # self-defined connection style
            #
    return img_rgb
