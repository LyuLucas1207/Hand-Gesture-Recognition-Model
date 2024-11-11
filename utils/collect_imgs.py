"""
collect_imgs Function:

This function is used to collect images for multiple classes using a video feed. It saves the captured images into organized directories based on class names.

Parameters:
!- number_of_classes (list): A list of integers representing class labels. Each class will have its own folder for storing images. Default is [0, 1, 2].
!- dataset_size (int): The number of images to collect for each class. Default is 100.
!- data_dir (str): The base directory where the class folders and images will be stored. Default is './data'.
!- video_source (int or str): The source of the video feed. Default is 0 (default webcam). It can also accept a video file path.

Workflow:
1. Creates the base `data_dir` and individual folders for each class in `number_of_classes`.
2. Displays a prompt to start collecting images for a class using `show_text`.
3. Determines the maximum image number in the current class folder to avoid overwriting existing images.
4. Captures images from the video feed and saves them to the respective class folder.
5. Releases the video capture object and closes all OpenCV windows at the end.

Dependencies:
- Requires the `utils.filetools` module for utility functions like `create_dir`, `show_text`, `get_max_list_image`, and `write_images`.
"""
import os
import cv2

import utils.filetools as fts
import utils.graphtools as gts

def collect_imgs(number_of_classes = [0, 1, 2], dataset_size=100, data_dir='./images', video_source=0):
    DATA_DIR = fts.create_dir(data_dir)

    #! Create a VideoCapture object
    cap = cv2.VideoCapture(video_source) 
    for folder in number_of_classes:
        IMAGE_DIR = fts.create_dir(os.path.join(DATA_DIR, str(folder)))
        print('Collecting data for class {}'.format(folder))

        content = 'Press "S" to start collecting data for class {}'.format(folder)
        gts.show_text(cap, content)

        max_images, _ = fts.get_max_list_image(IMAGE_DIR)
        print('Max number: {}'.format(max_images))
        fts.write_images(cap, max_images+1, dataset_size, DATA_DIR, folder)

    # Release the VideoCapture object
    cap.release() 
    # Close all OpenCV windows
    cv2.destroyAllWindows() 



