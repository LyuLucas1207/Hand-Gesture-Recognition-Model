# Hand Gesture Recognition Model - File Utility Functions
# 
# List of Functions:
#! 1. create_dir(dir_): 
#     - Creates a directory if it does not exist.
#
#! 2. get_max_list_folder(folder_path): 
#     - Returns the maximum numeric folder name and a list of all folder names in a directory.
#
#! 3. get_max_list_image(folder_path): 
#     - Returns the maximum numeric image file name and a list of numeric file names in a folder.
#
#! 4. write_images(cap, start_num, data_size, DATA_DIR, folder): 
#     - Captures and saves images from a video feed.
#
#! 5. process_directories(dir_path, min_dir, max_dir, dir_all, min_image, max_image, image_all):
#     - Processes directories and images based on the provided range or settings.
#

import os, cv2

def create_dir(dir_):
    """
    Create a directory if it does not exist.

    Parameters:
    dir_ (str): The path of the directory to be created.

    Returns:
    str: The path of the created or existing directory.
    """

    if not os.path.exists(dir_):
        os.makedirs(dir_)
    return dir_

def get_max_list_folder(folder_path):
    """
    Get the maximum numeric folder name and a list of all folder names in a directory.

    Parameters:
    folder_path (str): The path of the directory to check.

    Returns:
    tuple:
        - int: The maximum numeric folder name (returns -1 if no numeric folders exist).
        - list: A list of all folder names in the directory.
    """
    max_number = -1  # initialize to -1 to handle empty folder case
    folder_list = []
    for folder_name in os.listdir(folder_path):
        folder_path_full = os.path.join(folder_path, folder_name)
        if os.path.isdir(folder_path_full) and folder_name.isdigit():
            max_number = max(max_number, int(folder_name))
            folder_list.append(folder_name)
    return max_number, folder_list

def get_max_list_image(folder_path):
    """
    Get the maximum numeric image file name (without extension) and a list of numeric file names in a folder.

    Parameters:
    folder_path (str): The path of the directory to check.

    Returns:
    tuple:
        - int: The maximum numeric image file name (returns -1 if no numeric files exist).
        - list: A list of all numeric image file names in the directory.
    """
    max_number = -1  # 初始化为 -1，处理空文件夹情况
    image_list = []
    
    for file_name in os.listdir(folder_path):
        file_path_full = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path_full) and file_name.split('.')[0].isdigit():
            number = int(file_name.split('.')[0])
            max_number = max(max_number, number)
            image_list.append(number)
    
    return max_number, image_list
        
def write_images(cap, start_num, data_size, DATA_DIR, folder):
    """
    Capture and save images from a video feed.

    Parameters:
    cap (cv2.VideoCapture): The video capture object.
    start_num (int): The starting number for naming the saved images.
    data_size (int): The total number of images to save.
    DATA_DIR (str): The base directory where images will be saved.
    folder (str): The folder name within DATA_DIR to save the images.

    Returns:
    None
    """
    counter = 0
    image_name = '{}.jpg'.format
    while counter < data_size:
        path = os.path.join(DATA_DIR, str(folder), image_name(start_num + counter))
        ret, frame = cap.read()
        if ret == False or frame is None:
            print('Camera not found')
            break

        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(path, frame)
        counter += 1

def process_directories(dir_path, min_dir, max_dir, dir_all, min_image, max_image, image_all):
    """
    Process directories and images based on the provided range or settings.

    Parameters:
    dir_path (str): Base directory path.
    min_dir (int): Minimum directory index.
    max_dir (int): Maximum directory index.
    dir_all (bool): If True, use all directories.
    min_image (int): Minimum image index.
    max_image (int): Maximum image index.
    image_all (bool): If True, use all images in each directory.
    Returns:
    list: List of tuples with directory name and image file paths.
    """
    # Determine directory range
    # dir_list = os.listdir(dir_path)
    # sort the directories in ascending order
    dir_list = sorted(
        [d for d in os.listdir(dir_path) if d.isdigit()],
        key=lambda x: int(x)
    )

    if dir_all:
        min_dir = 0
        max_dir = len(dir_list)
    dir_list = dir_list[min_dir:max_dir]

    processed_data = []

    # Iterate over directories
    for dir_ in dir_list:
        folder_path = os.path.join(dir_path, dir_)
        if not os.path.isdir(folder_path):
            continue

        # Determine image range
        image_list = os.listdir(folder_path)
        if image_all:
            min_image = 0
            max_image = len(image_list)
        image_list = image_list[min_image:max_image]

        # Collect directory and image data
        processed_data.append((dir_, [os.path.join(folder_path, img) for img in image_list]))
        
    # print(f"Processed directory: {processed_data}")
    return processed_data