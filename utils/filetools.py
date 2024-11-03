# filetools.py

import os, cv2

def create_dir(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    return dir_

def get_max_list_folder(folder_path):
    max_number = -1  # initialize to -1 to handle empty folder case
    folder_list = []
    for folder_name in os.listdir(folder_path):
        folder_path_full = os.path.join(folder_path, folder_name)
        if os.path.isdir(folder_path_full) and folder_name.isdigit():
            max_number = max(max_number, int(folder_name))
            folder_list.append(folder_name)
    return max_number, folder_list

def get_max_list_image(folder_path):
    max_number = -1  # 初始化为 -1，处理空文件夹情况
    image_list = []
    
    for file_name in os.listdir(folder_path):
        file_path_full = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path_full) and file_name.split('.')[0].isdigit():
            number = int(file_name.split('.')[0])
            max_number = max(max_number, number)
            image_list.append(number)
    
    return max_number, image_list

def show_text(
        cap,
        content = 'Press "S" to start collecting data for class {}'.format,
        start = (100, 50),
        font_type = cv2.FONT_HERSHEY_SIMPLEX,
        font_scale = 1.0,
        font_color = (0, 0, 0),
        font_thickness = 3,
        line_type = cv2.LINE_AA
    ):
    while True:
        ret, frame = cap.read()
        if ret == False:
            print('Camera not found')
            break
        cv2.putText(frame, content, start, font_type, font_scale, font_color, font_thickness, line_type)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
        
def write_images(cap, start_num, data_size, DATA_DIR, folder):
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


