import os
import cv2

import utils.filetools as fts

def collect_imgs(number_of_classes = [0, 1, 2], dataset_size=100, data_dir='./data'):
    DATA_DIR = fts.create_dir(data_dir)

    cap = cv2.VideoCapture(0) #! Change this to 0 if you are using a webcam
    for folder in number_of_classes:
        IMAGE_DIR = fts.create_dir(os.path.join(DATA_DIR, str(folder)))
        print('Collecting data for class {}'.format(folder))

        content = 'Press "S" to start collecting data for class {}'.format(folder)
        fts.show_text(cap, content)

        max_images, _ = fts.get_max_list_image(IMAGE_DIR)
        print('Max number: {}'.format(max_images))
        fts.write_images(cap, max_images+1, dataset_size, DATA_DIR, folder)

    cap.release() # Release the VideoCapture object
    cv2.destroyAllWindows() # Close all OpenCV windows



