# How to run: python -m ui.image_collection
from utils import videotools as vt

# Collects images from the webcam and saves them in the data directory


def main():
    #!number of classes
    number_of_classes = [33]
    dataset_size = 100
    #!relative path from the root directory
    data_dir = "./images"
    vt.collect_imgs(number_of_classes, dataset_size, data_dir)


# 运行 main() 函数
if __name__ == "__main__":
    main()
