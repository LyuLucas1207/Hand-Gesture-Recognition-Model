# 在 image_collection.py 中
from utils import collect_imgs as ci

# Collects images from the webcam and saves them in the data directory

def main():
    number_of_classes = [0, 1, 2]
    dataset_size = 100
    data_dir = './data' #!relative path from the root directory
    ci.collect_imgs(number_of_classes, dataset_size, data_dir)

# 运行 main() 函数
if __name__ == "__main__":
    main()
