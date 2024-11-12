# How to run: python -m ui.image_showing

from utils import graphtools as gts

# show image


def main():
    min_dir = 0  # from directory 0
    max_dir = 4  # to directory 3
    dir_all = False
    min_image = 0  # from image 0
    max_image = 1  # to image 1
    image_all = False
    dir_path = "./images"

    gts.show_image(dir_path, min_dir, max_dir, dir_all, min_image, max_image, image_all)

    gts.show_hand(dir_path, min_dir, max_dir, dir_all, min_image, max_image, image_all)


# Run the main function
if __name__ == "__main__":
    main()
