# How to run: python -m ui.image_process
from utils import datatools as dt


def main():
    dir_path = "./images"
    min_dir = 0
    max_dir = 1  # non-inclusive
    dir_all = True
    min_image = 0
    max_image = 1  # non-inclusive
    image_all = True
    output_file = "./data/data.pickle"

    dt.save_hand_data(
        dir_path,
        min_dir,
        max_dir,
        dir_all,
        min_image,
        max_image,
        image_all,
        output_file,
    )


# run the main() function

if __name__ == "__main__":
    main()
