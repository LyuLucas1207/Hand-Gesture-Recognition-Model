# How to run: python -m ui.image_process

from utils import datatools as dt
from utils.printtools import print_boxed_message, print_boxed_options

def get_user_input(prompt, default_value, input_type=str):
    """
    Get user input with a default value.

    Parameters:
    prompt (str): The prompt to display to the user.
    default_value (any): The default value to use if no input is provided.
    input_type (type): The type to which the input should be converted.

    Returns:
    any: The user input converted to the specified type, or the default value.
    """
    user_input = input(f"{prompt} [Default: {default_value}]: ").strip()
    if user_input == "":
        return default_value
    try:
        return input_type(user_input)
    except ValueError:
        print(f"Invalid input. Using default value: {default_value}")
        return default_value

def main():
    # Display a welcome message
    print_boxed_message("Hand Gesture Recognition - Main Menu", width=80)

    # Define menu options
    options = {
        1: "Save Hand Data (Extract and Save Hand Landmark Data)",
        2: "Crop Hand Images (Extract Hand Regions and Save Cropped Images)",
    }

    while True:
        # Print menu options
        print_boxed_options(options, additional_option="Other. Exit", width=80)

        # Prompt user for input
        choice = input("Enter your choice (1, 2, or other to exit): ").strip()

        if choice == "1":
            print_boxed_message("Executing Save Hand Data...", width=80)
            # Ask user for parameters with default values
            min_dir = get_user_input("Enter min_dir", 0, int)
            max_dir = get_user_input("Enter max_dir (non-inclusive)", 1, int)
            dir_all = get_user_input("Process all directories? (True/False)", True, lambda x: x.lower() == "true")
            min_image = get_user_input("Enter min_image", 0, int)
            max_image = get_user_input("Enter max_image (non-inclusive)", 1, int)
            image_all = get_user_input("Process all images? (True/False)", True, lambda x: x.lower() == "true")

            # Execute the save_hand_data function
            dt.save_hand_data(
                dir_path="./images",
                min_dir=min_dir,
                max_dir=max_dir,
                dir_all=dir_all,
                min_image=min_image,
                max_image=max_image,
                image_all=image_all,
                output_file="./data/data.pickle",
            )
        elif choice == "2":
            print_boxed_message("Executing Crop Hand Images...", width=80)
            margin = get_user_input("Enter margin (e.g., 0.1 for 10%)", 0.1, float)
            greyscale = get_user_input("Convert images to grayscale? (True/False)", False, lambda x: x.lower() == "true")
            target_size = get_user_input("Enter target size (e.g., 224)", 128, int)

            # Execute the crop_hand_images function
            dt.crop_hand_images(
                input_dir="./images",
                output_dir="./data/images",
                margin=margin,
                greyscale=greyscale,
                target_size=target_size,
            )
        else:
            print_boxed_message("Exiting the program. Goodbye!", width=80)
            break  # Exit the loop

if __name__ == "__main__":
    main()
