# Hand Gesture Recognition Model - Print Utility Functions
#
# This module provides utility functions for printing formatted and visually appealing messages or options in the console.
# It includes functions for displaying aligned key-value pairs, boxed messages, and neatly formatted options
# for user interaction.
#
# List of Functions:
#! 1. print_aligned_model_config(model_config)
#       - Prints aligned key-value pairs for a configuration dictionary.
#
#! 2. print_boxed_message(message, width=100)
#       -Displays a centered boxed message with a customizable width.
#
#! 3. print_boxed_options(options, additional_option="Other. Exit", width=100)
#       -Prints a boxed menu of numbered options.
#
#! 4. print_boxed_yes_no(prompt, option_1="Yes", option_2="No", width=100)
#       -Displays a boxed yes/no prompt for user interaction.


def print_aligned_model_config(model_config):
    """
    Prints aligned key-value pairs for a configuration dictionary.

    Parameters:
    model_config (dict): A dictionary containing key-value pairs to display.

    Example Output:
    key1   : value1
    key2   : value2
    """
    max_key_length = max(len(key) for key in model_config.keys())
    for key, value in model_config.items():
        # print(f"{key.ljust(max_key_length)}: {value}")
        # determine if the value is a dictionary again
        if isinstance(value, dict):
            # 缩进一个tab
            print(f"{key.ljust(max_key_length)}: ")
            for k, v in value.items():
                print(f"    {k.ljust(max_key_length)}: {v}")

        else:
            print(f"{key.ljust(max_key_length)}: {value}")


def print_boxed_message(message, width=100):
    """
    Displays a centered boxed message.

    Parameters:
    message (str): The message to display.
    width (int): The width of the box (default is 100).

    Example Output:
    ==================================================
                        Message
    ==================================================
    """
    print("=" * width)
    print(f"{message.center(width)}")
    print("=" * width)


def print_boxed_options(
    options, additional_option="Other. Exit", ifadd=True, width=100
):
    """
    Prints a boxed menu of numbered options.

    Parameters:
    options (dict): A dictionary where keys are option numbers and values are the corresponding descriptions.
    additional_option (str): The description for the "Other" option (default is "Other. Exit").
    width (int): The width of the box (default is 100).

    Example Output:
    ==================================================
    | 1. Option 1                                    |
    | 2. Option 2                                    |
    | Other. Exit                                    |
    ==================================================
    """
    print("=" * width)
    for key, value in options.items():
        line = f"{key}. {value}"
        print(f"| {line.ljust(width - 4)}|")
    if ifadd:
        other_line = f"{additional_option}"
    print(f"| {other_line.ljust(width - 4)}|")
    print("=" * width)


def print_boxed_yes_no(prompt, option_1="Yes", option_2="No", ifadd=True, width=100):
    """
    Displays a boxed yes/no prompt for user interaction.

    Parameters:
    prompt (str): The prompt to display.
    option_1 (str): The label for the first option (default is "Yes").
    option_2 (str): The label for the second option (default is "No").
    width (int): The width of the box (default is 100).

    Example Output:
    ==================================================
                        Prompt
    ==================================================
    | 1. Yes                                         |
    | 2. No                                          |
    | Other. Exit                                    |
    ==================================================
    """
    print("=" * width)
    print(f"{prompt.center(width)}")
    print("=" * width)
    print(f"| 1. {option_1}".ljust(width - 1) + "|")
    print(f"| 2. {option_2}".ljust(width - 1) + "|")
    if ifadd:
        print(f"| Other. Exit".ljust(width - 1) + "|")
    print("=" * width)
