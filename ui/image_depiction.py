# How to run: python -m ui.image_depiction
from utils import videotools as vt


def main():
    static_image_mode = True
    min_detection_confidence = 0.5

    vt.hand_depiction(static_image_mode, min_detection_confidence)


# Run the main() function
if __name__ == "__main__":
    main()
