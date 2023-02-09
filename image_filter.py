from PIL import Image
import numpy as np
import cv2


def filter_image(file_path):
    img = Image.open(file_path).convert("RGBA")
    img = np.array(img)

    # Define the desired color range
    lower_blue = np.array([0, 0, 252, 255], dtype=np.uint8)
    upper_blue = np.array([0, 252, 252, 255], dtype=np.uint8)
    lower_green = np.array([0, 131, 0, 255], dtype=np.uint8)
    upper_green = np.array([67, 223, 35, 255], dtype=np.uint8)
    lower_yellow = np.array([255, 187, 0, 255], dtype=np.uint8)
    upper_yellow = np.array([255, 255, 5, 255], dtype=np.uint8)

    # Create a mask of the desired color range
    mask = cv2.inRange(img, lower_blue, upper_blue)
    mask2 = cv2.inRange(img, lower_green, upper_green)
    mask3 = cv2.inRange(img, lower_yellow, upper_yellow)
    mask_combined = cv2.bitwise_or(mask, mask2)
    mask_combined = cv2.bitwise_or(mask_combined, mask3)

    # Get the original pixels
    result = cv2.bitwise_and(img, img, mask=mask_combined)

    return result
