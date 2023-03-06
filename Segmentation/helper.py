import colorsys
import numpy as np
import cv2


def keep_single_color(image, color):
    # Read image
    # Convert image to HSV format
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Extract pixels within specified color range
    lower_range = np.array([color[0] - 1, color[1] - 1, color[2] - 1])
    upper_range = np.array([color[0] + 1, color[1] + 1, color[2] + 1])
    mask = cv2.inRange(hsv_image, lower_range, upper_range)
    # Assign only pixels of specified color to new image
    new_image = np.zeros_like(image)
    new_image[mask>0] = image[mask>0]
    # Save new image
    return new_image,mask






def hex_to_hsv_cv(hex_color):
    # Convert hexadecimal color code to RGB value
    rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))

    # Convert RGB value to standard HSV value
    hsv_color = colorsys.rgb_to_hsv(rgb_color[0] / 255.0, rgb_color[1] / 255.0, rgb_color[2] / 255.0)

    # Map the HSV value to the range of 0 to 180 and 0 to 255
    h = int(hsv_color[0] * 180)
    s = int(hsv_color[1] * 255)
    v = int(hsv_color[2] * 255)

    # Convert the HSV value to the OpenCV format of color value with a maximum value of 255
    hsv_color_cv = np.array([h, s, v], dtype=np.uint8)

    return hsv_color_cv

def hsv_to_rgb_cv(hsv_color):
    # Convert HSV value to standard RGB value
    hsv_color = colorsys.hsv_to_rgb(hsv_color[0] / 180.0, hsv_color[1] / 255.0, hsv_color[2] / 255.0)

    # Map the HSV value to the range of 0 to 180 and 0 to 255
    r = int(hsv_color[0] * 255)
    g = int(hsv_color[1] * 255)
    b= int(hsv_color[2] * 255)

    # Convert the HSV value to the OpenCV format of color value with a maximum value of 255
    hsv_color_cv = np.array([b, g, r], dtype=np.uint8)

    return hsv_color_cv