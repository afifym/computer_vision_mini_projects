import cv2
import numpy as np


BLUE, GREEN, RED = (255, 0, 0), (0, 255, 0), (0, 0, 255)

WHITE_LOWER, WHITE_UPPER = (200, 200, 200), (255, 255, 255)
BLACK_LOWER, BLACK_UPPER = (0, 0, 0), (10, 10, 10)


def mask_from_color(image, lower, upper, erode=0, dilate=0):
    mask = cv2.inRange(image, lower, upper)
    
    if erode: mask = cv2.erode(mask, None, iterations=erode)
    if dilate: mask = cv2.dilate(mask, None, iterations=dilate)
    
    return mask

def resize_same(image, new_height=False, new_width=False):
    ar = image.shape[1]/image.shape[0]
    
    if new_height: 
        new_width = int(ar * new_height)
    elif new_width: 
        new_height = int(new_width/ar)
    
    return cv2.resize(image, dsize=(new_width, new_height))