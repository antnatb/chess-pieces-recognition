import cv2
import numpy as np

def resize(img, width=512, height=512):
    return cv2.resize(img, (width, height))

def read_image(path):
    return cv2.imread(path)

def detect_black_pieces(img):
    # convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # define and apply threshold
    threshold = 50
    img = cv2.inRange(img, 0, threshold)
    # apply morphological operationsto remove noise
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel) # close
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) # open
    # find contours of detected pieces
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # filter contours based on area
    min_area = 300
    max_area = 2300
    contours = [c for c in contours if max_area > cv2.contourArea(c) > min_area]
    return contours

def detect_white_pieces(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # define and apply threshold
    lower = np.array([20, 60, 110])
    upper = np.array([40, 255, 255])
    img = cv2.inRange(img, lower, upper)
    # apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel) # close
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) # open
    kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(img, kernel, iterations=1) # dilate
    # find contours of detected pieces
    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # filter contours based on area
    min_area = 250
    max_area = 2500
    contours = [c for c in contours if max_area > cv2.contourArea(c) > min_area]
    return contours

def detect_pieces(img):
    countours = detect_black_pieces(img) + detect_white_pieces(img)
    return countours

def draw_bounding_boxes(img, contours, extra_width=5, extra_height=10):
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        x -= extra_width
        y -= extra_height
        w += extra_width
        h += extra_height
        cv2.rectangle(img, (x, y), (x+w+5, y+h), (0, 255, 0), 2)
    return img




    




