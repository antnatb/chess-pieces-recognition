import cv2
import numpy as np

def resize(img, width=512, height=512):
    return cv2.resize(img, (width, height))

def read_image(path):
    return cv2.imread(path)

def detect_black_pieces(img):
    # resize
    img = resize(img)
    # convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # define and apply threshold
    threshold = 40
    img = cv2.inRange(img, 0, threshold, cv2.THRESH_BINARY)
    # apply morphological operationsto remove noise
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel) # close
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) # open
    # find contours of detected pieces
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # filter contours based on area
    min_area = 300
    contours = [c for c in contours if cv2.contourArea(c) > min_area]
    return contours
    



