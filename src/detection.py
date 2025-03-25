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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # define and apply threshold
    threshold = 40
    mask = cv2.inRange(gray, 0, threshold, cv2.THRESH_BINARY)
    # apply morphological operationsto remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # close
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # open
    # find contours of detected pieces
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # filter contours based on area
    min_area = 300
    contours = [c for c in contours if cv2.contourArea(c) > min_area]
    # draw bounding boxes
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img, (x-5, y-5), (x+w+5, y+h+5), (0, 255, 0), 2)
    return img
    



