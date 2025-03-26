import cv2
import numpy as np

def detect_black_pieces(img, threshold=50, min_area=250, max_area=2500):
    # convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # apply threshold
    img = cv2.inRange(img, 0, threshold)
    # apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel) # close
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) # open
    # find contours of detected pieces
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # filter contours based on area
    contours = [c for c in contours if max_area > cv2.contourArea(c) > min_area]
    return contours

def detect_white_pieces(img, lower=np.array[20, 60, 110], upper = np.array([40, 255, 255]), min_area=250, max_area=2500):
    # convert to HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # apply threshold
    img = cv2.inRange(img, lower, upper)
    # apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel) # close
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) # open
    # dilate
    kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(img, kernel, iterations=1) 
    # find contours of detected pieces
    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # filter contours based on area
    contours = [c for c in contours if max_area > cv2.contourArea(c) > min_area]
    return contours

def detect_pieces(img):
    contours = detect_black_pieces(img) + detect_white_pieces(img)
    return contours


# for the following two functions,
# some extra pixels are added to the bounding box
# to make sure the piece is not cropped

def draw_bounding_boxes(img, contours, extra_width=5, extra_height=10):
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        x -= extra_width
        y -= extra_height
        w += extra_width
        h += extra_height
        cv2.rectangle(img, (x, y), (x+w+5, y+h), (0, 255, 0), 2)
    return img

def crop_pieces(img, contours, extra_width=5, extra_height=10):
    pieces = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        x -= extra_width
        y -= extra_height
        w += extra_width
        h += extra_height
        pieces.append(img[y:y+h, x:x+w])
    return pieces







    




