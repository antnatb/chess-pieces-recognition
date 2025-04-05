import cv2
import numpy as np

# This file contains functions to detect chess pieces in an image
# The functions are based on color detection and contour detection
# The functions are not perfect and may not work in all cases

def detect_pieces(img):
    # resize 
    img = cv2.resize(img, (640, 480))
    # crop: extra pixels outside the chessboard are unnecessary
    img = img[10:-10, 5:-20]
    # detect black and white pieces
    # the contours of the detected pieces are returned
    black_contours = detect_black_pieces(img)
    white_contours = detect_white_pieces(img)
    # draw bounding boxes around detected pieces
    img = draw_black_bounding_boxes(img, black_contours)
    img = draw_white_bounding_boxes(img, white_contours)
    return img    


# The following function is used to detect black pieces
# The lower and upper values for the threshold were found by trial and error
# They are not perfect and may not work in all cases
def detect_black_pieces(img, threshold=55, min_area=150, max_area=2500):
    # convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # median blur
    img = cv2.medianBlur(img, 5)
    # apply threshold
    img = cv2.inRange(img, 0, threshold)
    # apply morphological operations to remove noise
    kernel = np.ones((7, 7), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel) # close
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) # open
    # find contours of detected pieces
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # filter contours based on area and height/width ratio
    contours = [c for c in contours
                 if max_area > cv2.contourArea(c) > min_area
                 and cv2.boundingRect(c)[2] <= cv2.boundingRect(c)[3] <= 3 * cv2.boundingRect(c)[2]]
    return contours


# The following function is used to detect white pieces
# The H and V channels are used 
# The lower and upper values for H and V were found by trial and error
# They are not perfect and may not work in all cases 
def detect_white_pieces(img, h_bounds = (23, 36),v_bounds = (64, 191), min_area=150, max_area=2500):
    # convert to HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # split HSV channels and take only H and V
    h, _, v = cv2.split(img)
    # apply threshold only on H and V channels
    h_mask = cv2.inRange(h, h_bounds[0],h_bounds[1])
    v_mask = cv2.inRange(v, v_bounds[0], v_bounds[1])
    # combine masks
    mask = cv2.bitwise_and(h_mask, v_mask)
    # apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # open
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # close
    # find contours of detected pieces
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # filter contours based on area
    contours = [c for c in contours 
                if max_area > cv2.contourArea(c) > min_area
                and cv2.boundingRect(c)[2] <= cv2.boundingRect(c)[3] <= 2.5 * cv2.boundingRect(c)[2]]
    return contours

# For the following functions,
# some extra pixels are added to the bounding box
# to make sure the piece is not cropped

def draw_black_bounding_boxes(img, black_contours, padding_width=0.25, padding_height=0.2):
    # draw bounding boxes around detected pieces
    for c in black_contours:
        x, y, w, h = cv2.boundingRect(c)
         # calculate extra width and height based on padding
        extra_width = int(padding_width * w)
        extra_height = int(padding_height * h)
        x -= extra_width
        y -= extra_height
        w += extra_width
        h += extra_height
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return img

def draw_white_bounding_boxes(img, white_contours, padding_width = 0.25, padding_height = 0.4):
    # draw bounding boxes around detected pieces
    for c in white_contours:
        x, y, w, h = cv2.boundingRect(c)
        # calculate extra width and height based on padding
        extra_width = int(padding_width * w)
        extra_height = int(padding_height * h)
        x -= extra_width
        y -= extra_height 
        w += 2*extra_width
        h += extra_height
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    return img

def crop_pieces(img, contours, extra_width=5, extra_height=10):
    pieces = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        
        x -= extra_width
        y -= extra_height
        w += extra_width
        h += extra_height
        pieces.append(img[y:y+h, x:x+w+5]) # add 5 pixels to the width
    return pieces




    




