import cv2
import numpy as np
from recognition.image_processing import order_points, warp, K


def get_board(full_img, out=512, inset=0.94, blur = False):
    #Blur to get rid of noise
    if blur:
        img = cv2.GaussianBlur(full_img, K(5), 0)
    else:
        img = full_img.copy()
        
    #Get orange mask for board recognition:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    orange = cv2.inRange(hsv, (10, 100, 90), (40, 255, 255))
    mask = cv2.morphologyEx(orange, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    #Find outer contours of blobs
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        print('None found')
        return None
    #Pick blob with largest area (presumably the boggle board)
    cnt = max(cnts, key=cv2.contourArea)
    box = cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.float32)
    box = order_points(box)
    c = box.mean(axis=0)
    box = (c + inset * (box - c)).astype(np.float32)
    #Cut and warp to board
    board = warp(img, box, out)
    return board