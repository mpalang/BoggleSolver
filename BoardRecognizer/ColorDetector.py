# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 21:15:58 2026

@author: morit
"""

import cv2
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

base = Path(__file__).parent/'TrainingImages'
images = list(base.glob('*.jpg'))
path = str(images[10])

class ColorOnClick:
    def __init__(self,path):
        
        img = cv2.imread(path)
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        def mouse_callback(event, x, y, flags, param):
        
            if event == cv2.EVENT_LBUTTONDOWN:
        
                b, g, r = img[y, x]
                h, s, v = hsv[y, x]
        
                print(f"Pixel ({x}, {y})")
                print(f"RGB : ({r}, {g}, {b})")
                print(f"HSV : ({h}, {s}, {v})")
                print()
                
        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", mouse_callback)
        
        while True:
        
            cv2.imshow("Image", img)
        
            key = cv2.waitKey(1)
        
            if key == 27:      # ESC
                break
        
        cv2.destroyAllWindows()
        
# ColorOnClick(path)


(14, 145, 214)
(10, 255, 136)
(14, 144, 229)
(14, 251, 186)
def plot_color(hsv):
    
    # Create figure and axis
    # OpenCV HSV color (H:0-179, S:0-255, V:0-255)
    hsv = np.uint8([[hsv]])

    # Convert to BGR
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Convert BGR -> RGB for matplotlib
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    plt.imshow(rgb)
    plt.axis("off")
    
# plot_color((14, 251, 186))
    


