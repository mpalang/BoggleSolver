# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 13:56:01 2026

@author: morit
"""
import cv2
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

#auxiliary
def timer(function):
    """wrapper that times a function"""
    from time import perf_counter as pc
    from functools import wraps
    @wraps(function)
    def wrapped(*args,**kwargs):
        t0 = pc()
        result = function(*args,**kwargs)
        print(f'{function.__name__} took {(pc()-t0):.2f} s')
        return result
    return wrapped
        
    
def show(img, title=""):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()
    
def order_points(pts):
    pts = pts.reshape(4,2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)

def warp(img, pts):
    pts = order_points(pts)

    size = 400  # or 800
    dst = np.array([
        [0,0],
        [size-1,0],
        [size-1,size-1],
        [0,size-1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (size, size))

    return warped

def load_image(path):
    img = cv2.imread(path)
    return img

def crop_image(img):
    # This is the template size and shoud therefore not be changed!
    crop_w = 850
    crop_h = 850
    
    H, W = img.shape[:2]

    # Image center
    cx = W // 2
    cy = H // 2
    
    # Crop around center
    x1 = cx - crop_w // 2
    x2 = cx + crop_w // 2
    y1 = cy - crop_h // 2
    y2 = cy + crop_h // 2
    
    cropped = img[y1:y2, x1:x2]
    return cropped

def resize_image(img):
    # This is the template size and shoud therefore not be changed!
    target_w = 850
    target_h = 850
    
    resized = cv2.resize(
        img,
        (target_w, target_h),
        interpolation=cv2.INTER_AREA
    )
    return resized

def rotate_image(img):
    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return rotated

def make_bw(img:'BGR image',
            white_threshold:'HSV thresholds'=[(0, 0, 180),(180, 80, 255)],
            black_threshold:'HSV thresholds'=[(0, 0, 120),(179, 255, 255)],
            blur_kernel=7,blur_sigma=0.5):
    img = cv2.GaussianBlur(img, (blur_kernel,blur_kernel), blur_sigma)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # pixels that should become white
    white_mask = cv2.inRange(hsv,*white_threshold)
    
    # pixels that should become black
    black_mask = cv2.inRange(hsv,*black_threshold)
    
    # start with white image
    bw = 255*np.ones(img.shape[:2], dtype=np.uint8)
    # assign black pixels
    bw[black_mask > 0] = 0
    
    # # start with white image
    # bw = 0*np.ones(img.shape[:2], dtype=np.uint8)
    # # assign white pixels
    # bw[white_mask > 0] = 255
    kernel = 1*np.ones((3,3), np.uint8)
    bw = cv2.morphologyEx(   
                        bw,
                        cv2.MORPH_OPEN,
                        kernel,
                        iterations=1
                    )
                        
    return bw

def load_templates():
    """loads template pictures and converts to black and white for easier comparison"""
    base = Path(Path(__file__).parent,'Templates')
    print(f'Loading templates from {base}')
    paths = list(base.glob('*.jpg'))
    templates={}
    if len(paths)<1:
        raise ValueError('No templates found')
    for path in paths:
        templates[Path(path).stem] = make_bw(load_image(path))
    return templates

def resize_templates():  
    """This function is making sure all templates are the same size"""
    templates = load_templates()
    for letter,template in templates.items():
        cropped = crop_image(template)
            
        show(cropped,title=f'{letter}:{cropped.shape[:2]}')
        
        # Uncomment to replace existing images
        # cv2.imwrite(str(Path('Templates') / f"{letter}.jpg"), cropped)