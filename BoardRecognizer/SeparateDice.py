import sys
import cv2
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from statistic_tools import kde

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

def get_board(img):
  try:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    orange = cv2.inRange(hsv, (10,100,90), (40,255,255))
    orange = cv2.GaussianBlur(orange, (3,3), 0)

    # 3. clean noise
    mask = cv2.morphologyEx(orange, cv2.MORPH_OPEN, np.ones((7,7), np.uint8))
    # kernel = np.ones((5, 5), np.uint8)
    # mask = cv2.erode(mask, kernel, iterations=2)
    
    # 4. largest contour inside
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours,key=cv2.contourArea)
    # loop over the contours
    screenCnt = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        
        screenCnt.append(approx)
            
    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(np.float32)
    def inset_rect(pts, scale=1):
        pts = order_points(pts)
        center = np.mean(pts, axis=0)
        return center + scale * (pts - center)
    box = inset_rect(box)
    
    # cv2.drawContours(img, [cnt], -1, (170, 155, 0), 50)
    # cv2.drawContours(img, [np.int32(box)], -1, (170, 155, 0), 50)
    # cv2.circle(img, tuple(np.int32(rect[0])), 50, (170, 155, 0), -1)
    warped = warp(img,box)
    return warped
  except Exception as e:
    print(f'Image failed: {e}')
    return orange

def get_dice(img,tolerance=0.5):
    h,w = img.shape[:2]
    A = h*w
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (5,5), 0)
    orange = cv2.inRange(hsv, (2,40,30), (30,255,255))

    inv = cv2.bitwise_not(orange)
    edges = cv2.Canny(inv, 30, 120)
    
    # rho=1, theta=pi/180, threshold=50, minLineLength=50, maxLineGap=10
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=20, maxLineGap=60)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 120, 255), 4)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    orange = cv2.inRange(hsv, (10,100,90), (40,255,255))
    inv = cv2.bitwise_not(orange)
    mask = cv2.morphologyEx(inv, cv2.MORPH_OPEN, np.ones((7,7), np.uint8))
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours,key=cv2.contourArea)
        
    cv2.drawContours(img, contours, -1, (30, 255, 100), 2)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (5,5), 0)
    orange = cv2.inRange(hsv, (1,40,30), (30,255,255))
    
    inv = cv2.bitwise_not(orange)
    
    contours, _ = cv2.findContours(inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #RETR_LIST
    contours = sorted(contours,key=cv2.contourArea)
    
    a = [cv2.contourArea(c) for c in contours]
    a = np.array([c for c in a if c>A/400])
    
    x,y = kde(a, bandwidth=100)  
    ipeak = np.argmax(y)
    
    A_estimate = A/6/6 # assume dice takes up 1/6 of the image length
    mask_bgr = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    screenCtr = [c for c in contours if A_estimate*(1-tolerance)<cv2.contourArea(c)<A_estimate*(1+tolerance)]
    
    squares = []
    for c in screenCtr:
        area = cv2.contourArea(c)
    
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        square_tolerance = 0.3
        if not (1-square_tolerance <= aspect_ratio <= 1+square_tolerance):
            continue
        if w*h*(1-square_tolerance) < area < w*h*(1+square_tolerance):
            squares.append(c)
    
    squares = sorted(squares, key=lambda c: cv2.boundingRect(c)[1])  # sort by y first
    
    print(len(squares))
    cv2.drawContours(mask_bgr,squares,-1,(30,255,100),3)
    
    cell_images = []
    for i, c in enumerate(squares):
        x, y, w, h = cv2.boundingRect(c)
    
        # optional padding (helps avoid cutting borders)
        pad = 2
        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + w + pad, img.shape[1])
        y2 = min(y + h + pad, img.shape[0])
    
        cell = img[y1:y2, x1:x2].copy()
    
        cell_images.append(cell)
    
        # optional save
        # cv2.imwrite(f"cell_{i}.png", cell)
    
    
    
    return cell_images
    
def read_letter(img):
    from openai import OpenAI

    client = OpenAI()

    result = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Extract all text from this image. Return only the text."},
                    {"type": "input_image", "image_data": img}
                    ]
                }
            ]
        )
    
    return result.output_text