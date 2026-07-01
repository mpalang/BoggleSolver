import sys
import cv2
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

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
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    orange = cv2.inRange(hsv, (10,100,90), (40,255,255))

    # 3. clean noise
    mask = cv2.morphologyEx(orange, cv2.MORPH_OPEN, np.ones((7,7), np.uint8))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    
    # 4. largest contour inside
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours,key=cv2.contourArea)
    
    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(np.float32)
    def inset_rect(pts, scale=0.88):
        pts = order_points(pts)
        center = np.mean(pts, axis=0)
        return center + scale * (pts - center)
    box = inset_rect(box)
    warped = warp(img,box)
    return warped

def get_dice(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(gray, 30, 120)
    
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    
    h_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel)
    
    grid = cv2.add(h_lines, v_lines)
    grid = cv2.dilate(grid, np.ones((3,3), np.uint8), iterations=2)
    
    h_proj = np.sum(h_lines, axis=1)
    v_proj = np.sum(v_lines, axis=0)
    
    from scipy.signal import find_peaks    
    h_peaks, _ = find_peaks(h_proj, distance=len(h_proj)//6)
    v_peaks, _ = find_peaks(v_proj, distance=len(v_proj)//6)
    
    # h_peaks = np.linspace(h_peaks[0], h_peaks[-1], 5).astype(int)
    # v_peaks = np.linspace(v_peaks[0], v_peaks[-1], 5).astype(int)
    
    cells = []
    def crop_cell(cell, pad=0.12):
        h, w = cell.shape[:2]
        x1, x2 = int(w*pad), int(w*(1-pad))
        y1, y2 = int(h*pad), int(h*(1-pad))
        return cell[y1:y2, x1:x2]

    for i in range(4):
        for j in range(4):
            cell = gray[
                h_peaks[i]:h_peaks[i+1],
                v_peaks[j]:v_peaks[j+1]
            ]
            cell = crop_cell(cell,pad=0.12)
            cells.append(cell)
    
    # debug = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # for y in h_peaks:
    #     cv2.line(debug, (0, y), (debug.shape[1], y), (0, 0, 255), 2)
    
    # for x in v_peaks:
    #     cv2.line(debug, (x, 0), (x, debug.shape[0]), (0, 255, 0), 2)
        
    return cells
    
base = Path(__file__).parent/'TrainingImages'
images = list(base.glob('*.jpg'))[-5:-1]
for image in images:
    img = load_image(image)
    img = get_board(img)  
    show(img)
    # dice = get_dice(img)
    # for die in dice:
    #     show(die)
    
