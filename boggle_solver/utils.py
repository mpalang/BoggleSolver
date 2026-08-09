# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 13:56:01 2026

@author: morit
"""
import cv2
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

def load_trainingsimages():
    paths = list(Path('board_recognizing/TrainingImages').glob('*.jpg'))
    return [load_image(path) for path in paths]


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

def warp(img, pts, out):
    pts = order_points(pts)

    size = out  # or 800
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

def crop_image(img,size=None):
    # This is the template size and shoud therefore not be changed!
    if size:
        crop_w = size
        crop_h = size
    else:
        crop_w = 128
        crop_h = 128
    
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

def resize_image(img,size=None):
    # This is the template size and shoud therefore not be changed!
    if size:
        target_w = size
        target_h = size
    else:
        target_w = 128
        target_h = 128
    
    resized = cv2.resize(
        img,
        (target_w, target_h),
        interpolation=cv2.INTER_AREA
    )
    return resized

def rotate_image(img):
    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return rotated

def make_bw(img, white_threshold=[(0, 0, 180),(180, 80, 255)],
            black_threshold=[(0, 0, 120),(179, 255, 255)],
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

def load_templates(raw=False,cropped=False, use_bw=False, use_canon=True):
    """loads template pictures and converts to black and white for easier comparison"""
    if raw:
        base = Path(Path(__file__).parent,'Templates_Original')
    elif cropped:
        base = Path(Path(__file__).parent,'Templates_Cropped')
    else:
        base = Path(Path(__file__).parent,'Templates')
    out_folder = Path(Path(__file__).parent,'Templates')
    # print(f'Loading templates from {base}')
    paths = list(base.glob('*.jpg'))
    templates={}
    if len(paths)<1:
        raise ValueError('No templates found')
    for path in paths:
        img = load_image(path)
        if use_canon:
            img = canon(img)      
        elif use_bw:
            img = make_bw(img)
        templates[Path(path).stem] = img
    return templates


def resize_templates(size=None):  
    """This function is making sure all templates are the same size"""
    templates = load_templates(cropped=True,bw=False)
    for letter,template in templates.items():
        cropped = resize_image(template,size=None)
            
        # show(cropped,title=f'{letter}:{cropped.shape[:2]}')
        
        # Uncomment to replace existing images
        path = str(Path('board_recognizing/Templates') / f"{letter}.jpg")
        check = cv2.imwrite(path, cropped)
        if not check:
            print(path)


def approx_quad(cnt):
    peri = cv2.arcLength(cnt, True)
    for eps in np.arange(0.01, 0.06, 0.005):
        a = cv2.approxPolyDP(cnt, eps * peri, True)
        if len(a) == 4 and cv2.isContourConvex(a):
            return a.reshape(4, 2).astype(np.float32)
    return None


def canon(img, size=64, margin=0, min_area=0.02):
    """This function collapses letters to inkbox. BGR or gray -> binarized, tight-cropped, centred, size x size."""

    def clear_border(bw):
        """Clear potentially white borders"""
        n, lab, st, _ = cv2.connectedComponentsWithStats(bw)
        H, W = bw.shape
        keep = [i for i in range(1, n)
                if st[i,0] > 0 and st[i,1] > 0
                and st[i,0]+st[i,2] < W and st[i,1]+st[i,3] < H]
        return np.isin(lab, keep).astype(np.uint8) * 255

    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (bw > 127).mean() > 0.5:          # ink should be the minority
        bw = cv2.bitwise_not(bw)
    bw = clear_border(bw)

    n, lab, st, _ = cv2.connectedComponentsWithStats(bw)
    if n < 2:
        return np.zeros((size, size), np.uint8)
    big = st[1:, cv2.CC_STAT_AREA].max()
    keep = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > min_area * big]
    m = np.isin(lab, keep).astype(np.uint8) * 255

    ys, xs = np.nonzero(m)
    g = m[ys.min():ys.max()+1, xs.min():xs.max()+1]
    h, w = g.shape
    s = (size - 2*margin) / max(w, h)
    g = cv2.resize(g, (max(1, int(w*s)), max(1, int(h*s))), interpolation=cv2.INTER_AREA)

    out = np.zeros((size, size), np.uint8)
    oy, ox = (size - g.shape[0])//2, (size - g.shape[1])//2
    out[oy:oy+g.shape[0], ox:ox+g.shape[1]] = g
    return out


class WordFinder:
    def __init__(self, board:np.ndarray[str], trie):
        self.board = board
        self.trie = trie
        self.solve()
        
    def solve(self):
        results = set()

        ROWS,COLS = self.board.shape
        
        for r in range(ROWS):
            for c in range(COLS):
                self.dfs(r, c, self.trie.root, "", set(), results)
        
        results = sorted(results, key=lambda x: (x,len(x)))
        self.valid_words = results
        return results
        
    def dfs(self,r, c, node, word, visited, results):
        DIRS = [(-1,-1),(-1,0),(0,-1),(0,1),(1,0),(1,1),(-1,1),(1,-1)]
        ROWS,COLS = self.board.shape
        
        letter = self.board[r,c].upper()
        if letter == 'QU':
            if 'Q' not in node.children:
                return
            word += 'Q'
            node = node.children['Q']
            if 'U' not in node.children:
                return
            word += 'U'
            node = node.children['U']
        
        else:
            if letter not in node.children:
                return
        
            node = node.children[letter]
            word += letter
        
            if node.is_word:
                results.add(word)
    
        visited.add((r, c))
    
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
    
            if (0 <= nr < ROWS and
                0 <= nc < COLS and
                (nr, nc) not in visited):
    
                self.dfs(nr, nc, node, word, visited, results)
    
        visited.remove((r, c))