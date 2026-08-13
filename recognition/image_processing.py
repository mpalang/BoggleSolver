"""general functions for image processing and opencv helpers."""
from pathlib import Path
import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils.paths import TRAININGSIMAGES



def show(img, title=""):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()


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


def order_points(pts):
    pts = pts.reshape(4,2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def K(n: int, shape = cv2.MORPH_ELLIPSE):
    """Creates Kernel of size n. Keep it uneven no shifting happens."""
    if n%2==0:
        raise ValueError('kernel size needs to be uneven')
    return cv2.getStructuringElement(shape, (n, n))


def load_image(path):
    if not Path(path).exists():
        raise ValueError(f"Path {path} doesn't exist")
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f'Reading image from {path} failed.')
    return img


def normalize_img(img, long_edge=1200):
    s = long_edge / max(img.shape[:2])
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    norm_img = cv2.resize(img, None, fx=s, fy=s, interpolation=interp)
    return norm_img


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


def rotate_image(img):
    rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return rotated


def resize_image(img,size=None):
    # This is the template size and shoud therefore not be changed!
    if size:
        target_w = size
        target_h = size
    else:
        target_w = 128
        target_h = 128
    resized = cv2.resize(img,
                        (target_w, target_h),
                        interpolation=cv2.INTER_AREA
                        )
    return resized


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


def approx_quad(cnt):
    peri = cv2.arcLength(cnt, True)
    for eps in np.arange(0.01, 0.06, 0.005):
        a = cv2.approxPolyDP(cnt, eps * peri, True)
        if len(a) == 4 and cv2.isContourConvex(a):
            return a.reshape(4, 2).astype(np.float32)
    return None



def _clear_border(bw):
    """Clear potentially white borders"""
    n, lab, st, _ = cv2.connectedComponentsWithStats(bw)
    H, W = bw.shape
    keep = [i for i in range(1, n)
            if st[i,0] > 0 and st[i,1] > 0
            and st[i,0]+st[i,2] < W and st[i,1]+st[i,3] < H]
    return np.isin(lab, keep).astype(np.uint8) * 255


def _threshold(g, core=0.6):
    H, W = g.shape
    ky, kx = int(H*(1-core)/2), int(W*(1-core)/2)
    thr, _ = cv2.threshold(g[ky:H-ky, kx:W-kx], 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.threshold(g, thr, 255, cv2.THRESH_BINARY)[1]


def _keep_glyph(m, expand=0.30, min_px=8):
    n, lab, st, cent = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    X, Y, W, H = st[big, :4]
    keep = [i for i in range(1, n)
            if st[i, cv2.CC_STAT_AREA] >= min_px
            and X - W*expand <= cent[i][0] <= X + W*(1+expand)
            and Y - H*expand <= cent[i][1] <= Y + H*(1+expand)]
    return np.isin(lab, keep).astype(np.uint8) * 255


def _try_polarity(bw, invert):
    if invert:
        bw = cv2.bitwise_not(bw)
    m = _keep_glyph(_clear_border(bw))                                       
    if m is None:
        return None, 0
    frac = (m > 0).mean()
    if not (0.01 < frac < 0.55):                                            
        return None, 0
    return m, frac


def canon(img, size=64, margin=0, core=0.6):
    """BGR or gray -> binarized, tight-cropped, centred, size x size."""
    bw = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # bw = _threshold(bw)
    
    _, bw = cv2.threshold(bw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    m, score = max((_try_polarity(bw, inv) for inv in (False, True)),
                   key=lambda c: c[1])
    if m is None or score == 0:
        return np.zeros((size, size), np.uint8)

    ys, xs = np.nonzero(m)
    g = m[ys.min():ys.max()+1, xs.min():xs.max()+1]
    h, w = g.shape
    s = (size - 2*margin) / max(w, h)
    g = cv2.resize(g, (max(1, int(w*s)), max(1, int(h*s))), interpolation=cv2.INTER_AREA)

    out = np.zeros((size, size), np.uint8)
    oy, ox = (size - g.shape[0])//2, (size - g.shape[1])//2
    out[oy:oy+g.shape[0], ox:ox+g.shape[1]] = g
    
    # if (out > 127).mean() < 0.10:
    #     warnings.warn("canon output nearly empty — bad crop?")
        
    return out


def load_trainingsimages():
    paths = list(TRAININGSIMAGES.glob('*.jpg'))
    return [load_image(path) for path in paths]  
