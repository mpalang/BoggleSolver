import cv2
import numpy as np
from skimage.feature import peak_local_max
from recognition.image_processing import K, order_points, rotate_image


def get_lines(mask,Kernel = 3, rho=5,theta=np.pi/360,threshold=500,):
    """ draw straight lines between dice. Takes bw mask."""
    edges = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, K(Kernel))   # 1-px outlines
    lines = cv2.HoughLines(edges,     
                        rho=rho,
                        theta=theta,
                        threshold=threshold)
    return lines


def draw_lines(input_img, lines, color=255):
    """draw lines on an image"""
    img = input_img.copy()
    h, w = img.shape[:2]
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # two points far along the line
        scale = max(w, h)
        x1 = int(x0 + scale * (-b))
        y1 = int(y0 + scale * (a))
        x2 = int(x0 - scale * (-b))
        y2 = int(y0 - scale * (a))
        cv2.line(img, (x1, y1), (x2, y2), color, 3)
    return img


def pretreat_board(board, use_lines = False):
    #Get orange mask:
    hsv = cv2.cvtColor(board, cv2.COLOR_BGR2HSV)
    orange = cv2.inRange(hsv, (5, 100, 90), (70, 255, 255))
    #Get rid of small holes:
    orange_morph = cv2.morphologyEx(orange, cv2.MORPH_OPEN, K(5))
    orange_morph = cv2.morphologyEx(orange_morph, cv2.MORPH_CLOSE, K(11))
    #Draw extra lines that strengthen the grid and improve dice separation:
    if use_lines:
        lines = get_lines(orange_morph)
        orange_morph = draw_lines(orange_morph,lines)
    #Bring dice into foreground:
    mask = cv2.bitwise_not(orange_morph)
    return mask


def distance_transform(mask):
    d = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    return d

def visualize_distance_transform(d):
    dv = cv2.applyColorMap(
        cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLORMAP_JET)
    return dv


def find_peaks(d, min_distance = 30, threshold = 0.5, method = 'skimage'):

    if method == 'skimage':
        peaks = peak_local_max(d, min_distance=min_distance, threshold_abs=threshold)
        return peaks

    elif method == 'open_cv':
        k = np.ones((2*min_distance+1, 2*min_distance+1), np.uint8)
        dil = cv2.dilate(d, k)
        m = ((d >= dil) & (d > threshold*d.max())).astype(np.uint8)

        n, lab, st, cent = cv2.connectedComponentsWithStats(m)      # collapse plateaus
        cand = [(float(d[int(cent[i][1]), int(cent[i][0])]), float(cent[i][0]), float(cent[i][1]))
                for i in range(1, n)]
        cand.sort(reverse=True)                                     # strongest first

        out = []
        for v, x, y in cand:                                        # greedy suppression
            if all((x-px)**2 + (y-py)**2 >= min_distance**2 for px, py in out):
                out.append((x, y))
        return np.array(out)


def visualize_peaks(mask, peaks, size=21, thickness=3, color=(0, 255, 255)):
    pv = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if mask.ndim == 2 else mask.copy()
    for y, x in peaks:
        cv2.drawMarker(pv, (int(x), int(y)), color,
                       markerType=cv2.MARKER_CROSS,
                       markerSize=size, thickness=thickness,
                       line_type=cv2.LINE_AA)
    return pv


def find_markers(mask, peaks, guide=None, seed_r=5):
    """mask: binary dice mask (gray or BGR). guide: image watershed follows; defaults to mask."""
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    if guide is None:
        guide = mask
    if guide.ndim == 2:
        guide = cv2.cvtColor(guide, cv2.COLOR_GRAY2BGR)
    if guide.shape[:2] != mask.shape:
        raise ValueError(f"guide {guide.shape[:2]} does not match mask {mask.shape}")

    markers = np.zeros(mask.shape, np.int32)
    markers[mask == 0] = 1                              # background
    for i, (y, x) in enumerate(peaks, start=2):         # seeds
        cv2.circle(markers, (int(x), int(y)), seed_r, i, -1)

    markers = cv2.watershed(guide, markers)

    return markers


def visualize_markers(base, markers, alpha=0.45, seed=0):
    """Colour each watershed label over a gray or BGR base. Dams red, background dim."""
    bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR) if base.ndim == 2 else base.copy()

    rng = np.random.default_rng(seed)
    hi = int(markers.max())
    lut = rng.integers(60, 255, (hi + 2, 3), dtype=np.uint8)
    lut[0] = (0, 0, 0)          # unknown
    lut[1] = (40, 40, 40)       # background

    overlay = lut[np.clip(markers, 0, None)]
    out = cv2.addWeighted(bgr, 1 - alpha, overlay, alpha, 0)
    out[markers == -1] = (0, 0, 255)                    # dams at full strength

    for i in range(2, hi + 1):
        ys, xs = np.nonzero(markers == i)
        if len(xs):
            cv2.putText(out, str(i), (int(xs.mean()) - 8, int(ys.mean()) + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def find_squares(board, markers, peaks):
  #Get Squares of dice:
    squares = []
    for i in range(2, len(peaks) + 2):
        blob = (markers == i).astype(np.uint8) 
        c = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if not c:
            continue    
        rect = cv2.minAreaRect(max(c, key=cv2.contourArea))

        (cx, cy), (w,h), angle = rect
        #Size filter:
        H, W = board.shape[:2]
        size = w*h
        if size<(H*W)/7**2:
            continue
        if size>(H*W)/3.8**2:
            continue
        
        #Square filter:
        tolerance = 0.5
        if not (1-tolerance)*w < h < (1+tolerance)*w:
            continue

        #Squareify
        a = (w+h)/2
        square = ((cx,cy), (a,a), angle)
        squares.append(square)
    
    #Make sure there's actually 16 dice.
    #TODO: maybe add a second check where the centers have to be approx. on a 4x4 grid?
    if len(squares)!=16:
        raise RuntimeError(f'Number of found dice is not 16 but {len(squares)}')

    return squares


def visualize_squares(mask, squares):
    sv = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if mask.ndim == 2 else mask.copy()
    for square in squares:
        box = cv2.boxPoints(square).astype(np.int32)
        cv2.drawContours(sv, [box], 0, (0, 255, 0), 2)
    return sv


def order_squares(squares):
    n = 4
    squares = sorted(squares, key=lambda r: r[0][1])           # by y
    squares = [s for k in range(n) for s in sorted(squares[k*n:(k+1)*n], key=lambda r: r[0][0])]
    return squares


def find_dice(board):
    """Find dice positions on board."""
    mask = pretreat_board(board)
    d = distance_transform(mask)
    p = find_peaks(d)
    m = find_markers(mask, p)
    squares = find_squares(board, m, p)
    squares = order_squares(squares)
    return squares


def cut_dice(board, dice_squares, size = 128, pad=5):
    """cut dice from board for letter recognition. Size should be bigger than template size (probably 64). Pad in %"""
    cells = []
    pad = 1+pad/100
    dst = np.array([[0,0], [size-1,0], [size-1,size-1], [0,size-1]], np.float32)
    for (cx, cy), (w, h), ang in dice_squares:
        s = (w + h) / 2 * pad
        src = cv2.boxPoints(((cx, cy), (s, s), ang)).astype(np.float32)
        src = order_points(src)
        M = cv2.getPerspectiveTransform(src, dst)
        cells.append(cv2.warpPerspective(board, M, (size, size)))
    return cells


# def draw_dice(board: np.ndarray, 
#               dice_squares: list[tuple], 
#               letters: list[str],
#               confidences: list[float] | None = None,
#               rotations: list[int] | None = None,
#               templates: dict[str,np.ndarray] | None = None,
#               scale=0.6, #scales the templates
#               good = 0.1 #decides which confidence value qualifies as good.
#               ):
#     """Overlay each die's square with recognized letter on the board."""

#     n = len(dice_squares)
#     confidences = confidences if confidences is not None else [None] * n
#     rotations   = rotations   if rotations is not None else [0] * n

#     vis = board.copy()
#     for s, l, c, r in zip(dice_squares, letters, confidences, rotations, strict=True):
#         color = (0, 255, 0) if c >= good else (0, 0, 255)   # BGR

#         t = templates[l] #TODO: add option to write machine letters if no templates is passed. For now it will raise an error.
#         for n in range((4-r)%4):
#             t = rotate_image(t)
#         cnts, _ = cv2.findContours(t, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

#         size = t.shape[0]
#         src = np.array([[0,0],[size-1,0],[size-1,size-1],[0,size-1]], np.float32)

#         dst = cv2.boxPoints(s).astype(np.float32)  
#         dst = order_points(dst)
#         c = dst.mean(axis=0) 
#         dst = (c + scale * (dst - c)).astype(np.float32)

#         M = cv2.getPerspectiveTransform(src, dst)
#         mapped = [np.int32(cv2.perspectiveTransform(c.astype(np.float32), M))
#                 for c in cnts]
        
#         cv2.drawContours(vis, mapped, -1, color, 2)
#     return vis


def draw_dice(board: np.ndarray,
              dice_squares: list[tuple],
              letters: list[str],
              confidences: list[float] | None = None,
              rotations: list[int] | None = None,
              templates: dict[str, np.ndarray] | None = None,
              scale: float = 0.6,
              good: float = 0.1):
    """Overlay each die's square with the recognized letter on the board."""
    n = len(dice_squares)
    confidences = confidences if confidences is not None else [None] * n
    rotations   = rotations   if rotations   is not None else [0] * n

    vis = board.copy() if board.ndim == 3 else cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)

    for square, letter, conf, rot in zip(dice_squares, letters, confidences,
                                         rotations, strict=True):
        color = (0, 255, 0) if conf is not None and conf >= good else (0, 0, 255)

        dst = order_points(cv2.boxPoints(square).astype(np.float32))
        ctr = dst.mean(axis=0)
        dst = (ctr + scale * (dst - ctr)).astype(np.float32)

        tmpl = templates.get(letter) if templates else None
        if tmpl is None:                                  # fall back to typed text
            org = tuple(np.int32(ctr) - np.int32([14 * len(letter), -14]))
            cv2.putText(vis, letter, org, cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        color, 3, cv2.LINE_AA)
            continue

        for _ in range((4 - rot) % 4):
            tmpl = rotate_image(tmpl)

        cnts, _ = cv2.findContours(tmpl, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        size = tmpl.shape[0]
        src = np.array([[0, 0], [size-1, 0], [size-1, size-1], [0, size-1]], np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        mapped = [np.int32(cv2.perspectiveTransform(cnt.astype(np.float32), M))
                  for cnt in cnts]
        cv2.drawContours(vis, mapped, -1, color, 2)

    return vis

