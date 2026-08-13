import cv2
import numpy as np
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


def find_dice(board, use_lines = False):
    """Find dice in board."""
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

    #Improve die separation with watershedding:
    d = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    
    #sweep sensitivity to find exacly 16 die.
    best = None
    for f in np.arange(0.30, 0.85, 0.025): 
        peaks = (d > f * d.max()).astype(np.uint8)
        n = cv2.connectedComponents(peaks)[0] - 1
        if best is None or abs(n - 16) < abs(best[1] - 16):
            best = (peaks, n)
        if n == 16:
            break
    peaks = best[0] * 255                     

    #Now define markers for dice
    nlab, markers = cv2.connectedComponents(peaks)
    markers += 1
    markers[cv2.subtract(mask, peaks) == 255] = 0
    markers = cv2.watershed(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), markers)
    
    #Get Squares of dice:
    squares = []
    for i in range(2, nlab + 1):
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

    #Order dice:
    row_tol = 80
    def sort_key(rect):
        (x,y),_,_ = rect
        row = round(y / row_tol)
        return (row, x)
    squares = sorted(squares, key=sort_key) # sort from top left to bottom right

    #Make sure there's actually 16 dice.
    #TODO: maybe add a second check where the centers have to be approx. on a 4x4 grid?
    if len(squares)!=16:
        raise RuntimeError(f'Number of found dice is not 16: {len(squares)}')

    return squares


def draw_squares(board, squares, thickness=2, color=(0, 255, 0), numbers = True):
    """draw dice on image"""
    vis = board.copy()
    rects = [cv2.boxPoints(square) for square in squares]
    cnts = [np.int32(c) for c in rects]
    cv2.drawContours(vis, cnts, -1, color, thickness)
    if numbers:
        for i, b in enumerate(rects, start=1):
            tr = b[np.argmax(b[:, 0] - b[:, 1])]        # corner with max x-y
            org = tuple(np.int32(tr) + np.array([-16, 16]))
            cv2.putText(vis, str(i), org, cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)
    return vis


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


def draw_dice(board: np.ndarray, 
              dice_squares: list[tuple], 
              letters: list[str],
              confidences: list[float] | None = None,
              rotations: list[int] | None = None,
              templates: dict[str,np.ndarray] | None = None,
              scale=0.6, #scales the templates
              good = 0.1 #decides which confidence value qualifies as good.
              ):
    """Overlay each die's square with recognized letter on the board."""

    n = len(dice_squares)
    confidences = confidences if confidences is not None else [None] * n
    rotations   = rotations   if rotations is not None else [0] * n

    vis = board.copy()
    for s, l, c, r in zip(dice_squares, letters, confidences, rotations, strict=True):
        color = (0, 255, 0) if c >= good else (0, 0, 255)   # BGR

        t = templates[l] #TODO: add option to write machine letters if no templates is passed. For now it will raise an error.
        for n in range((4-r)%4):
            t = rotate_image(t)
        cnts, _ = cv2.findContours(t, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        size = t.shape[0]
        src = np.array([[0,0],[size-1,0],[size-1,size-1],[0,size-1]], np.float32)

        dst = cv2.boxPoints(s).astype(np.float32)  
        dst = order_points(dst)
        c = dst.mean(axis=0) 
        dst = (c + scale * (dst - c)).astype(np.float32)

        M = cv2.getPerspectiveTransform(src, dst)
        mapped = [np.int32(cv2.perspectiveTransform(c.astype(np.float32), M))
                for c in cnts]
        
        cv2.drawContours(vis, mapped, -1, color, 2)
    return vis



