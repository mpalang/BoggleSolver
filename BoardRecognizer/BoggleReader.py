import cv2
import numpy as np
from BoardRecognizer.auxiliary import (show,order_points,warp,resize_image,
                       make_bw,rotate_image,load_templates,
                       timer)

#% custom functions:
def get_board(img):
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

def draw_lines(img,blur_kernel=5,blur_sigma=0,
               canny_thresholds=(80,200),
               color_thresholds=[(2,40,30), (30,255,255)],
               rho=0.5,theta=np.pi/360,threshold=100,
               white_thresholds:'HSV thresholds'=[(0, 0, 180),(179, 80, 235)],):

    """ draw straight lines between dice"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (blur_kernel,blur_kernel), blur_sigma)
    # pixels that should become black
    black_mask = cv2.inRange(hsv,*white_thresholds)
    # orange = cv2.inRange(hsv, *color_thresholds)
    img[black_mask>0] = 255
    # show(img)
    edges = cv2.Canny(img, *canny_thresholds)

    lines = cv2.HoughLines(edges,     
                           rho=rho,
                           theta=theta,
                           threshold=threshold)

    if lines is not None:
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
    
            cv2.line(img, (x1, y1), (x2, y2), (0,120,255), 3)
            
    
                
    return img

def detect_cells(img,size_tolerance=0.7,square_tolerance=0.5,
                 erode_kernel=5,erode_iterations=1,
                 blur_kernel=7,blur_sigma=0.9):
    """detect square dice cells"""
    h,w = img.shape[:2]
    A = h*w
    
    # get not orange mask
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (blur_kernel,blur_kernel), blur_sigma)
    orange = cv2.inRange(hsv, (10,100,90), (40,255,255))
    inv = cv2.bitwise_not(orange)
    mask = cv2.morphologyEx(inv, cv2.MORPH_OPEN, np.ones((7,7), np.uint8))
    # show(mask)
    
    #Amplify dice separation:
    kernel = np.ones((erode_kernel,erode_kernel), np.uint8)
    separated = cv2.erode(mask, kernel, iterations=erode_iterations)
    # show(separated)
    
    # get contours (keep general. Dice are sorted out later)
    contours, _ = cv2.findContours(separated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours,key=cv2.contourArea)
    
    # # # debug contours:
    # img_with_ctrs = img.copy()
    # cv2.drawContours(img_with_ctrs , contours, -1, (0, 255, 0), 2)  
    # show(img_with_ctrs)
    
    # filter contours by size:
    A_estimate = A/6/6 # assume dice takes up 1/6 of the image length
    mask_bgr = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    contours_size_filtered = []
    for c in contours:
        if A_estimate*(1-size_tolerance)<cv2.contourArea(c)<A_estimate*(1+size_tolerance):
            contours_size_filtered.append(c)
    # #debug size filtered contours:
    # img_with_ctrs = img.copy()
    # cv2.drawContours(img_with_ctrs, contours_size_filtered, -1, (0, 255, 0), 2)  
    # show(img_with_ctrs)      
    
    # #filter out square like shapes:
    squares = []
    for c in contours_size_filtered:
        area = cv2.contourArea(c)
        (x,y), (w,h), angle = rect = cv2.minAreaRect(c)
        aspect_ratio = w / float(h)
        if not (1-square_tolerance <= aspect_ratio <= 1+square_tolerance):
            continue
        if not (w*h*(1-square_tolerance) < area < w*h*(1+square_tolerance)):
            continue
        
        square = cv2.boxPoints([(x,y),(w,h),angle])
        square = np.int32(square)
        
        squares.append(square)
    
    row_tol = 80  # pixels; adjust as needed

    def sort_key(c):
        x, y, w, h = cv2.boundingRect(c)
        row = round(y / row_tol)
        return (row, x)
    
    squares = sorted(squares, key=sort_key) # sort from top left to bottom right
    # #debug square filtered contours:
    # img_with_ctrs = img.copy()
    # cv2.drawContours(img_with_ctrs, squares, -1, (0, 255, 0), 2)  
    # show(img_with_ctrs,title=f'#squares:{len(squares)}')  
    # print(len(squares))
    return squares

def separate_dice(img,cells):
    """returns imgs of dice"""
    cell_images = []
    for i, c in enumerate(cells):
        x, y, w, h = cv2.boundingRect(c)
    
        # optional padding (helps avoid cutting borders)
        pad = 2
        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + w + pad, img.shape[1])
        y2 = min(y + h + pad, img.shape[0])
    
        cell = img[y1:y2, x1:x2].copy()
    
        cell_images.append(resize_image(cell))
    return cell_images
    
def read_letter(img,templates):
    """Reads the letters. Option A: compare to template images. Option B: Build machine learning 
    model that recognizes them regardles of orientation"""
    bw = make_bw(img)
    best_agreement=('',0)
    for r in range(4): # check all four rotations
        for letter in templates.keys():
            result = cv2.matchTemplate(
                 bw,
                 templates[letter],
                 cv2.TM_CCOEFF_NORMED) [0][0]
            if result>best_agreement[1]:
                best_agreement = (letter,result)
        bw = rotate_image(bw)
    
    return best_agreement[0]


# Executions:
def get_dice(img,linekwargs={},cellkwargs={}):
    """this function takes an already cropped boggle board 
    and returns the dice as separate images"""
    img = draw_lines(img,**linekwargs)
    # show(img)
    cells = detect_cells(img,**cellkwargs)
    if len(cells)!=16:
        print(f'Got {len(cells)} instead of 16.')
        return False
    cell_images = separate_dice(img,cells)
    return cell_images
    
def read_board(img,templates):
    """ reads boggle board and returns letters in np.array. 
    (templates are exported externally to avoid reimporting 
     if you loop over multiple images"""
    print(f'cropping board')
    board = get_board(img)
    print(f'detecting cells')
    dice_imgs = get_dice(board)
    if not dice_imgs:
        return False
    print(f'Reading letters')
    letters = []
    for n,dice_img in enumerate(dice_imgs):
        print(f'{n}/{len(dice_imgs)}')
        letter = read_letter(dice_img,templates)
        if not letter:
            letter = 'A' #TODO: just add default letter to test the code...needs fixing
        letters.append(letter)
    letters = np.reshape(np.array(letters),[4,4])    
    return letters
        
    
    