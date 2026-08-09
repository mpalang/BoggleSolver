import cv2
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from boggle_solver.utils import (order_points,warp,
                       rotate_image,load_templates,
                       canon, make_bw,
                       WordFinder)
from Dictionary.utils import load_trie


def K(n: int):
    """Creates Kernel of size n. Keep it uneven no shifting happens."""
    if n%2==0:
        raise ValueError('kernel size needs to be uneven')
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
    

class BoggleSolver:
    def __init__(self, source: str | Path | np.ndarray | None = None, auto_solve=True):
        self.raw_img = None
        self.img = None
        self.img_path = Path()
        self.templates = load_templates()
        self.trie = load_trie()
        
        self.valid_words = []

        if source is None:
            return
        if isinstance(source, str | Path):
            self.img_path = Path(source)
            self.load_image(source)
        elif isinstance(source, np.ndarray):
            self.raw_img = source
            self.normalize_img()
        else:
            raise TypeError(f'unsupported source type:{type(source).__name__}')
        
        if auto_solve:
            self.solve_board()

    def load_image(self,path):
        self.raw_img = cv2.imread(path)

    
    def normalize_img(self, long_edge=1200):
        s = long_edge / max(self.raw_img.shape[:2])
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
        self.img = cv2.resize(self.raw_img, None, fx=s, fy=s, interpolation=interp)


    def get_board(self, out=512, inset=0.94, blur = False, debug = True):

        #Blur to get rid of noise
        if blur:
            img = cv2.GaussianBlur(self.img, K(5), 0)
        else:
            img = self.img

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
        self.board = warp(self.img, box, out)

        return self.board
    

    def find_dice(self):
        """Detect dice in board."""
        board = self.board
        # board = cv2.GaussianBlur(board, K(5), 0)
        #Get orange mask:
        hsv = cv2.cvtColor(board, cv2.COLOR_BGR2HSV)
        orange = cv2.inRange(hsv, (5, 100, 90), (70, 255, 255))
        #Get rid of small holes:
        orange_morph = cv2.morphologyEx(orange, cv2.MORPH_OPEN, K(5))
        orange_morph = cv2.morphologyEx(orange_morph, cv2.MORPH_CLOSE, K(11))

        #Draw extra lines that strengthen the grid and improve dice separation:
        lines = self.get_lines(orange_morph)
        orange_with_lines = self.draw_lines(orange,lines)

        #Bring dice into foreground:
        mask = cv2.bitwise_not(orange_with_lines)

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
            c = cv2.findContours((markers == i).astype(np.uint8),
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            if not c:
                continue    
            rect = cv2.minAreaRect(max(c, key=cv2.contourArea))

            (cx, cy), (w,h), angle = rect
            #Size filter:
            H, W = self.board.shape[:2]
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

        #Write to class:
        self.dice_squares = squares
        self.dice_boxes = [cv2.boxPoints(square) for square in self.dice_squares]


    def get_lines(self,mask,blur_kernel=5,blur_sigma=0,
                canny_thresholds=(80,200),
                rho=5,theta=np.pi/360,threshold=500,):
        """ draw straight lines between dice. Takes bw mask."""
        edges = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, K(3))   # 1-px outlines
        lines = cv2.HoughLines(edges,     
                            rho=rho,
                            theta=theta,
                            threshold=threshold)
        return lines


    def draw_lines(self,input_img,lines, color=255):
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


    def draw_dice(self, thickness=2, color=(0, 255, 0)):
        """draw dice on image"""
        vis = self.board.copy()
        cnts = [np.int32(c) for c in self.dice_boxes]
        cv2.drawContours(vis, cnts, -1, color, thickness)
        return vis


    def cut_dice(self):
        """cut dice from board for letter recognition"""
        cells = []
        pad = 1.05
        size = 128 #has to match template size
        dst = np.array([[0,0], [size-1,0], [size-1,size-1], [0,size-1]], np.float32)
        for (cx, cy), (w, h), ang in self.dice_squares:
            s = (w + h) / 2 * pad
            src = cv2.boxPoints(((cx, cy), (s, s), ang)).astype(np.float32)
            M = cv2.getPerspectiveTransform(src, dst)
            cells.append(cv2.warpPerspective(self.board, M, (size, size)))

        self.dice = cells


    def read_letters(self, use_canon=True):
        """Reads the letters. Option A: compare to template images. TODO: Option B: Build machine learning 
        model that recognizes them regardles of orientation"""
        letters = []
        agreement = []
        confidence = []
        rotation = []
        for d in self.dice:
            if use_canon:
                bw = canon(d)
            else:
                bw = make_bw(d)
            best_agreement=('',1e-6,1,None)
            for r in range(4): # check all four rotations
                for letter in self.templates.keys():
                    score = cv2.matchTemplate(
                        bw,
                        self.templates[letter],
                        cv2.TM_CCOEFF_NORMED) [0][0]
                    if score>best_agreement[1]:
                        best_agreement = (letter,score,score/best_agreement[1],r)
                bw = rotate_image(bw)

            letters.append(best_agreement[0])
            agreement.append(best_agreement[1])
            confidence.append(best_agreement[2])
            rotation.append(best_agreement[3])

        self.letters = np.array(letters).reshape((4,4))
        self.score = np.array(agreement).reshape((4,4))
        self.confidence = np.array(confidence).reshape((4,4))
        self.rotation = rotation


    def read_letter(self,n=0, use_canon=True):
        """read single die (debug etc...)"""
        img = self.dice[n].copy()
        bw = make_bw(img)
        if use_canon:
            bw = canon(bw)
        if img.shape[:2] != self.templates['A'].shape[:2]:
            print(f'Dice size {img.shape[:2]} must match template size {self.templates['A'].shape[:2]} but.')
            return
        
        results = dict()
        for letter in self.templates.keys():
            results[letter] = []
            for r in range(4): # check all four rotations
                score = cv2.matchTemplate(
                    bw,
                    self.templates[letter],
                    cv2.TM_CCOEFF_NORMED) [0][0]

                results[letter].append(round(float(score),3))
                bw = rotate_image(bw)
        return results


    def solve_board(self):
        self.get_board()
        self.find_dice()
        self.cut_dice()
        self.read_letters()
        finder = WordFinder(self.letters, self.trie)
        
        min_word_length = 4
        self.valid_words = [word for word in finder.valid_words if len(word)>=min_word_length]
        self.valid_words = sorted(self.valid_words, key=lambda w: (len(w), w))
        
        
    
    