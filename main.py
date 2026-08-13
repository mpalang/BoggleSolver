from BoggleSolver import BoggleSolver
from utils.paths import TRAININGSIMAGES as ti
from recognition.image_processing import show
import numpy as np

img_paths = list(ti.glob('*.jpg'))
b = BoggleSolver(img_paths[1],auto_solve=False)
b.get_board()

#%%
from recognition.dice import find_dice,debug_watershed
from recognition.image_processing import show
import cv2

mask, d, p, mr, m = find_dice(b.board, debug=True)
dv, pv, mrv, mv = debug_watershed(mask,d,p,mr,m)
show(dv)
show(pv)