# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 18:46:15 2026

@author: morit

"""
from pathlib import Path
import cv2
import numpy as np
from matplotlib import pyplot as plt
from statistic_tools import kde
from scipy.signal import find_peaks
from SeparateDice import get_board,get_dice,load_image,show,read_letter

images = list(Path('TrainingImages').glob('*.jpg'))[1:2]
for n,image in enumerate(images):
    # try:
        img = load_image(image)
        img = get_board(img)  
        # show(img)
        dice = get_dice(img,tolerance=0.2)
        # for d in dice:
        #     show(d)

show(dice[0])

letter = read_letter(dice[0])
print(letter)
