# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 18:46:15 2026

@author: morit

"""
from pathlib import Path
from auxiliary import load_image,show,timer,load_templates,make_bw
from BoggleReader import (read_board,read_letter,
                        get_board,get_dice,
                        draw_lines,detect_cells,separate_dice)
import numpy as np

paths = list(Path('TrainingImages').glob('*.jpg'))
images = [load_image(path) for path in paths]
templates = load_templates()
# for template in templates.values():
#     show(template)


#%%

boards=[]
for image in images[0:1]:
    board = get_board(image)  
    letters = read_board(board,templates)
    if not isinstance(letters,np.ndarray):
        continue
    boards.append(letters)


#%%

# for n,image in enumerate(images):
#     show(get_board(image),title=f'({len(boards[n])}) {boards[n]}')
