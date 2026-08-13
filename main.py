from BoggleSolver import BoggleSolver
from utils.paths import TRAININGSIMAGES as ti
from recognition.image_processing import show
import numpy as np

img_paths = list(ti.glob('*.jpg'))
b = BoggleSolver(img_paths[5])
show(b.draw_dice())
print(b.valid_words)

#%%
# from recognition.letters import read_letter
# from recognition.image_processing import canon, rotate_image

# result = read_letter(b.dice[0], b.templates)

# #%%

# show(b.dice[9])
# show(canon(b.dice[9]))
# show(b.templates['M'])
