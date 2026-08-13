import cv2
from pathlib import Path
import numpy as np

from utils.tools import progress_bar
from recognition.image_processing import (load_image, resize_image, make_bw, canon, rotate_image)
from utils.paths import RESOURCES


def load_templates(raw=False,cropped=False, use_bw=False, use_canon=True):
    """loads template pictures and converts to black and white for easier comparison"""
    if raw:
        base = RESOURCES / 'templates_original'
    elif cropped:
        base = RESOURCES / 'templates_cropped'
    else:
        base = RESOURCES / 'templates'

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


def resize_templates(size=None, out_path = RESOURCES / "templates"):  
    """This function is making sure all templates are the same size"""
    templates = load_templates(cropped=True,bw=False)
    for letter,template in templates.items():
        cropped = resize_image(template,size=size)
        path = str(out_path / f"{letter}.jpg")
        check = cv2.imwrite(path, cropped)
        if not check:
            raise RuntimeError(f'Failed writing template:{path}')


def read_letters(dice: list[np.ndarray], 
                templates: dict[str,np.ndarray] | None = None
                ):
    """Reads the letters. Compare to template images"""

    if templates is None:
        templates = load_templates()

    results = []
    for n,d in enumerate(dice):
        progress_bar(n+1,len(dice),'Reading dice:')
        results.append(read_letter(d,templates))

    def margin(res):
        best = res[0][0]
        second = next((s for l, s, _ in res if l != best), -1.0)
        return res[0][1] - second
    
    letters = [r[0][0] for r in results]
    confidences = [margin(r) for r in results]
    rotations = [r[0][2] for r in results]
    return letters, confidences, rotations


def read_letter(dice: np.ndarray, templates: dict[str, np.ndarray], use_canon=True):
    """reads letter from boggle die """
    result = []
    if use_canon:
        type(dice)
        bw = canon(dice)
    else:
        bw = make_bw(dice)
    for letter in templates.keys():
        for r in range(4): # check all four rotations
            score = cv2.matchTemplate(
                    bw,
                    templates[letter],
                    cv2.TM_CCOEFF_NORMED
                    ) [0][0]
            result.append((letter,score,r))
            bw = rotate_image(bw)
    return sorted(result, key = lambda x: x[1], reverse = True)

    

