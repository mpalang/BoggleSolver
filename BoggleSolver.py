from pathlib import Path
from numpy import typing as nt
import numpy as np
from recognition.image_processing import (load_image, normalize_img, decode_image)
from recognition.board import (get_board)
from recognition.dice import (find_dice, cut_dice, draw_dice)
from recognition.letters import (load_templates, read_letters)
from solving.WordFinder import WordFinder
import cv2

MINWORDLENGTH = 4


class BoggleSolver:
    def __init__(self, source: str | Path | nt.NDArray | None = None, 
                 auto_solve = True,
                 dictionary = 'small'):
        
        self.img_path = None
        self.raw_img = None
        self.img = None
        self.board = None
        self.dice_squares = None
        self.dice = None
        self.templates = load_templates()
        self.letters = None
        self.valid_words = []

        self.wordFinder = WordFinder(dictionary = dictionary)

        self.post_init(source, auto_solve=auto_solve)


    def post_init(self, source, auto_solve=True):
        if source is None:
            return

        if isinstance(source, str | Path):
            self.img_path = Path(source)
            self.raw_img = load_image(self.img_path)
        elif isinstance(source, np.ndarray):
            self.raw_img = source
        elif isinstance(source, bytes | bytearray | memoryview):
            self.raw_img = decode_image(source)
        else:
            raise TypeError(f"unsupported source type: {type(source).__name__}")

        self.normalize_img()
        if auto_solve:
            self.solve()

    @classmethod
    def from_path(cls, path: str | Path, **kwargs):
        return cls(Path(path), **kwargs)

    @classmethod
    def from_array(cls, img: np.ndarray, **kwargs):
        return cls(img, **kwargs)

    @classmethod
    def from_bytes(cls, data: bytes, **kwargs):
        """Decode raw image bytes (e.g. an HTTP upload)."""
        return cls(decode_image(data), **kwargs)
    
    
    def load_image(self):
        self.raw_img = load_image(self.img_path)
        return self.raw_img

    
    def normalize_img(self):
        self.img = normalize_img(self.raw_img)
        return self.img


    def get_board(self):
        self.board = get_board(self.img)
        return self.board
    

    def find_dice(self):
        self.dice_squares = find_dice(self.board)
        return self.dice_squares


    def cut_dice(self):
        self.dice = cut_dice(self.board, self.dice_squares)
        return self.dice


    def read_letters(self):
        self.letters, self.confidences, self.rotations = read_letters(self.dice, templates=self.templates)
        return self.letters, self.confidences, self.rotations


    def find_words(self):
        self.wordFinder.solve(self.letters)
        self.valid_words = self.wordFinder.valid_words
        self.valid_words = [word for word in self.valid_words if len(word)>=MINWORDLENGTH]
        self.valid_words = sorted(self.valid_words, key=lambda w: (len(w), w))


    def solve(self):
        print('detecting board')
        self.get_board()
        print('finding dice')
        self.find_dice()
        self.cut_dice()
        print('reading letters')
        self.read_letters()
        print('looking for words')
        self.find_words()

        print(f"{len(self.valid_words)} words found with at least {MINWORDLENGTH} letters.")


    def set_dictionary(self,dictionary: Path | str):
        self.wordFinder = WordFinder(dictionary = dictionary) 
        return self.wordFinder


    def draw_squares(self):
        return draw_squares(self.board, self.dice_squares, numbers = True)


    def draw_dice(self):
        board_with_dice = draw_dice(
                        self.board,
                        self.dice_squares,
                        self.letters,
                        self.confidences,
                        self.rotations,
                        self.templates,
                        )
        return board_with_dice


    @property
    def processed_board(self) -> np.ndarray:
        return self.draw_dice()

    def as_jpeg(self, img=None, quality=90) -> bytes:
        img = self.processed_board if img is None else img
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise ValueError("JPEG encoding failed")
        return buf.tobytes()


    

    
        
    
    