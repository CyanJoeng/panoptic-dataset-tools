'''
Author: Cyan
Date: Mon Jul 26 23:33:45 CST 2021
'''
from cv2 import cv2 as cv

class ColorReader:
    def __init__(self, color_file) -> None:
        self._frames = cv.VideoCapture(color_file) 
        self._count = 0
        if self._frames.isOpened():
            self._count = int(self._frames.get(cv.CAP_PROP_FRAME_COUNT))
        
    def __len__(self):
        return self._count

    def __getitem__(self, idx):
        print(f'get color id: {idx}/{self._count}')
        if idx > self._count:
            raise Exception('color index out of range')
        self._frames.set(cv.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self._frames.read()
        if not ret:
            raise Exception(f'error frame by idx:{idx}')
        return frame

