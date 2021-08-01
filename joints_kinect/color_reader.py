'''
Author: Cyan
Date: Mon Jul 26 23:33:45 CST 2021
'''
import os
from cv2 import cv2 as cv

class ColorReader:
    def __init__(self, color_dir):
        frame_files = os.listdir(color_dir)
        frame_files.sort()
        self._count = 0
        if len(frame_files) != 0:
            self._count = len(frame_files)
            print(f'color frame count {self._count}')
            self._frames_path = [os.path.join(color_dir, name) for name in frame_files]

    def __len__(self):
        return self._count

    def __getitem__(self, idx):
        if idx > self._count:
            raise Exception('color index out of range')
        frame = cv.imread(self._frames_path[idx])
        return frame

