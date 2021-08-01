'''
Author: Cyan
Date: Mon Jul 26 23:46:26 CST 2021
'''
import numpy as np

class DepthReader:
    shape = (424, 512)
    frame_len = 2 * 424 * 512

    def __init__(self, depth_file) -> None:
        self._depth_file = depth_file
        frames = open(depth_file, 'rb')
        frames.seek(0, 2)
        self._count = frames.tell() // DepthReader.frame_len

        self._frames = None

    def __len__(self):
        return self._count

    def __getitem__(self, idx):
        if self._frames is None:
            self._frames = open(self._depth_file, 'rb')

        self._frames.seek(idx * DepthReader.frame_len, 0)
        raw_data = self._frames.read(DepthReader.frame_len)
        frame = np.frombuffer(raw_data, dtype=np.uint16)
        return np.array(frame).reshape(DepthReader.shape[0], DepthReader.shape[1])
