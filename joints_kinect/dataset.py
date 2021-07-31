'''
Author: Cyan
Date: Mon Jul 26 23:26:05 CST 2021
'''
from joints_kinect.calib_reader import CalibReader
from os import path

import numpy as np
from cv2 import cv2 as cv

from .color_reader import ColorReader
from .depth_reader import DepthReader
from .sync_reader import SyncReader


class Dataset:
    def __init__(self, database_dir, node):
        root = database_dir

        self._color = ColorReader(path.join(root, 'kinectVideos', f'kinect_50_{node:02d}.mp4'))
        self._depth = DepthReader(path.join(root, 'kinect_shared_depth', f'KINECTNODE{node}', 'depthdata.dat'))

        dataset_label = path.basename(database_dir)
        self._sync = SyncReader(path.join(root, f'ksynctables_{dataset_label}.json'), path.join(root, 'hdPose3d_stage1_coco19'), node)

        self._calib = CalibReader(path.join(root, f'calibration_{dataset_label}.json'), path.join(root, f'kcalibration_{dataset_label}.json'), node)

        self._data = list(self._check())

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        if idx > len(self._data):
            raise Exception("out of range")
        joints, depth_id, color_id = self._data[idx]
        print(depth_id >= len(self._depth), color_id >= len(self._color))
        depth = self._depth[depth_id]
        color = self._color[color_id]

        return joints, depth, color

    def _check(self):
        depth_len = len(self._depth)
        color_len = len(self._color)
        for i in range(len(self._sync)):
            bodies_idx = self._sync[i]
            if bodies_idx is None:
                break
            bodies, d_id, c_id = bodies_idx

            if d_id >= depth_len or c_id >= color_len:
                continue

            depth = d_id
            color = c_id

            yield (bodies, depth, color)

    def joints_proj(self, joints):
        K, d = self._calib.color_proj()
        R, t = self._calib.joints_k_color()

        joints = joints[:, :3]

        r, _ = cv.Rodrigues(R)
        pts, _ = cv.projectPoints(joints, r, t, K, d)
        return pts

    def depth_proj(self, depth):
        d_K, d_d = self._calib.depth_proj()
        c_K, c_d = self._calib.color_proj()
        R, t = self._calib.k_depth_color()

        dfx, dfy, dcx, dcy = d_K[0, 0], d_K[1, 1], d_K[0, 2], d_K[1, 2]

        d_d = d_d[:5]

        depth = cv.undistort(depth, d_K, d_d)
        cloud = np.array([[(u - dcx) / dfx * depth[v, u], (v - dcy) / dfy * depth[v, u], depth[v, u]] for v in range(depth.shape[0]) for u in range(depth.shape[1])])

        cloud = np.expand_dims(cloud, axis=0)
        proj_pts, _ = cv.projectPoints(cloud, R, t, c_K, c_d)
        return proj_pts

