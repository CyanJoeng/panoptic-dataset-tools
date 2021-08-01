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

        self._color = ColorReader(path.join(root, 'kinectImgs', f'50_{node:02d}'))
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

    def joints_to_color(self, joints):
        R, t = self._calib.joints_k_color()

        joints = joints[:, :3]

        joints = joints @ np.linalg.inv(R) + t.reshape(1, 3)
        return joints

    def depth_to_color_cloud(self, depth, shape):
        d_K, d_d = self._calib.depth_proj()
        R, t = self._calib.k_depth_color()

        dfx, dfy, dcx, dcy = d_K[0, 0], d_K[1, 1], d_K[0, 2], d_K[1, 2]

        d_d = d_d[:5]

        depth = cv.undistort(depth, d_K, d_d) / 1e3

        cloud = np.array([[(u - dcx) / dfx * depth[v, u], (v - dcy) / dfy * depth[v, u], depth[v, u]] for v in range(depth.shape[0]) for u in range(depth.shape[1]) if depth[v, u] < 3 and depth[v, u] > 1.5])
        cloud = cloud @ np.linalg.inv(R) + t.reshape(1, 3)

        img_pts = self.project_pts(cloud)
        img_pts = np.array(img_pts + 0.5).astype(np.int32)

        img_pts_filter, cloud_filter = self._filter_pts(img_pts, cloud, shape[:2])
        print(f'pts len {img_pts.shape} -> {img_pts_filter.shape}')

        proj_cloud = np.zeros(shape=shape)
        proj_cloud[img_pts_filter[:, 1], img_pts_filter[:, 0], :] = cloud_filter

        return proj_cloud

    def _filter_pts(self, img, cloud, shape):
        img_pts, cloud_pts = img, cloud
        h, w = shape

        for _idx in [
                lambda pts: 0 < pts[:, 0],
                lambda pts: pts[:, 0] < w,
                lambda pts: 0 < pts[:, 1],
                lambda pts: pts[:, 1] < h
                ]:
            idx = np.array(_idx(img_pts))
            img_pts = img_pts[idx]
            cloud_pts = cloud_pts[idx]

        return img_pts, cloud_pts

    def project_pts(self, cloud):
        K, d = self._calib.color_proj()
        proj_pts, _ = cv.projectPoints(cloud, np.zeros(3), np.zeros(3), K, d)
        proj_pts = proj_pts.reshape(-1, 2)
        return proj_pts

