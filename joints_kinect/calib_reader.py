'''
Author: Cyan
Date: Mon Jul 26 23:59:24 CST 2021
'''
import json
import numpy as np


class CalibReader:
    def __init__(self, calib_file, kcalib_file, node_id) -> None:
        with open(calib_file) as f:
            calib = json.load(f)
            camera = calib['cameras'][-11 + node_id]
            self._pan_calib = camera

        with open(kcalib_file) as kf:
            calib = json.load(kf)
            camera = calib['sensors'][node_id - 1]
            self._k_calib = camera

    def joints_k_color(self):
        R = self._pan_calib['R']
        t = self._pan_calib['t']
        R, t = np.array(R), np.array(t) / 100
        return R, t

    def k_depth_color(self):
        T = self._k_calib['M_color']
        T = np.array(T)
        T[:3, :3] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) * T[:3, :3]
        return T[:3, :3], T[:3, 3:]

    def depth_proj(self):
        K = self._k_calib['K_depth']
        d = self._k_calib['distCoeffs_depth']
        K, d = np.array(K), np.array(d)
        return K, d

    def color_proj(self):
        K = self._k_calib['K_color']
        d = self._k_calib['distCoeffs_color']
        K, d = np.array(K), np.array(d)
        return K, d

