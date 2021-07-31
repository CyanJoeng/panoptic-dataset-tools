'''
Author: Cyan
Date: Tue Jul 27 00:17:02 CST 2021
'''
import os
import json
import numpy as np
from os import path

class SyncReader:
    def __init__(self, ksync_file, joint_dir, node_id) -> None:
        node_name = f'KINECTNODE{node_id}'
        with open(ksync_file) as kf:
            data = json.load(kf)
            color_time = data['kinect']['color'][node_name]['univ_time']
            depth_time = data['kinect']['depth'][node_name]['univ_time']
            
            self._color_time = color_time
            self._depth_time = depth_time

        assert path.isdir(joint_dir)

        timed_bodies = {}
        joints_file = os.listdir(joint_dir)
        for joint in sorted(joints_file):
            with open(path.join(joint_dir, joint)) as f:
                data = json.load(f)
                time = data['univTime']
                bodies = data['bodies']
                if len(bodies) == 0:
                    continue
                bodies = np.array(bodies[0]['joints19'])
                timed_bodies[time] = bodies

        self._timed_bodies = timed_bodies
        self._bodies_idx = list(self._sync())

    def __len__(self):
        return len(self._bodies_idx)

    def __getitem__(self, idx):
        if idx > len(self._bodies_idx):
            raise Exception('sync out of range')
        bodies, did, cid = self._bodies_idx[idx]
        bodies = np.array(bodies).reshape(-1, 4)[:, :3] / 100
        return bodies, did, cid

    @staticmethod
    def _time_iter(time_list):
        idx_time_iter = iter(enumerate(time_list))
        none_val = (-1, 1e10)
        return lambda cur=None: (none_val if cur is None else cur[1], next(idx_time_iter, none_val))

    @staticmethod
    def _pick(aim_time, pair):
        last, cur = pair
        diff_l, diff_r = aim_time - last[1], cur[1] - aim_time
        target = last if diff_l <= diff_r else cur
        # print(f'pick  {aim_time} {target[1]} ,  {target[1] - aim_time}')
        return target

    @staticmethod
    def _is_in(aim_time, pairs, sync_offset=0):
        last, cur = pairs
        # print('aim', aim_time, 'in', f'{last[1]}, {cur[1]}')
        return (last[1] - sync_offset) < aim_time and (cur[1] - sync_offset) >= aim_time

    def _nearest(self, next_color, next_depth):
        color_pair = next_color()
        depth_pair = next_depth()

        for aim_time in self._timed_bodies.keys():
            aim_time += 32

            while not SyncReader._is_in(aim_time, color_pair, 6.25):
                color_pair = next_color(color_pair)

            while not SyncReader._is_in(aim_time, depth_pair):
                depth_pair = next_depth(depth_pair)

            yield SyncReader._pick(aim_time, color_pair), SyncReader._pick(aim_time, depth_pair)
    
    def _sync(self):
        for bodies, (color, depth) in zip(self._timed_bodies.values(), self._nearest(self._time_iter(self._color_time), self._time_iter(self._depth_time))):

            print(f'iter {color} {depth}')

            if depth[0] == -1 or color[0] == -1:
                break
            if abs(depth[1] - color[1]) > 6.5:
                continue

            print(f'yield {color} {depth}')
            yield (bodies, depth[0], color[0])

