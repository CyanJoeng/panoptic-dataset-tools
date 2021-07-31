'''
Author: Cyan
Date: Wed Jul 28 00:43:51 CST 2021
'''
from os import path
from time import sleep
import sys
sys.path.insert(0, path.dirname(path.dirname(path.abspath(__file__))))

from cv2 import cv2 as cv, data
import open3d as o3d
import numpy as np

from joints_kinect import Dataset

import torch


def main(dataset_dir, node_id):
    dataset = Dataset(dataset_dir, node_id)
    yolov5 = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    obj_cloud = o3d.geometry.PointCloud()
    obj_joints = [o3d.geometry.TriangleMesh.create_sphere(0.05) for _ in range(19)]
    for obj in obj_joints:
        obj.paint_uniform_color(np.array([1, 1, 0]))
        vis.add_geometry(obj)
    vis.add_geometry(obj_cloud)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=.5))

    for i in range(len(dataset)):
        joints, depth, color = dataset[i]
        print(joints.shape)
        print(depth.shape)
        print(color.shape)

        joints = dataset.joints_to_color(joints)
        print(joints.shape)
        
        person = yolov5(color).xyxy[0]
        roi = (0, 0, color.shape[1], color.shape[0])
        if len(person) != 0:
            person = person[0]
            min_x, min_y, max_x, max_y, p, c = person
            if c.item() == 0:
                roi = (int(min_x), int(min_y), int(max_x), int(max_y))

        color_cloud, _ = dataset.depth_to_color_cloud(depth, color.shape, roi)
        print('cloud shape ', color_cloud.shape)

        if True:
            ''' display 3d objects '''

            color_cloud = color_cloud.reshape(-1, 3)
            obj_cloud.points = o3d.utility.Vector3dVector(color_cloud)
            obj_cloud.paint_uniform_color(np.array([0.3, 0.3, 0.3]))
            vis.update_geometry(obj_cloud)

            for obj, pt in zip(obj_joints, joints):
                print('joints', pt)
                center = obj.get_center()
                obj.translate(pt - center)
                vis.update_geometry(obj)

            vis.poll_events()
            vis.update_renderer()

            ''' display 2d images and joints '''
            proj_joints = dataset.project_pts(joints)
            for idx, pt in enumerate(proj_joints):
                u, v = int(pt[0]), int(pt[1])
                color = cv.circle(color, (u, v), 2, (255, 255, 0), 2)
                color = cv.putText(color, f'{idx}', (u,v), 0, 1, (255, 0, 255), 2)
            cv.imshow('color', color)
            cv.imshow('depth', depth / depth.max())
            if 113 == cv.waitKey(10):
                break
        else:
            sleep(1)

        print()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', '-d', help='cmu panopti dataset dir', type=str, required=True)
    parser.add_argument('--node_id', '-n', help='node id of kinect cameras', default=2)

    args = parser.parse_args()
    main(**args.__dict__)
