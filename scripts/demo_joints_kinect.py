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

from joints_kinect import Dataset


def main(dataset_dir, node_id):
    dataset = Dataset(dataset_dir, node_id)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    obj_cloud = o3d.geometry.PointCloud()
    obj_joints = o3d.geometry.PointCloud()
    vis.add_geometry(obj_joints)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame())

    for i in range(len(dataset)):
        joints, depth, color = dataset[i]
        print(joints.shape)
        print(depth.shape)
        print(color.shape)

        proj_joints = dataset.joints_proj(joints)
        print(proj_joints.shape)

        proj_cloud = dataset.depth_proj(depth)

        if True:
            ''' display 3d objects '''

            # obj_cloud.points = o3d.utility.Vector3dVector(proj_cloud.reshape(-1, 3))
            # obj_joints.points = o3d.utility.Vector3dVector(joints.reshape(-1, 3))
            # vis.update_geometry(obj_joints)
            # vis.poll_events()
            # vis.update_renderer()

            ''' display 2d images and joints '''
            for idx, pt in enumerate(proj_joints):
                u, v = int(pt[0, 0]), int(pt[0, 1])
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
