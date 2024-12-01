# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file: Process the point-cloud and prepare it for object detection
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------

# General package imports
import cv2
import easydict
import numpy as np
import open3d as o3d
import os
import sys
import torch
import zlib

# Add project directory to PYTHONPATH to enable relative imports
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# Simple Waymo Open Dataset Reader library
from tools.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2
from tools.waymo_reader.simple_waymo_open_dataset_reader import label_pb2

def _object_type_name(x: int) -> str:
    """Returns the class label mapped to the input class id."""
    return label_pb2.Label.Type.Name(x)

def _laser_name(x: int) -> str:
    """Returns the LiDAR sensor name mapped to the input id."""
    return dataset_pb2.LaserName.Name.Name(x)

def _camera_name(x: int) -> str:
    """Returns the camera name mapped to the input id."""
    return dataset_pb2.CameraName.Name.Name(x)

def _load_range_image(frame: dataset_pb2.Frame, lidar_name: int) -> np.ndarray:
    """Returns the range image from the `frame` captured by `lidar_name` sensor.

    Args:
        frame (dataset_pb2.Frame): The Waymo Open Dataset `Frame` instance.
        lidar_name (int): The integer id corresponding to the LiDAR sensor name.

    Returns:
        np.ndarray: The range image as a Numpy `ndarray` object.
    """
    laser_data = [laser for laser in frame.lasers if laser.name == lidar_name][0]
    ri = []
    if len(laser_data.ri_return1.range_image_compressed) > 0:
        ri = dataset_pb2.MatrixFloat()
        ri.ParseFromString(zlib.decompress(laser_data.ri_return1.range_image_compressed))
        ri = np.array(ri.data).reshape(ri.shape.dims)
    return ri

def show_pcl(pcl: np.ndarray):
    """Displays the LiDAR point cloud data in an Open3D viewer.

    Args:
        pcl (np.ndarray): The 3D point cloud to visualise.
    """
    def close_window(vis: o3d.visualization.Visualizer) -> bool:
        vis.close()
        return False

    visualiser = o3d.visualization.VisualizerWithKeyCallback()
    str_window = 'Visualising the Waymo Open Dataset: LiDAR Point Cloud data'
    visualiser.create_window(window_name=str_window, width=1280, height=720, left=50, top=50, visible=True)
    visualiser.register_key_callback(key=262, callback_func=close_window)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcl[:, :3])
    visualiser.add_geometry(pcd)
    visualiser.run()

def show_range_image(frame: dataset_pb2.Frame, lidar_name: int) -> np.ndarray:
    """Returns the range image given in the `frame` captured by `lidar_name`.

    Args:
        frame (dataset_pb2.Frame): The Waymo Open Dataset `Frame` instance.
        lidar_name (int): The integer id corresponding to the LiDAR sensor name.

    Returns:
        np.ndarray: The range and intensity channels of the range image as a Numpy `ndarray` object.
    """
    ri = _load_range_image(frame, lidar_name)
    ri_range = ri[:, :, 0]
    ri_intensity = ri[:, :, 1]
    MIN_RANGE = 0
    MAX_RANGE = 75 if lidar_name is dataset_pb2.LaserName.TOP else 20
    np.clip(ri_range, MIN_RANGE, MAX_RANGE)
    ri_range = ri_range * 255 / (np.amax(ri_range) - np.amin(ri_range))
    ri_range = ri_range.astype(np.uint8)
    ri_min = np.percentile(ri_intensity, 1)
    ri_max = np.percentile(ri_intensity, 99)
    np.clip(ri_intensity, a_min=ri_min, a_max=ri_max)
    ri_intensity = np.int_((ri_intensity - ri_min) * 255. / (ri_max - ri_min))
    ri_intensity = ri_intensity.astype(np.uint8)
    img_range_intensity = np.vstack((ri_range, ri_intensity))
    img_range_intensity = img_range_intensity.astype(np.uint8)
    return img_range_intensity

def bev_from_pcl(lidar_pcl: np.ndarray, configs: easydict.EasyDict, vis: bool = False):
    """Converts the point cloud to a BEV map.

    Args:
        lidar_pcl (np.ndarray): The point cloud to clip and convert to BEV map.
        configs (easydict.EasyDict): The EasyDict instance storing the viewing range and filtering params.
        vis (bool, optional): If True, will visualise the resulting BEV map using the Open3D `Visualizer` interface.

    Returns:
        torch.Tensor: The 3-channel RGB-like BEV map tensor.
    """
    pcl_range = lidar_pcl[:, 0]
    pcl_intensity = lidar_pcl[:, 1]
    pcl_elongation = lidar_pcl[:, 2]
    mask = np.where((pcl_range >= configs.lim_x[0]) & (pcl_range <= configs.lim_x[1]) &
                    (pcl_intensity >= configs.lim_y[0]) & (pcl_intensity <= configs.lim_y[1]) &
                    (pcl_elongation >= configs.lim_z[0]) & (pcl_elongation <= configs.lim_z[1]))
    lidar_pcl = lidar_pcl[mask]
    lidar_pcl[:, 2] = lidar_pcl[:, 2] - configs.lim_z[0]

    bev_interval = (configs.lim_x[1] - configs.lim_x[0]) / configs.bev_height
    bev_offset = (configs.bev_width + 1) / 2
    lidar_pcl_cpy = lidar_pcl.copy()
    lidar_pcl_cpy[:, 0] = np.int_(np.floor(lidar_pcl_cpy[:, 0] / bev_interval))
    lidar_pcl_cpy[:, 1] = np.int_(np.floor(lidar_pcl_cpy[:, 1] / bev_interval) + bev_offset)
    lidar_pcl_cpy[lidar_pcl_cpy < 0.0] = 0.0
    if vis:
        show_pcl(lidar_pcl_cpy)

    lidar_pcl_cpy[lidar_pcl_cpy[:, 3] > 1.0, 3] = 1.0
    intensity_map = np.zeros(shape=(configs.bev_height + 1, configs.bev_height + 1))
    idxs_intensity = np.lexsort(keys=(-lidar_pcl_cpy[:, 2], lidar_pcl_cpy[:, 1], lidar_pcl_cpy[:, 0]))
    lidar_pcl_top = lidar_pcl_cpy[idxs_intensity]
    _, idxs_top_unique, counts = np.unique(lidar_pcl_top[:, 0:2], axis=0, return_index=True, return_counts=True)
    lidar_pcl_top = lidar_pcl_cpy[idxs_top_unique]
    intensity_vals = lidar_pcl_top[:, 3]
    scale_factor_intensity = np.amax(intensity_vals) - np.amin(intensity_vals)
    intensity_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = lidar_pcl_top[:, 3] / scale_factor_intensity
    if vis:
        img_intensity = intensity_map * 256
        img_intensity = img_intensity.astype(np.uint8)
        str_title = "Bird's-eye view (BEV) map: normalised intensity channel values"
        cv2.imshow(str_title, img_intensity)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    height_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    scale_factor_height = float(np.abs(configs.lim_z[1] - configs.lim_z[0]))
    height_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = lidar_pcl_top[:, 2] / scale_factor_height
    if vis:
        img_height = height_map * 256
        img_height = img_height.astype(np.uint8)
        str_title = "Bird's-eye view (BEV) map: normalised height channel values"
        cv2.imshow(str_title, img_height)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    density_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    _, _, counts = np.unique(lidar_pcl_cpy[:, 0:2], axis=0, return_index=True, return_counts=True)
    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64))
    density_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = normalizedCounts

    bev_map = np.zeros((3, configs.bev_height, configs.bev_width))
    bev_map[2, :, :] = density_map[:configs.bev_height, :configs.bev_width]
    bev_map[1, :, :] = height_map[:configs.bev_height, :configs.bev_width]
    bev_map[0, :, :] = intensity_map[:configs.bev_height, :configs.bev_width]

    s1, s2, s3 = bev_map.shape
    bev_maps = np.zeros((1, s1, s2, s3))
    bev_maps[0] = bev_map
    bev_maps = torch.from_numpy(bev_maps)
    input_bev_maps = bev_maps.to(configs.device, non_blocking=True).float()
    return input_bev_maps