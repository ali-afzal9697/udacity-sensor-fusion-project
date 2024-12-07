# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file: Classes for sensor and measurement
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ---------------------------------------------------------------------

import numpy as np
import os
import sys
from typing import List, Union
from math import atan

# Add project directory to PYTHONPATH to enable relative imports
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# Simple Waymo Open Dataset Reader library
from tools.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2

# Measurement model imports
import misc.params as params

class Measurement(object):
    """The Measurement class.

    Args:
        num_frame (int): The frame id from which this measurement was captured.
        z (np.ndarray): The raw measurement vector.
        sensor (Sensor): The Sensor instance from which this measurement was generated.
    """

    def __init__(self, num_frame: int, z: np.ndarray, sensor: 'Sensor'):
        """Initialises a new Measurement instance."""
        self.t = (num_frame - 1) * params.dt
        self.sensor = sensor

        if sensor.name == 'lidar':
            sigma_lidar_x = params.sigma_lidar_x
            sigma_lidar_y = params.sigma_lidar_y
            sigma_lidar_z = params.sigma_lidar_z
            self.z = np.zeros((sensor.dim_meas, 1))
            self.z[0] = z[0]
            self.z[1] = z[1]
            self.z[2] = z[2]
            self.R = np.array([
                [sigma_lidar_x ** 2, 0., 0.],
                [0., sigma_lidar_y ** 2, 0.],
                [0., 0., sigma_lidar_z ** 2]
            ])
            self.width = z[4]
            self.length = z[5]
            self.height = z[3]
            self.yaw = z[6]
        elif sensor.name == 'camera':
            sigma_cam_i = params.sigma_cam_i
            sigma_cam_j = params.sigma_cam_j
            self.z = np.zeros((sensor.dim_meas, 1))
            self.z[0] = z[0]
            self.z[1] = z[1]
            self.R = np.array([
                [sigma_cam_i ** 2, 0.],
                [0., sigma_cam_j ** 2]
            ])
            self.width = z[2]
            self.length = z[3]
        else:
            raise ValueError(f"Invalid sensor type '{sensor.name}'")

class Sensor(object):
    """The Sensor class.

    Args:
        name (str): The name of this sensor instance, can be one of ['camera', 'lidar'].
        calib (Union[dataset_pb2.CameraCalibration, dataset_pb2.LaserCalibration]): The protobuf container corresponding to the sensor calibration attributes.
    """

    def __init__(self, name: str, calib: Union[dataset_pb2.CameraCalibration, dataset_pb2.LaserCalibration]):
        """Initialises a new Sensor instance."""
        self.name = name
        if name == 'lidar':
            self.dim_meas = 3
            self.sens_to_veh = np.array(np.identity(n=4))
            self.fov = [-np.pi / 2, np.pi / 2]
        elif name == 'camera':
            self.dim_meas = 2
            self.sens_to_veh = np.array(calib.extrinsic.transform).reshape(4, 4)
            self.f_i = calib.intrinsic[0]
            self.f_j = calib.intrinsic[1]
            self.c_i = calib.intrinsic[2]
            self.c_j = calib.intrinsic[3]
            self.fov = [-0.35, 0.35]
        else:
            raise ValueError(f"Invalid sensor type '{name}'")
        self.veh_to_sens = np.linalg.inv(self.sens_to_veh)

    def in_fov(self, x: np.ndarray) -> bool:
        """Checks if the given object `x` is within the sensor field of view.

        Args:
            x (np.ndarray): The object state vector to obtain the coordinates from.

        Returns:
            bool: Whether or not the object at its position can be seen by the sensor.
        """
        _p_veh = x[0:3]
        _p_veh = np.vstack([_p_veh, np.newaxis])
        _p_veh[3] = 1

        _p_sens = self.veh_to_sens @ _p_veh
        p_x, p_y, _ = _p_sens[0:3]
        if p_x == 0:
            raise ZeroDivisionError(f"Invalid coordinates (sensor frame) '{_p_sens.tolist()}'")
        alpha = atan(p_y / p_x)
        return np.min(self.fov) <= alpha <= np.max(self.fov)

    def get_hx(self, x: np.ndarray) -> np.ndarray:
        """Implements the non-linear camera measurement function.

        Args:
            x (np.ndarray): The track state vector used to calculate the expectation value.

        Returns:
            np.ndarray: The non-linear camera measurement function evaluated at the given `x`.
        """
        if self.name == 'lidar':
            _p_veh = np.vstack([x[0:3], np.newaxis])
            _p_veh[3] = 1
            _p_sens = self.veh_to_sens @ _p_veh
            return _p_sens[0:3]
        elif self.name == 'camera':
            _p_veh = np.vstack([x[0:3], np.newaxis])
            _p_veh[3] = 1
            _p_sens = self.veh_to_sens @ _p_veh
            if _p_sens[0, 0] == 0:
                raise ZeroDivisionError(f"Invalid coordinates (sensor frame) '{_p_sens.tolist()}'")
            i = self.c_i - self.f_i * _p_sens[1, 0] / _p_sens[0, 0]
            j = self.c_j - self.f_j * _p_sens[2, 0] / _p_sens[0, 0]
            return np.array([[i], [j]])

    def get_H(self, x: np.ndarray) -> np.ndarray:
        """Implements the linearised camera measurement function.

        Args:
            x (np.ndarray): The position estimate obtained from the track state vector.

        Returns:
            np.ndarray: The Jacobian of the non-linear camera measurement model.
        """
        H = np.array(np.zeros((self.dim_meas, params.dim_state)))
        R = self.veh_to_sens[0:3, 0:3]
        T = self.veh_to_sens[0:3, 3]
        if self.name == 'lidar':
            H[0:3, 0:3] = R
        elif self.name == 'camera':
            if R[0, 0] * x[0] + R[0, 1] * x[1] + R[0, 2] * x[2] + T[0] == 0:
                raise ZeroDivisionError(f'Jacobian not defined for this {x.tolist()}!')
            H[0, 0] = self.f_i * (-R[1, 0] / (R[0, 0] * x[0] + R[0, 1] * x[1] + R[0, 2] * x[2] + T[0]) + R[0, 0] * (R[1, 0] * x[0] + R[1, 1] * x[1] + R[1, 2] * x[2] + T[1]) / ((R[0, 0] * x[0] + R[0, 1] * x[1] + R[0, 2] * x[2] + T[0]) ** 2))
            H[1, 0] = self.f_j * (-R[2, 0] / (R[0, 0] * x[0] + R[0, 1] * x[1] + R[0, 2] * x[2] + T[0]) + R[0, 0] * (R[2, 0] * x[0] + R[2, 1] * x[1] + R[2, 2] * x[2] + T[2]) / ((R[0, 0] * x[0] + R[0, 1] * x[1] + R[0, 2] * x[2] + T[0]) ** 2))
            H[0, 1] = self.f_i * (-R[1, 1] / (R[0, 0] * x[0] + R[0, 1] * x[1] + R[0, 2] * x[2] + T[0]) + R[0, 1] * (R[1, 0] * x[0] + R[1, 1] * x[1] + R[1, 2] * x[2] + T[1]) / ((R[0, 0] * x[0] + R[0, 1] * x[1] + R[0, 2] * x[2] + T[0]) ** 2))
            H[1, 1] = self.f_j * (-R[2, 1] / (R[0, 0] * x[0] + R[0, 1] * x[1] + R[0, 2] * x[2] + T[0]) + R[0, 1] * (R[2, 0] * x[0] + R[2, 1] * x[1] + R[2, 2] * x[2] + T[2]) / ((R[0, 0] * x[0] + R[0, 1] * x[1] + R[0, 2] * x[2] + T[0]) ** 2))
            H[0, 2] = self.f_i * (-R[1, 2] / (R[0, 0] * x[0] + R[0, 1] * x[1] + R[0, 2] * x[2] + T[0]) + R[0, 2] * (R[1, 0] * x[0] + R[1, 1] * x[1] + R[1, 2] * x[2] + T[1]) / ((R[0, 0] * x[0] + R[0, 1] * x[1] + R[0, 2] * x[2] + T[0]) ** 2))
            H[1, 2] = self.f_j * (-R[2, 2] / (R[0, 0] * x[0] + R[0, 1] * x[1] + R[0, 2] * x[2] + T[0]) + R[0, 2] * (R[2, 0] * x[0] + R[2, 1] * x[1] + R[2, 2] * x[2] + T[2]) / ((R[0, 0] * x[0] + R[0, 1] * x[1] + R[0, 2] * x[2] + T[0]) ** 2))
        return H

    def generate_measurement(self, num_frame: int, z: np.ndarray, meas_list: List[Measurement]) -> List[Measurement]:
        """Initialises a new Measurement instance and returns it in a list.

        Args:
            num_frame (int): The frame id from which this measurement was captured.
            z (np.ndarray): The raw measurement vector.
            meas_list (List[Measurement]): The list of measurements to update with this object.

        Returns:
            List[Measurement]: The updated measurement list containing the new measurement object created from sensor reading `z`.
        """
        if self.name in {'lidar', 'camera'}:
            meas = Measurement(num_frame, z, self)
            meas_list.append(meas)
            return meas_list
        else:
            raise ValueError(f"Invalid sensor type '{self.name}'")