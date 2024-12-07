# ------------------------------------------------------------------------------
# Project "Multi-Target Tracking with Extended Kalman Filters and Sensor Fusion"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file: Define the `Track` and `TrackManagement` classes
#                       and their core functionality, i.e., the track state,
#                       track id, covariance matrices, and functions for
#                       track initialisation / deletion.
#
# You should have received a copy of the Udacity license with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
#
# NOTE: The current version of this programme relies on Numpy to perform data
#       manipulation, however, a platform-specific implementation, e.g.,
#       TensorFlow `tf.Tensor` data ops, is recommended.
# ------------------------------------------------------------------------------

# General package imports
import numpy as np
import os
import sys
from typing import List

# Add project directory to PYTHONPATH to enable relative imports
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# Tracking package imports
import misc.params as params
from student.measurements import Measurement


class Track(object):
    """The Track class.

    Args:
        meas (Measurement): The LiDAR measurement to associate with this track.
        _id (int): The unique id to assign this track.
    """

    def __init__(self, meas: Measurement, _id: int):
        """Initialises a new Track instance."""
        print('creating track no.', _id)
        M_rot = meas.sensor.sens_to_veh[0:3, 0:3]
        self.x = np.ones((6, 1))
        _z_sens = meas.z
        _z_sens = np.vstack((_z_sens, np.newaxis))
        _z_sens[3] = 1
        _T_sens2veh = meas.sensor.sens_to_veh
        self.x[0:4] = _T_sens2veh @ _z_sens
        self.P = np.zeros((6, 6))
        _M_rot = meas.sensor.sens_to_veh[0:3, 0:3]
        _R_sens = meas.R
        self.P[0:3, 0:3] = np.matmul(_M_rot @ _R_sens, _M_rot.T)
        self.P[3:6, 3:6] = np.diag([params.sigma_p44 ** 2, params.sigma_p55 ** 2, params.sigma_p66 ** 2])
        self.state = 'initialized'
        self.score = 1. / params.window
        self.id = _id
        self.width = meas.width
        self.length = meas.length
        self.height = meas.height
        self.yaw = np.arccos(M_rot[0, 0] * np.cos(meas.yaw) + M_rot[0, 1] * np.sin(meas.yaw))
        self.t = meas.t

    def set_x(self, x: np.ndarray):
        """Sets the state vector estimate to the given object."""
        self.x = x

    def set_P(self, P: np.ndarray):
        """Sets the estimation error covariance to the given object."""
        self.P = P

    def set_t(self, t: np.ndarray):
        """Sets the translation vector instance to the given object."""
        self.t = t

    def update_attributes(self, meas: Measurement):
        """Updates the track with the latest measurement.

        Args:
            meas (Measurement): The Measurement instance to update the track state with.
        """
        if meas.sensor.name == 'lidar':
            c = params.weight_dim
            self.width = c * meas.width + (1 - c) * self.width
            self.length = c * meas.length + (1 - c) * self.length
            self.height = c * meas.height + (1 - c) * self.height
            M_rot = meas.sensor.sens_to_veh
            self.yaw = np.arccos(M_rot[0, 0] * np.cos(meas.yaw) + M_rot[0, 1] * np.sin(meas.yaw))


class TrackManagement(object):
    """The Track Management class."""

    def __init__(self):
        """Initialises a new TrackManagement instance."""
        self.N = 0
        self.track_list = []
        self.result_list = []
        self.last_id = -1

    def manage_tracks(self, unassigned_tracks: List[Track], unassigned_meas: List[Measurement],
                      meas_list: List[Measurement]):
        """Runs the track management loop.

        Args:
            unassigned_tracks (List[Track]): The list of tracks that have not yet been assigned to a measurement.
            unassigned_meas (List[Measurement]): The list of measurements that have not yet been associated with a track.
            meas_list (List[Measurement]): The list of measurements from this current time-step.
        """
        tracks_to_delete = []
        for i in unassigned_tracks:
            track = self.track_list[i]
            if meas_list:
                if meas_list[0].sensor.in_fov(track.x):
                    track.score -= 1. / params.window
                    track.score = max(track.score, 0.)
                    if track.state == 'confirmed':
                        threshold = params.delete_threshold
                    elif track.state in {'initialized', 'tentative'}:
                        threshold = params.delete_init_threshold
                    else:
                        raise ValueError(f"Invalid track state '{track.state}'")
                    if track.score < threshold or track.P[0,0] > params.max_P or track.P[1,1] > params.max_P:
                        tracks_to_delete.append(track)
                else:
                    pass
            else:
                pass
        for track in tracks_to_delete:
            self.delete_track(track)
        for j in unassigned_meas:
            if meas_list[j].sensor.name == 'lidar':
                self.init_track(meas_list[j])

    def add_track_to_list(self, track: Track):
        """Adds the given track to the track manager.

        Args:
            track (Track): The new track to add to the track list.
        """
        self.track_list.append(track)
        self.N += 1
        self.last_id = track.id

    def delete_track(self, track: Track):
        """Removes the given track from the track list.

        Args:
            track (Track): The track instance to remove from the track list.
        """
        if track in self.track_list:
            print('deleting track no.', track.id)
            self.track_list.remove(track)

    def handle_updated_track(self, track: Track):
        """Updates the given track's score and state.

        Args:
            track (Track): The specific track to update.
        """
        _new_score = track.score + 1. / params.window
        track.score = min(_new_score, 1.0)
        if track.state in {'initialized', 'initialised'}:
            track.state = 'tentative'
        elif track.state == 'tentative':
            if track.score > params.confirmed_threshold:
                track.state = 'confirmed'
        elif track.state == 'confirmed':
            pass
        else:
            raise ValueError(f"Invalid track state '{track.state}'")

    def init_track(self, meas: Measurement):
        """Initialises a new track instance and adds it to the track list.

        Args:
            meas (Measurement): The LiDAR measurement to assign to a new track instance.
        """
        track = Track(meas, self.last_id + 1)
        self.add_track_to_list(track)
