# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file: Data association class with single nearest neighbor association and gating based on Mahalanobis distance
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------

# Imports
import numpy as np
from scipy.stats.distributions import chi2
from typing import List, Tuple

# Add project directory to PYTHONPATH to enable relative imports
import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import misc.params as params
from student.filter import Filter
from student.measurements import Measurement, Sensor
from student.track_management import Track, TrackManagement

class Association(object):
    """The Association class.

    Implements the Single Nearest Neighbor (SNN) association algorithm with validation gating.

    Args:
        association_matrix (np.ndarray): The data association matrix used to assign uncertain measurements to known tracks based on Mahalanobis distance.
        unassigned_tracks (List[int]): The list of known tracks that have not been assigned to a measurement.
        unassigned_meas (List[int]): The list of uncertain measurements that have not been assigned to a track.
    """

    def __init__(self):
        """Initialises the Association instance and its attributes."""
        self.association_matrix = np.array([])
        self.unassigned_tracks = []
        self.unassigned_meas = []

    def associate(self, track_list: List[Track], measurement_list: List[Measurement], kalman_filter: Filter):
        """Performs the data association step.

        Args:
            track_list (List[Track]): The list of current known assigned tracks.
            measurement_list (List[Measurement]): The list of current, already associated measurements.
            kalman_filter (Filter): The Kalman Filter instance containing the residual covariance function to compute matrix S.
        """
        self.unassigned_tracks = list(range(len(track_list)))
        self.unassigned_meas = list(range(len(measurement_list)))
        self.association_matrix = np.array(np.full((len(track_list), len(measurement_list)), fill_value=np.inf))
        for x_i, track in enumerate(track_list):
            for z_j, measurement in enumerate(measurement_list):
                dist = self.calc_mahalanobis_distance(track, measurement, kalman_filter)
                if self.gating(dist, measurement.sensor):
                    self.association_matrix[x_i, z_j] = dist

    def get_closest_track_and_meas(self) -> Tuple[int, int]:
        """Returns the closest-distance track-measurement pair.

        Returns:
            Tuple[int, int]: The indices of the closest track and measurement instances from the unassigned lists.
        """
        if np.min(self.association_matrix) == np.inf:
            return np.nan, np.nan
        idx_track, idx_measurement = np.unravel_index(np.argmin(self.association_matrix, axis=None), self.association_matrix.shape)
        track_closest = self.unassigned_tracks[idx_track]
        measurement_closest = self.unassigned_meas[idx_measurement]
        self.unassigned_tracks.remove(track_closest)
        self.unassigned_meas.remove(measurement_closest)
        self.association_matrix = np.delete(self.association_matrix, obj=idx_track, axis=0)
        self.association_matrix = np.delete(self.association_matrix, obj=idx_measurement, axis=1)
        return track_closest, measurement_closest

    def gating(self, dist_mh: float, sensor: Sensor) -> bool:
        """Checks if the measurement is inside the gating region of the track.

        Args:
            dist_mh (float): The Mahalanobis distance between track and measurement.
            sensor (Sensor): The Sensor instance containing the degrees of freedom of the measurement space.

        Returns:
            bool: Whether the measurement is within the gating region with high certainty.
        """
        ppf = chi2.ppf(q=params.gating_threshold, df=sensor.dim_meas)
        return dist_mh < ppf

    def calc_mahalanobis_distance(self, track: Track, meas: Measurement, kalman_filter: Filter) -> float:
        """Implements and returns the Mahalanobis distance calculation.

        Args:
            track (Track): The Track instance with known estimation error covariance.
            meas (Measurement): The Measurement instance with uncertain position estimate and corresponding measurement error covariance.
            kalman_filter (Filter): The Kalman Filter instance containing the residual covariance function used to compute S.

        Returns:
            float: The Mahalanobis distance measure between the given track and the measurement.
        """
        _H = meas.sensor.get_H(track.x)
        gamma = meas.z - meas.sensor.get_hx(track.x)
        _S = kalman_filter.S(track, meas, _H)
        dist = np.matmul(gamma.T @ np.linalg.inv(_S), gamma)
        return dist

    def associate_and_update(self, manager: TrackManagement, measurement_list: List[Measurement], kalman_filter: Filter):
        """Performs the association and update step.

        Args:
            manager (TrackManagement): The TrackManagement instance monitoring the current track_list and managing the track state/score updates.
            measurement_list (List[Measurement]): The list of measurements from this current time-step.
            kalman_filter (Filter): The Filter instance used to perform the innovation step.
        """
        self.associate(manager.track_list, measurement_list, kalman_filter)
        while self.association_matrix.shape[0] > 0 and self.association_matrix.shape[1] > 0:
            idx_track, idx_measurement = self.get_closest_track_and_meas()
            if np.isnan(idx_track):
                print('---no more associations---')
                break
            track = manager.track_list[idx_track]
            if not measurement_list[0].sensor.in_fov(track.x):
                continue
            print(f"Update track {track.id} with {measurement_list[idx_measurement].sensor.name}, measurement {idx_measurement}")
            kalman_filter.update(track, measurement_list[idx_measurement])
            manager.handle_updated_track(track)
            manager.track_list[idx_track] = track
        manager.manage_tracks(unassigned_tracks=self.unassigned_tracks, unassigned_meas=self.unassigned_meas, meas_list=measurement_list)
        for track in manager.track_list:
            print('track', track.id, 'score =', track.score)