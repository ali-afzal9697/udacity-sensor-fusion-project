# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file: Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ---------------------------------------------------------------------

# General package imports
import numpy as np
import os
import sys

# Add project directory to PYTHONPATH to enable relative imports
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# Tracking package imports
from student.measurements import Measurement
import misc.params as params
from student.track_management import Track

class Filter(object):
    """The Kalman filter class.

    Implements the Kalman filter in 3-D using state-space form.

    Args:
        dim_state (int): The dimensions of the process model `P`.
        dt (float): The discrete time-step, i.e., $\Delta{t}$, which is assumed to be fixed in this implementation.
        q (float): The design parameter of the covariance process noise matrix $Q$, which is selected w.r.t. the expected maximum change in velocity.
    """

    def __init__(self):
        """Initialises the Kalman filter object with attributes."""
        self.dim_state = params.dim_state
        self.dt = params.dt
        self.q = params.q

    def F(self) -> np.ndarray:
        """Implements the state transition function as a system matrix `F`.

        Returns:
            np.ndarray: The state transition matrix.
        """
        return np.array([[1., 0., 0., self.dt, 0., 0.],
                         [0., 1., 0., 0., self.dt, 0.],
                         [0., 0., 1., 0., 0., self.dt],
                         [0., 0., 0., 1., 0., 0.],
                         [0., 0., 0., 0., 1., 0.],
                         [0., 0., 0., 0., 0., 1.]])

    def Q(self) -> np.ndarray:
        """Implements the process noise covariance matrix.

        Returns:
            np.ndarray: The process noise covariance matrix.
        """
        _F = self.F()
        _Q = np.diag([0., 0., 0., self.q, self.q, self.q])
        _integral_factor = np.array([[self.dt / 3, 0., 0., self.dt / 2, 0., 0.],
                                     [0., self.dt / 3., 0., 0., self.dt / 2, 0.],
                                     [0., 0., self.dt / 3, 0., 0., self.dt / 2],
                                     [self.dt / 2, 0., 0., self.dt, 0., 0.],
                                     [0., self.dt / 2, 0., 0., self.dt, 0.],
                                     [0., 0., self.dt / 2, 0., 0., self.dt]])
        QT = _integral_factor * np.matmul(_F @ _Q, _F.T)
        return QT.T

    def predict(self, track: Track):
        """Implements the prediction step.

        Args:
            track (Track): The Track instance containing the state estimation and the covariance matrix from the previous time-step.
        """
        _F = self.F()
        _Q = self.Q()
        _x = _F @ track.x
        _P = np.matmul(_F @ track.P, _F.T) + _Q
        track.set_x(_x)
        track.set_P(_P)

    def update(self, track: Track, meas: Measurement):
        """Implements the update step.

        Args:
            track (Track): The Track instance containing the state estimate and the process noise covariance matrix updated from the prediction step.
            meas (Measurement): The Measurement instance containing the measurement vector and the measurement noise covariance.
        """
        _gamma = self.gamma(track, meas)
        _H = meas.sensor.get_H(track.x)
        _S = self.S(track, meas, _H)
        _K = np.matmul(track.P @ _H.T, np.linalg.inv(_S))
        _x = track.x + _K @ _gamma
        track.set_x(_x)
        _I = np.identity(n=self.dim_state)
        _P = (_I - np.matmul(_K, _H)) @ track.P
        track.set_P(_P)
        track.update_attributes(meas)

    def gamma(self, track: Track, meas: Measurement) -> np.ndarray:
        """Helper function to compute and return the residual $\gamma$.

        Args:
            track (Track): The Track instance containing the state estimate and the process noise covariance matrix updated from the prediction step.
            meas (Measurement): The Measurement instance containing the measurement vector and the measurement noise covariance.

        Returns:
            np.ndarray: The measurement residual update.
        """
        return meas.z - meas.sensor.get_hx(track.x)

    def S(self, track: Track, meas: Measurement, H: np.ndarray) -> np.ndarray:
        """Helper function to compute and return the residual covariance $\mathrm{S}$.

        Args:
            track (Track): The Track instance containing the estimation error covariance.
            meas (Measurement): The Measurement instance containing the measurement noise covariance.
            H (np.ndarray): The Jacobian of the measurement model.

        Returns:
            np.ndarray: The estimation error covariance of the residual.
        """
        return np.matmul(H @ track.P, H.T) + meas.R