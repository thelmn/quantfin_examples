import pandas as pd
import numpy as np
from typing import Optional, Tuple


class Kalman:
  """
  Simple Kalman filter implementation for linear Gaussian state space models.
  The model is defined by the following equations:
  z_t = A z_{t-1} + B u_t + w_t, w_t ~ N(0, Q)
  y_t = C z_t + D u_t + v_t, v_t ~ N(0, R)
  Only implement inference (filtering and smoothing), 
  parameter estimation (EM algorithm) is too complex :)
  """
  def __init__(self, 
               A: np.ndarray, C: np.ndarray, 
               Q: np.ndarray, R: np.ndarray, 
               z0: np.ndarray, P0: np.ndarray,
               yt: np.ndarray, 
               ut: Optional[np.ndarray] = None,
               B: Optional[np.ndarray] = None, D: Optional[np.ndarray] = None
               ):
    self.A = A # State transition matrix
    self.C = C # Measurement matrix
    self.state_dim = A.shape[0]
    self.meas_dim = C.shape[0]
    self.B = B if B is not None else np.zeros((self.state_dim, 0))  # Control matrix
    self.D = D if D is not None else np.zeros((self.meas_dim, 0))  # Control matrix
    self.Q = Q
    self.R = R
    self.z0 = z0
    self.P0 = P0
    self.control_dim = B.shape[1] if B is not None else 0
    assert A.shape[1] == self.state_dim, "A must be square"
    assert C.shape[1] == self.state_dim, "C must have the same number of columns as A"
    if B is not None:
      assert B.shape[0] == self.state_dim, "B must have the same number of rows as A"
    if D is not None:
      assert D.shape[0] == self.meas_dim, "D must have the same number of rows as C"
      assert D.shape[1] == self.control_dim, "D must have the same number of columns as B"
    assert Q.shape[0] == Q.shape[1] == self.state_dim, "Q must be square and have the same number of rows as A"
    assert R.shape[0] == R.shape[1] == self.meas_dim, "R must be square and have the same number of rows as C"
    assert z0.shape[0] == self.state_dim, "z0 must have the same number of rows as A"
    assert P0.shape[0] == P0.shape[1] == self.state_dim, "P0 must be square and have the same number of rows as A"

    self.yt = yt
    self.T = yt.shape[1]
    self.ut = ut if ut is not None else np.zeros((self.control_dim, self.T))
    self.zt, ztT = None, None  # filtered and smoothed state estimates, shape (state_dim, len(yt)+1)
    self.Pt, PtT = None, None  # filtered and smoothed state covariance, shape (state_dim, state_dim, len(yt)+1)

  def _predict(self):
    # initialize state and covariance matrices
    self.zt = np.zeros((self.state_dim, self.T+1))
    self.Pt = np.zeros((self.state_dim, self.state_dim, self.T+1))
    # initialize kalman gain matrix
    self.Kt = np.zeros((self.state_dim, self.meas_dim, self.T))
    self.zt[:, 0] = self.z0
    self.Pt[:, :, 0] = self.P0
    # predict step
    for t in range(1, self.T+1):
      self.zt[:, t] = self.A @ self.zt[:, t-1] + self.B @ self.ut[:, t-1]
      self.Pt[:, :, t] = self.A @ self.Pt[:, :, t-1] @ self.A.T + self.Q

  def _update(self):
    # correction step
    for t in range(1, self.T+1):
      St = self.C @ self.Pt[:, :, t] @ self.C.T + self.R
      self.Kt[:, :, t-1] = self.Pt[:, :, t] @ self.C.T @ np.linalg.inv(St)
      self.zt[:, t] = self.zt[:, t] + self.Kt[:, :, t-1] @ (self.yt[:, t-1] - self.C @ self.zt[:, t])
      self.Pt[:, :, t] = (np.eye(self.state_dim) - self.Kt[:, :, t-1] @ self.C) @ self.Pt[:, :, t]
  
  def filter(self):
    self._predict()
    self._update()
    return self.zt, self.Pt
  
  def smooth(self):
    # initialize smoothed state and covariance matrices
    self.ztT = np.zeros((self.state_dim, self.T+1))
    self.PtT = np.zeros((self.state_dim, self.state_dim, self.T+1))
    # P_{t,t-1|T} cross-covariance of z_t and z_{t-1} given all observations
    self.Ptt1_T = np.zeros((self.state_dim, self.state_dim, self.T))  
    # initialize last time step for state and covariance matrices
    self.ztT[:, -1] = self.zt[:, -1]
    self.PtT[:, :, -1] = self.Pt[:, :, -1]
    # initialize last time step for cross-covariance matrix
    self.Ptt1_T[:, :, -1] = (np.eye(self.state_dim) - self.Kt[:, :, -1] @ self.C) @ self.A @ self.Pt[:, :, -2]
    # smoothing step (and compute cross-covariance matrix)
    z_t_tn1, P_t_tn1, J_tn1 = None, None, None
    for t in range(self.T-1, -1, -1):
      if z_t_tn1 is not None and P_t_tn1 is not None and J_tn1 is not None:
        z_tp1_t, P_tp1_t, Jt = z_t_tn1, P_t_tn1, J_tn1
      else:
        z_tp1_t = self.A @ self.zt[:, t]
        P_tp1_t = self.A @ self.Pt[:, :, t] @ self.A.T + self.Q
        Jt = self.Pt[:, :, t] @ self.A.T @ np.linalg.inv(P_tp1_t)
      self.ztT[:, t] = self.zt[:, t] + Jt @ (self.ztT[:, t+1] - z_tp1_t)
      self.PtT[:, :, t] = self.Pt[:, :, t] + Jt @ (self.PtT[:, :, t+1] - P_tp1_t) @ Jt.T

      if t > 0:
        z_t_tn1 = self.A @ self.zt[:, t-1]
        P_t_tn1 = self.A @ self.Pt[:, :, t-1] @ self.A.T + self.Q
        J_tn1 = self.Pt[:, :, t-1] @ self.A.T @ np.linalg.inv(P_t_tn1)
        self.Ptt1_T[:, :, t-1] = self.Pt[:, :, t] @ J_tn1.T + Jt @ (self.Ptt1_T[:, :, t] - self.A @ self.Pt[:, :, t]) @ J_tn1.T
    return self.ztT, self.PtT, self.Ptt1_T
    


