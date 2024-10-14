import pandas as pd
import numpy as np
from typing import Optional


class Kalman:
  def __init__(self, 
               A: np.ndarray, C: np.ndarray, 
               Q: np.ndarray, R: np.ndarray, 
               z0: np.ndarray, P0: np.ndarray,
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

  def fit(self, yt, ut):
    self.yt = yt
    self.ut = ut
    assert len(y) == len(u), "y and u must have the same length"
    self.ztt_1, self.zt, self.ztT = (
      np.zeros((self.state_dim, len(yt)+1)),  # Predicted state estimate
      np.zeros((self.state_dim, len(yt)+1)),  # Corrected state estimate
      np.zeros((self.state_dim, len(yt)+1)),  # Smoothed state estimate
    )
    self.Ptt_1, self.Pt, self.PtT = (
      np.zeros((self.state_dim, self.state_dim, len(yt)+1)),  # Predicted state covariance
      np.zeros((self.state_dim, self.state_dim, len(yt)+1)),  # Corrected state covariance
      np.zeros((self.state_dim, self.state_dim, len(yt)+1)),  # Smoothed state covariance
    )
    self.ztt_1[:, 0], self.Ptt_1[:, :, 0] = self.z0, self.P0
    self.zt[:, 0], self.Pt[:, :, 0] = self.z0, self.P0
    self.Kt = np.zeros((self.state_dim, self.meas_dim, len(yt)))  # Kalman gain
    self.St = np.zeros((self.meas_dim, self.meas_dim, len(yt)))  # Innovation covariance
    self.Jt = np.zeros((self.state_dim, self.state_dim, len(yt)))  # Smoothing gain (backward Kalman gain)
    pass

  def _predict(self):
    for i in range(1, len(self.yt)+1):
      self.ztt_1[:, i] = self.A @ self.ztt_1[:, i-1] + self.B @ self.ut[i-1]
      self.Ptt_1[:, :, i] = self.A @ self.Ptt_1[:, :, i-1] @ self.A.T + self.Q

  def _update(self):
    for i in range(1, len(self.yt)+1):
      self.St[:, :, i-1] = self.C @ self.Ptt_1[:, :, i] @ self.C.T + self.R
      self.Kt[:, :, i-1] = self.Ptt_1[:, :, i] @ self.C.T @ np.linalg.inv(self.St[:, :, i-1])
      yhat_t = self.C @ self.ztt_1[:, i]
      self.zt[:, i] = self.ztt_1[:, i] + self.Kt[:, :, i-1] @ (self.yt[i-1] - yhat_t)
      self.Pt[:, :, i] = self.Ptt_1[:, :, i] - self.Kt[:, :, i-1] @ self.C @ self.Ptt_1[:, :, i]
  
  def filter(self):
    self._predict()
    self._update()
    return self.zt, self.Pt
  
  def smooth(self):
    for i in range(len(self.yt)-1, -1, -1):
      self.JtT[:, :, i] = self.Pt[:, :, i] @ self.A.T @ np.linalg.inv(self.Ptt_1[:, :, i+1])
      self.ztT[:, i] = self.zt[:, i] + self.JtT[:, :, i] @ (self.ztT[:, i+1] - self.ztt_1[:, i+1])
      self.PtT[:, :, i] = self.Pt[:, :, i] + self.JtT[:, :, i] @ (self.PtT[:, :, i+1] - self.Ptt_1[:, :, i+1]) @ self.JtT[:, :, i].T
    


