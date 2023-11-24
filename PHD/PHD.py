import numpy as np


class PHD:
    def __init__(self, w, m, P):
        self.w = w
        self.m = m
        self.P = P
        # self.P_aposterior=self.P_aprior


    def predict(self, ps, A, Q):
        self.w = ps * self.w
        self.m = A @ self.m
        self.P = Q + A @ self.P @ A.T

    def updateComponents(self, H, R):
        self.ny = H @ self.m
        self.S = R + H @ self.P @ H.T
        self.K = self.P @ H.T @ np.linalg.inv(self.S)
        self.P_apost = self.P.copy()
        self.P = (np.eye(len(self.K)) - self.K @ H) @ self.P

    def update(self, pd):
        self.w = (1 - pd) * self.w
        self.m = self.m
        self.P = self.P_apost
        # self.P_aposterior = self.P_aprior


