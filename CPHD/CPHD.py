import numpy as np
class CPHD:
    def __init__(self,w,m,P):
        self.w = w
        self.m = m
        self.P = P

    def predict(self, ps, A, Q):
        self.w = ps * self.w
        self.m = A @ self.m
        self.P = Q + A @ self.P @ A.T

    def updateComponents(self, H, R):
        self.eta = H @ self.m
        self.S = R + H @ self.P @ H.T
        self.K = self.P @ H.T @ np.linalg.inv(self.S)
        self.P_apost = self.P.copy()
        self.P = (np.eye(len(self.K)) - self.K @ H) @ self.P


    def update(self, pd, psi):
        self.w = (1 - pd) * self.w * psi
        self.m = self.m
        self.P = self.P_apost