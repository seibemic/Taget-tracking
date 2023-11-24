import numpy as np
from scipy.stats import chi2

class LocalHypotheses:
    def __init__(self, w, r, m, P, updatedBy):
        self.r = r
        self.w = w
        self.m = m
        self.P = P
        self.updatedBy = updatedBy
        # self.P_aposterior=self.P_aprior


    def predict(self, ps, A, Q):
        self.w = self.w
        self.r = ps * self.r
        self.m = A @ self.m
        self.P = Q + A @ self.P @ A.T

    def updateComponents(self, H, R):
        self.ny = H @ self.m
        self.S = R + H @ self.P @ H.T
        self.K = self.P @ H.T @ np.linalg.inv(self.S)
        self.P_apost = self.P.copy()
        self.P = (np.eye(len(self.K)) - self.K @ H) @ self.P

    def update(self, pd):
        # print("update: ", self.w, " ", self.r, " ", pd)
        self.w = self.w * (1-self.r + self.r * (1 - pd))
        # print("     ", self.w)
        self.r = (self.r*(1-pd)) / (1-self.r + self.r*(1-pd))
        self.m = self.m
        self.P = self.P_apost
        # self.P_aposterior = self.P_aprior

    def applyGating(self, z, counter, Pg=0.99):
        self.Z_indexes = []
        self.Z_gating = []
        covInv = np.linalg.inv(self.S)
        gamma = chi2.ppf(Pg, df=2)
        for i, z_ in enumerate(z):
            # print((z_ - self.ny).T @ covInv @ (z_ - self.ny), " ", gamma)
            if ((z_ - self.ny).T @ covInv @ (z_ - self.ny)) <= gamma:
                self.Z_gating.append(z_)
                self.Z_indexes.append([i, counter])
                counter +=1
        self.Z_indexes.append([-1, counter])
        counter +=1
        return counter