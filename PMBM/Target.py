import math

from LocalHypotheses import LocalHypotheses
from scipy.stats import multivariate_normal as mvn
import numpy as np
from copy import copy, deepcopy


class BernoulliTarget:
    def __init__(self, w_k, r_k, m_k, P_k):
        self.trackers = []
        self.trackers.append(LocalHypotheses(w_k, r_k, m_k, P_k, updatedBy=-1))

    def add(self, w_k, r_k, m_k, P_k):
        self.trackers.append(LocalHypotheses(w_k, r_k, m_k, P_k, updatedBy=-1))

    def predict(self, ps, F, Q):
        for target in self.trackers:
            target.predict(ps, F, Q)

    def applyGating(self, z):
        self.targetsZids = []
        counter = 0
        for i, target in enumerate(self.trackers):
            counter = target.applyGating0(z, counter=counter)
            self.targetsZids.extend(target.Z_indexes)
        for i, target in enumerate(self.trackers):
            counter = target.applyGating1(z, counter=counter)
            self.targetsZids.extend(target.Z_indexes)

        return self.targetsZids

    def updateComponents(self, H, R):
        for target in self.trackers:
            target.updateComponents(H, R)

    def update(self, pd, lambd):
        cnt = 0
        Jk = len(self.trackers)
        for i in range(Jk):

            cnt += 1
            for j, z in enumerate(self.trackers[i].Z_gating):  # measuremets for i-th target
                r = 1
                # print("NEW update: ", self.trackers[i].w, " ", self.trackers[i].r, " ", mvn(self.trackers[i].ny, self.trackers[i].S).pdf(z), " ", lambd)
                w = ((self.trackers[i].w) + np.log(self.trackers[i].r * pd)
                     + mvn(self.trackers[i].ny, self.trackers[i].S).logpdf(z) - np.log(lambd))
                # print("     ",w)
                m = self.trackers[i].m + self.trackers[i].K @ (z - self.trackers[i].ny)
                P = self.trackers[i].P
                # tmpTrackers.append(LocalHypotheses(w, r, m, P, updatedBy = self.trackers[i].Z_indexes[j]))
                self.trackers.append(
                    LocalHypotheses(w, r, m, P, updatedBy=self.trackers[i].Z_indexes[j]))  # add new targets
                cnt += 1

            self.trackers[i].update(pd)  # update no measurement


from scipy.stats import chi2


class PoissonTarget:
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

    def applyGating(self, z, pd, lambd, H):
        self.targetsZids = []
        counter = 0

        Pg = 0.95
        gamma = chi2.ppf(Pg, df=2)
        covInv = np.linalg.inv(self.S)
        newBernoulli = False
        all_w = 0
        Btarget = None
        for i, z_ in enumerate(z):
            if (z_ - self.ny).T @ covInv @ (z_ - self.ny) <= gamma:
                e = pd * self.w * mvn.pdf(z_, self.ny, self.S)
                ro = e + lambd
                r = e / ro
                w = self.w * mvn.pdf(z_, self.ny, self.S)
                all_w += w
                m = self.m + self.P @ H.T @ covInv @ (z_ - H @ self.m)
                P = self.P - self.P @ H.T @ covInv @ (self.P @ H.T).T
                if not newBernoulli:
                    Btarget = BernoulliTarget(w, r, m, P)
                    self.targetsZids.append([i, counter])
                    counter +=1
                    newBernoulli = True
                else:
                    Btarget.add(w,r,m,P)
                    self.targetsZids.append([i, counter])
                    counter += 1
        if Btarget:
            for i in range(len(Btarget.trackers)):
                Btarget.trackers[i].w /= all_w
            return Btarget, self.targetsZids
        return None, None



    def update(self, pd):
        self.w = (1 - pd) * self.w
