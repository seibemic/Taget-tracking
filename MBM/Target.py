import math

from LocalHypotheses import LocalHypotheses
from scipy.stats import multivariate_normal as mvn
import numpy as np
from copy import copy, deepcopy
class Target:
    def __init__(self, id, w_k, r_k, m_k, P_k):
        self.origin_id = id
        self.trackers = []
        self.trackers.append(LocalHypotheses(w_k, r_k, m_k, P_k, updatedBy=-1))
    def add(self,w_k, r_k, m_k, P_k):
        self.trackers.append(LocalHypotheses(w_k, r_k, m_k, P_k, updatedBy=-1))
    def predict(self, ps, F, Q):
        for target in self.trackers:
            target.predict(ps, F, Q)

    def applyGating(self,z):
        self.targetsZids = []
        counter = 0
        for i, target in enumerate(self.trackers):
            counter = target.applyGating0(z, counter = counter)
            self.targetsZids.extend(target.Z_indexes)
        for i, target in enumerate(self.trackers):
            counter = target.applyGating1(z, counter = counter)
            self.targetsZids.extend(target.Z_indexes)

        return self.targetsZids

    def updateComponents(self, H, R):
        for target in self.trackers:
            target.updateComponents(H, R)

    def update(self,pd,lambd):
        Jk = len(self.trackers)
        print("num of trackers: ", Jk)
        self.trackersMeasurement_w = []
        self.trackersNoMeasurement_w = []
        self.mapping = []
        tmpTrackers = []
        cnt = 0
        for i in range(Jk):
            # print("i:",i)
            temp_w = []
            temp_map = []
            # tmpTrackers.append(self.trackers[i])
            # print(self.trackers[i].P_apost)
            # print(self.trackers[i].P)
            # print()
            # print(tmpTrackers[i].P)
            # print(tmpTrackers[i].P_apost)
          #  print(tmpTrackers[i].P_apost)
            cnt += 1
            for j, z in enumerate(self.trackers[i].Z_gating): # measuremets for i-th target
                r = 1
                # print("NEW update: ", self.trackers[i].w, " ", self.trackers[i].r, " ", mvn(self.trackers[i].ny, self.trackers[i].S).pdf(z), " ", lambd)
                w = ((self.trackers[i].w) + np.log(self.trackers[i].r * pd)
                     + mvn(self.trackers[i].ny, self.trackers[i].S).logpdf(z) - np.log(lambd))
                # print("     ",w)
                m = self.trackers[i].m + self.trackers[i].K @ (z - self.trackers[i].ny)
                P = self.trackers[i].P
                # tmpTrackers.append(LocalHypotheses(w, r, m, P, updatedBy = self.trackers[i].Z_indexes[j]))
                self.trackers.append(LocalHypotheses(w, r, m, P, updatedBy = self.trackers[i].Z_indexes[j]))# add new targets
                temp_w.append(w)
                temp_map.append(cnt)
                cnt +=1
            # self.trackersMeasurement_w.append(temp_w)
            # self.mapping.append(temp_map)
            # tmpTrackers[i].update(pd)
            print("update no measurement")
            print(self.trackers[i].r)
            self.trackers[i].update(pd)# update no measurement
            self.trackersNoMeasurement_w.append(self.trackers[i].w)

        # self.trackers = tmpTrackers
            # cnt +=1

