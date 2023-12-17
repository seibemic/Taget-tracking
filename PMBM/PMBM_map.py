import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal as mvn
from confidence_ellipse import confidence_ellipse
from MAP2 import Radar
import itertools
from Target import BernoulliTarget, PoissonTarget
import math
from scipy.special import logsumexp


class PMBM_map:
    def __init__(self, radar, ps, pd):
        self.U = 4
        self.radarMap = radar
        self.F = radar.A
        self.H = radar.H
        self.Q = radar.Q
        self.R = radar.R
        self.ps = ps
        self.pd = pd
        self.lambd = radar.lambd
        self.targets = []
        self.Ptargets = []
        self.pmbmsToPlot = []
        self.measurements = radar.getAllMeasurementsWithinRadarRadius()
        self.globalHyp_mat = [[]]
        self.globalHyp_w = []

        self.model_colors = ["saddlebrown", "black", "magenta", "slategray"]

    def predictionForBirthTargets(self):

        for airport in self.radarMap.getAirports():
            # w_k = 1
            # r_k = airport.weight
            # m_k = np.array(airport.pos)
            # P_k = airport.cov
            # self.targets.append(BernoulliTarget(w_k, r_k, m_k, P_k))
            w = airport.weight
            m = np.array(airport.pos)
            P = airport.cov
            self.Ptargets.append(PoissonTarget(w, m, P))

    def predictionForExistingTargets(self):
        for target in self.targets:
            target.predict(self.ps, self.F, self.Q)
        for targer in self.Ptargets:
            targer.predict(self.ps, self.F, self.Q)
        # for i in range(len(self.phds)):
        #     self.phds[i].predict(self.ps, self.F, self.Q)

    def getMeasurements(self, time):
        print(self.measurements[time])
        return self.measurements[time]

    def applyGating(self, z, t):
        self.Zids = []
        for i, target in enumerate(self.targets):
            self.Zids.append(target.applyGating(z))
        print("Z_ids after Bernoulli")
        print(self.Zids)
        self.newTargetsCnt = 0
        PtargetsToDelete = []
        newTargets = []
        newZids = []
        for i, target in enumerate(self.Ptargets):
            Btarget, Zids = target.applyGating(z, self.pd, self.lambd, self.H)
            if Btarget:
                PtargetsToDelete.append(i)
                self.newTargetsCnt += 1
                newTargets.append(Btarget)
                newZids.append(Zids)
                self.targets.append(Btarget)
                self.Zids.append(Zids)
        print("New Zids:")
        print(newZids)
        print("Z_ids after Poisson")
        print(self.Zids)
        for index in sorted(PtargetsToDelete, reverse=True):
            del self.Ptargets[index]

    def getHypothesis(self):
        print("Z_ids")
        print(self.Zids)
        combinations = list(itertools.product(*self.Zids))
        target_id = []
        z_id = []
        for c in combinations:
            x = [element[0] for element in c]
            if all(x.count(element) <= 1 or element == -1 for element in x):
                target_id.append([element[1] for element in c])
                z_id.append([element[0] for element in c])
        # print("num of combinations after:", len(target_id))
        return np.array(target_id), np.array(z_id)

    def makeGlobalMatWeights(self):
        weights = np.zeros_like(np.array(self.globalHyp_mat, dtype="float32").T)
        print("weights:")
        print(weights)
        for i, target in enumerate(np.array(self.globalHyp_mat).T):
            for j, target_id in enumerate(target):
                weights[i][j] = self.targets[i].trackers[target_id].w
        self.globalHyp_w = np.zeros_like(weights[0])

        for i, w in enumerate(weights.T):
            self.globalHyp_w[i] = np.sum(w)
        self.globalHyp_w /= logsumexp(self.globalHyp_w)

    def makeCostMatrix(self):
        ## ADD measurements outside gating
        self.costMatrix = []
        for i, target in enumerate(self.targets):

            for j, allW in enumerate(target.trackersMeasurement_w):
                C_row = []
                for w in allW:
                    C_row.append(-math.log(w / target.trackersNoMeasurement_w[j]))
                self.costMatrix.append(C_row)

        print("Cost mat")
        print(self.costMatrix)

    def selectKBestHypothese(self, K):
        K_best = np.argsort(-self.globalHyp_w)[:min(K, len(self.globalHyp_w))]
        K_bestHypothesis = []
        for i, k in enumerate(K_best):
            K_bestHypothesis.append(self.globalHyp_mat[k])
        self.K_bestHypothesis = np.array(K_bestHypothesis)
        print("K best hyp: ")
        print(self.K_bestHypothesis)
        self.bestHypothesis = self.K_bestHypothesis[0]
        print("best hyp:")
        print(self.bestHypothesis)

        self.bestTrackers = []
        for i, tracker_id in enumerate(self.bestHypothesis):
            self.bestTrackers.append(self.targets[i].trackers[tracker_id])

    def pruneByExistenceProbabiltyAndWeights(self):
        # prune by r
        new_z = []
        for j, target in enumerate(self.targets):
            indexes = []
            for i, hyp in enumerate(target.trackers):
                if hyp.r < 1e-4:# or hyp.w < 1e-4:
                    indexes.append(i)
            # lowerEl = len([x for x in indexes if x < self.bestHypothesis[j]])
            # self.bestHypothesis[j] -= lowerEl
            for index in sorted(indexes, reverse=True):
                del target.trackers[index]

            temp_z = []
            for i, local_hyp in enumerate(self.Zids[j]):
                if local_hyp[1] not in indexes:
                    temp_z.append(self.Zids[j][i])
            if len(temp_z) != 0:
                temp_z = np.array(temp_z)
                temp_z.T[1] = np.arange(0, len(temp_z.T[1]), 1)
                temp_z = temp_z.tolist()
                new_z.append(temp_z)
        self.Zids = new_z
        # delete targets with no localHypothese
        indexes = []
        for i, target in enumerate(self.targets):
            if len(target.trackers) == 0:
                indexes.append(i)
                # del self.bestHypothesis[i]
        for index in sorted(indexes, reverse=True):
            del self.targets[index]


    def pruneTargets(self):
        for i, target in enumerate(self.targets):
            indexes = set((self.K_bestHypothesis.T)[i])
            #deleted = set((self.globalHyp_mat.T)[i]) - indexes
            deleted = set(range(len(self.targets[i].trackers))) - indexes
            print("glob hyp : ", i)
            print(set((self.globalHyp_mat.T)[i]))
            print("deleted:")
            print(deleted)
            lowerEl = len([x for x in deleted if x < self.bestHypothesis[i]])
            print('lower el: ', i, " ", lowerEl)
            self.bestHypothesis[i] -= lowerEl
            target.trackers = [e for j, e in enumerate(target.trackers) if j in indexes]
        lens = []
        for i, target in enumerate(self.targets):
            lens.append(len(target.trackers))

        # delete targets with no localHypothese
        indexes = []
        for i, target in enumerate(self.targets):
            if len(target.trackers) == 0:
                indexes.append(i)
                del self.bestHypothesis[i]
        for index in sorted(indexes, reverse=True):
            del self.targets[index]


    def showGatingMeasurements(self):
        for i, target in enumerate(self.targets):
            print("target ", i, ":")
            print(target.Z_indexes)

    def updateComponents(self):
        for target in self.targets:
            target.updateComponents(self.H, self.R)
        for target in self.Ptargets:
            target.updateComponents(self.H, self.R)

    def update(self):
        for i in range(len(self.targets) - self.newTargetsCnt):
            self.targets[i].update(self.pd, self.lambd)

    def updateWithMeasurements(self, time):
        self.updateComponents()
        z = np.array(self.measurements[time]).T
        self.applyGating(z, time)
        self.update()
    def prunePoissonByMaxWeight(self, w):
        filters_to_stay = []
        # print(" filteres before pruning: ", len(self.phds))
        for filter in self.Ptargets:
            if filter.w > w:
                filters_to_stay.append(filter)
        print("prune len: ", len(filters_to_stay))
        self.Ptargets = filters_to_stay

    def addNoMeasurementUpdate(self,steps = 2):
        targetToDelete = []
        for i, target in enumerate(self.K_bestHypothesis.T):
            if np.sum(target) == 0:
                self.targets[i].noMeasurementUpdateCnt += 1
                if self.targets[i].noMeasurementUpdateCnt == steps:
                    targetToDelete.append(i)
            else:
                self.targets[i].noMeasurementUpdateCnt = 0
        for index in sorted(targetToDelete, reverse=True):
            del self.targets[index]

    def KLdistance(self, t1, t2):
        KL = 0
        p2_inv = np.linalg.inv(t2.P)
        det1 = np.linalg.det(t1.P)
        det2 = np.linalg.det(t2.P)
        nx = len(t1.P)
        if t1.r == 0 or t1.r == 1 or t2.r == 0 or t2.r == 1:
            KL = t1.r / 2 * (np.trace(p2_inv @ t1.P) - np.log(det1 / det2) - nx
                             + (t2.m - t1.m).T @ p2_inv @ (t2.m - t1.m))
        else:
            KL = ((1 - t1.r) * np.log((1 - t1.r) / (1 - t2.r))
                  + t1.r * np.log(t1.r / t2.r)
                  + t1.r / 2 * (np.trace(p2_inv @ t1.P) - np.log(det1 / det2) - nx
                                + (t2.m - t1.m).T @ p2_inv @ (t2.m - t1.m)))

        return KL

    def mergeLocalHypotheses(self):
        print("merge")
        for target_id, target in enumerate(self.targets):
            print("target: ", target_id)
            w = len(target.trackers)
            if w == 1:
                continue
            KL_matrix = np.ones(shape=(w, w)) * 10000
            for i, locHyp in enumerate(target.trackers):
                for j, locHyp2 in enumerate(target.trackers):
                    if i != j:
                        KL_matrix[i, j] = self.KLdistance(locHyp, locHyp2)
            print("KL:")
            print(KL_matrix)
            print("argmin:")
            ind = np.unravel_index(np.argmin(KL_matrix, axis=None), KL_matrix.shape)
            print(ind)
            print(KL_matrix[ind])
    def run(self):

        fig, ax = plt.subplots(figsize=(10, 10))
        for t in range(self.radarMap.ndat):
            print("-----------------time: ", t, "---------------")
            self.predictionForExistingTargets()
            if t < 2:
                self.predictionForBirthTargets()
            self.updateWithMeasurements(t)
            print("B. targets + P. Targets ")
            print(len(self.targets), " + ", len(self.Ptargets))

            for i, target in enumerate(self.targets):
                print("target: ", i)
                for hyp in target.trackers:
                    print("          r: ", hyp.r)
            print("Z_ids before prunning:")
            print(self.Zids)
            self.pruneByExistenceProbabiltyAndWeights()
            self.prunePoissonByMaxWeight(0.1)
            print("after prunning: B. targets + P. Targets ")
            print(len(self.targets), " + ", len(self.Ptargets))
            print("Z_ids after prunning:")
            print(self.Zids)
            self.globalHyp_mat, self.globalHyp_mat_z = self.getHypothesis()
            self.makeGlobalMatWeights()
            self.selectKBestHypothese(30)
            for i, target in enumerate(self.targets):
                print("target: ", i)
                for hyp in target.trackers:
                    print("          r: ")
            self.pruneTargets()
            print("best hyp")
            print(self.bestHypothesis)
            self.radarMap.animateRadar(t, ax)
            ax.set_title(f"{t}")
            for i,target in enumerate(self.targets):
                print("target: ", i)
                for hyp in target.trackers:
                    print("          r: ", hyp.r, " ny:", hyp.m, " P: ", np.diagonal(hyp.P))
            for i, target_id in enumerate(self.bestHypothesis):
                ax.plot(self.targets[i].trackers[target_id].m[0], self.targets[i].trackers[target_id].m[1], "+",
                        color=self.model_colors[i % len(self.model_colors)], label=f"MBM_{i}")
                confidence_ellipse([self.targets[i].trackers[target_id].m[0], self.targets[i].trackers[target_id].m[1]],
                                   self.targets[i].trackers[target_id].P, ax=ax,
                                   edgecolor=self.model_colors[i % len(self.model_colors)])

            ax.legend(loc='center left', bbox_to_anchor=(0.9, 0.5))

            # self.logWeights()

            self.addNoMeasurementUpdate(3)
            self.mergeLocalHypotheses()
            plt.waitforbuttonpress()
            # plt.pause(2)


if __name__ == '__main__':
    dt = 1
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    Q = np.diag([0.1, 0.1, 0.1, 0.1])
    R = np.diag([5, 5]) * 5
    H = np.diag([1, 1])  # 2x4
    H = np.lib.pad(H, ((0, 0), (0, 2)), 'constant', constant_values=(0))

    ndat = 100
    lambd = 0.00001
    Pd = 0.9
    Ps = 0.95

    r = Radar(F, Q, R, H, ndat, lambd)
    r.setRadarPosition([100, 100])
    # r.setRadarRadius(500)
    sx = R[0, 0] * 5
    sy = R[1, 1] * 5
    airportCov = np.array([[sx, 0, sx, 0],
                           [0, sy, 0, sy],
                           [sx, 0, 2 * sx, 0],
                           [0, sy, 0, 2 * sy]])

    r.addNRandomAirports(2, airportCov, 0.15)
    # r.addNBorderAirports(36, airportCov, 0.1)
    # seed = 123
    seed = np.random.randint(1000)
    # r.addSingleTrajectory([-150, 350, 2, -2], seed, ndat, 0, False)
    r.makeRadarMap(full_trajectories=3, short_trajectories=None, global_clutter=True, startFromAirport=True,
                   borned_trajectories=0)
    # r.makeRadarMap(full_trajectories=2, short_trajectories=[50], global_clutter=False, startFromAirport=False)

    filter = PMBM_map(r, Ps, Pd)
    filter.run()
