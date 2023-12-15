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
        self.Zids = []
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
            self.Ptargets.append(PoissonTarget(w,m, P))

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

        for i, target in enumerate(self.targets):
            self.Zids.append(target.applyGating(z))

    def getHypothesis(self):

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
                if hyp.r < 1e-4 or hyp.w < 1e-4:
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
            deleted = set((self.globalHyp_mat.T)[i]) - indexes
            lowerEl = len([x for x in deleted if x < self.bestHypothesis[i]])
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

    def update(self):
        for target in self.targets:
            target.update(self.pd, self.lambd)

    def updateWithMeasurements(self, time):


        self.updateComponents()
        z = np.array(self.measurements[time]).T
        self.applyGating(z, time)
        self.update()


    def run(self):

        fig, ax = plt.subplots(figsize=(10, 10))
        for t in range(self.radarMap.ndat):
            print("-----------------time: ", t, "---------------")
            self.predictionForExistingTargets()
            self.predictionForBirthTargets()
            self.updateWithMeasurements(t)
            self.pruneByExistenceProbabiltyAndWeights()

            self.globalHyp_mat, self.globalHyp_mat_z = self.getHypothesis()
            self.makeGlobalMatWeights()
            self.selectKBestHypothese(10)
            self.pruneTargets()

            self.radarMap.animateRadar(t, ax)
            ax.set_title(f"{t}")
            for i, target_id in enumerate(self.bestHypothesis):
                ax.plot(self.targets[i].trackers[target_id].m[0], self.targets[i].trackers[target_id].m[1], "+",
                        color=self.model_colors[i % len(self.model_colors)], label=f"MBM_{i}")
                confidence_ellipse([self.targets[i].trackers[target_id].m[0], self.targets[i].trackers[target_id].m[1]],
                                   self.targets[i].trackers[target_id].P, ax=ax,
                                   edgecolor=self.model_colors[i % len(self.model_colors)])

            ax.legend(loc='center left', bbox_to_anchor=(0.9, 0.5))
            fig.savefig(f'./pics/t_{t}.png')
            # self.logWeights()
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
    lambd = 0.0001
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
