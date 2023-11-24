import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal as mvn
from confidence_ellipse import confidence_ellipse
from MAP2 import Radar
import itertools
from Target import Target
import math

class MBM_map:
    def __init__(self, radar, ps, pd):
        self.radarMap = radar
        self.F = radar.A
        self.H = radar.H
        self.Q = radar.Q
        self.R = radar.R
        self.ps = ps
        self.pd = pd
        self.lambd = radar.lambd
        self.targets = []
        self.mbmsToPlot = []
        self.measurements = radar.getAllMeasurementsWithinRadarRadius()
        self.globalHyp_mat = [[]]
        self.globalHyp_w = []

        x=0
        if x:
            tmp = self.measurements[0][0][0]
            tmp2 = self.measurements[0][1][0]
            self.measurements[0][0] = []
            self.measurements[0][0].append(tmp)
            self.measurements[0][1] = []
            self.measurements[0][1].append(tmp2)
            self.measurements[1][0] = []
            self.measurements[1][1] = []

            tmp = self.measurements[2][0][0]
            tmp2 = self.measurements[2][1][0]
            tmp3 = self.measurements[2][0][1]
            tmp4 = self.measurements[2][1][1]
            self.measurements[2][0] = []
            self.measurements[2][0].append(tmp)
            self.measurements[2][1] = []
            self.measurements[2][1].append(tmp2)
            self.measurements[2][0].append(tmp3)
            self.measurements[2][1].append(tmp4)

        self.model_colors = ["saddlebrown", "black", "magenta", "slategray"]

    def predictionForBirthTargets(self):
        for airport in self.radarMap.getAirports():
            w_k = 1
            r_k = airport.weight
            m_k = np.array(airport.pos)
            P_k = airport.cov
            self.targets.append(Target(len(self.targets), w_k, r_k, m_k, P_k))

    def predictionForExistingTargets(self):
        for target in self.targets:
            target.predict(self.ps, self.F, self.Q)
        # for i in range(len(self.phds)):
        #     self.phds[i].predict(self.ps, self.F, self.Q)


    def getMeasurements(self, time):
        print(self.measurements[time])
        return self.measurements[time]

    def applyGating(self, z):
        self.Zids = []
        for i, target in enumerate(self.targets):
            self.Zids.append(target.applyGating(z))
            # targetsZids[i] = hyp.Z_indexes
        #print("tmpZ : ", tmpZids)
        # tmpZids=tmpZids.flatten()
        # targetsZids= dict(enumerate(tmpZids.flatten(), 0))
        # connectedTargets = self.findConnectedTargets(targetsZids)
        #print("connected: ")
        # print(connectedTargets)

    def getHypothesis(self):
        combinations = list(itertools.product(*self.Zids))

        unique_combinations = []
        for c in combinations:
            x = [element[0] for element in c]
            if all(x.count(element) <= 1 or element == -1 for element in x):
                unique_combinations.append([element[1] for element in c])

        return np.array(unique_combinations)

    def makeGlobalMatWeights(self):
        print("hypotheses:")
        print(np.array(self.globalHyp_mat))
        print("........")
        weights=np.zeros_like(np.array(self.globalHyp_mat,dtype="float32").T)
        for i, target in enumerate(np.array(self.globalHyp_mat).T):
            for j, target_id in enumerate(target):
                # print("i: ", i)
                # print("target_id: ", target_id)
                # print("w: ", self.targets[i].trackers[target_id].w)
                weights[i][j] = self.targets[i].trackers[target_id].w
        self.globalHyp_w = np.zeros_like(weights[0])
        # print("weights:")
        # print(weights)
        # print("...........")
        for i,w in enumerate(weights.T):
            self.globalHyp_w[i] = np.prod(w)
        self.globalHyp_w /= sum(self.globalHyp_w)

    def makeCostMatrix(self):
        ## ADD measurements outside gating
        self.costMatrix = []
        for i, target in enumerate(self.targets):

            for j, allW in enumerate(target.trackersMeasurement_w):
                C_row = []
                for w in allW:
                    C_row.append(-math.log(w/target.trackersNoMeasurement_w[j]))
                self.costMatrix.append(C_row)

        print("Cost mat")
        print(self.costMatrix)


    def selectKBestHypothese(self, K):
        # print("Global hyp_w: ")
        # print(self.globalHyp_w)
        K_best = np.argsort(self.globalHyp_w)[-min(K, len(self.globalHyp_w)):]
        K_bestHypothesis = []
        for i, k in enumerate(K_best):
            K_bestHypothesis.append(self.globalHyp_mat[k])
        self.K_bestHypothesis = np.array(K_bestHypothesis)
        self.bestHypothesis = self.K_bestHypothesis[0]

        self.bestTrackers = []
        for i, tracker_id in enumerate(self.bestHypothesis):
            self.bestTrackers.append(self.targets[i].trackers[tracker_id])


    def pruneTargets(self):
        # print("K_best: ")
        # print(self.K_bestHypothesis)
        # print("best: ")
        # print(self.bestHypothesis)
        # prune by hypotheses
        for i, target in enumerate(self.targets):
            indexes = set((self.K_bestHypothesis.T)[i])
            # lowerEl = len([x for x in indexes if x < self.bestHypothesis[i]])
            # self.bestHypothesis[i] -= lowerEl
            target.trackers = [e for j, e in enumerate(target.trackers) if j in indexes]

        # for i, target in enumerate(self.targets):
        #     print(i, " :", len(target.trackers))
        #     for hyp in target.trackers:
        #         print("w: ", hyp.w)
        # print("Targets after pruning by hypothese: ")
        # for i, target in enumerate(self.targets):
        #     print(i, " :", len(target.trackers))


        # prune by r
        for j, target in enumerate(self.targets):
            indexes = []
            for i, hyp in enumerate(target.trackers):
                if hyp.r < 10e-5:
                    indexes.append(i)
            # lowerEl = len([x for x in indexes if x < self.bestHypothesis[j]])
            # self.bestHypothesis[j] -= lowerEl
            for index in sorted(indexes, reverse=True):
                del target.trackers[index]
        # print("Targets after pruning by r: ")
        # for i, target in enumerate(self.targets):
        #     print(i, " :", len(target.trackers))


        # delete targets with no localHypothese
        indexes = []
        for i, target in enumerate(self.targets):
            if len(target.trackers) == 0:
                indexes.append(i)
                # del self.bestHypothesis[i]
        for index in sorted(indexes, reverse=True):
            del self.targets[index]

        # print("Targets after pruning no local hyp: ")
        # for i, target in enumerate(self.targets):
        #     print(i, " :", len(target.trackers))

    def showGatingMeasurements(self):
        for i,target in enumerate(self.targets):
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
        # print("Z: ", z)
        self.applyGating(z)
        # connectedTargets =
        self.update()
        #self.makeHypotheses(connectedTargets)
        #self.showGatingMeasurements()


        #for l, z in enumerate(np.array(self.measurements[time]).T):


    """
    def updateWithMeasurements(self, time):
        Jk = len(self.phds)
        #print("Jk: ",Jk)
        self.updateComponents2()
        #for l, z in enumerate(zip(self.measurements[time][0], self.measurements[time][1])):
        for l, z in enumerate(np.array(self.measurements[time]).T):
            print("z: ",z)
           # z = np.array(z)
            phds_sum = 0
            for j in range(Jk):
                w = self.pd * self.phds[j].w * mvn(self.phds[j].ny, self.phds[j].S).pdf(z)
                m = self.phds[j].m + self.phds[j].K @ (z - self.phds[j].ny)
                P = self.phds[j].P
                phds_sum += w
                self.phds.append(PHD(w, m, P))
            for j in range(Jk):
                self.phds[(l + 1) * Jk + j].w = self.phds[(l + 1) * Jk + j].w / (self.radarMap.lambd + phds_sum)

        for j in range(Jk):
            self.phds[j].update(self.pd)
    """





    def run(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        for t in range(self.radarMap.ndat):
            print("-----------------time: ", t,"---------------")
            #print("     num of phds (before): ", len(self.mbms))
            self.predictionForExistingTargets()
            self.predictionForBirthTargets()
            #print("     num of phds (after predict): ", len(self.mbms))

            self.updateWithMeasurements(t)

            self.globalHyp_mat=self.getHypothesis()
            self.makeGlobalMatWeights()
            # print(self.globalHyp_w)
            self.selectKBestHypothese(10)
            self.pruneTargets()
            # self.makeCostMatrix()


            # print("hypothese: ",len(self.globalHyp_mat))
            # print(self.globalHyp_mat)
           # self.getPHDsToPlot()
            self.radarMap.animateRadar(t, ax)
            print("best hyp:")
            print(self.bestHypothesis)
            for i, target in enumerate(self.targets):
                print(i,": ")
                for hyp in target.trackers:
                    try:
                        print("    r: ", hyp.r, " w: ", hyp.w, "z: ", hyp.updatedBy)
                    except:
                        print("    r: ", hyp.r, " w: ", hyp.w, "z: ")

            for i, target in enumerate(self.bestTrackers):
                ax.plot(target.m[0],target.m[1], "+", color=self.model_colors[i % len(self.model_colors)], label="MBM")
                confidence_ellipse([target.m[0], target.m[1]],
                                   target.P, ax=ax, edgecolor = self.model_colors[i % len(self.model_colors)])
            # for i, target in enumerate(self.targets):
            #     for j, tracker in enumerate(target.trackers):
            #     ax.plot(filter.m[0], filter.m[1], "+", color=self.model_colors[i % len(self.model_colors)], label="PHD")
            #     confidence_ellipse([filter.m[0], filter.m[1]], filter.P, ax=ax,
            #                        edgecolor=self.model_colors[i % len(self.model_colors)])
                # print(filter.P.diagonal())
            # print(filter.P)
            # plt.plot(filter)
            # ax.legend(loc='center left', bbox_to_anchor=(0.9, 0.5))
            plt.waitforbuttonpress()
            #plt.pause(2)


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

    r.addNRandomAirports(1, airportCov, 0.15)
    #r.addNBorderAirports(36, airportCov, 0.1)
    # seed = 123
    seed=np.random.randint(1000)
    #r.addSingleTrajectory([-150, 350, 2, -2], seed, ndat, 0, False)
    r.makeRadarMap(full_trajectories=2, short_trajectories=None, global_clutter=True, startFromAirport=True,
                   borned_trajectories=0)
     # r.makeRadarMap(full_trajectories=2, short_trajectories=[50], global_clutter=False, startFromAirport=False)

    filter = MBM_map(r, Ps, Pd)
    filter.run()

