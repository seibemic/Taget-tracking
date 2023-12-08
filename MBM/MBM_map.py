import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal as mvn
from confidence_ellipse import confidence_ellipse
from MAP2 import Radar
import itertools
from Target import Target
import math
from scipy.special import logsumexp
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

    def applyGating(self, z,t):
        self.Zids = []
        for i, target in enumerate(self.targets):
            self.Zids.append(target.applyGating(z))
        print("Z ids:")
        print(self.Zids)
        for t_z in self.Zids:
            for inner_z in t_z:
                print(inner_z, end=": ")
                if inner_z[0] == -1:
                    print(-1, end=", ")
                else:
                    print(self.measurements[t][0][inner_z[0]], " ", self.measurements[t][1][inner_z[0]], end=", ")
                print()
            # targetsZids[i] = hyp.Z_indexes
        #print("tmpZ : ", tmpZids)
        # tmpZids=tmpZids.flatten()
        # targetsZids= dict(enumerate(tmpZids.flatten(), 0))
        # connectedTargets = self.findConnectedTargets(targetsZids)
        #print("connected: ")
        # print(connectedTargets)

    def getHypothesis(self):
        print("Z_ids")
        print(self.Zids)
        print("Targets lens:")
        for i, target in enumerate(self.targets):
            print(len(target.trackers), end=", ")
        print()
        combinations = list(itertools.product(*self.Zids))
        print("num of combinations:", len(combinations))
        target_id = []
        z_id = []
        for c in combinations:
            x = [element[0] for element in c]
            if all(x.count(element) <= 1 or element == -1 for element in x):
                target_id.append([element[1] for element in c])
                z_id.append([element[0] for element in c])
        print("num of combinations after:", len(target_id))
        return np.array(target_id), np.array(z_id)

    def makeGlobalMatWeights(self):

        # print("hypotheses Z:")
        # for i, hyp in enumerate(np.array(self.globalHyp_mat_z)):
        #     print(i, ": ", hyp)
        # print("........")
        # print("Global hyp mat: ")
        # print(self.globalHyp_mat)
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
            self.globalHyp_w[i] = np.sum(w)
        #print("logsumexp: ", logsumexp(self.globalHyp_w))
        self.globalHyp_w /= logsumexp(self.globalHyp_w)
        #self.globalHyp_w /= sum(self.globalHyp_w)
        # print(self.globalHyp_w)
        # print("hypotheses W:")
        # for i, hyp in enumerate(np.array(self.globalHyp_w)):
        #     print(i, ": ", hyp)
        print("hypotheses    |    hypotheses Z     |     hypotheses W    |    targets W   |   targets r")
        i=0
        for hyp, hyp_z, hyp_w in (zip(self.globalHyp_mat, self.globalHyp_mat_z, self.globalHyp_w)):
            print(i,": ", hyp, " | ", hyp_z, "  |  ", hyp_w, end="   |   ")
            for j, x in enumerate(hyp):
                print(self.targets[j].trackers[x].w, end = ", ")
            print("    |    ", end = " ")
            for j, x in enumerate(hyp):
                print(self.targets[j].trackers[x].r, end=", ")
            print()

            i+=1
        print("--------------------")

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
        # K_best = sorted(self.globalHyp_w,reverse=True)#[:min(K, len(self.globalHyp_w))]
        # print("K_best")
        # print(K_best)
        K_best = np.argsort(-self.globalHyp_w)[:min(K, len(self.globalHyp_w))]
        # print("K_best")
        # print(K_best)
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
                if hyp.r < 1e-2 or hyp.w < 1e-2:
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

            # print("Targets after pruning by r: ")
            # for i, target in enumerate(self.targets):
            #     print(i, " :", len(target.trackers))
    def pruneTargets(self):
        # print("K_best: ")
        # print(self.K_bestHypothesis)
        # print("best: ")
        # print(self.bestHypothesis)
        # prune by hypotheses
        lens = []
        for i, target in enumerate(self.targets):
            lens.append(len(target.trackers))
        print("lens before hyp prune: ", lens)
        for i, target in enumerate(self.targets):
            indexes = set((self.K_bestHypothesis.T)[i])
            deleted = set((self.globalHyp_mat.T)[i])-indexes
            # print("deleted: ", deleted)
            # r = set([x for x in range(0, self.bestHypothesis[i])])
            # deleted = r - deleted
            # print("deleted: ", deleted)
            # print("indexes: ", indexes)
            # print("deleted: ", deleted)
            lowerEl = len([x for x in deleted if x < self.bestHypothesis[i]])
            self.bestHypothesis[i] -= lowerEl
            target.trackers = [e for j, e in enumerate(target.trackers) if j in indexes]
            # if len(target.trackers) == 1:
            #     self.bestHypothesis[i] = 0
            # print(i, " final len: ", len(target.trackers))
        lens = []
        for i, target in enumerate(self.targets):
            lens.append(len(target.trackers))
        print("lens after hyp prune: ", lens)
        # for i, target in enumerate(self.targets):
        #     print(i, " :", len(target.trackers))
        #     for hyp in target.trackers:
        #         print("w: ", hyp.w)
        # print("Targets after pruning by hypothese: ")
        # for i, target in enumerate(self.targets):
        #     print(i, " :", len(target.trackers))






        # delete targets with no localHypothese
        indexes = []
        for i, target in enumerate(self.targets):
            if len(target.trackers) == 0:
                indexes.append(i)
                del self.bestHypothesis[i]
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
        self.applyGating(z,time)
        # connectedTargets =
        self.update()
        #self.makeHypotheses(connectedTargets)
        #self.showGatingMeasurements()


        #for l, z in enumerate(np.array(self.measurements[time]).T):

    def logWeights(self):
        for target in self.targets:
            target.logWeights()
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
        import os
        import glob

        files = glob.glob('./pics/*.png')
        for f in files:
            os.remove(f)
        fig, ax = plt.subplots(figsize=(10, 10))
        for t in range(self.radarMap.ndat):
            print("-----------------time: ", t,"---------------")
            #print("     num of phds (before): ", len(self.mbms))
            self.predictionForExistingTargets()
           # if t %10==0 or 1:
            self.predictionForBirthTargets()
            #print("     num of phds (after predict): ", len(self.mbms))

            self.updateWithMeasurements(t)


            # self.pruneByWeights(10e-5)

            self.pruneByExistenceProbabiltyAndWeights()
            self.globalHyp_mat, self.globalHyp_mat_z=self.getHypothesis()
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
                        print("    r: ", hyp.r, " w: ", hyp.w, "P: ", np.diagonal(hyp.P))
                    except:
                        print("    r: ", hyp.r, " w: ", hyp.w, "z: ")
            ax.set_title(f"{t}")
            for i, target_id in enumerate(self.bestHypothesis):
                ax.plot(self.targets[i].trackers[target_id].m[0],self.targets[i].trackers[target_id].m[1], "+",
                        color=self.model_colors[i % len(self.model_colors)], label=f"MBM_{i}")
                confidence_ellipse([self.targets[i].trackers[target_id].m[0], self.targets[i].trackers[target_id].m[1]],
                                   self.targets[i].trackers[target_id].P, ax=ax, edgecolor = self.model_colors[i % len(self.model_colors)])
            # for i, target in enumerate(self.targets):
            #     for j, tracker in enumerate(target.trackers):
            #     ax.plot(filter.m[0], filter.m[1], "+", color=self.model_colors[i % len(self.model_colors)], label="PHD")
            #     confidence_ellipse([filter.m[0], filter.m[1]], filter.P, ax=ax,
            #                        edgecolor=self.model_colors[i % len(self.model_colors)])
                # print(filter.P.diagonal())
            # print(filter.P)
            # plt.plot(filter)
            ax.legend(loc='center left', bbox_to_anchor=(0.9, 0.5))
            fig.savefig(f'./pics/t_{t}.png')
            #self.logWeights()
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

    r.addNRandomAirports(2, airportCov, 0.15)
    #r.addNBorderAirports(36, airportCov, 0.1)
    # seed = 123
    seed=np.random.randint(1000)
    #r.addSingleTrajectory([-150, 350, 2, -2], seed, ndat, 0, False)
    r.makeRadarMap(full_trajectories=3, short_trajectories=None, global_clutter=True, startFromAirport=True,
                   borned_trajectories=0)
     # r.makeRadarMap(full_trajectories=2, short_trajectories=[50], global_clutter=False, startFromAirport=False)

    filter = MBM_map(r, Ps, Pd)
    filter.run()

