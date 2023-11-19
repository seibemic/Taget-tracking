import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal as mvn
from confidence_ellipse import confidence_ellipse
from MBM import MBM
from MAP2 import Radar


class MBM_map:
    def __init__(self, radar, ps, pd):
        self.radarMap = radar
        self.F = radar.A
        self.H = radar.H
        self.Q = radar.Q
        self.R = radar.R
        self.ps = ps
        self.pd = pd
        self.mbms = []
        self.mbmsToPlot = []
        self.measurements = radar.getAllMeasurementsWithinRadarRadius()


        self.model_colors = ["saddlebrown", "black", "magenta", "slategray"]

    def predictionForBirthTargets(self):
        for airport in self.radarMap.getAirports():
            w_k = 1
            r_k = airport.weight
            m_k = np.array(airport.pos)
            P_k = airport.cov
            self.mbms.append(MBM(w_k, r_k, m_k, P_k))

    def predictionForExistingTargets(self):
        for target in self.mbms:
            target.predict(self.ps, self.F, self.Q)
        # for i in range(len(self.phds)):
        #     self.phds[i].predict(self.ps, self.F, self.Q)

    def updateComponents(self):
        for target in self.mbms:
            target.updateComponents(self.H, self.R)

    def getMeasurements(self, time):
        print(self.measurements[time])
        return self.measurements[time]

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




    def run(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        for t in range(self.radarMap.ndat):
            print("time: ", t)
            print("     num of phds (before): ", len(self.phds))
            self.predictionForExistingTargets()
            self.predictionForBirthTargets()
            print("     num of phds (after predict): ", len(self.phds))

            self.updateWithMeasurements(t)

            self.getPHDsToPlot()
            self.radarMap.animateRadar(t, ax)
            for i, filter in enumerate(self.phdsToPlot):
                ax.plot(filter.m[0], filter.m[1], "+", color=self.model_colors[i % len(self.model_colors)], label="PHD")
                confidence_ellipse([filter.m[0], filter.m[1]], filter.P, ax=ax,
                                   edgecolor=self.model_colors[i % len(self.model_colors)])
                # print(filter.P.diagonal())
            # print(filter.P)
            # plt.plot(filter)
            ax.legend(loc='center left', bbox_to_anchor=(0.9, 0.5))
            plt.pause(0.3)


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
    lambd = 0.0000
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
    r.makeRadarMap(full_trajectories=1, short_trajectories=None, global_clutter=True, startFromAirport=True,
                   borned_trajectories=0)
     # r.makeRadarMap(full_trajectories=2, short_trajectories=[50], global_clutter=False, startFromAirport=False)

    filter = PHD_map(r, Ps, Pd)
    filter.run()

