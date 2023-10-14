import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.stats import poisson, uniform
import matplotlib.pyplot as plt
import time


class Trajectory:
    def __init__(self, A, Q, H, R, z0, seed=123, ndat=200, startTime=0, borned=False):
        self.ndat = ndat
        self.seed = seed
        self.A = A
        self.Q = Q
        self.H = H
        self.R = R
        self.m0 = z0
        self.startTime = startTime
        self.bornedFromAnotherTrajectory = borned
        # self.m0 = np.array([4000., 80000., 0., 0.])
        self.X = np.zeros(shape=(self.A.shape[0], self.ndat))  # true positions
        self.Y = np.zeros(shape=(self.H.shape[0], self.ndat))  # measured positions
        self._simulate()

    def _simulate(self):
        np.random.seed(self.seed)

        x = self.m0
        for t in range(self.ndat):
            x = self.A.dot(x) + mvn.rvs(cov=self.Q)
            y = self.H.dot(x) + mvn.rvs(cov=self.R)
            self.X[:, t] = x.flatten()
            self.Y[:, t] = y.flatten()


class Airport:
    def __init__(self):
        print("TODO")
class Radar:
    def __init__(self):
        self.radarPosition = [0., 0.]
        self.radarRadius = 300
        self.map = None


        self.mapMeasurements = None
        self.radarMeasurements = []
        self.trajectoriesMeasurements = None
        self.Airports = None

        self.meas_colors = ["orange", "lime", "indigo", "steelblue"]
        self.traj_colors = ["red", "green", "blue", "dodgerblue"]
        self.model_colors = ["saddlebrown", "black", "magenta", "slategray"]
    def setRadarPosition(self, position):
        self.radarPosition = position

    def setRadarRadius(self, radius):
        self.radarRadius = radius

    def addMap(self, A, Q, R, H, ndat, area_vol, lambd):
        self.map = MapGenerator(A, Q, R, H, ndat, area_vol, lambd)

    def makeMap(self, full_trajectories=1, short_trajectories=None, borned_trajectories=0,
                global_clutter=False, singlePoints=0, trajectory_clutters=False, crossNTrajectories=0):
        if not self.map:
            raise Exception("map not added yet")
        minMax=[self.radarPosition[0]-self.radarRadius,self.radarPosition[0]+self.radarRadius,self.radarPosition[1]-self.radarRadius,self.radarPosition[1]+self.radarRadius]
        self.mapMeasurements = self.map.makeMap(full_trajectories=full_trajectories, short_trajectories=short_trajectories, borned_trajectories=borned_trajectories,
                               global_clutter=global_clutter, singlePoints=singlePoints, trajectory_clutters=trajectory_clutters, crossNTrajectories=crossNTrajectories,
                                minMaxClutter=minMax)
        self.trajectoriesMeasurements = self.map.getAllTrueMeasurements()
        self.getAllMeasurementsWithinRadarRadius()

    def getAllMeasurementsWithinRadarRadius(self):
        print(self.mapMeasurements)
        # print(len(self.mapMeasurements)) #[time][coord(x,xy)][measurement_id]
        print("----------------------")
        for t in range(len(self.mapMeasurements)): # time
            radarMeasuremens = []
            radarMeasuremensX = []
            radarMeasuremensY = []
            allX=self.mapMeasurements[t][0]
            allY=self.mapMeasurements[t][1]

            for m_id in range(len(allX)):
                if (allX[m_id]-self.radarPosition[0])**2 + (allY[m_id]-self.radarPosition[1])**2 < self.radarRadius**2:
                    radarMeasuremensX.append(allX[m_id])
                    radarMeasuremensY.append(allY[m_id])
            radarMeasuremens.append(radarMeasuremensX)
            radarMeasuremens.append(radarMeasuremensY)
            self.radarMeasurements.append(radarMeasuremens)
        return self.radarMeasurements


    def animateRadar(self,showTrueTrajectories=True, showTrueTrajectoriesMeasurements=True, showRadar = True,
                     showMapClutter=True, showRadarClutter=True):
        fig, ax = plt.subplots(figsize=(10, 10))
        for t in range(self.map.ndat):
            ax.cla()
            ax.set_xlim(self.radarPosition[0] - self.radarRadius - 20, self.radarPosition[0] + self.radarRadius + 20)
            ax.set_ylim(self.radarPosition[1] - self.radarRadius - 20, self.radarPosition[1] + self.radarRadius + 20)
            if showTrueTrajectories:
                for i, traj in enumerate(self.map.trajectories):
                    plt.plot(traj.X[0], traj.X[1], "-", color=self.traj_colors[i % len(self.traj_colors)])
            if showMapClutter:
                plt.plot(self.mapMeasurements[t][0], self.mapMeasurements[t][1], "*", color="gold")
            if showRadarClutter:
                plt.plot(self.radarMeasurements[t][0], self.radarMeasurements[t][1], "*", color="orange")
            if showTrueTrajectoriesMeasurements:
                plt.plot(self.trajectoriesMeasurements[t][0], self.trajectoriesMeasurements[t][1], "*r")
            if showRadar:
                radar=plt.Circle((self.radarPosition),self.radarRadius, color="b", fill=False)
                plt.plot(self.radarPosition,"*b")
            ax.add_patch(radar)
            # p1=[200,200]
            #
            # if( ((p1[0] - self.radarPosition[0])**2 + (p1[1] - self.radarPosition[1])**2) < self.radarRadius**2):
            #     plt.plot(p1[0],p1[1], marker="*", color="black")
            # else:
            #     plt.plot(p1[0],p1[1],  marker="*", color="orange")
            plt.pause(0.3)


class MapGenerator:
    # def __init__(self, ndat=200, global_clutter = False, trajectory_clutters = False,
    #            full_trajectories = 1, new_trajectories=[], borned_trajectories=[[]]):
    def __init__(self, A, Q, R, H, ndat, area_vol, lambd):
        self.allMeasurements = []
        self.A = A
        self.Q = Q
        self.R = R
        self.H = H
        self.ndat = ndat
        self.area_vol = area_vol
        self.lambd = lambd
        self.trajectories = []

        self.meas_colors = ["orange", "lime", "indigo", "steelblue"]
        self.traj_colors = ["red", "green", "blue", "dodgerblue"]
        self.model_colors = ["saddlebrown", "black", "magenta", "slategray"]

    def setA(self, A):
        self.A = A

    def setQ(self, Q):
        self.Q = Q

    def setR(self, R):
        self.R = R

    def setH(self, H):
        self.H = H

    def addAirport(self):
        print("TODO")
        raise Exception("This method is not done yet.")

    def addBorderAirports(self):
        print("TODO")
        raise Exception("This method is not done yet.")

    def addSingleTrajectory(self, z0, seed=123, lenght=0, startingTime=0, borned=False):  # add starting time
        if (len(z0) != len(self.A)):
            z0 = np.zeros(len(self.A))
        if (lenght + startingTime > self.ndat):
            raise Exception("ERR lenght + startingTime > self.ndat")
        self.trajectories.append(Trajectory(self.A, self.Q, self.H, self.R, z0, seed, lenght, startingTime, borned))

    def addNTrajectories(self, N, startingTime=0, borned=False):
        for i in range(N):
            z0 = np.zeros(len(self.A))
            for j in range(len(self.R)):
                z0[j] = np.random.randint(100)

            self.addSingleTrajectory(z0, np.random.randint(1000), self.ndat - startingTime, startingTime, borned)

    def addNShortTrajectories(self, startingTimes):
        if not isinstance(startingTimes, list):
            raise Exception(startingTimes, " is not a list")
        for time in startingTimes:
            self.addNTrajectories(1, time, False)

    def crossTrajectories(self, trajStable, trajToMove, eps=2):
        arg_p1 = np.random.randint(len(trajStable.X[1]) * .3, len(trajStable.X[1]) * .9)

        point_t1 = np.zeros(len(self.R))
        point_t2 = np.zeros(len(self.R))
        for i in range(len(self.R)):
            point_t1[i] = trajStable.X[i][arg_p1]
            point_t2[i] = trajToMove.X[i][arg_p1]

        eps = mvn.rvs(cov=np.eye(len(self.R)) * eps)

        for i in range(len(self.R)):
            trajToMove.X[i] += (point_t1 - point_t2)[i] + eps[i]
            trajToMove.Y[i] += (point_t1 - point_t2)[i] + eps[i]
        return trajToMove

    def crossNRandomTrajectories(self, N, eps=2):
        nonMovedTrajectories = []
        for i, traj in enumerate(self.trajectories):
            if not traj.bornedFromAnotherTrajectory:
                nonMovedTrajectories.append(i)
        if (len(nonMovedTrajectories) < N + 1):
            raise Exception("not enough trajectories to cross")
        for i in range(N):
            r_m = np.random.randint(len(nonMovedTrajectories))
            idm = nonMovedTrajectories[r_m]
            trajToMove = self.trajectories[idm]
            nonMovedTrajectories.remove(nonMovedTrajectories[r_m])
            r_s = np.random.randint(len(nonMovedTrajectories))
            trajStable = self.trajectories[nonMovedTrajectories[r_s]]
            self.trajectories[idm] = self.crossTrajectories(trajStable, trajToMove, eps)

    def addBornedTrajectory(self, fromTrajectoryID=0, lenght=0, bornedTime=0):
        z0 = np.zeros(len(self.trajectories[fromTrajectoryID].X))
        z0[0:len(self.R)] = self.trajectories[fromTrajectoryID].X[:2, bornedTime]
        self.addSingleTrajectory(z0, np.random.randint(1000), lenght, bornedTime, True)

    def getRandomNotBornedTrajectoryID(self):
        trajsID = []
        for i, traj in enumerate(self.trajectories):
            if not traj.bornedFromAnotherTrajectory:
                trajsID.append(i)
        r = np.random.randint(len(trajsID))
        return trajsID[r]

    def addNRandomlyBornedTrajectories(self, N):
        for i in range(N):
            id = self.getRandomNotBornedTrajectoryID()
            bornedTime = np.random.randint(self.trajectories[id].ndat * 0.25, self.trajectories[id].ndat * 0.75)
            lenght = self.trajectories[id].ndat - bornedTime
            self.addBornedTrajectory(id, lenght, bornedTime)

    def getTrajectoriesIDInTimeT(self, T):
        trIDs = []
        for i, traj in enumerate(self.trajectories):
            if traj.startTime <= T and traj.startTime + traj.ndat > T:
                trIDs.append(i)
        return trIDs

    def getAllTrueMeasurements(self):
        measurementsL2 = []
        for t in range(self.ndat):
            actTrajectories = self.getTrajectoriesIDInTimeT(t)
            measurementsL1 = []
            for coord in range(len(self.R)):
                measurementsL0 = []
                for trID in actTrajectories:
                    measurementsL0.append(self.trajectories[trID].Y[coord][t - self.trajectories[trID].startTime])
                measurementsL1.append(measurementsL0)
            measurementsL2.append(measurementsL1)
        return measurementsL2

    def getMaxMin(self, measurements, coord):
        Xmin = min(measurements[0][coord])
        Xmax = max(measurements[0][coord])

        for i in range(len(measurements)):
            if max(measurements[i][coord]) > Xmax:
                Xmax = max(measurements[i][coord])
            if min(measurements[i][coord]) < Xmin:
                Xmin = min(measurements[i][coord])
        return Xmax, Xmin

    def addGlobalClutter(self, NSinglePoints=0, offset=0.001, minMax=None):
        if (len(self.R) > 2):
            raise Exception("method addGlobalClutter is done for only 2D map")
        trueMeasurements = self.getAllTrueMeasurements()
        X0max, X0min, X1max, X1min = [0, 0, 0, 0]

        if not minMax:
            X0max, X0min = self.getMaxMin(trueMeasurements, 0)
            X1max, X1min = self.getMaxMin(trueMeasurements, 1)
        else:
            X0min, X0max = minMax[0:2]
            X1min, X1max = minMax[2:4]

        self.area_vol = abs(X0max - X0min) * abs(X1max - X1min)
        clutterCount = poisson.rvs(self.area_vol * self.lambd, size=self.ndat)

        for t in range(self.ndat):
            if t < NSinglePoints:
                Y = np.zeros((2, len(trueMeasurements[t][0])))
                Y[0, 0:] = np.array(trueMeasurements[t][0])
                Y[1, 0:] = np.array(trueMeasurements[t][1])
                self.allMeasurements.append(Y)
                continue

            Y = np.zeros((2, len(trueMeasurements[t][0]) + clutterCount[t]))
            Y[0, 0:len(trueMeasurements[t][0])] = np.array(trueMeasurements[t][0])
            Y[1, 0:len(trueMeasurements[t][1])] = np.array(trueMeasurements[t][1])

            Y[0, len(trueMeasurements[t][0]):] = np.random.uniform(low=X0min * (1 - np.sign(X0min) * offset),
                                                                   high=X0max * (1 + np.sign(X0max) * offset),
                                                                   size=clutterCount[t])
            Y[1, len(trueMeasurements[t][0]):] = np.random.uniform(low=X1min * (1 - np.sign(X1min) * offset),
                                                                   high=X1max * (1 + np.sign(X1max) * offset),
                                                                   size=clutterCount[t])
            self.allMeasurements.append(Y)
        return self.allMeasurements

    def addTrajectoriesClutter(self):
        print("TODO")
        raise Exception("This method is not done yet.")

    def makeMap(self, full_trajectories=1, short_trajectories=None, borned_trajectories=0,
                global_clutter=False, singlePoints=0, trajectory_clutters=False, crossNTrajectories=0, minMaxClutter=None):
        self.addNTrajectories(full_trajectories)
        if short_trajectories:
            self.addNShortTrajectories(short_trajectories)
        self.addNRandomlyBornedTrajectories(borned_trajectories)
        self.crossNRandomTrajectories(crossNTrajectories)
        if global_clutter:
            measurements = self.addGlobalClutter(NSinglePoints=singlePoints,minMax=minMaxClutter)
            return measurements
        else:
            return None



        # map.addGlobalClutter(2)

        # map.animateAll()
        # map.addBornedTrajectory(0, 50, 50)

    def printTrajectories(self):
        for traj in self.trajectories:
            plt.plot(traj.X[0], traj.X[1], "-")
        plt.show()

    def animateAll(self, showTrueTrajectories=True, showTrueTrajectoriesMeasurements=True):
        fig, ax = plt.subplots(figsize=(10, 10))
        true = self.getAllTrueMeasurements()
        minX, maxX = self.getMaxMin(true, 0)
        minY, maxY = self.getMaxMin(true, 1)
        for t in range(self.ndat):
            ax.cla()
            ax.set_xlim((minX, maxX))
            ax.set_ylim((minY, maxY))
            # ax.set_xlim(self.radarPosition[0] - self.radarRadius - 20, self.radarPosition[0] + self.radarRadius + 20)
            # ax.set_ylim(self.radarPosition[1] - self.radarRadius - 20, self.radarPosition[1] + self.radarRadius + 20)
            if showTrueTrajectories:
                for i, traj in enumerate(self.trajectories):
                    plt.plot(traj.X[0], traj.X[1], "-", color=self.traj_colors[i % len(self.traj_colors)])
            plt.plot(self.allMeasurements[t][0], self.allMeasurements[t][1], "*", color="yellow")
            if showTrueTrajectoriesMeasurements:
                plt.plot(true[t][0], true[t][1], "*r")
            # if showRadar:
            #     radar=plt.Circle((self.radarPosition),self.radarRadius, color="b", fill=False)
            #     plt.plot(self.radarPosition,"*b")
            # ax.add_patch(radar)
            # p1=[200,200]
            #
            # if( ((p1[0] - self.radarPosition[0])**2 + (p1[1] - self.radarPosition[1])**2) < self.radarRadius**2):
            #     plt.plot(p1[0],p1[1], marker="*", color="black")
            # else:
            #     plt.plot(p1[0],p1[1],  marker="*", color="orange")
            plt.pause(0.3)


if __name__ == '__main__':
    dt = 1
    A = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    Q = np.diag([0.1, 0.1, 0.01, 0.01])
    R = np.diag([5, 5])
    H = np.diag([1, 1])  # 2x4
    H = np.lib.pad(H, ((0, 0), (0, 2)), 'constant', constant_values=(0))

    ndat = 50
    area_vol = 20
    lambd = 0.0001
    n_objects = 2
    Pg = 0.99
    Pd = 0.9
    r0 = 0.98
    Ps = 0.95

    # map = MapGenerator(A, Q, R, H, ndat, area_vol, lambd)
    # map.addNTrajectories(2)
    # map.crossNRandomTrajectories(1)
    # map.addNShortTrajectories([2])
    # map.addGlobalClutter(2)
    r= Radar()
    r.addMap(A, Q, R, H, ndat, area_vol, lambd)
    r.makeMap(full_trajectories=2, short_trajectories=[10], global_clutter=True)
    # r.getAllMeasurementsWithinRadarRadius()
    r.animateRadar()
# map.animateAll()
# map.addBornedTrajectory(0, 50, 50)
# map.addNRandomlyBornedTrajectories(2)

# map.printTrajectories()


#  traj=SingleTrajectory(A,Q,H,R,0,123,200)


#    plt.plot(traj.X[0],traj.X[1],"-.")
#   plt.show()
#    for t in range(ndat):
#       plt.plot()
