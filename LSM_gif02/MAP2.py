import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.stats import poisson, uniform
import matplotlib.pyplot as plt
import math
from confidence_ellipse import confidence_ellipse


class Trajectory:

    def __init__(self, A, Q, H, R, z0, seed=123, ndat=200, startTime=0, borned=False):
        """
        Class to simulate one Trajectory.
        :param A: state transition matrix x_k = A * x_(k-1) + Q
        :param Q: process noise covariance matrix of state estimate N(0, Q)
        :param H: Observation matrix
        :param R: observation noise covariance matrix N(0, R)
        :param z0: vector of starting coordinates
        :param seed: random seed
        :param ndat: num of time steps
        :param startTime: starting time of given trajectory on whole map
        :param borned: bool, True if trajectory was borned from another trajectory
        """
        self.ndat = ndat
        self.seed = seed
        self.A = A
        self.Q = Q
        self.H = H
        self.R = R
        self.m0 = z0
        self.startTime = startTime
        self.bornedFromAnotherTrajectory = borned

        self.X = np.zeros(shape=(self.A.shape[0], self.ndat))  # true positions
        self.Y = np.zeros(shape=(self.H.shape[0], self.ndat))  # measured positions
        self._simulate()

    def _simulate(self):
        """
        Simulation of trajectory.
        :return:
        """
        np.random.seed(self.seed)

        x = self.m0
        for t in range(self.ndat):
            x = self.A.dot(x) + mvn.rvs(cov=self.Q)
            y = self.H.dot(x) + mvn.rvs(cov=self.R)
            self.X[:, t] = x.flatten()
            self.Y[:, t] = y.flatten()


class Airport:
    def __init__(self, pos, cov, weight):
        """
        Class Airport for making Airports (borning places) for radar.
        :param pos: Position of an Airport. This Position will be given to Trajectorie borned in this Airport.
        :param cov: Covariance matrix of an Airport. This Covariance will be given to Trajectorie borned in this Airport.
        :param weight: Weight to give to borned trajectory from this airport
        """
        self.pos = pos
        self.cov = cov
        self.weight = weight


class MapGenerator:
    def __init__(self, A, Q, R, H, ndat, lambd):
        """

        :param A: state transition matrix x_k = A * x_(k-1) + Q
        :param Q: process noise covariance matrix of state estimate N(0, Q)
        :param H: Observation matrix
        :param R: observation noise covariance matrix N(0, R)
        :param ndat: Num of time steps.
        :param lambd: Intensity of Poisson Clutter
        """
        self.allMeasurements = []  # All measurements on the map (Targets + Clutter)
        self.A = A
        self.Q = Q
        self.R = R
        self.H = H
        self.ndat = ndat
        self.lambd = lambd
        self.trajectories = []  # Trajectories on the map

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

    def addSingleTrajectory(self, z0, seed=123, lenght=0, startingTime=0, borned=False):
        """
        Method to add a single Trajectory to the map.
        :param z0: Starting position of trajectory.
        :param seed: seed
        :param lenght: Length (num of time steps) of trajectory
        :param startingTime: Starting time of the trajectory
        :param borned: bool; True if trajectory was borned from another trajectory
        :return:
        """

        if (len(z0) != len(self.A)):
            z0 = np.zeros(len(self.A))
        if (lenght + startingTime > self.ndat):
            raise Exception("ERR lenght + startingTime > self.ndat")
        self.trajectories.append(Trajectory(self.A, self.Q, self.H, self.R, z0, seed, lenght, startingTime, borned))

    def addNTrajectories(self, N, startingTime=0, borned=False, z0=None):
        """
        Method to add N single trajectories with starting time startingTime.
        :param N: Num of trajectories to add.
        :param startingTime: Starting time of trajectories
        :param borned: bool; True if trajectory was borned from another trajectory
        :param z0: Starting position of trajectories. If None is given. Random vector with elements 0-100 is given.
        :return:
        """
        if not z0:
            for i in range(N):
                z0 = np.zeros(len(self.A))
                for j in range(len(self.R)):
                    z0[j] = np.random.randint(100)
                s = np.random.randint(1000)
                self.addSingleTrajectory(z0, s, self.ndat - startingTime, startingTime, borned)
        else:
            for i in range(N):
                s = np.random.randint(1000)
                self.addSingleTrajectory(z0, s, self.ndat - startingTime, startingTime, borned)

    def addNShortTrajectories(self, startingTimes, z0=None):
        """
        Method to add N shorter Trajectories.
        :param startingTimes: list of starting Times of trajectories. Num of short strajectories is based on len of this list
        :param z0: Starting position of trajectories. If None is given. Random vector with elements 0-100 is given.
        :return:
        """
        if not isinstance(startingTimes, list):
            raise Exception(startingTimes, " is not a list")

        for time in startingTimes:
            self.addNTrajectories(1, time, False, z0)

    def crossTrajectories(self, trajStable, trajToMove, eps=2):
        """
        Method to cross two trajectories. Usefull for example for JPDA filter, where we want to see the performance of a filter
        when two trajectories meet each other in given timestep
        :param trajStable: Instance of Trajectory. This trajectory wont move and stays in same position.
        :param trajToMove: Instance of Trajectory. This trajectory will move and its position wont stay the same.
        :param eps: epsilon variable to make some noise in crossing coordinates
        :return: trajToMove;
        """
        arg_p1 = np.random.randint(len(trajStable.X[1]) * .3, len(
            trajStable.X[1]) * .9)  # random point (time step) in which trajectories will cross (between 30% -90%)

        point_t1 = np.zeros(len(self.R))  # State vector of trajectory trajStable in time step arg_p1
        point_t2 = np.zeros(len(self.R))  # State vector of trajectory trajToMove in time step arg_p1
        for i in range(len(self.R)):
            point_t1[i] = trajStable.X[i][arg_p1]
            point_t2[i] = trajToMove.X[i][arg_p1]

        eps = mvn.rvs(cov=np.eye(len(self.R)) * eps)

        for i in range(len(self.R)):
            trajToMove.X[i] += (point_t1 - point_t2)[i] + eps[i]  # Moving of measurements of trajToMove
            trajToMove.Y[i] += (point_t1 - point_t2)[i] + eps[i]  # Moving of true trajectory of trajToMove

        return trajToMove

    def crossNRandomTrajectories(self, N, eps=2):
        """
        Method to cross N Random trajectories. One trajectory cant be moved more than once in one call of this method.
        :param N: num of Trajectories to cross
        :param eps: epsilon variable to make some noise in crossing coordinates
        :return:
        """
        nonMovedTrajectories = []
        for i, traj in enumerate(self.trajectories):
            if not traj.bornedFromAnotherTrajectory:
                nonMovedTrajectories.append(i)
        if (len(nonMovedTrajectories) < N + 1):
            raise Exception("not enough trajectories to cross")
        #startPositions = []
        for i in range(N):
            r_m = np.random.randint(len(nonMovedTrajectories))
            idm = nonMovedTrajectories[r_m]
            trajToMove = self.trajectories[idm]
            nonMovedTrajectories.remove(nonMovedTrajectories[r_m])
            r_s = np.random.randint(len(nonMovedTrajectories))
            trajStable = self.trajectories[nonMovedTrajectories[r_s]]
            self.trajectories[idm] = self.crossTrajectories(trajStable, trajToMove, eps)
            #startPositions.append(self.trajectories[idm])

    def addBornedTrajectory(self, fromTrajectoryID=0, lenght=0, bornedTime=0):
        """
        Method to add Single Trajectory that was borned from another trajectory.
        :param fromTrajectoryID: Index of trajectory form which new trajectory will be borned
        :param lenght: lenght (num of time steps) of this new trajectory
        :param bornedTime: Time in which the trajectory will be borned
        :return:
        """

        z0 = np.zeros(len(self.trajectories[fromTrajectoryID].X))
        z0[0:len(self.R)] = self.trajectories[fromTrajectoryID].X[:2, bornedTime]
        self.addSingleTrajectory(z0, np.random.randint(1000), lenght, bornedTime, True)

    def getRandomNotBornedTrajectoryID(self):
        """
        Method to get index of trajectory that was not borned from another trajectory
        :return: index of random not borned trajectory.
        """
        trajsID = []
        for i, traj in enumerate(self.trajectories):
            if not traj.bornedFromAnotherTrajectory:
                trajsID.append(i)
        r = np.random.randint(len(trajsID))
        return trajsID[r]

    def addNRandomlyBornedTrajectories(self, N):
        """
        Method to add N trajectories that will be borned from another trajectories.
        :param N: Num of new borned trajectories.
        :return:
        """
        for i in range(N):
            id = self.getRandomNotBornedTrajectoryID()
            bornedTime = np.random.randint(self.trajectories[id].ndat * 0.25, self.trajectories[
                id].ndat * 0.75)  # borning time is between 25% and 75% of len of trajectory
            lenght = self.trajectories[id].ndat - bornedTime
            self.addBornedTrajectory(id, lenght, bornedTime)

    def getTrajectoriesIDInTimeT(self, T):
        """
        Method to get index of trajectories that exist at Time T
        :param T: time T
        :return: list of indexes of existing trajectories in time T
        """
        trIDs = []
        for i, traj in enumerate(self.trajectories):
            if traj.startTime <= T and traj.startTime + traj.ndat > T:
                trIDs.append(i)
        return trIDs

    def getAllTrueMeasurements(self):
        """
        Method to get all Measurements of trajectories in one list
        :return: list of measurements of trajectories in every timestep. [timestep][coord (x/y)][mesurements]
        """
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
        """
        Method to get max and min across all timesteps in given coordinate.
        :param measurements: Measurements
        :param coord: coordinate from which to get max and min
        :return: max and min in measurements in given coordinate
        """
        Xmin = min(measurements[0][coord])
        Xmax = max(measurements[0][coord])

        for i in range(len(measurements)):
            if max(measurements[i][coord]) > Xmax:
                Xmax = max(measurements[i][coord])
            if min(measurements[i][coord]) < Xmin:
                Xmin = min(measurements[i][coord])
        return Xmax, Xmin

    def addGlobalClutter(self, NSinglePoints=0, offset=0.001, minMax=None):
        """
        Method to add Poisson Clutter uniformly across whole map.
        :param NSinglePoints: num of time steps from beggining, where no clutter will be added
        :param offset: offset multiplicator to min and max of the map (adding points slightly outside of border)
        :param minMax: [list]; min and max for every axis, where to add the uniformly distributed clutter
        :return: All measurements = clutter + measurements from trajectories
        """
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
        clutterCount = poisson.rvs(self.area_vol * self.lambd,
                                   size=self.ndat)  # Poisson point clutter Poiss (V * lambda), E(X) = V * lambda

        for t in range(self.ndat):
            if t < NSinglePoints:  # no clutter added
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
                global_clutter=False, singlePoints=0, trajectory_clutters=False, crossNTrajectories=0,
                minMaxClutter=None):
        """
        Method to make partially random map automatically.
        :param full_trajectories: Num of trajectories living from time 0 to ndat
        :param short_trajectories: [list]; list of times, the corresponding trajectories will be borned at
        :param borned_trajectories: Num of trajectories that will be borned from another trajectory (from which trajectory is random)
        :param global_clutter: [bool]; True if you want to add global clutter
        :param singlePoints: num of single points for method addGlobalClutter method (num of time steps from beggining without clutter)
        :param trajectory_clutters: Not done yet
        :param crossNTrajectories: How many trajectories will be crossed (random trajectories)
        :param minMaxClutter: min and max for all axis for uniform clutter
        :return: all meeasurements from clutter if global_clutter == True
        """
        self.addNTrajectories(full_trajectories)
        if short_trajectories:
            self.addNShortTrajectories(short_trajectories)
        self.addNRandomlyBornedTrajectories(borned_trajectories)
        self.crossNRandomTrajectories(crossNTrajectories)
        if global_clutter:
            measurements = self.addGlobalClutter(NSinglePoints=singlePoints, minMax=minMaxClutter)
            return measurements
        else:
            return None

    def printTrajectories(self):
        """
        Method to print True trajectories
        :return: plot
        """
        for traj in self.trajectories:
            plt.plot(traj.X[0], traj.X[1], "-")
        plt.show()

    def animate(self, showTrueTrajectories=True, showTrueTrajectoriesMeasurements=True):
        """
        Method to animate map in matplotlib.
        :param showTrueTrajectories: [bool]; if True, True trajectories will be shown
        :param showTrueTrajectoriesMeasurements: if True, True trajectories measurements will be shown
        :return: plot
        """
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
            plt.pause(0.3)


class Radar(MapGenerator):
    def __init__(self, A, Q, R, H, ndat, lambd):
        """
        Class Radar is subclass of MapGenerator. Radar class adds radar, airports (birth points) to the map.
        Radar position and radius can be changed. Radar also computes measuremets (from trajectories and clutter) that are inside radar radius.
        :param A: state transition matrix
        :param Q: process noise covariance matrix
        :param R: observation noise covariance matrix
        :param H: observation matrix
        :param ndat: num of time steps
        :param lambd: clutter intensity
        """
        super().__init__(A, Q, R, H, ndat, lambd)
        self.radarPosition = np.array([0., 0.])
        self.radarRadius = 300

        self.mapMeasurements = []
        self.radarMeasurements = []  # measurements inside radar radius
        self.trajectoriesMeasurements = []

        self.airports = []  # True airports generating targets
        self.borderAirports = []  # "fake" airports on the border of radius, to catch trajectories comming outside of the radar

        self.meas_colors = ["orange", "lime", "indigo", "steelblue"]
        self.traj_colors = ["red", "green", "blue", "dodgerblue"]
        self.model_colors = ["saddlebrown", "black", "magenta", "slategray"]

    def setRadarPosition(self, position):
        self.radarPosition = position

    def setRadarRadius(self, radius):
        self.radarRadius = radius

    def adjustAirports(self):
        targets = []
        for traj in self.trajectories:
            targets.append([traj.X[0][0],traj.X[1][0]])
        targets = np.array(targets)
        print("targets: ", len(targets))
        print("AP: ", len(self.airports))
        targets = np.unique(targets,axis=0)
        for i, traj in enumerate(targets):
            #print(traj)
            self.airports[i].pos = np.append(traj, [0,0])


    def makeRadarMap(self, full_trajectories=1, short_trajectories=None, startFromAirport=False, borned_trajectories=0,
                     global_clutter=False, singlePoints=0, trajectory_clutters=False, crossNTrajectories=0, adjustAirports = False):
        """
        Method to make a random map with radar.
        :param full_trajectories: num of trajectories strating in time 0 to ndat
        :param short_trajectories: [list]; list of times, corresponding trajectories will start at. Num of these trajectories is based on num of elements in the list
        :param startFromAirport: [bool]; If true, z0 for all trajectoies will be on random airport
        :param borned_trajectories: num of trajectories, that will be borned from another
        :param global_clutter: [bool]; if true, map clutter will be added to the map
        :param singlePoints: num of time steps from beggining without clutter
        :param trajectory_clutters:
        :param crossNTrajectories: num of trajectories that will cross another trajectory
        :return:
        """
        if startFromAirport:
            for i in range(full_trajectories):
                z0 = np.random.randint(len(self.airports))
                self.addNTrajectories(1, 0, False, self.airports[z0].pos)
            if short_trajectories:
                for time in short_trajectories:
                    z0 = np.random.randint(len(self.airports))
                    self.addNShortTrajectories([time], self.airports[z0].pos)

        else:
            self.addNTrajectories(full_trajectories)
            if short_trajectories:
                self.addNShortTrajectories(short_trajectories)

        self.addNRandomlyBornedTrajectories(borned_trajectories)
        self.crossNRandomTrajectories(crossNTrajectories)
        #print("make")
        if adjustAirports:
            self.adjustAirports()

        if global_clutter:
            minMax = [self.radarPosition[0] - self.radarRadius, self.radarPosition[0] + self.radarRadius,
                      self.radarPosition[1] - self.radarRadius, self.radarPosition[1] + self.radarRadius]
            self.addGlobalClutter(NSinglePoints=singlePoints, minMax=minMax)

        self.trajectoriesMeasurements = self.getAllTrueMeasurements()
        self.getAllMeasurementsWithinRadarRadius()

    def getAllMeasurementsWithinRadarRadius(self):
        """
        Method to get all measurements inside Radar radius.
        :return: measurements inside radar radius
        """
        if (len(self.allMeasurements) == 0):
            self.allMeasurements = self.getAllTrueMeasurements()
        for t in range(len(self.allMeasurements)):  # time
            radarMeasurements = []
            radarMeasurementsX = []
            radarMeasurementsY = []
            allX = self.allMeasurements[t][0]
            allY = self.allMeasurements[t][1]

            for m_id in range(len(allX)):
                if (allX[m_id] - self.radarPosition[0]) ** 2 + (
                        allY[m_id] - self.radarPosition[1]) ** 2 < self.radarRadius ** 2:
                    radarMeasurementsX.append(allX[m_id])
                    radarMeasurementsY.append(allY[m_id])
            radarMeasurements.append(radarMeasurementsX)
            radarMeasurements.append(radarMeasurementsY)
            self.radarMeasurements.append(radarMeasurements)
        return self.radarMeasurements

    def addAirport(self, pos, cov, weight):
        """
        Method to add airport to the map.
        :param pos: position of an airport
        :param cov: Covariance matrix that will be given to borned targets
        :param weight: weight that will be given to borned targets
        :return:
        """
        self.airports.append(Airport(pos, cov, weight))

    def addNRandomAirports(self, N, cov, weight):
        """
        Method to add N random airports on the map.
        :param N: num of airports to add
        :param cov: Covariance matrices for the airport - gives to targets
        :param weight: weight to the airport - gives to targets
        :return:
        """
        for i in range(N):
            r = np.random.randint(0, self.radarRadius * 0.9)
            rad = np.random.randint(0, 360)
            x = math.sin(math.radians(rad)) * r + self.radarPosition[0]
            y = math.cos(math.radians(rad)) * r + self.radarPosition[1]
            self.addAirport([x, y, 0, 0], cov, weight)

    def addNBorderAirports(self, N, cov, weight):
        """
        Method to add N border airports - airports located on the border of radar to detect targets from outside of radar.
        :param N: num of border airports to add
        :param cov: Covariance matrix for border airport - gives to targets
        :param weight: weight for border airports - gives to targets
        :return:
        """
        for i in range(N):
            rad = 360 / N * i
            x = math.sin(math.radians(rad)) * self.radarRadius + self.radarPosition[0]
            y = math.cos(math.radians(rad)) * self.radarRadius + self.radarPosition[1]
            self.borderAirports.append(Airport([x, y, 0, 0], cov, weight))

    def getAirports(self):
        return self.airports + self.borderAirports

    def animateRadarTMP(self, showTrueTrajectories=True, showTrueTrajectoriesMeasurements=True, showRadar=True,
                        showMapClutter=True, showRadarClutter=True, showBorderAirports=True,
                        showAirports=True):
        fig, ax = plt.subplots(figsize=(10, 10))
        for t in range(self.ndat):
            ax.cla()
            ax.set_xlim(self.radarPosition[0] - self.radarRadius - 20, self.radarPosition[0] + self.radarRadius + 20)
            ax.set_ylim(self.radarPosition[1] - self.radarRadius - 20, self.radarPosition[1] + self.radarRadius + 20)
            if showTrueTrajectories:
                for i, traj in enumerate(self.trajectories):
                    plt.plot(traj.X[0], traj.X[1], "-", color=self.traj_colors[i % len(self.traj_colors)])
            if showMapClutter:
                plt.plot(self.allMeasurements[t][0], self.allMeasurements[t][1], "*", color="gold")
            if showRadarClutter:
                plt.plot(self.radarMeasurements[t][0], self.radarMeasurements[t][1], "*", color="orange")
            if showTrueTrajectoriesMeasurements:
                plt.plot(self.trajectoriesMeasurements[t][0], self.trajectoriesMeasurements[t][1], "*r")
            if showRadar:
                radar = plt.Circle((self.radarPosition), self.radarRadius, color="b", fill=False)
                plt.plot(self.radarPosition[0], self.radarPosition[1], "*b")
                ax.add_patch(radar)
            if showBorderAirports:
                for airPort in self.borderAirports:
                    confidence_ellipse(airPort.pos[:2], airPort.cov, ax=ax, edgecolor="red")
            if showAirports:
                for airPort in self.airports:
                    confidence_ellipse(airPort.pos[:2], airPort.cov, ax=ax, edgecolor="black")
            plt.pause(0.01)

    def animateRadar(self, t, ax,laserCounter,showTrueTrajectories=True, showTrueTrajectoriesMeasurements=True, showRadar=True,
                     showMapClutter=True, showRadarClutter=True, showBorderAirports=True,
                     showAirports=True):
        """
        Method to animate radar map.
        :param t: current time step
        :param ax: ax for plot
        :param showTrueTrajectories: [booo]; if true True trajecotries tracks will be shown
        :param showTrueTrajectoriesMeasurements: [bool]; if true, shows measuremets from targets
        :param showRadar: [bool]; if true, radar radius and center will be shown
        :param showMapClutter: [bool]; if true, shows map clutter
        :param showRadarClutter: [bool]; if true, show clutter inside radar radius
        :param showBorderAirports: [bool]; if true, shows border "fake" airports
        :param showAirports: [bool]; if true, shows generated airports inside radar
        :return:
        """

        ax.cla()
        ax.set_facecolor("black")
        ax.set_xlim(self.radarPosition[0] - self.radarRadius - 20,
                    self.radarPosition[0] + self.radarRadius + 20)
        ax.set_ylim(self.radarPosition[1] - self.radarRadius - 20,
                    self.radarPosition[1] + self.radarRadius + 20)
        if showTrueTrajectories:
            for i, traj in enumerate(self.trajectories):
                ax.plot(traj.X[0][:t], traj.X[1][:t], "-", color="white",
                        label=f"True trajectory, T={traj.startTime}")
        if showMapClutter:
            ax.plot(self.allMeasurements[t][0], self.allMeasurements[t][1], "*", color="grey", label="Map Clutter", markersize=1)
        # if showRadarClutter:
        #     ax.plot(self.radarMeasurements[t][0], self.radarMeasurements[t][1], "*", color="grey", label="Radar Clutter")
        if showTrueTrajectoriesMeasurements:
            ax.plot(self.trajectoriesMeasurements[t][0], self.trajectoriesMeasurements[t][1], "*r",
                   label="Trajectory measurement")

        if showRadar:
            from matplotlib.patches import Circle, Polygon, Wedge
            radar = plt.Circle((self.radarPosition), self.radarRadius, color="lime", fill=False, label="Radar")
            radar1 = plt.Circle((self.radarPosition), self.radarRadius/5, color="lime", fill=False)
            radar2 = plt.Circle((self.radarPosition), self.radarRadius / 5*4, color="lime", fill=False)
            radar3 = plt.Circle((self.radarPosition), self.radarRadius / 5*3, color="lime", fill=False)
            radar4 = plt.Circle((self.radarPosition), self.radarRadius / 5*2, color="lime", fill=False)
            # ax.plot(self.radarPosition[0], self.radarPosition[1], "*b")
            ax.add_patch(radar)
            ax.add_patch(radar1)
            ax.add_patch(radar2)
            ax.add_patch(radar3)
            ax.add_patch(radar4)
            w = Wedge((self.radarPosition), self.radarRadius, 45+(t+laserCounter)*5, 75+(t+laserCounter)*5, color="lime", alpha=0.5)
            ax.add_patch(w)
        if showBorderAirports:
            for airPort in self.borderAirports:
                confidence_ellipse(airPort.pos[:2], airPort.cov, ax=ax, edgecolor="red")
        if showAirports:
            for airPort in self.airports:
                confidence_ellipse(airPort.pos[:2], airPort.cov, ax=ax, edgecolor="blue", label="Airport")
