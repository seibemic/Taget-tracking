import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal as mvn
from scipy.stats import poisson
from confidence_ellipse import confidence_ellipse
from CPHD import CPHD
from MAP2 import Radar
from scipy.special import binom
from scipy.special import perm
from itertools import combinations
class CPHD_map:
    def __init__(self, radar, ps, pd, card, Nmax, birthCard):
        self.radarMap = radar
        self.F = radar.A
        self.H = radar.H
        self.Q = radar.Q
        self.R = radar.R
        self.ps = ps
        self.pd = pd
        self.cphds = []
        self.card=card
        self.Nmax=Nmax
        self.birthCard= birthCard
        self.v = 0
        #36self.birthIntensity = birthIntensity
        self.J_gamma = len(self.radarMap.getAirports())
        self.cphdsToPlot = []
        self.measurements = radar.getAllMeasurementsWithinRadarRadius()


    def calculatePredictCardinality(self, n):
        res=0
        for j in range(n):
            innerRes=0
            for l in range(j, self.Nmax):
                innerRes+=binom(l,j) * self.card[l] * self.ps**j * (1-self.ps)**(l-j)
            res += self.birthCard[n-j] * innerRes
        return res

    def predictionForExistingTargets(self):
        for target in self.cphds:
            target.predict(self.ps, self.F, self.Q)

    def predictionForBirthTargets(self):
        for airport in self.radarMap.getAirports():
            w_k = airport.weight
            m_k = np.array(airport.pos)
            P_k = airport.cov
            self.cphds.append(CPHD(w_k, m_k, P_k))
    def predict(self):
        """ cardinality (23)"""
        for n in range(self.Nmax):
            self.card[n]=self.calculatePredictCardinality(n)

        """ v_k|k-1 (24)"""
        self.predictionForExistingTargets()
        self.predictionForBirthTargets()

    def getWeights(self):
        w=[]
        for target in self.cphds:
            w.append(target.w)
        return np.array(w)

    def elementarySymmetricPolynomial(self, j, Z):
        if j == 0:
            return 1
        comb = combinations(Z, j)
        res=0
        for c in comb:
            res += np.prod(np.array(c))
        return res

    def getq(self,z):
        q = []
        for target in self.cphds:
            eta = self.H @ target.m
            S = self.H @ target.P @ self.H.T + self.R
            q.append(mvn(eta,S).pdf(z))
        return np.array(q)
    def LAMBDA(self, w, Z):
        lambdaSet = []
        for z in Z:
            kappa = len(Z)/self.radarMap.radarRadius**2
            q = self.getq(z)
            lambdaSet.append(kappa * self.pd * w.T @ q)
        return lambdaSet
    def psi(self, w, Z, n, u):
        res = []
        w = self.getWeights()
        for j in range(min(len(Z), n)):
            pK = poisson.pmf(len(Z), len(Z) - j)
            res.append( (len(Z)-j) * pK * perm(n, j+u, exact=True) * (1-self.pd)**(n-(j+u)) / np.inner(1, w)**(j+u) * )

    def updateCardinality(self):

    def updateComponents(self):
    def update(self):

        print("TODO")

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
    r.addNBorderAirports(36, airportCov, 0.1)
    # seed = 123
    seed=np.random.randint(1000)
    r.addSingleTrajectory([-150, 350, 2, -2], seed, ndat, 0, False)
    r.makeRadarMap(full_trajectories=2, short_trajectories=None, global_clutter=True, startFromAirport=True,
                   borned_trajectories=0)
     # r.makeRadarMap(full_trajectories=2, short_trajectories=[50], global_clutter=False, startFromAirport=False)
    Nmax=10
    card = np.ones(Nmax)*1/10
    filter = CPHD_map(r, Ps, Pd,card, Nmax,card )
    filter.elementarySymmetricPolynomial(2,[1,2,3,4])
    #filter.run()

