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
import math
class CPHD_map:
    def __init__(self, radar, ps, pd, card, Nmax, birthCard):
        self.radarMap = radar
        self.area_vol = (self.radarMap.radarRadius*2)**2
        self.F = radar.A
        self.H = radar.H
        self.Q = radar.Q
        self.R = radar.R
        self.ps = ps
        self.pd = pd
        self.cphds = []
        self.card=card
        self.Nmax=Nmax
        self.birthCard = birthCard
        self.v = 0
        #36self.birthIntensity = birthIntensity
        self.J_gamma = len(self.radarMap.getAirports())
        self.cphdsToPlot = []
        self.measurements = radar.getAllMeasurementsWithinRadarRadius()

        self.cphdsToPlot = []
        self.T = 10e-5
        self.U = 4
        self.J_max = 10

        self.model_colors = ["saddlebrown", "black", "magenta", "slategray"]

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


    """ ---------------- UPDATE ----------------"""
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

    def elementarySymmetricPolynomial2(self, j, Z):
        if j == 0:
            return 1
        if len(Z) < j:
            return 0

        return self.elementarySymmetricPolynomial2(j, Z[1:]) + Z[0] * self.elementarySymmetricPolynomial2(j-1, Z[1:])
    def q(self, z):
        Q = []
        for target in self.cphds:
            Q.append(mvn.pdf(z,target.eta,target.S))
        # for j in range(len(self.cphds)):
        #     S = self.H @ self.cphds[j].P @ self.H.T + self.R
        #     eta = self.H @ self.cphds[j].m
        #     Q.append(mvn.pdf(z,eta,S))
        return np.array(Q)
    def LAMBDA(self, w, Z):
        res = []
        w_ = self.area_vol / self.radarMap.lambd * self.pd * w
        for z in Z:
            res.append( w_.T @ self.q(z))
        return np.array(res)
    def PSI(self, u, w, Z, n):
        res = 0
        lambd = self.LAMBDA(w, Z)
        for j in range(min(len(Z), n)):
            pK = poisson.pmf(int(self.area_vol * self.radarMap.lambd), len(Z) - j)+0.0001
            # print("PSI: ",((len(Z)-j) * pK * perm(n, j+u, exact=True)
            #        * (1-self.pd)**(n-(j+u)) / np.sum(w)**(j+u)))
            # print(self.area_vol, self.radarMap.lambd, self.area_vol*self.radarMap.lambd)
            # print(len(Z)-j)
            # print("pk: ", pK)
            # print("perm: ", perm(n, j+u, exact=True))
            # print("zbytek: ", (1-self.pd)**(n-(j+u)) / np.sum(w)**(j+u))
            res += (((len(Z)-j) * pK * perm(n, j+u, exact=True)
                   * (1-self.pd)**(n-(j+u)) / np.sum(w)**(j+u))
                   * self.elementarySymmetricPolynomial2(j, lambd))

        return res
    def updateComponents(self):
        for target in self.cphds:
            target.updateComponents(self.H, self.R)
    def getNewHypotheses(self,Z):
        denominator = 0
        w = self.getWeights()
        for i in range(self.Nmax):
            denominator += self.PSI(0,w,Z,i) * self.card[i]
        newTargets = []
        for l, z in enumerate(Z):
            # print("new hypotheses l: ",l)
            nominator = 0
            Z_copy = Z.copy()
            np.delete(Z_copy, l, axis=0)
            print("nom")
            for i in range(self.Nmax):
                # print("new hypotheses i: ", i)
                nominator += self.PSI(1, w, Z_copy, i) * self.card[i]

            psi_res = nominator / denominator * self.area_vol / self.radarMap.lambd
            # print("psi res: ", psi_res)
            # print(nominator / denominator)
            # print("nom: ", nominator)
            # print("denom: ", denominator)
            for target in self.cphds:
                w_ = self.pd * target.w * mvn.pdf(z, target.eta, target.S) * psi_res
                m = target.m + target.K @ (z - target.eta)
                newTargets.append(CPHD(w_, m, target.P))
        return newTargets, denominator

    def updateOld(self, denominator, Z):
        nominator = 0
        w = self.getWeights()
        for i in range(self.Nmax):
            nominator += self.PSI(1, w, Z, i) * self.card[i]
        res = nominator / denominator
        for target in self.cphds:
            target.update(self.pd, res)

    def updateCardinality(self, Z):
        w = self.getWeights()
        for i in range(self.Nmax):
            self.card[i] *= self.PSI(0, w, Z, i)
        self.card /= sum(self.card)

    def mergeOldAndNewTargets(self, newTargets):
        self.cphds.extend(newTargets)



    def pruneByMaxWeight(self, w):
        filters_to_stay = []
        for filter in self.cphds:
            if filter.w > w:
                filters_to_stay.append(filter)
        self.cphds = filters_to_stay

    def argMax(self, filters):
        maX = 0
        argmaX = 0
        for i, filter in enumerate(filters):
            if filter.w > maX:
                maX = filter.w
                argmaX = i
        return argmaX

    def getCPHDsToPlot(self):
        self.cphdsToPlot=[]
        for filter in self.cphds:
            if filter.w > 0.5:
                self.cphdsToPlot.append(filter)
        return self.cphdsToPlot
    def mergeTargets(self):
        filters_to_stay = []
        mixed_filters = []
        for filter in self.cphds:
            if filter.w > self.T:
                filters_to_stay.append(filter)

        while len(filters_to_stay) != 0:
            j = self.argMax(filters_to_stay)
            L = []  # indexes
            for i in range(len(filters_to_stay)):
                if ((filters_to_stay[i].m - filters_to_stay[j].m).T @
                    np.linalg.inv(filters_to_stay[i].P) @ (filters_to_stay[i].m - filters_to_stay[j].m)) < self.U:
                    L.append(i)
            # print(len(L))
            w_mix = 0
            for t_id in L:
                w_mix += filters_to_stay[t_id].w
            m_mix = np.zeros(4)
            for t_id in L:
                m_mix += filters_to_stay[t_id].w * filters_to_stay[t_id].m
            m_mix /= w_mix
            P_mix = np.zeros_like(filters_to_stay[0].P, dtype="float64")
            for t_id in L:
                P_mix += filters_to_stay[t_id].w * (
                            filters_to_stay[t_id].P + np.outer((m_mix - filters_to_stay[t_id].m),
                                                               (m_mix - filters_to_stay[t_id].m).T))
            P_mix /= w_mix
            mixed_filters.append(CPHD(w_mix,m_mix,P_mix))
            removed = np.delete(filters_to_stay, L)
            filters_to_stay = removed.tolist()

        if len(mixed_filters) > self.J_max:
            self.cphds = mixed_filters
            self.pruneByMaxWeight(0.1)
        else:
            self.cphds = mixed_filters
    def update(self, Z):
        print("update Components")
        self.updateComponents()
        Z=np.array(Z).T
        print("new Hypotheses")
        newTargets, denominator = self.getNewHypotheses(Z)
        print("update Old")
        self.updateOld(denominator, Z)
        self.updateCardinality(Z)
        self.mergeOldAndNewTargets(newTargets)

    def run(self):
        print("run")
        fig, ax = plt.subplots(figsize=(10, 10))
        for t in range(self.radarMap.ndat):
            print("t: ", t)
            print("predict")
            self.predict()
            print("update")
            self.update(self.measurements[t])
            print("merge")
            self.mergeTargets()
            self.getCPHDsToPlot()
            print("num of cphds: ", len(self.cphds))
            print("num of cphds to plot: ", len(self.cphdsToPlot))

            self.radarMap.animateRadar(t, ax)
            for i, filter in enumerate(self.cphdsToPlot):
                ax.plot(filter.m[0], filter.m[1], "+", color=self.model_colors[i % len(self.model_colors)], label="CPHD")
                confidence_ellipse([filter.m[0], filter.m[1]], filter.P, ax=ax,
                                   edgecolor=self.model_colors[i % len(self.model_colors)])
                # print(filter.P.diagonal())
            # print(filter.P)
            # plt.plot(filter)
            ax.legend(loc='center left', bbox_to_anchor=(0.9, 0.5))
            plt.pause(0.3)

        #for z in self.measurements:
        #    print(z)
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
    lambd = 0.00005
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
    Nmax=5
    card = np.ones(Nmax)*1/Nmax
    filter = CPHD_map(r, Ps, Pd,card, Nmax,card.copy() )
    filter.run()
    # filter.elementarySymmetricPolynomial(2,[1,2,3,4])
    #filter.run()

