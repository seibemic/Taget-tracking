import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.stats import poisson, uniform


class trajectory():

    def __init__(self, A, Q, H, R, z0, seed=123, ndat=200):
        self.ndat = ndat
        self.seed = seed
        self.A = A
        self.Q = Q
        self.H = H
        self.R = R
        self.m0 = z0
        self.X = np.zeros(shape=(self.A.shape[0], self.ndat))
        self.Y = np.zeros(shape=(self.H.shape[0], self.ndat))
        self._simulate()

    def _simulate(self):
        np.random.seed(self.seed)

        x = self.m0
        for t in range(self.ndat):
            x = self.A.dot(x) + mvn.rvs(cov=self.Q)
            y = self.H.dot(x) + mvn.rvs(cov=self.R)
            self.X[:, t] = x.flatten()
            self.Y[:, t] = y.flatten()

    def addClutter(self, area_vol, num_single_points, clutter_size):
        self.Y_with_clutter = []

        for t in range(self.ndat):
            if t < num_single_points:
                Yt = np.zeros((2, 1))
                Yt[:, 0] = self.Y[:, t]
                self.Y_with_clutter.append(Yt)
                continue
            Yt = np.zeros((2, clutter_size[t] + 1))
            Yt[:, 0] = self.Y[:, t]
            Yt[0, 1:] = np.random.uniform(low=self.Y[0, t] - area_vol / 2, high=self.Y[0, t] + area_vol / 2,
                                          size=clutter_size[t])
            Yt[1, 1:] = np.random.uniform(low=self.Y[1, t] - area_vol / 2, high=self.Y[1, t] + area_vol / 2,
                                          size=clutter_size[t])
            self.Y_with_clutter.append(Yt)


def crossTrajectories(trajStable, trajToMove, eps):
    arg_p1 = np.random.randint(len(trajStable.X[1]) * .3, len(trajStable.X[1]) * .9)
    #print(arg_p1)

    point_t1 = np.array([trajStable.X[0][arg_p1], trajStable.X[1][arg_p1]])
    point_t2 = np.array([trajToMove.X[0][arg_p1], trajToMove.X[1][arg_p1]])

    a, b = mvn.rvs(cov=np.array(np.diag([eps, eps])))
    trajToMove.X[0] += (point_t1 - point_t2)[0] + a
    trajToMove.X[1] += (point_t1 - point_t2)[1] + b

    trajToMove.Y[0] += (point_t1 - point_t2)[0] + a
    trajToMove.Y[1] += (point_t1 - point_t2)[1] + b

    return trajToMove


def getClutter(ndat, area_vol, lambd, trajs, c):
    clutterSize = []
    for t in range(ndat):
        cnts = []
        for i, tr1 in enumerate(trajs):
            p = []
            for tr2 in trajs:
                p.append(mvn([tr1.Y[0][t], tr1.Y[1][t]], cov=c*area_vol * np.eye(2)).pdf([tr2.Y[0][t], tr2.Y[1][t]]))
            prob = (p[i] / np.sum(p))
            cnts.append(poisson.rvs(area_vol ** 2 * lambd * prob, size=1)[0])
        clutterSize.append(cnts)

    return np.array(clutterSize)

def generateMapClutter(Xmin,Xmax, Ymin, Ymax, lambd, ndat, n_objects=1):
    area=abs(Xmax-Xmin)*abs(Ymax-Ymin)
    clutterCount=poisson.rvs(area*lambd,size=ndat)
    
    clutter=[]
    
    for t in range(ndat):
        C=np.zeros((2, max(clutterCount[t],n_objects)))
        C[0,0:] = np.random.uniform(low=Xmin*(1-lambd), high=Xmax*(1+lambd), size=max(clutterCount[t],n_objects))
        C[1,0:] = np.random.uniform(low=Ymin*(1-lambd), high=Ymax*(1+lambd), size=max(clutterCount[t],n_objects))
        
        clutter.append(C)
    return clutter
        
        
        
    
    
    