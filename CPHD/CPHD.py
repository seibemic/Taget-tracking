class CPHD:
    def __init__(self,w,m,P):
        self.w = w
        self.m = m
        self.P = P

    def predict(self, ps, A, Q):
        self.w = ps * self.w
        self.m = A @ self.m
        self.P = Q + A @ self.P @ A.T