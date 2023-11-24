import numpy as np

class GlobalHypothese:
    def __init__(self):
        self.hypotheses_mat = [[]]
        self.weights = []

    def addMatrix(self, mat):
        self.hypotheses_mat = mat
