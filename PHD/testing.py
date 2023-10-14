import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.stats import poisson, uniform
import matplotlib.pyplot as plt


a=[[1,1,1],[2,2,2],[3,3,3],[4,4,4]]
b=[[10,10,10],[20,20,20],[30,30,30],[40,40,40]]

tr=[]
tr.append(a)
tr.append(b)
tr=np.array(tr)
#print(tr[0][0])
#measurements=np.zeros((2,len(tr[0][0])))
#print(measurements)
#for j in range(len(tr[0][0])):
#    measurements[0][j]=(tr[:,0,j])
 #   measurements[1][j]=(tr[:,1,j])

#print(measurements)

b=[1,2,3]
# print(max(b))
a=[4,5]

print(a+b)