import numpy as np

a = [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]
b = [[10, 10, 10], [20, 20, 20], [30, 30, 30], [40, 40, 40]]

tr = []
tr.append(a)
tr.append(b)
tr = np.array(tr)
# print(tr[0][0])
# measurements=np.zeros((2,len(tr[0][0])))
# print(measurements)
# for j in range(len(tr[0][0])):
#    measurements[0][j]=(tr[:,0,j])
#   measurements[1][j]=(tr[:,1,j])

# print(measurements)

b = [1, 2, 3]
# print(max(b))
a = [4, 5]

print(a + b)

c = [1, 2, 3, 4, 5]
c += [6]
print(c)
for el in c:
    print("remove ", el)
    c.remove(el)
print(np.eye(2))

from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
dym = np.zeros(1000)


def fibonacci(n):
    # Taking 1st two fibonacci numbers as 0 and 1
    f = [0, 1]

    for i in range(2, n + 1):
        f.append(f[i - 1] + f[i - 2])
    return f[n]


a=np.array([1,2,3])
b=np.array([3,4,5])

print("a * b = \n", a*b.T)
print("outer = \n", np.outer(a,b))
print("a*b + outer = \n",  a*b.T + np.outer(a,b))
#t=mvn.pdf(z, eta, S)
#print("T: ", t)


