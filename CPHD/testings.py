import math

from scipy.stats import poisson
import numpy as np
from itertools import combinations
from datetime import datetime
a = [[[123.71589969726391, -31.469102634455552, 264.7997679960376, 71.92610890984429, 316.32034867420117], [202.6007032948559, -30.953411288436598, 217.49252804721118, 214.3961117116944, -83.04182921990635]], [[119.4620730389617, -32.99759199893193, 322.7739191002442], [190.11647151795376, -42.891720267768264, 166.22250422558108]]]
# print(a[0])
# a=np.array(a[0])
# print(a.shape)
# print(a.T)
# print(a.T.shape)
# b=a.T
# print(b)
# print(np.delete(b, 1,axis=0))

z = [1,2,3,4,5]
z= [i for i in range(10)]
l = [6.90809070e+09, 8.59167138e+09, 3.07280200e+05, 5.72166648e+05,
 1.50557020e+06, 2.25355452e+01, 8.47223024e-06, 5.60593052e+06,
 4.02632267e-27, 2.79664543e-11, 8.67759870e-15, 3.86157350e+06,
 2.33331100e+03, 4.72896147e-11, 1.80600236e-12, 2.03953286e+09,
 7.89402991e+03, 1.34369871e-43]
def elementarySymmetricPolynomial( j, Z):
    if j == 0:
        return 1
    comb = combinations(Z, j)
    res = 0
    i=0
    for c in comb:
        i+=1
        res += np.prod(np.array(c))
    print("len: ", i)
    return res
t1 = datetime.now().microsecond
# print(elementarySymmetricPolynomial(2,z))
# t2 =datetime.now().microsecond
# print("alg 1: ", t2-t1)
# print(2+3+4+5+6+8+10+12+15+20)

print(elementarySymmetricPolynomial(len(l),l))
X=["x1", "x2", "x3","x4"]
C= combinations(X,3)
for c in C:
    print(c)
def elementary_symmetric_function_recursive(roots, k):
    if k == 0:
        return 1
    if len(roots) < k:
        return 0

    return elementary_symmetric_function_recursive(roots[1:], k) + roots[0] * elementary_symmetric_function_recursive(roots[1:], k - 1)

# Example usage:
# roots = [1, 2, 3, 4,5]
# k = 2
# t1 = datetime.now().microsecond
# result = elementary_symmetric_function_recursive(z, k)
# t2 = datetime.now().microsecond
# print("alg 2: ", t2-t1)
# print(f"The {k}-th elementary symmetric function is {result}")

# t1 = datetime.now().microsecond
# for i in range(len(z)):
#     temp=z[0]
#     z[:-1] = z[1:]
#     print(z[:-1])
    # z[-1] = temp
# t2 = datetime.now().microsecond
# print("alg 1: ", t2-t1)
# t1 = datetime.now().microsecond
# for i in range(len(z)):
#     Z_copy = z.copy()
#     np.delete(Z_copy, i, axis=0)
# t2 = datetime.now().microsecond
# print("alg 2: ", t2-t1)

w=np.array([0.03846154, 0.03846154, 0.02564103, 0.02564103, 0.02564103, 0.02564103,
 0.02564103, 0.02564103, 0.02564103, 0.02564103, 0.02564103, 0.02564103])
print(50/1.25e-5)
print(600**2/0.005)
print(w*600**2/0.005*0.95)
print(np.arange(10,0,-1)/10.)

