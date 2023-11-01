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
z= [i for i in range(500)]
def elementarySymmetricPolynomial( j, Z):
    if j == 0:
        return 1
    comb = combinations(Z, j)
    res = 0
    for c in comb:
        res += np.prod(np.array(c))
    return res
t1 = datetime.now().microsecond
print(elementarySymmetricPolynomial(2,z))
t2 =datetime.now().microsecond
print("alg 1: ", t2-t1)
print(2+3+4+5+6+8+10+12+15+20)


def elementary_symmetric_function_recursive(roots, k):
    if k == 0:
        return 1
    if len(roots) < k:
        return 0

    return elementary_symmetric_function_recursive(roots[1:], k) + roots[0] * elementary_symmetric_function_recursive(roots[1:], k - 1)

# Example usage:
roots = [1, 2, 3, 4,5]
k = 2
t1 = datetime.now().microsecond
result = elementary_symmetric_function_recursive(z, k)
t2 = datetime.now().microsecond
print("alg 2: ", t2-t1)
print(f"The {k}-th elementary symmetric function is {result}")