from itertools import product
import itertools
import numpy as np
def product(hyp, repeat=1):
    # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
    #pools = [tuple(pool) for pool in args] * repeat
    pools = hyp
    #print("pools: ", pools)
    result = [[]]
    for pool in pools:
        temp_result = []
        for x in result:
            for y in pool:
                temp_result.append(x + [y])
        result = temp_result

    #print("first res: ", result)
    for res in result:
        print(res)

hypotheses80 = [[1, 2, 3, 4, 5, 6], [1, 2, 3], [1, 2, 3]]
#res=product(hypotheses2)


def product2(hyp, repeat=1):
    # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
    #pools = [tuple(pool) for pool in args] * repeat
    pools = hyp
    # print("pools: ", pools)
    result = [[]]
    for pool in pools:
        temp_result = []
        for x in result:
            for y in pool:
                temp_result.append(x + [y])
        result = temp_result

    for res in result:
        print(res)

hypotheses4 = [[[-1,1],[0,2],[1,3],[-1,4],[0,5],[1,6]], [[-1,1],[0,2],[1,3]], [[-1,1],[0,2],[1,3]]]
#res=product2(hypotheses4)

#res=itertools.product(*hypotheses4)

def getHypothesis(ids):

    combinations = list(itertools.product(*ids))

    unique_combinations = []
    for c in combinations:
        x = [element[0] for element in c]
        if all(x.count(element) <=1 or element == -1 for element in x):
            unique_combinations.append([element[1] for element in c])

    return np.array(unique_combinations)
# Example usage:
hypotheses2 = [[[-1,0],[0,1],[1,2],[-1,3],[0,4],[1,5]], [[-1,1],[0,2],[1,3]], [[-1,1],[0,2],[1,3]]]

combinations2 = getHypothesis(hypotheses2)

# print("Combinations 2:", combinations2)
# print("Total combinations 2:", len(combinations2))

def getHypothesis2( ids):
    # Get all combinations of elements from the sets
    combinations = list(itertools.product(*ids))

    # print(combinations)
    # Filter out combinations with repeated elements

    for c in combinations:
        print("----", c, "------")
        for element in c:
            print("     ", element)
    print("**********************************")
    unique_combinations = []
    for c in combinations:
        if all(c.count(element) <= 1 or element == -1 for element in c):
            unique_combinations.append(c)
    # zeros = tuple(np.zeros(len(betas)))
    # unique_combinations.append(zeros)
    return np.array(unique_combinations)

hypotheses3 = [[1, 2, 3, 1,2,3], [1, 2, 3], [1, 2, 3]]
res= getHypothesis2(hypotheses3)

def getHypothesis3(hypotheses):
    # Extract the first elements of each tuple in the inner lists
    first_elements = [set(h[0] for h in hypothesis) for hypothesis in hypotheses]

    # Get all combinations of elements from the sets
    combinations = list(itertools.product(*first_elements))

    # Filter out combinations with repeated elements
    unique_combinations = []
    for c in combinations:
        if all(c.count(element) <= 1 or element == -1 for element in c):
            # Extract the corresponding elements from the original hypotheses
            result = [next(h[1] for h in hypotheses[i] if h[0] == j) for i, j in enumerate(c)]
            unique_combinations.append(result)

    return np.array(unique_combinations)
# Example usage
# hypotheses4 = [[(-1,1),(0,2),(1,3),(-1,4),(0,5),(1,6)], [(-1,1),(0,2),(1,3)], [(-1,1),(0,2),(1,3)]]
# hypotheses5 = [[(-1, 0), (0, 1), (1, 2), (-1, 3), (0, 4), (1, 5)], [(-1, 0), (0, 1), (1, 2)], [(-1, 0), (0, 1), (1, 2)]]
# result = getHypothesis3(hypotheses5)
# print("len: ", len(result))
# print(result)


def getHypothesis4(hypotheses):
    # Extract the first elements of each tuple in the inner lists
    first_elements = [h[0] for h in hypotheses]

    # Get all combinations of elements from the sets based on the first elements
    combinations = list(itertools.product(*first_elements))

    # Extract the corresponding tuples from the original hypotheses
    unique_combinations = [tuple(hypotheses[i][first_elements[i].index(element)] for i, element in enumerate(c)) for c in combinations]

    return np.array(unique_combinations)

# Example usage
hypotheses3 = [[(-1, 0), (0, 1), (1, 2), (-1, 3), (0, 4), (1, 5)], [(-1, 0), (0, 1), (1, 2)], [(-1, 0), (0, 1), (1, 2)]]
result = getHypothesis4(hypotheses3)
# print(result)

a=[1,2,2,3,3,4,5,6]
best = 4
a =set(a)
j2 = [x for x in a if x < best]
# print("x: ",j2)
# indexes = [2, 3, 5]
# for index in sorted(indexes, reverse=True):
#     del a[index]
#
# print(a)

w = np.array([0.15349248,0.12577309,0.2494964,0.049825754,0.0124701755])
K=2
K_best = np.argsort(-w)[:min(K, len(w))]
#print(K_best)

z_ids = [[[-1, 0], [0, 1], [1, 2], [-1, 3], [0, 4], [1, 5]], [[-1, 0], [0, 1], [1, 2]],[[-1, 0], [0, 1], [1, 2],[2,3]]]
indexes = [0,1,2]
new_z = []
for j in range(len(z_ids)):
    temp_z = []
    for i, local_hyp in enumerate(z_ids[j]):
        if local_hyp[1] not in indexes:
            temp_z.append(z_ids[j][i])

    if len(temp_z) != 0:
        print(temp_z)
        temp_z = np.array(temp_z)
        temp_z.T[1] = np.arange(0,len(temp_z.T[1]),1)
        temp_z = temp_z.tolist()
        # print(temp_z)
        new_z.append(temp_z)
    #     del z_ids[j]
print(new_z)

print(10**(-3))
print(1e-3)
