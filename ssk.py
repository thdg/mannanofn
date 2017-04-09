from math import sqrt
from functools import lru_cache
import numpy as np
import sys


@lru_cache(maxsize=None)
def kh(s, t, n, l):
    @lru_cache(maxsize=None)
    def kmm(n, si, ti):
        if n == 0:
            return 1
        if min(si, ti) < n:
            return 0
        if s[si-1] == t[ti-1]:
            return l * (kmm(n, si, ti-1) + l * km(n-1, si-1, ti-1))
        else:
            return l * kmm(n, si, ti-1)

    @lru_cache(maxsize=None)
    def km(n, si, ti):
        if n == 0:
            return 1
        if min(si, ti) < n:
            return 0
        return l*km(n, si-1, ti) + kmm(n, si, ti)

    @lru_cache(maxsize=None)
    def k(n, si, ti):
        if min(si, ti) < n:
            return 0
        return k(n, si-1, ti) + sum(km(n-1, si-1, j) for j in range(ti) if t[j] == s[si-1]) * l**2

    return k(n, len(s), len(t))


def ssk(s, t, n, l, ss=None, tt=None):
    if ss and tt:
        return kh(s, t, n, l) / sqrt(ss*tt)
    k = kh(s, t, n, l)
    if k == 0: return k
    return k / sqrt(kh(s, s, n, l) * kh(t, t, n, l))


# build kernel matrix of string list s and string list t with ssk
def build_gram_matrix(sList, tList, l, n, K=ssk):
    lenS = len(sList)
    lenT = len(tList)
    # optimize calculation of kernel gram matrix when sList equals to tList
    # in our case, this is to save calculation for kernel gram matrix of training data
    if sList is tList:  # sList is the same object as tList
        gramMat = np.eye(lenS, lenT, dtype=np.float64)

        for i in range(lenS):
            for j in range(i+1, lenT):
                gramMat[i][j] = gramMat[j][i] = K(sList[i], tList[j], n, l)
            # print("gramMat[{}][{}] = {}".format(i, j, gramMat[i][j]))
            # without the two list equal to one another, we have to calculate every element for gram matrix
            # in our case, this is for kernel gram matrix of training data and test data
    else:
        gramMat = np.zeros((lenS, lenT), dtype=np.float64)
        for i in range(lenS):
            for j in range(lenT):
                gramMat[i][j] = K(sList[i], tList[j], n, l)  # here to calculate the ssk value

                # print("gramMat[{}][{}] = {}".format(i, j, gramMat[i][j]))

    return gramMat

