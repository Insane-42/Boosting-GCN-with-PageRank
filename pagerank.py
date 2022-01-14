# coding=UTF-8

import numpy as np
import networkx as nx
# from data import load_data

def GtoM(G):
    N = np.shape(G)[0]
    M = np.zeros((N, N))
    for i in range(N):
        D_i = sum(G[i]) # 出度之和
        if D_i == 0:
            continue
        for j in range(N):
            # 每个点的分数，由所有 指向它的节点的分数 除以 这个节点的出度数 求和所替代
            # 可以把指向自己理解成别人是否来访问自己，出度表示自己总共流失出去了多少个
            M[j][i] = G[i][j] / D_i # watch out! M_j_i instead of M_i_j #等于转置位置分数除以出度和
    return M

def PageRank(G, T=10000, eps=1e-8, beta=0.5):
    # 初始时rank一样
    M = GtoM(G)
    N = np.shape(G)[0]
    R = np.ones(N) / N
    teleport = np.ones(N) / N
    for _ in range(T):
        R_new = beta * np.dot(M, R) + (1-beta)*teleport
        if np.linalg.norm(R_new - R) < eps:
            break
        R = R_new.copy()
    return R_new

