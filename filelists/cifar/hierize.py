import json, sys
import numpy as np
from scipy.sparse import csr_matrix, csgraph

with open('novel.json') as b:
    labels = json.load(b)['label_names']
hfname = 'hier.json'
hierlen = 26
with open(hfname) as h:
    hier = json.load(h)

am = np.zeros((hierlen, hierlen))
tc = {}
for k, v in hier.items():
    tc[k] = []
    if k not in labels:
        labels.append(k)
    am[labels.index(k)][hierlen - 1] = 1
    am[hierlen - 1][labels.index(k)] = 1
    for k2, v2 in v.items():
        tc[k].append((k2,1))
        tc[k2] = []
        if k2 not in labels:
            labels.append(k2)
        am[labels.index(k)][labels.index(k2)] = 1
        am[labels.index(k2)][labels.index(k)] = 1
        for k3, v3 in v2.items():
            tc[k].append((k3,2))
            tc[k2].append((k3,1))
            tc[k3] = []
            if k3 not in labels:
                labels.append(k3)
            am[labels.index(k3)][labels.index(k2)] = 1
            am[labels.index(k2)][labels.index(k3)] = 1
            for k4, v4 in v3.items():
                tc[k].append((k4,3))
                tc[k2].append((k4,2))
                tc[k3].append((k4,1))
                tc[k4] = []
                if k4 not in labels:
                    labels.append(k4)
                am[labels.index(k3)][labels.index(k4)] = 1
                am[labels.index(k4)][labels.index(k3)] = 1
                for k5, v5 in v4.items():                  
                    tc[k].append((k5,4))           
                    tc[k2].append((k5,3))                             
                    tc[k3].append((k5,2))               
                    tc[k4].append((k5,1))
                    tc[k5] = []
                    if k5 not in labels:  
                        labels.append(k5)
                    am[labels.index(k4)][labels.index(k5)] = 1
                    am[labels.index(k5)][labels.index(k4)] = 1

dists = csgraph.shortest_path(csr_matrix(am), directed=False)[:-1,:-1]
print(dists.mean())
print(len(tc), len(labels))
tc_npy = np.zeros((len(tc), len(tc)))
np.fill_diagonal(tc_npy, 1)
for w, hypo in tc.items():
    for h in hypo:
        tc_npy[labels.index(h[0])][labels.index(w)] = 1

print(tc_npy.mean())
np.save('tc.npy', tc_npy)
np.save('dist.npy', dists)
