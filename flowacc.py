import numpy as np
import pandas as pd
import igraph

self = fine_grid
dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
di = self.view('dir', mask=False)
cdi = di[1:-1, 1:-1]
shape = cdi.shape

go_to = (
  0 - shape[1],
  1 - shape[1],
  1 + 0,
  1 + shape[1],
  0 + shape[1],
 -1 + shape[1],
 -1 + 0,
 -1 - shape[1]
        )

go_to = dict(zip(dirmap, go_to))

startnodes = []
endnodes = []

for i in dirmap:
    j_0 = np.ravel_multi_index(np.array(np.where(cdi == i)), shape)
    j_1 = j_0 + go_to[i]
    startnodes.extend(j_0)
    endnodes.extend(j_1)

startnodes = np.asarray(startnodes)
endnodes = np.asarray(endnodes)
data = np.ones(len(startnodes), dtype=np.bool)

startnodes = startnodes[endnodes >= 0]
endnodes = endnodes[endnodes >= 0]
data = data[endnodes >= 0]

s = np.full(cdi.size, -1, dtype=int)
e = np.full(cdi.size, -1, dtype=int)
c = np.ones(cdi.size, dtype=int)

s[startnodes] = startnodes
e[startnodes] = endnodes

next_s = s.copy()
next_e = e[(e != -1) & (e <= cdi.size)]

for i in range(10000):
    len_prev = len(next_e)
    next_s = s[next_e]
    next_s = next_s[next_s != -1]
    next_e = e[next_s]
    next_e = next_e[(next_e != -1) & (next_e <= cdi.size)]
    print 'next e:', next_e, len(next_e)
    len_next = len(next_e)
    if len_next == len_prev:
        break
    else:
        c[next_e] += np.bincount(next_e, minlength=next_e.max())[next_e]
#        c[next_e] += 1    #THIS NEEDS TO ACCOUNT FOR ENDNODES LISTED MORE THAN ONCE IN THE INDEXER

### HOLY SHIT THIS ACTUALLY WORKS!!!^^^^^
