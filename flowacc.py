import numpy as np

outacc = np.zeros(g.shape)
idx = np.indices(g.shape)

#def goto_cell(i):
#    dirs = [[0,0], [-1,0], [-1,1], [0,1], [1,1], [1,0], [1,-1], [0,-1], [-1,-1]]
#    return dirs[d[tuple(i)]]

def goto_cell_r(i, j):
    print i, j
    inner.append(i)
    dirs = [[0,0], [-1,0], [-1,1], [0,1], [1,1], [1,0], [1,-1], [0,-1], [-1,-1]]
    move = dirs[d[tuple(j)]]
    next_i = i + move[1] + move[0]*d.shape[1]
    next_cell = j + move
    if d[tuple(j)] == 0:
        print 'Done!'
        return j
    elif (next_cell < 0).any(): #SHOULD ALSO ACCOUNT FOR N > SHAPE[0], SHAPE[1]
        print 'Out of bounds!'
        return j
    else:
        return goto_cell_r(next_i, next_cell)

coverage = []

iterarr = np.vstack(np.dstack(idx))
iterange = np.arange(iterarr.shape[0])

outer = []

for i in iterange:
    if not i in coverage:
        inner = []
        coverage.append(i)
        j = iterarr[i]
        goto_cell_r(i, j)
        outer.append(inner)
