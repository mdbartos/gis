import numpy as np

g = np.random.uniform(200,210,100).reshape(10,10)
#np.vstack(np.dstack(np.indices(g.shape)))

def build_args(arr):
    
    idx = np.indices(g.shape)
    
    #CORNERS
    c = {
    'nw' : idx[:,0,0],
    'ne' : idx[:,0,-1],
    'sw' : idx[:,-1,0],
    'se' : idx[:,-1,-1],
    }

    #edges
    edges = {
    'n' : idx[:,0,1:-1] ,
    'w' : idx[:,1:-1,0],
    'e' : idx[:,1:-1,-1],
    's' : idx[:,-1,1:-1],
    }

    #body
    body = idx[:, 1:-1, 1:-1]

    outmap = np.empty(g.shape)

def select_surround(i, j):
    return [i+0, i+1, i+1, i+1, i+0, i-1, i-1, i-1], [j+1, j+1, j+0, j-1, j-1, j-1, j+0, j+1]

f = np.vectorize(select_surround)

# ONE-BY-ONE
for i, j in np.nditer([body[0], body[1]]):
    dif = (g[i,j] - g[f(i,j)])
    if (dif > 0).any():
        outmap[i,j] = dif.argmax() + 1
    else:
        outmap[i,j] = 0

# ROW-BY-ROW
for i, j in np.nditer([body[0], body[1]], flags=['external_loop']):
    dat = g[i,j]
    sur = g[select_surround(i,j)]
    a = ((dat - sur) > 0).any(axis=0)
    b = np.argmax((dat - sur), axis=0) + 1
    c = 0
    outmap2[i,j] = np.where(a,b,c)
