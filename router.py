import numpy as np

g = np.random.uniform(200,210,16).reshape(4,4)

def d8_flowdir(g, nodata=0):
    
    idx = np.indices(g.shape)
    
    # ALL GOING CLOCKWISE
    #CORNERS
    c = {
    'nw' : {'k' : tuple(idx[:,0,0]), 'v' : [[0,1,1], [1,1,0]], 'pad': 3},
    'ne' : {'k' : tuple(idx[:,0,-1]), 'v' : [[1,1,0], [-1,-2,-2]], 'pad': 5},
    'sw' : {'k' : tuple(idx[:,-1,0]), 'v' : [[-2,-2,-1], [0,1,1]], 'pad': 1},
    'se' : {'k' : tuple(idx[:,-1,-1]), 'v' : [[-1,-2,-2], [-2,-2,-1]], 'pad': 7}
    }

    #edges
    edge = {
    'n' : {'k' : tuple(idx[:,0,1:-1]), 'pad' : np.array([3,4,5,6,7])},
    'w' : {'k' : tuple(idx[:,1:-1,0]), 'pad' : np.array([1,2,3,4,5])},
    'e' : {'k' : tuple(idx[:,1:-1,-1]), 'pad' : np.array([1,5,6,7,8])},
    's' : {'k' : tuple(idx[:,-1,1:-1]), 'pad' : np.array([1,2,3,7,8])}
    }

    #body
    body = idx[:, 1:-1, 1:-1]

    outmap = np.zeros(g.shape, dtype=np.int8)


    def select_surround(i, j):
        return ([i-1, i-1, i+0, i+1, i+1, i+1, i+0, i-1],
               [j+0, j+1, j+1, j+1, j+0, j-1, j-1, j-1])


    def select_edge_sur(k):
        i,j = edge[k]['k']
        if k == 'n':
            return [i+0, i+1, i+1, i+1, i+0], [j+1, j+1, j+0, j-1, j-1]
        elif k =='e':
            return [i-1, i+1, i+1, i+0, i-1], [j+0, j+0, j-1, j-1, j-1]
        elif k =='s':
            return [i-1, i-1, i+0, i+0, i-1], [j+0, j+1, j+1, j-1, j-1]
        elif k == 'w':
            return [i-1, i-1, i+0, i+1, i+1], [j+0, j+1, j+1, j+1, j+0]

    # FILL CORNERS
    for i in c.keys():
        dat = g[c[i]['k']]
        sur = g[c[i]['v']]
        if ((dat - sur) > 0).any():
            outmap[c[i]['k']] = np.argmax(dat - sur) + c[i]['pad']
        else:
            outmap[c[i]['k']] = 0

    # FILL BODY
    for i, j in np.nditer(tuple(body), flags=['external_loop']):
        dat = g[i,j]
        sur = g[select_surround(i,j)]
        a = ((dat - sur) > 0).any(axis=0)
        b = np.argmax((dat - sur), axis=0) + 1
        c = 0
        outmap[i,j] = np.where(a,b,c)

    #FILL EDGES
    for x in edge.keys():
        dat = g[edge[x]['k']]
        sur = g[select_edge_sur(x)]
        a = ((dat - sur) > 0).any(axis=0)
        b = edge[x]['pad'][np.argmax((dat - sur), axis=0)]
        c = 0
        outmap[edge[x]['k']] = np.where(a,b,c)

    return outmap
