import numpy as np
import pandas as pd

g = np.random.uniform(200,210,100).reshape(10,10)

class d8():
        
    def __init__(self, g, data_type='dem', input_type='ascii', band=1, nodata=-1, pour=9, bbox=None, autorun=False):

        if input_type == 'ascii':
            pass

	if input_type == 'raster':
	    import rasterio
	    f = rasterio.open(g)
            self.crs = f.crs
            self.bbox = tuple(f.bounds)
            self.shape = f.shape
            self.fill = f.nodatavals[0]
	    if len(f.indexes) > 1:
	        self.data = np.ma.filled(f.read_band(band))
	    else:
	        self.data = np.ma.filled(f.read())
            f.close()
            self.data = self.data.reshape(self.shape)

        if input_type == 'array':
            self.g = g
            self.shape = g.shape
            if bbox:
                self.bbox = bbox
            else:
                self.bbox = (0, 0, g.shape[0], g.shape[1])
	    
        self.idx = np.indices(g.shape)

        if autorun == True:
            if input_type == 'dem':
                self.d = self.flowdir(g)
            elif input_type == 'flowdir':
                self.d = g
	
	self.outer = None
	self.o = None

    def clip_array(self, new_bbox, inplace=False):
        df = pd.DataFrame(self.data,
                          index=np.linspace(b.bbox[1], b.bbox[3],
                                b.shape[0], endpoint=False),
                          columns=np.linspace(b.bbox[0], b.bbox[2],
                                b.shape[1], endpoint=False))
        df = df.loc[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2]]

        if inplace == False:
            return df

        else:
            self.data = df.values
            self.bbox = new_bbox
            self.shape = self.data.shape

    def flowdir(self, data=self.data): 

        #corners
        c = {
        'nw' : {'k' : tuple(idx[:,0,0]),
	        'v' : [[0,1,1], [1,1,0]],
		'pad': np.array([3,4,5])},
        'ne' : {'k' : tuple(idx[:,0,-1]),
	        'v' : [[1,1,0], [-1,-2,-2]],
		'pad': np.array([5,6,7])},
        'sw' : {'k' : tuple(idx[:,-1,0]),
	        'v' : [[-2,-2,-1], [0,1,1]],
		'pad': np.array([1,2,3])},
        'se' : {'k' : tuple(idx[:,-1,-1]),
	        'v' : [[-1,-2,-2], [-2,-2,-1]],
		'pad': np.array([7,8,1])}
        }
    
        #edges
        edge = {
        'n' : {'k' : tuple(idx[:,0,1:-1]),
	       'pad' : np.array([3,4,5,6,7])},
        'w' : {'k' : tuple(idx[:,1:-1,0]),
	       'pad' : np.array([1,2,3,4,5])},
        'e' : {'k' : tuple(idx[:,1:-1,-1]),
	       'pad' : np.array([1,5,6,7,8])},
        's' : {'k' : tuple(idx[:,-1,1:-1]),
	       'pad' : np.array([1,2,3,7,8])}
        }
    
        #body
        body = idx[:, 1:-1, 1:-1]
    
        #output
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
                outmap[c[i]['k']] = c[i]['pad'][np.argmax(dat - sur)]
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

    def prep_accum(self):

	coverage = []
        iterarr = np.vstack(np.dstack(idx))
        iterange = np.arange(d.size) 
        outer = {}

        def goto_cell_r(i, j):
            inner.append(i)
            dirs = [[0,0], [-1,0], [-1,1], [0,1], [1,1],
		    [1,0], [1,-1], [0,-1], [-1,-1]]
            move = dirs[d[tuple(j)]]
            next_i = i + move[1] + move[0]*d.shape[1]
            next_cell = j + move
            if d[tuple(j)] == 0:
                return i
            elif (next_cell < 0).any(): #SHOULD ALSO ACCOUNT FOR N > SHAPE[0], SHAPE[1]
                return i
            elif next_i in coverage:
                return next_i
            else:
                coverage.append(next_i)
                return goto_cell_r(next_i, next_cell)

        def apply_to_zeros(lst, dtype=np.int64):
            inner_max_len = max(map(len, lst))
            result = np.full([len(lst), inner_max_len], -1)
            for i, row in enumerate(lst):
                for j, val in enumerate(row):
                    result[i][j] = val
            return result
        
        def pad_and_cat(a):
            b = a.copy()
            f = np.vectorize(lambda x: x.shape[1])
            ms = f(b).max()
            print ms
            for i in range(len(b)):
                b[i] = np.pad(
			b[i],
			((0,0), (0, ms-b[i].shape[1])),
			mode='constant',
			constant_values=-1)
            return np.vstack(b)

        for w in iterange:
            if not w in coverage:
                inner = []
                coverage.append(w)
                v = iterarr[w]
                h = goto_cell_r(w, v)
                if not h in outer.keys():
                    outer.update({h : []})
                inner = np.array(inner)
                inner = inner[np.where(inner != h)]
                outer[h].append(inner)
        
        for h in outer.keys():
            outer[h] = np.array(outer[h])

        self.outer = pd.Series(outer)
        self.o = pad_and_cat(np.array([apply_to_zeros(i) for i in outer.values]))

    def accum(self):

        iterange = np.arange(d.size) 

        if (not self.outer) or (not self.o):
            self.prep_accum()

        def get_accumulation(n):
            k = outer.index.values
            u = np.unique(o)
            # PRIMARY
            if (not n in k) and (n in u):
                return np.where(o==n)[1].sum()
            # INTERMEDIATE
            elif (n in k) and (n in u):
                prim = len(np.concatenate(outer[n]))
                sec = np.where(o==n)[1].sum()
                return prim + sec
            # FINAL
            elif (n in k) and (not n in u):
                upcells = np.concatenate(outer[n])
                prim = len(upcells)
                if np.in1d(upcells, k).any():
                    sec = np.concatenate(outer.loc[upcells].dropna().values).size
                else:
                    sec = 0
                return prim + sec

        acc_out = np.vectorize(get_accumulation)
        return acc_out(iterange).reshape(d.shape)

    def catchment(self, n):

        iterange = np.arange(d.size) 

        if (not self.outer) or (not self.o):
            self.prep_accum()

        def get_catchment(n):
            k = outer.index.values
            u = np.unique(o)
            # PRIMARY
            if (not n in k) and (n in u):
                q = np.where(o==n)
                return o[q[0], :q[1]]
            # INTERMEDIATE
            elif (n in k) and (n in u):
                prim = np.concatenate(outer[n])
                q = np.where(o==n)
                sec = o[q[0], :q[1]]
                return np.concatenate([prim.ravel(), sec.ravel()])
            # FINAL
            elif (n in k) and (not n in u):
                upcells = np.concatenate(outer[n])
                if np.in1d(upcells, k).any():
                    sec = np.concatenate(outer.loc[upcells].dropna().values)
                    return np.concatenate([upcells.ravel(), sec.ravel()])
                else:
                    return upcells.ravel()
    
            def delineate_catchment(n):
                catchment = np.where(np.in1d(iterange, get_catchment(n)),
            	                 d.ravel(), np.nan)
                catchment[n] = self.pour
                return catchment.reshape(d.shape)

        return delineate_catchment(n)
