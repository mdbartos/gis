import numpy as np
import pandas as pd

g = np.random.uniform(200,210,100).reshape(10,10)

class d8():
        
    def __init__(self, data, data_type='dem', input_type='ascii', band=1, nodata=-1, pour=9, bbox=None, autorun=False):

        if input_type == 'ascii':
            pass

	if input_type == 'raster':
	    import rasterio
	    f = rasterio.open(data)
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
            self.data = data
            self.shape = data.shape
            if bbox:
                self.bbox = bbox
            else:
                self.bbox = (0, 0, data.shape[0], data.shape[1])
	
        self.pour = pour
        self.nodata = nodata    
        self.idx = np.indices(self.data.shape)

        if autorun == True:
            if input_type == 'dem':
                self.d = self.flowdir(data)
            elif input_type == 'flowdir':
                self.d = data

            self.branches, self.pos = self.prep_accum()
            self.accumulation = self.accum()
            self.catchment = self.catch()

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

    def flowdir(self): 

        #corners
        c = {
        'nw' : {'k' : tuple(self.idx[:,0,0]),
	        'v' : [[0,1,1], [1,1,0]],
		'pad': np.array([3,4,5])},
        'ne' : {'k' : tuple(self.idx[:,0,-1]),
	        'v' : [[1,1,0], [-1,-2,-2]],
		'pad': np.array([5,6,7])},
        'sw' : {'k' : tuple(self.idx[:,-1,0]),
	        'v' : [[-2,-2,-1], [0,1,1]],
		'pad': np.array([1,2,3])},
        'se' : {'k' : tuple(self.idx[:,-1,-1]),
	        'v' : [[-1,-2,-2], [-2,-2,-1]],
		'pad': np.array([7,8,1])}
        }
    
        #edges
        edge = {
        'n' : {'k' : tuple(self.idx[:,0,1:-1]),
	       'pad' : np.array([3,4,5,6,7])},
        'w' : {'k' : tuple(self.idx[:,1:-1,0]),
	       'pad' : np.array([1,2,3,4,5])},
        'e' : {'k' : tuple(self.idx[:,1:-1,-1]),
	       'pad' : np.array([1,5,6,7,8])},
        's' : {'k' : tuple(self.idx[:,-1,1:-1]),
	       'pad' : np.array([1,2,3,7,8])}
        }
    
        #body
        body = self.idx[:, 1:-1, 1:-1]
    
        #output
        outmap = np.zeros(self.data.shape, dtype=np.int8)
    
    
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
            dat = self.data[c[i]['k']]
            sur = self.data[c[i]['v']]
            if ((dat - sur) > 0).any():
                outmap[c[i]['k']] = c[i]['pad'][np.argmax(dat - sur)]
            else:
                outmap[c[i]['k']] = self.nodata
    
        # FILL BODY
        for i, j in np.nditer(tuple(body), flags=['external_loop']):
            dat = self.data[i,j]
            sur = self.data[select_surround(i,j)]
            a = ((dat - sur) > 0).any(axis=0)
            b = np.argmax((dat - sur), axis=0) + 1
            c = self.nodata
            outmap[i,j] = np.where(a,b,c)
    
        #FILL EDGES
        for x in edge.keys():
            dat = self.data[edge[x]['k']]
            sur = self.data[select_edge_sur(x)]
            a = ((dat - sur) > 0).any(axis=0)
            b = edge[x]['pad'][np.argmax((dat - sur), axis=0)]
            c = self.nodata
            outmap[edge[x]['k']] = np.where(a,b,c)
    
        return outmap

    def prep_accum(self):

	coverage = []
        iterarr = np.vstack(np.dstack(self.idx)) 
        iterange = np.arange(self.d.size) 
        outer = {}

        def goto_cell_r(i, j):
            inner.append(i)
            dirs = [[0,0], [-1,0], [-1,1], [0,1], [1,1],
		    [1,0], [1,-1], [0,-1], [-1,-1]]
            move = dirs[self.d[tuple(j)]]
            next_i = i + move[1] + move[0]*self.d.shape[1]
            next_cell = j + move
            if self.d[tuple(j)] == self.nodata:
                return i
            elif (next_cell < 0).any(): #SHOULD ALSO ACCOUNT FOR N > SHAPE[0], SHAPE[1]
                return i
            elif next_i in coverage:
                return next_i
            else:
                coverage.append(next_i)
                return goto_cell_r(next_i, next_cell)

        def pad_inner(lst, dtype=np.int64):
            inner_max_len = max(map(len, lst))
            result = np.full([len(lst), inner_max_len], -1)
            for i, row in enumerate(lst):
                for j, val in enumerate(row):
                    result[i][j] = val
            return result
        
        def pad_outer(a):
            b = a.copy()
            f = np.vectorize(lambda x: x.shape[1])
            ms = f(b).max()
            print ms
            for i in range(len(b)):
                b[i] = np.pad(
			b[i],
			((0,0), (0, ms-b[i].shape[1])),
			mode='constant',
			constant_values=self.nodata)
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

        outer = pd.Series(outer)
        return (outer.apply(np.concatenate), pad_outer(np.array([pad_inner(i) for i in outer.values])))

    def accum(self):

        if not hasattr(self, 'd'):
            self.d = self.flowdir()

        iterange = np.arange(self.d.size) 

        if (not hasattr(self, 'branches')) or (not hasattr(self, 'pos')):
            self.branches, self.pos = self.prep_accum()

        def get_accumulation(n):
            k = self.branches.index.values
            u = np.unique(self.pos)
            # PRIMARY
            if (not n in k) and (n in u):
                return np.where(self.pos==n)[1].sum()
            # INTERMEDIATE
            elif (n in k) and (n in u):
                prim = len(self.branches[n])
                sec = np.where(self.pos==n)[1].sum()
                return prim + sec
            # FINAL
            elif (n in k) and (not n in u):
                upcells = self.branches[n]
                prim = len(upcells)
                if np.in1d(upcells, k).any():
                    sec = np.concatenate(self.branches.loc[upcells].dropna().values).size
                else:
                    sec = 0
                return prim + sec

        acc_out = np.vectorize(get_accumulation)
        return acc_out(iterange).reshape(self.d.shape)

    def catch(self, n):

        if not hasattr(self, 'd'):
            self.d = self.flowdir()

        iterange = np.arange(self.d.size) 

        if (not hasattr(self, 'branches')) or (not hasattr(self, 'pos')):
            self.branches, self.pos = self.prep_accum()

        if isinstance(n, int):
            pass
        elif isinstance(n, (tuple, list, np.ndarray)):
            n = n[0] + n[1]*self.d.shape[1]

        def get_catchment(n):
            k = self.branches.index.values
            u = np.unique(self.pos)
            # PRIMARY
            if (not n in k) and (n in u):
                q = np.where(self.pos==n)
                return self.pos[q[0], :q[1]]
            # INTERMEDIATE
            elif (n in k) and (n in u):
                prim = self.branches[n]
                q = np.where(self.pos==n)
                sec = self.pos[q[0], :q[1]]
                return np.concatenate([prim.ravel(), sec.ravel()])
            # FINAL
            elif (n in k) and (not n in u):
                upcells = self.branches[n]
                if np.in1d(upcells, k).any():
                    sec = np.concatenate(self.branches.loc[upcells].dropna().values)
                    return np.concatenate([upcells.ravel(), sec.ravel()])
                else:
                    return upcells.ravel()
    
        catchment = np.where(np.in1d(iterange, get_catchment(n)),
            	             self.d.ravel(), self.nodata)
        catchment[n] = self.pour
        return catchment.reshape(self.d.shape)

