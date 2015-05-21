import numpy as np
import pandas as pd
import sys
import ast

g = np.random.uniform(300,500,100).reshape(10,10)

class d8():        

    def __init__(self, data, data_type='dem', input_type='ascii', band=1, nodata=0, bbox=None, include_edges=True):

        if input_type == 'ascii':
            with open(data) as header:
                ncols = int(header.readline().split()[1])
                nrows = int(header.readline().split()[1])
                xll = ast.literal_eval(header.readline().split()[1])
                yll = ast.literal_eval(header.readline().split()[1])
                self.cellsize = ast.literal_eval(header.readline().split()[1])
                self.fill = ast.literal_eval(header.readline().split()[1])
                self.shape = (nrows, ncols)
                self.bbox = (xll, yll, xll + ncols*self.cellsize, yll + nrows*self.cellsize)
            data = np.loadtxt(data, skiprows=6)

        if input_type == 'raster':
            import rasterio
            f = rasterio.open(data)
            self.crs = f.crs
            self.bbox = tuple(f.bounds)
            self.shape = f.shape
            self.fill = f.nodatavals[0]
            self.cellsize = f.affine[0]    #ASSUMES THAT CELLS ARE SQUARE
            if len(f.indexes) > 1:
                data = np.ma.filled(f.read_band(band))
            else:
                data = np.ma.filled(f.read())
                f.close()
                data = data.reshape(self.shape)

        if input_type == 'array':
            self.shape = data.shape
            if bbox:
                self.bbox = bbox
#            MIGHT BE CONFUSING
#            else:
#                self.bbox = (0, 0, self.shape[0], self.shape[1])
            
        self.shape_min = np.min_scalar_type(max(self.shape))
        self.size_min = np.min_scalar_type(data.size)
        self.nodata = nodata    
        self.idx = np.indices(self.shape, dtype=self.shape_min)

        if data_type == 'dem':
            self.dir = self.flowdir(data, include_edges=include_edges)
        elif data_type == 'flowdir':
            self.dir = data

    def __repr__(self):
        return repr(self.dir)

    def clip_array(self, data, new_bbox, inplace=False):
        df = pd.DataFrame(data,
                          index=np.linspace(self.bbox[1], self.bbox[3],
                                self.shape[0], endpoint=False),
                          columns=np.linspace(self.bbox[0], self.bbox[2],
                                self.shape[1], endpoint=False))
        df = df.loc[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2]]

        if inplace == False:
            return df

        else:
#NOT WORKING
            self.data = df.values
            self.bbox = new_bbox
            self.shape = self.data.shape

    def flowdir(self, data, include_edges):

        #corners
        corner = {
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
        outmap = np.full(self.shape, self.nodata, dtype=np.int8)
    
    
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
     
        # FILL BODY
        for i, j in np.nditer(tuple(body), flags=['external_loop']):
            dat = data[i,j]
            sur = data[select_surround(i,j)]
            a = ((dat - sur) > 0).any(axis=0)
            b = np.argmax((dat - sur), axis=0) + 1
            c = self.nodata
            outmap[i,j] = np.where(a,b,c)

        if include_edges == True:

            # FILL CORNERS
            for i in corner.keys():
                dat = data[corner[i]['k']]
                sur = data[corner[i]['v']]
                if ((dat - sur) > 0).any():
                    outmap[corner[i]['k']] = corner[i]['pad'][np.argmax(dat - sur)]
                else:
                    outmap[corner[i]['k']] = self.nodata

            #FILL EDGES
            for x in edge.keys():
                dat = data[edge[x]['k']]
                sur = data[select_edge_sur(x)]
                a = ((dat - sur) > 0).any(axis=0)
                b = edge[x]['pad'][np.argmax((dat - sur), axis=0)]
                c = self.nodata
                outmap[edge[x]['k']] = np.where(a,b,c)
    
        return outmap

    def catchment(self, x, y, pour_value=None, dirmap=[5,6,7,8,1,2,3,4], recursionlimit=15000):

        sys.setrecursionlimit(recursionlimit)
        self.collect = np.array([], dtype=int)
        self.dir = np.pad(self.dir, 1, mode='constant')
        padshape = self.dir.shape
        # KEEP IN MIND THAT WHEN AN EXCEPTION IS RAISED IT SCREWS EVERYTHING UP IN PLACE
	self.dir = self.dir.ravel()
        pour_point = np.ravel_multi_index(np.array([y+1, x+1]), padshape)

        def select_surround_ravel(i):
            return np.array([i + 0 - padshape[1],
                             i + 1 - padshape[1],
                             i + 1 + 0,
                             i + 1 + padshape[1],
                             i + 0 + padshape[1],
                             i - 1 + padshape[1],
                             i - 1 + 0,
                             i - 1 - padshape[1]]).T


        def catchment_search(j):
            self.collect = np.append(self.collect, j)
            selection = select_surround_ravel(j)
            next_idx = selection[np.where(self.dir[selection] == dirmap)]
            if next_idx.any():
                return catchment_search(next_idx)

        catchment_search(pour_point)
        outcatch = np.zeros(padshape, dtype=np.int16)
        outcatch.flat[self.collect] = self.dir[self.collect]
        self.dir = self.dir.reshape(padshape)[1:-1, 1:-1]
        outcatch = outcatch[1:-1, 1:-1]
        del self.collect

        if pour_value is not None:
            outcatch[y,x] = pour_value 
        return outcatch

b = d8('./na_dir_30s/DRT_8th_FDR_globe.asc', data_type='flowdir', input_type='ascii')
q = d8('./na_dir_30s/na_dir_30s/w001001.adf', data_type='flowdir', input_type='raster')
catch = q.catchment(5831, 3797, 9, [4, 8, 16, 32, 64, 128, 1, 2], recursionlimit=15000)
nz = np.nonzero(catch)
#nz = np.column_stack([nz[1], nz[0]])

cell_ratio = int(b.cellsize/q.cellsize)

q_index = np.linspace(q.bbox[1], q.bbox[3], q.shape[0], endpoint=False)
q_columns = np.linspace(q.bbox[0], q.bbox[2], q.shape[1], endpoint=False)
qdf = pd.DataFrame(q.dir, index=q_index, columns=q_columns)

b_index = np.linspace(b.bbox[1], b.bbox[3], b.shape[0], endpoint=False)
b_columns = np.linspace(b.bbox[0], b.bbox[2], b.shape[1], endpoint=False)
bdf = pd.DataFrame(np.arange(b.dir.size).reshape(b.shape), index=b_index, columns=b_columns)

bri = bdf.reindex(qdf.index, method='nearest').reindex_axis(qdf.columns, axis=1, method='nearest')
result_idx = bri.values[np.where(catch != 0, True, False)]

### VIOLA!!!
result = (np.bincount(result_idx, minlength=bdf.size).astype(float)/(cell_ratio**2)).reshape(bdf.shape)
