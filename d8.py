import numpy as np
import pandas as pd
import sys
import ast

g = np.random.uniform(300,500,100).reshape(10,10)

class d8():        

    def __init__(self, **kwargs):
        if kwargs['data']:
            self.read_input(**kwargs)
        else:
            pass

    def read_input(self, data, data_type='dir', input_type='ascii', band=1, nodata=0, bbox=None):
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
            
        self.shape_min = np.min_scalar_type(max(self.shape))
        self.size_min = np.min_scalar_type(data.size)
        self.nodata = nodata    
        self.idx = np.indices(self.shape, dtype=self.shape_min)
        setattr(self, data_type, data)

    def __repr__(self):
        return repr(self.dir)

    def nearest_cell(self, lon, lat):

        coords = np.meshgrid(
            np.linspace(self.bbox[0], self.bbox[2],
                        self.shape[1], endpoint=False),
            np.linspace(self.bbox[1], self.bbox[3],
                        self.shape[0], endpoint=False)[::-1])

        nearest = np.unravel_index(np.argmin(np.sqrt((
                                   coords[0] - lon)**2 + (coords[1] - lat)**2)),
                                   self.shape)
        return nearest

    def clip_array(self, data, new_bbox):
        df = pd.DataFrame(data,
                          index=np.linspace(self.bbox[1], self.bbox[3],
                                self.shape[0], endpoint=False),
                          columns=np.linspace(self.bbox[0], self.bbox[2],
                                self.shape[1], endpoint=False))
        df = df.loc[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2]]
        return df

    def flowdir(self, data, include_edges, dirmap=[1,2,3,4,5,6,7,8]):

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
    
        if dirmap != [1,2,3,4,5,6,7,8]:
            dir_d = dict(zip([1,2,3,4,5,6,7,8], dirmap))
            outmap = pd.DataFrame(outmap).apply(lambda x: x.map(dir_d), axis=1).values

        return outmap

    def catchment(self, x, y, pour_value=None, dirmap=[5,6,7,8,1,2,3,4], recursionlimit=15000, inplace=True):

        sys.setrecursionlimit(recursionlimit)
        self.collect = np.array([], dtype=int)
        self.cdir = np.pad(self.dir, 1, mode='constant')
        padshape = self.cdir.shape
	self.cdir = self.cdir.ravel()
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
            next_idx = selection[np.where(self.cdir[selection] == dirmap)]
            if next_idx.any():
                return catchment_search(next_idx)

        catchment_search(pour_point)
        outcatch = np.zeros(padshape, dtype=np.int16)
        outcatch.flat[self.collect] = self.cdir[self.collect]
#        self.cdir = self.cdir.reshape(padshape)[1:-1, 1:-1]
        outcatch = outcatch[1:-1, 1:-1]
        del self.cdir
        del self.collect

        if pour_value is not None:
            outcatch[y,x] = pour_value 

        if inplace == True:
            self.catch = outcatch
        else:
            return outcatch

    def fraction(self, other, inplace=True):

        assert hasattr(self, 'dir')
        assert hasattr(other, 'dir')
        assert hasattr(other, 'catch')

        cell_ratio = int(self.cellsize/other.cellsize)
        selfdf = pd.DataFrame(
                np.arange(self.dir.size).reshape(self.shape),
                index=np.linspace(self.bbox[1], self.bbox[3],
                                  self.shape[0], endpoint=False)[::-1],
                columns=np.linspace(self.bbox[0], self.bbox[2],
                                    self.shape[1], endpoint=False)
                )
        otherdf = pd.DataFrame(
                other.dir,
                index=np.linspace(other.bbox[1], other.bbox[3],
                                  other.shape[0], endpoint=False)[::-1],
                columns=np.linspace(other.bbox[0], other.bbox[2],
                                    other.shape[1], endpoint=False)
                )
        result = selfdf.reindex(otherdf.index, method='nearest').reindex_axis(otherdf.columns, axis=1, method='nearest')
        result = result.values[np.where(other.catch != 0, True, False)]
        result = (np.bincount(result, minlength=selfdf.size).astype(float)/(cell_ratio**2)).reshape(selfdf.shape)

        if inplace == True:
            self.frac = result
        else:
            return result
