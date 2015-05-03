import numpy as np
import pandas as pd
import sys

g = np.random.uniform(300,500,100).reshape(10,10)

class d8():        

    def __init__(self, data, data_type='dem', input_type='ascii', band=1, nodata=0, bbox=None, include_edges=True):

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
                data = np.ma.filled(f.read_band(band))
            else:
                data = np.ma.filled(f.read())
                f.close()
                data = self.data.reshape(self.shape)

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

    def catchment(self, x, y, pour_value=None):

        self.collect = np.array([], dtype=int)
        self.dir = np.pad(self.dir, 1, mode='constant')
        padshape = self.dir.shape
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
            next_idx = selection[np.where(self.dir[selection] == [5,6,7,8,1,2,3,4])]
            if next_idx.any():
                return catchment_search(next_idx)

        catchment_search(pour_point)
        outcatch = np.zeros(padshape, dtype=np.int8)
        outcatch.flat[self.collect] = self.dir[self.collect]
        self.dir = self.dir.reshape(padshape)[1:-1, 1:-1]
        outcatch = outcatch[1:-1, 1:-1]
        del self.collect

        if pour_value is not None:
            outcatch[y,x] = pour_value 
        return outcatch
