import numpy as np
import pandas as pd
import sys
import ast


class flow_grid():        
    """
    Container class for holding and manipulating gridded VIC routing data.
    Can be instantiated with optional keyword arguments. These keyword arguments
    will be used to add a dataset (dem, flowdir, accumulation) to the flow_grid instance.

    Parameters
    ----------
    data : File name (string) or numpy ndarray
           If data is from a file, 'input_type' should be set to the appropriate value
           ('ascii' or 'raster').
    data_type : 'dem', 'dir', 'acc'
                 How to interpret the input data:
                     'dem' : digital elevation data
                     'dir' : flow direction data
                     'acc' : flow accumulation (upstream area) data
                
    input_type : 'raster', 'ascii' or 'array'
                 Type of input data.
    band : int
           For raster data, the band number to read.
    nodata : int or float
             Value indicating no data.

    Attributes (Optional)
    ---------------------
    dem : digital elevation grid
    dir : flow direction grid
    acc : flow accumulation grid
    catch : Catchment delineated from 'dir' and a given pour point
    frac : fractional contributing area grid
    bbox : The geographical bounding box of the gridded dataset
           (xmin, ymin, xmax, ymax)
    shape : The shape of the gridded data (nrows, ncolumns)
    cellsize : The length/width of each grid cell (assumed to be square).
    nodata : The value to use for gridcells with no data.

    Methods
    -------
    read_input : add a gridded dataset (dem, flowdir, accumulation) 
                 to flow_grid instance.
    nearest_cell : Returns the index (column, row) of the cell closest
                   to a given geographical coordinate (x, y).
    flowdir : Generate a flow direction grid from a given digital elevation
              dataset (dem).
    catchment : Delineate the watershed for a given pour point (x, y)
                or (column, row).
    fraction : Generate the fractional contributing area for a coarse
               scale flow direction grid based on a fine-scale flow
               direction grid.
    """

    def __init__(self, **kwargs):
        if 'data' in kwargs:
            self.read_input(**kwargs)
        else:
            pass

    def read_input(self, data, data_type='dir', input_type='ascii', band=1, nodata=0, bbox=None):
        """
        Reads data into a named attribute of flow_grid
        (name of attribute determined by 'data_type').

        Parameters
        ----------
        data : File name (string) or numpy ndarray
               If data is from a file, 'input_type' should be set to the appropriate value
               ('ascii' or 'raster').
        data_type : 'dem', 'dir', 'acc'
                     How to interpret the input data:
                         'dem' : digital elevation data
                         'dir' : flow direction data
                         'acc' : flow accumulation (upstream area) data
                    
        input_type : 'raster', 'ascii' or 'array'
                     Type of input data.
        band : int
               For raster data, the band number to read.
        nodata : int or float
                 Value indicating no data.
        bbox : tuple or list
               Bounding box, if none provided.

        """
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
            self.cellsize = f.affine[0]
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

    def nearest_cell(self, lon, lat):
        """
        Returns the index of the cell (column, row) closest
        to a given geographical coordinate.

        Parameters
        ----------
        lon : int or float
              x coordinate.
        lat : int or float
              y coordinate.
        """

        coords = np.meshgrid(
            np.linspace(self.bbox[0] + self.cellsize/2.0,
                        self.bbox[2] + self.cellsize/2.0,
                        self.shape[1], endpoint=False),
            np.linspace(self.bbox[1] + self.cellsize/2.0,
                        self.bbox[3] + self.cellsize/2.0,
                        self.shape[0], endpoint=False)[::-1])

        nearest = np.unravel_index(np.argmin(np.sqrt((
                                   coords[0] - lon)**2 + (coords[1] - lat)**2)),
                                   self.shape)
        return nearest[1], nearest[0]

    def flowdir(self, data, include_edges, dirmap=[1,2,3,4,5,6,7,8]):
        """
        Generates a flow direction grid from a DEM grid.

        Parameters
        ----------
        data : numpy ndarray
               Array representing DEM grid
        include_edges : bool
                        Whether to include outer rim of grid.
        dirmap : list or tuple
                 List of integer values representing the following
                 cardinal directions (in order):
                 [N, NE, E, SE, S, SW, W, NW]
        """

        #initialize indices of corners
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
    
        #initialize indices of edges
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
    
        #initialize indices of body
        body = self.idx[:, 1:-1, 1:-1] #Shouldn't use self.idx
    
        #initialize output array
        outmap = np.full(self.shape, self.nodata, dtype=np.int8) #Shouldn't be int8
    
    
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
     
        # Fill body
        for i, j in np.nditer(tuple(body), flags=['external_loop']):
            dat = data[i,j]
            sur = data[select_surround(i,j)]
            a = ((dat - sur) > 0).any(axis=0)
            b = np.argmax((dat - sur), axis=0) + 1
            c = self.nodata
            outmap[i,j] = np.where(a,b,c)

        if include_edges == True:

            # Fill corners
            for i in corner.keys():
                dat = data[corner[i]['k']]
                sur = data[corner[i]['v']]
                if ((dat - sur) > 0).any():
                    outmap[corner[i]['k']] = corner[i]['pad'][np.argmax(dat - sur)]
                else:
                    outmap[corner[i]['k']] = self.nodata

            #Fill edges
            for x in edge.keys():
                dat = data[edge[x]['k']]
                sur = data[select_edge_sur(x)]
                a = ((dat - sur) > 0).any(axis=0)
                b = edge[x]['pad'][np.argmax((dat - sur), axis=0)]
                c = self.nodata
                outmap[edge[x]['k']] = np.where(a,b,c)
    
        # If dirmap isn't range(1,9), convert values of outmap.
        if dirmap != [1,2,3,4,5,6,7,8]:
            dir_d = dict(zip([1,2,3,4,5,6,7,8], dirmap))
            outmap = pd.DataFrame(outmap).apply(lambda x: x.map(dir_d), axis=1).values

        return outmap

    def catchment(self, x, y, pour_value=None, dirmap=[1,2,3,4,5,6,7,8], xytype='index', recursionlimit=15000, inplace=True):
        """
        Delineates a watershed from a given pour point (x, y).
        Returns a grid 

        Parameters
        ----------
        x : int or float
            x coordinate of pour point
        y : int or float
            y coordinate of pour point
        pour_value : int or None
                     If not None, value to represent pour point in catchment grid
                     (required by some programs).
        dirmap : list or tuple
                 List of integer values representing the following
                 cardinal directions (in order):
                 [N, NE, E, SE, S, SW, W, NW] 
        xytype : 'index' or 'label'
                 How to interpret parameters 'x' and 'y'.
                     'index' : x and y represent the column and row 
                               indices of the pour point.
                     'label' : x and y represent geographic coordinates
                               (will be passed to self.nearest_cell).
        recursionlimit : int
                         Recursion limit--may need to be raised if
                         recursion limit is reached.
        inplace : bool
                  If True, catchment will be written to attribute 'catch'.
        """

        if xytype == 'label':
            x, y = self.nearest_cell(x, y)
        elif xytype == 'index':
            pass
        sys.setrecursionlimit(recursionlimit)
        self.collect = np.array([], dtype=int)
        self.cdir = np.pad(self.dir, 1, mode='constant')
        padshape = self.cdir.shape
	self.cdir = self.cdir.ravel()
        pour_point = np.ravel_multi_index(np.array([y+1, x+1]), padshape)
        dirmap = np.array(dirmap)[[4,5,6,7,0,1,2,3]].tolist()

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
        """
        Generates a grid representing the fractional contributing area for a
        coarse-scale flow direction grid.

        Parameters
        ----------
        other : Another flow_grid instance containing fine-scale flow direction data.
                The ratio of self.cellsize/other.cellsize must be a positive integer.
                Grid cell boundaries must have some overlap.
                Must have attributes 'dir' and 'catch' (i.e. must have a flow direction grid,
                along with a delineated catchment).
                
        inplace : bool
                  If True, appends fraction grid to attribute 'frac'.
        """

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
