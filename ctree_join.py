import pandas as pd
import numpy as np
import fiona
from shapely import geometry
from shapely import ops
from itertools import chain
from pyproj import Proj, transform
from scipy import spatial


class quick_spatial_join():
    def __init__(self, shp1, shp2, convert_crs=0):
        
        self.shapes = {
                        'shp1' : {
                            'file' : fiona.open(shp1, 'r')
                            },
                
                        'shp2': {
                            'file' : fiona.open(shp2, 'r')
                            }
                }

	self.shapes['shp1'].update({'crs': self.shapes['shp1']['file'].crs})
	self.shapes['shp1'].update({'types': self.geom_types(self.shapes['shp1']['file']).dropna()})
	self.shapes['shp2'].update({'crs': self.shapes['shp2']['file'].crs})
	self.shapes['shp2'].update({'types': self.geom_types(self.shapes['shp2']['file']).dropna()})


        if self.shapes['shp1']['crs'] != self.shapes['shp2']['crs']:
            if convert_crs==0:
                self.shapes['shp1'].update({'shp' : self.convert_crs('shp1', self.shapes['shp1']['crs'], self.shapes['shp2']['crs'])})
                self.shapes['shp2'].update({'shp' : self.homogenize_inputs('shp2')})
            elif convert_crs==1:
                self.shapes['shp2'].update({'shp' : self.convert_crs('shp2', self.shapes['shp2']['crs'], self.shapes['shp1']['crs'])})
                self.shapes['shp1'].update({'shp' : self.homogenize_inputs('shp1')})
        else:
            self.shapes['shp1'].update({'shp' : self.homogenize_inputs('shp1')})
            self.shapes['shp2'].update({'shp' : self.homogenize_inputs('shp2')})
                

        self.shapes['shp1'].update({'poly' : self.poly_return('shp1')})
        self.shapes['shp2'].update({'poly' : self.poly_return('shp2')})
      

        
    def homogenize_inputs(self, shp):
        print 'homogenizing inputs for %s...' % (shp)
        
        d = {}
        
        bv = self.poly_vectorize(self.shapes[shp]['file']).dropna()
        gtypes = self.shapes[shp]['types']

        poly = bv.loc[gtypes=='Polygon'] 
        mpoly = bv.loc[gtypes=='MultiPolygon'] 

        apoly = poly.apply(lambda x: list(chain(*x)))
        a_mpoly = mpoly.apply(lambda x: list(chain(*x)))
        
        #### HOMOGENIZE POLYGONS

        if len(poly) > 0:
            polyarrays = pd.Series(apoly.apply(lambda x: np.array(x)))
            p_x_arrays = polyarrays.apply(lambda x: np.array(x)[:,0])
            p_y_arrays = polyarrays.apply(lambda x: np.array(x)[:,1])
            p_trans_arrays = pd.concat([p_x_arrays, p_y_arrays], axis=1)

            d['p_geom'] = pd.Series(zip(p_trans_arrays[0], p_trans_arrays[1]), index=p_trans_arrays.index).apply(np.column_stack)
        
        #### HOMOGENIZE MULTIPOLYGONS
        
        if len(mpoly) > 0:            
            mpolydims = a_mpoly.apply(lambda x: np.array(x).ndim)

            ##ndim==1

            if (mpolydims==1).any():
                m_x_arrays_1 = a_mpoly[mpolydims==1].apply(pd.Series).stack().apply(lambda x: np.array(x)[:,0])
                m_y_arrays_1 = a_mpoly[mpolydims==1].apply(pd.Series).stack().apply(lambda x: np.array(x)[:,1])

                mp = pd.concat([m_x_arrays_1, m_y_arrays_1], axis=1)

                m_geom_1_s = pd.Series(zip(mp[0], mp[1])).apply(np.column_stack)

                empty_s = pd.Series(range(len(mp)), index=mp.index)
                empty_s = empty_s.reset_index()
                empty_s[0] = m_geom_1_s

                d['m_geom_1'] = empty_s.groupby('level_0').apply(lambda x: tuple(list(x[0])))

            ##ndim==3

            if (mpolydims==3).any():
                m_arrays_3 = a_mpoly[mpolydims==3].apply(pd.Series).stack().apply(lambda x: np.array(x)[:,[0,1]])

                d['m_geom_3'] = m_arrays_3.reset_index().groupby('level_0').apply(lambda x: tuple(list(x[0])))
        
        return pd.concat(d.values()).sort_index()

        
    def convert_crs(self, shp, crsfrom, crsto):
        print 'converting coordinate reference system of %s...' % (shp)
        
        crsfrom = Proj(crsfrom, preserve_units=True)
        crsto = Proj(crsto, preserve_units=True)
        
        d = {}
        
        bv = self.poly_vectorize(self.shapes[shp]['file']).dropna()
        gtypes = self.shapes[shp]['types']

        poly = bv.loc[gtypes=='Polygon'] 
        mpoly = bv.loc[gtypes=='MultiPolygon'] 

        apoly = poly.apply(lambda x: list(chain(*x)))
        a_mpoly = mpoly.apply(lambda x: list(chain(*x)))
        
        #### CONVERT POLYGONS
        
        if len(poly) > 0:
            polyarrays = pd.Series(apoly.apply(lambda x: np.array(x)))
            p_x_arrays = polyarrays.apply(lambda x: np.array(x)[:,0])
            p_y_arrays = polyarrays.apply(lambda x: np.array(x)[:,1])
            p_trans_arrays = pd.concat([p_x_arrays, p_y_arrays], axis=1).apply(lambda x: transform(crsfrom, crsto, x[0], x[1]), axis=1)
        
            d['p_trans_geom'] = p_trans_arrays.apply(np.array).apply(np.column_stack)
        
        #### CONVERT MULTIPOLYGONS
        
        if len(mpoly) > 0:
            mpolydims = a_mpoly.apply(lambda x: np.array(x).ndim)
        
            ##ndim==1
            
            if (mpolydims==1).any():
                m_x_arrays_1 = a_mpoly[mpolydims==1].apply(pd.Series).stack().apply(lambda x: np.array(x)[:,0])
                m_y_arrays_1 = a_mpoly[mpolydims==1].apply(pd.Series).stack().apply(lambda x: np.array(x)[:,1])
                mp = pd.concat([m_x_arrays_1, m_y_arrays_1], axis=1)
                m_x_flat_arrays_1 = pd.Series([j[:,0] for j in [np.column_stack(i) for i in np.column_stack([mp[0].values, mp[1].values])]])
                m_y_flat_arrays_1 = pd.Series([j[:,0] for j in [np.column_stack(i) for i in np.column_stack([mp[0].values, mp[1].values])]])
                m_trans_arrays_1 = pd.concat([m_x_flat_arrays_1, m_y_flat_arrays_1], axis=1).apply(lambda x: transform(crsfrom, crsto, x[0], x[1]), axis=1)
                m_trans_geom_1_s = m_trans_arrays_1.apply(np.array).apply(np.column_stack)
                empty_s = pd.Series(range(len(mp)), index=mp.index).reset_index()
                empty_s[0] = m_trans_geom_1_s

                d['m_trans_geom_1'] = empty_s.groupby('level_0').apply(lambda x: tuple(list(x[0])))
        
            ##ndim==3
            if (mpolydims==3).any():
                m_trans_arrays_3 = a_mpoly[mpolydims==3].apply(pd.Series).stack().apply(lambda x: np.array(x)[:,[0,1]]).apply(lambda x: transform(crsfrom, crsto, x[:,0], x[:,1]))
                m_trans_geom_3_s = m_trans_arrays_3.apply(np.array).apply(np.column_stack)
                m_trans_geom_3_u = m_trans_geom_3_s.unstack()

                d['m_trans_geom_3'] = pd.Series(zip(m_trans_geom_3_u[0], m_trans_geom_3_u[1]), index=m_trans_geom_3_u.index)
        
        return pd.concat(d.values()).sort_index()
    
    
    def poly_vectorize(self, shpfile):
        s = pd.Series(range(len(shpfile)))
        
        def return_coords(x):
            try:
                return shpfile[x]['geometry']['coordinates']
            except:
                return np.nan
            
        return s.apply(return_coords)
      
    
    def handle_topo_err(self, k):
        if k.is_valid:
            return k
        else:
            return k.boundary.convex_hull
    
    def poly_return(self, shp):
        print 'creating polygons for %s...' % (shp)
        poly_df = pd.Series(index=self.shapes[shp]['shp'].index)
        
        p = self.shapes[shp]['shp'].loc[self.shapes[shp]['types']=='Polygon'].apply(lambda x: geometry.Polygon(x))
        poly_df.loc[p.index] = p
        
        mp = self.shapes[shp]['shp'].loc[self.shapes[shp]['types']== 'MultiPolygon'].apply(lambda x: (pd.Series(list(x)))).stack().apply(geometry.Polygon).apply(self.handle_topo_err).reset_index().groupby('level_0').apply(lambda x: ops.cascaded_union(list(x[0])))
        poly_df.loc[mp.index] = mp
        
        return poly_df
            
    def geom_types(self, shp):
        s = pd.Series(range(len(shp)))
        def return_geom(x):
            try:
                return shp[x]['geometry']['type']
            except:
                return np.nan
        return s.apply(return_geom)
      
    def make_kdtree(self):

        valmap = self.shapes['shp2']['poly'].apply(lambda x: x.buffer(-0.001*(x.area/x.length))).apply(pd.Series).stack().apply(lambda x: x.exterior.xy).apply(lambda x: np.column_stack([x[0], x[1]]))
	
        valshape = valmap.apply(lambda x: x.shape[0]).apply(lambda x: np.arange(x)).reset_index()

        idx_arrays = []

        for i in valshape.values:
            for j in i[2]:
                idx_arrays.append(np.array([i[0], i[1], i[2][j]]))

        idx_arrays=np.array(idx_arrays)

        midx = pd.MultiIndex.from_arrays([idx_arrays[:,0], idx_arrays[:,1], idx_arrays[:,2]])

        vals = np.concatenate(valmap.values)

        self.kdtree = spatial.cKDTree(vals)
	
        self.shp2_idx = np.array(midx.get_level_values(0))

    def query_tree(self, neighbors=1):
        shp1_c = self.shapes['shp1']['poly'].apply(lambda x: x.centroid.coords[0])
    	x_1 = shp1_c.apply(lambda x: x[0])
	y_1 = shp1_c.apply(lambda x: x[1])
	shp1_c_array = np.column_stack([x_1, y_1])
	self.shp1_centroids = shp1_c
	del shp1_c
	del x_1
	del y_1

	self.treequery = self.kdtree.query(shp1_c_array, k=neighbors)


    def check_matches(self):
        s = pd.Series(self.shapes['shp1']['shp'].index)

        geom = self.shapes['shp2']['types']
        ct_c = self.shp1_centroids.apply(geometry.Point)

	match_d = {}

	if self.treequery[1].ndim == 1:
            containing_p = self.shapes['shp2']['poly'].loc[self.shp2_idx[self.treequery[1]]].reset_index()
	    containing_bool = s.apply(lambda x: containing_p.iloc[x, 1].contains(ct_c[x]))
	    match_d.update({i : containing_p['index'].where(containing_bool)})
             
        else:
	    for i in range(self.treequery[1].shape[1]):
                containing_p = self.shapes['shp2']['poly'].loc[self.shp2_idx[self.treequery[1][:,i]]].reset_index()
	        containing_bool = s.apply(lambda x: containing_p.iloc[x, 1].contains(ct_c[x]))
	        match_d.update({i : containing_p['index'].where(containing_bool)})

        return match_d


book400 = '/home/akagi/GIS/2014_All_Parcel_Shapefiles/2014_Book400.shp'
tracts = '/home/akagi/Desktop/census_az/Arizona_Census_Tracts.shp'

b = quick_spatial_join(book400, tracts)
b.make_kdtree()
b.query_tree(neighbors=4)
matches = b.check_matches()

