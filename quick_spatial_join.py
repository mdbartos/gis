import pandas as pd
import numpy as np
import fiona
from shapely import geometry
from shapely import ops
from itertools import chain
from pyproj import Proj, transform

states = fiona.open('/home/akagi/GIS/census/cb_2013_us_state_500k/cb_2013_us_state_500k.shp', 'r')

county = fiona.open('/home/akagi/GIS/census/cb_2013_us_county_500k/cb_2013_us_county_500k.shp', 'r')

book400 = fiona.open('/home/akagi/GIS/2014_All_Parcel_Shapefiles/2014_Book400.shp', 'r')

zipcodes = fiona.open('/home/akagi/Desktop/census_az/Arizona_ZIP_Codes.shp', 'r') 


class quick_spatial_join():
    def __init__(self, shp1, shp2, convert_crs=0):
        
        self.shapes = {
                        'shp1' : {
                            'file' : shp1,
                            'crs' : shp1.crs,
                            'types' : self.geom_types(shp1)
                            },
                
                        'shp2': {
                            'file' : shp2,
                            'crs' : shp2.crs,
                            'types' : self.geom_types(shp2)
                            }
                }
        
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
      
    def poly_len(self, shpfile):
        s = pd.Series(range(len(shpfile)))
        return s.apply(lambda x: len(shpfile[x]['geometry']['coordinates']))
    
    def poly_return(self, shp):
        print 'creating polygons for %s...' % (shp)
        poly_df = pd.Series(index=self.shapes[shp]['shp'].index)
        
        p = self.shapes[shp]['shp'].loc[self.shapes[shp]['types']=='Polygon'].apply(lambda x: geometry.Polygon(x))
        poly_df.loc[p.index] = p
        
        mp = self.shapes[shp]['shp'].loc[self.shapes[shp]['types']== 'MultiPolygon'].apply(lambda x: (pd.Series(list(x)))).stack().apply(geometry.Polygon).reset_index().groupby('level_0').apply(lambda x: geometry.MultiPolygon(list(x[0])))
        poly_df.loc[mp.index] = mp
        
        return poly_df
            
    def geom_types(self, shp):
        s = pd.Series(range(len(shp)))
        return s.apply(lambda x: shp[x]['geometry']['type'])
    
    def pt_in_bounds(self, x_coord, y_coord, shp_bounds):
        return list(shp_bounds.loc[(shp_bounds['xmin'] < x_coord) & (shp_bounds['ymin'] < y_coord) & (shp_bounds['xmax'] > x_coord) & (shp_bounds['ymax'] > y_coord)].index)
    
    def join_shapes(self):
        shp1_c = self.shapes['shp1']['poly'].apply(lambda x: pd.Series(x.centroid.coords[0])).rename(columns={0:'x', 1:'y'})
        shp2_bounds = self.shapes['shp2']['poly'].apply(lambda x: pd.Series(x.bounds)).rename(columns={0:'xmin', 1:'ymin', 2:'xmax', 3:'ymax'})
        c_memb = shp1_c.apply(lambda x: self.pt_in_bounds(x['x'], x['y'], shp2_bounds), axis=1).apply(pd.Series)
        c_memb.columns = [str(i) for i in c_memb.columns]
        geom = self.shapes['shp2']['types']
        ct_c = shp1_c.apply(geometry.Point, axis=1)
    
        def points_in_poly(): 
            c_poly = c_memb.dropna(how='all').dropna(axis=1, how='all').stack().astype(int).reset_index().set_index(0).sort_index()['level_0'].reset_index()        
            
            c_poly['poly'] = c_poly[0].map(self.shapes['shp2']['poly'])
            c_poly['pt'] = c_poly['level_0'].map(ct_c)
            
            return c_poly[[0, 'level_0']][c_poly.apply(lambda x: x['poly'].contains(x['pt']), axis=1)].set_index('level_0')

      
        return points_in_poly().sort_index()
