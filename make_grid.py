import numpy as np
import pandas as pd
from shapely import geometry
import geopandas as gpd

#### MAKE GRID

x = np.arange(-130,-65, 0.125)
y = np.arange(24, 50, 0.125)

ssize = 0.125

xy_ll = np.vstack(np.dstack(np.meshgrid(x, y)))
xy_ur = xy_ll + ssize
xy_lr = np.column_stack([xy_ll[:,0] + ssize, xy_ll[:,1]])
xy_ul = np.column_stack([xy_ll[:,0], xy_ll[:,1] + ssize])

cs = np.column_stack([xy_ur, xy_ul, xy_ll, xy_lr, xy_ur])
cs_shape = cs.shape
cs = cs.reshape(cs_shape[0], cs_shape[1]/2, 2)

#CHEESY WAY
gs = gpd.GeoSeries(pd.Series(cs.tolist()).apply(lambda x: geometry.Polygon(x)))

#MORE ELEGANT, BUT SLOWER
#gpd.GeoSeries(map(geometry.Polygon, cs))

#ABOUT AS SLOW AS MAP
#[geometry.Polygon(i) for i in cs]
