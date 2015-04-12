import numpy as np
import pandas as pd
from shapely import geometry
import geopandas as gpd

x = np.arange(-130,-65, 0.125)
y = np.arange(24, 50, 0.125)

def rect_grid(bbox, hdim, vdim=None, out='polygon', pointref='centroid',
              how='fixed_step', anchor_point='ll', endpoint=True):

    if isinstance(bbox, (gpd.geoseries.GeoSeries,
                         gpd.geodataframe.GeoDataFrame)):
        bbox = bbox.total_bounds
    elif isinstance(bbox, (geometry.Point, geometry.LineString,
                           geometry.LinearRing, geometry.Polygon,
                           geometry.MultiPoint, geometry.MultiLineString,
                           geometry.MultiPolygon)):
        bbox = bbox.bounds

    x0 = bbox[0]
    y0 = bbox[1]
    x1 = bbox[2]
    y1 = bbox[3]

    if how == 'fixed_step':

        hstep = hdim
        if vdim == None:
            vstep = hstep
        else:
            vstep = vdim

        if anchor_point.lower()[1] == 'l':
            x = np.arange(x0, x1, hstep, dtype=float)
        elif anchor_point.lower()[1] == 'r':
            x = np.arange(x1, x0, -hstep, dtype=float)[::-1]
        if anchor_point.lower()[0] == 'l':
            y = np.arange(y0, y1, vstep, dtype=float)
        elif anchor_point.lower()[0] == 'u':
            y = np.arange(y1, y0, -vstep, dtype=float)[::-1]

    elif how == 'fixed_number':
        if vdim == None:
            vdim = hdim

        if anchor_point.lower()[1] == 'l':
            x, hstep = np.linspace(x0, x1, hdim, retstep=True, dtype=float, endpoint=endpoint)
        elif anchor_point.lower()[1] == 'r':
            x, hstep = np.linspace(x1, x0, hdim, retstep=True, dtype=float, endpoint=endpoint) 
            x, hstep = x[::-1], -hstep
        if anchor_point.lower()[0] == 'l':
            y, vstep = np.linspace(y0, y1, vdim, retstep=True, dtype=float, endpoint=endpoint)
        elif anchor_point.lower()[0] == 'u':
            y, vstep = np.linspace(y1, y0, vdim, retstep=True, dtype=float, endpoint=endpoint)
            y, vstep = y[::-1], -vstep


    xy_ll = np.vstack(np.dstack(np.meshgrid(x, y)))

    if out == 'point':
        if pointref == 'centroid':
            out_arr = np.column_stack([xy_ll[:,0] + hstep/2.0,
                                       xy_ll[:,1] + vstep/2.0])
            return gpd.GeoSeries(map(geometry.asPoint, out_arr))

        elif pointref == 'll':
            return gpd.GeoSeries(map(geometry.asPoint, xy_ll))

        elif pointref == 'lr':
            out_arr = np.column_stack([xy_ll[:,0] + hstep, xy_ll[:,1]])
            return gpd.GeoSeries(map(geometry.asPoint, out_arr))

        elif pointref == 'ur':
            out_arr = np.column_stack([xy_ll[:,0] + hstep, xy_ll[:,1] + vstep])
            return gpd.GeoSeries(map(geometry.asPoint, out_arr))

        elif pointref == 'ul':
            out_arr = np.column_stack([xy_ll[:,0], xy_ll[:,1] + vstep])
            return gpd.GeoSeries(map(geometry.asPoint, out_arr))

    elif out == 'line':
#        if how == 'fixed_step':
#        y1 = y1 + vstep
#        x1 = x1 + hstep

        vlines = np.hstack([
                            np.column_stack([x, np.repeat(y0, len(x))]),
                            np.column_stack([x, np.repeat(y1, len(x))])
                            ])
        vlines = vlines.reshape(vlines.shape[0], 2, 2)

        hlines = np.hstack([
                            np.column_stack([np.repeat(x0, len(y)), y]),
                            np.column_stack([np.repeat(x1, len(y)), y])
                            ])
        hlines = hlines.reshape(hlines.shape[0], 2, 2)

        out_arr = np.vstack([vlines, hlines])
        del vlines
        del hlines

        return gpd.GeoSeries(map(geometry.asLineString, out_arr))

    elif out == 'polygon':
        xy_ur = np.column_stack([xy_ll[:,0] + hstep, xy_ll[:,1] + vstep])
        out_arr = np.column_stack([
                            xy_ur,
                            np.column_stack([xy_ll[:,0], xy_ll[:,1] + vstep]),
                            xy_ll,
                            np.column_stack([xy_ll[:,0] + hstep, xy_ll[:,1]]),
                            xy_ur
                            ])
        del xy_ll
        del xy_ur

        out_arr = out_arr.reshape(out_arr.shape[0], out_arr.shape[1]/2, 2)
        return gpd.GeoSeries(map(geometry.asPolygon, out_arr))

