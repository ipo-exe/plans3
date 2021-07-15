'''
Tools and scripts for pre-processing. Requires GDAL.
'''

def convert_mapbiomas(fmapb, flulc, load=True):
    """
    Convertion from MAPBIOMAS datasets to a simpler classification:

    0 - none
    1 - water
    2 - urban
    3 - forest
    4 - pasture
    5 - crops

    :param fmapb: string file path to mapbiomas .tif raster file
    :param flulc: string file path to new .tif raster file
    :param load: boolean to load layer to QGIS
    :return: none
    """
    import gdal
    import os
    import numpy as np

    def reclassify(array, upvalues, classes):
        """
        utility function -
        Reclassify array based on list of upper values and list of classes values

        :param array: 2d numpy array to reclassify
        :param upvalues: 1d numpy array of upper values
        :param classes: 1d array of classes values
        :return: 2d numpy array reclassified
        """
        new = array * 0.0
        for i in range(len(upvalues)):
            if i == 0:
                new = new + ((array <= upvalues[i]) * classes[i])
            else:
                new = new + ((array > upvalues[i - 1]) * (array <= upvalues[i]) * classes[i])
        return new

    # extract array from mapbiomas raster file
    rst_input1 = gdal.Open(fmapb, 0)
    mapb = rst_input1.GetRasterBand(1).ReadAsArray()

    # create output dataset
    #
    # driver
    driver_tiff = gdal.GetDriverByName('GTiff')
    # size
    x = rst_input1.RasterXSize
    y = rst_input1.RasterYSize
    # raster
    rst_out = driver_tiff.Create(flulc, xsize=x, ysize=y, bands=1, eType=gdal.GDT_Float32)
    # set geotransform
    rst_out.SetGeoTransform(rst_input1.GetGeoTransform())
    # set projection
    rst_out.SetProjection(rst_input1.GetProjection())
    # read band 1 array
    bnd1 = rst_out.GetRasterBand(1).ReadAsArray()

    # processing section:
    upvals = [0, 9, 15, 21, 24, 33]
    clasvals = [0, 3, 4, 5, 2, 1]
    # reclassify
    bnd1 = reclassify(mapb, upvals, clasvals)

    # exporting section
    # overwrite raster band
    rst_out.GetRasterBand(1).WriteArray(bnd1)
    # fechar a camada
    rst_out = None
    if load:
        # carregar no QGIS
        iface.addRasterLayer(flulc)



