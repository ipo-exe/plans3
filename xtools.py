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

    # from geo.py
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


def bulk_ndvi(red_dir, nir_dir, out_dir='C:/bin', label='calib_ndvi', load=True):
    """

    Bulk tool for processing NDVI files.
    All raster tif file names must present the date at the end. Example:
    ../red_band_2013-01-08.tif
    All raster files must match in number and dates

    :param red_dir: string path to RED directory
    :param nir_dir: string path to NIR directory
    :param out_dir: string path to OUTPUT directory
    :return: none
    """
    import gdal
    import os
    import numpy as np

    def ndvi(nir, red):
        """
        Normalized Difference Vegetation Index - NDVI
        :param nir: 2d numpy array of NIR (Near Infrared) - Band 5 of Landsat 8
        :param red: 2d numpy array of Red - Band 4 of Landsat 8
        :return: 2d numpy array of NDVI
        """
        return (nir - red) / (nir + red)

    # load files
    files_red = list()
    for file in os.listdir(red_dir):
        if file.endswith(".tif"):
            files_red.append(file)
    files_nir = list()
    for file in os.listdir(nir_dir):
        if file.endswith(".tif"):
            files_nir.append(file)

    for t in range(len(files_red)):
        # get local date
        lcl_date = files_red.split('.')[0][-10:]

        lcl_red_file = red_dir + '/' +  files_red[t]
        lcl_nir_file = nir_dir + '/' + files_nir[t]

        rst_input1 = gdal.Open(lcl_red_file, 0)
        rst_input2 = gdal.Open(lcl_nir_file, 0)
        lcl_red = rst_input1.GetRasterBand(1).ReadAsArray()
        lcl_nir = rst_input2.GetRasterBand(1).ReadAsArray()
        #
        #
        # 3 e 4) Obter os metadados da camada de entrada e criar camada de saida
        fnm_output = out_dir + '/' + label + '_' + lcl_date + '.tif'

        # carregar o driver para GeoTiff
        driver_tiff = gdal.GetDriverByName('GTiff')
        # criar um raster de saida com os metadados da camada de entrada
        x = rst_input1.RasterXSize
        y = rst_input1.RasterYSize
        # camada raster de saida
        rst_out = driver_tiff.Create(fnm_output, xsize=x, ysize=y, bands=1, eType=gdal.GDT_Float32)
        # configurar a geotransformacao com base na camada de entrada
        rst_out.SetGeoTransform(rst_input1.GetGeoTransform())
        # configurar a projecao com base na camada de entrada
        rst_out.SetProjection(rst_input1.GetProjection())
        # ler a banda 1 da camada de saida
        bnd1 = rst_out.GetRasterBand(1).ReadAsArray()
        #
        #
        #
        bnd1 = ndvi(nir=lcl_nir, red=lcl_red)
        #
        #
        # 6 e 7) EXPORTAR CAMADA DE SAIDA
        # sobrescrever a banda 1 do raster de saida
        rst_out.GetRasterBand(1).WriteArray(bnd1)
        # fechar a camada
        rst_out = None
        if load:
            # carregar no QGIS
            iface.addRasterLayer(fnm_output)


def bulk_ndwi_v(swir_dir, nir_dir, out_dir='C:/bin', label='calib_ndwi_v', load=True):
    """

    Bulk tool for processing NDWI_V files.
    All raster tif file names must present the date at the end. Example:
    ../nir_band_2013-01-08.tif
    All raster files must match in number and dates

    :param swir_dir: string path to SWIR directory
    :param nir_dir: string path to NIR directory
    :param out_dir: string path to OUTPUT directory
    :return: none
    """
    import gdal
    import os
    import numpy as np

    def ndwi_v(nir, swir):
        """
        Normalized Difference Wetness Index of Vegetation - NDWIv
        :param nir: 2d numpy array of NIR (Near Infrared) - Band 5 of Landsat 8
        :param swir: 2d numpy array of SWIR (Shortwave Infrared) - Band 6 of Landsat 8
        :return: 2d numpy array of NDWI_v
        """
        return (nir - swir) / (nir + swir)

    # load files
    files_swir = list()
    for file in os.listdir(swir_dir):
        if file.endswith(".tif"):
            files_swir.append(file)
    files_nir = list()
    for file in os.listdir(nir_dir):
        if file.endswith(".tif"):
            files_nir.append(file)

    for t in range(len(files_swir)):
        # get local date
        lcl_date = files_swir.split('.')[0][-10:]

        lcl_swir_file = swir_dir + '/' + files_swir[t]
        lcl_nir_file = nir_dir + '/' + files_nir[t]

        rst_input1 = gdal.Open(lcl_swir_file, 0)
        rst_input2 = gdal.Open(lcl_nir_file, 0)
        lcl_swir = rst_input1.GetRasterBand(1).ReadAsArray()
        lcl_nir = rst_input2.GetRasterBand(1).ReadAsArray()
        #
        #
        # 3 e 4) Obter os metadados da camada de entrada e criar camada de saida
        fnm_output = out_dir + '/' + label + '_' + lcl_date + '.tif'

        # carregar o driver para GeoTiff
        driver_tiff = gdal.GetDriverByName('GTiff')
        # criar um raster de saida com os metadados da camada de entrada
        x = rst_input1.RasterXSize
        y = rst_input1.RasterYSize
        # camada raster de saida
        rst_out = driver_tiff.Create(fnm_output, xsize=x, ysize=y, bands=1, eType=gdal.GDT_Float32)
        # configurar a geotransformacao com base na camada de entrada
        rst_out.SetGeoTransform(rst_input1.GetGeoTransform())
        # configurar a projecao com base na camada de entrada
        rst_out.SetProjection(rst_input1.GetProjection())
        # ler a banda 1 da camada de saida
        bnd1 = rst_out.GetRasterBand(1).ReadAsArray()
        #
        #
        #
        bnd1 = ndwi_v(nir=lcl_nir, swir=lcl_swir)
        #
        #
        # 6 e 7) EXPORTAR CAMADA DE SAIDA
        # sobrescrever a banda 1 do raster de saida
        rst_out.GetRasterBand(1).WriteArray(bnd1)
        # fechar a camada
        rst_out = None
        if load:
            # carregar no QGIS
            iface.addRasterLayer(fnm_output)


def bulk_ndwi_w(green_dir, nir_dir, out_dir='C:/bin', label='calib_ndwi_w', load=True):
    """

    Bulk tool for processing NDWI_W files.
    All raster tif file names must present the date at the end. Example:
    ../nir_band_2013-01-08.tif
    All raster files must match in number and dates

    :param green_dir: string path to GREEN directory
    :param nir_dir: string path to NIR directory
    :param out_dir: string path to OUTPUT directory
    :return: none
    """
    import gdal
    import os
    import numpy as np

    def ndwi_w(green, nir):
        """
        Normalized Difference Wetness Index of Water - NDWIw
        :param green: 2d numpy array of Green - Band 3 of Landsat 8
        :param nir: 2d numpy array of NIR (Near Infrared) - Band 5 of Landsat 8
        :return: 2d numpy array of NDWIw
        """
        return (green - nir) / (green + nir)

    # load files
    files_green = list()
    for file in os.listdir(green_dir):
        if file.endswith(".tif"):
            files_green.append(file)
    files_nir = list()
    for file in os.listdir(nir_dir):
        if file.endswith(".tif"):
            files_nir.append(file)

    for t in range(len(files_green)):
        # get local date
        lcl_date = files_green.split('.')[0][-10:]

        lcl_green_file = green_dir + '/' + files_green[t]
        lcl_nir_file = nir_dir + '/' + files_nir[t]

        rst_input1 = gdal.Open(lcl_green_file, 0)
        rst_input2 = gdal.Open(lcl_nir_file, 0)
        lcl_green = rst_input1.GetRasterBand(1).ReadAsArray()
        lcl_nir = rst_input2.GetRasterBand(1).ReadAsArray()
        #
        #
        # 3 e 4) Obter os metadados da camada de entrada e criar camada de saida
        fnm_output = out_dir + '/' + label + '_' + lcl_date + '.tif'

        # carregar o driver para GeoTiff
        driver_tiff = gdal.GetDriverByName('GTiff')
        # criar um raster de saida com os metadados da camada de entrada
        x = rst_input1.RasterXSize
        y = rst_input1.RasterYSize
        # camada raster de saida
        rst_out = driver_tiff.Create(fnm_output, xsize=x, ysize=y, bands=1, eType=gdal.GDT_Float32)
        # configurar a geotransformacao com base na camada de entrada
        rst_out.SetGeoTransform(rst_input1.GetGeoTransform())
        # configurar a projecao com base na camada de entrada
        rst_out.SetProjection(rst_input1.GetProjection())
        # ler a banda 1 da camada de saida
        bnd1 = rst_out.GetRasterBand(1).ReadAsArray()
        #
        #
        #
        bnd1 = ndwi_w(green=lcl_green, nir=lcl_nir)
        #
        #
        # 6 e 7) EXPORTAR CAMADA DE SAIDA
        # sobrescrever a banda 1 do raster de saida
        rst_out.GetRasterBand(1).WriteArray(bnd1)
        # fechar a camada
        rst_out = None
        if load:
            # carregar no QGIS
            iface.addRasterLayer(fnm_output)
