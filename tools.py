import numpy as np
import pandas as pd
import input, output, geo
import matplotlib.pyplot as plt


def map_cn(flulc, flulcparam, fsoils, fsoilsparam, folder='C:', filename='cn'):
    """
    derive the CN map based on LULC and Soils groups
    :param flulc: string file path to lulc .asc raster file
    :param flulcparam: string file path to lulc parameters .txt file. Separator = ;
    Example:

     Value     Name  NBS  CN-A  CN-B  CN-C  CN-D
     1        Water    0   100   100   100   100
     2        Urban    0    77    85    90    92
     3       Forest    0    30    55    70    77
     4      Pasture    0    68    79    86    89
     5        Crops    0    72    81    88    91
     6   Forest NBS    1    36    60    74    80
     7  Pasture NBS    1    39    61    74    80
     8    Crops NBS    1    62    71    78    81

    :param fsoils: string path to soils.asc raster file
    :param fsoilsparam: string path to soils parameters .txt file. Separator = ;

     Value Group
     1     A
     2     B
     3     C
     4     D

    Sep=';' and must have a 'Value' field enconding the raster soil values
    :param folder: string path to destination folder
    :param filename: string name of file
    :return: string file path
    """
    #
    # import data
    metalulc, lulc = input.asc_raster(flulc)
    metasoils, soils = input.asc_raster(fsoils)
    lulc_param_df = pd.read_csv(flulcparam, sep=';')
    soils_param_df = pd.read_csv(fsoilsparam, sep=';')
    lulc_classes = lulc_param_df['Value'].values
    soils_classes = soils_param_df['Value'].values
    cn_a = lulc_param_df['CN-A'].values
    cn_b = lulc_param_df['CN-B'].values
    cn_c = lulc_param_df['CN-C'].values
    cn_d = lulc_param_df['CN-D'].values
    cn_values = (cn_a, cn_b, cn_c, cn_d)
    #
    #
    # process data
    cn_map = geo.cn(lulc=lulc, soils=soils, cnvalues=cn_values, lulcclasses=lulc_classes, soilclasses=soils_classes)
    #
    # export data
    export_file = output.asc_raster(cn_map, metalulc, folder, filename)
    return export_file


def map_grad(fslope, folder='C:', filename='grad'):
    """
    derive the topographical gradient tan(B) from the slope in degrees
    :param fslope: string path to slope in degrees raster .asc file
    :param folder: string path to destination folder
    :param filename: string of file name
    :return: string path to file
    """
    # import data
    meta, slope = input.asc_raster(fslope)
    #
    # process data
    grad = geo.grad(slope)
    #
    # export data
    export_file = output.asc_raster(grad, meta, folder, filename)
    return export_file


def lulc_areas_2(flulc, flulcparam, folder='C:', filename='lulc_areas', unit='ha'):
    """
    derive csv file of the classes areas of lulc raster
    :param flulc:  string file path to lulc .asc raster file
    :param flulcparam: string file path to lulc parameters .txt file. Separator = ;
    Example:

     Value     Name  NBS  CN-A  CN-B  CN-C  CN-D
     1        Water    0   100   100   100   100
     2        Urban    0    77    85    90    92
     3       Forest    0    30    55    70    77
     4      Pasture    0    68    79    86    89
     5        Crops    0    72    81    88    91
     6   Forest NBS    1    36    60    74    80
     7  Pasture NBS    1    39    61    74    80
     8    Crops NBS    1    62    71    78    81

    :param folder: string path to destination folder
    :param filename: string name of file
    :param unit: 'ha' for hectares, 'sqkm' for squared km
    :return: string file path
    """
    factor = 1
    if unit == 'ha':
        factor = 100
    elif unit == 'sqkm':
        factor = 1000
    else:
        factor = 1
    #
    # import data
    meta, lulc = input.asc_raster(file=flulc)
    cellsize = meta['cellsize']
    lulc_param_df = pd.read_csv(flulcparam, sep=';')
    lulc_classes = lulc_param_df['Value'].values
    lulc_names = lulc_param_df['Name'].values

    #
    # process data
    areas = geo.areas(array=lulc, cellsize=cellsize, values=lulc_classes, factor=factor)
    fractions = areas/np.sum(areas)
    #
    # export data
    exp_df = pd.DataFrame({'Value':lulc_classes, 'Name':lulc_names, 'Area':areas, 'Fraction':fractions})
    print(exp_df)
    export_file = folder + '/' +  filename + '.txt'
    exp_df.to_csv(export_file, sep=';', index=False)
    return export_file


def lulc_areas(flulcseries, flulcparam, faoi, folder='C:', filename='lulc_areas', unit='ha'):
    """
    derive csv file of the classes areas of lulc raster
    :param flulc:series  string file path to lulc series .txt file
    :param flulcparam: string file path to lulc parameters .txt file. Separator = ;
    Example:

     Value     Name  NBS  CN-A  CN-B  CN-C  CN-D
     1        Water    0   100   100   100   100
     2        Urban    0    77    85    90    92
     3       Forest    0    30    55    70    77
     4      Pasture    0    68    79    86    89
     5        Crops    0    72    81    88    91
     6   Forest NBS    1    36    60    74    80
     7  Pasture NBS    1    39    61    74    80
     8    Crops NBS    1    62    71    78    81

    :param faoi: string path to aoi.asc raster file
    :param folder: string path to destination folder
    :param filename: string name of file
    :param unit: 'ha' for hectares, 'sqkm' for squared km
    :return: string file path
    """
    factor = 1
    if unit == 'ha':
        factor = 100
    elif unit == 'sqkm':
        factor = 1000
    else:
        factor = 1
    #
    # import data
    lulc_series_df = pd.read_csv(flulcseries, sep=';')
    dates = lulc_series_df['Date'].values
    files = lulc_series_df['File'].values
    lulc_param_df = pd.read_csv(flulcparam, sep=';')
    lulc_classes = lulc_param_df['Value'].values
    lulc_names = lulc_param_df['Name'].values
    metaaoi, aoi = input.asc_raster(faoi)
    cellsize = metaaoi['cellsize']
    dct = dict()
    dct['Date'] = dates
    for i in range(len(lulc_names)):
        dct[lulc_names[i]] = list()
    #
    # process data
    for i in range(len(dates)):
        lcl_meta, lcl_lulc = input.asc_raster(files[i])
        lcl_lulc = lcl_lulc * aoi
        areas = geo.areas(array=lcl_lulc, cellsize=cellsize, values=lulc_classes, factor=factor)
        print(np.sum(areas))
        #fracs = areas / np.sum(areas)
        for j in range(len(areas)):
            dct[lulc_names[j]].append(areas[j])
    # export data
    exp_df = pd.DataFrame(dct)
    #print(exp_df.to_string(index=False))
    export_file = folder + '/' + filename + '.txt'
    exp_df.to_csv(export_file, sep=';', index=False)
    return export_file


def import_lulc_series(flulcseries, rasterfolder='C:', folder='C:', filename='lulc_series'):
    """
    import lulc series data set
    :param flulcseries: string for the input files.
    :param rasterfolder: string path to raster dataset folder
    :param folder: string path to file folder
    :param filename: string file name
    :return: string - file path
    """
    #
    # import data
    lulc_series_df = pd.read_csv(flulcseries, sep=';')
    #print(lulc_series_df)
    dates = lulc_series_df['Date'].values
    files = lulc_series_df['File'].values
    #
    # process data
    new_files = list()
    for i in range(len(dates)):
        lcl_meta, lcl_lulc = input.asc_raster(files[i])
        lcl_date = dates[i]
        lcl_filenm = 'lulc_' + str(lcl_date)
        lcl_expf = output.asc_raster(lcl_lulc, lcl_meta, folder=rasterfolder, filename=lcl_filenm)
        #print(lcl_expf)
        new_files.append(lcl_expf)
    #
    # export data
    exp_df = pd.DataFrame({'Date':dates, 'File':new_files})
    exp_file = folder + '/' + filename + '.txt'
    exp_df.to_csv(exp_file, sep=';', index=False)
    return exp_file


