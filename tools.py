import numpy as np
import pandas as pd
import input, output, geo
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter


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


def map_twi(fslope, fcatcha, folder='C:', filename='twi'):
    """
    Derive the Topographical Wetness Index of TOPMODEL (Beven & Kirkby, 1979)
    :param fslope: string path to slope in degrees raster .asc file
    :param fcatcha: string path to catchment area in squared meters raster .asc file
    :param folder: string path to destination folder
    :param filename: string of file name
    :return: string path to file
    """
    # import data
    meta, slope = input.asc_raster(fslope)
    meta, catcha = input.asc_raster(fcatcha)
    # process data
    grad = geo.grad(slope)
    twi = geo.twi(catcha, grad, cellsize=meta['cellsize'])
    # export data
    export_file = output.asc_raster(twi, meta, folder, filename)
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
    from shutil import copyfile
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
        src = files[i]
        lcl_date = dates[i]
        lcl_filenm = 'lulc_' + str(lcl_date) + '.asc'
        dst = rasterfolder + '/' + lcl_filenm
        copyfile(src=src, dst=dst)
        #print(lcl_expf)
        new_files.append(dst)
    #
    # export data
    exp_df = pd.DataFrame({'Date':dates, 'File':new_files})
    exp_file = folder + '/' + filename + '.txt'
    exp_df.to_csv(exp_file, sep=';', index=False)
    return exp_file


def cn_series(flulcseries, flulcparam, fsoils, fsoilsparam, rasterfolder='C:', folder='C:', filename='cn_series'):
    """
    derive the CN series raster and txt file
    :param flulcseries: string filepath to lulc series txt
    :param flulcparam: string filepath to lulc param txt
    :param fsoils: string filepath to soils .asc rasterfiles
    :param fsoilsparam: string filepath to soils parameters
    :param rasterfolder: string filepath to raster folder
    :param folder: string filepath to return file
    :param filename: string of filename
    :return: string filepath for txt file
    """
    # import data
    lulc_series_df = pd.read_csv(flulcseries, sep=';')
    # print(lulc_series_df)
    dates = lulc_series_df['Date'].values
    files = lulc_series_df['File'].values
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
    # process data
    new_files = list()
    for i in range(len(dates)):
        metalulc, lulc = input.asc_raster(files[i])
        lcl_date = dates[i]
        # process data
        cn_map = geo.cn(lulc=lulc, soils=soils, cnvalues=cn_values, lulcclasses=lulc_classes, soilclasses=soils_classes)
        lcl_filenm = 'cn_' + str(lcl_date)
        dst = output.asc_raster(cn_map, metalulc, rasterfolder, lcl_filenm)
        # print(lcl_expf)
        new_files.append(dst)
    #
    # export data
    exp_df = pd.DataFrame({'Date': dates, 'File': new_files})
    exp_file = folder + '/' + filename + '.txt'
    exp_df.to_csv(exp_file, sep=';', index=False)
    return exp_file


def import_climpat(fclimmonth, rasterfolder='C:', folder='C:', filename='clim_month', alias='p'):
    """

    :param fclimmonth: string filepath to climate raster monthly pattern series txt file
    :param rasterfolder: string filepath to raster folder
    :param folder: string filepath to folder
    :param filename: string of filename
    :param alias: string climate alies
    :return: string filepath
    """
    from shutil import copyfile

    # import data
    clim_df = pd.read_csv(fclimmonth, sep=';')
    #print(clim_df)
    months = clim_df['Month'].values
    files = clim_df['File'].values
    #
    # process data
    new_files = list()
    for i in range(len(months)):
        src = files[i]
        lcl_month = months[i]
        lcl_filenm =  alias + 'pat_' + str(lcl_month) + '.asc'
        dst = rasterfolder + '/' + lcl_filenm
        copyfile(src=src, dst=dst)
        # print(lcl_expf)
        new_files.append(dst)
    #
    # export data
    exp_df = pd.DataFrame({'Month': months, 'File': new_files})
    exp_file = folder + '/' + filename + '.txt'
    exp_df.to_csv(exp_file, sep=';', index=False)
    return exp_file


def series_calib_month(fseries, faoi, folder='C:', filename='series_calib_month'):
    """
    Derive the monthly series of calibration file and ET and C variables (monthly)
    Variables must be: Date, Prec, Flow, Temp. Units: mm, m3/s, celsius

    output variables: Date, Temp, Prec, ET, Flow, C. Units: mm, mm, mm, mm/mm

    :param fseries: string filepath to txt file of daily timeseries
    :param faoi: string filepatht o aoi.asc raster file (watershed area in boolean image)
    :param folder: string filepath to folder
    :param filename: string filename
    :return: string filepath to monthtly series
    """
    import resample, hydrology
    # import data
    series_df = pd.read_csv(fseries, sep=';')
    meta, aoi = input.asc_raster(faoi)
    cell = meta['cellsize']
    # process data
    area = np.sum(aoi) * cell * cell
    #print('Area {}'.format(area))
    series_temp = resample.d2m_clim(series_df, var_field='Temp')
    series_flow = resample.d2m_flow(series_df, var_field='Flow')
    series_prec = resample.d2m_prec(series_df, var_field='Prec')
    #
    prec = series_prec['Sum'].values
    # convert flow to mm
    flow = series_flow['Sum'].values
    spflow = 1000 * flow / area
    #
    # monthly C and evap
    c_month = spflow/prec
    evap_month = prec - spflow
    #
    # export data
    exp_df = pd.DataFrame({'Date':series_temp['Date'], 'Temp':series_temp['Mean'],
                           'Prec':prec, 'ET':evap_month, 'Flow':spflow, 'C':c_month})
    print(exp_df.head())
    exp_file = folder + '/' + filename + '.txt'
    exp_df.to_csv(exp_file, sep=';', index=False)
    return exp_file


def run_topmodel(fseries, fparam, faoi, ftwi, fcn, folder='C:', tui=False):
    """

    :param fseries: string file path to input series. Required fields: 'Date', 'Prec', 'Temp'
    :param fparam: string file path to parameters dataframe.
    Parameters names must be in index. Parameters values must be in a filed called 'Set'.
    Order and names: m, ksat, qo, a, c, k, n
    :param faoi: string file path to AOI raster in .asc format
    :param ftwi: string file path to TWI raster in .asc format
    :param fcn: string file path to CN raster in .asc format
    :param folder: string file path to destination folder
    :param tui: boolean control to printout messages
    :return:
    """
    from hydrology import avg_2d, topmodel_hist, topmodel_sim
    import time
    #
    if tui:
        print('loading series...')
    lcl_df = pd.read_csv(fseries, sep=';', parse_dates=['Date'])
    #lcl_df.query('Date > "2005-03-15" and Date <= "2005-07-15"', inplace=True)
    #lcl_df.query('Date > "2005-03-15" and Date <= "2010-03-15"', inplace=True)
    #
    # import aoi raster
    if tui:
        print('loading aoi...')
    meta, aoi = input.asc_raster(faoi)
    cell = meta['cellsize']
    #
    # import twi raster
    if tui:
        print('loading twi...')
    meta, twi = input.asc_raster(ftwi)
    #
    # import CN raster
    if tui:
        print('loading cn...')
    meta, cn = input.asc_raster(fcn)
    #
    # import parameters
    if tui:
        print('loading parameters...')
    df_param = pd.read_csv(fparam, sep=';', index_col='Parameter')
    m = df_param.loc['m'].values[0]
    ksat = df_param.loc['ksat'].values[0]
    qo = df_param.loc['qo'].values[0]
    a = df_param.loc['a'].values[0]
    c = df_param.loc['c'].values[0]
    k = df_param.loc['k'].values[0]
    n = df_param.loc['n'].values[0]
    qt0 = 0.007  # mm/d
    lamb = avg_2d(var2d=twi, weight=aoi)
    #
    # compute histograms
    if tui:
        print('computing histograms...', end='\t\t')
    init = time.time()
    countmatrix, twihist, cnhist = topmodel_hist(twi=twi, cn=cn, aoi=aoi)
    end = time.time()
    if tui:
        print('Enlapsed time: {:.3f} seconds'.format(end - init))
    #
    # run topmodel simulation
    if tui:
        print('running simulation...', end='\t\t')
    init = time.time()
    sim_df = topmodel_sim(lcl_df, twihist, cnhist, countmatrix, lamb=lamb, ksat=ksat, m=m, qo=qo, a=a, c=c, qt0=qt0, k=k, n=n)
    end = time.time()
    if tui:
        print('Enlapsed time: {:.3f} seconds'.format(end - init))
    #
    # export files
    if tui:
        print('exporting run parameters...')
    exp_df = pd.DataFrame({'Parameter':('m', 'ksat', 'qo', 'a', 'c', 'k', 'n'),
                           'Set':(m, ksat, qo, a, c, k, n)})
    exp_file1 = folder + '/' + 'parameters.txt'
    exp_df.to_csv(exp_file1, sep=';', index=False)
    #
    # export histograms
    if tui:
        print('exporting histograms...')
    exp_df = pd.DataFrame(countmatrix, index=twihist[0], columns=cnhist[0])
    exp_file2 = folder + '/' + 'histograms.txt'
    exp_df.to_csv(exp_file2, sep=';', index_label='TWI')
    #
    # export simulation
    if tui:
        print('exporting simulation results...')
    exp_file3 = folder + '/' + 'simseries.txt'
    sim_df.to_csv(exp_file3, sep=';', index=False)
    #
    return (exp_file1, exp_file2, exp_file3)

