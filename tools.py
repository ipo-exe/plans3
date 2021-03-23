import numpy as np
import pandas as pd
import input, output, geo
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter


def map_cn(flulc, flulcparam, fsoils, fsoilsparam, folder='C:/bin', filename='cn'):
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


def map_grad(fslope, folder='C:/bin', filename='grad'):
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


def map_twi(fslope, fcatcha, folder='C:/bin', filename='twi'):
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


def lulc_areas(flulcseries, flulcparam, faoi, folder='C:/bin', filename='lulc_areas', unit='ha'):
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


def import_lulc_series(flulcseries, rasterfolder='C:/bin', folder='C:/bin', filename='lulc_series'):
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


def cn_series(flulcseries, flulcparam, fsoils, fsoilsparam, rasterfolder='C:/bin', folder='C:/bin', filename='cn_series'):
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


def map_cn_avg(fcnseries, fseries, folder='C:/bin', filename='cn_calib'):
    """
    Derive the average CN given a time series.
    :param fcnseries: string file path to CN series dataframe. Required fields: 'Date' and 'File'
    :param fseries: string file path to series dataframe. Required field: 'Date'
    :param folder: string file path to destination directory
    :param filename: string file name (without extension)
    :return: string file path to derived file
    """
    cnseries_df = pd.read_csv(fcnseries, sep=';', parse_dates=['Date'])
    series_df = pd.read_csv(fseries, sep=';', parse_dates=['Date'])
    datemin = series_df['Date'].min()
    datemax = series_df['Date'].max()
    expression = 'Date >= "{}" and Date <= "{}"'.format(datemin, datemax)
    cnseries_df.query(expr=expression, inplace=True)
    files = cnseries_df['File'].values
    size = len(files)
    for i in range(size):
        if i == 0:
            meta, lcl_cn1 = input.asc_raster(files[i])
        else:
            meta, lcl_cn2 = input.asc_raster(files[i])
            lcl_cn1 = lcl_cn1 + lcl_cn2
    cn_avg = lcl_cn1 / size
    #
    exp_file = output.asc_raster(array=cn_avg, meta=meta, folder=folder, filename=filename)
    return exp_file


def import_climpat(fclimmonth, rasterfolder='C:/bin', folder='C:/bin', filename='clim_month', alias='p'):
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


def series_calib_month(fseries, faoi, folder='C:/bin', filename='series_calib_month'):
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


def run_topmodel(fseries, fparam, faoi, ftwi, fcn, folder='C:/bin', tui=False, mapback=False, mapvar='R-ET-S1-S2-Qv'):
    """

    Run the PLANS3 TOPMODEL

    :param fseries: string file path to input series dataframe.
    Field separator = ';'
    Required fields: 'Date', 'Prec', 'Temp'

    Date = daily data in YYYY-MM-DD
    Prec = precipitation in mm
    Temp = temperature in Celsius

    Example of dataframe formatting:

    Date;		Prec;	Temp;
    2005-01-01;	0.0;	25.96;
    2005-01-02;	0.0;	26.96;
    2005-01-03;	7.35;	25.2;
    2005-01-04;	3.95;	27.76;
    2005-01-05;	0.0;	27.24;
       ...		...	    ...
    2015-01-06;	0.0;	27.52;
    2015-01-07;	0.0;	28.84;
    2015-01-08;	12.5;	28.44;

    :param fparam: string file path to parameters dataframe.

    Required fields: 'Parameter' and 'Set' , where:
    Parameter =  string of parameter name
    Set = float of parameter value

    Order and names: m, ksat, qo, a, c, k, n

    Example of dataframe formatting:

    Parameter;  Set;
    m;          5.0;
    ksat;       2.0;
    qo;         1.0;
    a;          1.5;
    c;          100.0;
    lat;        -30.0;
    k;          1.1;
    n;          2.1;

    :param faoi: string file path to AOI raster in .asc format.
    The AOI raster should be a pseudo-boolean image of the watershed area

    :param ftwi: string file path to TWI raster in .asc format.

    :param fcn: string file path to CN raster in .asc format.

    Note: all rasters must have the same shape (rows x columns)

    :param folder: string file path to destination folder
    :param tui: boolean control to allow terminal messages
    :param mapback: boolean control to map simulated variables
    :param mapvar: string code to set mapped variables (see topmodel routine for code formatting)
    :return:
    when mapback=False:
    tuple of 3 string file paths:
    1) simulated parameters dataframe
    2) simulated histograms dataframes (count matrix used)
    3) simulated of global variables series dataframe
    4) visual pannel string file path
    when mapback=True:
    tuple of 4 elements:
    1) string file path of simulated parameters dataframe
    2) string file path of simulated histograms dataframes (count matrix used)
    3) string file path of simulated of global variables series dataframe
    5) visual pannel string file path
    4) tuple of string file paths of map lists files
    """
    #
    from hydrology import avg_2d, topmodel_hist, topmodel_sim, map_back
    from visuals import pannel_topmodel
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
    lat = df_param.loc['lat'].values[0]
    k = df_param.loc['k'].values[0]
    n = df_param.loc['n'].values[0]
    qt0 = 0.2  # mm/d
    lamb = avg_2d(var2d=twi, weight=aoi)
    #
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
    #
    # run topmodel simulation
    if tui:
        print('running simulation...', end='\t\t')
    init = time.time()
    # mapback conditionals:
    if mapback:
        sim_df, mapped = topmodel_sim(lcl_df, twihist, cnhist, countmatrix, lamb=lamb, ksat=ksat, m=m, qo=qo, a=a, c=c,
                                      lat=lat, qt0=qt0, k=k, n=n, tui=False, mapback=mapback, mapvar=mapvar)
    else:
        sim_df = topmodel_sim(lcl_df, twihist, cnhist, countmatrix, lamb=lamb, ksat=ksat, m=m, qo=qo, a=a, c=c,
                              qt0=qt0, k=k, n=n, tui=False, mapback=mapback)
    end = time.time()
    if tui:
        print('Enlapsed time: {:.3f} seconds'.format(end - init))
    #
    #
    #
    # export files
    if tui:
        print('exporting run parameters...')
    exp_df = pd.DataFrame({'Parameter':('m', 'ksat', 'qo', 'a', 'c', 'lat', 'k', 'n'),
                           'Set':(m, ksat, qo, a, c, lat, k, n)})
    exp_file1 = folder + '/' + 'parameters.txt'
    exp_df.to_csv(exp_file1, sep=';', index=False)
    #
    # export histograms
    if tui:
        print('exporting histograms...')
    exp_df = pd.DataFrame(countmatrix, index=twihist[0], columns=cnhist[0])
    exp_file2 = folder + '/' + 'histograms.txt'
    exp_df.to_csv(exp_file2, sep=';', index_label='TWI\CN')
    #
    # export simulation
    if tui:
        print('exporting simulation results...')
    exp_file3 = folder + '/' + 'simseries.txt'
    sim_df.to_csv(exp_file3, sep=';', index=False)
    #
    # export visual pannel
    if tui:
        print('exporting visual results...')
    sim_df['Qobs'] = lcl_df['Q']
    exp_file4 = pannel_topmodel(sim_df, grid=False, show=False, qobs=True, folder=folder)
    #
    if mapback:
        if tui:
            print('exporting variable maps...', end='\t\t')
        init = time.time()
        #
        from os import mkdir
        mapvar_lst = mapvar.split('-')  # load string variables alias to list
        mapfiles_lst = list()
        stamp = pd.to_datetime(sim_df['Date'], format='%y-%m-%d')
        for var in mapvar_lst:  # loop across all variables
            lcl_folder = folder + '/' + var
            mkdir(lcl_folder)  # make diretory
            lcl_files = list()
            for t in range(len(stamp)):  # loop across all timesteps
                lcl_filename = var + '_' + str(stamp[t]).split(sep=' ')[0] + '.txt'
                lcl_file = lcl_folder + '/' + lcl_filename
                lcl_files.append(lcl_file)
                # export local dataframe to text file in local folder
                lcl_exp_df = pd.DataFrame(mapped[var][t], index=twihist[0], columns=cnhist[0])
                lcl_exp_df.to_csv(lcl_file, sep=';', index_label='TWI\CN')
                # map = map_back(zmatrix=mapped[var][t], a1=twi, a2=cn, bins1=twihist[0], bins2=cnhist[0])
                # plt.imshow(map[550:1020, 600:950], cmap='jet_r')
                # plt.show()
            # export map list file to main folder:
            lcl_exp_df = pd.DataFrame({'Date': sim_df['Date'], 'File': lcl_files})
            lcl_file = folder + '/' + var + '_maps' + '.txt'
            lcl_exp_df.to_csv(lcl_file, sep=';', index=False)
            mapfiles_lst.append(lcl_file)
        #
        mapfiles_lst = tuple(mapfiles_lst)
        end = time.time()
        if tui:
            print('Enlapsed time: {:.3f} seconds'.format(end - init))
    #
    if mapback:
        (exp_file1, exp_file2, exp_file3, exp_file4, mapfiles_lst)
    else:
        return (exp_file1, exp_file2, exp_file3, exp_file4)


def integrate_map(ftwi, fcn, faoi, fmaps, filename, yfield='TWI\CN', folder='C:/bin', tui=False, show=False):
    """

    Integrate (adds up) variable maps based on Z-Maps of TOPMODEL.

    :param ftwi: string file path to TWI raster in .asc format.
    :param fcn: string file path to CN raster in .asc format.
    :param faoi: string file path to AOI raster in .asc format.
    The AOI raster should be a pseudo-boolean image of the watershed area

    Note: all rasters must have the same shape (rows x columns)

    :param fmaps: string file maps to .txt file of listed z-map files of desired variable
    :param filename: string of output file name (without path and extension)
    :param yfield: string of fieldname Y variable on z-map dataframe
    :param folder: string path to output directory
    :param tui: boolean to control TUI displays
    :param show: boolean to control plotting displays
    :return: string file path to output file
    """
    from hydrology import map_back
    # import twi
    meta, twi = input.asc_raster(ftwi)
    #
    # import cn
    meta, cn = input.asc_raster(fcn)
    #
    # import aoi
    meta, aoi = input.asc_raster(faoi)
    #
    # read file of maps files
    maps_df = pd.read_csv(fmaps, sep=';')
    #
    # extract the length of integral
    integral_len = len(maps_df['File'])
    #
    # set up a blank map:
    integral_map = np.zeros(shape=np.shape(twi))
    #
    # loop across all maps:
    for i in range(integral_len):
        if tui:
            print('{:.1f} %'.format(100 * i / integral_len))
        #
        # retrieve zmap file:
        lcl_fmap = maps_df['File'][i]
        #
        # extract zmap and histograms bins
        lcl_zmap, hist_twi, hist_cn = input.zmap(file=lcl_fmap, yfield=yfield)
        #
        # hidden plotting procedure:
        """
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # Make data.
        X = np.arange(0, len(lcl_xhist))
        Y = np.arange(0, len(lcl_yhist))
        X, Y = np.meshgrid(X, Y)
        #R = np.sqrt(X ** 2 + Y ** 2)
        #Z = np.sin(R)
        Z = lcl_zmap
        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=False)
        plt.show()
        """
        #
        # map back:
        lcl_map = map_back(zmatrix=lcl_zmap, a1=twi, a2=cn, bins1=hist_twi, bins2=hist_cn)
        #
        # accumulate values:
        integral_map = integral_map + lcl_map
    #
    # show conditional
    if show:
        im =geo.mask(integral_map, aoi)
        plt.imshow(im, cmap='jet')
        plt.show()
    # export file
    fout = output.asc_raster(integral_map, meta=meta, filename=filename, folder=folder)
    return fout



def frames_topmodel_twi(fseries, ftwi, faoi, fparam, var1='Prec', var2='Qb', var3='D',
                       size=100, start=0, framef=0.2, watchf=0.2, folder='C:/bin'):
    """
    Create frames for video watch of topmodel simulation with TWI map alongside 3 three variables
    :param fseries: string filepath to simulation series dataframe
    :param ftwi: string filepath to raster asc file of TWI
    :param faoi: string filepath to raster asc file of AOI
    :param fparam: string filepath to simulation parameters dataframe
    :param var1: string of first variable field name
    :param var2: string of second variable field name
    :param var3: string of second variable field name
    :param size: int number of frames
    :param start: int index of first frame
    :param framef: float of frame ratio to full size (0 to 1)
    :param watchf: float of watch line ratio to full frame (0 to 1)
    :param folder: string path to destination folder
    :return: none
    """
    from hydrology import topmodel_di, avg_2d
    from visuals import pannel_1image_3series
    # utility function
    def suffix(t):
        if t < 10:
            suff = '000' + str(t)
        elif t >= 10 and t < 100:
            suff = '00' + str(t)
        elif t >= 100 and t < 1000:
            suff = '0' + str(t)
        else:
            suff = str(t)
        return suff
    # import series
    series_df = pd.read_csv(fseries, sep=';', parse_dates=['Date'])
    #
    # verify size
    if len(series_df['Date'].values) < start + size:
        size = len(series_df['Date'].values) - start
    #
    # extract arrays
    date = series_df['Date'].values[start: start + size]
    prec = series_df[var1].values[start: start + size]
    prec_max = np.max(prec)
    base = series_df[var2].values[start: start + size]
    base_max = np.max(base)
    deficit = series_df[var3].values[start: start + size]
    defic_max = np.max(deficit)
    #
    #
    ttls = ('Local D (mm)', var1 + ' (mm)', var2 + ' (mm)', var3 + ' (mm)')
    # import twi
    meta, twi = input.asc_raster(ftwi)
    #
    # import aoi
    meta, aoi = input.asc_raster(faoi)
    #
    # extract lamb
    lamb = avg_2d(twi, aoi)
    #
    # import parameter m
    df_param = pd.read_csv(fparam, sep=';', index_col='Parameter')
    m = df_param.loc['m'].values[0]
    #
    di = topmodel_di(d=defic_max, twi=twi, m=m, lamb=lamb)
    di_max = 0.5 * np.max(di)
    frame = int(size * framef)
    lcl_t_index = int(frame * watchf)
    for t in range(size - frame):
        print(t)
        # set suffix
        suff = suffix(t)
        #
        # extract local arrays
        lcl_date = date[t: t + frame]
        lcl_p = prec[t: t + frame]
        lcl_qb = base[t: t + frame]
        lcl_dfc = deficit[t: t + frame]
        #
        # extract local Deficit
        lcl_d = lcl_dfc[lcl_t_index]
        #
        # compute local distributed Deficit
        lcl_di = topmodel_di(d=lcl_d, twi=twi, m=m, lamb=lamb)
        # mask down imagem
        lcl_di_mask = geo.mask(lcl_di, aoi) #[90:1940, 50:1010]
        # export image
        pannel_1image_3series(image=lcl_di_mask, imax=di_max,
                              t=lcl_date, x1=lcl_p, x2=lcl_qb, x3=lcl_dfc,
                              x1max=prec_max, x2max=base_max, x3max=defic_max, titles=ttls,
                              vline=lcl_t_index, folder=folder, suff=suff)


def frames_topmodel4(fseries, ftwi, fcn, faoi, fparam, fmaps, varseries, cmaps, imtitles, ylabels, ytitles,
                     size=200, start=0, framef=0.3, watchf=0.2, folder='C:/bin', show=False):
    """
    Create frames for video watch of topmodel simulation with TWI map + 3 other maps alongside 4 variables

    4 maps and 4 variables

    :param fseries: string file path to simulation series .txt file
    :param ftwi: string file path to TWI raster .asc file
    :param fcn: string file path to CN raster .asc file
    :param faoi: string file path to AOI raster .asc file
    :param fparam: string file path to simulation parameters .txt file
    :param fmaps: list or tuple storing 3 string file paths to directory of 3 maps
    :param varseries: list or tuple with 4 strings of variables field names (must exist in series DataFrame)
    :param cmaps: tuple of 4 strings kargs to color maps. Example: ('viridis_r', 'viridis_r', 'viridis_r', 'jet')
    :param imtitles: tuple of 4 strings with image titles.
    :param ylabels: tuple of 4 strings with y axis labels in series.
    :param ytitles: tuple of 4 strings of series titles
    :param size: int number of frames
    :param start: int index of first frame
    :param framef: float of frame ratio to full size (0 to 1)
    :param watchf: float of watch line ratio to full frame (0 to 1)
    :param folder: string path to destination folder
    :return: none
    """
    from hydrology import topmodel_di, avg_2d, map_back
    from visuals import pannel_4image_4series
    #
    # import series
    series_df = pd.read_csv(fseries, sep=';', parse_dates=['Date'])
    #
    # extract arrays
    date = series_df['Date'].values[start: start + size]
    prec = series_df['Prec'].values[start: start + size]
    imax = 0.1 * np.max(prec)
    #
    series1 = series_df[varseries[0]].values[start: start + size]
    series1_max = np.max(series1)
    #
    series2 = series_df[varseries[1]].values[start: start + size]
    series2_max = np.max(series2)
    #
    series3 = series_df[varseries[2]].values[start: start + size]
    series3_max = np.max(series3)
    #
    series4 = series_df[varseries[3]].values[start: start + size]
    series4_max = np.max(series4)
    #
    deficit = series_df['D'].values[start: start + size]
    defic_max = np.max(deficit)
    #
    # extract maplists
    map1_df = pd.read_csv(fmaps[0], sep=';')
    map2_df = pd.read_csv(fmaps[1], sep=';')
    map3_df = pd.read_csv(fmaps[2], sep=';')
    #
    # import twi
    meta, twi = input.asc_raster(ftwi)
    #
    # import cn
    meta, cn = input.asc_raster(fcn)
    #
    # import aoi
    meta, aoi = input.asc_raster(faoi)
    #
    # extract lamb
    lamb = avg_2d(twi, aoi)
    #
    # import parameter m
    df_param = pd.read_csv(fparam, sep=';', index_col='Parameter')
    m = df_param.loc['m'].values[0]
    #
    di = topmodel_di(d=defic_max, twi=twi, m=m, lamb=lamb)
    di_max = 0.5 * np.max(di)
    #
    # frame loops
    frame = int(size * framef)
    lcl_t_index = int(frame * watchf)
    for t in range(size - frame):
        print(t)
        # extract local arrays
        lcl_date = date[t: t + frame]
        lcl_s1 = series1[t: t + frame]
        lcl_s2 = series2[t: t + frame]
        lcl_s3 = series3[t: t + frame]
        lcl_s4 = series4[t: t + frame]
        #plt.plot(lcl_date, lcl_s1)
        #plt.show()
        lcl_dfc = deficit[t: t + frame]
        #
        # extract image 1
        lcl_im_files = map1_df['File'].values[t: t + frame]
        lcl_im_path = lcl_im_files[lcl_t_index]
        zmap, hist_twi, hist_cn = input.zmap(file=lcl_im_path)
        im1 = map_back(zmatrix=zmap, a1=twi, a2=cn, bins1=hist_twi, bins2=hist_cn)
        im1 = geo.mask(im1, aoi)  # [90:1940, 50:1010]
        #
        # extract image 2
        lcl_im_files = map2_df['File'].values[t: t + frame]
        lcl_im_path = lcl_im_files[lcl_t_index]
        zmap, hist_twi, hist_cn = input.zmap(file=lcl_im_path)
        im2 = map_back(zmatrix=zmap, a1=twi, a2=cn, bins1=hist_twi, bins2=hist_cn)
        im2 = geo.mask(im2, aoi)  # [90:1940, 50:1010]
        #
        # extract image 2
        lcl_im_files = map3_df['File'].values[t: t + frame]
        lcl_im_path = lcl_im_files[lcl_t_index]
        zmap, hist_twi, hist_cn = input.zmap(file=lcl_im_path)
        im3 = map_back(zmatrix=zmap, a1=twi, a2=cn, bins1=hist_twi, bins2=hist_cn)
        im3 = geo.mask(im3, aoi)  # [90:1940, 50:1010]
        #
        # extract local Deficit
        lcl_d = lcl_dfc[lcl_t_index]
        lcl_di = topmodel_di(d=lcl_d, twi=twi, m=m, lamb=lamb)
        im4 = geo.mask(lcl_di, aoi)  # [90:1940, 50:1010]
        #
        # load parameters
        im = (im1, im2, im3, im4)
        y4 = (lcl_s1, lcl_s2, lcl_s3, lcl_s4)
        y4max = (series1_max, series2_max, series3_max, series4_max)
        y4min = (0, 0, 0, 0)
        #
        stamp = pd.to_datetime(lcl_date, format='%y-%m-%d')
        suff = str(stamp[lcl_t_index]).split(' ')[0]
        #
        # export image
        pannel_4image_4series(im, imax, stamp, y4, y4max, y4min, cmaps, vline=lcl_t_index,
                              imtitles=imtitles, ytitles=ytitles, ylabels=ylabels, show=show, suff=suff)


def frames_topmodel_maps(fseries, ftwi, fcn, faoi, fparam, fs1maps, fs2maps, ftfmaps, finfmaps, frmaps, fqvmaps, fetmaps,
                         fevmaps, ftpmaps, ftpgwmaps, size=600, start=0, framef=0.25, watchf=0.2, folder='C:/bin',
                         show=False):
    from hydrology import topmodel_di, avg_2d, map_back
    from visuals import pannel_topmodel_maps
    #
    # import series
    series_df = pd.read_csv(fseries, sep=';', parse_dates=['Date'])
    #
    # extract arrays
    date = series_df['Date'].values[start: start + size]
    prec = series_df['Prec'].values[start: start + size]
    pet = series_df['PET'].values[start: start + size]
    et = series_df['ET'].values[start: start + size]
    qb = series_df['Qb'].values[start: start + size]
    deficit = series_df['D'].values[start: start + size]
    #
    # extract maps dataframes
    maps_dfs_files = (fs1maps, fs2maps, ftfmaps, finfmaps, frmaps, fqvmaps, fetmaps, fevmaps, ftpmaps, ftpgwmaps)
    maps_dfs = list()  # list of dataframes
    for i in range(len(maps_dfs_files)):
        lcl_df = pd.read_csv(maps_dfs_files[i], sep=';')
        maps_dfs.append(lcl_df.copy())

    # import twi
    meta, twi = input.asc_raster(ftwi)
    #
    # import cn
    meta, cn = input.asc_raster(fcn)
    #
    # import aoi
    meta, aoi = input.asc_raster(faoi)
    #
    # extract lamb
    lamb = avg_2d(twi, aoi)
    #
    # import parameter m
    df_param = pd.read_csv(fparam, sep=';', index_col='Parameter')
    m = df_param.loc['m'].values[0]
    #
    # extract parameters
    imax = 0.25 * np.max(prec)
    precmax = np.max(prec)
    qbmax = np.max(qb)
    etmax = np.max(pet)
    #
    # frame loops
    frame = int(size * framef)
    lcl_t_index = int(frame * watchf)
    for t in range(size - frame):
        print(t)
        # extract local arrays
        lcl_date = date[t: t + frame]
        lcl_prec = prec[t: t + frame]
        lcl_qb = qb[t: t + frame]
        lcl_pet = pet[t: t + frame]
        lcl_et = et[t: t + frame]
        lcl_dfc = deficit[t: t + frame]
        #
        # extract maps
        lcl_d = lcl_dfc[lcl_t_index]
        lcl_di = topmodel_di(d=lcl_d, twi=twi, m=m, lamb=lamb)
        di_map = geo.mask(lcl_di, aoi)
        vsai_map = geo.mask((di_map <= 0) * 1.0, aoi)
        lcl_p = lcl_prec[lcl_t_index]
        pi_map = geo.mask(lcl_p + (lcl_di * 0.0), aoi)
        maps_lst = [pi_map, vsai_map, di_map]
        #
        for i in range(0, len(maps_dfs)):
            lcl_im_files = maps_dfs[i]['File'].values[t: t + frame]
            lcl_im_path = lcl_im_files[lcl_t_index]
            zmap, hist_twi, hist_cn = input.zmap(file=lcl_im_path)
            im = map_back(zmatrix=zmap, a1=twi, a2=cn, bins1=hist_twi, bins2=hist_cn)
            im = geo.mask(im, aoi)
            maps_lst.append(im)
        #
        #
        # load params
        stamp = pd.to_datetime(lcl_date, format='%y-%m-%d')
        suff = str(stamp[lcl_t_index]).split(' ')[0]
        pannel_topmodel_maps(t=lcl_date, prec=lcl_prec, precmax=precmax, qb=lcl_qb, qbmax=qbmax, pet=lcl_pet, et=lcl_et,
                             etmax=etmax, maps=maps_lst, mapsmax=imax, vline=lcl_t_index, suff=suff, show=show)

