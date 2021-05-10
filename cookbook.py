
#
#
#
#
#
# verify if deprecated:
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
    maps_df = pd.read_csv(fmaps, sep=';', engine='python')
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

# verify if deprecated:
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
    series_df = pd.read_csv(fseries, sep=';\s+', engine='python')
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

# verify if deprecated:
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
    series_df = pd.read_csv(fseries, sep=';', engine='python', parse_dates=['Date'])
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
    df_param = pd.read_csv(fparam, sep=';\s+', engine='python', index_col='Parameter')
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

# verify if deprecated:
def frames_topmodel4(fseries, ftwi, fcn, faoi, fparam, fmaps, varseries, cmaps, imtitles, ylabels, ytitles,
                     size=200, start=0, framef=0.3, watchf=0.2, folder='C:/bin', show=False):
    """
    Create frames of topmodel simulation with TWI map + 3 other maps alongside 4 variables

    4 maps and 4 variables

    :param fseries: string file path to simulation series .txt file
    :param ftwi: string file path to TWI raster .asc file
    :param fcn: string file path to CN raster .asc file
    :param faoi: string file path to AOI raster .asc file
    :param fparam: string file path to simulation parameters .txt file
    :param fmaps: list or tuple storing 3 string file paths of the mapfiles
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
    series_df = pd.read_csv(fseries, sep=';', engine='python', parse_dates=['Date'])
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
    map1_df = pd.read_csv(fmaps[0], sep=';\s+', engine='python')
    map2_df = pd.read_csv(fmaps[1], sep=';\s+', engine='python')
    map3_df = pd.read_csv(fmaps[2], sep=';\s+', engine='python')
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
    df_param = pd.read_csv(fparam, sep=';\s+', engine='python', index_col='Parameter')
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

# verify if deprecated:
def frames_topmodel_maps(fseries, ftwi, fcn, faoi, fparam, fs1maps, fs2maps, ftfmaps, finfmaps, frmaps, fqvmaps,
                         fetmaps, fevmaps, ftpmaps, ftpgwmaps, size=600, start=0, framef=0.25, watchf=0.2,
                         folder='C:/bin', show=False):
    """
    Create frames of topmodel simulation with all 13 processes maps alongside with 3 variables series: Precipitation,
    base flow and ET (with PET)
    :param fseries: string file path to simulation series .txt file
    :param ftwi: string file path to TWI raster .asc file
    :param fcn: string file path to CN raster .asc file
    :param faoi: string file path to AOI raster .asc file
    :param fparam: string file path to simulation parameters .txt file
    :param fs1maps: string file path to S1 .txt mapfile
    :param fs2maps: string file path to S1 .txt mapfile
    :param ftfmaps: string file path to S1 .txt mapfile
    :param finfmaps: string file path to S1 .txt mapfile
    :param frmaps: string file path to S1 .txt mapfile
    :param fqvmaps: string file path to S1 .txt mapfile
    :param fetmaps: string file path to S1 .txt mapfile
    :param fevmaps: string file path to S1 .txt mapfile
    :param ftpmaps: string file path to S1 .txt mapfile
    :param ftpgwmaps: string file path to S1 .txt mapfile
    :param size: int number of frames
    :param start: int index of first frame
    :param framef: float of frame ratio to full size (0 to 1)
    :param watchf: float of watch line ratio to full frame (0 to 1)
    :param folder: string path to destination folder
    :param show: boolean control to show or save frame
    :return: none
    """
    from hydrology import topmodel_di, avg_2d, map_back
    from visuals import pannel_topmodel_maps
    #
    # import series
    series_df = pd.read_csv(fseries, sep=';', engine='python', parse_dates=['Date'])
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
        lcl_df = pd.read_csv(maps_dfs_files[i], sep=';\s+', engine='python')
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
    df_param = pd.read_csv(fparam, sep=';\s+', engine='python', index_col='Parameter')
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
        # load params
        stamp = pd.to_datetime(lcl_date, format='%y-%m-%d')
        suff = str(stamp[lcl_t_index]).split(' ')[0]
        pannel_topmodel_maps(t=lcl_date, prec=lcl_prec, precmax=precmax, qb=lcl_qb, qbmax=qbmax, pet=lcl_pet, et=lcl_et,
                             etmax=etmax, maps=maps_lst, mapsmax=imax, vline=lcl_t_index, suff=suff, show=show)

# verify if deprecated:
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
    lulc_series_df = pd.read_csv(flulcseries, sep=';\s+', engine='python')
    dates = lulc_series_df['Date'].values
    files = lulc_series_df['File'].values
    lulc_param_df = pd.read_csv(flulcparam, sep=';\s+', engine='python')
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


def watch_maps_byzmaps():
    """
    Cookbook to watch maps from a simulation run
    :return:
    """
    import tools, input, geo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from hydrology import map_back
    from visuals import plot_map_view
    #
    # define output directory
    folder_output = 'C:/bin'
    #
    # directory of input data
    folder_input = 'C:/Plans3/demo/datasets/observed'
    # files paths to raster maps
    ftwi = folder_input + '/' + 'calib_twi.asc'
    fshru = folder_input + '/' + 'calib_shru.asc'
    fbasin = folder_input + '/' + 'calib_basin.asc'
    #
    # import raster maps
    meta, twi = input.asc_raster(ftwi)
    meta, shru = input.asc_raster(fshru)
    meta, basin = input.asc_raster(fbasin)
    #
    # directory of simulated data r'C:\Plans3\demo\runbin\optimization\calib_hydro_KGElog_2021-04-19-18-04-14\bestset\calibration_period' #
    folder_sim = 'C:/Plans3/demo/runbin/simulation/calib_SLH_2021-04-21-07-25-42'
    # simulated time series
    file = folder_sim + '/' + 'sim_series.txt'
    # import time series
    df_series = pd.read_csv(file, sep=';')
    #
    # define here list of variable maps
    varmaps = ['ET']
    size = 300  # define how many frames to watch
    #
    # loop in variables
    for v in varmaps:
        # assumed file in simulation
        file = folder_sim + '/' + 'sim_zmaps_{}.txt'.format(v)  # open map series
        # import map series
        df = pd.read_csv(file, sep=';')
        # extract min and max values from series (global average values)
        v_min = np.min(df_series[v].values[:size])
        v_max = np.max(df_series[v].values[:size])
        #
        # loop in each frame
        for i in range(size):
            print('frame {}'.format(i))
            # extract local Z-Map file path and date
            lcl_file = df['File'].values[i]
            lcl_date = df['Date'].values[i]
            # import Z-Map
            zmap, hist_twi, hist_shru = input.zmap(file=lcl_file)
            # Map back to raster
            mp = map_back(zmatrix=zmap, a1=twi, a2=shru, bins1=hist_twi, bins2=hist_shru)
            # mask it by basin
            #mp = geo.mask(mp, basin)
            #
            #
            # smart mapid selector
            if v == 'D':
                mapid = 'deficit'
            elif v in set(['Cpy', 'Sfs', 'Unz']):
                mapid = 'stock'
            elif v in set(['R', 'RSE', 'RIE', 'Inf', 'TF', 'IRA', 'IRI', 'Qv', 'P']):
                mapid = 'flow'
            elif v in set(['ET', 'Evc', 'Evs', 'Tpun', 'Tpgw']):
                mapid = 'flow_v'
            elif v == 'VSA':
                mapid = 'VSA'
            elif v == 'RC':
                mapid = 'VSA'
            else:
                mapid = 'flow'
            #
            # PLOT
            plot_map_view(mp, meta, ranges=(v_min, v_max), mapid=mapid, mapttl='{} | {}'.format(v, lcl_date),
                          show=False, metadata=False, folder=folder_output, filename='{}_{}'.format(v, lcl_date))


def create_etpat_input():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    import input, output

    # directory of input data
    folder_input = 'C:/Plans3/demo/datasets/observed'
    file = folder_input + '/calib_etpat_series_input_old.txt'
    output_folder = 'C:/Plans3/demo/datasets/observed/bin'
    df = pd.read_csv(file, sep=';')
    df = input.dataframe_prepro(df, strfields='Date,File')
    print(df)
    # files paths to raster maps
    ftwi = folder_input + '/' + 'calib_twi.asc'
    #
    # import raster maps
    meta, twi = input.asc_raster(ftwi)
    new_files = list()
    for i in range(len(df)):
        randim = np.random.random(size=np.shape(twi))
        filter = gaussian_filter(randim, sigma=8)
        raster = twi + (500 * filter)
        filename = 'input_etpat_' + df['Date'].values[i]
        filer = output.asc_raster(raster, meta, folder=output_folder, filename=filename)
        new_files.append(filer)
        #plt.imshow(twi + (500 * filter))
        #plt.show()
    df['File'] = new_files
    out = folder_input + '/calib_etpat_series_input_2.txt'
    df.to_csv(out, sep=';', index=False)


def visual_map_analyst():
    import input, analyst
    from visuals import  plot_map_analyst
    import numpy as np
    from scipy.ndimage import gaussian_filter
    import matplotlib.pyplot as plt
    # define output directory
    folder_output = 'C:/bin'
    #
    # directory of input data
    folder_input = 'C:/Plans3/demo/datasets/observed'
    # files paths to raster maps
    ftwi = folder_input + '/' + 'calib_twi.asc'
    fobs = r"C:\Plans3\demo\datasets\observed\etpat\calib_etpat_2012-10-01.asc"
    fsim = r"C:\Plans3\demo\runbin\optimization\calib_hydro_KGElog_2021-04-21-13-04-49\bestset\calibration_period\sim_ETPat\raster_ETPat_2012-10-01.asc"
    #
    # import raster maps
    meta, obs = input.asc_raster(fobs)
    meta, sim = input.asc_raster(fsim)
    metric = analyst.error(obs, sim)

    obs_signal = obs.flatten()
    sim_signal = sim.flatten()
    metric_signal = analyst.error(obs_signal, sim_signal)
    #plt.imshow(metric, cmap='seismic')
    #plt.show()
    metrics_dct = {'Error': 666.666, 'SqErr': 666.666, 'RMSE': 666.666, 'NSE':666.3, 'KGE':666.666, 'R':666.666}
    ranges = (0, 1)
    metricranges = (-1, 1)
    plot_map_analyst(obs, sim, metric, obs_signal, sim_signal, metric_signal, ranges=ranges, metricranges=metricranges,
                     metrics_dct=metrics_dct)


def demo_obs_sim_map_analyst(fseries, type, var='ETPat', filename='obssim_maps_analyst', folder='C:/bin', tui=True):
    from input import dataframe_prepro
    import input
    import pandas as pd
    import numpy as np
    #
    import time, datetime
    import analyst
    from visuals import plot_map_analyst

    def extract_series_data(dataframe, fld, type='raster'):
        maps_lst = list()
        signal_lst = list()
        for i in range(len(dataframe)):
            map_file = dataframe[fld].values[i]
            if type == 'zmap':
                map, ybins, xbins = input.zmap(map_file)
            elif type == 'raster':
                meta, map = input.asc_raster(map_file)
            signal = map.flatten()
            maps_lst.append(map)
            signal_lst.append(signal)
        full_signal = np.array(maps_lst).flatten()
        return maps_lst, signal_lst, full_signal

    # report setup
    t0 = time.time()
    report_lst = list()
    report_lst.append('Execution timestamp: {}\n'.format(datetime.datetime.now()))
    report_lst.append('Process: OBS-SIM DATA ANALYST\n')
    input_files_df = pd.DataFrame({'Input files': (fseries,)})
    report_lst.append(input_files_df.to_string(index=False))
    #
    if tui:
        from tui import status
        status('performing obs vs. sim map analysis')
    #
    # extract Dataframe
    def_df = pd.read_csv(fseries, sep=';', engine='python')
    def_df = dataframe_prepro(def_df, strfields='File_obs,File_sim,Date')
    #
    maps_obs_lst, signal_obs_lst, full_signal_obs = extract_series_data(def_df, 'File_obs', type=type)
    maps_sim_lst, signal_sim_lst, full_signal_sim = extract_series_data(def_df, 'File_sim', type=type)
    #
    # 3) compute local map metrics and append to a new dataframe
    map_errors = list()
    map_sqerr = list()
    map_rmse = list()
    map_nse = list()
    map_kge = list()
    map_r = list()
    metric_maps = list()
    metric_signal = list()
    for i in range(len(def_df)):
        #
        lcl_metric = analyst.error(maps_obs_lst[i], maps_sim_lst[i])
        metric_maps.append(lcl_metric)
        #
        lcl_error = analyst.error(signal_obs_lst[i], signal_sim_lst[i])
        metric_signal.append(lcl_error)
        map_errors.append(np.mean(lcl_error))
        #
        lcl_sqerr = analyst.sq_error(signal_obs_lst[i], signal_sim_lst[i])
        map_sqerr.append(np.mean(lcl_sqerr))
        #
        lcl_rmse = analyst.rmse(signal_obs_lst[i], signal_sim_lst[i])
        map_rmse.append(lcl_rmse)
        #
        lcl_nse = analyst.nse(signal_obs_lst[i], signal_sim_lst[i])
        map_nse.append(lcl_nse)
        #
        lcl_kge = analyst.kge(np.append([1], signal_obs_lst[i]), np.append([1], signal_sim_lst[i]) )
        map_kge.append(lcl_kge)
        #
        lcl_r = analyst.linreg(np.append([1], signal_obs_lst[i]), np.append([1], signal_sim_lst[i]) )['R']
        map_r.append(lcl_r)
    #
    # built data frame
    def_df['Error'] = map_errors
    def_df['SqErr'] = map_sqerr
    def_df['RMSE'] = map_rmse
    def_df['NSE'] = map_nse
    def_df['KGE'] = map_kge
    def_df['R'] = map_r
    out_df = def_df[['Date', 'Error', 'SqErr', 'RMSE', 'NSE', 'KGE', 'R', 'File_obs', 'File_sim']]
    #
    #
    # Export dataframe
    out_file = folder + '/' + var + '_' + filename + '.txt'
    out_df.to_csv(out_file, sep=';', index=False)
    #
    # Export visuals
    map_vmin = 0.0 #np.min((maps_obs_lst, maps_obs_lst))
    map_vmax = np.max((maps_obs_lst, maps_obs_lst))
    ranges = (map_vmin, map_vmax)
    map_metric_vmin = np.min(metric_maps)
    map_metric_vmax = np.max(metric_maps)
    mapmax = np.max((np.abs(map_metric_vmin), np.abs(map_metric_vmax)))
    metricranges = (-mapmax, mapmax)
    for i in range(len(out_df)):
        print(i)
        lcl_date = out_df['Date'].values[i]
        lcl_filename = var + '_map_analyst_' + lcl_date
        lcl_ttl = '{} | {}'.format(var, lcl_date)
        metrics_dct = {'Error': out_df['Error'].values[i],
                       'SqErr': out_df['SqErr'].values[i],
                       'RMSE': out_df['RMSE'].values[i],
                       'NSE':out_df['NSE'].values[i],
                       'KGE':out_df['KGE'].values[i],
                       'R':out_df['R'].values[i]}
        vis_file = plot_map_analyst(obs=maps_obs_lst[i], sim=maps_sim_lst[i], metric=metric_maps[i],
                                    obs_sig=signal_obs_lst[i], sim_sig=signal_sim_lst[i], ranges=ranges,
                                    metricranges=metricranges, metric_sig=metric_signal[i], metrics_dct=metrics_dct,
                                    filename=lcl_filename, folder=folder, ttl=lcl_ttl)


def demo_watch():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from visuals import pannel_local
    import input
    from hydrology import map_back
    folder ='C:/Plans3/demo/datasets/observed'
    ftwi = '{}/calib_twi.asc'.format(folder)
    fshru = '{}/calib_shru.asc'.format(folder)

    meta, twi = input.asc_raster(ftwi)
    meta, shru = input.asc_raster(fshru)

    folder = r"C:\Plans3\demo\runbin\simulation\calib_SLH_2021-04-25-14-32-52\calibration_period"
    fseries = folder  + r'\sim_series.txt'
    series = pd.read_csv(fseries, sep=';', parse_dates=['Date'])

    date_init = '2011-01-01'
    date_end = '2011-05-01'
    query_str = 'Date > "{}" and Date < "{}"'.format(date_init, date_end)
    series = series.query(query_str)
    print(len(series))
    #vars = 'Cpy-Sfs-Unz-D-TF-Tpun-Tpgw-Evc-Evs-ET-R-Inf-RIE-RSE-IRA-IRI-Qv-VSA'.split('-')
    # for QV vars = 'D-Qv-Inf-Cpy-IRA-IRI-Unz-Sfs'.split('-')
    # for ET
    vars = 'D-ET-IRA-IRI-Tpun-Tpgw-Evc-Evs-Qv-Inf-Cpy-Unz-Sfs-R-RIE-RSE-Qv-VSA'.split('-')
    #for R vars = 'D-IRA-IRI-R-RIE-RSE-VSA-TF'.split('-')
    #
    #
    # 1) load zmap series dataframes in a dict for each
    zmaps_series = dict()
    for var in vars:
        lcl_file = '{}/sim_zmaps_series_{}.txt'.format(folder, var)  # next will be sim_zmaps_series_{}.txt
        lcl_df = pd.read_csv(lcl_file, sep=';', parse_dates=['Date'])
        lcl_df = lcl_df.query(query_str)
        #print(lcl_df.tail().to_string())
        zmaps_series[var] = lcl_df.copy()
    #
    #
    # load rasters from zmap files
    raster_series = dict()
    rasters_maxval = dict()
    rasters_minval = dict()
    for var in vars:
        lcl_df = zmaps_series[var]
        raster_list = list()
        print('computing raster maps of {} ... '.format(var))
        for i in range(len(series)):
            lcl_file = lcl_df['File'].values[i]
            lcl_zmap, ybins, xbins = input.zmap(lcl_file)
            lcl_raster = map_back(lcl_zmap, a1=twi, a2=shru, bins1=ybins, bins2=xbins)
            raster_list.append(lcl_raster)
        raster_nd = np.array(raster_list)
        raster_series[var] = raster_nd
        rasters_minval[var] = np.min(raster_nd)
        rasters_maxval[var] = np.max(raster_nd)
        print('{} maps loaded'.format(len(raster_nd)))
    #print(zmaps_series['R'].head().to_string())
    #print(rasters_minval['R'])
    #print(rasters_maxval['R'])
    offsetback = 10
    offsetfront = 40
    prec_rng = (0, np.max(series['Prec'].values))
    irri_rng = (0, np.max((series['IRA'].values, series['IRA'].values)))
    for t in range(offsetback, len(series) - offsetfront -1):
        print('t = {} | plotting date {}'.format(t, series['Date'].values[t]))
        pannel_local(series, star=raster_series['ET'][t],
                     deficit=raster_series['D'][t],
                     sups=[raster_series['IRA'][t], raster_series['IRI'][t]],
                     mids=[raster_series['Evc'][t], raster_series['Evs'][t],
                           raster_series['Tpun'][t], raster_series['Tpgw'][t]],
                     star_rng=(rasters_minval['ET'], rasters_maxval['ET']),
                     deficit_rng=(rasters_minval['D'], rasters_maxval['D']),
                     sup1_rng=prec_rng, sup2_rng=irri_rng, sup3_rng=irri_rng, sup4_rng=irri_rng,
                     mid1_rng=(rasters_minval['ET'], rasters_maxval['ET']),
                     mid2_rng=(rasters_minval['ET'], rasters_maxval['ET']),
                     mid3_rng=(rasters_minval['ET'], rasters_maxval['ET']),
                     mid4_rng=(rasters_minval['ET'], rasters_maxval['ET']),
                     t=t, type='ET', show=True, offset_back=offsetback, offset_front=offsetfront)


def demo_simulation():
    from tools import slh
    import backend
    import pandas as pd

    files_input = backend.get_input2simbhydro(aoi=False)
    folder = 'C:/Plans3/demo/datasets/observed'
    fseries ='{}/{}'.format(folder, files_input[0])
    fhydroparam = '{}/{}'.format(folder, files_input[1])
    fshruparam = '{}/{}'.format(folder, files_input[2])
    fhistograms = '{}/{}'.format(folder, files_input[3])
    fbasinhists = '{}/{}'.format(folder, files_input[4])
    fbasin = '{}/{}'.format(folder, files_input[5])
    ftwi = '{}/{}'.format(folder, files_input[6])
    fshru = '{}/{}'.format(folder, files_input[7])

    series = pd.read_csv(fseries, sep=';', parse_dates=['Date'])
    date_init = '2011-01-01'
    date_end = '2012-01-01'
    query_str = 'Date >= "{}" and Date < "{}"'.format(date_init, date_end)
    series = series.query(query_str)

    vars = 'IRI-IRA'
    #vars = 'all'

    mapdates = ' & '.join(series['Date'].astype('str'))
    #mapdates = 'all'
    #print(mapdates)
    outfolder = 'C:/bin'
    out_dct = slh(fseries=fseries, fhydroparam=fhydroparam, fshruparam=fshruparam,
                  fhistograms=fhistograms, fbasinhists=fbasinhists, fbasin=fbasin,
                  ftwi=ftwi, fshru=fshru, folder=outfolder,
                  wkpl=True,
                  tui=True,
                  mapback=True,
                  mapraster=True,
                  mapvar=vars, integrate=True,
                  mapdates=mapdates, qobs=True)


def diags():
    import pandas as pd
    from tools import sdiag
    fseries = r"C:\Plans3\pardo\runbin\simulation\calib_SLH_2021-04-30-09-27-27\calibration_period\sim_series.txt"
    series = pd.read_csv(fseries, sep=';', parse_dates=['Date'])
    sdiag(fseries, tui=True)


def plot_sal_frames():
    import numpy as np
    from visuals import sal_deficit_frame
    from input import asc_raster
    from hydrology import topmodel_di, topmodel_vsai, avg_2d

    def stamped(g):
        if g < 10:
            stamp = '0000' + str(g)
        elif g >= 10 and g < 100:
            stamp = '000' + str(g)
        elif g >= 100 and g < 1000:
            stamp = '00' + str(g)
        elif g >= 1000 and g < 10000:
            stamp = '0' + str(g)
        else:
            stamp = str(g)
        return stamp


    meta, twi = asc_raster('./samples/calib_twi.asc')

    lamb = avg_2d(twi, weight=(1 + (twi * 0)))
    print(lamb)
    m1 = 10
    m2 = 30
    d_nd = np.arange(0, 151)
    print(d_nd)
    d1_lst = list()
    d2_lst = list()
    for d in d_nd:
        di1 = topmodel_di(d, twi, m=m1, lamb=lamb)
        d1_lst.append(di1)
        di2 = topmodel_di(d, twi, m=m2, lamb=lamb)
        d2_lst.append(di2)
    vmax = np.max((d1_lst, d2_lst))
    print(vmax)
    count = 0
    for i in range(len(d1_lst)):
        vsai1 = topmodel_vsai(d1_lst[i])
        vsai2 = topmodel_vsai(d2_lst[i])
        filename = 'SAL_d_frame_{}'.format(stamped(count))
        print(filename)
        sal_deficit_frame(d_nd[i], d1=d1_lst[i], vsa1=vsai1, d2=d2_lst[i], vsa2=vsai2, m1=m1, m2=m2, vmax=vmax, vmin=0,
                          dgbl_max=np.max(d_nd), filename=filename)
        count = count + 1
    for i in range(len(d1_lst) - 1, -1, -1):
        vsai1 = topmodel_vsai(d1_lst[i])
        vsai2 = topmodel_vsai(d2_lst[i])
        filename = 'SAL_d_frame_{}'.format(stamped(count))
        print(filename)
        sal_deficit_frame(d_nd[i], d1=d1_lst[i], vsa1=vsai1, d2=d2_lst[i], vsa2=vsai2, m1=m1, m2=m2, vmax=vmax, vmin=0,
                          dgbl_max=np.max(d_nd), filename=filename)
        count = count + 1


def plot_gens_evolution(folder='C:/bin'):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    def stamped(g):
        if g < 10:
            stamp = '0000' + str(g)
        elif g >= 10 and g < 100:
            stamp = '000' + str(g)
        elif g >= 100 and g < 1000:
            stamp = '00' + str(g)
        elif g >= 1000 and g < 10000:
            stamp = '0' + str(g)
        else:
            stamp = str(g)
        return stamp

    f = r"C:\Plans3\pardo\runbin\optimization\calib_Hydrology_KGElog_2021-05-05-13-20-14\generations\population\generations_population.txt"
    df = pd.read_csv(f, sep=';')
    param = 'sfmax'
    likelihood = 'Qb_C'
    for i in range(len(df)):
        lcl_f = df['File'].values[i]
        lcl_df = pd.read_csv(lcl_f, sep=';')
        if i == 0:
            x = lcl_df[param]
            y = lcl_df[likelihood].values / 100
        else:
            x = np.append(x, lcl_df[param])
            y = np.append(y, lcl_df[likelihood].values / 100)
        plt.scatter(x, y, cmap='Spectral', c=y)
        plt.ylim((-1, 1))
        plt.ylabel(likelihood)
        plt.xlabel('{} parameter'.format(param))
        plt.title('Generation {}'.format(i + 1))
        filename = 'generation_{}_{}_{}.jpg'.format(param, likelihood, stamped(i))
        print(filename)
        exp = '{}/{}'.format(folder, filename)
        plt.savefig(exp)

