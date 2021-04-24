import numpy as np
import pandas as pd
import input, output, geo
from input import dataframe_prepro
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter


def export_report(report_lst, filename='report', folder='C:/bin', tui=False):
    # todo docstring
    from backend import header_plans, header
    filepath = folder + '/' + filename + '.txt'
    fle = open(filepath, 'w+')
    header = header('output report')
    report_lst.insert(0, header)
    header = header_plans()
    report_lst.insert(0, header)
    fle.writelines(report_lst[:])
    fle.close()
    if tui:
        for e in report_lst[1:]:
            print(e)


def view_imported(filename, folder):
    # todo docstring
    from visuals import plot_map_view, plot_calib_series
    file = folder + '/' + filename
    quantmaps = ('aoi_dem.asc', 'aoi_catcha.asc', 'aoi_basin.asc',
                 'calib_dem.asc', 'calib_catcha.asc', 'calib_basin.asc')
    if filename == 'calib_lulc.asc':
        print()
    elif filename in set(quantmaps):
        mapid = filename.split('.')[0].split('_')[1]
        meta, rmap = input.asc_raster(file)
        ranges = (np.min(rmap), np.max(rmap))
        plot_map_view(rmap, meta, ranges, mapid=mapid, filename=filename.split('.')[0], folder=folder, metadata=True)
    elif filename == 'calib_series.txt':
        series_df = pd.read_csv(file, sep=';')
        series_df = dataframe_prepro(series_df, strf=False, date=True, datefield='Date')
        plot_calib_series(series_df, filename=filename, folder=folder, show=False)


def map_shru(flulc, flulcparam, fsoils, fsoilsparam, fshruparam, folder='C:/bin', filename='shru'):
    """

    :param flulc: string file path to lulc .asc raster file
    :param flulcparam: string file path to lulc parameters .txt file. Separator = ;

    :param fsoils: string path to soils.asc raster file
    :param fsoilsparam: string path to soils parameters .txt file. Separator = ;

    :param folder: string path to destination folder
    :param filename: string name of file
    :return: string file path
    """
    from visuals import plot_shrumap_view
    #
    # import data
    metalulc, lulc = input.asc_raster(flulc)
    metasoils, soils = input.asc_raster(fsoils)
    lulc_param_df = pd.read_csv(flulcparam, sep=';', engine='python')
    lulc_param_df = dataframe_prepro(lulc_param_df, 'LULCName,ConvertTo,ColorLULC')
    soils_param_df = pd.read_csv(fsoilsparam, sep=';', engine='python')
    soils_param_df = dataframe_prepro(soils_param_df, 'SoilName,ColorSoil')
    lulc_ids = lulc_param_df['IdLULC'].values
    soils_ids = soils_param_df['IdSoil'].values
    shru_df = pd.read_csv(fshruparam, sep=';')
    shru_df = dataframe_prepro(shru_df, 'SHRUName,LULCName,SoilName,ColorLULC,ColorSoil')
    #
    # process data
    shru_map = geo.xmap(map1=lulc, map2=soils, map1ids=lulc_ids, map2ids=soils_ids, map1f=100, map2f=1)
    #plt.imshow(shru_map)
    #plt.show()
    plot_shrumap_view(lulc, soils, metalulc, shru_df, filename=filename, folder=folder, metadata=True)
    # export data
    export_file = output.asc_raster(shru_map, metalulc, folder, filename)
    return export_file


def map_fto(fsoils, fsoilsparam, folder='C:/bin', filename='fto'):
    """
    Derive the f_To map from soils map and soils parameters
    :param fsoils: string path to soils raster .asc file
    :param fsoilsparam: string path to soils parameters .txt file. Separator = ;
    :param folder: string path to destination folder
    :param filename: string name of file
    :return: string file path
    """
    from visuals import plot_map_view
    # import data
    meta, soils = input.asc_raster(fsoils)
    soils_df = pd.read_csv(fsoilsparam, sep=';', engine='python')
    soils_df = dataframe_prepro(soils_df, strfields='SoilName,ColorSoil')
    # process data
    fto = geo.reclassify(soils, upvalues=soils_df['IdSoil'].values, classes=soils_df['f_To'].values)
    #plt.imshow(fto)
    #plt.show()
    # export data
    export_file = output.asc_raster(fto, meta, folder, filename)
    ranges = (np.min(fto), np.max(fto))
    plot_map_view(fto, meta, ranges, mapid='fto', filename=filename, folder=folder, metadata=True)
    return export_file


def map_slope(fdem, folder='C:/bin', filename='slope'):
    """
    Derive slope of terrain in degrees
    :param fdem: string path to DEM (digital elevation model) raster .asc file
    :param folder: string path to destination folder
    :param filename: string of file name
    :return: string path to file
    """
    from visuals import plot_map_view
    # import data
    meta, dem = input.asc_raster(fdem)
    #plt.imshow(dem)
    #plt.show()
    # process data
    slp = geo.slope(dem, meta['cellsize'], degree=True)
    #plt.imshow(slp)
    #plt.show()
    #
    # export
    export_file = output.asc_raster(slp, meta, folder, filename)
    ranges = (np.min(slp), np.max(slp))
    plot_map_view(slp, meta, ranges, mapid='slope', filename=filename, folder=folder, metadata=True)
    return export_file


def map_twi(fslope, fcatcha, ffto, folder='C:/bin', filename='twi'):
    """
    Derive the Topographical Wetness Index of TOPMODEL (Beven & Kirkby, 1979)
    :param fslope: string path to slope in degrees raster .asc file
    :param fcatcha: string path to catchment area in squared meters raster .asc file
    :param ffto: string path to transmissivity factor (f_To) raster .asc file
    :param folder: string path to destination folder
    :param filename: string of file name
    :return: string path to file
    """
    from visuals import plot_map_view
    # import data
    meta, slope = input.asc_raster(fslope)
    meta, catcha = input.asc_raster(fcatcha)
    meta, fto = input.asc_raster(ffto)
    # process data
    grad = geo.grad(slope)
    twi = geo.twi(catcha, grad, fto, cellsize=meta['cellsize'])
    #plt.imshow(twi)
    #plt.show()
    # export data
    export_file = output.asc_raster(twi, meta, folder, filename)
    ranges = (np.min(twi), np.max(twi))
    plot_map_view(twi, meta, ranges, mapid='twi', filename=filename, folder=folder, metadata=True)
    return export_file


def compute_histograms(fshruparam, fshru, ftwi, faoi='none', ntwibins=15, folder='C:/bin', filename='histograms',
                       tui=False):
    # todo doctring
    import time
    from hydrology import count_matrix, flatten_clear
    from visuals import plot_histograms
    if tui:
        init = time.time()
        from tui import status
        status('loading SHRU parameters')
    shru_df = pd.read_csv(fshruparam, sep=';')
    shru_df = dataframe_prepro(shru_df, 'SHRUName,LULCName,SoilName')
    shrubins = shru_df['IdSHRU'].values
    #
    # import shru raster
    if tui:
        status('loading shru raster')
    meta, shru = input.asc_raster(fshru)
    #
    # import twi raster
    if tui:
        status('loading twi raster')
    meta, twi = input.asc_raster(ftwi)
    #
    if faoi == 'none':
        aoi = aoi = 1.0 + (twi * 0.0)
    else:
        # import twi raster
        if tui:
            status('loading aoi raster')
        meta, aoi = input.asc_raster(faoi)
    #
    if tui:
        end = time.time()
        print('\nloading enlapsed time: {:.3f} seconds\n'.format(end - init))
    #
    #
    #
    # compute count matrix
    if tui:
        init = time.time()
        status('computing histograms')
    #
    # flat and clear:
    twi_flat = twi.flatten()
    #
    # extract histogram of TWI
    twi_hist, twi_bins = np.histogram(twi_flat, bins=ntwibins)
    twibins = twi_bins[1:]
    count, twibins, shrubins = count_matrix(twi, shru, aoi, shrubins, twibins)
    print('SUM = {}'.format(np.sum(count)))
    if tui:
        end = time.time()
        print('\nProcessing enlapsed time: {:.3f} seconds\n'.format(end - init))
    #
    #
    # export histograms
    if tui:
        status('exporting histograms')
    exp_df = pd.DataFrame(count, index=twibins, columns=shrubins)
    if tui:
        print(exp_df.to_string())
    exp_file = folder + '/' + filename + '.txt'
    exp_df.to_csv(exp_file, sep=';', index_label='TWI\SHRU')
    #
    #
    #
    # plot histogram view
    matrix = count
    matrix_t = np.transpose(count)
    x_twi = twibins
    y_twi = np.zeros(len(matrix))
    for i in range(0, len(matrix)):
        y_twi[i] = np.sum(matrix[i])
    y_twi = 100 * y_twi / np.sum(matrix)
    y_shru = np.zeros(shape=len(matrix_t))
    for i in range(0, len(matrix_t)):
        y_shru[i] = np.sum(matrix_t[i])
    y_shru = 100 * y_shru / np.sum(matrix_t)
    x_shru = exp_df.columns
    plot_histograms(matrix, x_shru, y_shru, x_twi, y_twi, filename=filename, folder=folder)
    return exp_file


def import_map_series(fmapseries, rasterfolder='C:/bin', folder='C:/bin', filename='map_series', rasterfilename='map'):
    """
    import map series data set
    :param fmapseries: string for the input time series data frame. Must have 'Date' and 'File" as fields
    :param rasterfolder: string path to raster dataset folder
    :param folder: string path to file folder
    :param filename: string file name
    :param suff: string of filename suffix
    :return: string - file path
    """
    from shutil import copyfile
    #
    # import data
    map_series_df = pd.read_csv(fmapseries, sep=';', engine='python')
    map_series_df = dataframe_prepro(dataframe=map_series_df, strfields='Date,File')
    dates = map_series_df['Date'].values
    files = map_series_df['File'].values
    #
    # process data
    new_files = list()
    for i in range(len(dates)):
        src = files[i]
        lcl_date = dates[i]
        lcl_filenm = rasterfilename + '_' + str(lcl_date) + '.asc'
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


def import_etpat_series(finputseries, rasterfolder='C:/bin', folder='C:/bin', filename='map_series',
                        rasterfilename='map', tui=False):
    if tui:
        from tui import status
        status('importing dataframe')
    # import data
    map_series_df = pd.read_csv(finputseries, sep=';', engine='python')
    map_series_df = dataframe_prepro(dataframe=map_series_df, strfields='Date,File')
    dates = map_series_df['Date'].values
    files = map_series_df['File'].values
    #
    # process data
    if tui:
        status('normalizing rasters')
    rasters = list()
    for i in range(len(dates)):
        src = files[i]
        meta, lcl_raster = input.asc_raster(src)
        rasters.append(lcl_raster)
    rasters = np.array(rasters)
    min_value = np.min(rasters)
    max_value = np.max(rasters)
    a_value = 1 / (max_value - min_value)
    b_value = 1 - (a_value * max_value)
    rasters_norm = (a_value * rasters) + b_value
    #
    #
    # export data
    if tui:
        status('exporting rasters')
    new_files = list()
    for i in  range(len(dates)):
        lcl_date = dates[i]
        lcl_filename = rasterfilename + '_' + lcl_date
        lcl_file = output.asc_raster(rasters_norm[i], meta, folder=rasterfolder, filename=lcl_filename)
        new_files.append(lcl_file)
    #
    if tui:
        status('exporting series')
    exp_df = pd.DataFrame({'Date': dates, 'File': new_files})
    exp_file = folder + '/' + filename + '.txt'
    exp_df.to_csv(exp_file, sep=';', index=False)
    return exp_file


def get_views_rasters(fmapseries, mapvar='ET', mapid='etpat', vmin='local', vmax='local', tui=False):
    from visuals import plot_map_view
    import os
    # import data
    map_series_df = pd.read_csv(fmapseries, sep=';', engine='python')
    map_series_df = dataframe_prepro(dataframe=map_series_df, strfields='Date,File')
    dates = map_series_df['Date'].values
    files = map_series_df['File'].values
    if tui:
        from tui import status
        status('exporting raster views')
    # process data
    filenames_lst = list()
    for i in range(len(dates)):
        lcl_filename = os.path.basename(files[i])
        lcl_filename = lcl_filename.split('.')[0]
        lcl_folder = os.path.dirname(files[i])
        meta, mp = input.asc_raster(files[i])
        if vmin == 'local':
            v_min = np.min(mp)
        else:
            v_min = float(vmin)
        if vmax == 'local':
            v_max = np.max(mp)
        else:
            v_max = float(vmax)
        ranges = [v_min, v_max]
        plot_map_view(mp, meta, ranges, mapid, mapttl='{} | {}'.format(mapvar, dates[i]),
                      filename=lcl_filename, folder=lcl_folder)


def compute_zmap_series(fvarseries, ftwi, fshru, fhistograms, var, filename='var_zmap_series', folder='C:/bin',
                        folderseries='C:/bin', tui=False):
    """
    compute zmaps from raster series
    :param fvarseries:
    :param ftwi:
    :param fshru:
    :param fhistograms:
    :param filename:
    :param folder:
    :param folderseries:
    :param tui:
    :return:
    """
    from output import zmap
    from hydrology import built_zmap
    import os

    def extract_histdata(fhistograms):
        dataframe = pd.read_csv(fhistograms, sep=';')
        dataframe = dataframe_prepro(dataframe, strf=False)
        dataframe = dataframe.set_index(dataframe.columns[0])
        shru_ids = dataframe.columns.astype('int')
        twi_bins = dataframe.index.values
        count_matrix = dataframe.values
        return count_matrix, twi_bins, shru_ids
    #
    # import data
    if tui:
        from tui import status
        status('loading series')
    map_series_df = pd.read_csv(fvarseries, sep=';', engine='python')
    map_series_df = dataframe_prepro(dataframe=map_series_df, strfields='Date,File')
    dates = map_series_df['Date'].values
    files = map_series_df['File'].values
    #
    if tui:
        status('loading rasters')
    meta, twi = input.asc_raster(ftwi)
    meta, shru = input.asc_raster(fshru)
    #
    if tui:
        status('loading histograms')
    count, twibins, shrubins = extract_histdata(fhistograms=fhistograms)
    #
    # process data
    if tui:
        status('computing zmaps')
    new_files = list()
    for i in range(len(dates)):
        #
        lcl_folder = os.path.dirname(files[i])
        lcl_filenm = os.path.basename(files[i])
        lcl_new_filename = 'zmap_' +  var + '_' + dates[i]
        #
        meta, lcl_var = input.asc_raster(files[i])
        lcl_zmap = built_zmap(varmap=lcl_var, twi=twi, shru=shru, twibins=twibins, shrubins=shrubins)
        exp_file = output.zmap(zmap=lcl_zmap, twibins=twibins, shrubins=shrubins, folder=lcl_folder, filename=lcl_new_filename)
        new_files.append(exp_file)
    #
    # export data
    exp_df = pd.DataFrame({'Date': dates, 'File': new_files})
    exp_file = folderseries + '/' + filename + '.txt'
    exp_df.to_csv(exp_file, sep=';', index=False)
    return exp_file


def import_shru_series(flulcseries, flulcparam, fsoils, fsoilsparam, fshruparam, rasterfolder='C:/bin', folder='C:/bin',
                       filename='shru_series', suff='', tui=False):
    # todo docstring
    # import data
    lulc_series_df = pd.read_csv(flulcseries, sep=';', engine='python')
    lulc_series_df = dataframe_prepro(dataframe=lulc_series_df, strfields='Date,File')
    #print(lulc_series_df)
    dates = lulc_series_df['Date'].values
    files = lulc_series_df['File'].values
    #
    # process data
    new_files = list()
    for i in range(len(dates)):
        lcl_date = dates[i]
        if suff == '':
            lcl_filename = 'shru_' + str(lcl_date)
        else:
            lcl_filename = suff + '_' + 'shru_' + str(lcl_date)
        if tui:
            print('procesing file:\t{}'.format(lcl_filename))
        # process data
        shru_file = map_shru(flulc=files[i], flulcparam=flulcparam, fsoils=fsoils, fsoilsparam=fsoilsparam,
                             fshruparam=fshruparam, folder=rasterfolder, filename=lcl_filename)
        # print(lcl_expf)
        new_files.append(shru_file)
    #
    # export data
    exp_df = pd.DataFrame({'Date': dates, 'File': new_files})
    exp_file = folder + '/' + filename + '.txt'
    exp_df.to_csv(exp_file, sep=';', index=False)
    return exp_file


def shru_param(flulcparam, fsoilsparam, folder='C:/bin', filename='shru_param'):
    # todo docstring
    # extract data
    lulc_df = pd.read_csv(flulcparam, sep=';', engine='python')
    lulc_df = dataframe_prepro(lulc_df, strfields='LULCName,ConvertTo,ColorLULC')
    #print(lulc_df.to_string())
    soils_df = pd.read_csv(fsoilsparam, sep=';', engine='python')
    soils_df = dataframe_prepro(soils_df, strfields='SoilName,ColorSoil')
    #print(soils_df.to_string())
    lulc_ids = lulc_df['IdLULC'].values
    soils_ids = soils_df['IdSoil'].values
    #
    # process
    shru_ids = list()
    shru_nm = list()
    shru_lulc_ids = list()
    shru_soils_ids = list()
    for i in range(len(lulc_ids)):
        for j in range(len(soils_ids)):
            lcl_shru_id = lulc_ids[i] * 100 + soils_ids[j]
            lcl_shru_nm = lulc_df['LULCName'].values[i] + '_' + soils_df['SoilName'].values[j]
            shru_ids.append(lcl_shru_id)
            shru_nm.append(lcl_shru_nm)
            shru_lulc_ids.append(lulc_ids[i])
            shru_soils_ids.append(soils_ids[j])
    shru_df = pd.DataFrame({'IdSHRU':shru_ids, 'SHRUName': shru_nm, 'IdLULC':shru_lulc_ids, 'IdSoil':shru_soils_ids})
    #print(shru_df.to_string())
    shru_df = shru_df.join(lulc_df.set_index('IdLULC'), on='IdLULC')
    #print(shru_df.to_string())
    shru_df = shru_df.join(soils_df.set_index('IdSoil'), on='IdSoil')
    shru_df['f_EfRootZone'] = shru_df['Porosity'].values * shru_df['f_RootDepth'].values
    print(shru_df.to_string())
    #
    # export
    exp_file = folder + '/' + filename + '.txt'
    shru_df.to_csv(exp_file, sep=';', index=False)
    return exp_file


def osa(fseries, fld_obs='Qobs', fld_sim='Q', fld_date='Date', folder='C:/bin', tui=False, var=True, log=True):
    """
    Analyst of Observed - Simulated series
    :param fseries: string filepath to series dataframe
    :param fld_obs: string of field of observed series data
    :param fld_sim: string of field of simulated series data
    :param fld_date: string of date field
    :param folder: string of export directory
    :param tui: boolean to control TUI displays
    :return: tuple of strings of exported files
    """
    import time, datetime
    import analyst
    from visuals import pannel_obs_sim_analyst
    #
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
        status('performing obs vs. sim analysis')
    #
    #
    #
    #
    # extract Dataframe
    def_df = pd.read_csv(fseries, sep=';', engine='python')
    if var:
        def_df = dataframe_prepro(def_df, strf=False, date=True, datefield=fld_date)
    else:
        def_df = dataframe_prepro(def_df, strf=False)
    #
    # extract obs and sim arrays:
    obs = def_df[fld_obs].values
    #sim = obs - 0.1 * obs
    sim = def_df[fld_sim].values
    if log:
        obslog = np.log10(obs)
        simlog = np.log10(sim)
    #
    #
    # **** Series Analysis ****
    # Error series analyst
    e = analyst.error(obs=obs, sim=sim)
    se = analyst.sq_error(obs=obs, sim=sim)
    if log:
        elog = analyst.error(obs=obslog, sim=simlog)
        selog = analyst.sq_error(obs=obslog, sim=simlog)
    # built Dataframe
    if var:
        series_df = pd.DataFrame({'Date':def_df[fld_date], 'Obs':obs, 'Sim':sim, 'E':e, 'SE':se})
    else:
        series_df = pd.DataFrame({'Id':def_df.index.values, 'Obs':obs, 'Sim':sim, 'E':e, 'SE':se})
    if log:
        series_df['Obslog'] = obslog
        series_df['Simlog'] = simlog
        series_df['Elog'] = elog
        series_df['SElog'] = selog
    # coefs analyst of series
    pbias = analyst.pbias(obs=obs, sim=sim)
    rmse = analyst.rmse(obs=obs, sim=sim)
    nse = analyst.nse(obs=obs, sim=sim)
    linreg = analyst.linreg(obs=obs, sim=sim)
    kge = analyst.kge(obs=obs, sim=sim)
    if log:
        rmselog = analyst.rmse(obs=obslog, sim=simlog)
        nselog = analyst.nse(obs=obslog, sim=simlog)
        kgelog = analyst.kge(obs=obslog, sim=simlog)
    #
    #
    #
    #
    #
    # **** Frequency analysis ****
    freq_obs = analyst.frequency(series=obs)
    freq_sim = analyst.frequency(series=sim)
    obs_freq = freq_obs['Values']
    sim_freq = freq_sim['Values']
    if log:
        obslog_freq = np.log10(obs_freq)
        simlog_freq = np.log10(sim_freq)
    #
    # Error frequency analyst
    e_freq = analyst.error(obs=obs_freq, sim=sim_freq)
    se_freq = analyst.sq_error(obs=obs_freq, sim=sim_freq)
    if log:
        elog_freq = analyst.error(obs=obslog_freq, sim=simlog_freq)
        selog_freq = analyst.sq_error(obs=obslog_freq, sim=simlog_freq)
    #
    # built dataframe
    freq_df = pd.DataFrame({'Percentiles': freq_obs['Percentiles'], 'Exeedance': freq_obs['Exeedance'],
                            'ProbabObs': freq_obs['Probability'], 'ValuesObs': freq_obs['Values'],
                            'ProbabSim': freq_sim['Probability'],
                            'ValuesSim': freq_sim['Values'],
                            'E':e_freq, 'SE':se_freq })
    if log:
        freq_df['ValuesObslog'] = obslog_freq
        freq_df['ValuesSimlog'] = simlog_freq
        freq_df['Elog'] = elog_freq
        freq_df['SElog'] = selog_freq
    #
    # coefs analyst of series
    rmse_freq = analyst.rmse(obs=obs_freq, sim=sim_freq)
    linreg_freq = analyst.linreg(obs=obs_freq, sim=sim_freq)
    if log:
        rmselog_freq = analyst.rmse(obs=obslog_freq, sim=simlog_freq)
        linreg_freq_log = analyst.linreg(obs=obslog_freq, sim=simlog_freq)
    #
    # built dataframe of parameters
    if log:
        params = ('PBias', 'RMSE', 'RMSElog', 'NSE', 'NSElog', 'KGE', 'KGElog', 'A', 'B', 'R', 'P', 'SD',
                  'RMSE-CFC', 'RMSElog-CFC', 'R-CFC', 'Rlog-CFC')
        values = (pbias, rmse, rmselog, nse, nselog, kge, kgelog, linreg['A'], linreg['B'], linreg['R'], linreg['P'],
                  linreg['SD'], rmse_freq, rmselog_freq, linreg_freq['R'], linreg_freq_log['R'])
        param_df = pd.DataFrame({'Parameter': params, 'Value': values})
    else:
        params = ('PBias', 'RMSE', 'NSE', 'KGE', 'A', 'B', 'R', 'P', 'SD',
                  'RMSE-CFC', 'R-CFC')
        values = (pbias, rmse, nse, kge, linreg['A'], linreg['B'], linreg['R'], linreg['P'],
                  linreg['SD'], rmse_freq, linreg_freq['R'])
        param_df = pd.DataFrame({'Parameter': params, 'Value': values})
    #
    report_lst.append('\n\nAnalyst Parameter Results:\n\n')
    report_lst.append(param_df.to_string(index=False))
    report_lst.append('\n\n')
    #
    #
    #
    # **** Export Data ****
    if tui:
        print('exporting analysis data and visuals...')
    # 1) series data
    exp_file1 = folder + '/' + 'analyst_series.txt'
    series_df.to_csv(exp_file1, sep=';', index=False)
    # 2) frequency data
    exp_file2 = folder + '/' + 'analyst_freq.txt'
    freq_df.to_csv(exp_file2, sep=';', index=False)
    # 3) parameters data
    exp_file3 = folder + '/' + 'analyst_params.txt'
    param_df.to_csv(exp_file3, sep=';', index=False)
    #
    # export visual:
    if var:
        exp_file4 = pannel_obs_sim_analyst(series=series_df, freq=freq_df, params=param_df, folder=folder,
                                           title='Series Analysis', units='flow')
    else:
        exp_file4 = pannel_obs_sim_analyst(series=series_df, freq=freq_df, params=param_df, units='signal',
                                           folder=folder, log=log, fld_date='Id', title='Signal Analysis')
    #
    #
    tf = time.time()
    output_df = pd.DataFrame({'Output files': (exp_file1, exp_file2, exp_file3, exp_file4)})
    report_lst.append(output_df.to_string(index=False))
    report_lst.append('\n\n')
    report_lst.insert(2, 'Execution enlapsed time: {:.3f} seconds\n'.format(tf - t0))
    export_report(report_lst, filename='REPORT__analyst', folder=folder, tui=tui)
    #
    return (exp_file1, exp_file2, exp_file3, exp_file4)


def osa_map(fseries, fhistograms, type, var='ETPat', filename='obssim_maps_analyst', folder='C:/bin', tui=True):
    import time, datetime
    from os import mkdir
    import analyst
    from hydrology import extract_zmap_signal
    from visuals import plot_map_analyst

    def extract_histdata(fhistograms):
        dataframe = pd.read_csv(fhistograms, sep=';')
        dataframe = dataframe_prepro(dataframe, strf=False)
        dataframe = dataframe.set_index(dataframe.columns[0])
        shru_ids = dataframe.columns.astype('int')
        twi_bins = dataframe.index.values
        count_matrix = dataframe.values
        return count_matrix, twi_bins, shru_ids

    def extract_series_data(dataframe, count, fld, type='raster'):
        maps_lst = list()
        signal_lst = list()
        print(dataframe.to_string())
        for i in range(len(dataframe)):
            map_file = dataframe[fld].values[i]
            print(map_file)
            if type == 'zmap':
                lcl_map, ybins, xbins = input.zmap(map_file)
                signal = extract_zmap_signal(lcl_map, count)
            elif type == 'raster':
                meta, lcl_map = input.asc_raster(map_file)
                signal = lcl_map.flatten()
            maps_lst.append(lcl_map)
            signal_lst.append(signal)
        full_signal = np.array(signal_lst).flatten()
        #lcl_x = np.arange(0, len(full_signal))
        #plt.plot(lcl_x, full_signal)
        #plt.show()
        return maps_lst, signal_lst, full_signal

    #
    # report setup  # todo report
    t0 = time.time()
    report_lst = list()
    report_lst.append('Execution timestamp: {}\n'.format(datetime.datetime.now()))
    report_lst.append('Process: OBS-SIM MAP ANALYST\n')
    input_files_df = pd.DataFrame({'Input files': (fseries,)})
    report_lst.append(input_files_df.to_string(index=False))
    out_files = list()
    #
    if tui:
        from tui import status
        status('performing obs vs. sim map analysis')
    #
    # extract Dataframe
    def_df = pd.read_csv(fseries, sep=';', engine='python')
    def_df = dataframe_prepro(def_df, strfields='File_obs,File_sim,Date')
    #
    count, twibins, shrubins = extract_histdata(fhistograms=fhistograms)
    #
    maps_obs_lst, signal_obs_lst, full_signal_obs = extract_series_data(def_df, count, 'File_obs', type=type)
    maps_sim_lst, signal_sim_lst, full_signal_sim = extract_series_data(def_df,  count, 'File_sim', type=type)
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
        lcl_kge = analyst.kge(np.append([1], signal_obs_lst[i]), np.append([1], signal_sim_lst[i]))
        map_kge.append(lcl_kge)
        #
        lcl_r = analyst.linreg(np.append([1], signal_obs_lst[i]), np.append([1], signal_sim_lst[i]))['R']
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
    out_file1 = folder + '/{}_{}_map_analyst_local.txt'.format(var, type)
    out_df.to_csv(out_file1, sep=';', index=False)
    out_files.append(out_file1)
    #
    #
    # 4) compute metrics for full signal
    full_serial_df = pd.DataFrame({'Id':np.arange(0, len(full_signal_obs)), 'Obs':full_signal_obs, 'Sim':full_signal_sim})
    out_file2 = folder + '/{}_{}_map_analyst_signals.txt'.format(var, type)
    full_serial_df.to_csv(out_file2, sep=';', index=False)
    out_files.append(out_file2)
    #
    #
    if tui:
        status('processing full signal')
    osa(out_file2, fld_obs='Obs', fld_sim='Sim', fld_date='Id', log=False, var=False, folder=folder)
    #
    #
    #
    #
    # Export visuals
    if tui:
        status('exporting map visuals')
    map_vmin = 0.0  # np.min((maps_obs_lst, maps_obs_lst))
    map_vmax = np.max((maps_obs_lst, maps_obs_lst))
    ranges = (map_vmin, map_vmax)
    map_metric_vmin = np.min(metric_maps)
    map_metric_vmax = np.max(metric_maps)
    mapmax = np.max((np.abs(map_metric_vmin), np.abs(map_metric_vmax)))
    metricranges = (-mapmax, mapmax)
    visual_dir = folder + '/{}_{}_map_analyst_visuals'.format(var, type)
    mkdir(visual_dir)
    for i in range(len(out_df)):
        lcl_date = out_df['Date'].values[i]
        lcl_filename = '{}_{}_map_analyst_local_{}'.format(var, type, lcl_date)
        lcl_ttl = '{} | {}'.format(var, lcl_date)
        metrics_dct = {'Error': out_df['Error'].values[i],
                       'SqErr': out_df['SqErr'].values[i],
                       'RMSE': out_df['RMSE'].values[i],
                       'NSE': out_df['NSE'].values[i],
                       'KGE': out_df['KGE'].values[i],
                       'R': out_df['R'].values[i]}
        lcl_out = plot_map_analyst(obs=maps_obs_lst[i], sim=maps_sim_lst[i], metric=metric_maps[i],
                                     obs_sig=signal_obs_lst[i], sim_sig=signal_sim_lst[i], ranges=ranges,
                                     metricranges=metricranges, metric_sig=metric_signal[i], metrics_dct=metrics_dct,
                                     filename=lcl_filename, folder=visual_dir, ttl=lcl_ttl)
        out_files.append(lcl_out)
        if tui:
            status('exporting visual of {}'.format(lcl_ttl))
    #
    tf = time.time()
    output_df = pd.DataFrame({'Output files': out_files})
    report_lst.append(output_df.to_string(index=False))
    report_lst.append('\n\n')
    report_lst.insert(2, 'Execution enlapsed time: {:.3f} seconds\n'.format(tf - t0))
    export_report(report_lst, filename='REPORT__{}_{}_map_analyst'.format(var, type), folder=folder, tui=tui)
    #
    return {'Local':out_file1, 'Full':out_file2, 'Visuals':out_files}


def slh_calib(fseries, fhydroparam, fshruparam, fhistograms, fbasinhists, fbasin, ftwi, fshru, mapback=False,
              mapraster=False, mapvar='all', mapdates='all', qobs=True, folder='C:/bin', wkpl=False, label='',
              tui=False):
    from backend import create_rundir
    from visuals import pannel_calib_valid
    from os import mkdir

    def extract_calib_valid(dataframe, fvalid=0.333):
        size = len(dataframe)
        cut_id = int(size * (1 - fvalid))
        cut_date = dataframe['Date'].values[cut_id]
        calib_df = dataframe.query('Date < "{}"'.format(cut_date))
        valid_df = dataframe.query('Date >= "{}"'.format(cut_date))
        return calib_df, valid_df, cut_date

    #
    # 1) Run Folder setup
    if tui:
        from tui import status
        status('setting folders')
    if wkpl:  # if the passed folder is a workplace, create a sub folder
        if label != '':
            label = label + '_'
        folder = create_rundir(label=label + 'SLH', wkplc=folder)
    #
    calibration_folder = folder + '/calibration_period'
    mkdir(calibration_folder)
    #
    validation_folder = folder + '/validation_period'
    mkdir(validation_folder)
    #
    full_folder = folder + '/full_period'
    mkdir(full_folder)
    #
    # 2) import series and split
    if tui:
        status('importing series')
    series_df = pd.read_csv(fseries, sep=';')
    series_df = dataframe_prepro(series_df, strf=False, date=True, datefield='Date')
    calib_df, valid_df, cut_date = extract_calib_valid(series_df, fvalid=0.33)
    #
    #
    # 3) export separate series:
    if tui:
        status('splitting series')
    fcalib_series = folder + '/' + 'input_series_calibration_period.txt'
    calib_df.to_csv(fcalib_series, sep=';', index=False)
    #
    fvalid_series = folder + '/' + 'input_series_validation_period.txt'
    valid_df.to_csv(fvalid_series, sep=';', index=False)
    #
    ffull_series = folder + '/' + 'input_series_full_period.txt'
    series_df.to_csv(ffull_series, sep=';', index=False)
    #
    #
    # 4) run SLH for calibration period
    if tui:
        status('running SLH for calibration period')
    calib_dct = slh(fseries=fcalib_series, fhydroparam=fhydroparam, fshruparam=fshruparam, fhistograms=fhistograms,
                    fbasinhists=fbasinhists, fbasin=fbasin, ftwi=ftwi, fshru=fshru,
                    mapback=mapback, mapraster=mapraster, mapvar=mapvar, mapdates=mapdates, qobs=qobs,
                    folder=calibration_folder, wkpl=False, label=label, tui=tui)
    fsim_calib = calib_dct['Series']
    #
    # 5) sun OSA for calibration period
    if tui:
        status('running OSA for calibration period')
    osa_files1 = osa(fseries=fsim_calib, fld_obs='Qobs', fld_sim='Q', fld_date='Date', folder=calibration_folder, tui=tui)
    #
    #
    # 6) run SLH for validation period
    if tui:
        status('running SLH for validation period')
    valid_dct = slh(fseries=fvalid_series, fhydroparam=fhydroparam, fshruparam=fshruparam, fhistograms=fhistograms,
                    fbasinhists=fbasinhists, fbasin=fbasin, ftwi=ftwi, fshru=fshru,
                    mapback=mapback, mapraster=mapraster, mapvar=mapvar, mapdates=mapdates, qobs=qobs,
                    folder=validation_folder, wkpl=False, label=label, tui=tui)
    fsim_valid = valid_dct['Series']
    #
    # 7) sun OSA for validation period
    osa_files2 = osa(fseries=fsim_valid, fld_obs='Qobs', fld_sim='Q', fld_date='Date', folder=validation_folder, tui=tui)
    #
    #
    #
    # 6) run SLH for full period
    if tui:
        status('running SLH for full period')
    full_dct = slh(fseries=ffull_series, fhydroparam=fhydroparam, fshruparam=fshruparam, fhistograms=fhistograms,
                    fbasinhists=fbasinhists, fbasin=fbasin, ftwi=ftwi, fshru=fshru,
                    mapback=False, mapraster=mapraster, mapvar=mapvar, mapdates=mapdates, qobs=qobs,
                    folder=full_folder, wkpl=False, label=label, tui=tui)
    fsim_full = full_dct['Series']
    #
    # 7) sun OSA for full period
    osa_files3 = osa(fseries=fsim_full, fld_obs='Qobs', fld_sim='Q', fld_date='Date', folder=full_folder, tui=tui)
    #
    if tui:
        status('exporting pannel')
    full = osa_files3[0]
    cal = osa_files1[0]
    val = osa_files2[0]
    freq = osa_files3[1]
    pfull = osa_files3[2]
    pcal = osa_files1[2]
    pval = osa_files2[2]
    df_full = pd.read_csv(full, sep=';', parse_dates=['Date'])
    df_cal = pd.read_csv(cal, sep=';', parse_dates=['Date'])
    df_val = pd.read_csv(val, sep=';', parse_dates=['Date'])
    df_freq = pd.read_csv(freq, sep=';')
    p_full = pd.read_csv(pfull, sep=';')
    p_cal = pd.read_csv(pcal, sep=';')
    p_val = pd.read_csv(pval, sep=';')
    pannel_calib_valid(df_full, df_cal, df_val, df_freq, p_full, p_cal, p_val, folder=folder)

    return {'Folder':folder, 'CalibFolder':calibration_folder,
            'ValidFolder':validation_folder, 'FullFolder':full_folder}


def slh(fseries, fhydroparam, fshruparam, fhistograms, fbasinhists, fbasin, ftwi, fshru, mapback=False,
        mapraster=False, mapvar='all', mapdates='all', qobs=False, folder='C:/bin', wkpl=False, label='',
        tui=False):
    # todo 1) docstring
    import time, datetime
    from shutil import copyfile
    from hydrology import topmodel_sim, map_back
    from visuals import pannel_topmodel
    from backend import create_rundir

    def extract_histdata(fhistograms):
        dataframe = pd.read_csv(fhistograms, sep=';')
        dataframe = dataframe_prepro(dataframe, strf=False)
        dataframe = dataframe.set_index(dataframe.columns[0])
        shru_ids = dataframe.columns.astype('int')
        twi_bins = dataframe.index.values
        count_matrix = dataframe.values
        return count_matrix, twi_bins, shru_ids

    def extract_twi_avg(twibins, count):
        twi_sum = 0
        for i in range(len(twibins)):
            twi_sum = twi_sum + (twibins[i] * np.sum(count[i]))
        return twi_sum / np.sum(count)
    #
    #
    # Run Folder setup
    if wkpl:  # if the passed folder is a workplace, create a sub folder
        if label != '':
            label = label + '_'
        folder = create_rundir(label=label + 'SLH', wkplc=folder)
    #
    #
    # time and report setup
    t0 = time.time()
    report_lst = list()
    report_lst.append('Execution timestamp: {}\n'.format(datetime.datetime.now()))
    report_lst.append('Process: STABLE LULC HYDROLOGY\n')
    input_files_df = pd.DataFrame({'Input files': (fseries, fhydroparam, fshruparam, fhistograms, fbasinhists,
                                                   fbasin, ftwi, fshru)})
    report_lst.append(input_files_df.to_string(index=False))
    #
    #
    #
    #
    # Importing section
    init = time.time()
    if tui:
        from tui import status
        print('\n\t**** Load Data Protocol ****\n')
        status('loading time series')

    series_df =  pd.read_csv(fseries, sep=';')
    series_df = dataframe_prepro(series_df, strf=False, date=True, datefield='Date')
    #
    if tui:
        status('loading hydrology parameters')
    hydroparam_df = pd.read_csv(fhydroparam, sep=';')
    hydroparam_df = dataframe_prepro(hydroparam_df, 'Parameter')
    #
    # extract set values
    m = hydroparam_df[hydroparam_df['Parameter'] == 'm']['Set'].values[0]
    qo = hydroparam_df[hydroparam_df['Parameter'] == 'qo']['Set'].values[0]
    cpmax = hydroparam_df[hydroparam_df['Parameter'] == 'cpmax']['Set'].values[0]
    sfmax = hydroparam_df[hydroparam_df['Parameter'] == 'sfmax']['Set'].values[0]
    erz = hydroparam_df[hydroparam_df['Parameter'] == 'erz']['Set'].values[0]
    ksat = hydroparam_df[hydroparam_df['Parameter'] == 'ksat']['Set'].values[0]
    c = hydroparam_df[hydroparam_df['Parameter'] == 'c']['Set'].values[0]
    lat = hydroparam_df[hydroparam_df['Parameter'] == 'lat']['Set'].values[0]
    k = hydroparam_df[hydroparam_df['Parameter'] == 'k']['Set'].values[0]
    n = hydroparam_df[hydroparam_df['Parameter'] == 'n']['Set'].values[0]
    #
    # Shru parameters
    if tui:
        status('loading SHRU parameters')
    shru_df = pd.read_csv(fshruparam, sep=';')
    shru_df = dataframe_prepro(shru_df, 'SHRUName,LULCName,SoilName')
    #
    # extract count matrix (full map extension)
    if tui:
        status('loading histograms of full extension')
    count, twibins, shrubins = extract_histdata(fhistograms=fhistograms)
    #
    # extract count matrix (basin)
    if tui:
        status('loading histograms of basin')
    basincount, twibins2, shrubins2 = extract_histdata(fhistograms=fbasinhists)
    #
    # get basin boundary conditions
    if tui:
        status('loading boundary conditions')
    meta = input.asc_raster_meta(fbasin)
    area = np.sum(basincount) * meta['cellsize'] * meta['cellsize']
    qt0 = 0.01  # fixed
    if qobs:
        qt0 = series_df['Q'].values[0]
    lamb = extract_twi_avg(twibins, basincount)
    #
    end = time.time()
    report_lst.append('\n\nLoading enlapsed time: {:.3f} seconds\n'.format(end - init))
    if tui:
        print('\nloading enlapsed time: {:.3f} seconds'.format(end - init))
    #
    #
    #
    #
    # ****** Simulation section ******
    init = time.time()
    if tui:
        print('\n\t**** Simulation Protocol ****\n')
        status('running simulation')
    sim_dct = topmodel_sim(series=series_df, shruparam=shru_df, twibins=twibins, countmatrix=count, lamb=lamb,
                           qt0=qt0, m=m, qo=qo, cpmax=cpmax, sfmax=sfmax, erz=erz, ksat=ksat, c=c, lat=lat,
                           k=k, n=n, area=area, basinshadow=basincount, tui=False, qobs=qobs, mapback=mapback,
                           mapvar=mapvar,
                           mapdates=mapdates)
    sim_df = sim_dct['Series']
    if mapback:
        mapped = sim_dct['Maps']
    end = time.time()
    report_lst.append('Simulation enlapsed time: {:.3f} seconds\n'.format(end - init))
    if tui:
        print('\nsimulation enlapsed time: {:.3f} seconds'.format(end - init))
    #
    #
    #
    # exporting section
    init = time.time()
    if tui:
        print('\n\t**** Export Data Protocol ****\n')
        status('exporting simulated time series')
    #
    # export time series
    exp_file1 = folder + '/' + 'sim_series.txt'
    sim_df.to_csv(exp_file1, sep=';', index=False)
    #
    # exporthistograms
    if tui:
        status('exporting histograms')
    exp_file2 = folder + '/' + 'sim_histograms.txt'
    copyfile(src=fhistograms, dst=exp_file2)
    #
    # export parameters
    if tui:
        status('exporting run parameters')
    exp_file3 = folder + '/' + 'sim_parameters.txt'
    copyfile(src=fhydroparam, dst=exp_file3)
    #
    # export visual pannel
    if tui:
        status('exporting visual results')
    exp_file4 = pannel_topmodel(sim_df, grid=True, show=False, qobs=qobs, folder=folder)
    #
    #
    #
    # export maps
    if mapback:
        if tui:
            status('exporting variable maps')
        from os import mkdir
        if mapvar == 'all':
            mapvar = 'P-Temp-IRA-IRI-PET-D-Cpy-TF-Sfs-R-RSE-RIE-RC-Inf-Unz-Qv-Evc-Evs-Tpun-Tpgw-ET-VSA'
        mapvar_lst = mapvar.split('-')  # load string variables alias to list
        # get stamps
        stamp = sim_dct['MappedDates']
        #
        #
        # Zmaps exporting
        mapfiles_lst = list()
        zmaps_dct = dict()
        for var in mapvar_lst:  # loop across all variables
            if tui:
                status('exporting {} zmaps'.format(var))
            #
            # make dir
            lcl_folder = folder + '/sim_' + var
            mkdir(lcl_folder)  # make new diretory
            lcl_files = list()
            for t in range(len(stamp)):  # loop across all timesteps
                lcl_filename = 'zmap_' + var + '_' + str(stamp[t]).split(sep=' ')[0] + '.txt'
                lcl_file = lcl_folder + '/' + lcl_filename
                lcl_files.append(lcl_file)
                # export local dataframe to text file in local folder
                lcl_exp_df = pd.DataFrame(mapped[var][t], index=twibins, columns=shrubins)
                lcl_exp_df.to_csv(lcl_file, sep=';', index_label='TWI\SHRU')
            #
            # export map list file to main folder:
            lcl_exp_df = pd.DataFrame({'Date': stamp, 'File': lcl_files})
            lcl_file = folder + '/' + 'sim_zmaps_' + var + '.txt'
            lcl_exp_df.to_csv(lcl_file, sep=';', index=False)
            zmaps_dct[var] = lcl_file
            mapfiles_lst.append(lcl_file)
        #
        # Raster maps exporting
        if mapraster:
            from hydrology import map_back
            from geo import mask
            from visuals import plot_map_view
            if tui:
                status('raster map export section')
            if tui:
                status('importing twi raster')
            meta, twi = input.asc_raster(ftwi)
            if tui:
                status('importing shru raster')
            meta, shru = input.asc_raster(fshru)
            # loop in variables
            raster_dct = dict()
            for var in mapvar_lst:
                if tui:
                    status('exporting {} raster maps'.format(var))
                lcl_folder = folder + '/sim_' + var
                lcl_files = list()
                for t in range(len(stamp)):  # loop across all timesteps
                    stamp_str = str(stamp[t]).split(sep=' ')[0]
                    lcl_filename = 'raster_' + var + '_' + stamp_str
                    lcl_file = lcl_folder + '/' + lcl_filename
                    lcl_files.append(lcl_file + '.asc')
                    #
                    # express the map
                    mp = map_back(zmatrix=mapped[var][t], a1=twi, a2=shru, bins1=twibins, bins2=shrubins)
                    #
                    # export view
                    # smart mapid selector
                    if var == 'D':
                        mapid = 'deficit'
                    elif var in set(['Cpy', 'Sfs', 'Unz']):
                        mapid = 'stock'
                    elif var in set(['R', 'Inf', 'TF', 'IRA', 'IRI', 'Qv', 'P']):
                        mapid = 'flow'
                    elif var in set(['ET', 'Evc', 'Evs', 'Tpun', 'Tpgw']):
                        mapid = 'flow_v'
                    elif var == 'VSA':
                        mapid = 'VSA'
                    else:
                        mapid = 'flow'
                    ranges = [0, np.max(sim_df[var].values)]
                    plot_map_view(mp, meta, ranges, mapid, mapttl='{} | {}'.format(var, stamp_str),
                                  folder=lcl_folder, filename=lcl_filename, show=False)
                    # export raster map
                    output.asc_raster(mp, meta, lcl_folder, lcl_filename)
                #
                # export map list file to main folder:
                lcl_exp_df = pd.DataFrame({'Date': stamp, 'File': lcl_files})
                lcl_file = folder + '/' + 'sim_raster_' + var + '.txt'
                lcl_exp_df.to_csv(lcl_file, sep=';', index=False)
                mapfiles_lst.append(lcl_file)
                raster_dct[var] = lcl_file
    #
    #
    #
    #
    # report protocols
    end = time.time()
    report_lst.append('Exporting enlapsed time: {:.3f} seconds\n'.format(end - init))
    if tui:
        print('Exporting enlapsed time: {:.3f} seconds'.format(end - init))
    #
    tf = time.time()
    if tui:
        print('\nExecution enlapsed time: {:.3f} seconds'.format(tf - t0))
        print('\n\n\n')
    #
    report_lst.insert(2, 'Execution enlapsed time: {:.3f} seconds\n'.format(tf - t0))
    if mapback:
        outfiles = [exp_file1, exp_file2, exp_file3, exp_file4]
        for e in mapfiles_lst:
            outfiles.append(e)
        output_df = pd.DataFrame({'Output files': outfiles})
    else:
        output_df = pd.DataFrame({'Output files': (exp_file1, exp_file2, exp_file3, exp_file4)})
    report_lst.append(output_df.to_string(index=False))
    export_report(report_lst, filename='REPORT__simulation', folder=folder, tui=tui)
    #
    #
    #
    #
    # return section
    out_dct = {'Series':exp_file1, 'Histograms':exp_file2, 'Parameters':exp_file3, 'Pannel':exp_file4, 'Folder':folder}
    if mapback:
        out_dct['ZMaps'] = zmaps_dct
        if mapraster:
            out_dct['Raster'] = raster_dct
    return out_dct


def calib_hydro(fseries, fhydroparam, fshruparam, fhistograms, fbasinhists, fbasin, ftwi, fshru, fetpatzmaps,
                fetpatseries, folder='C:/bin', tui=False, mapback=False, mapvar='all', mapdates='all', qobs=True,
                generations=100, popsize=200, metric='NSE', label=''):
    # todo docstring
    from hydrology import avg_2d, topmodel_sim, map_back, topmodel_calib
    from visuals import pannel_topmodel
    from backend import create_rundir
    import time
    from os import mkdir

    def extract_calib_valid(dataframe, fvalid=0.333):
        size = len(dataframe)
        cut_id = int(size * (1 - fvalid))
        cut_date = dataframe['Date'].values[cut_id]
        calib_df = dataframe.query('Date < "{}"'.format(cut_date))
        return calib_df, cut_date

    def extract_histdata(fhistograms):
        dataframe = pd.read_csv(fhistograms, sep=';')
        dataframe = dataframe_prepro(dataframe, strf=False)
        dataframe = dataframe.set_index(dataframe.columns[0])
        shru_ids = dataframe.columns.astype('int')
        twi_bins = dataframe.index.values
        count_matrix = dataframe.values
        return count_matrix, twi_bins, shru_ids

    def extract_twi_avg(twibins, count):
        twi_sum = 0
        for i in range(len(twibins)):
            twi_sum = twi_sum + (twibins[i] * np.sum(count[i]))
        return twi_sum / np.sum(count)

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
    #
    #
    # Run Folder setup
    if label != '':
        label = label + '_'
    folder = create_rundir(label=label + 'Hydrology_' + metric, wkplc=folder)
    #
    t0 = time.time()
    if tui:
        from tui import status
        init = time.time()
        print('\n\t**** Load Data Protocol ****\n')
        status('loading time series')
    #
    #
    #
    # IMPORT DATA
    #
    # Series
    series_df =  pd.read_csv(fseries, sep=';')
    series_df = dataframe_prepro(series_df, strf=False, date=True, datefield='Date')
    calib_df, cut_date = extract_calib_valid(series_df, fvalid=0.33)
    #
    if tui:
        status('loading hydrology parameters') #print(' >>> loading hydrology parameters...')
    hydroparam_df = pd.read_csv(fhydroparam, sep=';')
    hydroparam_df = dataframe_prepro(hydroparam_df, 'Parameter')
    #
    # extract set range values
    m_min = hydroparam_df[hydroparam_df['Parameter'] == 'm']['Min'].values[0]
    qo_min = hydroparam_df[hydroparam_df['Parameter'] == 'qo']['Min'].values[0]
    cpmax_min = hydroparam_df[hydroparam_df['Parameter'] == 'cpmax']['Min'].values[0]
    sfmax_min = hydroparam_df[hydroparam_df['Parameter'] == 'sfmax']['Min'].values[0]
    erz_min = hydroparam_df[hydroparam_df['Parameter'] == 'erz']['Min'].values[0]
    ksat_min = hydroparam_df[hydroparam_df['Parameter'] == 'ksat']['Min'].values[0]
    c_min = hydroparam_df[hydroparam_df['Parameter'] == 'c']['Min'].values[0]
    k_min = hydroparam_df[hydroparam_df['Parameter'] == 'k']['Min'].values[0]
    n_min = hydroparam_df[hydroparam_df['Parameter'] == 'n']['Min'].values[0]
    #
    #
    m_max = hydroparam_df[hydroparam_df['Parameter'] == 'm']['Max'].values[0]
    qo_max = hydroparam_df[hydroparam_df['Parameter'] == 'qo']['Max'].values[0]
    cpmax_max = hydroparam_df[hydroparam_df['Parameter'] == 'cpmax']['Max'].values[0]
    sfmax_max = hydroparam_df[hydroparam_df['Parameter'] == 'sfmax']['Max'].values[0]
    erz_max = hydroparam_df[hydroparam_df['Parameter'] == 'erz']['Max'].values[0]
    ksat_max = hydroparam_df[hydroparam_df['Parameter'] == 'ksat']['Max'].values[0]
    c_max = hydroparam_df[hydroparam_df['Parameter'] == 'c']['Max'].values[0]
    k_max = hydroparam_df[hydroparam_df['Parameter'] == 'k']['Max'].values[0]
    n_max = hydroparam_df[hydroparam_df['Parameter'] == 'n']['Max'].values[0]
    lat = hydroparam_df[hydroparam_df['Parameter'] == 'lat']['Max'].values[0]
    #
    # ranges setup
    m_rng = (m_min, m_max)
    qo_rng = (qo_min, qo_max)
    cpmax_rng = (cpmax_min, cpmax_max)
    sfmax_rng = (sfmax_min, sfmax_max)
    erz_rng = (erz_min, erz_max)
    ksat_rng = (ksat_min, ksat_max)
    c_rng = (c_min, c_max)
    k_rng = (k_min, k_max)
    n_rng = (n_min, n_max)
    #
    if tui:
        status('loading SHRU parameters')
    shru_df = pd.read_csv(fshruparam, sep=';')
    shru_df = dataframe_prepro(shru_df, 'SHRUName,LULCName,SoilName')
    #
    #
    # extract count matrix (full map extension)
    if tui:
        status('loading histograms of full extension')
    count, twibins, shrubins = extract_histdata(fhistograms=fhistograms)
    #
    #
    # extract count matrix (basin)
    if tui:
        status('loading histograms of basin')
    basincount, twibins2, shrubins2 = extract_histdata(fhistograms=fbasinhists)
    #
    #
    # extract etpat zmaps series for calibration
    if tui:
        status('loading OBS ETPat Z-maps')
    #
    # Import Observed Zmaps
    etpat_zmaps_obs_df = pd.read_csv(fetpatzmaps, sep=';')
    etpat_zmaps_obs_df = dataframe_prepro(etpat_zmaps_obs_df, strfields='File', date=True)
    # split dataframes for later
    etpat_zmaps_obs_calib_df = etpat_zmaps_obs_df.query('Date < "{}"'.format(cut_date))
    etpat_zmaps_obs_valid_df = etpat_zmaps_obs_df.query('Date >= "{}"'.format(cut_date))
    #
    # get a dataframe to store each date series
    etpat_calib_dates = pd.DataFrame({'Date': etpat_zmaps_obs_calib_df['Date']})
    etpat_valid_dates = pd.DataFrame({'Date': etpat_zmaps_obs_valid_df['Date']})
    #
    # extract dates for calibration
    etpat_dates_str_calib = ' & '.join(etpat_zmaps_obs_calib_df['Date'].astype('str').values)  # for calibration!!
    etpat_dates_str_full = ' & '.join(etpat_zmaps_obs_df['Date'].astype('str').values)  # for full
    #
    # load zmaps for calibration
    etpat_zmaps_obs_calib = list()
    for i in range(len(etpat_zmaps_obs_calib_df)):
        zmap_file = etpat_zmaps_obs_calib_df['File'].values[i]
        zmap, ybins, xbins = input.zmap(zmap_file)
        etpat_zmaps_obs_calib.append(zmap)
    #
    # extract etpat raster series for calibration
    if tui:
        status('loading OBS ETPat raster dataframe')
    etpat_raster_obs_df = pd.read_csv(fetpatseries, sep=';')
    etpat_raster_obs_df = dataframe_prepro(etpat_raster_obs_df, strfields='File', date=True)
    #
    # split dataframes for later
    etpat_raster_obs_calib_df = etpat_raster_obs_df.query('Date < "{}"'.format(cut_date))
    etpat_raster_obs_valid_df = etpat_raster_obs_df.query('Date >= "{}"'.format(cut_date))
    #
    #
    # get boundary conditions
    if tui:
        status('loading boundary conditions')
    meta = input.asc_raster_meta(fbasin)
    area = np.sum(basincount) * meta['cellsize'] * meta['cellsize']
    qt0 = 0.01  # fixed
    if qobs:
        qt0 = series_df['Q'].values[0]
    lamb = extract_twi_avg(twibins, basincount)
    #
    end = time.time()
    if tui:
        print('\nloading enlapsed time: {:.3f} seconds'.format(end - init))
    #
    #
    #
    #
    #
    #
    # ************* CALIBRATION *************
    #
    if tui:
        init = time.time()
        print('\n\t**** Calibration Protocol ****\n')
        status('running calibration')
    pset, traced, tracedpop = topmodel_calib(series=calib_df, shruparam=shru_df, twibins=twibins, countmatrix=count,
                                             lamb=lamb, qt0=qt0, lat=lat, area=area, basinshadow=basincount,
                                             m_range=m_rng, qo_range=qo_rng,
                                             cpmax_range=cpmax_rng, sfmax_range=sfmax_rng, erz_range=erz_rng,
                                             ksat_range=ksat_rng, c_range=c_rng, k_range=ksat_rng, n_range=n_rng,
                                             etpatdates=etpat_dates_str_calib, etpatzmaps=etpat_zmaps_obs_calib,
                                             tui=tui, generations=generations, popsize=popsize, metric=metric,
                                             tracefrac=1, tracepop=True)
    end = time.time()
    if tui:
        print('\nCalibration enlapsed time: {:.3f} seconds'.format(end - init))
    #
    #
    #
    #
    #
    #
    # ********* EXPORT GENERATIONS *********
    #
    if tui:
        init = time.time()
        print('\n\t**** Export Generations Data Protocol ****\n')
        status('exporting generations dataframes')
    # export generations
    lcl_folder = folder + '/generations'
    mkdir(lcl_folder)  # make diretory
    lcl_folder1 = lcl_folder + '/parents'
    mkdir(lcl_folder1)  # make diretory
    lcl_folder2 = lcl_folder + '/population'
    mkdir(lcl_folder2)  # make diretory
    #
    # export parents
    lcl_files = list()
    lcl_gen_ids = list()
    for g in range(len(traced)):
        stamp = stamped(g=g)
        lcl_filepath = lcl_folder1 + '/gen_' + stamp + '.txt'
        traced[g].to_csv(lcl_filepath, sep=';', index=False)
        lcl_gen_ids.append(g)
        lcl_files.append(lcl_filepath)
    traced_df = pd.DataFrame({'Gen': lcl_gen_ids, 'File': lcl_files})
    exp_file5 = lcl_folder1 + '/' + 'generations_parents.txt'
    traced_df.to_csv(exp_file5, sep=';', index=False)
    #
    # export full population
    lcl_files = list()
    lcl_gen_ids = list()
    for g in range(len(tracedpop)):
        stamp = stamped(g=g)
        lcl_filepath = lcl_folder2 + '/gen_' + stamp + '.txt'
        tracedpop[g].to_csv(lcl_filepath, sep=';', index=False)
        lcl_gen_ids.append(g)
        lcl_files.append(lcl_filepath)
    traced_df = pd.DataFrame({'Gen': lcl_gen_ids, 'File': lcl_files})
    exp_file6 = lcl_folder2 + '/' + 'generations_population.txt'
    traced_df.to_csv(exp_file6, sep=';', index=False)
    #
    #
    #
    #
    # BEST SET simulation on the calibration period
    #
    # create best set folder
    bestset_folder = folder + '/' + 'bestset'
    mkdir(bestset_folder)
    #
    #
    # best set parameters export
    if tui:
        print(' >>> exporting best set run parameters...')
    exp_file3 = bestset_folder + '/' + 'bestset_parameters.txt'
    hydroparam_df['Set'] = [pset[0], pset[1], pset[2], pset[3], pset[4], pset[5], pset[6], lat, pset[7], pset[8]]
    hydroparam_df.to_csv(exp_file3, sep=';', index=False)
    #
    # run SLH for calibration basin, asking for mapping the ET:
    slh_dct = slh_calib(fseries=fseries, fhydroparam=fhydroparam, fshruparam=fshruparam,
                        fhistograms=fhistograms, fbasinhists=fbasinhists, fbasin=fbasin, ftwi=ftwi, fshru=fshru,
                        mapback=True, mapraster=True, mapvar='ET', mapdates=etpat_dates_str_full,
                        qobs=True, tui=tui, folder=bestset_folder)
    # extract folders:
    calib_folder = slh_dct['CalibFolder']
    valid_folder = slh_dct['ValidFolder']
    full_folder = slh_dct['ValidFolder']
    #
    # old code for simulation
    '''    #
    # run simulation of calib period
    if tui:
        init = time.time()
        print('\n\t**** Best Set Simulation Protocol ****\n')
        status('running simulation of calibration period')
    fseries_calib = bestset_folder_calib + '/' + 'calib_series.txt'
    calib_df.to_csv(fseries_calib, sep=';', index=False)
    bestset_files_calib = slh(fseries=fseries_calib, fhydroparam=exp_file3, fshruparam=fshruparam,
                              fhistograms=fhistograms, fbasinhists=fbasinhists, fbasin=fbasin, ftwi=ftwi,
                              fshru=fshru, folder=bestset_folder_calib,
                              mapback=True, mapvar='ET', mapraster=True,
                              mapdates=etpat_dates_str_calib, qobs=True, tui=tui)
    # run analyst calib period
    bestset_analyst_calib = osa(fseries=bestset_files_calib['Series'], fld_obs='Qobs', fld_sim='Q',
                                folder=bestset_folder_calib, tui=True)
    #
    # run full period simulation
    if tui:
        status('running simulation of full period')
    bestset_files_full = slh(fseries=fseries, fhydroparam=exp_file3, fshruparam=fshruparam,
                             fhistograms=fhistograms, fbasinhists=fbasinhists,
                             fbasin=fbasin, ftwi=ftwi, fshru=fshru, folder=bestset_folder_full,
                             mapback=True, mapvar='ET', mapraster=True,
                             mapdates=etpat_dates_str_full, qobs=True, tui=tui)
    # run analyst full period
    bestset_analyst_full = osa(fseries=bestset_files_full['Series'], fld_obs='Qobs', fld_sim='Q',
                               folder=bestset_folder_full, tui=True)
    #
    # now extract just the validation period from the full period series
    if tui:
        status('analysis of validation period')
    full_df = pd.read_csv(bestset_files_full['Series'], sep=';', parse_dates=['Date'])
    valid_df = full_df.query('Date >= "{}"'.format(cut_date))
    fseries_valid = bestset_folder_valid + '/' + 'valid_series.txt'
    valid_df.to_csv(fseries_valid, sep=';', index=False)
    #
    # run analyst in valid period
    bestset_analyst_valid = osa(fseries=fseries_valid, fld_obs='Qobs', fld_sim='Q',
                                folder=bestset_folder_valid, tui=tui)
    if tui:
        end = time.time()
        print('\nsimulation enlapsed time: {:.3f} seconds'.format(end - init))
    #
'''    #
    #
    #
    # BEST SET ASSESMENT OF ET PAT DATA
    #
    # CALIBRATION period
    #
    # RASTERS | first compute the simulated etpat rasters
    #
    # create a dir
    lcl_folder = calib_folder + '/' + 'sim_ETPat'
    mkdir(lcl_folder)
    #
    # find the simulated mapseries file for period
    fet_raster_sim = calib_folder + '/sim_raster_ET.txt'
    #
    # compute the ETPat rasters
    if tui:
        status('computing SIM ETPat rasters of calibration period')
    fetpat_raster_sim_calib = import_etpat_series(finputseries=fet_raster_sim, rasterfolder=lcl_folder,
                                                  rasterfilename='raster_ETPat', tui=tui,
                                                  folder=calib_folder, filename='sim_raster_ETPat')
    # export raster views
    get_views_rasters(fetpat_raster_sim_calib, mapvar='ETPat', mapid='etpat', vmin=0, vmax=1, tui=tui)
    #
    # create a OBS-SIM dataframe for analysis
    etpat_raster_sim_calib_df = pd.read_csv(fetpat_raster_sim_calib, sep=';')
    obssim_etpat_raster_df = pd.DataFrame({'Date': etpat_calib_dates['Date'],
                                           'File_obs': etpat_raster_obs_calib_df['File'],
                                           'File_sim': etpat_raster_sim_calib_df['File']})
    exp_file8 = calib_folder + '/' + 'raster_etpat_obssim_series.txt'
    obssim_etpat_raster_df.to_csv(exp_file8, sep=';', index=False)
    # analyste rasters
    osa_map(fseries=exp_file8, fhistograms=fhistograms, type='raster', filename='raster_analyst', folder=lcl_folder,
            tui=tui)

    # standby code
    '''
    #
    # BEST SET ASSESMENT OF ET PAT DATA
    #
    # CALIBRATION period
    #
    # RASTERS | first compute the simulated etpat rasters
    #
    # create a dir
    lcl_folder = bestset_folder_calib + '/' + 'sim_ETPat_raster'
    mkdir(lcl_folder)
    #
    # find the simulated mapseries file for period
    fet_raster_sim = bestset_files_calib['Raster']['ET']
    #
    # compute the ETPat rasters
    if tui:
        status('computing SIM ETPat rasters of calibration period')
    fetpat_raster_sim_calib = import_etpat_series(finputseries=fet_raster_sim, rasterfolder=lcl_folder,
                                                  rasterfilename='raster_ETPat', tui=tui,
                                                  folder=bestset_folder_calib, filename='sim_raster_ETPat')
    # export raster views
    get_views_rasters(fetpat_raster_sim_calib, mapvar='ETPat', mapid='etpat', vmin=0, vmax=1, tui=tui)
    #
    # create a OBS-SIM dataframe for analysis
    etpat_raster_sim_calib_df = pd.read_csv(fetpat_raster_sim_calib, sep=';')
    obssim_etpat_raster_df = pd.DataFrame({'Date':etpat_calib_dates['Date'], 'File_obs':etpat_raster_obs_calib_df['File'],
                                           'File_sim':etpat_raster_sim_calib_df['File']})
    exp_file8 = bestset_folder_calib + '/' + 'raster_etpat_obssim_series.txt'
    obssim_etpat_raster_df.to_csv(exp_file8, sep=';', index=False)
    # analyste rasters
    osa_map(fseries=exp_file8, fhistograms=fhistograms, type='raster',
            filename='raster_analyst', folder=lcl_folder, tui=tui)
    #
    #
    # ZMAP | first compute the simulated etpat rasters
    #
    # create a dir
    lcl_folder = bestset_folder_calib + '/' + 'sim_ETPat_zmaps'
    mkdir(lcl_folder)
    #
    if tui:
        status('computing SIM ETPat ZMaps of calibration period')
    fetpat_zmaps_sim_calib = compute_zmap_series(fetpat_raster_sim_calib, ftwi, fshru, fhistograms, var='ETPat',
                                                 filename='ETPat_zmaps_series', tui=tui,
                                                 folder=lcl_folder, folderseries=bestset_folder_calib)
    #
    # create a OBS-SIM dataframe
    if tui:
        status('OBS-SIM analysis of ETPat ZMaps')
    etpat_zmaps_sim_calib_df = pd.read_csv(fetpat_zmaps_sim_calib, sep=';')
    obssim_etpat_zmaps_df = pd.DataFrame({'Date': etpat_calib_dates['Date'], 'File_obs': etpat_zmaps_obs_calib_df['File'],
                                           'File_sim': etpat_zmaps_sim_calib_df['File']})
    exp_file7 = bestset_folder_calib + '/' + 'zmaps_etpat_obssim_series.txt'
    obssim_etpat_zmaps_df.to_csv(exp_file7, sep=';', index=False)
    #
    osa_map(fseries=exp_file7, fhistograms=fhistograms, type='zmap',
            filename='zmap_analyst', folder=lcl_folder, tui=tui)
    #
    
    
    
    
    '''
    #
    # return dictionary
    return {'Folder':folder}










