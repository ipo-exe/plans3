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


def compute_zmap_series(fvarseries, ftwi, fshru, fhistograms, filename='var_zmap_series', folder='C:/bin', tui=False):
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
        lcl_new_filename = lcl_filenm.split('.')[0] + '_zmap'
        #
        meta, lcl_var = input.asc_raster(files[i])
        lcl_zmap = built_zmap(varmap=lcl_var, twi=twi, shru=shru, twibins=twibins, shrubins=shrubins)
        exp_file = output.zmap(zmap=lcl_zmap, twibins=twibins, shrubins=shrubins, folder=lcl_folder, filename=lcl_new_filename)
        new_files.append(exp_file)
    #
    # export data
    exp_df = pd.DataFrame({'Date': dates, 'File': new_files})
    exp_file = folder + '/' + filename + '.txt'
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


def obs_sim_analyst(fseries, fld_obs='Qobs', fld_sim='Q', fld_date='Date', folder='C:/bin', tui=False):
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
        print('performing obs vs. sim analysis...')
    #
    # extract Dataframe
    def_df = pd.read_csv(fseries, sep=';', engine='python')
    def_df = dataframe_prepro(def_df, strf=False, date=True, datefield=fld_date)
    #
    # extract obs and sim arrays:
    obs = def_df[fld_obs].values
    #sim = obs - 0.1 * obs
    sim = def_df[fld_sim].values
    obslog = np.log10(obs)
    simlog = np.log10(sim)
    #
    #
    # **** Series Analysis ****
    # Error series analyst
    e = analyst.error(obs=obs, sim=sim)
    se = analyst.sq_error(obs=obs, sim=sim)
    elog = analyst.error(obs=obslog, sim=simlog)
    selog = analyst.sq_error(obs=obslog, sim=simlog)
    # built Dataframe
    series_df = pd.DataFrame({'Date':def_df[fld_date], 'Obs':obs, 'Sim':sim, 'Obslog':obslog, 'Simlog':simlog, 'E':e,
                              'Elog':elog, 'SE':se, 'SElog':selog})
    # coefs analyst of series
    pbias = analyst.pbias(obs=obs, sim=sim)
    rmse = analyst.rmse(obs=obs, sim=sim)
    rmselog = analyst.rmse(obs=obslog, sim=simlog)
    nse = analyst.nse(obs=obs, sim=sim)
    nselog = analyst.nse(obs=obslog, sim=simlog)
    linreg = analyst.linreg(obs=obs, sim=sim)
    kge = analyst.kge(obs=obs, sim=sim)
    kgelog = analyst.kge(obs=obslog, sim=simlog)
    #
    #
    # **** Frequency analysis ****
    freq_obs = analyst.frequency(series=obs)
    freq_sim = analyst.frequency(series=sim)
    obs_freq = freq_obs['Values']
    sim_freq = freq_sim['Values']
    obslog_freq = np.log10(obs_freq)
    simlog_freq = np.log10(sim_freq)
    #
    # Error frequency analyst
    e_freq = analyst.error(obs=obs_freq, sim=sim_freq)
    se_freq = analyst.sq_error(obs=obs_freq, sim=sim_freq)
    elog_freq = analyst.error(obs=obslog_freq, sim=simlog_freq)
    selog_freq = analyst.sq_error(obs=obslog_freq, sim=simlog_freq)
    #
    # built dataframe
    freq_df = pd.DataFrame({'Percentiles': freq_obs['Percentiles'], 'Exeedance': freq_obs['Exeedance'],
                            'ProbabObs': freq_obs['Probability'], 'ValuesObs': freq_obs['Values'],
                            'ValuesObslog': obslog_freq, 'ProbabSim': freq_sim['Probability'],
                            'ValuesSim': freq_sim['Values'], 'ValuesSimlog': simlog_freq,
                            'E':e_freq, 'Elog':elog_freq,  'SE':se_freq, 'SElog':selog_freq
                            })
    #
    # coefs analyst of series
    rmse_freq = analyst.rmse(obs=obs_freq, sim=sim_freq)
    rmselog_freq = analyst.rmse(obs=obslog_freq, sim=simlog_freq)
    linreg_freq = analyst.linreg(obs=obs_freq, sim=sim_freq)
    linreg_freq_log = analyst.linreg(obs=obslog_freq, sim=simlog_freq)
    #
    # built dataframe of parameters
    params = ('PBias', 'RMSE', 'RMSElog', 'NSE', 'NSElog', 'KGE', 'KGElog', 'A', 'B', 'R', 'P', 'SD',
              'RMSE-CFC', 'RMSElog-CFC', 'R-CFC', 'Rlog-CFC')
    values = (pbias, rmse, rmselog, nse, nselog, kge, kgelog, linreg['A'], linreg['B'], linreg['R'], linreg['P'],
              linreg['SD'], rmse_freq, rmselog_freq, linreg_freq['R'], linreg_freq_log['R'])
    param_df = pd.DataFrame({'Parameter': params, 'Value': values})
    #
    report_lst.append('\n\nAnalyst Parameter Results:\n\n')
    report_lst.append(param_df.to_string(index=False))
    report_lst.append('\n\n')
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
    exp_file4 = pannel_obs_sim_analyst(series=series_df, freq=freq_df, params=param_df, folder=folder)
    #
    #
    if tui:
        print('\n\n\n')
    tf = time.time()
    output_df = pd.DataFrame({'Output files': (exp_file1, exp_file2, exp_file3, exp_file4)})
    report_lst.append(output_df.to_string(index=False))
    report_lst.append('\n\n')
    report_lst.insert(2, 'Execution enlapsed time: {:.3f} seconds\n'.format(tf - t0))
    export_report(report_lst, filename='REPORT__analyst', folder=folder, tui=tui)
    #
    return (exp_file1, exp_file2, exp_file3, exp_file4)


def stable_lulc_hydro(fseries, fhydroparam, fshruparam, fhistograms, fbasinhists, fbasin, ftwi, fshru, mapback=False,
                      mapraster=False, mapvar='all', mapdates='all', qobs=False, folder='C:/bin', tui=False,):
    # todo 1) docstring
    import time, datetime
    from shutil import copyfile
    from hydrology import topmodel_sim, map_back
    from visuals import pannel_topmodel

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
    # simulation section
    init = time.time()
    if tui:
        print('\n\t**** Simulation Protocol ****\n')
        status('running simulation')
    if mapback:
        sim_df, mapped = topmodel_sim(series=series_df, shruparam=shru_df, twibins=twibins, countmatrix=count, lamb=lamb,
                                      qt0=qt0, m=m, qo=qo, cpmax=cpmax, sfmax=sfmax, erz=erz, ksat=ksat, c=c, lat=lat,
                                      k=k, n=n, area=area, basinshadow=basincount, tui=False, qobs=qobs, mapback=mapback, mapvar=mapvar,
                                      mapdates=mapdates)
    else:
        sim_df = topmodel_sim(series=series_df, shruparam=shru_df, twibins=twibins, countmatrix=count, lamb=lamb,
                              qt0=qt0, m=m, qo=qo, cpmax=cpmax, sfmax=sfmax, erz=erz, ksat=ksat, c=c, lat=lat,
                              k=k, n=n, area=area, basinshadow=basincount, tui=False, qobs=qobs, mapback=mapback, mapvar=mapvar,
                              mapdates=mapdates)
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
    # time series
    exp_file1 = folder + '/' + 'sim_series.txt'
    sim_df.to_csv(exp_file1, sep=';', index=False)
    # histograms
    if tui:
        status('exporting histograms')
    exp_file2 = folder + '/' + 'sim_histograms.txt'
    copyfile(src=fhistograms, dst=exp_file2)
    #
    # parameters
    if tui:
        status('exporting run parameters')
    exp_file3 = folder + '/' + 'sim_parameters.txt'
    copyfile(src=fhydroparam, dst=exp_file3)
    ## hydroparam_df.to_csv(exp_file3, sep=';', index=False)
    #
    # export visual pannel
    if tui:
        status('exporting visual results')
    exp_file4 = pannel_topmodel(sim_df, grid=False, show=False, qobs=qobs, folder=folder)
    #
    # maps
    if mapback:
        if tui:
            status('exporting variable maps')
        from os import mkdir
        if mapvar == 'all':
            mapvar = 'P-Temp-IRA-IRI-PET-D-Cpy-TF-Sfs-R-RSE-RIE-RC-Inf-Unz-Qv-Evc-Evs-Tpun-Tpgw-ET-VSA'
        mapvar_lst = mapvar.split('-')  # load string variables alias to list
        # get stamps
        if mapdates == 'all':
            stamp = pd.to_datetime(sim_df['Date'], format='%y-%m-%d')  # get stamp
        else:
            mapdates_df = pd.DataFrame({'Date': mapdates.split('&')})
            mapdates_df['Date'] = mapdates_df['Date'].str.strip()
            stamp = mapdates_df['Date'].values
        #
        # Zmaps exporting
        mapfiles_lst = list()
        zmaps_dct = dict()
        for var in mapvar_lst:  # loop across all variables
            if tui:
                status('exporting {} zmaps'.format(var))
            lcl_folder = folder + '/sim_zmaps_' + var
            mkdir(lcl_folder)  # make new diretory
            lcl_files = list()
            for t in range(len(stamp)):  # loop across all timesteps
                lcl_filename = 'zmap_' + var + '_' + str(stamp[t]).split(sep=' ')[0] + '.txt'
                lcl_file = lcl_folder + '/' + lcl_filename
                lcl_files.append(lcl_file)
                # export local dataframe to text file in local folder
                lcl_exp_df = pd.DataFrame(mapped[var][t], index=twibins, columns=shrubins)
                lcl_exp_df.to_csv(lcl_file, sep=';', index_label='TWI\SHRU')
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
                lcl_folder = folder + '/sim_raster_' + var
                mkdir(lcl_folder)  # make new diretory
                lcl_files = list()
                for t in range(len(stamp)):  # loop across all timesteps
                    stamp_str = str(stamp[t]).split(sep=' ')[0]
                    lcl_filename = 'raster_' + var + '_' + stamp_str
                    lcl_file = lcl_folder + '/' + lcl_filename
                    lcl_files.append(lcl_file)
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
    out_dct = {'Series':exp_file1, 'Histograms':exp_file2, 'Parameters':exp_file3, 'Pannel':exp_file4}
    if mapback:
        out_dct['ZMaps'] = zmaps_dct
        if mapraster:
            out_dct['Raster'] = raster_dct
    return out_dct



def calib_hydro(fseries, fhydroparam, fshruparam, fhistograms, fbasinhists, fbasin, ftwi, fshru, fetpatzmaps,
                fetpatseries, folder='C:/bin', tui=False, mapback=False, mapvar='all', mapdates='all', qobs=True,
                generations=100, popsize=200, metric='NSE'):
    # todo docstring
    from hydrology import avg_2d, topmodel_sim, map_back, topmodel_calib
    from visuals import pannel_topmodel
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
    #
    t0 = time.time()
    if tui:
        from tui import status
        init = time.time()
        print('\n\t**** Load Data Protocol ****\n')
        status('loading time series')
    series_df =  pd.read_csv(fseries, sep=';')
    series_df = dataframe_prepro(series_df, strf=False, date=True, datefield='Date')
    calib_df, cut_date = extract_calib_valid(series_df, fvalid=0.33)
    #
    if tui:
        status('loading hydrology parameters') #print(' >>> loading hydrology parameters...')
    hydroparam_df = pd.read_csv(fhydroparam, sep=';')
    hydroparam_df = dataframe_prepro(hydroparam_df, 'Parameter')
    #
    #
    # extract set values
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
        status('loading OBS ETpat Z-maps')
    #
    # Import Observed Zmaps
    etpat_zmaps_obs_df = pd.read_csv(fetpatzmaps, sep=';')
    etpat_zmaps_obs_df = dataframe_prepro(etpat_zmaps_obs_df, strfields='File', date=True)
    # split dataframes for later
    etpat_zmaps_obs_calib_df = etpat_zmaps_obs_df.query('Date < "{}"'.format(cut_date))
    etpat_zmaps_obs_valid_df = etpat_zmaps_obs_df.query('Date >= "{}"'.format(cut_date))
    #
    # extract dates for calibration
    etpat_dates_str = ' & '.join(etpat_zmaps_obs_calib_df['Date'].astype('str').values)  # for calibration!!
    etpat_dates_str_full = ' & '.join(etpat_zmaps_obs_df['Date'].astype('str').values)  # for full
    #
    # load zmaps for calibration
    etpat_zmaps_obs_calib = list()
    for i in range(len(etpat_zmaps_obs_calib_df)):
        zmap_file = etpat_zmaps_obs_calib_df['File'].values[i]
        zmap, ybins, xbins = input.zmap(zmap_file)
        etpat_zmaps_obs_calib.append(zmap)
    #
    # load zmaps for validation
    etpat_zmaps_obs_valid = list()
    for i in range(len(etpat_zmaps_obs_valid_df)):
        zmap_file = etpat_zmaps_obs_valid_df['File'].values[i]
        zmap, ybins, xbins = input.zmap(zmap_file)
        etpat_zmaps_obs_valid.append(zmap)
    #
    # extract max value for ETpat
    #etpat_min_calib = np.min(np.array(etpat_zmaps_obs_calib))
    #etpat_min_valid = np.min(np.array(etpat_zmaps_obs_valid))
    #etpat_min = np.min((etpat_min_calib, etpat_min_valid))
    #
    etpat_max_calib = np.max(np.array(etpat_zmaps_obs_calib))
    etpat_max_valid = np.max(np.array(etpat_zmaps_obs_valid))
    #etpat_max = np.max((etpat_min_calib, etpat_min_valid))
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
    #
    end = time.time()
    if tui:
        print('\nloading enlapsed time: {:.3f} seconds'.format(end - init))
    #
    #
    #
    #
    # run calibration
    if tui:
        init = time.time()
        print('\n\t**** Calibration Protocol ****\n')
        status('running calibration')
    pset, traced, tracedpop = topmodel_calib(series=calib_df, shruparam=shru_df, twibins=twibins, countmatrix=count,
                                             lamb=lamb, qt0=qt0, lat=lat, area=area, basinshadow=basincount,
                                             m_range=m_rng, qo_range=qo_rng,
                                             cpmax_range=cpmax_rng, sfmax_range=sfmax_rng, erz_range=erz_rng,
                                             ksat_range=ksat_rng, c_range=c_rng, k_range=ksat_rng, n_range=n_rng,
                                             etpatdates=etpat_dates_str, etpatzmaps=etpat_zmaps_obs_calib,
                                             tui=tui, generations=generations, popsize=popsize, metric=metric,
                                             tracefrac=1, tracepop=True)
    end = time.time()
    if tui:
        print('\nCalibration enlapsed time: {:.3f} seconds'.format(end - init))
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
    bestset_folder_calib = bestset_folder + '/' + 'calibration_period'
    mkdir(bestset_folder_calib)
    bestset_folder_full = bestset_folder + '/' + 'full_period'
    mkdir(bestset_folder_full)
    bestset_folder_valid = bestset_folder + '/' + 'valid_period'
    mkdir(bestset_folder_valid)
    #
    #
    # parameters export
    if tui:
        print(' >>> exporting best set run parameters...')
    exp_file3 = bestset_folder + '/' + 'bestset_parameters.txt'
    hydroparam_df['Set'] = [pset[0], pset[1], pset[2], pset[3], pset[4], pset[5], pset[6], lat, pset[7], pset[8]]
    hydroparam_df.to_csv(exp_file3, sep=';', index=False)
    #
    #
    # run simulation of calib period
    if tui:
        init = time.time()
        print('\n\t**** Best Set Simulation Protocol ****\n')
        status('running simulation of calibration period')
    fseries_calib = bestset_folder_calib + '/' + 'calib_series.txt'
    calib_df.to_csv(fseries_calib, sep=';', index=False)
    bestset_files_calib = stable_lulc_hydro(fseries=fseries_calib, fhydroparam=exp_file3, fshruparam=fshruparam,
                                            fhistograms=fhistograms, fbasinhists=fbasinhists, fbasin=fbasin, ftwi=ftwi,
                                            fshru=fshru, folder=bestset_folder_calib,
                                            mapback=True, mapvar='ET', mapraster=True,
                                            mapdates=etpat_dates_str, qobs=True, tui=tui)
    # run analyst calib period
    bestset_analyst_calib = obs_sim_analyst(fseries=bestset_files_calib['Series'], fld_obs='Qobs', fld_sim='Q',
                                            folder=bestset_folder_calib, tui=True)
    #
    #
    # run full period simulation
    if tui:
        status('running simulation of full period')
    bestset_files_full = stable_lulc_hydro(fseries=fseries, fhydroparam=exp_file3, fshruparam=fshruparam,
                                           fhistograms=fhistograms, fbasinhists=fbasinhists,
                                           fbasin=fbasin, ftwi=ftwi, fshru=fshru, folder=bestset_folder_full,
                                           mapback=True, mapvar='ET',  mapraster=True,
                                           mapdates=etpat_dates_str_full, qobs=True, tui=tui)
    # run analyst full period
    bestset_analyst_full = obs_sim_analyst(fseries=bestset_files_full['Series'], fld_obs='Qobs', fld_sim='Q',
                                           folder=bestset_folder_full, tui=True)
    #
    # now extract just the validation period from the full period series
    if tui:
        status('analysis of validation period')
    full_df = pd.read_csv(bestset_files_full['Series'], sep=';', parse_dates=['Date'])
    valid_df = full_df.query('Date >= "{}"'.format(cut_date))
    fseries_valid = bestset_folder_valid + '/' + 'valid_series.txt'
    valid_df.to_csv(fseries_valid, sep=';', index=False)
    bestset_analyst_valid = obs_sim_analyst(fseries=fseries_valid, fld_obs='Qobs', fld_sim='Q',
                                            folder=bestset_folder_valid, tui=True)
    if tui:
        end = time.time()
        print('\nsimulation enlapsed time: {:.3f} seconds'.format(end - init))
    #
    #
    #
    # BEST SET ASSESMENT OF ET PAT DATA
    #
    # Import Observed rasters
    etpat_raster_obs_df = pd.read_csv(fetpatseries, sep=';')
    etpat_raster_obs_df = dataframe_prepro(etpat_raster_obs_df, strfields='File', date=True)
    # split dataframes for later
    etpat_raster_obs_calib_df = etpat_raster_obs_df.query('Date < "{}"'.format(cut_date))
    etpat_raster_obs_valid_df = etpat_raster_obs_df.query('Date >= "{}"'.format(cut_date))
    #
    # Import Simulated Zmaps of ET
    fet_zmaps_sim = bestset_files_full['ZMaps']['ET']
    et_zmaps_sim_df = pd.read_csv(fet_zmaps_sim, sep=';')
    et_zmaps_sim_df = dataframe_prepro(et_zmaps_sim_df, strfields='File', date=True)
    # split dataframes for later
    et_zmaps_sim_calib_df = et_zmaps_sim_df.query('Date < "{}"'.format(cut_date))
    et_zmaps_sim_valid_df = et_zmaps_sim_df.query('Date >= "{}"'.format(cut_date))
    #
    # Import Simulared Rasters of ET
    fet_raster_sim = bestset_files_full['Raster']['ET']
    et_raster_sim_df = pd.read_csv(fet_raster_sim, sep=';')
    et_raster_sim_df = dataframe_prepro(et_raster_sim_df, strfields='File', date=True)
    # split dataframes for later
    et_raster_sim_calib_df = et_raster_sim_df.query('Date < "{}"'.format(cut_date))
    et_raster_sim_valid_df = et_raster_sim_df.query('Date >= "{}"'.format(cut_date))
    #
    # Extract the max value of ET for calib and valid periods
    et_zmaps_sim_calib = list()
    for i in range(len(et_zmaps_sim_calib_df)):
        zmap_file = et_zmaps_sim_calib_df['File'].values[i]
        zmap, ybins, xbins = input.zmap(zmap_file)
        et_zmaps_sim_calib.append(zmap)
    et_zmaps_sim_valid = list()
    for i in range(len(et_zmaps_sim_valid_df)):
        zmap_file = et_zmaps_sim_valid_df['File'].values[i]
        zmap, ybins, xbins = input.zmap(zmap_file)
        et_zmaps_sim_valid.append(zmap)
    et_max_calib = np.max(np.array(et_zmaps_sim_calib))
    et_max_valid = np.max(np.array(et_zmaps_sim_valid))
    #
    # Calibration period analysis
    for t in range(len(etpat_zmaps_obs_calib_df)):
        lcl_date = etpat_zmaps_obs_calib_df['Date'].values[t]
        #
        # 1) import local zmap for obs (ETpat) and sim (ET)
        lcl_zmap_obs = etpat_zmaps_obs_calib[t]
        lcl_zmap_sim, ybins, xbins = input.zmap(et_zmaps_sim_calib_df['File'].values[t])
        # normalize zmaps:
        lcl_zmap_obs = lcl_zmap_obs / etpat_max_calib # etpat_max
        lcl_zmap_sim = lcl_zmap_sim / et_max_calib # et max
        #
        # 2) extract the local zmap signal for obs (ETpat) and sim (ET)
        lcl_zmap_sim_signal = lcl_zmap_sim.flatten()
        lcl_zmap_obs_signal = lcl_zmap_obs.flatten()
        x_signals = np.arange(0, len(lcl_zmap_obs_signal))
        plt.scatter(lcl_zmap_obs_signal, lcl_zmap_sim_signal, alpha=0.5)
        #plt.plot(x_signals, lcl_zmap_sim_signal)
        #plt.plot(x_signals, lcl_zmap_obs_signal)
        plt.show()
        # 4) import local raster map for obs and sim
        # 5) compute metrics for zmap signal
        # 6) compute metrics for zmap
        # 7) compute metrics for raster



    return exp_file3

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









