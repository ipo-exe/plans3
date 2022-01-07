import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tui import status
import inp, out


def step00_set_input_series(folder):
    import resample
    from hydrology import convert_q2sq, convert_sq2q
    import inp
    #
    # combine the available series
    status('combining avaliable series')
    in_folder = r'C:/000_myFiles/myDrive/myProjects/104_paper_castelhano/insumos'
    files = ['ANA-flow_85830000_1978-2021__by-2021-12-16.txt',
             'ANA-prec_02952035_2004-2020__by-2020-12-29.txt',
             'ANA-prec_02952036_2004-2020__by-2020-12-29.txt',
             'ANA-prec_02952037_2004-2021__by-2021-12-16.txt']
    names = ['Sta Cruz', 'Herveiras', 'Boqueirao', 'Deodoro']
    fields = ['Flow', 'Prec', 'Prec', 'Prec']
    #
    # read and resample
    df_lst = dict()
    for i in range(0, len(files)):
        _path = in_folder + '/{}'.format(files[i])
        _df = pd.read_csv(_path, sep=';', parse_dates=['Date'])
        _df = _df.query('Date >= "2011-01-01" and Date < "2015-01-01"')
        _gaps = np.sum(_df[fields[i]].isna())
        if _gaps > 0:
            _df = resample.interpolate_gaps(_df, fields[i], size=3, type='zero')
            _df = _df[['Date', 'Interpolation']]
            _df = _df.rename(columns={'Interpolation': fields[i]})
        df_lst[names[i]] = _df.copy()
    #
    # merge series
    for i in range(len(names)):
        _field = fields[i] + '_' + names[i]
        if i == 0:
            _odf = df_lst[names[i]]
            _odf = _odf.rename(columns={fields[i]: _field})
        else:
            _df = df_lst[names[i]]
            _df = _df.rename(columns={fields[i]: _field})
            _odf = pd.merge(_odf, _df, left_on='Date', right_on='Date')
    ##print(_odf.head().to_string())
    fraster = r"C:\000_myFiles\myDrive\Plans3\pardinho\datasets\observed\calib_basin.asc"
    meta, rmap = inp.asc_raster(file=fraster)
    _area = np.sum(rmap) * meta['cellsize'] * meta['cellsize']
    _odf['Q_StaCruz'] = convert_q2sq(q=_odf['Flow_Sta Cruz'].values, area=_area)
    _of = folder + '/step01a__obs_data_all.txt'
    _odf.to_csv(_of, sep=';', index=False)
    view_obs_data_analyst(fseries=_of, folder=folder, show=False)
    #
    #
    status('defining precipitation sets')
    _df = pd.read_csv(_of, sep=';', parse_dates=['Date'])
    areas_calib = {'Prec_Herveiras': 56788,
                   'Prec_Boqueirao': 26542,
                   'Prec_Deodoro': 2207}
    areas_aoi = {'Prec_Herveiras': 0,
                 'Prec_Boqueirao': 4823,
                 'Prec_Deodoro': 3351}
    # calib:
    area_total = 0
    for k in areas_calib:
        area_total = area_total + areas_calib[k]
    factors = dict()
    for k in areas_calib:
        factors[k] = areas_calib[k] / area_total
        _df[k + '_calibf'] = factors[k]
    #
    # aoi
    area_total = 0
    for k in areas_aoi:
        area_total = area_total + areas_aoi[k]
    factors = dict()
    for k in areas_aoi:
        factors[k] = areas_aoi[k] / area_total
        _df[k + '_aoif'] = factors[k]
    _df['Prec_calib_S1'] = 0
    for k in areas_calib:
        _df['Prec_calib_S1'] = _df['Prec_calib_S1'] + (_df[k] * _df[k + '_calibf'])
    _df['Prec_calib_S2'] = (_df['Prec_Boqueirao'] * (_df['Prec_Herveiras_calibf'] + _df['Prec_Boqueirao_calibf'])) + (
            _df['Prec_Deodoro'] * _df['Prec_Deodoro_calibf'])
    _df['Prec_calib_S3'] = _df['Prec_calib_S2']
    _df['Prec_AOI'] = 0
    for k in areas_calib:
        _df['Prec_AOI'] = _df['Prec_AOI'] + (_df[k] * _df[k + '_aoif'])
    _of = folder + '/step01b__obs_data_precsets.txt'
    _df.to_csv(_of, sep=';', index=False)
    #
    status('setting input series')
    #
    # load other inputs
    fcalib = r"C:\000_myFiles\myDrive\Plans3\pardinho\datasets\observed\calib_series.txt"
    calib_df = pd.read_csv(fcalib, sep=';', parse_dates=['Date'])
    calib_s1_df = calib_df.copy()
    calib_s2_df = calib_df.copy()
    s3_frame = 'Date >= "2013-01-01"'
    calib_s3_df = calib_df.query(s3_frame).copy()
    faoi = r"C:\000_myFiles\myDrive\Plans3\pardinho\datasets\observed\aoi_series.txt"
    aoi_df = pd.read_csv(faoi, sep=';', parse_dates=['Date'])
    #
    fprec_sets = _of
    sets_df = pd.read_csv(fprec_sets, sep=';', parse_dates=['Date'])
    sets_df_s3 = sets_df.query(s3_frame).copy()
    calib_s1_df['Prec'] = sets_df['Prec_calib_S1']
    calib_s2_df['Prec'] = sets_df['Prec_calib_S2']
    calib_s3_df['Prec'] = sets_df_s3['Prec_calib_S3']
    aoi_df['Prec'] = sets_df['Prec_AOI']
    _dfs = [calib_s1_df, calib_s2_df, calib_s3_df, aoi_df]
    _odct = {'s1': calib_s1_df, 's2': calib_s2_df, 's3': calib_s3_df, 'aoi': aoi_df}
    _names = ['calib_s1_series.txt', 'calib_s2_series.txt', 'calib_s3_series.txt', 'aoi_series.txt']
    # output
    for i in range(len(_dfs)):
        _of = folder + '/' + _names[i]
        _dfs[i].to_csv(_of, sep=';', index=False)
    return _odct


def step01_search_models(folder, projectfolder):
    import backend
    from tools import calibrate
    status('searching models')
    # define the output workplace folder
    outfolder = folder
    # get folder of observed datasets
    # get observed datasets standard names
    files_input = backend.get_input2calibhydro()
    folder = projectfolder + '/datasets/observed'
    fseries = folder + '/' + files_input[0]
    fhydroparam = folder + '/' + files_input[1]
    fshruparam = folder + '/' + files_input[2]
    fhistograms = folder + '/' + files_input[3]
    fbasinhists = folder + '/' + files_input[4]
    fbasin = folder + '/' + files_input[5]
    ftwi = folder + '/' + files_input[11]  # + files_input[6]
    fshru = folder + '/' + files_input[10]  # + files_input[7]
    fetpatzmaps = folder + '/' + files_input[8]
    fcanopy = folder + '/' + files_input[9]
    # Options: 'NSE', 'NSElog', 'RMSE', 'RMSElog', 'KGE', 'KGElog', 'PBias', 'RMSE-CFC', 'RMSElog-CFC'
    likelihood = 'KGElog'
    generations = 10
    popsize = 100
    calibfiles = calibrate(fseries=fseries,
                           fhydroparam=fhydroparam,
                           fshruparam=fshruparam,
                           fhistograms=fhistograms,
                           fbasinhists=fbasinhists,
                           fbasin=fbasin,
                           fetpatzmaps=fetpatzmaps,
                           ftwi=ftwi,
                           fshru=fshru,
                           fcanopy=fcanopy,
                           folder=outfolder,
                           label='calib',
                           generations=generations,
                           popsize=popsize,
                           likelihood=likelihood,
                           tui=True,
                           normalize=False,
                           etpat=True,
                           cutdatef=0.2,
                           tail=True,
                           )
    return calibfiles['Folder']


def step02_select_models(folder, calibfolder, projectfolder):
    from backend import get_input2calibhydro
    from tools import glue
    #
    outfolder = folder
    # get folder of observed datasets
    folder = projectfolder + '/datasets/observed'
    # get observed datasets standard names
    files_input = get_input2calibhydro()
    fshruparam = folder + '/' + files_input[2]
    fhistograms = folder + '/' + files_input[3]
    fbasinhists = folder + '/' + files_input[4]
    fbasin = folder + '/' + files_input[5]
    ftwi = folder + '/' + files_input[6]
    fshru = folder + '/' + files_input[7]
    fcanopy = folder + '/' + files_input[9]
    #
    # calibration folder
    calib_folder = calibfolder
    fseries = calib_folder + '/MLM/full_period/sim_series.txt'
    fhydroparam = calib_folder + '/MLM/mlm_parameters.txt'
    fmodels = calib_folder + '/generations/population.txt'
    gluefiles = glue(fseries=fseries,
                     fmodels=fmodels,
                     fhydroparam=fhydroparam,
                     fhistograms=fhistograms,
                     fbasinhists=fbasinhists,
                     fshruparam=fshruparam,
                     fbasin=fbasin,
                     fcanopy=fcanopy,
                     likelihood='L',
                     nmodels=5000,
                     behavioural=-2,
                     run_ensemble=True,
                     folder=outfolder,
                     wkpl=True,
                     normalize=False,
                     tui=True)
    # filter
    status('selecting models')
    fmodels = calib_folder + '/generations/population.txt'
    models_df = pd.read_csv(fmodels, sep=';')
    models_df = models_df.drop_duplicates(subset=['SetIds'])
    #models_df['PBias_abs'] = models_df['PBias'].abs()

    # activate this:
    #models_df = models_df.query('L > -0.65 and KGElog > 0.5 and NSElog > 0 and Qb_R > 0.25')


    #models_df = models_df.sort_values(by='PBias_abs', ascending=True)
    #print(models_df.head(20).to_string())
    #print(len(models_df))
    selected_models = gluefiles['Folder'] + '/selection.txt'
    models_df.to_csv(selected_models, sep=';', index=False)
    #
    #
    view_evolution_1(folder=outfolder, calibfolder=calib_folder, gluefolder=gluefiles['Folder'])
    view_evolution_2(folder=outfolder, calibfolder=calib_folder)
    #
    #
    return gluefiles['Folder']


def step03_map_processes(folder, gluefolder, projectfolder, vars='Qv-R'):
    from tools import bat_slh
    ftwi = projectfolder + "/datasets/observed/aoi_twi.asc"
    fseries = projectfolder + "/datasets/observed/calib_series.txt"
    fshru_param = projectfolder + "/datasets/observed/calib_shru_param.txt"
    fbasin = projectfolder + "/datasets/observed/aoi_basin.asc"
    fcanopy = projectfolder +  "/datasets/observed/calib_canopy_series.txt"
    fhydrop = projectfolder +  "/datasets/observed/hydro_param.txt"
    fmodels = gluefolder + '/selection.txt'
    #
    mapvars = vars
    #
    # pre
    fshru = projectfolder + "/datasets/projected/scn__lulc_predc/aoi_shru.asc"
    fhists = projectfolder + "/datasets/projected/scn__lulc_predc/aoi_histograms.txt"
    fbasin_hists = projectfolder + "/datasets/projected/scn__lulc_predc/aoi_basin_histograms.txt"
    pre_folder = bat_slh(fmodels=fmodels,
                            fseries=fseries,
                            fhydroparam=fhydrop,
                            fshruparam=fshru_param,
                            fshru=fshru,
                            ftwi=ftwi,
                            fhistograms=fhists,
                            fbasinhists=fbasin_hists,
                            fbasin=fbasin,
                            fcanopy=fcanopy,
                            model_id='Id',
                            wkpl=True,
                            tui=True,
                            mapback=True,
                            mapvar=mapvars,
                            integrate=True,
                            integrate_only=True,
                            qobs=True,
                            pannel=False,
                            ensemble=True,
                            stats=True,
                            annualize=True,
                            label='PRE',
                            folder=folder)
    #
    #
    # pos
    fshru = projectfolder + "/datasets/observed/aoi_shru.asc"
    fhists = projectfolder + "/datasets/observed/aoi_histograms.txt"
    fbasin_hists = projectfolder + "/datasets/observed/aoi_basin_histograms.txt"
    pos_folder = bat_slh(fmodels=fmodels,
                            fseries=fseries,
                            fhydroparam=fhydrop,
                            fshruparam=fshru_param,
                            fshru=fshru,
                            ftwi=ftwi,
                            fhistograms=fhists,
                            fbasinhists=fbasin_hists,
                            fbasin=fbasin,
                            fcanopy=fcanopy,
                            model_id='Id',
                            wkpl=True,
                            tui=True,
                            mapback=True,
                            mapvar=mapvars,
                            integrate=True,
                            integrate_only=True,
                            qobs=True,
                            pannel=False,
                            ensemble=True,
                            stats=True,
                            annualize=True,
                            label='POS',
                            folder=folder)
    return {'Pre_folder':pre_folder['Folder'],
            'Pos_folder':pos_folder['Folder']}


def step04_map_asla(folder, prefolder, posfolder, projectfolder):
    """
    uses the median runoff as the runoff for ASLA assessment
    :return:
    """
    from tools import asla
    #
    fseries = projectfolder + "/datasets/observed/calib_series.txt"
    flulc_param = projectfolder + "/datasets/observed/aoi_lulc_param.txt"
    fslope = projectfolder + "/datasets/observed/aoi_slope.asc"
    fsoils =  projectfolder + "/datasets/observed/aoi_soils.asc"
    fsoils_param =  projectfolder + "/datasets/observed/aoi_soils_param.txt"
    #
    # PRE
    frunoff = prefolder + "/annual_R_Median.asc"
    flulc =  projectfolder + "/datasets/projected/scn__lulc_predc/aoi_lulc_predc.asc"
    label = 'PRE'
    # run asla for pre
    pre_asla = asla(fmap_r=frunoff,
                     fslope=fslope,
                     flulc=flulc,
                     fsoils=fsoils,
                     flulcparam=flulc_param,
                     fsoilsparam=fsoils_param,
                     fseries=fseries,
                     aero=6000,
                     label=label,
                     wkpl=True,
                     tui=True,
                     nutrients=True,
                     folder=folder)
    # POS
    frunoff = posfolder + "/annual_R_Median.asc"
    flulc = projectfolder + "/datasets/observed/aoi_lulc.asc"
    label = 'POS'
    # run asla for pos
    pos_asla = asla(fmap_r=frunoff,
                     fslope=fslope,
                     flulc=flulc,
                     fsoils=fsoils,
                     flulcparam=flulc_param,
                     fsoilsparam=fsoils_param,
                     fseries=fseries,
                     aero=6000,
                     label=label,
                     wkpl=True,
                     tui=True,
                     nutrients=True,
                     folder=folder)
    return {'Pre_folder': pre_asla['Folder'], 'Pos_folder': pos_asla['Folder']}


def step05_compute_anomaly(folder, hy_prefolder, hy_posfolder, asla_prefolder, asla_posfolder, mapvars='Qv-R'):
    import os
    import numpy as np
    from visuals import plot_map_view
    from backend import create_rundir
    import inp, out
    outfolder = create_rundir(label='Anomaly', wkplc=folder)
    #
    stats = ['Median']
    # variables
    vars = mapvars.split('-')
    # POS
    pos_folder = hy_posfolder
    pos_all_files = os.listdir(pos_folder)
    # PRE
    pre_folder = hy_prefolder
    pre_all_files = os.listdir(pre_folder)
    for s in stats:
        for i in range(len(vars)):
            # find file path
            for f in pre_all_files:
                if vars[i] + '_' in f and s in f and '.asc' in f:
                    lcl_pre_file_path = '{}/{}'.format(pre_folder, f)
            # find file path
            for f in pos_all_files:
                if vars[i] + '_' in f and s in f and '.asc' in f:
                    lcl_pos_file_path = '{}/{}'.format(pos_folder, f)
            print(lcl_pre_file_path)
            print(lcl_pos_file_path)
            status('loading raster maps')
            meta, pre_map = inp.asc_raster(file=lcl_pre_file_path, dtype='float32')
            meta, pos_map = inp.asc_raster(file=lcl_pos_file_path, dtype='float32')
            meta['NODATA_value'] = -99999
            #
            #
            anom_map = pos_map - pre_map
            #
            lcl_filename = 'annual_{}_{}_anomaly'.format(vars[i], s)
            status('exporting')
            out_file = out.asc_raster(anom_map, meta, folder=outfolder, filename=lcl_filename, dtype='float32')
            rng = (np.abs(np.min(anom_map)), np.abs(np.max(anom_map)))
            rng = (-np.max(rng), np.max(rng))
            plot_map_view(map=anom_map,
                          meta=meta,
                          ranges=rng,
                          mapid='anom',
                          filename=lcl_filename,
                          folder=outfolder,
                          metadata=False,
                          mapttl='{} {} anomaly'.format(vars[i], s),
                          nodata=-99999)
    #
    #
    # ASLA
    vars = ['asl', 'pload', 'nload']
    # POS
    pos_folder = asla_posfolder
    pos_all_files = os.listdir(pos_folder)
    # PRE
    pre_folder = asla_prefolder
    pre_all_files = os.listdir(pre_folder)
    for i in range(len(vars)):
        # find file path
        for f in pre_all_files:
            if vars[i] in f and '.asc' in f and 'log' not in f:
                lcl_pre_file_path = '{}/{}'.format(pre_folder, f)
        # find file path
        for f in pos_all_files:
            if vars[i] in f and '.asc' in f and 'log' not in f:
                lcl_pos_file_path = '{}/{}'.format(pos_folder, f)
        print(lcl_pre_file_path)
        print(lcl_pos_file_path)
        status('loading raster maps')
        meta, pre_map = inp.asc_raster(file=lcl_pre_file_path, dtype='float32')
        meta, pos_map = inp.asc_raster(file=lcl_pos_file_path, dtype='float32')
        meta['NODATA_value'] = -99999
        #
        #
        anom_map = pos_map - pre_map
        #
        lcl_filename = 'annual_{}_anomaly'.format(vars[i])
        status('exporting')
        out_file = out.asc_raster(anom_map, meta, folder=outfolder, filename=lcl_filename, dtype='float32')
        rng = (np.abs(np.min(anom_map)), np.abs(np.max(anom_map)))
        rng = (-np.max(rng), np.max(rng))
        plot_map_view(map=anom_map,
                      meta=meta,
                      ranges=rng,
                      mapid='anom',
                      filename=lcl_filename,
                      folder=outfolder,
                      metadata=False,
                      mapttl='{} anomaly'.format(vars[i]),
                      nodata=-99999)
    view_anomaly(anomfolder=outfolder,
                 hy_prefolder=hy_prefolder,
                 hy_posfolder=hy_posfolder,
                 asla_posfolder=asla_posfolder,
                 asla_prefolder=asla_prefolder,
                 mapvars=mapvars, show=False)
    return outfolder


def step06_compute_uncertainty(folder, hy_prefolder, hy_posfolder, mapvars='Qv-R'):
    import os
    import numpy as np
    from visuals import plot_map_view
    from backend import create_rundir
    #
    # variables
    vars = mapvars.split('-')
    #
    # PRE and POS unc
    pos_folder = hy_posfolder
    pre_folder = hy_prefolder
    folders = [pos_folder, pre_folder]
    pos_outfolder = create_rundir('POS_Uncertainty', folder)
    pre_outfolder = create_rundir('PRE_Uncertainty', folder)
    outfolders = [pos_outfolder, pre_outfolder]
    for j in range(len(folders)):
        lcl_folder = folders[j]
        print(lcl_folder)
        all_files = os.listdir(lcl_folder)
        for i in range(len(vars)):
            # find file path
            for f in all_files:
                if vars[i] + '_' in f and 'Range_90' in f and '.asc' in f:
                    lcl_range_file_path = '{}/{}'.format(lcl_folder, f)
            # find file path
            for f in all_files:
                if vars[i] + '_' in f and 'Median' in f and '.asc' in f:
                    lcl_median_file_path = '{}/{}'.format(lcl_folder, f)
            print(lcl_range_file_path)
            print(lcl_median_file_path)
            print('loading raster maps...')
            meta, range_map = inp.asc_raster(file=lcl_range_file_path, dtype='float32')
            meta, median_map = inp.asc_raster(file=lcl_median_file_path, dtype='float32')
            meta['NODATA_value'] = -99999
            #
            #
            unc_map = 100 * (range_map / (median_map + (1 * (median_map == 0))))  # range 90 / median
            #
            lcl_filename = 'annual_{}_uncertainty'.format(vars[i])
            print('exporting...')
            out_file = out.asc_raster(unc_map, meta, folder=outfolders[j], filename=lcl_filename, dtype='float32')
            rng = (0, np.percentile(unc_map, q=95))
            plot_map_view(map=unc_map,
                          meta=meta,
                          ranges=rng,
                          mapid='unc',
                          filename=lcl_filename,
                          folder=outfolders[j],
                          metadata=False,
                          mapttl='{} uncertainty'.format(vars[i]),
                          nodata=-99999)
    #
    # Avg Unc
    pos_folder = pos_outfolder
    pos_all = os.listdir(pos_folder)
    pre_folder = pre_outfolder
    pre_all = os.listdir(pre_folder)
    folders = [pos_folder, pre_folder]
    outfolder = create_rundir('AVG_Uncertainty', folder)
    for v in vars:
        for f in pos_all:
            if v + '_' in f and '.asc' in f:
                lcl_pos_file = '{}/{}'.format(pos_folder, f)
                print(lcl_pos_file)
        for f in pre_all:
            if v + '_' in f and '.asc' in f:
                lcl_pre_file = '{}/{}'.format(pre_folder, f)
                print(lcl_pre_file)
        meta, pre_map = inp.asc_raster(file=lcl_pre_file, dtype='float32')
        meta, pos_map = inp.asc_raster(file=lcl_pos_file, dtype='float32')
        meta['NODATA_value'] = -99999
        #
        #
        avg_unc = (pre_map + pos_map) / 2
        #
        lcl_filename = 'avg_{}_uncertainty'.format(v)
        print('exporting...')
        out_file = out.asc_raster(avg_unc, meta, folder=outfolder, filename=lcl_filename, dtype='float32')
        rng = (0, np.percentile(avg_unc, q=95))
        plot_map_view(map=avg_unc,
                      meta=meta,
                      ranges=rng,
                      mapid='unc',
                      filename=lcl_filename,
                      folder=outfolder,
                      metadata=False,
                      mapttl='{} average uncertainty'.format(v),
                      nodata=-99999)


def view_obs_data_analyst(fseries, folder='C:/bin/pardinho/produtos_v2', show=True):
    import matplotlib as mpl
    import resample
    _df = pd.read_csv(fseries, sep=';', parse_dates=['Date'])
    #print(_df.head().to_string())
    fields = list(_df.columns[2:])
    month_dct = dict()
    # process
    for field in fields:
        month_dct[field] = dict()
        _df_yr = resample.d2m_prec(dataframe=_df, var_field=field)
        dct_q = resample.group_by_month(dataframe=_df_yr, var_field='Sum', zeros=True)
        months = 'J-F-M-A-M-J-J-A-S-O-N-D'.split('-')
        months_ids = np.arange(12)
        mean = np.zeros(12)
        p95 = np.zeros(12)
        p05 = np.zeros(12)
        count = 0
        for keys in dct_q:
            mean[count] = np.mean(dct_q[keys]['Sum'].values)
            p95[count] = np.percentile(dct_q[keys]['Sum'].values, q=75)
            p05[count] = np.percentile(dct_q[keys]['Sum'].values, q=25)
            count = count + 1
        month_dct[field] = pd.DataFrame({'Month': months,
                                         'Month_id': months_ids,
                                         'Mean': mean,
                                         'P75': p95,
                                         'P25': p05})
    # plot
    fig = plt.figure(figsize=(16, 6))  # Width, Height
    gs = mpl.gridspec.GridSpec(4, 9, wspace=1, hspace=0.8, left=0.05, bottom=0.05, top=0.95, right=0.95)
    order = ['Prec_Herveiras', 'Prec_Boqueirao', 'Prec_Deodoro', 'Q_StaCruz']
    names = {'Prec_Herveiras': 'Precip. Herveiras Station',
             'Prec_Boqueirao': 'Precip. Boqueirao Station',
             'Prec_Deodoro': 'Precip. Deodoro Station',
             'Q_StaCruz': 'Streamflow Santa Cruz Station'}
    count = 0
    for field in order:
        #
        # series plot
        plt.subplot(gs[count: count + 1, :6])
        plt.title(names[field], loc='left')
        color = 'tab:blue'
        if 'Q' in field:
            color = 'navy'
        plt.plot(_df['Date'], _df[field], color)
        if 'Prec' in field:
            plt.ylim((0, 150))
        plt.xlim((_df['Date'].values[0], _df['Date'].values[-1]))
        plt.ylabel('mm')
        #
        # monthly plot
        plt.subplot(gs[count: count + 1, 6:8])
        plt.title('monthly yield', loc='left')
        plt.plot(month_dct[field]['Month_id'], month_dct[field]['Mean'], color)
        plt.xticks(ticks=month_dct[field]['Month_id'].values, labels=month_dct[field]['Month'].values)
        plt.fill_between(x=month_dct[field]['Month_id'],
                         y1=month_dct[field]['P25'],
                         y2=month_dct[field]['P75'],
                         color='tab:grey',
                         alpha=0.4,
                         edgecolor='none')
        plt.ylim(0, 350)
        plt.ylabel('mm')
        #
        # yearly plot
        plt.subplot(gs[count: count + 1, 8:])
        plt.title('yearly yield', loc='left')
        _y = 365 * (_df[field].sum() / len(_df))
        _x = 0.5
        plt.bar(x=_x, height=_y, width=0.3, tick_label='')
        plt.text(x=_x, y=1.1 * _y, s=str(int(_y)))
        plt.xlim(0.1, 1.1)
        plt.ylim(0, 3000)
        plt.ylabel('mm')
        #
        count = count + 1
    # export
    if show:
        plt.show()
        plt.close(fig)
    else:
        filepath = folder + '/step01a__obs_data_all.png'
        plt.savefig(filepath, dpi=400)
        plt.close(fig)


def view_evolution_1(folder, calibfolder, gluefolder):
    full_f = calibfolder + '/generations/population.txt'
    pop_df = pd.read_csv(full_f, sep=';')
    behav_f = gluefolder + '/behavioural.txt'
    behav_df = pd.read_csv(behav_f, sep=';')
    select_f = gluefolder +  '/selection.txt'
    select_df = pd.read_csv(select_f, sep=';')
    fig = plt.figure(figsize=(7, 7), )  # Width, Height
    plt.scatter(x=pop_df['L_ET'], y=pop_df['L_Q'], marker='.', c='tab:grey', alpha=0.4, edgecolors='none')
    plt.scatter(x=behav_df['L_ET'], y=behav_df['L_Q'], marker='.', c='black')
    plt.scatter(x=select_df['L_ET'], y=select_df['L_Q'], marker='.', c='magenta')
    plt.ylim((0, 0.75))
    plt.xlim((-1, -0.4))
    plt.grid(True)
    expfile = folder + '/pop_zoom.png'
    plt.savefig(expfile, dpi=400)
    plt.close(fig)


def view_evolution_2(folder, calibfolder):
    fig = plt.figure(figsize=(7, 4), )  # Width, Height
    full_f = calibfolder + '/generations/population.txt'
    pop_df = pd.read_csv(full_f, sep=';')
    beha_df = pop_df.query('L > -0.65')
    pop_df = pop_df.query('L <= -0.65')
    print(pop_df.head().to_string())
    noise = np.random.normal(loc=0, scale=0.1, size=len(pop_df))
    noise2 = np.random.normal(loc=0, scale=0.1, size=len(beha_df))
    plt.scatter(x=pop_df['Gen'] + noise, y=pop_df['L'], marker='.', c='tab:grey', alpha=0.1, edgecolors='none')
    plt.scatter(x=beha_df['Gen'] + noise2, y=beha_df['L'], marker='.', c='black', alpha=0.3, edgecolors='none')
    plt.xlim((-1, 10))
    plt.ylim((-3, -0.5))
    plt.ylabel('Ly[M|y]')
    plt.xlabel('Generations')
    plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
               labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    expfile = folder + '/evolution.png'
    plt.savefig(expfile, dpi=400)
    plt.close(fig)


def view_anomaly(anomfolder, hy_prefolder, hy_posfolder, asla_prefolder, asla_posfolder, mapvars='Qv-R', show=True):
    from visuals import _custom_cmaps
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    #
    folder = anomfolder
    # variables
    vars = mapvars.split('-')
    #
    _cmaps = _custom_cmaps()
    cmaps = {'R':_cmaps['flow'],
             'RIE':_cmaps['flow'],
             'Qv':_cmaps['flow'],
             'Inf':_cmaps['flow'],
             'ET':_cmaps['flow_v'],
             'Tpgw':_cmaps['flow_v'],
             'VSA':'Blues',
             'asl':_cmaps['sed'],
             'pload':_cmaps['sed'],
             'nload':_cmaps['sed']}
    units = {'R': 'mm',
             'RIE': 'mm',
             'Qv': 'mm',
             'Inf': 'mm',
             'ET': 'mm',
             'Tpgw': 'mm',
             'VSA': '%',
             'asl': 'ton/yr',
             'pload': 'kgP/yr',
             'nload': 'kgN/yr'
             }
    for v in vars:
        print(v)
        if v in ['asl', 'pload', 'nload']:
            fpos = "{}/{}.asc".format(asla_posfolder, v)
            fpre = "{}/{}.asc".format(asla_prefolder, v)
            fanm = "{}/annual_{}_anomaly.asc".format(anomfolder, v)
        else:
            fpos = "{}/annual_{}_Median.asc".format(hy_posfolder, v)
            fpre = "{}/annual_{}_Median.asc".format(hy_prefolder, v)
            fanm = "{}/annual_{}_Median_anomaly.asc".format(anomfolder, v)
        files_lst = [fpre, fpos, fanm]
        maps_lst = list()
        for f in files_lst:
            meta, rmap = inp.asc_raster(file=f, dtype='float32')
            rmap = rmap[600:1200, 500:900]
            maps_lst.append(rmap.copy())
        #
        # get values
        _vmax = np.max((np.percentile(maps_lst[0], q=90), np.percentile(maps_lst[1], q=90)))
        _vanm = np.max((np.abs(np.min(maps_lst[2])), np.abs(np.max(maps_lst[2]))))
        #
        fig = plt.figure(figsize=(10, 4))  # Width, Height
        gs = mpl.gridspec.GridSpec(3, 9, wspace=0.8, hspace=0.6)
        fig.suptitle('{} median anomaly'.format(v))
        #
        plt.subplot(gs[:4, :3])
        im = plt.imshow(maps_lst[0], cmap=cmaps[v], vmin=0, vmax=1600)
        plt.title('pre-develop. ({})'.format(units[v]))
        plt.colorbar(im, shrink=0.4)
        plt.axis('off')
        #
        plt.subplot(gs[:4, 3:6])
        im = plt.imshow(maps_lst[1], cmap=cmaps[v], vmin=0, vmax=1600)
        plt.title('post-develop. ({})'.format(units[v]))
        plt.colorbar(im, shrink=0.4)
        plt.axis('off')
        #
        plt.subplot(gs[:4, 6:9])
        im = plt.imshow(maps_lst[2], cmap='seismic_r', vmin=-_vanm, vmax=_vanm)
        plt.title('anomaly ({})'.format(units[v]))
        plt.colorbar(im, shrink=0.4)
        plt.axis('off')
        #
        filename = '{}_median_anomaly'.format(v)
        if show:
            plt.show()
            plt.close(fig)
        else:
            filepath = folder + '/' + filename + '.png'
            plt.savefig(filepath, dpi=400)
            plt.close(fig)



def main(folder, projectfolder):
    from backend import create_rundir
    import os
    # -01 SET UP FOLDERS
    label = 'run_'
    folder = create_rundir(label=label, wkplc=folder)
    folder_s1 = folder + '/s1'
    folder_s2 = folder + '/s2'
    folder_s3 = folder + '/s3'
    folder_lst = [folder_s1, folder_s2, folder_s3]
    for f in folder_lst:
        os.mkdir(f)
    #
    # 00 SET INPUT SERIES
    dct = step00_set_input_series(folder=folder)
    lbls = ['s1', 's2', 's3', 'aoi']
    for i in lbls:
        print(dct[i].head().to_string())
    #
    #
    # loop in sets
    for folder in folder_lst:
        print(folder)
        #
        # 01 SEARCH MODELS
        calib_folder = step01_search_models(folder=folder,
                                            projectfolder=projectfolder)
        #
        # 02 SELECT BEHAVIOURAL MODELS
        glue_folder = step02_select_models(folder=folder,
                                           calibfolder=calib_folder,
                                           projectfolder=projectfolder)
        #
        # 03 MAP HYDRO PROCESSES FOR AOI PRE and POS
        map_folders = step03_map_processes(folder=folder,
                                           gluefolder=glue_folder,
                                           projectfolder=projectfolder)
        #
        # 04 MAP ASLA PROCESSES FOI AOI PRE AND POS
        asla_folders = asla_folders = step04_map_asla(folder=folder,
                                                      prefolder=map_folders['Pre_folder'],
                                                      posfolder=map_folders['Pos_folder'],
                                                      projectfolder=projectfolder)
        #
        # 05 COMPUTE ANOMALY
        step05_compute_anomaly(folder='C:/bin',
                               hy_prefolder=map_folders['Pre_folder'],
                               hy_posfolder=map_folders['Pos_folder'],
                               asla_prefolder=asla_folders['Pre_folder'],
                               asla_posfolder=asla_folders['Pos_folder'])
        #
        # 06 COMPUTE UNCERTAINTY
        step06_compute_uncertainty(folder='C:/bin',
                                   hy_prefolder=map_folders['Pre_folder'],
                                   hy_posfolder=map_folders['Pos_folder'])
        #


main(folder='C:/bin',
     projectfolder='C:/000_myFiles/myDrive/Plans3/pardinho')
'''
folders = step03_map_processes(folder='C:/bin',
                     gluefolder='C:/bin/pardinho/produtos_v2/run__2022-01-06-16-11-24/s1/GLUE_L_2022-01-06-16-13-09',
                     projectfolder='C:/000_myFiles/myDrive/Plans3/pardinho')

asla_folders = step04_map_asla(folder='C:/bin',
                                prefolder='C:/bin/PRE_batSLH_2022-01-07-09-32-57',
                                posfolder='C:/bin/POS_batSLH_2022-01-07-09-36-07',
                                projectfolder='C:/000_myFiles/myDrive/Plans3/pardinho')

step05_compute_anomaly(folder='C:/bin',
                       hy_prefolder='C:/bin/PRE_batSLH_2022-01-07-09-32-57',
                       hy_posfolder='C:/bin/POS_batSLH_2022-01-07-09-36-07',
                       asla_prefolder=asla_folders['Pre_folder'],
                       asla_posfolder=asla_folders['Pos_folder'])

step06_compute_uncertainty(folder='C:/bin',
                           hy_prefolder='C:/bin/PRE_batSLH_2022-01-07-09-32-57',
                           hy_posfolder='C:/bin/POS_batSLH_2022-01-07-09-36-07',)


step05_compute_anomaly(folder='C:/bin',
                       hy_prefolder='C:/bin/PRE_batSLH_2022-01-07-09-32-57',
                       hy_posfolder='C:/bin/POS_batSLH_2022-01-07-09-36-07',
                       asla_prefolder='C:/bin/PRE_ASLA_2022-01-07-10-00-55',
                       asla_posfolder='C:/bin/POS_ASLA_2022-01-07-10-01-32')

'''

