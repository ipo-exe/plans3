import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tui import status
import inp, out


def step00_set_input_series(folder, infolder, projectfolder):
    import resample
    from hydrology import convert_q2sq, convert_sq2q
    import inp
    #
    # combine the available series
    status('combining avaliable series')
    in_folder = infolder
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
    fraster = projectfolder + "/datasets/observed/calib_basin.asc"
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
    fcalib = projectfolder + "/datasets/observed/calib_series.txt"
    calib_df = pd.read_csv(fcalib, sep=';', parse_dates=['Date'])
    calib_s1_df = calib_df.copy()
    calib_s2_df = calib_df.copy()
    s3_frame = 'Date >= "2013-01-01"'
    calib_s3_df = calib_df.query(s3_frame).copy()
    faoi = projectfolder + "/datasets/observed/aoi_series.txt"
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
    _ofiles = list()
    for i in range(len(_dfs)):
        _of = folder + '/' + _names[i]
        _dfs[i].to_csv(_of, sep=';', index=False)
        _ofiles.append(_of)
    return _ofiles


def step01_search_models(folder, projectfolder, fseries):
    import backend
    from tools import calibrate
    status('searching models')
    # define the output workplace folder
    outfolder = folder
    # get folder of observed datasets
    # get observed datasets standard names
    files_input = backend.get_input2calibhydro()
    folder = projectfolder + '/datasets/observed'
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
    generations = 20
    popsize = 1000
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
                     behavioural=-0.65,
                     run_ensemble=True,
                     folder=outfolder,
                     wkpl=True,
                     normalize=False,
                     tui=True)
    # filter
    status('selecting models')
    fmodels = gluefiles['Folder'] + '/behavioural.txt' #calib_folder + '/generations/population.txt'
    models_df = pd.read_csv(fmodels, sep=';')
    models_df = models_df.drop_duplicates(subset=['SetIds'])
    models_df = models_df.sort_values(by='L', ascending=False)
    models_df = models_df.head(20)
    print(len(models_df))
    #models_df['PBias_abs'] = models_df['PBias'].abs()
    # activate this:
    #models_df = models_df.query('L > -0.65 and KGElog > 0.4 and NSElog > 0 and Qb_R > 0.25')
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


def step03_map_processes(folder, gluefolder, projectfolder, fseries, vars='Qv-R'):
    from tools import bat_slh
    ftwi = projectfolder + "/datasets/observed/aoi_twi.asc"
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


def step04_map_asla(folder, prefolder, posfolder, projectfolder, fseries):
    """
    uses the median runoff as the runoff for ASLA assessment
    :return:
    """
    from tools import asla
    #
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


def step07_zonal_stats(fcar_map, anom_folder, unc_folder):
    import matplotlib.pyplot as plt
    import pandas as pd
    import geo
    fcar = fcar_map
    meta, car = inp.asc_raster(file=fcar, dtype='int16')
    _ids = np.unique(car)
    vars = ['R', 'RIE', 'Qv', 'Inf', 'ET', 'Tpgw', 'asl', 'pload', 'nload']
    _out_df = pd.DataFrame({'id_car':_ids})
    for v in vars:
        # annual anomaly
        if v in ['asl', 'pload', 'nload']:
            fanm = "{}/annual_{}_anomaly.asc".format(anom_folder, v)
        else:
            fanm = "{}/annual_{}_Mean_anomaly.asc".format(anom_folder, v)
        meta, lcl_map = inp.asc_raster(file=fanm, dtype='float32')
        _dct = geo.zonalstats(field=lcl_map, zones=car, tui=True)
        _out_df['{}_aa_mean'.format(v)] = _dct['Mean']
        # average uncertainty
        if v in ['asl', 'pload', 'nload']:
            pass
        else:
            fanm = "{}/avg_{}_uncertainty.asc".format(unc_folder, v)
            meta, lcl_map = inp.asc_raster(file=fanm, dtype='float32')
            _dct = geo.zonalstats(field=lcl_map, zones=car, tui=True)
            _out_df['{}_un_mean'.format(v)] = _dct['Mean']
    print(_out_df.head(10).to_string())
    fout = 'C:/bin/pardinho/produtos/aoi_car_zonal_stats.txt'
    _out_df.to_csv(fout, sep=';', index=False)


def step08_map_index(folder, fcar_stats, fcar_basic, show=True):
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    import shapefile
    folder1 = folder
    folder2 = '{}/hist_maps'.format(folder)

    fbasic = fcar_basic
    fstats = fcar_stats
    fbasin = r"C:\000_myFiles\myDrive\gis\pnh\misc\aoi_basin.shp"

    basic_df = pd.read_csv(fbasic, sep=';')
    # print(basic_df.head().to_string())
    stats_df = pd.read_csv(fstats, sep=';')
    # print(stats_df.head().to_string())
    df = pd.merge(basic_df, stats_df, 'inner', left_on='id_imovel', right_on='id_car')

    vars = ['R', 'Qv', 'RIE', 'Inf', 'ET', 'Tpgw', 'asl', 'nload', 'pload', 'Prx', 'area']
    _ptiles1 = [0.20, 0.40, 0.60, 0.80, 0.95]
    _ptiles2 = [0.05, 0.20, 0.40, 0.60, 0.80]
    _classes1 = [1, 2, 3, 4, 5, 6]
    _classes2 = [6, 5, 4, 3, 2, 1]
    ptiles = {'aa': {'R': _ptiles1,
                     'Qv': _ptiles2,
                     'RIE': _ptiles1,
                     'Inf': _ptiles2,
                     'ET': _ptiles2,
                     'Tpgw': _ptiles2,
                     'asl': _ptiles1,
                     'nload': _ptiles1,
                     'pload': _ptiles1,
                     'Prx': _ptiles2,
                     'area': _ptiles1},
              'un': {'R': _ptiles2,
                     'Qv': _ptiles2,
                     'RIE': _ptiles2,
                     'Inf': _ptiles2,
                     'ET': _ptiles2,
                     'Tpgw': _ptiles2}}
    classes = {'aa': {'R': _classes1,
                      'Qv': _classes2,
                      'RIE': _classes1,
                      'Inf': _classes2,
                      'ET': _classes2,
                      'Tpgw': _classes2,
                      'asl': _classes1,
                      'nload': _classes1,
                      'pload': _classes1,
                      'Prx': _classes2,
                      'area':_classes1},
               'un': {'R': _classes2,
                      'Qv': _classes2,
                      'RIE': _classes2,
                      'Inf': _classes2,
                      'ET': _classes2,
                      'Tpgw': _classes2}}
    _colors1 = ['aqua', 'lime', 'gold', 'orange', 'red', 'black']
    _colors2 = ['black', 'red', 'orange', 'gold', 'lime', 'aqua']
    _colors3 = ['black', 'grey', 'darkgrey', 'silver', 'lightgrey', 'whitesmoke', ]
    colors = {'aa': {'R': _colors1,
                     'Qv': _colors2,
                     'RIE': _colors1,
                     'Inf': _colors2,
                     'ET': _colors2,
                     'Tpgw': _colors2,
                     'asl': _colors1,
                     'nload': _colors1,
                     'pload': _colors1,
                     'Prx': _colors2,
                     'area':_colors1},
              'un': {'R': _colors3,
                     'Qv': _colors3,
                     'RIE': _colors3,
                     'Inf': _colors3,
                     'ET': _colors3,
                     'Tpgw': _colors3}}
    #
    units = {'aa': {'R': 'mm',
                    'Qv': 'mm',
                    'RIE': 'mm',
                    'Inf': 'mm',
                    'ET': 'mm',
                    'Tpgw': 'mm',
                    'asl': 'ton/year',
                    'nload': 'kg-N/year',
                    'pload': 'kg-P/year',
                    'Prx': 'm',
                    'area': 'ha'},
             'un': {'R': '%',
                    'Qv': '%',
                    'RIE': '%',
                    'Inf': '%',
                    'ET': '%',
                    'Tpgw': '%'}}
    #
    #
    stats_types = ['aa', 'un']
    for stats in stats_types:
        for v in vars:
            print(v)
            if stats == 'un' and v in ['asl', 'nload', 'pload', 'Prx', 'area']:
                pass
            else:
                # get ptiles
                if v in ['Prx', 'area']:
                    _lcl_field = v
                else:
                    _lcl_field = '{}_{}_mean'.format(v, stats)
                _lcl_x = df[_lcl_field].values
                _ptiles = np.quantile(a=_lcl_x, q=ptiles[stats][v])
                df['{}_{}_ind'.format(v, stats)] = 0
                for i in range(len(df)):
                    if _lcl_x[i] < _ptiles[0]:
                        df['{}_{}_ind'.format(v, stats)].values[i] = classes[stats][v][0]
                    elif _lcl_x[i] < _ptiles[1] and _lcl_x[i] >= _ptiles[0]:
                        df['{}_{}_ind'.format(v, stats)].values[i] = classes[stats][v][1]
                    elif _lcl_x[i] < _ptiles[2] and _lcl_x[i] >= _ptiles[1]:
                        df['{}_{}_ind'.format(v, stats)].values[i] = classes[stats][v][2]
                    elif _lcl_x[i] < _ptiles[3] and _lcl_x[i] >= _ptiles[2]:
                        df['{}_{}_ind'.format(v, stats)].values[i] = classes[stats][v][3]
                    elif _lcl_x[i] < _ptiles[4] and _lcl_x[i] >= _ptiles[3]:
                        df['{}_{}_ind'.format(v, stats)].values[i] = classes[stats][v][4]
                    else:
                        df['{}_{}_ind'.format(v, stats)].values[i] = classes[stats][v][5]
                output = folder1 + '/aoi_car_full_indices.txt'
                df.to_csv(output, sep=';', index=False)
                #
                #
                # plot
                sf = shapefile.Reader(fbasin)
                fig = plt.figure(figsize=(10, 4.5))  # Width, Height
                fig.suptitle('{} {}'.format(stats, v))
                gs = mpl.gridspec.GridSpec(3, 5, wspace=0.05, hspace=0.05)
                ax = fig.add_subplot(gs[:, 3:])
                for i in range(len(_classes1)):
                    _lcl_df = df.query('{}_{}_ind == {}'.format(v, stats, classes[stats][v][i]))
                    plt.scatter(_lcl_df['long'], _lcl_df['lat'],
                                c=colors[stats][v][i],
                                marker='.',
                                zorder=classes[stats][v][i],
                                label=str(classes[stats][v][i]))
                plt.legend()
                # overlay shapefile
                patch = plt.Polygon(sf.shape(0).points, facecolor='none', edgecolor='black', linewidth=1, zorder=10)
                ax.add_patch(patch)
                ax.axis('scaled')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlim([360694, 385296])
                ax.set_ylim([6721596, 6752258])
                #
                #
                ax = fig.add_subplot(gs[1:, :3])
                _hist = plt.hist(x=_lcl_x, bins=100, color='grey')
                for i in range(len(_ptiles)):
                    plt.vlines(x=_ptiles[i], ymin=0, ymax=1.2 * np.max(_hist[0]), colors='tab:red')
                for i in range(len(_classes1)):
                    if i == 0:
                        lcl_lcl_x = (_ptiles[i] + np.min(_lcl_x)) / 2
                    elif i < len(_classes1) - 1:
                        lcl_lcl_x = (_ptiles[i] + _ptiles[i - 1]) / 2
                    else:
                        lcl_lcl_x = (_ptiles[i - 1] + np.max(_lcl_x)) / 2
                    plt.plot(lcl_lcl_x, 1.1 * np.max(_hist[0]),
                             marker='o',
                             color=colors[stats][v][i],
                             markersize=10)
                plt.ylim(0, 1.2 * np.max(_hist[0]))

                if stats == 'un' and v == 'Qv':
                    plt.xlim(0, 500)

                plt.ylabel('freq.')
                plt.xlabel(units[stats][v])
                #
                if show:
                    plt.show()
                    plt.close(fig)
                else:
                    filepath = folder2 + '/{}_{}.png'.format(stats, v)
                    plt.savefig(filepath, dpi=400)
                    plt.close(fig)


def deprec__step08_priority(folder, fcar_index, show=False):
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    import shapefile
    _df = pd.read_csv(fcar_index, sep=';')

    print(_df.head().to_string())
    aa_vars = ['R']
    s = ''
    for i in range(len(aa_vars)):
        s = s + ' {}_aa_ind >= 5 and'.format(aa_vars[i])
    s = s + ' R_un_ind >= 5'
    print(s)
    priority_df = _df.query(s)
    print(priority_df.head().to_string())
    print(len(priority_df))
    qv = np.sum((priority_df['Qv_aa_mean'].values * priority_df['area'].values)) / priority_df['area'].sum()
    print(qv)
    r = np.sum((priority_df['R_aa_mean'].values * priority_df['area'].values)) / priority_df['area'].sum()
    print(r)

    fout = '{}/priority_policy.txt'.format(folder)
    priority_df.to_csv(fout, sep=';', index=False)

    fbasin = r"C:\000_myFiles\myDrive\gis\pnh\misc\aoi_basin.shp"
    # plot
    sf = shapefile.Reader(fbasin)
    fig = plt.figure(figsize=(6, 4.5))  # Width, Height
    ax = fig.add_subplot()
    plt.scatter(priority_df['long'], priority_df['lat'], c='tab:grey', marker='o')
    #
    # overlay shapefile
    patch = plt.Polygon(sf.shape(0).points, facecolor='none', edgecolor='black', linewidth=1, zorder=10)
    ax.add_patch(patch)
    ax.axis('scaled')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([360694, 385296])
    ax.set_ylim([6721596, 6752258])
    plt.title(s)
    # export
    if show:
        plt.show()
        plt.close(fig)
    else:
        fout = '{}/priority_policy.png'.format(folder)
        plt.savefig(fout, dpi=400)
        plt.close(fig)


def step08_priority_index(folder, fcar_index, show=False):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import shapefile

    investment = 100000 # R$

    ppsa = 200 # R$ / ha

    max_area = investment / ppsa

    print('area : {}'.format(max_area))

    _df = pd.read_csv(fcar_index, sep=';')

    vars_aa = ['R', 'Qv', 'Inf', 'Qv', 'asl', 'nload', 'pload'] #, 'Inf', 'Qv', 'asl', 'nload', 'pload']
    vars_un = ['R', 'Qv', 'Inf', 'Qv', 'R', 'R', 'R'] #, 'Inf', 'Qv', 'R', 'R', 'R']
    #
    # get reference priority index
    ip_0 = np.zeros(len(_df))
    for v in vars_aa:
        ip_0 = ip_0 + _df['{}_aa_ind'.format(v)].values
    ip_0 = ip_0 / len(vars_aa)
    _df['IP_0'] = ip_0 #reclass(v=ip_0)
    #
    # define order
    _df = _df.sort_values(by='IP_0', ascending=False)
    ip_0_rnk = np.zeros(len(_df))
    area_count = 0
    for i in range(len(_df)):
        area_count = area_count + _df['area'].values[i]
        if area_count <= max_area:
            ip_0_rnk[i] = i + 1
        else:
            ip_0_rnk[i] = 0
    _df['IP_0_rnk'] = ip_0_rnk

    #
    # get priority considering uncertainty
    ip_1 = np.zeros(len(_df))
    ip_1_iu_sum = np.zeros(len(_df))
    for i in range(len(vars_aa)):
        ip_1 = ip_1 + (_df['{}_aa_ind'.format(vars_aa[i])].values * _df['{}_un_ind'.format(vars_un[i])].values)
        ip_1_iu_sum = ip_1_iu_sum + _df['{}_un_ind'.format(vars_un[i])].values
    ip_1 = ip_1 / ip_1_iu_sum
    _df['IP_1'] = ip_1 #reclass(v=ip_1)
    #
    # define order
    _df = _df.sort_values(by='IP_1', ascending=False)
    ip_1_rnk = np.zeros(len(_df))
    area_count = 0
    for i in range(len(_df)):
        area_count = area_count + _df['area'].values[i]
        if area_count <= max_area:
            ip_1_rnk[i] = i + 1
        else:
            ip_1_rnk[i] = 0
    _df['IP_1_rnk'] = ip_1_rnk

    _df['IP_1_rnk_diff'] = _df['IP_1_rnk'] - _df['IP_0_rnk']

    _rnk_min = np.min([_df['IP_0_rnk'].values, _df['IP_1_rnk'].values])
    _rnk_max = np.max([_df['IP_0_rnk'].values, _df['IP_1_rnk'].values])

    _rnk_dff_min = _df['IP_1_rnk_diff'].min()
    _rnk_dff_max = _df['IP_1_rnk_diff'].max()
    _rnk_dff_rng = np.max([_rnk_dff_min, _rnk_dff_max])

    print(_df.head(10).to_string())

    fout = '{}/priority_policy.txt'.format(folder)
    _df.to_csv(fout, sep=';', index=False)

    # load shapefile
    fbasin = r"C:\000_myFiles\myDrive\gis\pnh\misc\aoi_basin.shp"
    sf = shapefile.Reader(fbasin)
    #
    # plot
    _p_cmp = 'RdYlGn'

    fig = plt.figure(figsize=(12, 5))  # Width, Height
    fig.suptitle('R${} | R${}/ha | {}ha'.format(investment, ppsa, max_area))
    gs = mpl.gridspec.GridSpec(3, 9, wspace=0.8, hspace=0.6)
    #
    # IP0
    ax = fig.add_subplot(gs[:4, :3])
    _df0 = _df.query('IP_0_rnk > 0')
    _long = np.append(_df0['long'].values, [386300, 386300])
    _lat = np.append(_df0['lat'].values, [6752259, 6752259])
    _z = np.append(_df0['IP_0_rnk'].values, [_rnk_min, _rnk_max])
    plt.scatter(_long, _lat, c=_z, cmap=_p_cmp, marker='.')
    plt.colorbar(shrink=0.4)
    #
    # overlay shapefile
    patch = plt.Polygon(sf.shape(0).points, facecolor='none', edgecolor='black', linewidth=1, zorder=10)
    ax.add_patch(patch)
    ax.axis('scaled')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([360694, 385296])
    ax.set_ylim([6721596, 6752258])
    plt.title('priority rank:\nanomaly only')
    #
    # IP1
    ax = fig.add_subplot(gs[:4, 3:6])
    _df1 = _df.query('IP_1_rnk > 0')
    _long = np.append(_df1['long'].values, [386300, 386300])
    _lat = np.append(_df1['lat'].values, [6752259, 6752259])
    _z = np.append(_df1['IP_1_rnk'].values, [_rnk_min, _rnk_max])
    plt.scatter(_long, _lat, c=_z, cmap=_p_cmp, marker='.')
    plt.colorbar(shrink=0.4)
    #
    # overlay shapefile
    patch = plt.Polygon(sf.shape(0).points, facecolor='none', edgecolor='black', linewidth=1, zorder=10)
    ax.add_patch(patch)
    ax.axis('scaled')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([360694, 385296])
    ax.set_ylim([6721596, 6752258])
    plt.title('priority rank:\nanomaly & uncertainty')

    ax = fig.add_subplot(gs[:4, 6:9])
    _long = np.append(_df1['long'].values, [386300, 386300])
    _lat = np.append(_df1['lat'].values, [6752259, 6752259])
    _z = np.append(_df1['IP_1_rnk_diff'].values, [-_rnk_dff_rng, _rnk_dff_max])
    plt.scatter(_long, _lat, c=_z, cmap='seismic', marker='.')
    plt.colorbar(shrink=0.4)
    #
    # overlay shapefile
    patch = plt.Polygon(sf.shape(0).points, facecolor='none', edgecolor='black', linewidth=1, zorder=10)
    ax.add_patch(patch)
    ax.axis('scaled')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([360694, 385296])
    ax.set_ylim([6721596, 6752258])
    plt.title('rank difference')
    #
    filename = 'priority_ranks'
    if show:
        plt.show()
        plt.close(fig)
    else:
        filepath = folder + '/' + filename + '.png'
        plt.savefig(filepath, dpi=400)
        plt.close(fig)





def step09_compare_policies(folder, show=False):
    """
    comparing datasets policies
    :param folder:
    :param show:
    :return:
    """
    ffull_s1 = '{}/s1/aoi_car_full_indices.txt'.format(folder)
    fpolicy_s1 = '{}/s1/priority_policy.txt'.format(folder)
    fpolicy_s2 = '{}/s2/priority_policy.txt'.format(folder)
    fpolicy_s3 = '{}/s3/priority_policy.txt'.format(folder)
    # load dataframe
    full_df = pd.read_csv(ffull_s1, sep=';')
    policy_s1_df = pd.read_csv(fpolicy_s1, sep=';')
    print(len(policy_s1_df))
    policy_s2_df = pd.read_csv(fpolicy_s2, sep=';')
    print(len(policy_s2_df))
    policy_s3_df = pd.read_csv(fpolicy_s3, sep=';')
    print(len(policy_s3_df))
    policies = [policy_s1_df, policy_s2_df, policy_s3_df]
    labels = ['s1', 's2', 's3']
    variables = ['R', 'Qv', 'Inf', 'ET', 'asl', 'nload', 'pload']
    out_df = pd.DataFrame({'Var':variables})
    for i in range(0, len(policies)):
        lcl_df_0 = policies[i]
        lcl_df = pd.merge(full_df, policies[i], how='right', on='id_car', suffixes=('_s1', '_' + labels[i]))
        out_df[labels[i]] = 0
        out_df[labels[i] + 's1'] = 0
        out_df[labels[i] + 's1_d'] = 0
        for j in range(len(variables)):
            v = variables[j]
            v_d0_lbl = '{}_aa_mean'.format(v)
            v_d1_lbl = '{}_aa_mean_s1'.format(v)
            v_d0 = np.sum(lcl_df_0[v_d0_lbl].values * lcl_df_0['area'].values) / np.sum(lcl_df_0['area'].values)
            v_d1 = np.sum(lcl_df[v_d1_lbl].values * lcl_df['area_s1'].values) / np.sum(lcl_df['area_s1'].values)
            v_disc = v_d0 - v_d1
            out_df[labels[i]].values[j] = v_d0
            out_df[labels[i] + 's1'].values[j] = v_d1
            out_df[labels[i] + 's1_d'].values[j] = v_disc
    print(out_df.to_string())
    fout = '{}/compare_policies.txt'.format(folder)
    out_df.to_csv(fout, sep=';', index=False)
    #
    # plot
    labels = {'s2':['ds0' ,'dsA', 'dsA|ds0', 'discrep.'],
              's3':['ds0' ,'dsB', 'dsB|ds0', 'discrep.']}

    for i in range(len(variables)):
        lcl_policies = ['s2', 's3']
        for j in range(len(lcl_policies)):
            values1 = (out_df['s1'].values[i], out_df[lcl_policies[j]].values[i],
                       out_df['{}s1'.format(lcl_policies[j])].values[i],
                       out_df['{}s1_d'.format(lcl_policies[j])].values[i])
            #values2 = (out_df['s3'].values[0], out_df['s3s1'].values[0], out_df['s3s1_d'].values[0])
            x = np.arange(len(labels[lcl_policies[j]]))  # the label locations
            width = 0.4  # the width of the bars
            fig = plt.figure(figsize=(5, 2.5))  # Width, Height
            plt.subplot(111)
            plt.bar(x, values1, width, label='ok', color='tab:grey')
            #plt.bar(x + (width / 2), values2, width, label='ok', color='maroon')
            #
            # Add some text for labels, title and custom x-axis tick labels, etc.
            plt.ylabel('mm')
            plt.xticks(x, labels[lcl_policies[j]])
            plt.title(variables[i] + ' {}'.format(lcl_policies[j]))
            plt.grid(True, axis='y')
            if show:
                plt.show()
                plt.close(fig)
            else:
                filepath = folder + '\{}_{}_compare.png'.format(variables[i], lcl_policies[j])
                plt.savefig(filepath, dpi=400)
                plt.close(fig)


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


def view_evolution_1(folder, calibfolder, gluefolder, show=True):
    full_f = calibfolder + '/generations/population.txt'
    pop_df = pd.read_csv(full_f, sep=';')
    behav_f = gluefolder + '/behavioural.txt'
    behav_df = pd.read_csv(behav_f, sep=';')
    select_f = gluefolder + '/selection.txt'
    select_df = pd.read_csv(select_f, sep=';')
    fig = plt.figure(figsize=(7, 7), )  # Width, Height
    plt.scatter(x=pop_df['L_ET'], y=pop_df['L_Q'], marker='.', c='tab:grey', alpha=0.4, edgecolors='none')
    plt.scatter(x=behav_df['L_ET'], y=behav_df['L_Q'], marker='.', c='black')
    plt.scatter(x=select_df['L_ET'], y=select_df['L_Q'], marker='.', c='magenta')
    plt.ylim((0, 0.9))
    plt.xlim((-1, -0.4))
    plt.grid(True)
    filename = 'zoom_pop'
    if show:
        plt.show()
        plt.close(fig)
    else:
        filepath = folder + '/' + filename + '.png'
        plt.savefig(filepath, dpi=400)
        plt.close(fig)


def view_evolution_2(folder, calibfolder, show=True):
    fig = plt.figure(figsize=(7, 4), )  # Width, Height
    full_f = calibfolder + '/generations/population.txt'
    pop_df = pd.read_csv(full_f, sep=';')
    _x = list()
    _y = list()
    for g in range(0, 20):
        lcl_df = pop_df.query('Gen == {}'.format(g))
        lcl_y = np.percentile(lcl_df['L'].values, 50)
        lcl_x = g
        _x.append(lcl_x)
        _y.append(lcl_y)
    beha_df = pop_df.query('L > -0.9')
    beha_df = beha_df.nlargest(1000, columns=['L'])
    pop_df = pop_df.query('L <= {}'.format(beha_df['L'].min()))
    # print(pop_df.head().to_string())
    noise = np.random.normal(loc=0, scale=0.1, size=len(pop_df))
    noise2 = np.random.normal(loc=0, scale=0.1, size=len(beha_df))
    plt.scatter(x=pop_df['Gen'] + noise, y=pop_df['L'], marker='.', c='tab:grey', alpha=0.1, edgecolors='none')
    plt.scatter(x=beha_df['Gen'] + noise2, y=beha_df['L'], marker='.', c='black', alpha=0.3, edgecolors='none')
    plt.plot(_x, _y, 'blue')
    plt.xlim((-1, 20))
    plt.ylim((-3, -0.5))
    plt.ylabel('Ly[M|y]')
    plt.xlabel('Generations')
    plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
               labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    filename = 'evolution'
    if show:
        plt.show()
        plt.close(fig)
    else:
        filepath = folder + '/' + filename + '.png'
        plt.savefig(filepath, dpi=400)
        plt.close(fig)


def view_evolution_3(folder, show=True):
    fig = plt.figure(figsize=(7, 4), )  # Width, Height
    sets_lst = ['s1', 's2', 's3']
    colors_dct = {'s1':'green', 's2':'orange', 's3':'maroon'}
    for s in sets_lst:
        full_f = '{}/{}/search/generations/population.txt'.format(folder, s)
        pop_df = pd.read_csv(full_f, sep=';')
        _x = list()
        _y50 = list()
        _y25 = list()
        _y75 = list()
        for g in range(0, 20):
            lcl_df = pop_df.query('Gen == {}'.format(g))
            lcl_y50 = np.percentile(lcl_df['L'].values, 97.5)
            lcl_y25 = np.percentile(lcl_df['L'].values, 96)
            lcl_y75 = np.percentile(lcl_df['L'].values, 99)
            lcl_x = g
            _x.append(lcl_x)
            _y50.append(lcl_y50)
            _y25.append(lcl_y25)
            _y75.append(lcl_y75)
        plt.plot(_x, _y50, c=colors_dct[s])
        plt.fill_between(x=_x, y1=_y25, y2=_y75, color=colors_dct[s], alpha=0.2, edgecolor='none')
    plt.xlim((1, 19))
    plt.ylim((-1, -0.57))
    plt.ylabel('Ly[M|y]')
    plt.xlabel('Generations')
    plt.xticks(ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
               labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    filename = 'evolution_all'
    if show:
        plt.show()
        plt.close(fig)
    else:
        filepath = folder + '/' + filename + '.png'
        plt.savefig(filepath, dpi=400)
        plt.close(fig)


def view_evolution_4(folder, show=True):
    sets_lst = ['s1', 's2', 's3']
    colors_dct = {'s1': 'green', 's2': 'orange', 's3': 'maroon'}
    fig = plt.figure(figsize=(7, 7), )  # Width, Height
    for s in sets_lst:
        behav_f = '{}/{}/select/behavioural.txt'.format(folder, s)
        behav_df = pd.read_csv(behav_f, sep=';')
        select_f = '{}/{}/select/selection.txt'.format(folder, s)
        select_df = pd.read_csv(select_f, sep=';')
        plt.scatter(x=behav_df['L_ET'], y=behav_df['L_Q'], marker='.', c=colors_dct[s], alpha=0.3, edgecolors='none')
        plt.scatter(x=select_df['L_ET'], y=select_df['L_Q'], marker='.', c=colors_dct[s])
    plt.ylim((0.3, 0.9))
    plt.xlim((-0.65, -0.45))
    plt.grid(True)
    filename = 'zoom_pop'
    if show:
        plt.show()
        plt.close(fig)
    else:
        filepath = folder + '/' + filename + '.png'
        plt.savefig(filepath, dpi=400)
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


def view_uncertainty(folder, pos_folder, show=True):
    from visuals import _custom_cmaps
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    #
    _xs = [100, 150, 320, 290]
    _ys = [400, 100, 460, 300]
    labels = ['s1', 's2', 's3', 's4']

    folder = folder
    vars = ['R', 'Qv'] #, 'Inf', 'ET', 'Tpgw', 'VSA']
    units = {'R':'mm',
             'RIE':'mm',
             'Qv':'mm',
             'Inf':'mm',
             'ET':'mm',
             'Tpgw':'mm',
             'VSA':'%'}
    colors = {'R': 'white',
             'RIE': 'white',
             'Qv': 'white',
             'Inf': 'white',
             'ET': 'black',
             'Tpgw': 'black',
             'VSA': 'black'}
    _cmaps = _custom_cmaps()
    cmaps = {'R':_cmaps['flow'],
             'RIE':_cmaps['flow'],
             'Qv':_cmaps['flow'],
             'Inf':_cmaps['flow'],
             'ET':_cmaps['flow_v'],
             'Tpgw':_cmaps['flow_v'],
             'VSA':'Blues'}
    for v in vars:
        print(v)
        #values = sample_histograms(x=_xs, y=_ys, var=v)
        fmed = pos_folder + "/annual_{}_Mean.asc".format(v)
        frng = pos_folder +  "/annual_{}_Range_90.asc".format(v)
        func = pos_folder + "/annual_{}_uncertainty.asc".format(v)
        files_lst = [fmed, frng, func]
        maps_lst = list()
        for f in files_lst:
            meta, rmap = inp.asc_raster(file=f, dtype='float32')
            rmap = rmap[600:1200, 500:900]
            maps_lst.append(rmap.copy())
        meds = list()
        rngs = list()
        for i in range(len(labels)):
            lcl_y = _ys[i]
            lcl_x = _xs[i]
            lcl_med = maps_lst[0][lcl_y][lcl_x]
            meds.append(lcl_med)
            lcl_rng = maps_lst[1][lcl_y][lcl_x]
            rngs.append(lcl_rng)
        rngs = np.array(rngs)
        #
        # get values
        #
        fig = plt.figure(figsize=(10, 6))  # Width, Height
        gs = mpl.gridspec.GridSpec(6, 9, wspace=0.8, hspace=0.6)
        fig.suptitle('{} uncertainty'.format(v))
        #
        plt.subplot(gs[:4, :3])
        im = plt.imshow(maps_lst[0], cmap=cmaps[v], vmin=0, vmax=1600)
        plt.title('Median ({})'.format(units[v]))
        for p in range(len(_xs)):
            plt.plot(_xs[p], _ys[p], '.', color=colors[v])
            plt.text(_xs[p] + 10, _ys[p] + 10, s=labels[p], color=colors[v])
        plt.colorbar(im, shrink=0.4)
        plt.axis('off')
        #
        plt.subplot(gs[:4, 3:6])
        im = plt.imshow(maps_lst[1], cmap=cmaps[v], vmin=0, vmax=np.max(maps_lst[1]))
        plt.title('90% range ({})'.format(units[v]))
        for p in range(len(_xs)):
            plt.plot(_xs[p], _ys[p], '.', color=colors[v])
            plt.text(_xs[p] + 10, _ys[p] + 10, s=labels[p], color=colors[v])
        plt.colorbar(im, shrink=0.4)
        plt.axis('off')
        #
        plt.subplot(gs[:4, 6:9])
        im = plt.imshow(maps_lst[2], cmap='Greys', vmin=0, vmax=100)
        plt.title('uncertainty (%)')
        for p in range(len(_xs)):
            plt.plot(_xs[p], _ys[p], '.', color='black')
            plt.text(_xs[p] + 10, _ys[p] + 10, s=labels[p], color='black')
        plt.colorbar(im, shrink=0.4)
        plt.axis('off')
        #
        plt.subplot(gs[4:, :3])
        plt.bar(labels, meds, yerr=rngs/2, color='tab:grey')
        plt.ylabel(units[v])
        #
        filename = '{}_uncertainty'.format(v)
        if show:
            plt.show()
            plt.close(fig)
        else:
            filepath = folder + '/' + filename + '.png'
            plt.savefig(filepath, dpi=400)
            plt.close(fig)


def view_ensemble_q(calibfolder, gluefolder, outputfolder, show=True):
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    #
    folder = outputfolder
    #
    f_ensemble_et = gluefolder + "/ensemble_et.txt"
    f_ensemble_q = gluefolder + "/ensemble_q.txt"
    f_global_et = calibfolder + "/MLM/full_period/osa_zmaps/analyst_sim_series.txt"
    #
    #
    q_df = pd.read_csv(f_ensemble_q, sep=';', parse_dates=['Date'])
    q_obs_df = pd.read_csv(f_global_et, sep=';', parse_dates=['Date'])
    # print(et_df.head().to_string())
    fig = plt.figure(figsize=(16, 2.5))  # Width, Height
    plt.fill_between(x=q_df['Date'], y1=q_df['Lo_5'], y2=q_df['Hi_95'],
                     color='silver')
    plt.plot(q_df['Date'], q_df['Mid_50'], 'tab:blue')
    plt.plot(q_obs_df['Date'], q_obs_df['Qobs'], 'k.')
    plt.xlim((q_df['Date'].values[0], q_df['Date'].values[-1]))
    plt.ylim((0.001, 35))
    plt.yscale('log')
    plt.grid(True)
    # plt.ylabel('mm')
    filename = 'q_series'
    if show:
        plt.show()
        plt.close(fig)
    else:
        filepath = folder + '/' + filename + '.png'
        plt.savefig(filepath, dpi=400)
        plt.close(fig)


def view_ensemble_et(calibfolder, gluefolder, outputfolder, show=True):
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    #
    folder = outputfolder
    #
    f_ensemble_et = gluefolder + "/ensemble_et.txt"
    f_ensemble_q = gluefolder + "/ensemble_q.txt"
    f_global_et = calibfolder + "/MLM/full_period/osa_zmaps/analyst_sim_series.txt"
    #
    #
    et_df = pd.read_csv(f_ensemble_et, sep=';', parse_dates=['Date'])
    et_obs_df = pd.read_csv(f_global_et, sep=';', parse_dates=['Date'])
    #print(et_df.head().to_string())
    fig = plt.figure(figsize=(16, 2.5))  # Width, Height
    plt.fill_between(x=et_df['Date'], y1=et_df['Lo_5'], y2=et_df['Hi_95'],
                     color='silver')
    plt.plot(et_df['Date'], et_df['Mid_50'], 'tab:red')
    plt.plot(et_obs_df['Date'], et_obs_df['ETobs'], 'ko')
    plt.xlim((et_df['Date'].values[0], et_df['Date'].values[-1]))
    plt.ylim((0, 6))
    plt.grid(True)
    #plt.ylabel('mm')
    filename = 'et_series'
    if show:
        plt.show()
        plt.close(fig)
    else:
        filepath = folder + '/' + filename + '.png'
        plt.savefig(filepath, dpi=400)
        plt.close(fig)


def view_et_pannel(folder, calibfolder, show=False):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import inp
    from hydrology import map_back
    from visuals import _custom_cmaps
    #
    folder = folder
    _cmaps = _custom_cmaps()
    #
    date = '2014-07-07'
    etobs_sebal_raster_f = "C:/000_myFiles/myDrive/Plans3/pardinho/datasets/observed/etpat/calib_etpat_{}.asc".format(date)
    etobs_sampled_zmap_f = "C:/000_myFiles/myDrive/Plans3/pardinho/datasets/observed/etpat/zmap_etpat_{}.txt".format(date)
    ftwi = "C:/000_myFiles/myDrive/Plans3/pardinho/datasets/observed/calib_twi_window.asc"
    fshru = "C:/000_myFiles/myDrive/Plans3/pardinho/datasets/observed/calib_shru_window.asc"
    f_etsim_raster = calib_folder + "/full_period/sim_ET/raster_ET_{}.asc".format(date)
    #
    #
    # import stuff
    meta, twi = inp.asc_raster(file=ftwi, dtype='float32')
    meta, shru = inp.asc_raster(file=fshru, dtype='float32')
    meta, et_sebal = inp.asc_raster(file=etobs_sebal_raster_f, dtype='float32')
    meta, et_sim = inp.asc_raster(file=f_etsim_raster, dtype='float32')
    zmap_et, twi_bins, shru_bins = inp.zmap(file=etobs_sampled_zmap_f)
    et_sebal_sampled = map_back(zmatrix=zmap_et, a1=twi, a2=shru, bins1=twi_bins, bins2=shru_bins)
    #
    #
    v_max = 2
    fig = plt.figure(figsize=(16, 5))  # Width, Height
    gs = mpl.gridspec.GridSpec(1, 4, wspace=0.1, hspace=0.1)
    plt.subplot(gs[0, 0])
    im = plt.imshow(et_sebal, _cmaps['flow_v'], vmin=0, vmax=v_max)
    plt.colorbar(im, shrink=0.4)
    plt.axis('off')
    #
    plt.subplot(gs[0, 1])
    im = plt.imshow(et_sebal_sampled, _cmaps['flow_v'], vmin=0, vmax=v_max)
    plt.colorbar(im, shrink=0.4)
    plt.axis('off')
    #
    plt.subplot(gs[0, 2])
    im = plt.imshow(et_sim, _cmaps['flow_v'], vmin=0, vmax=v_max)
    plt.colorbar(im, shrink=0.4)
    plt.axis('off')
    #
    plt.subplot(gs[0, 3])
    im = plt.imshow(et_sebal_sampled - et_sim, 'seismic_r', vmin=-v_max, vmax=v_max)
    plt.colorbar(im, shrink=0.4)
    plt.axis('off')
    #
    #
    filename = 'et_pannel'
    if show:
        plt.show()
        plt.close(fig)
    else:
        filepath = folder + '/' + filename + '.png'
        plt.savefig(filepath, dpi=400)
        plt.close(fig)


def view_pre_pos(folder, pre_folder, pos_folder, show=True):
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    fpos = "{}\series_ensemble.txt".format(pos_folder)
    fpre = "{}\series_ensemble.txt".format(pre_folder)
    pos_df = pd.read_csv(fpos, sep=';', parse_dates=['Date'])
    pre_df = pd.read_csv(fpre, sep=';', parse_dates=['Date'])
    print(pos_df.head().to_string())
    vars = ['Prec','TF', 'ET', 'Evc', 'Evs', 'Tpun', 'Tpgw', 'R', 'RIE', 'RSE', 'Inf', 'Qv', 'Q', 'Qb', 'Qs']
    pre_50 = list()
    pos_50 = list()
    pre_rng = list()
    pos_rng = list()
    for v in vars:
        print(v)
        if v == 'Prec':
            pre_50.append(365 * np.sum(pre_df['Prec'.format(v)].values) / len(pre_df))
            pos_50.append(365 * np.sum(pos_df['Prec'.format(v)].values) / len(pre_df))
            _lo = 365 * np.sum(pre_df['Prec'.format(v)].values) / len(pre_df)
            _hi = 365 * np.sum(pre_df['Prec'.format(v)].values) / len(pre_df)
            pre_rng.append(_hi - _lo)
            _lo = 365 * np.sum(pos_df['Prec'.format(v)].values) / len(pre_df)
            _hi = 365 * np.sum(pos_df['Prec'.format(v)].values) / len(pre_df)
            pos_rng.append(_hi - _lo)
        else:
            pre_50.append(365 * np.sum(pre_df['{}_50'.format(v)].values) / len(pre_df))
            pos_50.append(365 * np.sum(pos_df['{}_50'.format(v)].values) / len(pre_df))
            _lo = 365 * np.sum(pre_df['{}_05'.format(v)].values) / len(pre_df)
            _hi = 365 * np.sum(pre_df['{}_95'.format(v)].values) / len(pre_df)
            pre_rng.append(_hi - _lo)
            _lo = 365 * np.sum(pos_df['{}_05'.format(v)].values) / len(pre_df)
            _hi = 365 * np.sum(pos_df['{}_95'.format(v)].values) / len(pre_df)
            pos_rng.append(_hi - _lo)
    labels = vars
    x = np.arange(len(labels))  # the label locations
    width = 0.3  # the width of the bars
    fig = plt.figure(figsize=(10, 4))  # Width, Height
    plt.subplot(111)
    plt.bar(x - (width / 2), pre_50, width, yerr=np.array(pre_rng) / 2, label='Pre-development', color='tab:green')
    plt.bar(x + (width / 2), pos_50, width, yerr=np.array(pos_rng) / 2, label='Post-development', color='tab:blue')
    #
    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel('mm')
    plt.xticks(x, vars)
    plt.grid(True, axis='y')
    plt.legend()
    if show:
        plt.show()
        plt.close(fig)
    else:
        filepath = folder +  '\Flows_prepost.png'.format(v)
        plt.savefig(filepath, dpi=400)
        plt.close(fig)
    #
    #
    vars = ['Cpy', 'Sfs', 'Unz']
    pre_50 = list()
    pos_50 = list()
    pre_rng = list()
    pos_rng = list()
    for v in vars:
        print(v)
        pre_50.append(np.mean(pre_df['{}_50'.format(v)].values))
        pos_50.append(np.mean(pos_df['{}_50'.format(v)].values))
        _lo = np.mean(pre_df['{}_05'.format(v)].values)
        _hi = np.mean(pre_df['{}_95'.format(v)].values)
        pre_rng.append(_hi - _lo)
        _lo = np.mean(pos_df['{}_05'.format(v)].values)
        _hi = np.mean(pos_df['{}_95'.format(v)].values)
        pos_rng.append(_hi - _lo)
    labels = vars
    x = np.arange(len(labels))  # the label locations
    width = 0.3  # the width of the bars
    fig = plt.figure(figsize=(3.5, 4))  # Width, Height
    plt.subplot(111)
    plt.bar(x - width / 2, pre_50, width, yerr=np.array(pre_rng) / 2, label='Pre-devel.', color='tab:green')
    plt.bar(x + width / 2, pos_50, width, yerr=np.array(pos_rng) / 2, label='Post-devel.', color='tab:blue')
    #
    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel('mm')
    plt.xticks(x, vars)
    plt.grid(True, axis='y')
    plt.legend()
    if show:
        plt.show()
        plt.close(fig)
    else:
        filepath = folder + '\Stocks_prepost.png'.format(v)
        plt.savefig(filepath, dpi=400)
        plt.close(fig)
    #
    # vars
    vars = ['Cpy', 'Sfs', 'Unz', 'TF', 'ET', 'Evc', 'Evs', 'Tpun', 'Tpgw', 'R', 'Inf', 'Qv', 'Q', 'Qb']
    for v in vars:
        print(v)
        fig = plt.figure(figsize=(16, 2.5))  # Width, Height
        plt.fill_between(x=pos_df['Date'],
                         y1=pos_df['{}_05'.format(v)],
                         y2=pos_df['{}_95'.format(v)],
                         color='tab:green',
                         alpha=0.4,
                         edgecolor='none')
        plt.fill_between(x=pre_df['Date'],
                         y1=pre_df['{}_05'.format(v)],
                         y2=pre_df['{}_95'.format(v)],
                         color='tab:blue',
                         alpha=0.4,
                         edgecolor='none')
        plt.plot(pos_df['Date'], pos_df['{}_50'.format(v)], 'tab:blue', label='post-development')
        plt.plot(pre_df['Date'], pre_df['{}_50'.format(v)], 'tab:green', label='pre-development')
        plt.xlim((pre_df['Date'].values[0], pre_df['Date'].values[-1]))
        plt.legend(loc='upper right')
        plt.title('{}'.format(v))
        if show:
            plt.show()
            plt.close(fig)
        else:
            filepath = folder + '\{}_series_prepost.png'.format(v)
            plt.savefig(filepath, dpi=400)
            plt.close(fig)


def main(folder, infolder, projectfolder):
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
    series_lst = step00_set_input_series(folder=folder,
                                  infolder=infolder,
                                  projectfolder=projectfolder)
    lbls = ['s1', 's2', 's3', 'aoi']
    mapvars = 'Qv-R'
    #
    #
    # loop in sets
    for i in len(folder_lst):
        folder = folder_lst[i]
        fseries = series_lst[i]
        #
        # 01 SEARCH MODELS
        calib_folder = step01_search_models(folder=folder,
                                            projectfolder=projectfolder,
                                            fseries=fseries)
        #
        # 02 SELECT BEHAVIOURAL MODELS
        glue_folder = step02_select_models(folder=folder,
                                           calibfolder=calib_folder,
                                           projectfolder=projectfolder)
        #
        # 03 MAP HYDRO PROCESSES FOR AOI PRE and POS
        map_folders = step03_map_processes(folder=folder,
                                           gluefolder=glue_folder,
                                           projectfolder=projectfolder,
                                           fseries=fseries, vars=mapvars)
        #
        # 04 MAP ASLA PROCESSES FOI AOI PRE AND POS
        asla_folders = asla_folders = step04_map_asla(folder=folder,
                                                      prefolder=map_folders['Pre_folder'],
                                                      posfolder=map_folders['Pos_folder'],
                                                      projectfolder=projectfolder,
                                                      fseries=fseries)
        #
        # 05 COMPUTE ANOMALY
        step05_compute_anomaly(folder='C:/bin',
                               hy_prefolder=map_folders['Pre_folder'],
                               hy_posfolder=map_folders['Pos_folder'],
                               asla_prefolder=asla_folders['Pre_folder'],
                               asla_posfolder=asla_folders['Pos_folder'],
                               mapvars=mapvars)
        #
        # 06 COMPUTE UNCERTAINTY
        step06_compute_uncertainty(folder='C:/bin',
                                   hy_prefolder=map_folders['Pre_folder'],
                                   hy_posfolder=map_folders['Pos_folder'],
                                   mapvars=mapvars)
        #


sets_lst = ['s1', 's2', 's3']
for s in sets_lst:
    folder = 'C:/bin/pardinho/produtos_v2/run_02a/{}'.format(s)
    calib_folder = '{}/search'.format(folder)
    glue_folder = '{}/select'.format(folder)
    pos_folder = '{}/pos_bat'.format(folder)
    pre_folder = '{}/pre_bat'.format(folder)
    fcar_index = '{}/aoi_car_full_indices.txt'.format(folder)
    step08_priority_index(folder=folder, fcar_index=fcar_index, show=False)
