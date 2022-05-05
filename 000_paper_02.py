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


def step03_map_processes(folder, gluefolder, projectfolder, fseries,
                         vars='R-Qv-D-VSA',
                         pre=True,
                         pos=True):
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
    out_dct = dict()
    # pre
    if pre:
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
                                ensemble=False,
                                stats=True,
                                annualize=True,
                                stats_raster=False,
                                label='PRE',
                                folder=folder)
        out_dct['Pre_folder'] = pre_folder['Folder']
    #
    #
    # pos
    if pos:
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
                                ensemble=False,
                                stats=True,
                                annualize=True,
                                stats_raster=False,
                                label='POS',
                                folder=folder)
        out_dct['Pos_folder'] = pos_folder['Folder']
    return out_dct


def step03_map_raster(folder, mapvar='D-R-Evs', pre=True, pos=True):
    import inp, out
    from visuals import plot_map_view
    from hydrology import map_back
    from tui import status
    from backend import get_mapid
    mapvars = mapvar.split('-')
    ftwi = 'C:/bin/pardinho/produtos_v2/inputs/pardinho/datasets/observed/aoi_twi.asc'
    fpos_shru = 'C:/bin/pardinho/produtos_v2/inputs/pardinho/datasets/observed/aoi_shru.asc'
    fpre_shru = "C:/bin/pardinho/produtos_v2/inputs/pardinho/datasets/projected/scn__lulc_predc/aoi_shru.asc"
    # import maps
    status('importing maps')
    meta, twi = inp.asc_raster(file=ftwi, dtype='float32')
    stats = ['Range_90']
    for s in stats:
        if pre:
            for v in mapvars:
                meta, shru = inp.asc_raster(file=fpre_shru, dtype='float32')
                lcl_file = '{}/pre_bat/zmaps/annual_{}_{}.txt'.format(folder, v, s)
                status('loading zmap of {} {}'.format(v, s))
                lcl_zmap, twi_bins, shru_bins = inp.zmap(file=lcl_file)
                status('mapping back {} {}'.format(v, s))
                lcl_raster = map_back(zmatrix=lcl_zmap,
                                      a1=twi,
                                      a2=shru,
                                      bins1=twi_bins,
                                      bins2=shru_bins)
                status('exporting raster of {} {}'.format(v, s))
                out.asc_raster(array=lcl_raster,
                               meta=meta,
                               folder='{}/pre_bat/rasters'.format(folder),
                               filename='annual_{}_{}'.format(v, s),
                               dtype='float32')
                status('plotting view of {} {}'.format(v, s))
                mapid = get_mapid(v)
                plot_map_view(lcl_raster, meta,
                              ranges=[0, np.max(lcl_raster)],
                              mapid=mapid,
                              mapttl='{} {}'.format(v, s),
                              filename='annual_{}_{}'.format(v, s),
                              folder='{}/pre_bat/rasters'.format(folder))

        if pos:
            for v in mapvars:
                meta, shru = inp.asc_raster(file=fpos_shru, dtype='float32')
                lcl_file = '{}/pos_bat/zmaps/annual_{}_{}.txt'.format(folder, v, s)
                status('loading zmap of {} {}'.format(v, s))
                lcl_zmap, twi_bins, shru_bins = inp.zmap(file=lcl_file)
                status('mapping back {} {}'.format(v, s))
                lcl_raster = map_back(zmatrix=lcl_zmap,
                                      a1=twi,
                                      a2=shru,
                                      bins1=twi_bins,
                                      bins2=shru_bins)
                status('exporting raster of {} {}'.format(v, s))
                out.asc_raster(array=lcl_raster,
                               meta=meta,
                               folder='{}/pos_bat/rasters'.format(folder),
                               filename='annual_{}_{}'.format(v, s),
                               dtype='float32')
                status('plotting view of {} {}'.format(v, s))
                mapid = get_mapid(v)
                plot_map_view(lcl_raster, meta,
                              ranges=[0, np.max(lcl_raster)],
                              mapid=mapid,
                              mapttl='{} {}'.format(v, s),
                              filename='annual_{}_{}'.format(v, s),
                              folder='{}/pos_bat/rasters'.format(folder))


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
    frunoff = prefolder + "/annual_R_Mean.asc"
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
    frunoff = posfolder + "/annual_R_Mean.asc"
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


def step05_compute_anomaly(folder, hy_prefolder, hy_posfolder, asla_prefolder, asla_posfolder,
                           mapvars='Qv-R', asla=True):
    import os
    import numpy as np
    from visuals import plot_map_view
    from backend import create_rundir
    import inp, out
    outfolder = create_rundir(label='Anomaly', wkplc=folder)
    #
    stats = ['Mean']
    # variables
    mapvars = mapvars.split('-')
    # POS
    pos_folder = hy_posfolder
    pos_all_files = os.listdir(pos_folder)
    # PRE
    pre_folder = hy_prefolder
    pre_all_files = os.listdir(pre_folder)
    # stats loop
    for s in stats:
        for i in range(len(mapvars)):
            # find file path
            lcl_pre_file_path = '{}/rasters/annual_{}_Mean.asc'.format(pre_folder, mapvars[i])
            # find file path
            lcl_pos_file_path = '{}/rasters/annual_{}_Mean.asc'.format(pos_folder, mapvars[i])
            print(lcl_pre_file_path)
            print(lcl_pos_file_path)
            status('loading raster maps')
            meta, pre_map = inp.asc_raster(file=lcl_pre_file_path, dtype='float32')
            meta, pos_map = inp.asc_raster(file=lcl_pos_file_path, dtype='float32')
            meta['NODATA_value'] = -99999
            #
            # compute anomaly
            anom_map = pos_map - pre_map
            #
            lcl_filename = 'annual_{}_{}_anomaly'.format(mapvars[i], s)
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
                          mapttl='{} {} anomaly'.format(mapvars[i], s),
                          nodata=-99999,
                          show=False)
            view_anomaly(anomfolder=outfolder,
                         premap=pre_map,
                         posmap=pos_map,
                         anommap=anom_map,
                         show=False,
                         mapvar=mapvars[i])
            if mapvars[i] == 'R':
                pre_map = pre_map / 2000
                pos_map = pos_map / 2000
                anom_map = pos_map - pre_map
                lcl_filename = 'annual_CR_{}_anomaly'.format(s)
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
                              mapttl='CR {} anomaly'.format(s),
                              nodata=-99999,
                              show=False)
                view_anomaly(anomfolder=outfolder,
                             premap=pre_map,
                             posmap=pos_map,
                             anommap=anom_map,
                             show=False,
                             mapvar=mapvars[i])

    #
    #
    # ASLA
    if asla:
        mapvars = ['c_usle', 'k_usle', 's_rusle', 'l_rusle'] #'asl', 'asllog', 'pload', 'nload',
        # POS
        pos_folder = asla_posfolder
        pos_all_files = os.listdir(pos_folder)
        # PRE
        pre_folder = asla_prefolder
        pre_all_files = os.listdir(pre_folder)
        for i in range(len(mapvars)):
            lcl_pre_file_path = '{}/{}.asc'.format(pre_folder, mapvars[i])
            lcl_pos_file_path = '{}/{}.asc'.format(pos_folder, mapvars[i])
            print(lcl_pre_file_path)
            print(lcl_pos_file_path)
            status('loading raster maps')
            meta, pre_map = inp.asc_raster(file=lcl_pre_file_path, dtype='float32')
            meta, pos_map = inp.asc_raster(file=lcl_pos_file_path, dtype='float32')
            meta['NODATA_value'] = -99999
            #
            # compute anomaly
            anom_map = pos_map - pre_map
            #
            lcl_filename = 'annual_{}_anomaly'.format(mapvars[i])
            status('exporting raster map')
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
                          mapttl='{} anomaly'.format(mapvars[i]),
                          nodata=-99999)
            view_anomaly(anomfolder=outfolder,
                         premap=pre_map,
                         posmap=pos_map,
                         anommap=anom_map,
                         show=False,
                         mapvar=mapvars[i])
    return outfolder


def step06_compute_uncertainty(folder, hy_prefolder, hy_posfolder, mapvars='Qv-R'):
    import os
    import numpy as np
    from visuals import plot_map_view
    from backend import create_rundir
    #
    # variables
    mapvars = mapvars.split('-')
    #
    # PRE and POS unc
    pos_folder = hy_posfolder
    pre_folder = hy_prefolder
    folders = [pre_folder, pos_folder,]
    unc_folder = create_rundir('Uncertainty', folder)
    pos_outfolder = create_rundir('POS_Uncertainty', unc_folder)
    pre_outfolder = create_rundir('PRE_Uncertainty', unc_folder)
    outfolders = [pre_outfolder, pos_outfolder]
    # variable loop
    for i in range(len(mapvars)):
        print(mapvars[i])
        # pre and post analysis
        unc_maps_lst = list()
        rng_maps_lst = list()
        mean_maps_lst = list()
        for j in range(len(folders)):
            range_fpath = '{}/rasters/annual_{}_Range_90.asc'.format(folders[j], mapvars[i])
            mean_fpath = '{}/rasters/annual_{}_Mean.asc'.format(folders[j], mapvars[i])
            # load
            print('loading raster maps...')
            meta, range_map = inp.asc_raster(file=range_fpath, dtype='float32')
            meta, mean_map = inp.asc_raster(file=mean_fpath, dtype='float32')
            meta['NODATA_value'] = -99999
            #
            """
            '+ 1' to avoid zero division and '!= 0' to set zero where mean is zero 
            """
            print('computing uncertainty')
            unc_map = (100 * (range_map + 1) / (mean_map + 1)) * (mean_map != 0) #

            unc_maps_lst.append(unc_map.copy())
            rng_maps_lst.append(range_map.copy())
            mean_maps_lst.append(mean_map.copy())

            lcl_filename = 'annual_{}_uncertainty'.format(mapvars[i])
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
                          mapttl='{} uncertainty'.format(mapvars[i]),
                          nodata=-99999,
                          show=False)
        #
        # compute avg unc
        avg_unc = (unc_maps_lst[0] + unc_maps_lst[1]) / 2
        #
        lcl_filename = 'avg_{}_uncertainty'.format(mapvars[i])
        print('exporting...')
        out_file = out.asc_raster(avg_unc, meta, folder=unc_folder, filename=lcl_filename, dtype='float32')
        rng = (0, np.percentile(avg_unc, q=95))
        plot_map_view(map=avg_unc,
                      meta=meta,
                      ranges=rng,
                      mapid='unc',
                      filename=lcl_filename,
                      folder=unc_folder,
                      metadata=False,
                      mapttl='{} average uncertainty'.format(mapvars[i]),
                      nodata=-99999,
                      show=False)
        #
        # plot pannel
        view_uncertainty(uncfolder=unc_folder,
                         premean=mean_maps_lst[0],
                         posmean=mean_maps_lst[1],
                         prerng=rng_maps_lst[0],
                         posrng=rng_maps_lst[1],
                         preunc=unc_maps_lst[0],
                         posunc=unc_maps_lst[1],
                         uncmap=avg_unc,
                         mapvar=mapvars[i],
                         show=False)


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


def step09_priority_index(folder, fcar_index, show=False, wkpl=False):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import shapefile

    if wkpl:
        from backend import create_rundir
        folder = create_rundir(label='step09', wkplc=folder)

    investment = 200000 # R$

    ppsa = 200 # R$ / ha

    max_area = investment / ppsa

    print('area : {}'.format(max_area))

    _df = pd.read_csv(fcar_index, sep=';')

    vars_aa = ['R', 'Qv', 'Inf', 'Qv', 'asl', 'nload', 'pload', 'ET', 'Tpgw'] #, 'Inf', 'Qv', 'asl', 'nload', 'pload']
    vars_un = ['R', 'Qv', 'Inf', 'Qv', 'R', 'R', 'R', 'ET', 'Tpgw'] #, 'Inf', 'Qv', 'R', 'R', 'R']
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

    view_rank_diff(folder=folder, policy_df=_df, show=show)

    fout = '{}/priority_policy.txt'.format(folder)
    _df.to_csv(fout, sep=';', index=False)

    # load shapefile
    fbasin = r"C:\000_myFiles\myDrive\gis\pnh\misc\aoi_basin.shp"
    sf = shapefile.Reader(fbasin)
    #
    # plot
    _p_cmp = 'YlOrRd_r'

    fig = plt.figure(figsize=(12, 5))  # Width, Height
    fig.suptitle('RS{} | RS{}/ha | {:.2f}ha'.format(investment, ppsa, max_area))
    gs = mpl.gridspec.GridSpec(3, 9, wspace=0.2, hspace=0.1)
    #
    # IP0
    ax = fig.add_subplot(gs[:4, :3])
    _df0 = _df.query('IP_0_rnk > 0')
    print(len(_df0))
    print(_df0['area'].sum())
    _long = np.append(_df0['long'].values, [386300, 386300])
    _lat = np.append(_df0['lat'].values, [6752259, 6752259])
    _z = np.append(_df0['IP_0_rnk'].values, [_rnk_min, _rnk_max])
    plt.scatter(_long, _lat, c=_z, cmap=_p_cmp, marker='.')
    cbar = plt.colorbar(shrink=0.4)
    cbar.ax.invert_yaxis()
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
    print(len(_df1))
    print(_df1['area'].sum())
    _long = np.append(_df1['long'].values, [386300, 386300])
    _lat = np.append(_df1['lat'].values, [6752259, 6752259])
    _z = np.append(_df1['IP_1_rnk'].values, [_rnk_min, _rnk_max])
    plt.scatter(_long, _lat, c=_z, cmap=_p_cmp, marker='.')
    cbar = plt.colorbar(shrink=0.4)
    cbar.ax.invert_yaxis()
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
    #
    # difference
    ax = fig.add_subplot(gs[:4, 6:9])
    _df_aux = _df1.query('IP_0_rnk == 0')
    print(_df_aux.head(10).to_string())
    plt.scatter(_df_aux['long'], _df_aux['lat'], c='tab:green', marker='.')
    _df1 = _df1.query('IP_0_rnk > 0')
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
    plt.title('rank difference | {} new'.format(len(_df_aux)))
    #
    filename = 'priority_ranks'
    if show:
        plt.show()
        plt.close(fig)
    else:
        filepath = folder + '/' + filename + '.png'
        plt.savefig(filepath, dpi=400)
        plt.close(fig)


def step10_compare_policies(folder, show=False, wkpl=False):
    """
    comparing datasets policies
    :param folder:
    :param show:
    :return:
    """
    import matplotlib.pyplot as plt
    import shapefile
    # get filepaths first
    ffull_s1 = '{}/ds0/aoi_car_full_indices.txt'.format(folder)
    fpolicy_ds0 = '{}/ds0/priority_policy.txt'.format(folder)
    fpolicy_dsA = '{}/dsA/priority_policy.txt'.format(folder)
    fpolicy_dsB = '{}/dsB/priority_policy.txt'.format(folder)

    if wkpl:
        from backend import create_rundir
        folder = create_rundir(label='step10', wkplc=folder)
    # load policies dataframes
    full_df = pd.read_csv(ffull_s1, sep=';')
    policy_ds0_df = pd.read_csv(fpolicy_ds0, sep=';')
    policy_ds0_df = policy_ds0_df.query('IP_1_rnk > 0')
    policy_ds0_s_df = policy_ds0_df[['id_car', 'lat', 'long', 'area']]

    policy_dsA_df = pd.read_csv(fpolicy_dsA, sep=';')
    policy_dsA_df = policy_dsA_df.query('IP_1_rnk > 0')
    policy_dsA_s_df = policy_dsA_df[['id_car', 'lat', 'long', 'area']]

    policy_dsB_df = pd.read_csv(fpolicy_dsB, sep=';')
    policy_dsB_df = policy_dsB_df.query('IP_1_rnk > 0')
    policy_dsB_s_df = policy_dsB_df[['id_car', 'lat', 'long', 'area']]

    # create dsA outer join
    dfA0 = pd.merge(policy_dsA_s_df, policy_ds0_s_df,
                    how='outer',
                    left_on='id_car',
                    right_on='id_car',
                    suffixes=('_A', '_0'))
    dfA0['Kind'] = ''
    for i in range(len(dfA0)):
        if pd.isna(dfA0['area_A'].values[i]):
            dfA0['Kind'].values[i] = 'Missing'
        elif pd.notna(dfA0['area_A'].values[i]) and pd.isna(dfA0['area_0'].values[i]):
            dfA0['Kind'].values[i] = 'Extra'
        else:
            dfA0['Kind'].values[i] = 'Shared'
    # create dsB outer join
    dfB0 = pd.merge(policy_dsB_s_df, policy_ds0_s_df,
                    how='outer',
                    left_on='id_car',
                    right_on='id_car',
                    suffixes=('_B', '_0'))
    dfB0['Kind'] = ''
    for i in range(len(dfB0)):
        if pd.isna(dfB0['area_B'].values[i]):
            dfB0['Kind'].values[i] = 'Missing'
        elif pd.notna(dfB0['area_B'].values[i]) and pd.isna(dfB0['area_0'].values[i]):
            dfB0['Kind'].values[i] = 'Extra'
        else:
            dfB0['Kind'].values[i] = 'Shared'
    #
    #
    # create summary dataframe

    dfA0_shared = dfA0.query('Kind == "Shared"')
    dfB0_shared = dfB0.query('Kind == "Shared"')
    dfA0_extra = dfA0.query('Kind == "Extra"')
    dfB0_extra = dfB0.query('Kind == "Extra"')
    dfA0_missing = dfA0.query('Kind == "Missing"')
    dfB0_missing = dfB0.query('Kind == "Missing"')


    summary_df = pd.DataFrame({'Policy': ['ds0', 'dsA', 'dsB'],
                               'data_years': [16, 8, 4],
                               'data_diff': [0, 8, 12],
                               'n': [len(policy_ds0_s_df), len(policy_dsA_s_df), len(policy_dsB_s_df)],
                               'n_shared':[0, len(dfA0_shared), len(dfB0_shared)],
                               'n_extra':[0, len(dfA0_extra), len(dfB0_extra)],
                               'n_missing':[0, len(dfA0_missing), len(dfB0_missing)],
                               'psa': [200, 200, 200],
                               'area': [policy_ds0_s_df['area'].sum(),
                                        dfA0_shared['area_A'].sum() + dfA0_extra['area_A'].sum(),
                                        dfB0_shared['area_B'].sum() + dfB0_extra['area_B'].sum()],
                               'area_missing': [0,
                                                dfA0_missing['area_0'].sum(),
                                                dfB0_missing['area_0'].sum()]
                               })
    summary_df['ann_cost_est'] = summary_df['psa'] * summary_df['area']
    summary_df['ann_cost_miss'] = summary_df['psa'] * summary_df['area_missing']
    summary_df['ann_cost'] = summary_df['ann_cost_est'] + summary_df['ann_cost_miss']
    summary_df['ann_cost_avoid'] = summary_df['ann_cost'] - summary_df['ann_cost_est']
    print(summary_df.to_string())
    fout = '{}/summary_policies.txt'.format(folder)
    summary_df.to_csv(fout, sep=';', index=False)

    # compare this variables:
    variables = ['R', 'Qv', 'Inf', 'ET', 'asl', 'nload', 'pload']
    out_df = pd.DataFrame({'Var':variables})
    out_df['ds0'] = 0.0
    out_df['dsA'] = 0.0
    out_df['ds0dsA_d'] = 0.0
    out_df['dsB'] = 0.0
    out_df['ds0dsB_d'] = 0.0
    # variable loop:
    for i in range(len(variables)):
        lcl_v = variables[i]
        out_df['ds0'].values[i] = policy_ds0_df['{}_aa_mean'.format(lcl_v)].mean()
        out_df['dsA'].values[i] = policy_dsA_df['{}_aa_mean'.format(lcl_v)].mean()
        out_df['dsB'].values[i] = policy_dsB_df['{}_aa_mean'.format(lcl_v)].mean()
    out_df['ds0dsA_d'] = out_df['ds0'] - out_df['dsA']
    out_df['ds0dsB_d'] = out_df['ds0'] - out_df['dsB']
    print(out_df.to_string())
    fout = '{}/compare_policies.txt'.format(folder)
    out_df.to_csv(fout, sep=';', index=False)
    # bar plot of variables
    labels = {'dsA':['ds0' ,'dsA', 'ds0 - dsA'],
              'dsB':['ds0' ,'dsB', 'ds0 - dsB']}
    for i in range(len(variables)):
        lcl_policies = ['dsA', 'dsB']
        lcl_max = 1.3 * np.max([out_df['ds0'].values[i],
                                out_df['dsA'].values[i],
                                out_df['ds0dsA_d'].values[i],
                                out_df['dsB'].values[i],
                                out_df['ds0dsB_d'].values[i]])
        lcl_min = 1.3 * np.min([out_df['ds0'].values[i],
                                out_df['dsA'].values[i],
                                out_df['ds0dsA_d'].values[i],
                                out_df['dsB'].values[i],
                                out_df['ds0dsB_d'].values[i]])
        if lcl_min >= 0:
            lcl_min = 0
        print('{}\nmax {}\n min {}\n\n'.format(variables[i], lcl_max, lcl_min))
        for j in range(len(lcl_policies)):
            values1 = [out_df['ds0'].values[i],
                       out_df[lcl_policies[j]].values[i],
                       out_df['ds0{}_d'.format(lcl_policies[j])].values[i]]
            values1 = np.round(values1, 2)
            #values2 = (out_df['s3'].values[0], out_df['s3s1'].values[0], out_df['s3s1_d'].values[0])
            x = np.arange(len(labels[lcl_policies[j]]))  # the label locations
            width = 0.4  # the width of the bars

            fig, ax = plt.subplots(figsize=(5, 2.5))  # Width, Height
            p = ax.bar(x, values1, width, label='', color='tab:grey')
            ax.bar_label(p)
            #plt.bar(x + (width / 2), values2, width, label='ok', color='maroon')
            #
            # Add some text for labels, title and custom x-axis tick labels, etc.
            plt.ylabel('mm')
            plt.ylim((lcl_min, lcl_max))
            plt.xticks(x, labels[lcl_policies[j]])
            plt.title(variables[i] + ' {}'.format(lcl_policies[j]))
            plt.grid(True, axis='y')
            #plt.show()
            if show:
                plt.show()
                plt.close(fig)
            else:
                filepath = folder + '\{}_{}_compare.png'.format(variables[i], lcl_policies[j])
                plt.savefig(filepath, dpi=400)
                plt.close(fig)
    #
    #
    #
    # plot policies
    policies = [policy_ds0_df, dfA0, dfB0]
    labels = ['ds0', 'dsA', 'dsB']
    for i in range(0, len(policies)):
        # load shapefile
        fbasin = r"C:\000_myFiles\myDrive\gis\pnh\misc\aoi_basin.shp"
        sf = shapefile.Reader(fbasin)
        fig = plt.figure(figsize=(5, 6))  # Width, Height
        fig.suptitle(labels[i])
        ax = fig.add_subplot()
        if i == 0:
            plt.scatter(policies[i]['long'], policies[i]['lat'], c='tab:green', marker='o')
        elif i == 1:
            shared_df = policies[i].query('Kind == "Shared"')
            extra_df = policies[i].query('Kind == "Extra"')
            missing_df = policies[i].query('Kind == "Missing"')
            plt.scatter(shared_df['long_A'], shared_df['lat_A'], c='tab:green', marker='o', zorder=3)
            plt.scatter(extra_df['long_A'], extra_df['lat_A'], c='orange', marker='o', zorder=2)
            plt.scatter(missing_df['long_0'], missing_df['lat_0'], c='tab:red', marker='o', zorder=1)
        else:
            shared_df = policies[i].query('Kind == "Shared"')
            extra_df = policies[i].query('Kind == "Extra"')
            missing_df = policies[i].query('Kind == "Missing"')
            plt.scatter(shared_df['long_B'], shared_df['lat_B'], c='tab:green', marker='o', zorder=3)
            plt.scatter(extra_df['long_B'], extra_df['lat_B'], c='orange', marker='o', zorder=2)
            plt.scatter(missing_df['long_0'], missing_df['lat_0'], c='tab:red', marker='o', zorder=1)
        #
        # overlay shapefile
        patch = plt.Polygon(sf.shape(0).points, facecolor='none', edgecolor='black', linewidth=1, zorder=10)
        ax.add_patch(patch)
        ax.axis('scaled')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([360694, 385296])
        ax.set_ylim([6721596, 6752258])
        filename = '{}_policy'.format(labels[i])
        #show = True
        if show:
            plt.show()
            plt.close(fig)
        else:
            filepath = folder + '/' + filename + '.png'
            plt.savefig(filepath, dpi=400)
            plt.close(fig)
        #

def view_rank_diff_hist(folder='C:/bin/pardinho/produtos_v2/run_02a/ds0', show=False):
    from analyst import frequency
    fcar_index = r"C:\bin\pardinho\produtos_v2\run_02a\ds0\priority_policy.txt"
    _df = pd.read_csv(fcar_index, sep=';')
    _df1 = _df.query('IP_1_rnk > 0')
    _df1['IP_1_rnk_diff_abs'] = np.abs(_df1['IP_1_rnk_diff'].values)
    #
    # hist
    fig, ax = plt.subplots(figsize=(3, 2.5), tight_layout=True) # Width, Height
    n, bins, patches = plt.hist(_df1['IP_1_rnk_diff_abs'], 50, density=True,
                                facecolor='tab:grey', alpha=1)
    plt.xlim(0, 250)
    filename = 'policy_abschange_hist'
    if show:
        plt.show()
        plt.close(fig)
    else:
        filepath = folder + '/' + filename + '.png'
        plt.savefig(filepath, dpi=400)
        plt.close(fig)
    
    df_freq = frequency(dataframe=_df1, var_field='IP_1_rnk_diff_abs')
    fig, ax = plt.subplots(figsize=(3, 2.5), tight_layout=True) # Width, Height
    plt.plot(df_freq['Values'], df_freq['Percentiles'], 'tab:grey')
    plt.ylim(0, 100)
    plt.xlim(0, 250)
    filename = 'policy_abschange_cfc'
    if show:
        plt.show()
        plt.close(fig)
    else:
        filepath = folder + '/' + filename + '.png'
        plt.savefig(filepath, dpi=400)
        plt.close(fig)
    
    

def view_rank_diff(folder, policy_df, show=False):
    _df0 = policy_df.query('IP_0_rnk > 0')
    x0 = np.ones(len(_df0))
    _df1 = policy_df.query('IP_1_rnk > 0')
    x1 = np.ones(len(_df1)) * 2
    fig, ax = plt.subplots(figsize=(3, 2.5), tight_layout=True) # Width, Height
    for i in range(len(_df1)):
        _al = 0.9 * np.abs(_df1['IP_1_rnk_diff'].values[i]) / len(_df1)
        if _df1['IP_1_rnk_diff'].values[i] > 0 and _df1['IP_0_rnk'].values[i] != 0:
            _c = 'red'
        elif _df1['IP_0_rnk'].values[i] == 0:
            _df1['IP_0_rnk'].values[i] = len(_df1)
            _c = 'tab:green'
            _al = 0.9 * np.abs(len(_df1) - _df1['IP_1_rnk'].values[i]) / len(_df1)
        else:
            _c = 'blue'
        plt.plot([1, 2],
                 [_df1['IP_0_rnk'].values[i], _df1['IP_1_rnk'].values[i]],
                 color=_c,
                 alpha=_al)
    ax1 = ax.twinx()
    ax1.plot([3, 3], [0, len(_df1)])
    ax.set_ylim(0, len(_df1))
    ax1.set_ylim(0, len(_df1))
    ax.invert_yaxis()
    ax1.invert_yaxis()
    plt.xlim(1, 2)
    plt.xticks([1, 2], ['', ''])
    filename = 'rank_change'
    if show:
        plt.show()
        plt.close(fig)
    else:
        filepath = folder + '/' + filename + '.png'
        plt.savefig(filepath, dpi=400)
        plt.close(fig)


def view_obs_data_analyst(fseries, fetobs, folder='C:/bin/pardinho/produtos_v2', show=True):
    import matplotlib as mpl
    import resample
    _df = pd.read_csv(fseries, sep=';', parse_dates=['Date'])
    _df2 = pd.read_csv(fetobs, sep=';', parse_dates=['Date'])
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
    fig = plt.figure(figsize=(16, 7))  # Width, Height
    gs = mpl.gridspec.GridSpec(5, 9, wspace=1, hspace=0.8, left=0.05, bottom=0.05, top=0.95, right=0.95)
    order = ['Prec_Deodoro', 'Prec_Boqueirao', 'Prec_Herveiras',  'Q_StaCruz']
    names = {'Prec_Herveiras': 'Precip. Herveiras Station',
             'Prec_Boqueirao': 'Precip. Boqueirao Station',
             'Prec_Deodoro': 'Precip. Deodoro Station',
             'Q_StaCruz': 'Streamflow Santa Cruz Station'}
    count = 0
    for field in order:
        #
        # series plot
        plt.subplot(gs[count: count + 1, :6])
        #plt.title(names[field], loc='left')
        color = 'tab:grey'
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

    ax = plt.subplot(gs[count: count + 1, :6])

    ax1 = ax.twinx()
    color = 'tab:orange'
    ax1.plot(_df2['Date'], _df2['Temp'], color, zorder=10)
    ax1.set_xlim((_df['Date'].values[0], _df['Date'].values[-1]))
    ax1.set_ylim(0, 35)
    ax1.set_ylabel('°C')

    color = 'darkred'
    ax.plot(_df2['Date'], _df2['ETobs'], color, marker='.', zorder=0)
    ax.set_xlim((_df['Date'].values[0], _df['Date'].values[-1]))
    ax.set_ylabel('mm')
    ax.set_ylim(0, 10)

    # export
    if show:
        plt.show()
        plt.close(fig)
    else:
        filepath = folder + '/step01a__obs_data_all.png'
        plt.savefig(filepath, dpi=400)
        plt.close(fig)


def view_evolution_lspace(folder, calibfolder, gluefolder, show=True):
    full_f = calibfolder + '/generations/population.txt'
    pop_df = pd.read_csv(full_f, sep=';')
    behav_f = gluefolder + '/behavioural.txt'
    behav_df = pd.read_csv(behav_f, sep=';')
    select_f = gluefolder + '/selection.txt'
    select_df = pd.read_csv(select_f, sep=';')
    fig = plt.figure(figsize=(7, 7), )  # Width, Height
    plt.scatter(x=pop_df['L_ET'], y=pop_df['L_Q'], marker='.', c='tab:grey', alpha=0.4, edgecolors='none')
    plt.scatter(x=behav_df['L_ET'], y=behav_df['L_Q'], marker='.', c='black')
    plt.scatter(x=select_df['L_ET'], y=select_df['L_Q'], marker='.', c='orange')
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


def view_evolution_gens(folder, calibfolder, show=True):
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


def view_evolution_datasets(folder, show=True):
    fig = plt.figure(figsize=(7, 4), )  # Width, Height
    sets_lst = ['ds0', 'dsA', 'dsB']
    colors_dct = {'ds0': 'orange', 'dsA': 'silver', 'dsB': 'peru'}
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
    sets_lst = ['ds0', 'dsA', 'dsB']
    colors_dct = {'ds0': 'orange', 'dsA': 'silver', 'dsB': 'peru'}
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


def view_evolution_scatter(dir_glue, dir_calib, fparam, show=True):
    import matplotlib as mpl
    import inp
    from visuals import glue_scattergram

    def extract_ranges(fhydroparam):
        dct, hydroparam_df = inp.hydroparams(fhydroparam=fhydroparam)
        #
        # extract set range values
        out_dct = {'Params_df': hydroparam_df,
                   'm_rng': (dct['m']['Min'], dct['m']['Max']),
                   'lamb_rng': (dct['lamb']['Min'], dct['lamb']['Max']),
                   'qo_rng': (dct['qo']['Min'], dct['qo']['Max']),
                   'cpmax_rng': (dct['cpmax']['Min'], dct['cpmax']['Max']),
                   'sfmax_rng': (dct['sfmax']['Min'], dct['sfmax']['Max']),
                   'erz_rng': (dct['erz']['Min'], dct['erz']['Max']),
                   'ksat_rng': (dct['ksat']['Min'], dct['ksat']['Max']),
                   'c_rng': (dct['c']['Min'], dct['c']['Max']),
                   'k_rng': (dct['k']['Min'], dct['k']['Max']),
                   'n_rng': (dct['n']['Min'], dct['n']['Max']),
                   'lat': dct['lat']['Set']}
        return out_dct

    full_f = dir_calib + '/generations/population.txt'
    pop_df = pd.read_csv(full_f, sep=';')

    f_behav = '{}/behavioural.txt'.format(dir_glue)
    df_behav = pd.read_csv(f_behav, sep=';')
    print(df_behav.head().to_string())
    print(len(df_behav))
    print(df_behav['L'].min())
    f_select = '{}/selection.txt'.format(dir_glue)
    df_select = pd.read_csv(f_select, sep=';')
    print(df_select.head().to_string())
    print(len(df_select))

    rng_dct = extract_ranges(fhydroparam=fparam)
    print(rng_dct)

    models_df = df_behav.copy()
    likelihood = 'L'
    criteria = '>'
    behavioural = df_behav['L'].min()
    folder = dir_glue
    filename = 'scattergrams_2'


    fig = plt.figure(figsize=(14, 6), )  # Width, Height
    fig.suptitle('GLUE | Likelihood scattergrams of behavioural models'
                 ' | Criteria: {} {} {} | N = {}'.format(likelihood, criteria, behavioural, len(models_df)))
    rows = 2
    cols = 5
    gs = mpl.gridspec.GridSpec(rows, cols, wspace=0.55, hspace=0.45)
    #
    params = ('m', 'lamb', 'qo', 'cpmax', 'sfmax', 'erz', 'ksat', 'c', 'k', 'n')
    units = ('mm', 'twi', 'mm/d', 'mm', 'mm', 'mm', 'mm/d', '°C', 'days', 'n')
    #
    #
    if behavioural >= 0:
        ymin = 0
    else:
        ymin = behavioural + 0.5 * behavioural
    ind = 0
    for i in range(rows):
        for j in range(cols):
            lcl_prm = params[ind]
            lcl_units = units[ind]
            ax = fig.add_subplot(gs[i, j])
            plt.title('{}'.format(lcl_prm))
            plt.scatter(pop_df[lcl_prm].values, pop_df[likelihood].values,
                     marker='.', c='tab:grey', alpha=0.2, edgecolors='none')
            plt.plot(models_df[lcl_prm].values, models_df[likelihood].values, 'k.', zorder=1)
            plt.scatter(df_select[lcl_prm].values, df_select[likelihood].values,
                        marker='.', c='orange', edgecolors='none', zorder=2)
            plt.hlines(y=behavioural,
                       xmin=rng_dct['{}_rng'.format(lcl_prm)][0],
                       xmax=rng_dct['{}_rng'.format(lcl_prm)][1],
                       colors='tab:red', linestyles='--')
            # plt.plot(models_df[lcl_prm].values, criteria_line, 'tab:red')
            plt.ylabel('Ly[M|y]')
            plt.xlabel('{}'.format(lcl_units))
            plt.xlim(rng_dct['{}_rng'.format(lcl_prm)])
            plt.ylim((-0.65, -0.55))
            # plt.ylim((ymin, 1.1))
            ind = ind + 1
    #
    if show:
        plt.show()
        plt.close(fig)
    else:
        expfile = folder + '/' + filename + '.png'
        plt.savefig(expfile, dpi=400)
        plt.close(fig)
        return expfile


def view_anomaly(anomfolder, premap, posmap, anommap, mapvar='R', show=True):
    from visuals import _custom_cmaps
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    #
    folder = anomfolder
    # variables
    #
    _cmaps = _custom_cmaps()
    cmaps = {'R': _cmaps['flow'],
             'RIE': _cmaps['flow'],
             'RSE': _cmaps['flow'],
             'Qv': _cmaps['flow'],
             'Inf': _cmaps['flow'],
             'ET': _cmaps['flow_v'],
             'Evc': _cmaps['flow_v'],
             'Evs': _cmaps['flow_v'],
             'Tpgw': _cmaps['flow_v'],
             'Tpun': _cmaps['flow_v'],
             'VSA': 'Blues',
             'D': _cmaps['D'],
             'Cpy': _cmaps['stk'],
             'Sfs': _cmaps['stk'],
             'Unz': _cmaps['stk'],
             'asl': _cmaps['sed'],
             'asllog': _cmaps['sed'],
             'pload': _cmaps['sed'],
             'nload': _cmaps['sed'],
             'c_usle': 'YlGn_r',
             'k_usle': 'Oranges',
             's_rusle': 'OrRd',
             'l_rusle': 'OrRd'
             }
    units = {'R': 'mm',
             'RIE': 'mm',
             'RSE': 'mm',
             'Qv': 'mm',
             'Inf': 'mm',
             'ET': 'mm',
             'Evc': 'mm',
             'Evs': 'mm',
             'Tpgw': 'mm',
             'Tpun': 'mm',
             'Tpgw': 'mm',
             'VSA': '%',
             'D': 'mm',
             'Cpy': 'mm',
             'Sfs': 'mm',
             'Unz': 'mm',
             'asl': 'ton/yr',
             'asllog': 'log(ton/yr)',
             'pload': 'kgP/yr',
             'nload': 'kgN/yr',
             'c_usle': '-',
             'k_usle': 'ton h MJ-1 mm-1 ',
             's_rusle': '-',
             'l_rusle': '-'
             }

    ext_xmin = 550
    ext_xmax = 1150
    ext_ymin = 400
    ext_ymax = 1000

    premap = premap[ext_xmin:ext_xmax, ext_ymin:ext_ymax]
    posmap = posmap[ext_xmin:ext_xmax, ext_ymin:ext_ymax]
    anommap = anommap[ext_xmin:ext_xmax, ext_ymin:ext_ymax]
    v_max = np.max((np.percentile(premap, q=98), np.percentile(posmap, q=98)))
    v_min = np.max((np.percentile(premap, q=2), np.percentile(posmap, q=2)))


    maps_lst = [premap, posmap, anommap]
    maps_lst_ttl = ['pre', 'pos', 'anomaly']
    lims = {'pre': {'min': v_min,
                    'max': v_max},
            'pos': {'min': v_min,
                    'max': v_max},
            'anomaly': {'min': -np.max(np.abs(anommap)),
                        'max': np.max(np.abs(anommap))}}

    fig = plt.figure(figsize=(12, 4))  # Width, Height
    gs = mpl.gridspec.GridSpec(2, 9, wspace=0.2, hspace=0.2, left=0.05, bottom=0.05, top=0.95, right=0.95)
    fig.suptitle('{} mean anomaly'.format(mapvar))
    anchors_l = [0, 3, 6]
    anchors_r = [3, 6, 9]
    for i in range(len(maps_lst)):
        lcl_cmap = cmaps[mapvar]
        if i == 2:
            lcl_cmap = 'seismic_r'
        wind_left = 0
        wind_right = 3
        plt.subplot(gs[:, anchors_l[i]:anchors_r[i]])
        im = plt.imshow(maps_lst[i],
                        cmap=lcl_cmap,
                        vmin=lims[maps_lst_ttl[i]]['min'],
                        vmax=lims[maps_lst_ttl[i]]['max'])
        #plt.title('{} ({})'.format(maps_lst_ttl[i], units[mapvar]))
        plt.colorbar(im, shrink=0.3)
        plt.axis('off')
    filename = 'annual_{}_Mean_anomaly'.format(mapvar)
    if show:
        plt.show()
        plt.close(fig)
    else:
        filepath = folder + '/' + filename + '.png'
        plt.savefig(filepath, dpi=400)
        plt.close(fig)


def view_uncertainty(uncfolder, premean, posmean, prerng, posrng, preunc, posunc, uncmap,
                     mapvar='R', show=True):
    from visuals import _custom_cmaps
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    #
    folder = uncfolder
    # variables
    #
    _cmaps = _custom_cmaps()
    cmaps = {'R': _cmaps['flow'],
             'RIE': _cmaps['flow'],
             'RSE': _cmaps['flow'],
             'Qv': _cmaps['flow'],
             'Inf': _cmaps['flow'],
             'ET': _cmaps['flow_v'],
             'Evc': _cmaps['flow_v'],
             'Evs': _cmaps['flow_v'],
             'Tpgw': _cmaps['flow_v'],
             'Tpun': _cmaps['flow_v'],
             'VSA': 'Blues',
             'D': _cmaps['D'],
             'Cpy': _cmaps['stk'],
             'Sfs': _cmaps['stk'],
             'Unz': _cmaps['stk'],
             'asl': _cmaps['sed'],
             'asllog': _cmaps['sed'],
             'pload': _cmaps['sed'],
             'nload': _cmaps['sed']}
    units = {'R': 'mm',
             'RIE': 'mm',
             'RSE': 'mm',
             'Qv': 'mm',
             'Inf': 'mm',
             'ET': 'mm',
             'Evc': 'mm',
             'Evs': 'mm',
             'Tpgw': 'mm',
             'Tpun': 'mm',
             'Tpgw': 'mm',
             'VSA': '%',
             'D': 'mm',
             'Cpy': 'mm',
             'Sfs': 'mm',
             'Unz': 'mm',
             'asl': 'ton/yr',
             'asllog': 'log(ton/yr)',
             'pload': 'kgP/yr',
             'nload': 'kgN/yr'
             }

    ext_xmin = 550
    ext_xmax = 1150
    ext_ymin = 400
    ext_ymax = 1000
    # clip extents:
    premean = premean[ext_xmin:ext_xmax, ext_ymin:ext_ymax]
    posmean = posmean[ext_xmin:ext_xmax, ext_ymin:ext_ymax]
    prerng = prerng[ext_xmin:ext_xmax, ext_ymin:ext_ymax]
    posrng = posrng[ext_xmin:ext_xmax, ext_ymin:ext_ymax]
    preunc = preunc[ext_xmin:ext_xmax, ext_ymin:ext_ymax]
    posunc = posunc[ext_xmin:ext_xmax, ext_ymin:ext_ymax]
    uncmap = uncmap[ext_xmin:ext_xmax, ext_ymin:ext_ymax]
    # get min max
    v_max = np.max((np.percentile(premean, q=98), np.percentile(posmean, q=98)))
    v_min = np.max((np.percentile(premean, q=2), np.percentile(posmean, q=2)))
    vunc_max = np.max((np.percentile(preunc, q=98), np.percentile(posunc, q=98)))
    vunc_min = np.max((np.percentile(preunc, q=2), np.percentile(posunc, q=2)))
    #
    maps_lst = [[premean, prerng, preunc],
                [posmean, posrng, posunc]]
    maps_lst_ttl = ['pre', 'pos', 'unc']
    lims = {'pre': {'min': v_min,
                    'max': v_max},
            'pos': {'min': v_min,
                    'max': v_max},
            'unc': {'min': vunc_min,
                    'max': vunc_max}}

    fig = plt.figure(figsize=(12, 8))  # Width, Height
    gs = mpl.gridspec.GridSpec(6, 9, wspace=0.01, hspace=0.07, left=0.05, bottom=0.05, top=0.95, right=0.95)
    fig.suptitle('{} uncertainty'.format(mapvar))
    col_anchors_left = [0, 3, 6]
    col_anchors_right = [3, 6, 9]
    row_anchors_upper = [0, 2]
    row_anchors_lower = [2, 4]
    for j in range(len(maps_lst)):
        for i in range(len(maps_lst[j])):
            lcl_cmap = cmaps[mapvar]
            if i == 2:
                lcl_cmap = 'Greys'
            plt.subplot(gs[row_anchors_upper[j]:row_anchors_lower[j],
                        col_anchors_left[i]:col_anchors_right[i]])
            im = plt.imshow(maps_lst[j][i],
                            cmap=lcl_cmap,
                            vmin=lims[maps_lst_ttl[i]]['min'],
                            vmax=lims[maps_lst_ttl[i]]['max'])
            #plt.title('{} ({})'.format(maps_lst_ttl[i], units[mapvar]))
            plt.colorbar(im, shrink=0.3)
            plt.axis('off')
    # avg unc
    plt.subplot(gs[4:6, 6:9])
    im = plt.imshow(uncmap,
                    cmap='Greys',
                    vmin=lims['unc']['min'],
                    vmax=lims['unc']['max'])
    # plt.title('{} ({})'.format(maps_lst_ttl[i], units[mapvar]))
    plt.colorbar(im, shrink=0.3)
    plt.axis('off')
    filename = 'annual_{}_Mean_uncertainty'.format(mapvar)
    if show:
        plt.show()
        plt.close(fig)
    else:
        filepath = folder + '/' + filename + '.png'
        plt.savefig(filepath, dpi=400)
        plt.close(fig)


def view_ensemble(calibfolder, gluefolder, outputfolder, show=True):
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    #
    folder = outputfolder
    #

    f_ensemble = gluefolder + "/ensemble_q.txt"
    #f_ensemble = gluefolder + "/ensemble_et.txt"
    f_global = calibfolder + "/MLM/full_period/osa_zmaps/analyst_sim_series.txt" # "/et_obs.txt"
    #f_global = calibfolder + "/etobs_series.txt"
    #
    #
    ensemb_df = pd.read_csv(f_ensemble, sep=';', parse_dates=['Date'])
    v_obs_df = pd.read_csv(f_global, sep=';', parse_dates=['Date'])
    # print(et_df.head().to_string())

    obs_field = 'Qobs'
    #obs_field = 'ETobs'
    filename = '{}_series'.format(obs_field)
    dct_colors = {'Qobs': 'tab:blue',
                  'ETobs': 'tab:red'}
    dct_lims = {'Qobs': {'Main': (0.001, 40), 'Unc': (0, 500), 'Enc': (0, 101)},
                'ETobs': {'Main': (0.0, 6), 'Unc': (0, 250), 'Enc': (0, 20)}}
    dct_mark = {'Qobs': 'k.',
                  'ETobs': 'ko'}

    join_df = pd.merge(ensemb_df[['Date', 'Lo_5', 'Mid_50', 'Hi_95']],
                       v_obs_df[['Date', obs_field]], 'left', on='Date')

    print(join_df.head().to_string())
    join_df['Encaps'] = 0
    for i in range(len(join_df)):
        lcl_obs = join_df[obs_field].values[i]
        lcl_lo = join_df['Lo_5'].values[i]
        lcl_hi = join_df['Hi_95'].values[i]
        if lcl_obs >= lcl_lo and lcl_obs <= lcl_hi:
            join_df['Encaps'].values[i] = 1
    join_df['Encaps_sum7'] = 0
    window = 7
    for i in range(window - 1, len(join_df)):
        lcl_sum = np.sum(join_df['Encaps'].values[i - window: i])
        join_df['Encaps_sum7'].values[i] = lcl_sum
    join_df['Encaps_sum7'] = 100 * join_df['Encaps_sum7'] / window
    print(join_df.head().to_string())
    join_df['Unc_coef'] = 100 * (join_df['Hi_95'] - join_df['Lo_5']) / (join_df['Mid_50'] + 0.001)
    join_df['Unc_coef_sum7'] = 0
    for i in range(6, len(join_df)):
        lcl_sum = np.sum(join_df['Unc_coef'].values[i - 7: i])
        if pd.isna(lcl_sum):
            join_df['Unc_coef_sum7'].values[i] = np.nan
        else:
            join_df['Unc_coef_sum7'].values[i] = lcl_sum
    join_df['Unc_coef_sum7'] = join_df['Unc_coef_sum7'] / 7
    print(join_df.head().to_string())

    calib_join_df = join_df.query('Date < "2013-10-19"')
    valid_join_df = join_df.query('Date >= "2013-10-19"')
    calib_join_df['Encaps_mean'] = calib_join_df['Encaps_sum7'].mean()
    valid_join_df['Encaps_mean'] = valid_join_df['Encaps_sum7'].mean()
    print(calib_join_df['Encaps_sum7'].mean())
    print(valid_join_df['Encaps_sum7'].mean())
    calib_join_df['Unc_mean'] = calib_join_df['Unc_coef_sum7'].mean()
    valid_join_df['Unc_mean'] = valid_join_df['Unc_coef_sum7'].mean()
    print(calib_join_df['Unc_coef_sum7'].mean())
    print(valid_join_df['Unc_coef_sum7'].mean())

    join_df.to_csv('{}/{}_ensemble.txt'.format(folder, obs_field), sep=';', index=False)
    #
    #
    #plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 6))  # Width, Height
    gs = mpl.gridspec.GridSpec(7, 1, wspace=0.3, hspace=0.9, left=0.05, bottom=0.05, top=0.95, right=0.95)

    ax = fig.add_subplot(gs[0:3, 0])
    plt.fill_between(x=ensemb_df['Date'], y1=ensemb_df['Lo_5'], y2=ensemb_df['Hi_95'],
                     color='silver')
    plt.plot(ensemb_df['Date'], ensemb_df['Mid_50'], dct_colors[obs_field])
    plt.plot(v_obs_df['Date'], v_obs_df[obs_field], dct_mark[obs_field])
    plt.vlines(x=valid_join_df['Date'].values[0], ymin=0.001, ymax=35, colors='tab:red')
    plt.xlim((ensemb_df['Date'].values[0], ensemb_df['Date'].values[-1]))
    plt.ylim(dct_lims[obs_field]['Main'])
    if obs_field == 'Qobs':
        plt.yscale('log')
    plt.grid(True)

    ax = fig.add_subplot(gs[3:5, 0])
    plt.plot(join_df['Date'], join_df['Unc_coef_sum7'], 'tab:grey')
    plt.vlines(x=valid_join_df['Date'].values[0], ymin=0, ymax=500, colors='tab:red')
    plt.plot(calib_join_df['Date'], calib_join_df['Unc_mean'], '--k')
    plt.plot(valid_join_df['Date'], valid_join_df['Unc_mean'], '--k')
    plt.xlim((ensemb_df['Date'].values[0], ensemb_df['Date'].values[-1]))
    plt.ylim(dct_lims[obs_field]['Unc'])

    ax = fig.add_subplot(gs[5:, 0])
    plt.plot(join_df['Date'], join_df['Encaps_sum7'], 'tab:grey')
    if obs_field == 'ETobs':
        pass
    else:
        plt.plot(calib_join_df['Date'], calib_join_df['Encaps_mean'], '--k')
        plt.plot(valid_join_df['Date'], valid_join_df['Encaps_mean'], '--k')
    plt.vlines(x=valid_join_df['Date'].values[0], ymin=0, ymax=120, colors='tab:red')
    plt.xlim((join_df['Date'].values[0], join_df['Date'].values[-1]))
    plt.ylim(dct_lims[obs_field]['Enc'])
    if show:
        plt.show()
        plt.close(fig)
    else:
        filepath = folder + '/' + filename + '.png'
        plt.savefig(filepath, dpi=400)
        plt.close(fig)

    if obs_field == 'Qobs':
        import analyst
        kge_mid = analyst.kge(obs=np.log10(join_df['Qobs'].values),
                              sim=np.log10(join_df['Mid_50'].values))
        print('KGE mid = {}'.format(kge_mid))
        nse_mid = analyst.nse(obs=np.log10(join_df['Qobs'].values),
                              sim=np.log10(join_df['Mid_50'].values))
        print('NSE mid = {}'.format(nse_mid))

        r_mid = analyst.linreg(obs=np.log10(join_df['Qobs'].values),
                               sim=np.log10(join_df['Mid_50'].values))
        print(r_mid)
        r_lo = analyst.linreg(obs=np.log10(join_df['Qobs'].values),
                               sim=np.log10(join_df['Lo_5'].values))
        print(r_lo)
        r_hi = analyst.linreg(obs=np.log10(join_df['Qobs'].values),
                               sim=np.log10(join_df['Hi_95'].values))
        print(r_hi)
        # scatter plot
        fig = plt.figure(figsize=(4, 4))
        plt.scatter(join_df['Qobs'], join_df['Lo_5'], color='b', alpha=0.2, marker='o', edgecolors='none')
        plt.scatter(join_df['Qobs'], join_df['Hi_95'], color='r', alpha=0.2, marker='o', edgecolors='none')
        plt.scatter(join_df['Qobs'], join_df['Mid_50'], color='k', alpha=0.7, marker='o', edgecolors='none')
        plt.ylim(0.01, 50)
        plt.xlim(0.01, 50)
        plt.yscale('log')
        plt.xscale('log')
        plt.grid(True)
        if show:
            plt.show()
            plt.close(fig)
        else:
            filepath = folder + '/' + filename + '_scatter.png'
            plt.savefig(filepath, dpi=400)
            plt.close(fig)
        # CFCs
        fig = plt.figure(figsize=(4, 4))
        freq_obs_df = analyst.frequency(dataframe=join_df, var_field='Qobs')
        freq_mid_df = analyst.frequency(dataframe=join_df, var_field='Mid_50')
        freq_lo_df = analyst.frequency(dataframe=join_df, var_field='Lo_5')
        freq_hi_df = analyst.frequency(dataframe=join_df, var_field='Hi_95')
        print(freq_obs_df.head().to_string())
        plt.plot(freq_mid_df['Exceedance'], freq_mid_df['Values'], 'tab:blue')
        plt.plot(freq_obs_df['Exceedance'], freq_obs_df['Values'], 'ko')
        plt.fill_between(x=freq_obs_df['Exceedance'],
                         y1=freq_lo_df['Values'],
                         y2=freq_hi_df['Values'],
                         color='silver')
        plt.yscale('log')
        plt.grid(True)
        if show:
            plt.show()
            plt.close(fig)
        else:
            filepath = folder + '/' + filename + '_cfc.png'
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
    f_global_et = calibfolder + "/etobs_series.txt"
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
    from analyst import frequency
    fpos = "{}\series_ensemble.txt".format(pos_folder)
    fpre = "{}\series_ensemble.txt".format(pre_folder)
    pos_df = pd.read_csv(fpos, sep=';', parse_dates=['Date'])
    pre_df = pd.read_csv(fpre, sep=';', parse_dates=['Date'])
    print(pos_df.head().to_string())
    #
    # CFC plots
    vars = ['Q', 'Qb', 'ET']
    for v in vars:
        cfc_mid_pre_df = frequency(dataframe=pre_df, var_field='{}_50'.format(v))
        cfc_lo_pre_df = frequency(dataframe=pre_df, var_field='{}_05'.format(v))
        cfc_hi_pre_df = frequency(dataframe=pre_df, var_field='{}_95'.format(v))
        cfc_mid_pos_df = frequency(dataframe=pos_df, var_field='{}_50'.format(v))
        cfc_lo_pos_df = frequency(dataframe=pos_df, var_field='{}_05'.format(v))
        cfc_hi_pos_df = frequency(dataframe=pos_df, var_field='{}_95'.format(v))
        print(cfc_mid_pre_df.head().to_string())
        fig = plt.figure(figsize=(4, 4))  # Width, Height
        plt.fill_between(x=cfc_hi_pos_df['Exceedance'],
                         y1=cfc_lo_pos_df['Values'],
                         y2=cfc_hi_pos_df['Values'],
                         color='tab:blue',
                         alpha=0.4,
                         edgecolor='none')
        plt.fill_between(x=cfc_hi_pre_df['Exceedance'],
                         y1=cfc_lo_pre_df['Values'],
                         y2=cfc_hi_pre_df['Values'],
                         color='tab:green',
                         alpha=0.4,
                         edgecolor='none')
        plt.plot(cfc_mid_pre_df['Exceedance'], cfc_mid_pre_df['Values'], 'tab:green')
        plt.plot(cfc_mid_pos_df['Exceedance'], cfc_mid_pos_df['Values'], 'tab:blue')
        plt.xlim(0, 100)
        if v == 'Q':
            plt.yscale('log')
        plt.grid(True)
        if show:
            plt.show()
            plt.close(fig)
        else:
            filepath = folder + '\{}_cfc_prepost.png'.format(v)
            plt.savefig(filepath, dpi=400)
            plt.close(fig)

    #
    # Series plots
    # vars
    vars = ['Cpy', 'Sfs', 'Unz', 'TF', 'ET', 'Evc', 'Evs', 'Tpun', 'Tpgw', 'R', 'Inf', 'Qv', 'Q', 'Qb']
    for v in vars:
        print(v)
        fig = plt.figure(figsize=(16, 3))  # Width, Height
        plt.fill_between(x=pos_df['Date'],
                         y1=pos_df['{}_05'.format(v)],
                         y2=pos_df['{}_95'.format(v)],
                         color='tab:blue',
                         alpha=0.4,
                         edgecolor='none')
        plt.fill_between(x=pre_df['Date'],
                         y1=pre_df['{}_05'.format(v)],
                         y2=pre_df['{}_95'.format(v)],
                         color='tab:green',
                         alpha=0.4,
                         edgecolor='none')
        plt.plot(pos_df['Date'], pos_df['{}_50'.format(v)], 'tab:blue', label='post-development')
        plt.plot(pre_df['Date'], pre_df['{}_50'.format(v)], 'tab:green', label='pre-development')
        plt.xlim((pre_df['Date'].values[0], pre_df['Date'].values[-1]))
        plt.title('{}'.format(v))
        plt.grid(True)
        if v == 'Q':
            plt.yscale('log')
            plt.legend(loc='lower right')
        else:
            plt.legend(loc='upper right')
        if show:
            plt.show()
            plt.close(fig)
        else:
            filepath = folder + '\{}_series_prepost.png'.format(v)
            plt.savefig(filepath, dpi=400)
            plt.close(fig)

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


sets_lst = ['ds0']
folder = 'C:/bin/pardinho/produtos_v2/run_02a/ds0'
calib_folder = 'C:/bin/pardinho/produtos_v2/run_02a/ds0/search'
glue_folder = 'C:/bin/pardinho/produtos_v2/run_02a/ds0/select'
f_hydroparam =  'C:/bin/pardinho/produtos_v2/run_02a/ds0/search/MLM/mlm_parameters.txt'

#view_evolution_lspace(folder=folder, calibfolder=calib_folder, gluefolder=glue_folder, show=False)
#view_evolution_scatter(dir_glue=glue_folder, dir_calib=calib_folder, fparam=f_hydroparam, show=False)
#view_evolution_datasets(folder='C:/bin/pardinho/produtos_v2/run_02a', show=False)
#view_evolution_4(folder='C:/bin/pardinho/produtos_v2/run_02a', show=False)


project_folder = 'C:/bin/pardinho/produtos_v2/inputs/pardinho'
fseries = 'C:/bin/pardinho/produtos_v2/inputs/pardinho/datasets/observed/aoi_series.txt'
for s in sets_lst:
    folder = 'C:/bin/pardinho/produtos_v2/run_02a/{}'.format(s)
    pre_folder = 'C:/bin/pardinho/produtos_v2/run_02a/{}/pre_bat'.format(s)
    pos_folder = 'C:/bin/pardinho/produtos_v2/run_02a/{}/pos_bat'.format(s)
    asl_pre_folder = 'C:/bin/pardinho/produtos_v2/run_02a/{}/pre_asla'.format(s)
    asl_pos_folder = 'C:/bin/pardinho/produtos_v2/run_02a/{}/pos_asla'.format(s)
    """
    step03_map_processes(folder=folder,
                         gluefolder=glue_folder,
                         projectfolder=project_folder,
                         fseries=fseries,
                         vars='R-RIE-RSE-D-ET-Qv-Tpgw-Inf-Evc-Evs-Tpun-Cpy-Sfs-Unz',
                         pre=True,
                         pos=False)
                         
    step03_map_raster(folder=folder, mapvar='Evs-Evc')
    step05_compute_anomaly(folder,
                           hy_prefolder=pre_folder,
                           hy_posfolder=pos_folder,
                           asla_prefolder=asl_pre_folder,
                           asla_posfolder=asl_pos_folder,
                           mapvars='R-RIE',#R-RSE-Inf-Qv-ET-Tpun-Tpgw-Evc-Evs-D-Cpy-Sfs-Unz',
                           asla=False
                           )
    step06_compute_uncertainty(folder,
                               hy_prefolder=pre_folder,
                               hy_posfolder=pos_folder,
                               mapvars='R-RIE',
                               )
    """

view_rank_diff_hist()



