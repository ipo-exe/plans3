
def watch_maps_byzmaps():
    """
    Cookbook to watch maps from a simulation run
    :return:
    """
    import tools, inp, geo
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
    folder_input =  'C:/000_myFiles/myDrive/Plans3/demo/datasets/observed'
    # files paths to raster maps
    ftwi = folder_input + '/' + 'calib_twi.asc'
    fshru = folder_input + '/' + 'calib_shru.asc'
    fbasin = folder_input + '/' + 'calib_basin.asc'
    #
    # import raster maps
    meta, twi = inp.asc_raster(ftwi)
    meta, shru = inp.asc_raster(fshru)
    meta, basin = inp.asc_raster(fbasin)
    #
    # directory of simulated data
    folder_sim = r"C:\bin\demo\full_SLH_2021-08-26-12-38-53\calibration_period"
    # simulated time series
    file = folder_sim + '/' + 'sim_series.txt'
    # import time series
    df_series = pd.read_csv(file, sep=';')
    #
    # define here list of variable maps
    varmaps = ['D']
    size = 360  # define how many frames to watch
    start = 160
    #
    # loop in variables
    for v in varmaps:
        # assumed file in simulation
        file = folder_sim + '/' + 'sim_zmaps_series_{}.txt'.format(v)  # open map series
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
            lcl_file = df['File'].values[i + start]
            lcl_date = df['Date'].values[i + start]
            # import Z-Map
            zmap, hist_twi, hist_shru = inp.zmap(file=lcl_file)
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


def visual_map_analyst():
    import inp, analyst
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
    meta, obs = inp.asc_raster(fobs)
    meta, sim = inp.asc_raster(fsim)
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
    from inp import dataframe_prepro
    import inp
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
                map, ybins, xbins = inp.zmap(map_file)
            elif type == 'raster':
                meta, map = inp.asc_raster(map_file)
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
    import inp
    from hydrology import map_back
    folder ='C:/Plans3/demo/datasets/observed'
    ftwi = '{}/calib_twi.asc'.format(folder)
    fshru = '{}/calib_shru.asc'.format(folder)

    meta, twi = inp.asc_raster(ftwi)
    meta, shru = inp.asc_raster(fshru)

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
            lcl_zmap, ybins, xbins = inp.zmap(lcl_file)
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
    from inp import asc_raster
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


def demo_create_benchmark_series():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    fseries = r"C:\000_myFiles\myDrive\Plans3\demo\datasets\observed\calib_series.txt"
    df = pd.read_csv(fseries, sep=';', parse_dates=['Date'])
    plt.plot(df['Date'], df['Prec'])
    plt.plot(df['Date'], df['Temp'])
    plt.show()

    # let dry
    df1 = df.copy()
    df1['Prec'] = 0.0
    df1['Temp'] = -5
    fseries_1 = 'C:/bin/demo/series_letdry.txt'
    df1.to_csv(fseries_1, sep=';', index=False)
    plt.plot(df1['Date'], df1['Prec'])
    plt.plot(df1['Date'], df1['Temp'])
    plt.show()

    # let dry + 10d pulse of 50 mm
    df2 = df1.copy()
    for i in range(10):
        df2['Prec'].values[200 + i] = 5
    fseries_2 = 'C:/bin/demo/series_10dpulse_50mm.txt'
    df2.to_csv(fseries_2, sep=';', index=False)
    plt.plot(df2['Date'], df2['Prec'])
    plt.plot(df2['Date'], df2['Temp'])
    plt.show()

    # let dry + 100d pulse of 100 mm
    df2 = df1.copy()
    for i in range(100):
        df2['Prec'].values[200 + i] = 5
    fseries_2 = 'C:/bin/demo/series_100dpulse_100mm.txt'
    df2.to_csv(fseries_2, sep=';', index=False)
    plt.plot(df2['Date'], df2['Prec'])
    plt.plot(df2['Date'], df2['Temp'])
    plt.show()

    # let dry + 10d pulse of 100 mm + ET
    df2 = df1.copy()
    for i in range(100):
        df2['Prec'].values[200 + i] = 15
    fseries_2 = 'C:/bin/demo/series_10dpulse_150mm_ET.txt'
    df2['Temp'] = df['Temp']
    df2.to_csv(fseries_2, sep=';', index=False)
    plt.plot(df2['Date'], df2['Prec'])
    plt.plot(df2['Date'], df2['Temp'])
    plt.show()

    # let dry + 10d pulse of real rain + ET
    df2 = df1.copy()
    for i in range(100):
        df2['Prec'].values[200 + i] = df['Prec'].values[200 + i]
    fseries_2 = 'C:/bin/demo/series_10dpulse_real_ET.txt'
    df2['Temp'] = df['Temp']
    df2.to_csv(fseries_2, sep=';', index=False)
    plt.plot(df2['Date'], df2['Prec'])
    plt.plot(df2['Date'], df2['Temp'])
    plt.show()


def insert_irrigation(folder='C:/bin'):
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.ndimage import gaussian_filter
    fseries = r"C:\000_myFiles\myDrive\Plans3\demo\datasets\observed\calib_series.txt"
    df = pd.read_csv(fseries, sep=';', parse_dates=['Date'])
    #
    # irrigation by aspersion:
    pulse_ira = 100  # total mm per batch
    peak_ira = '01-15' # month and day
    size_ira = 40 # number of days
    sigma_ira = size_ira / 6
    for i in range(len(df)):
        if str(df['Date'].values[i])[5:10] == peak_ira:
            df['IRA'].values[i] = pulse_ira
    #plt.plot(df['Date'], df['IRA'])
    #plt.show()
    print(df['IRA'].sum())
    ira = np.zeros(shape=len(df))
    ira = gaussian_filter(df['IRA'].values, sigma=sigma_ira)
    df['IRA'] = ira
    #plt.plot(df['Date'], df['IRA'])
    #plt.show()
    print(df['IRA'].sum())
    #
    #
    # irrigation by inundation:
    pulse_iri = 100  # total mm per batch
    start_iri = '10-15'  # month and day
    size_iri = 100  # number of days
    sigma_iri = 3
    fed = False
    counter = 0
    for i in range(len(df)):
        if str(df['Date'].values[i])[5:10] == start_iri:
            fed = True
        if counter > size_iri:
            fed = False
            counter = 0
        if fed:
            df['IRI'].values[i] = pulse_iri / size_iri
            counter = counter + 1
    # plt.plot(df['Date'], df['IRA'])
    # plt.show()
    print(df['IRI'].sum())
    iri = np.zeros(shape=len(df))
    iri = gaussian_filter(df['IRI'].values, sigma=sigma_iri)
    df['IRI'] = iri
    plt.plot(df['Date'], df['IRI'])
    plt.plot(df['Date'], df['IRA'])
    plt.show()
    print(df['IRI'].sum())
    outfile = folder + '/series_with_irrigation.txt'
    df.to_csv(outfile, sep=';', index=False)

#watch_maps_byzmaps()
