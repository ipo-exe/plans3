# OK
def demo_watch_by_zmaps():
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
    folder_output = 'C:/bin/ibira/views'
    #
    # directory of input data
    folder_input =  'C:/000_myFiles/myDrive/Plans3/ibirapuita/datasets/observed'
    # files paths to raster maps
    ftwi = folder_input + '/' + '__calib_twi_window1.asc'
    fshru = folder_input + '/' + '__calib_shru_window1.asc'
    ##fbasin = folder_input + '/' + 'calib_basin.asc'
    #
    # import raster maps
    meta, twi = inp.asc_raster(ftwi, dtype='float32')
    meta, shru = inp.asc_raster(fshru, dtype='float32')
    ##meta, basin = inp.asc_raster(fbasin)
    #
    # directory of simulated data
    folder_sim = 'C:/bin/ibira/SLH_2021-10-04-15-57-38'
    # simulated time series
    file = folder_sim + '/' + 'sim_series.txt'
    # import time series
    df_series = pd.read_csv(file, sep=';')
    #
    # define here list of variable maps
    varmaps = ['D', 'ET', 'VSA', 'Qv', 'R']
    size = 50  # define how many frames to watch
    start = 0
    #
    # loop in variables
    for v in varmaps:
        # assumed file in simulation
        file = folder_sim + '/' + 'sim_zmaps_series_{}.txt'.format(v)  # open map series
        # import map series
        df = pd.read_csv(file, sep=';')
        # extract min and max values from series (global average values)
        v_min = 0 #np.min(df_series[v].values[:size])
        if v == 'D':
            v_max = np.max(df_series[v].values[start: start + size]) * 1
        elif v == 'VSA':
            v_max = 1
        else:
            v_max = np.max(df_series[v].values[start: start + size])
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
            #import matplotlib.pyplot as plt
            #plt.imshow(mp)
            #plt.show()
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


def demo_watch_pannels():
    from tools import export_local_pannels
    # define output directory
    folder_output = 'C:/bin/ibira/SLH_2021-10-04-16-51-52'
    #
    # directory of input data
    folder_input = 'C:/000_myFiles/myDrive/Plans3/ibirapuita/datasets/observed'
    # files paths to raster maps
    ftwi = folder_input + '/' + '__calib_twi_window1.asc'
    fshru = folder_input + '/' + '__calib_shru_window1.asc'

    export_local_pannels(ftwi, fshru,
                         folder=folder_output,
                         date_init='2014-01-01',
                         date_end='2014-03-01',
                         frametype='R',
                         filter_date=True,
                         tui=True)


def demo_slh():
    """
    Simulation demo routine
    :return:
    """
    from tools import slh
    import backend
    import pandas as pd

    # define output workplace
    outfolder = 'C:/bin/ibira'
    # define observed datasets folder
    folder = 'C:/000_myFiles/myDrive/Plans3/ibirapuita/datasets/observed'
    # get input files
    files_input = backend.get_input2simbhydro(aoi=False)
    fseries ='{}/{}'.format(folder, files_input[0])
    fhydroparam = '{}/{}'.format(folder, files_input[1])
    fshruparam = '{}/{}'.format(folder, files_input[2])
    fhistograms = '{}/{}'.format(folder, files_input[3])
    fbasinhists = '{}/{}'.format(folder, files_input[4])
    fbasin = '{}/{}'.format(folder, files_input[5])
    ftwi = '{}/{}'.format(folder, files_input[6])
    fshru = '{}/{}'.format(folder, files_input[7])
    fcanopy = '{}/{}'.format(folder, files_input[8])
    #
    # map back settings
    vars = 'IRI-IRA'
    mapdates = 'all'
    mapback = True
    if mapback:
        # define the variables to map back
        vars = 'all'
        # define the range of dates to map back
        date_init = '2014-01-01'
        date_end = '2015-01-01'
        series = pd.read_csv(fseries, sep=';', parse_dates=['Date'])
        query_str = 'Date >= "{}" and Date < "{}"'.format(date_init, date_end)
        series = series.query(query_str)
        mapdates = ' & '.join(series['Date'].astype('str'))
    # integration settings:
    integrate = False
    # raster mapping settings:
    mapraster = False
    #
    # call function
    out_dct = slh(fseries=fseries,
                  fhydroparam=fhydroparam,
                  fshruparam=fshruparam,
                  fhistograms=fhistograms,
                  fbasinhists=fbasinhists,
                  fbasin=fbasin,
                  ftwi=ftwi,
                  fshru=fshru,
                  fcanopy=fcanopy,
                  folder=outfolder,
                  wkpl=True,
                  tui=True,
                  mapback=mapback,
                  mapraster=mapraster,
                  mapvar=vars,
                  integrate=integrate,
                  mapdates=mapdates,
                  qobs=True)


def demo_slh_calib():
    from tools import hca
    import backend
    import pandas as pd

    # define output workplace
    outfolder = 'C:/bin/ibira'
    # define observed datasets folder
    folder = 'C:/000_myFiles/myDrive/Plans3/ibirapuita/datasets/observed'
    # get input files
    files_input = backend.get_input2simbhydro(aoi=False)
    fseries ='{}/{}'.format(folder, files_input[0])
    fhydroparam = '{}/{}'.format(folder, files_input[1])
    fshruparam = '{}/{}'.format(folder, files_input[2])
    fhistograms = '{}/{}'.format(folder, files_input[3])
    fbasinhists = '{}/{}'.format(folder, files_input[4])
    fbasin = '{}/{}'.format(folder, files_input[5])
    ftwi = r"C:\000_myFiles\myDrive\Plans3\ibirapuita\datasets\observed\__calib_twi_window1.asc"
    #ftwi = '{}/{}'.format(folder, files_input[6])
    fshru = r"C:\000_myFiles\myDrive\Plans3\ibirapuita\datasets\observed\__calib_shru_window1.asc"
    #fshru = '{}/{}'.format(folder, files_input[7])
    fcanopy = '{}/{}'.format(folder, files_input[8])
    fzmaps = r"C:\000_myFiles\myDrive\Plans3\ibirapuita\datasets\observed\calib_etpat_zmaps_note.txt" #'{}/{}'.format(folder, files_input[9])
    #
    #
    # call function
    out_dct = hca(fseries=fseries,
                  fhydroparam=fhydroparam,
                  fshruparam=fshruparam,
                  fhistograms=fhistograms,
                  fbasinhists=fbasinhists,
                  fbasin=fbasin,
                  ftwi=ftwi,
                  fshru=fshru,
                  fcanopy=fcanopy,
                  fzmaps=fzmaps,
                  folder=outfolder,
                  wkpl=True,
                  tui=True,
                  label='CALIB')


def demo_calibration():
    import backend
    from tools import calibrate

    # define the output workplace folder
    outfolder = 'C:/bin/ibira'

    # get folder of observed datasets
    folder = 'C:/000_myFiles/myDrive/Plans3/ibirapuita/datasets/observed'

    # get observed datasets standard names
    files_input = backend.get_input2calibhydro()
    fseries = folder + '/' + files_input[0]
    fhydroparam = folder + '/' + files_input[1]
    fshruparam = folder + '/' + files_input[2]
    fhistograms = folder + '/' + files_input[3]
    fbasinhists = folder + '/' + files_input[4]
    fbasin = folder + '/' + files_input[5]
    ftwi = r"C:\000_myFiles\myDrive\Plans3\ibirapuita\datasets\observed\__calib_twi_window1.asc"
    fshru = r"C:\000_myFiles\myDrive\Plans3\ibirapuita\datasets\observed\__calib_shru_window1.asc"
    #ftwi = folder + '/' + files_input[6]
    #fshru = folder + '/' + files_input[7]
    fetpatzmaps = folder + '/' + 'calib_etpat_zmaps_note.txt' #files_input[8]
    fcanopy = folder + '/' + files_input[9]

    # Options: 'NSE', 'NSElog', 'RMSE', 'RMSElog', 'KGE', 'KGElog', 'PBias', 'RMSE-CFC', 'RMSElog-CFC'

    likelihood = 'NSElog'
    generations = 10
    popsize = 200
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
                           normalize=False
                           )

    print('\nRun files sucessfully created at:\n{}\n'.format(calibfiles['Folder']))


def demo_glue():
    from backend import get_input2calibhydro
    from tools import glue
    #
    # get folder of observed datasets
    folder = 'C:/000_myFiles/myDrive/Plans3/pardinho/datasets/observed'
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
    calib_folder = r"C:\bin\pardinho\optimization\000_calib_Hydrology_KGElog_2021-10-27-18-01-16"
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
                     run_ensemble=False,
                     folder=calib_folder,
                     wkpl=True,
                     normalize=True,
                     tui=True)


def demo_bat_slh_pre_pos():
    from tools import bat_slh

    folder = r"C:/bin/pardinho/optimization/000_calib_Hydrology_KGElog_2021-10-27-18-01-16"
    ftwi = r"C:/000_myFiles/myDrive/Plans3/pardinho/datasets/observed/aoi_twi.asc"
    fseries = r"C:/000_myFiles/myDrive/Plans3/pardinho/datasets/observed/calib_series.txt"
    fshru_param = r"C:/000_myFiles/myDrive/Plans3/pardinho/datasets/observed/calib_shru_param.txt"
    fbasin = r"C:/000_myFiles/myDrive/Plans3/pardinho/datasets/observed/aoi_basin.asc"
    fcanopy = r"C:/000_myFiles/myDrive/Plans3/pardinho/datasets/observed/calib_canopy_series.txt"
    fhydrop = r"C:/000_myFiles/myDrive/Plans3/pardinho/datasets/observed/hydro_param.txt"
    fmodels = folder + '/generations/selection.txt'
    #
    #
    # pre
    fshru = r"C:/000_myFiles/myDrive/Plans3/pardinho/datasets/projected/scn__lulc_predc/aoi_shru.asc"
    fhists = r"C:/000_myFiles/myDrive/Plans3/pardinho/datasets/projected/scn__lulc_predc/aoi_histograms.txt"
    fbasin_hists = r"C:/000_myFiles/myDrive/Plans3/pardinho/datasets/projected/scn__lulc_predc/aoi_basin_histograms.txt"
    bat_slh(fmodels=fmodels,
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
            mapvar='Qv-R-Tpgw-Inf-RIE-ET-VSA',
            integrate=True,
            qobs=True,
            pannel=False,
            ensemble=True,
            stats=True,
            annualize=True,
            folder='C:/bin/pardinho')
    #
    #
    # pos
    fshru = r"C:/000_myFiles/myDrive/Plans3/pardinho/datasets/observed/aoi_shru.asc"
    fhists = r"C:/000_myFiles/myDrive/Plans3/pardinho/datasets/observed/aoi_histograms.txt"
    fbasin_hists = r"C:/000_myFiles/myDrive/Plans3/pardinho/datasets/observed/aoi_basin_histograms.txt"
    bat_slh(fmodels=fmodels,
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
            mapvar='Qv-R-Tpgw-Inf-RIE-ET-VSA',
            integrate=True,
            qobs=True,
            pannel=False,
            ensemble=True,
            stats=True,
            annualize=True,
            folder='C:/bin/pardinho')


def demo_asla(pre=True):
    from tools import asla
    if pre:
        frunoff = r"C:/000_myFiles/myDrive/myProjects/104_paper_castelhano/produtos/pre/avg_R_annual.asc"
        flulc = r"C:/000_myFiles/myDrive/Plans3/pardinho/datasets/projected/scn__lulc_predc/aoi_lulc_predc.asc"
        label = 'E6000_pre'
    else:
        frunoff = r"C:/000_myFiles/myDrive/myProjects/104_paper_castelhano/produtos/pos/avg_R_annual.asc"
        flulc = r"C:/000_myFiles/myDrive/Plans3/pardinho/datasets/observed/aoi_lulc.asc"
        label = 'E6000_pos'
    #
    fseries = r"C:/000_myFiles/myDrive/Plans3/pardinho/datasets/observed/calib_series.txt"
    flulc_param = r"C:/000_myFiles/myDrive/Plans3/pardinho/datasets/observed/aoi_lulc_param.txt"
    fslope = r"C:/000_myFiles/myDrive/Plans3/pardinho/datasets/observed/aoi_slope.asc"
    fsoils = r"C:/000_myFiles/myDrive/Plans3/pardinho/datasets/observed/aoi_soils.asc"
    fsoils_param = r"C:/000_myFiles/myDrive/Plans3/pardinho/datasets/observed/aoi_soils_param.txt"
    #
    # run asla
    asla(fmap_r=frunoff,
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
         folder='C:/bin/pardinho/assessment')


def demo_compute_zmaps_et():
    import os
    import inp, visuals
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import tools
    folder = 'C:/000_myFiles/myDrive/Plans3/pardinho/datasets/observed/etpat'
    l = os.listdir(folder)
    # filtrar com base no nome
    files = list()
    for f in l:
        if '.asc' in f and '.aux' not in f:
            files.append(f)
    print(files)
    print(len(files))
    # get htwi and shru
    ftwi = 'C:/000_myFiles/myDrive/Plans3/pardinho/datasets/observed/calib_twi_window.asc'
    fshru = 'C:/000_myFiles/myDrive/Plans3/pardinho/datasets/observed/calib_shru_window.asc'
    meta, twi = inp.asc_raster(ftwi, dtype='float32')
    meta, shru = inp.asc_raster(fshru, dtype='float32')
    #plt.imshow(twi)
    #plt.show()
    # import et24h
    et_maps = list()
    et_dates = list()
    max_values = list()
    paths = list()
    for i in range(len(files)):
        print(i)
        filepath = folder + '/' + files[i]
        date = files[i].split('_')[-1].split('.')[0]
        meta, et = inp.asc_raster(filepath, dtype='float32')
        et = (et / 1) * (et >= 0) + (-1 * (et < 0))
        #plt.imshow(et)
        #plt.show()
        et_maps.append(et)
        et_dates.append(date)
        paths.append(filepath)
        max_values.append(np.max(et))
    df = pd.DataFrame({'Date': et_dates, 'File': paths})
    print(df.to_string())
    fout = folder + '/calib_et_series.txt'
    df.to_csv(fout, index=False, sep=';')
    for i in range(len(et_maps)):
        print(i)
        title = 'ET 24h | ' + et_dates[i]
        file_name = 'calib_etpat_' + et_dates[i]
        et = et_maps[i]
        et[et == -1] = np.nan
        #plt.imshow(et)
        #plt.show()
        visuals.plot_map_view(et, meta, [0, 10],
                              mapid='flow_v',
                              mapttl=title,
                              filename=file_name,
                              metadata=True,
                              folder=folder, show=False)
    #
    # compute zmaps
    tools.compute_zmap_series(fvarseries=fout,
                              var='et24h',
                              ftwi=ftwi,
                              fshru=fshru,
                              fhistograms='C:/000_myFiles/myDrive/Plans3/pardinho/datasets/observed/calib_histograms.txt',
                              filename='calib_etpat_zmaps',
                              folder='C:/000_myFiles/myDrive/Plans3/pardinho/datasets/observed',
                              dtype='float32', tui=True, factor=1)


def demo_sal_by_lamb():
    from tools import sal_d_by_lamb
    ftwi = r"C:\000_myFiles\myDrive\Plans3\ibirapuita\datasets\observed\__calib_twi_window1.asc"
    sal_d_by_lamb(ftwi=ftwi,
                  m=5,
                  lamb1=3,
                  lamb2=10,
                  dmax=50,
                  size=30,
                  wkpl=True,
                  folder='C:/bin/ibira')

# todo revise
def demo_path_problem():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import evolution
    from backend import create_rundir
    from datetime import datetime

    def id_label(id):
        if id < 10:
            return '000' + str(id)
        elif id >= 10 and id < 100:
            return '00' + str(id)
        elif id >= 100 and id < 1000:
            return '0' + str(id)
        elif id >= 1000 and id < 10000:
            return str(id)

    _nucleo = ((1, 0, -1),
               (1, 0, -1))

    _field = evolution.get_cost_surface()
    _vmin = np.min(_field)
    _vmax = np.max(_field)
    _field[50][40:55] = 100
    fig, ax = plt.subplots()
    im = ax.imshow(_field, 'jet', extent=[0, 100, 0, 100], origin='lower', vmin=_vmin, vmax=_vmax)
    fig.colorbar(im, shrink=0.4)
    plt.show()

    seed = int(str(datetime.now())[-6:])

    # start
    x0 = 15
    y0 = 15
    # target
    xf = 75
    yf = 95

    _solution = _field.copy()
    _solution[yf][xf] = 0

    _path_size = 120
    _gene_size = (_path_size, _path_size)
    _pop_size = 2000
    _generations = 150

    # get initial pop
    print('get pop')
    _pop0 = evolution.generate_population(nucleotides=_nucleo,
                                          genesizes=_gene_size,
                                          popsize=_pop_size)
    # evolve pop
    traced = evolution.demo_evolve_path(pop0=_pop0,
                                        nucleotides=_nucleo,
                                        solution=_solution,
                                        seed=seed,
                                        generations=_generations,
                                        fittype='path',
                                        offsfrac=1,
                                        mutrate=0.5,
                                        puremutrate=0.9,
                                        cutfrac=0.5,
                                        tracefrac=0.1,
                                        tracepop=False,
                                        tui=True)

    print(type(traced))
    print(len(traced))
    last = traced[len(traced) - 1]
    # print(last)
    print(len(last['DNAs']))
    last_df = pd.DataFrame({'Id': last['Ids'], 'Score': last['Scores'], 'Prox': last['Prox'], 'Cost': last['Cost']})
    print(last_df.to_string())

    folder = create_rundir(label='ev', wkplc='C:/bin/fun')

    # plot generations
    for g in range(len(traced)):
        print('gen {}'.format(g))
        # retrieve paths
        _x_paths = np.zeros(shape=(len(traced[g]['Ids']), _path_size))
        _y_paths = np.zeros(shape=(len(traced[g]['Ids']), _path_size))
        _last_xs = np.zeros(shape=len(traced[g]['Ids']))
        _last_ys = np.zeros(shape=len(traced[g]['Ids']))
        for i in range(len(traced[g]['Ids']) - 1):
            _x_paths[i] = evolution.express_path(gene=traced[g]['DNAs'][i][0], p0=x0)
            _y_paths[i] = evolution.express_path(gene=traced[g]['DNAs'][i][1], p0=y0)
            _last_xs[i] = _x_paths[i][-1]
            _last_ys[i] = _y_paths[i][-1]
        # plot paths
        fig, ax = plt.subplots()
        im = ax.imshow(_field, 'jet', extent=[0, 100, 0, 100], origin='lower', vmin=_vmin, vmax=_vmax)
        fig.colorbar(im, shrink=0.4)
        for i in range(len(_x_paths)):
            ax.plot(_x_paths[i], _y_paths[i], 'tab:grey', alpha=0.5)
        ax.scatter(_last_xs, _last_ys, c='k', marker='.', zorder=10000)
        ax.plot(x0, y0, 'ko')
        ax.text(x0 - 4, y0, 'A')
        ax.plot(xf, yf, 'ko')
        ax.text(xf + 2, yf, 'B')
        plt.title('gen = {}'.format(g))
        lcl_id = id_label(g)
        expfile = '{}/gen_frame_{}.png'.format(folder, lcl_id)
        plt.savefig(expfile)
        plt.close(fig)

    # plot dynamic path
    dynamic = False
    if dynamic:
        stages = [999]
        for s in stages:
            subfolder = create_rundir(label=str(s), wkplc=folder)
            # retrieve paths
            last = traced[s]
            _x_paths = np.zeros(shape=(len(last['Ids']), _path_size))
            _y_paths = np.zeros(shape=(len(last['Ids']), _path_size))
            for i in range(len(last['Ids']) - 1):
                _x_paths[i] = evolution.express_path(gene=last['DNAs'][i][0], p0=x0)
                _y_paths[i] = evolution.express_path(gene=last['DNAs'][i][1], p0=y0)
            # plot
            for t in range(1, _path_size):
                print('gen {} | t = {}'.format(s + 1, t))
                fig, ax = plt.subplots()
                im = ax.imshow(_field, 'jet', extent=[0, 100, 0, 100], origin='lower', vmin=_vmin, vmax=_vmax)
                fig.colorbar(im, shrink=0.4)
                for i in range(len(_x_paths)):
                    ax.plot(_x_paths[i][:t], _y_paths[i][:t], 'tab:grey', alpha=0.5)
                for i in range(len(_x_paths)):
                    ax.plot(_x_paths[i][:t][-1], _y_paths[i][:t][-1], c='tab:red', marker='.')
                ax.plot(x0, y0, 'ko')
                ax.text(x0 - 4, y0, 'A')
                ax.plot(xf, yf, 'ko')
                ax.text(xf + 2, yf, 'B')
                plt.title('gen {} | t = {}'.format(s + 1, t))
                lcl_id = id_label(t)
                expfile = '{}/frame_{}.png'.format(subfolder, lcl_id)
                plt.savefig(expfile)
                plt.close(fig)


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

# todo revise
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


def __insert_irrigation(folder='C:/bin'):
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.ndimage import gaussian_filter
    fseries = r"C:\000_myFiles\myDrive\Plans3\ibirapuita\datasets\observed\calib_series.txt"
    df = pd.read_csv(fseries, sep=';', parse_dates=['Date'])
    #
    # irrigation by aspersion:
    pulse_ira = 0  # total mm per batch
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
    pulse_iri = 1200  # total mm per batch
    start_iri = '10-01'  # month and day
    size_iri = 5 * 30  # number of days
    sigma_iri = 15
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
    print(df['IRI'].sum()/6)
    outfile = folder + '/series_with_irrigation.txt'
    df.to_csv(outfile, sep=';', index=False)

# todo revise
def __visual_map_analyst():
    from tools import osa_zmaps
    fobs_series = r"C:\000_myFiles\myDrive\Plans3\sacre\datasets\observed\calib_etpat_zmaps_note.txt"
    fsim_series = r"C:\bin\sacre\SLH_2021-09-28-12-46-33\sim_zmaps_series_ET.txt"
    fhistograms = r"C:\000_myFiles\myDrive\Plans3\sacre\datasets\observed\calib_histograms.txt"
    fseries = r"C:\000_myFiles\myDrive\Plans3\sacre\datasets\observed\calib_series.txt"
    osa_zmaps(fobs_series, fsim_series, fhistograms, fseries)

# todo revise
def __demo_obs_sim_map_analyst(fseries, type, var='ETPat', filename='obssim_maps_analyst', folder='C:/bin', tui=True):
    from inp import dataframe_prepro
    import inp
    import pandas as pd
    import numpy as np
    #
    import time, datetime
    import analyst
    from visuals import plot_zmap_analyst

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
        vis_file = plot_zmap_analyst(obs=maps_obs_lst[i], sim=maps_sim_lst[i], error=metric_maps[i],
                                     obs_sig=signal_obs_lst[i], sim_sig=signal_sim_lst[i], ranges=ranges,
                                     metricranges=metricranges, error_sig=metric_signal[i], metrics_dct=metrics_dct,
                                     filename=lcl_filename, folder=folder, ttl=lcl_ttl)

