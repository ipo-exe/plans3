"""
Scripts for the final paper!!

"""
import os
import inp
import out
import numpy as np


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


def asla_median():
    """
    uses the median runoff as the runoff for ASLA assessment
    :return:
    """
    from tools import asla
    #
    fseries = r"C:/000_myFiles/myDrive/Plans3/pardinho/datasets/observed/calib_series.txt"
    flulc_param = r"C:/000_myFiles/myDrive/Plans3/pardinho/datasets/observed/aoi_lulc_param.txt"
    fslope = r"C:/000_myFiles/myDrive/Plans3/pardinho/datasets/observed/aoi_slope.asc"
    fsoils = r"C:/000_myFiles/myDrive/Plans3/pardinho/datasets/observed/aoi_soils.asc"
    fsoils_param = r"C:/000_myFiles/myDrive/Plans3/pardinho/datasets/observed/aoi_soils_param.txt"
    #
    # PRE
    frunoff = "C:/bin/pardinho/produtos/pre_batSLH_2021-11-27/annual_R_Median.asc"
    flulc = "C:/000_myFiles/myDrive/Plans3/pardinho/datasets/projected/scn__lulc_predc/aoi_lulc_predc.asc"
    label = 'ASLA_E6000_median_pre'
    # run asla for pre
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
         folder='C:/bin/pardinho/produtos')
    # POS
    frunoff = "C:/bin/pardinho/produtos/pos_batSLH_2021-11-27/annual_R_Median.asc"
    flulc = r"C:/000_myFiles/myDrive/Plans3/pardinho/datasets/observed/aoi_lulc.asc"
    label = 'ASLA_E6000_median_pos'
    # run asla for pos
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
         folder='C:/bin/pardinho/produtos')


def bat_anomaly():
    import os
    import numpy as np
    from visuals import plot_map_view
    outfolder = 'C:/bin/pardinho/produtos/anomaly'
    #
    stats = ['Median']
    # variables
    vars = ['Inf', 'Qv', 'R', 'RIE', 'ET', 'Tpgw', 'VSA']
    # POS
    pos_folder = 'C:/bin/pardinho/produtos/pos_batSLH_2021-11-27'
    pos_all_files = os.listdir(pos_folder)
    # PRE
    pre_folder = 'C:/bin/pardinho/produtos/pre_batSLH_2021-11-27'
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
            print('loading raster maps...')
            meta, pre_map = inp.asc_raster(file=lcl_pre_file_path, dtype='float32')
            meta, pos_map = inp.asc_raster(file=lcl_pos_file_path, dtype='float32')
            meta['NODATA_value'] = -99999
            #
            #
            anom_map = pos_map - pre_map
            #
            lcl_filename = 'annual_{}_{}_anomaly'.format(vars[i], s)
            print('exporting...')
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
    pos_folder = 'C:/bin/pardinho/produtos/pos_E6000_median_ASLA_2021-11-30'
    pos_all_files = os.listdir(pos_folder)
    # PRE
    pre_folder = 'C:/bin/pardinho/produtos/pre_E6000_median_ASLA_2021-11-30'
    pre_all_files = os.listdir(pre_folder)
    for i in range(len(vars)):
        # find file path
        for f in pre_all_files:
            if vars[i] in f and '.asc' in f:
                lcl_pre_file_path = '{}/{}'.format(pre_folder, f)
        # find file path
        for f in pos_all_files:
            if vars[i] in f and '.asc' in f:
                lcl_pos_file_path = '{}/{}'.format(pos_folder, f)
        print(lcl_pre_file_path)
        print(lcl_pos_file_path)
        print('loading raster maps...')
        meta, pre_map = inp.asc_raster(file=lcl_pre_file_path, dtype='float32')
        meta, pos_map = inp.asc_raster(file=lcl_pos_file_path, dtype='float32')
        meta['NODATA_value'] = -99999
        #
        #
        anom_map = pos_map - pre_map
        #
        lcl_filename = 'annual_{}_anomaly'.format(vars[i])
        print('exporting...')
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


def bat_uncertainty():
    import os
    import numpy as np
    from visuals import plot_map_view
    #
    # variables
    vars = ['Inf', 'Qv', 'R', 'RIE', 'ET', 'Tpgw', 'VSA']
    #
    pos_folder = 'C:/bin/pardinho/produtos/pos_batSLH_2021-11-27'
    pre_folder = 'C:/bin/pardinho/produtos/pre_batSLH_2021-11-27'
    folders = [pos_folder, pre_folder]
    pos_outfolder = 'C:/bin/pardinho/produtos/pos_uncertainty'
    pre_outfolder = 'C:/bin/pardinho/produtos/pre_uncertainty'
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
                if vars[i] + '_'  in f and 'Median' in f and '.asc' in f:
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


def bat_uncertainty_avg():
    import os
    import numpy as np
    from visuals import plot_map_view
    # variables
    vars = ['Inf', 'Qv', 'R', 'RIE', 'ET', 'Tpgw', 'VSA']
    #
    pos_folder = 'C:/bin/pardinho/produtos/pos_uncertainty'
    pos_all = os.listdir(pos_folder)
    pre_folder = 'C:/bin/pardinho/produtos/pre_uncertainty'
    pre_all = os.listdir(pre_folder)
    folders = [pos_folder, pre_folder]
    outfolder = 'C:/bin/pardinho/produtos/avg_uncertainty'
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


def view_anomaly(show=True):
    from visuals import _custom_cmaps
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    #
    folder = r"C:/000_myFiles/myDrive/myProjects/104_paper_castelhano/produtos/figs/anom"
    vars = ['R'] #['asl', 'pload', 'nload', 'R', 'RIE', 'Qv', 'Inf', 'ET', 'Tpgw', 'VSA']
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
            fpos = r"C:/bin/pardinho/produtos/pos_E6000_median_ASLA_2021-11-30/{}.asc".format(v)
            fpre = r"C:/bin/pardinho/produtos/pre_E6000_median_ASLA_2021-11-30/{}.asc".format(v)
            fanm = r"C:/bin/pardinho/produtos/anomaly/annual_{}_anomaly.asc".format(v)
        else:
            fpos = r"C:/bin/pardinho/produtos/pos_batSLH_2021-11-27/annual_{}_Median.asc".format(v)
            fpre = r"C:/bin/pardinho/produtos/pre_batSLH_2021-11-27/annual_{}_Median.asc".format(v)
            fanm = r"C:/bin/pardinho/produtos/anomaly/annual_{}_Median_anomaly.asc".format(v)
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


def sample_histograms(x, y, var='R'):
    folder = "C:/bin/pardinho/produtos/pos_batSLH_2021-11-27"
    all_files = os.listdir(folder)
    maps_lst = list()
    for f in all_files:
        if 'model' in f:
            lcl_file = '{}/{}/integration/raster_integral_{}.asc'.format(folder, f, var)
            print(lcl_file)
            meta, lcl_map = inp.asc_raster(file=lcl_file, dtype='float32')
            lcl_map = lcl_map[600:1200, 500:900]
            maps_lst.append(lcl_map)
    maps_array = np.array(maps_lst)
    # annualize:
    n_days = 1461
    annual_factor = n_days / 365
    maps_array = maps_array / annual_factor
    #
    #
    values = np.zeros(shape=(len(x), len(maps_array)))
    for i in range(len(values)):
        lcl_x = x[i]
        lcl_y = y[i]
        for j in range(len(values[i])):
            values[i][j] = maps_array[j][lcl_y][lcl_x]
    return values


def view_uncertainty(show=True):
    from visuals import _custom_cmaps
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    #
    _xs = [100, 150, 320, 290]
    _ys = [400, 100, 460, 300]
    labels = ['s1', 's2', 's3', 's4']

    folder = r"C:/000_myFiles/myDrive/myProjects/104_paper_castelhano/produtos/figs/unc"
    vars = ['R'] #, 'RIE', 'Qv', 'Inf', 'ET', 'Tpgw', 'VSA']
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
        fmed = r"C:/bin/pardinho/produtos/pos_batSLH_2021-11-27/annual_{}_Median.asc".format(v)
        frng = r"C:/bin/pardinho/produtos/pos_batSLH_2021-11-27/annual_{}_Range_90.asc".format(v)
        func = r"C:/bin/pardinho/produtos/pos_uncertainty/annual_{}_uncertainty.asc".format(v)
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
        im = plt.imshow(maps_lst[2], cmap='Greys', vmin=0, vmax=75)
        plt.title('uncertainty (%)')
        for p in range(len(_xs)):
            plt.plot(_xs[p], _ys[p], '.', color='tab:orange')
            plt.text(_xs[p] + 10, _ys[p] + 10, s=labels[p], color='tab:orange')
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


def zonal_stats_car():
    import matplotlib.pyplot as plt
    import pandas as pd
    import geo
    fcar = 'C:/bin/pardinho/produtos/aoi_car.asc'
    meta, car = inp.asc_raster(file=fcar, dtype='int16')
    _ids = np.unique(car)
    vars = ['R', 'RIE', 'Qv', 'Inf', 'ET', 'Tpgw', 'asl', 'pload', 'nload']
    _out_df = pd.DataFrame({'id_car':_ids})
    for v in vars:
        # annual anomaly
        if v in ['asl', 'pload', 'nload']:
            fanm = r"C:/bin/pardinho/produtos/anomaly/annual_{}_anomaly.asc".format(v)
        else:
            fanm = r"C:/bin/pardinho/produtos/anomaly/annual_{}_Median_anomaly.asc".format(v)
        meta, lcl_map = inp.asc_raster(file=fanm, dtype='float32')
        _dct = geo.zonalstats(field=lcl_map, zones=car, tui=True)
        _out_df['{}_aa_mean'.format(v)] = _dct['Mean']
        # average uncertainty
        if v in ['asl', 'pload', 'nload']:
            pass
        else:
            fanm = r"C:/bin/pardinho/produtos/avg_uncertainty/avg_{}_uncertainty.asc".format(v)
            meta, lcl_map = inp.asc_raster(file=fanm, dtype='float32')
            _dct = geo.zonalstats(field=lcl_map, zones=car, tui=True)
            _out_df['{}_un_mean'.format(v)] = _dct['Mean']
    print(_out_df.head(10).to_string())
    fout = 'C:/bin/pardinho/produtos/aoi_car_zonal_stats.txt'
    _out_df.to_csv(fout, sep=';', index=False)


def view_evolution_1():
    import matplotlib.pyplot as plt
    import pandas as pd
    full_f = r"C:\bin\pardinho\produtos\generations\population.txt"
    pop_df = pd.read_csv(full_f, sep=';')
    behav_f = r"C:\bin\pardinho\produtos\generations\behaviroural.txt"
    behav_df = pd.read_csv(behav_f, sep=';')
    select_f = r"C:\bin\pardinho\produtos\generations\selection.txt"
    select_df = pd.read_csv(select_f, sep=';')
    fig = plt.figure(figsize=(7, 7), )  # Width, Height
    plt.scatter(x=pop_df['L_ET'], y=pop_df['L_Q'], marker='.', c='tab:grey', alpha=0.4, edgecolors='none')
    plt.scatter(x=behav_df['L_ET'], y=behav_df['L_Q'], marker='.', c='black')
    plt.scatter(x=select_df['L_ET'], y=select_df['L_Q'], marker='.', c='magenta')
    plt.ylim((0, 0.75))
    plt.xlim((-1, -0.4))
    plt.grid(True)
    expfile = 'C:/bin/pardinho/produtos/generations/pop_zoom.png'
    plt.savefig(expfile, dpi=400)
    plt.close(fig)


def view_evolution_2():
    import matplotlib.pyplot as plt
    import pandas as pd
    fig = plt.figure(figsize=(7, 4), )  # Width, Height
    full_f = r"C:\bin\pardinho\produtos\generations\population.txt"
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
    expfile = 'C:/bin/pardinho/produtos/generations/evolution.png'
    plt.savefig(expfile, dpi=400)
    plt.close(fig)

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
                     run_ensemble=True,
                     folder='C:/bin/pardinho/produtos',
                     wkpl=True,
                     normalize=False,
                     tui=True)


def view_et_pannel(show=False):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import inp
    from hydrology import map_back
    from visuals import _custom_cmaps
    #
    folder = r"C:/000_myFiles/myDrive/myProjects/104_paper_castelhano/produtos/figs"
    _cmaps = _custom_cmaps()
    #
    etobs_sebal_raster_f = "C:/000_myFiles/myDrive/Plans3/pardinho/datasets/observed/etpat/calib_etpat_2013-08-05.asc"
    etobs_sampled_zmap_f = "C:/000_myFiles/myDrive/Plans3/pardinho/datasets/observed/etpat/zmap_etpat_2013-08-05.txt"
    ftwi = "C:/000_myFiles/myDrive/Plans3/pardinho/datasets/observed/calib_twi_window.asc"
    fshru = "C:/000_myFiles/myDrive/Plans3/pardinho/datasets/observed/calib_shru_window.asc"
    f_etsim_raster = "C:/bin/pardinho/optimization/000_calib_Hydrology_KGElog_2021-10-27-18-01-16/MLM/full_period/sim_ET/raster_ET_2013-08-05.asc"
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
    v_max = 3
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


def view_ensemble_et(show=True):
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    #
    folder = r"C:/000_myFiles/myDrive/myProjects/104_paper_castelhano/produtos/figs"
    #
    f_ensemble_et = "C:/bin/pardinho/produtos/GLUE_L_2021-12-06-14-31-40/ensemble_et.txt"
    f_ensemble_q = "C:/bin/pardinho/produtos/GLUE_L_2021-12-06-14-31-40/ensemble_q.txt"
    f_global_et = "C:/bin/pardinho/optimization/000_calib_Hydrology_KGElog_2021-10-27-18-01-16/MLM/full_period/osa_zmaps/analyst_sim_series.txt"
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


def view_ensemble_q(show=True):
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    #
    folder = r"C:/000_myFiles/myDrive/myProjects/104_paper_castelhano/produtos/figs"
    #
    f_ensemble_et = "C:/bin/pardinho/produtos/GLUE_L_2021-12-06-14-31-40/ensemble_et.txt"
    f_ensemble_q = "C:/bin/pardinho/produtos/GLUE_L_2021-12-06-14-31-40/ensemble_q.txt"
    f_global_et = "C:/bin/pardinho/optimization/000_calib_Hydrology_KGElog_2021-10-27-18-01-16/MLM/full_period/osa_zmaps/analyst_sim_series.txt"
    #
    #
    q_df = pd.read_csv(f_ensemble_q, sep=';', parse_dates=['Date'])
    q_obs_df = pd.read_csv(f_global_et, sep=';', parse_dates=['Date'])
    #print(et_df.head().to_string())
    fig = plt.figure(figsize=(16, 2.5))  # Width, Height
    plt.fill_between(x=q_df['Date'], y1=q_df['Lo_5'], y2=q_df['Hi_95'],
                     color='silver')
    plt.plot(q_df['Date'], q_df['Mid_50'], 'tab:blue')
    plt.plot(q_obs_df['Date'], q_obs_df['Qobs'], 'k.')
    plt.xlim((q_df['Date'].values[0], q_df['Date'].values[-1]))
    plt.ylim((0.001, 35))
    plt.yscale('log')
    plt.grid(True)
    #plt.ylabel('mm')
    filename = 'q_series'
    if show:
        plt.show()
        plt.close(fig)
    else:
        filepath = folder + '/' + filename + '.png'
        plt.savefig(filepath, dpi=400)
        plt.close(fig)


def view_maps_hist2(show=True):
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    import shapefile

    fbasic = r"C:\000_myFiles\myDrive\myProjects\104_paper_castelhano\produtos\aoi_car_basic.txt"
    fstats = r"C:\000_myFiles\myDrive\myProjects\104_paper_castelhano\produtos\aoi_car_zonal_stats.txt"
    fbasin = r"C:\000_myFiles\myDrive\gis\pnh\misc\aoi_basin.shp"

    basic_df = pd.read_csv(fbasic, sep=';')
    # print(basic_df.head().to_string())
    stats_df = pd.read_csv(fstats, sep=';')
    # print(stats_df.head().to_string())
    df = pd.merge(basic_df, stats_df, 'inner', left_on='id_imovel', right_on='id_car')
    print(df.head().to_string())
    #
    # read shapefile
    sf = shapefile.Reader(fbasin)
    vars = ['R', 'Qv'] #['R', 'Qv', 'RIE', 'Inf', 'ET', 'Tpgw', 'asl', 'nload', 'pload']
    _ptiles1 = [0.20, 0.40, 0.60, 0.80, 0.95]
    _ptiles2 = [0.05, 0.20, 0.40, 0.60, 0.80]
    ptiles = {'aa': {'R':_ptiles1,
                      'Qv':_ptiles2,
                     'RIE':_ptiles1,
                     'Inf':_ptiles2,
                     'ET':_ptiles2,
                     'Tpgw':_ptiles2,
                     'asl':_ptiles1,
                     'nload':_ptiles1,
                     'pload':_ptiles1},
              'un': {'R':_ptiles1,
                     'Qv':_ptiles1,
                     'RIE':_ptiles1,
                     'Inf':_ptiles1,
                     'ET':_ptiles1,
                     'Tpgw':_ptiles1,
                     'asl':_ptiles1,
                     'nload':_ptiles1,
                     'pload':_ptiles1}}
    _colors1 = ['aqua', 'lime', 'gold', 'orange', 'red', 'black']
    _colors2 = ['gold', 'black', 'red', 'orange', 'lime', 'aqua']
    _colors3 = ['whitesmoke', 'lightgrey', 'silver', 'darkgrey', 'grey', 'black']
    colors = {'aa':{'R':_colors1,
                    'Qv':_colors2,
                    'RIE':_colors1,
                    'Inf':_colors2,
                    'ET':_colors2,
                    'Tpgw':_colors2,
                    'asl':_colors1,
                    'nload':_colors1,
                    'pload':_colors1},
              'un': {'R':_colors3,
                     'Qv':_colors3,
                     'RIE':_colors3,
                     'Inf':_colors3,
                     'ET':_colors3,
                     'Tpgw':_colors3,
                     'asl':_colors3,
                     'nload':_colors3,
                     'pload':_colors3}}
    folders = {'aa': r"C:/000_myFiles/myDrive/myProjects/104_paper_castelhano/produtos/figs/anom",
               'un': r"C:/000_myFiles/myDrive/myProjects/104_paper_castelhano/produtos/figs/unc"}
    stats_types = ['un']#, 'aa'] #['un', 'aa']
    for stats in stats_types:
        folder = folders[stats]
        for v in vars:
            print(v)
            if stats == 'un' and v in ['asl', 'nload', 'pload']:
                pass
            else:
                # get ptiles
                _lcl_x = df['{}_{}_mean'.format(v, stats)].values
                _ptiles = np.quantile(a=_lcl_x, q=ptiles[stats][v])
                # plot histograms
                fig = plt.figure(figsize=(8, 3))
                _hist = plt.hist(x=_lcl_x, bins=100, color='grey')
                for i in range(len(_ptiles)):
                     plt.vlines(x=_ptiles[i], ymin=0, ymax=1.2 * np.max(_hist[0]), colors='tab:red')
                plt.ylim(0, 1.2 * np.max(_hist[0]))
                #if v in ['R', 'RIE', 'ET', 'Inf', 'Qv', 'Tpgw']:
                plt.xlim(0, 100)
                plt.title('{}'.format(v))
                # plt.xlim(-800, 800)
                if show:
                    plt.show()
                    plt.close(fig)
                else:
                    filepath = folder + '/hist/hist_{}_{}_mean.png'.format(v, stats)
                    plt.savefig(filepath, dpi=400)
                    plt.close(fig)
                # map
                fig = plt.figure(figsize=(6, 6))  # Width, Height
                # gs = mpl.gridspec.GridSpec(3, 6, wspace=0.1, hspace=0.1)
                #
                #
                ax = fig.add_subplot(111)
                patch = plt.Polygon(sf.shape(0).points, facecolor='none', edgecolor='black', linewidth=1)
                ax.add_patch(patch)
                ax.axis('scaled')
                ax.set_xticks([])
                ax.set_yticks([])
                _alphas = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
                _colors = colors[stats][v]
                for i in range(len(_ptiles) + 1):
                    if i == 0:
                        _lcl_df = df.query('{}_{}_mean <= {}'.format(v, stats, _ptiles[i]))
                    elif i > 0 and i <= len(_ptiles) - 1:
                        _lcl_df = df.query('{}_{}_mean <= {} and {}_{}_mean > {}'.format(v, stats, _ptiles[i], v, stats, _ptiles[i - 1]))
                    else:
                        _lcl_df = df.query('{}_{}_mean > {}'.format(v, stats, _ptiles[i - 1]))
                    plt.scatter(_lcl_df['long'], _lcl_df['lat'], c=_colors[i], alpha=_alphas[i], marker='.', edgecolors='none')
                ax.set_xlim([360694, 385296])
                ax.set_ylim([6721596, 6752258])
                plt.title('{}'.format(v))
                if show:
                    plt.show()
                    plt.close(fig)
                else:
                    filepath = folder + '/maps/map_{}_{}_mean.png'.format(v, stats)
                    plt.savefig(filepath, dpi=400)
                    plt.close(fig)


def view_maps_hist(show=False):
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    import shapefile
    folder1 = 'C:/000_myFiles/myDrive/myProjects/104_paper_castelhano/produtos'
    folder2 = 'C:/000_myFiles/myDrive/myProjects/104_paper_castelhano/produtos/figs/hists_maps'
    ffull = 'C:/000_myFiles/myDrive/myProjects/104_paper_castelhano/produtos/aoi_car_full.txt'
    fbasin = r"C:\000_myFiles\myDrive\gis\pnh\misc\aoi_basin.shp"

    df = pd.read_csv(ffull, sep=';')
    print(df.head(10).to_string())
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
                _lcl_x = df['{}_{}_mean'.format(v, stats)].values
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
                if stats == 'un':
                    if np.max(_lcl_x) > 100:
                        plt.xlim(0, 100)
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

def view_pre_pos():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    fpos = r"C:\bin\pardinho\produtos\pos_batSLH_2021-11-27\series_ensemble.txt"
    fpre = r"C:\bin\pardinho\produtos\pre_batSLH_2021-11-27\series_ensemble.txt"

    pos_df = pd.read_csv(fpos, sep=';', parse_dates=['Date'])
    pre_df = pd.read_csv(fpre, sep=';', parse_dates=['Date'])

    print(pos_df.head().to_string())
    vars = ['R', 'Q', 'Qb', 'ET', 'Qv', 'Inf', 'TF', 'RIE', 'RSE', 'Tpgw']

    pre_50 = list()
    pos_50 = list()
    pre_rng = list()
    pos_rng = list()
    for v in vars:
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
    plt.bar(x - width / 2, pre_50, width, yerr=np.array(pre_rng) / 2, label='Pre', color='tab:green')
    plt.bar(x + width / 2, pos_50, width, yerr=np.array(pos_rng) / 2, label='Pos', color='tab:blue')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel('mm')
    plt.xticks(x, vars)
    plt.legend()
    plt.show()

    for v in vars:
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
        plt.plot(pos_df['Date'], pos_df['{}_50'.format(v)], 'tab:blue', label='post')
        plt.plot(pre_df['Date'], pre_df['{}_50'.format(v)], 'tab:green', label='pre')
        if v == 'Q':
            plt.yscale('log')
        plt.xlim((pre_df['Date'].values[0], pre_df['Date'].values[-1]))
        plt.legend(loc='upper right')
        plt.show()