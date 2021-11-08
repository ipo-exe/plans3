''' 
UFRGS - Universidade Federal do Rio Grande do Sul
IPH - Instituto de Pesquisas Hidráulicas
WARP - Research Group in Water Resources Management and Planning
Porto Alegre, Rio Grande do Sul, Brazil

plans - planning nature-based solutions
Version: 3.0

This software is under the GNU GPL3.0 license

Source code repository: https://github.com/ipo-exe/plans3/
Authors: Ipora Possantti: https://github.com/ipo-exe

This file is under LICENSE: GNU General Public License v3.0
Permissions:
    Commercial use
    Modification
    Distribution
    Patent use
    Private use 
Limitations:
    Liability
    Warranty 
Conditions:
    License and copyright notice
    State changes
    Disclose source
    Same license 

Module description:
This module stores all frontend functions of plans3. 
Input parameters are all strings and booleans.
'''

import numpy as np
import pandas as pd

import backend
import inp, out, geo
import visuals
from out import export_report
from inp import dataframe_prepro
#import matplotlib.pyplot as plt
#from scipy.ndimage.filters import gaussian_filter


# this is a mess:
def view_imported_input(filename, folder, aux_folder=''):
    """

    Function to view an imported map

    :param filename: string file name of raster asc file
    :param folder: string filepath of source folder
    :param aux_folder: string file path of extra folder to extract auxiliar files
    :return: none
    """

    from inp import dataframe_prepro
    from os import listdir
    from backend import get_stringfields
    from visuals import plot_map_view, plot_qmap_view, pannel_calib_series, plot_shrumap_view, pannel_aoi_series

    def plot_lulc(fraster, fparams, filename, mapid='LULC'):
        lulc_param_df = pd.read_csv(fparams, sep=';', engine='python')
        lulc_param_df = dataframe_prepro(lulc_param_df, get_stringfields(fparams.split('/')[-1]))
        meta, rmap = inp.asc_raster(fraster)
        ranges = (np.min(lulc_param_df['IdLULC']), np.max(lulc_param_df['IdLULC']))
        plot_qmap_view(rmap,
                       meta,
                       colors=lulc_param_df['ColorLULC'].values,
                       names=lulc_param_df['LULCName'].values,
                       mapid=mapid,
                       ranges=ranges,
                       filename=filename,
                       folder=folder)

    # get file path
    file = folder + '/' + filename
    #
    quantmaps = ('aoi_dem.asc', 'aoi_basin.asc', 'aoi_twi.asc', 'calib_twi.asc',
                 'calib_dem.asc', 'calib_basin.asc')
    lulc_maps = ('aoi_lulc.asc', 'calib_lulc.asc')
    #
    # plot quant map
    if filename in set(quantmaps):
        mapid = filename.split('.')[0].split('_')[1]
        meta, rmap = inp.asc_raster(file)
        ranges = (np.min(rmap), np.max(rmap))
        plot_map_view(rmap, meta, ranges,
                      mapid=mapid,
                      filename=filename.split('.')[0],
                      folder=folder,
                      metadata=True)
    #
    # plot calib lulc from raster file
    elif filename == 'calib_lulc.asc':
        folder_files = listdir(folder)
        aux_filename = 'calib_lulc_param.txt'
        if aux_filename in set(folder_files):
            fraster = '{}/{}'.format(folder, filename)
            fparam = '{}/{}'.format(folder, aux_filename)
            plot_lulc(fraster=fraster, fparams=fparam, filename='calib_lulc')
    #
    # plot calib lulc from param file
    elif filename == 'calib_lulc_param.txt':
        folder_files = listdir(folder)
        aux_filename = 'calib_lulc.asc'
        if aux_filename in set(folder_files):
            fraster = '{}/{}'.format(folder, aux_filename)
            fparam = '{}/{}'.format(folder, filename)
            plot_lulc(fraster=fraster, fparams=fparam, filename='calib_lulc')
    #
    # plot aoi lulc from raster file
    elif filename == 'aoi_lulc.asc':
        folder_files = listdir(folder)
        aux_filename = 'aoi_lulc_param.txt'
        if aux_filename in set(folder_files):
            fraster = '{}/{}'.format(folder, filename)
            fparam = '{}/{}'.format(folder, aux_filename)
            plot_lulc(fraster=fraster, fparams=fparam, filename='aoi_lulc')
    #
    # plot calib lulc from param file
    elif filename == 'aoi_lulc_param.txt':
        folder_files = listdir(folder)
        aux_filename = 'aoi_lulc.asc'
        if aux_filename in set(folder_files):
            fraster = '{}/{}'.format(folder, aux_filename)
            fparam = '{}/{}'.format(folder, filename)
            plot_lulc(fraster=fraster, fparams=fparam, filename='aoi_lulc')
    #
    # plot calib soils from raster file
    elif filename == 'calib_soils.asc':
        folder_files = listdir(folder)
        aux_filename = 'calib_soils_param.txt'
        if aux_filename in set(folder_files):
            flulcparam = '{}/{}'.format(folder, aux_filename)
            soils_param_df = pd.read_csv(flulcparam, sep=';', engine='python')
            soils_param_df = dataframe_prepro(soils_param_df, 'SoilName,ColorSoil')
            meta, rmap = inp.asc_raster(file)
            ranges = (np.min(soils_param_df['IdSoil']), np.max(soils_param_df['IdSoil']))
            plot_qmap_view(rmap, meta,
                           colors=soils_param_df['ColorSoil'].values,
                           names=soils_param_df['SoilName'].values,
                           mapid='Soils',
                           ranges=ranges,
                           filename='calib_soils',
                           folder=folder)
    #
    # plot calib soils from param file
    elif filename == 'calib_soils_param.txt':
        folder_files = listdir(folder)
        aux_filename = 'calib_soils.asc'
        if aux_filename in set(folder_files):
            flulcparam = '{}/{}'.format(folder, filename)
            soils_param_df = pd.read_csv(flulcparam, sep=';', engine='python')
            soils_param_df = dataframe_prepro(soils_param_df, 'SoilName,ColorSoil')
            meta, rmap = inp.asc_raster('{}/{}'.format(folder, aux_filename))
            ranges = (np.min(soils_param_df['IdSoil']), np.max(soils_param_df['IdSoil']))
            plot_qmap_view(rmap, meta,
                           colors=soils_param_df['ColorSoil'].values,
                           names=soils_param_df['SoilName'].values,
                           mapid='Soils',
                           ranges=ranges,
                           filename='calib_soils',
                           folder=folder)
    #
    # plot aoi soils from raster map
    elif filename == 'aoi_soils.asc':
        folder_files = listdir(folder)
        aux_filename = 'aoi_soils_param.txt'
        if aux_filename in set(folder_files):
            flulcparam = '{}/{}'.format(folder, aux_filename)
            soils_param_df = pd.read_csv(flulcparam, sep=';', engine='python')
            soils_param_df = dataframe_prepro(soils_param_df, 'SoilName,ColorSoil')
            meta, rmap = inp.asc_raster(file)
            ranges = (np.min(soils_param_df['IdSoil']), np.max(soils_param_df['IdSoil']))
            plot_qmap_view(rmap, meta,
                           colors=soils_param_df['ColorSoil'].values,
                           names=soils_param_df['SoilName'].values,
                           mapid='Soils',
                           ranges=ranges,
                           filename='aoi_soils',
                           folder=folder)
    #
    # plot aoi soils from param file
    elif filename == 'aoi_soils_param.txt':
        folder_files = listdir(folder)
        aux_filename = 'aoi_soils.asc'
        if aux_filename in set(folder_files):
            flulcparam = '{}/{}'.format(folder, filename)
            soils_param_df = pd.read_csv(flulcparam, sep=';', engine='python')
            soils_param_df = dataframe_prepro(soils_param_df, 'SoilName,ColorSoil')
            meta, rmap = inp.asc_raster('{}/{}'.format(folder, aux_filename))
            ranges = (np.min(soils_param_df['IdSoil']), np.max(soils_param_df['IdSoil']))
            plot_qmap_view(rmap, meta,
                           colors=soils_param_df['ColorSoil'].values,
                           names=soils_param_df['SoilName'].values,
                           mapid='Soils',
                           ranges=ranges,
                           filename='aoi_soils',
                           folder=folder)
    #
    # plot calib series
    elif filename == 'calib_series.txt':
        series_df = pd.read_csv(file, sep=';')
        series_df = dataframe_prepro(series_df, strf=False, date=True, datefield='Date')
        pannel_calib_series(series_df, filename=filename.split('.')[0], folder=folder, show=False)
    #
    # plot calib series
    elif filename == 'aoi_series.txt':
        series_df = pd.read_csv(file, sep=';')
        series_df = dataframe_prepro(series_df, strf=False, date=True, datefield='Date')
        pannel_aoi_series(series_df, filename=filename.split('.')[0], folder=folder, show=False)


def export_local_pannels(ftwi, fshru,
                         folder='C:/bin',
                         chagezmapfolder=False,
                         zmapfolder='',
                         frametype='all',
                         filter_date=False,
                         date_init='2011-01-01',
                         date_end = '2011-04-01',
                         tui=False):
    """

    Export frames of local variables

    :param ftwi: string filepath to twi raster map
    :param fshru: string filepath to twi raster map
    :param folder: string filepath to simulation/output folder. The folder must contain all series files:
    'sim_series.txt' and all 'sim_zmaps_series_VAR.txt' VAR files
    :param chagezmapfolder: boolean to change zmap folder
    :param zmapfolder: string folder path to zmaps
    :param frametype: string code to frametypes. options: 'all', 'ET', 'Qv' and 'R"
    :param filter_date: boolean to filter dates
    :param date_init: string initial date to filter
    :param date_end: string end date to filter
    :param tui: boolean to display
    :return: none
    """
    from backend import get_all_lclvars
    from hydrology import map_back
    from visuals import pannel_local
    from os import mkdir
    if tui:
        from tui import status
    #
    # load heavy inputs
    if tui:
        status('loading raster maps')
    meta, twi = inp.asc_raster(ftwi, dtype='float32')
    meta, shru = inp.asc_raster(fshru, dtype='float32')
    #
    #
    # set view port
    upper_y = 0
    lower_x = 0
    size = 500
    twi = twi[upper_y: upper_y + size, lower_x: lower_x + size]
    shru = shru[upper_y: upper_y + size, lower_x: lower_x + size]
    #
    #
    # load series
    fseries = folder + '/sim_series.txt'
    series = pd.read_csv(fseries, sep=';', parse_dates=['Date'])
    #
    #
    # filter dates (remove this)

    if filter_date:
        query_str = 'Date > "{}" and Date < "{}"'.format(date_init, date_end)
        series = series.query(query_str)
    #
    dates_labels = pd.to_datetime(series['Date'], format='%Y%m%d')
    dates_labels = dates_labels.astype('str')
    #
    #
    if frametype == 'all':
        vars = 'D-ET-IRA-IRI-Tpun-Tpgw-Evc-Evs-Qv-Inf-Cpy-Unz-Sfs-TF-R-RIE-RSE-VSA'.split('-')
    elif frametype == 'ET':
        vars = 'D-ET-IRA-IRI-Tpun-Tpgw-Evc-Evs'.split('-')
    elif frametype == 'Qv':
        vars = 'D-Qv-IRA-IRI-Inf-Cpy-Unz-Sfs'.split('-')
    elif frametype == 'R':
        vars = 'D-R-IRA-IRI-TF-RIE-RSE-VSA'.split('-')
    # 1) load zmap series dataframes in a dict for each
    zmaps_series = dict()
    for var in vars:
        lcl_file = '{}/sim_zmaps_series_{}.txt'.format(folder, var)  # next will be sim_zmaps_series_{}.txt
        lcl_df = pd.read_csv(lcl_file, sep=';', parse_dates=['Date'])
        if filter_date:
            lcl_df = lcl_df.query(query_str)
        # print(lcl_df.tail().to_string())
        zmaps_series[var] = lcl_df.copy()
    #
    # todo improve to avoid memory crash - import rasters on the fly - get max min from zmaps
    # load rasters from zmap files
    raster_series = dict()
    rasters_maxval = dict()
    rasters_minval = dict()
    for var in vars:
        lcl_df = zmaps_series[var]
        raster_list = list()
        if tui:
            status('computing raster maps of {} ... '.format(var))
        for i in range(len(series)):
            lcl_file = lcl_df['File'].values[i]
            if chagezmapfolder:
                aux_lst = lcl_file.split('/')
                lcl_file = zmapfolder + '/' + aux_lst[len(aux_lst -1)]
                print(lcl_file)
            lcl_zmap, ybins, xbins = inp.zmap(lcl_file)
            lcl_raster = map_back(lcl_zmap, a1=twi, a2=shru, bins1=ybins, bins2=xbins)
            #print(i)
            #plt.imshow(lcl_raster)
            #plt.show()
            raster_list.append(lcl_raster)
        raster_nd = np.array(raster_list)
        raster_series[var] = raster_nd
        rasters_minval[var] = np.min(raster_nd)
        rasters_maxval[var] = np.max(raster_nd)
        if tui:
            status('{} maps loaded'.format(len(raster_nd)))
    #
    offsetback = 0
    offsetfront = 0
    #
    # plot ET pannels:
    if frametype == 'all' or frametype == 'ET':
        lcl_folder = '{}/ET_frames'.format(folder)
        mkdir(lcl_folder)
        vars = 'D-ET-IRA-IRI-Tpun-Tpgw-Evc-Evs'.split('-')
        star = 'ET'
        mids = ('Evc', 'Evs', 'Tpun', 'Tpgw')
        prec_rng = (0, np.max(series['Prec'].values))
        irri_rng = (0, np.max((series['IRA'].values, series['IRI'].values)))
        for t in range(len(series)):
            if tui:
                status('ET pannel t = {} | plotting date {}'.format(t, dates_labels.values[t]))
            pannel_local(series, star=raster_series[star][t],
                         deficit=raster_series['D'][t],
                         sups=[raster_series['IRA'][t], raster_series['IRI'][t]],
                         mids=[raster_series[mids[0]][t], raster_series[mids[1]][t],
                               raster_series[mids[2]][t], raster_series[mids[3]][t]],
                         star_rng=(rasters_minval[star], rasters_maxval[star]),
                         deficit_rng=(rasters_minval['D'], rasters_maxval['D']),
                         sup1_rng=prec_rng,
                         sup2_rng=irri_rng,
                         sup3_rng=irri_rng,
                         sup4_rng=irri_rng,
                         mid1_rng=(rasters_minval['ET'], rasters_maxval['ET']),
                         mid2_rng=(rasters_minval['ET'], rasters_maxval['ET']),
                         mid3_rng=(rasters_minval['ET'], rasters_maxval['ET']),
                         mid4_rng=(rasters_minval['ET'], rasters_maxval['ET']),
                         t=t,
                         type=star,
                         show=False,
                         offset_back=offsetback,
                         offset_front=offsetfront,
                         folder=lcl_folder)
    #
    #
    # plot QV pannels:
    if frametype == 'all' or frametype == 'Qv':
        lcl_folder = '{}/Qv_frames'.format(folder)
        mkdir(lcl_folder)
        vars = 'D-Qv-IRA-IRI-Inf-Cpy-Sfs-Unz'.split('-')
        star = 'Qv'
        mids = ('Inf', 'Cpy', 'Sfs', 'Unz')
        prec_rng = (0, np.max(series['Prec'].values))
        irri_rng = (0, np.max((series['IRA'].values, series['IRI'].values)))
        stockmax = np.max((rasters_maxval['Unz'], rasters_maxval['Cpy'], rasters_maxval['Sfs']))
        for t in range(len(series)):
            if tui:
                status('Qv pannel t = {} | plotting date {}'.format(t, dates_labels.values[t]))
            pannel_local(series, star=raster_series[star][t],
                         deficit=raster_series['D'][t],
                         sups=[raster_series['IRA'][t], raster_series['IRI'][t]],
                         mids=[raster_series[mids[0]][t], raster_series[mids[1]][t],
                               raster_series[mids[2]][t], raster_series[mids[3]][t]],
                         star_rng=(0, rasters_maxval['Inf']),
                         deficit_rng=(rasters_minval['D'], rasters_maxval['D']),
                         sup1_rng=prec_rng,
                         sup2_rng=irri_rng,
                         sup3_rng=irri_rng,
                         sup4_rng=irri_rng,
                         mid1_rng=(0, rasters_maxval['Inf']),
                         mid2_rng=(0, rasters_maxval['Cpy']),
                         mid3_rng=(0, rasters_maxval['Sfs']),
                         mid4_rng=(0, rasters_maxval['Unz']),
                         t=t,
                         type=star,
                         show=False,
                         offset=False,
                         offset_back=offsetback,
                         offset_front=offsetfront,
                         folder=lcl_folder)
        #
    # plot R pannels:
    if frametype == 'all' or frametype == 'R':
        lcl_folder = '{}/R_frames'.format(folder)
        mkdir(lcl_folder)
        vars = 'D-TF-IRA-IRI-R-RIE-RSE-VSA'.split('-')
        star = 'R'
        mids = ('TF', 'RIE', 'RSE', 'VSA')
        prec_rng = (0, np.max(series['Prec'].values))
        irri_rng = (0, np.max((series['IRA'].values, series['IRI'].values)))
        for t in range(len(series)):
            if tui:
                status('R pannel t = {} | plotting date {}'.format(t, dates_labels.values[t]))
            pannel_local(series, star=raster_series[star][t],
                         deficit=raster_series['D'][t],
                         sups=[raster_series['IRA'][t], raster_series['IRI'][t]],
                         mids=[raster_series[mids[0]][t], raster_series[mids[1]][t],
                               raster_series[mids[2]][t], raster_series[mids[3]][t]],
                         star_rng=(0, rasters_maxval['TF']),
                         deficit_rng=(rasters_minval['D'], rasters_maxval['D']),
                         sup1_rng=prec_rng,
                         sup2_rng=irri_rng,
                         sup3_rng=irri_rng,
                         sup4_rng=irri_rng,
                         mid1_rng=(0, rasters_maxval['TF']),
                         mid2_rng=(0, rasters_maxval['TF']),
                         mid3_rng=(0, rasters_maxval['TF']),
                         mid4_rng=(0, 1.0),
                         t=t,
                         type=star,
                         show=False,
                         offset_back=offsetback,
                         offset_front=offsetfront,
                         folder=lcl_folder)


def map_shru(flulc, flulcparam, fsoils, fsoilsparam, fshruparam, folder='C:/bin', filename='shru', viewlabel=''):
    """

    Function to get the SHRU raster map.

    :param flulc: string file path to lulc .asc raster file
    :param flulcparam: string file path to lulc parameters .txt file. Separator = ;
    :param fsoils: string path to soils.asc raster file
    :param fsoilsparam: string path to soils parameters .txt file. Separator = ;
    :param folder: string path to destination folder
    :param filename: string name of file
    :return: string file path
    """
    from visuals import plot_shrumap_view
    from backend import get_stringfields

    #
    # import data
    metalulc, lulc = inp.asc_raster(flulc)
    #print(np.shape(lulc))
    metasoils, soils = inp.asc_raster(fsoils)
    #print(np.shape(soils))
    lulc_param_df = pd.read_csv(flulcparam, sep=';', engine='python')
    lulc_param_df = dataframe_prepro(lulc_param_df, get_stringfields(flulcparam.split('/')[-1]))
    soils_param_df = pd.read_csv(fsoilsparam, sep=';', engine='python')
    soils_param_df = dataframe_prepro(soils_param_df, get_stringfields(fsoilsparam.split('/')[-1]))
    lulc_ids = lulc_param_df['IdLULC'].values
    soils_ids = soils_param_df['IdSoil'].values
    shru_df = pd.read_csv(fshruparam, sep=';')
    shru_df = dataframe_prepro(shru_df, get_stringfields(fshruparam.split('/')[-1]))
    #
    # process data
    shru_map = geo.xmap(map1=lulc, map2=soils, map1ids=lulc_ids, map2ids=soils_ids, map1f=100, map2f=1)
    #plt.imshow(shru_map)
    #plt.show()
    title = 'SHRU'
    if viewlabel != '':
        title = ' SHRU | {}'.format(viewlabel)
    plot_shrumap_view(lulc, soils, metalulc, shru_df, filename=filename, folder=folder,
                      metadata=True, ttl=title)
    # export data
    export_file = out.asc_raster(shru_map, metalulc, folder, filename, dtype='int16')
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
    meta, soils = inp.asc_raster(fsoils)
    soils_df = pd.read_csv(fsoilsparam, sep=';', engine='python')
    soils_df = dataframe_prepro(soils_df, strfields='SoilName,ColorSoil')
    # process data
    fto = geo.reclassify(soils, upvalues=soils_df['IdSoil'].values, classes=soils_df['f_To'].values)
    #plt.imshow(fto)
    #plt.show()
    # export data
    export_file = out.asc_raster(fto, meta, folder, filename)
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
    meta, dem = inp.asc_raster(fdem)
    #plt.imshow(dem)
    #plt.show()
    # process data
    slp = geo.slope(dem, meta['cellsize'], degree=True)
    #plt.imshow(slp)
    #plt.show()
    #
    # export
    export_file = out.asc_raster(slp, meta, folder, filename)
    ranges = (np.min(slp), np.max(slp))
    plot_map_view(slp, meta, ranges, mapid='slope', filename=filename, folder=folder, metadata=True)
    return export_file


def map_twi(fslope, fcatcha, ffto, folder='C:/bin', filename='twi'):
    """
    Derive the Topographical Wetness Index of TOPMODEL (Beven & Kirkby, 1979)

    TWI =  ln ( a / To tanB )

    Where a is the specific catchment area, To is the local transmissivity and tanB is the terrain gradient.


    :param fslope: string path to slope in degrees raster .asc file
    :param fcatcha: string path to catchment area in squared meters raster .asc file
    :param ffto: string path to transmissivity factor (f_To) raster .asc file
    :param folder: string path to destination folder
    :param filename: string of file name
    :return: string path to file
    """
    from visuals import plot_map_view
    # import data
    meta, slope = inp.asc_raster(fslope)
    meta, catcha = inp.asc_raster(fcatcha)
    meta, fto = inp.asc_raster(ffto, dtype='float32')
    # process data
    grad = geo.grad(slope)
    twi = geo.twi(catcha, grad, fto, cellsize=meta['cellsize'])
    #plt.imshow(twi)
    #plt.show()
    # export data
    export_file = out.asc_raster(twi, meta, folder, filename)
    ranges = (np.min(twi), np.max(twi))
    plot_map_view(twi, meta, ranges, mapid='twi', filename=filename, folder=folder, metadata=True)
    return export_file


def map_twi_hand_long(fslope, fcatcha, ffto, fhand,
                      hand_hi=15.0,
                      hand_lo=0.0,
                      hand_w=1,
                      twi_w=1,
                      folder='C:/bin',
                      filename='etwi'):
    """

    Derive the HAND-enhanced TWI raster map. Short version.

    :param fslope: string file path to the Slope asc raster map
    :param fcatcha: string file path to the Catcha asc raster map
    :param ffto: string file path to the fTo asc raster map
    :param fhand: string file path to the HAND asc raster map
    :param hand_hi: float of HAND higher threshold
    :param hand_lo: float of HAND lower threshold
    :param hand_w: float of HAND compouding weight
    :param twi_w: float of TWI compouding weight
    :param folder: string path to output folder
    :param filename: string of file name
    :return: string file path of output file
    """
    from visuals import plot_map_view
    # import maps
    meta, slope = inp.asc_raster(fslope)
    meta, catcha = inp.asc_raster(fcatcha)
    meta, fto = inp.asc_raster(ffto, dtype='float32')
    meta, hand = inp.asc_raster(fhand)
    #
    # process
    twi_hand = geo.twi_hand_long(catcha=catcha,
                                 slope=slope,
                                 fto=fto,
                                 hand=hand,
                                 cellsize=meta['cellsize'],
                                 hand_hi=hand_hi,
                                 hand_lo=hand_lo,
                                 hand_w=hand_w,
                                 twi_w=twi_w)
    # export data
    export_file = out.asc_raster(twi_hand, meta, folder, filename)
    # export plot
    ranges = (np.min(twi_hand), np.max(twi_hand))
    plot_map_view(twi_hand, meta, ranges, mapid='twi', filename=filename, folder=folder, metadata=True)
    return export_file


def map_twi_hand_short(ftwi, fhand, hand_hi=15.0, hand_lo=0.0, hand_w=1, twi_w=1, folder='C:/bin', filename='etwi'):
    """

    Derive the HAND-enhanced TWI raster map. Short version.

    :param ftwi: string file path to the TWI asc raster map
    :param fhand: string file path to the HAND asc raster map
    :param hand_hi: float of HAND higher threshold
    :param hand_lo: float of HAND lower threshold
    :param hand_w: float of HAND compouding weight
    :param twi_w: float of TWI compouding weight
    :param folder: string path to output folder
    :param filename: string of file name
    :return: string file path of output file
    """
    from visuals import plot_map_view
    # import maps
    meta, twi = inp.asc_raster(ftwi, dtype='float32')
    meta, hand = inp.asc_raster(fhand)
    #
    # process
    twi_hand = geo.twi_hand_short(twi=twi,
                                  hand=hand,
                                  cellsize=meta['cellsize'],
                                  hand_hi=hand_hi,
                                  hand_lo=hand_lo,
                                  hand_w=hand_w,
                                  twi_w=twi_w)
    # export data
    export_file = out.asc_raster(twi_hand, meta, folder, filename)
    # export plot
    ranges = (np.min(twi_hand), np.max(twi_hand))
    plot_map_view(twi_hand, meta, ranges, mapid='twi', filename=filename, folder=folder, metadata=True)
    return export_file


def map_twito(ftwi, ffto, folder='C:/bin', filename='twito'):
    """
    Derive the complete TWI raster map by inserting the To (transmissivity) term to the TWI formula.

    TWI =  ln ( a / To tanB ) =  ln ( a / tanB ) + ln ( 1 / To )

    :param ftwi: string filepath to TWI asc raster map computed without the To term, i.e., only ln ( a / tanB )
    :param ffto: string filepath to fTo asc raster map
    :param folder: string filepath to output folder
    :param filename: string of output file name
    :return: string filepath to output asc raster map
    """
    from visuals import plot_map_view
    # import
    meta, twi = inp.asc_raster(ftwi, dtype='float32')
    meta, fto = inp.asc_raster(ffto, dtype='float32')
    # process
    twito = twi + np.log(1 / fto)
    # export
    exp_file = out.asc_raster(twito, meta, folder, filename)
    # plot
    ranges = (np.min(twito), np.max(twito))
    plot_map_view(twito, meta, ranges, mapid='twi', filename=filename, folder=folder, metadata=True)
    return exp_file


def compute_histograms(fshruparam, fshru, ftwi, faoi='none', ntwibins=20, folder='C:/bin', filename='histograms',
                       tui=False):
    """
    Compute the 2d histogram
    :param fshruparam: string
    :param fshru: string
    :param ftwi: string
    :param faoi: string
    :param ntwibins: int
    :param folder: string
    :param filename: string
    :param tui: boolean
    :return:
    """
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
    meta, shru = inp.asc_raster(fshru, dtype='float32')
    #
    # import twi raster
    if tui:
        status('loading twi raster')
    meta, twi = inp.asc_raster(ftwi, dtype='float32')
    #
    if faoi == 'none':
        aoi = aoi = 1.0 + (twi * 0.0)
    else:
        # import twi raster
        if tui:
            status('loading aoi raster')
        meta, aoi = inp.asc_raster(faoi)
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


def import_map_series(fmapseries,
                      rasterfolder='C:/bin',
                      folder='C:/bin',
                      filename='map_series',
                      rasterfilename='map',
                      view=True):
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
        # plot view
        if view:
            view_imported_input(lcl_filenm, rasterfolder, folder)
    #
    # export data
    exp_df = pd.DataFrame({'Date':dates, 'File':new_files})
    exp_file = folder + '/' + filename + '.txt'
    exp_df.to_csv(exp_file, sep=';', index=False)
    return exp_file


def import_etpat_series(finputseries, rasterfolder='C:/bin', folder='C:/bin', filename='map_series',
                        rasterfilename='map', nodata=-1.0, normalize=True, tui=False):
    """
    Batch import function for ET Pat raster maps

    :param finputseries: string file path to input map series file
    :param rasterfolder: string file path to output raster folder
    :param folder: string file path to output folder
    :param filename: string name of output map series file
    :param rasterfilename: raster file name (date is appended)
    :param nodata: float of no data
    :param normalize: boolean to normalize the maps (ET pat). Otherwise Et pat is considered the observed ET, 
    :param tui: boolean to printouts
    :return: string file path to map series file
    """
    if normalize:
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
            status('importing rasters')
        # load all rasters
        rasters_lst = list()
        for i in range(len(dates)):
            src = files[i]
            meta, lcl_raster = inp.asc_raster(src, dtype='float32')
            rasters_lst.append(lcl_raster)
        rasters = np.array(rasters_lst)
        if tui:
            status('normalizing rasters')
        from hydrology import flatten_clear
        from geo import fuzzy_transition
        for e in range(len(rasters)):
            a_value = np.min(rasters[e])
            b_value = np.max(rasters[e])
            rasters[e] = fuzzy_transition(rasters[e], a=a_value, b=b_value, type='senoid')
        #
        # export data
        new_files = list()
        for i in  range(len(dates)):
            lcl_date = dates[i]
            lcl_filename = rasterfilename + '_' + lcl_date
            lcl_file = out.asc_raster(rasters[i], meta, folder=rasterfolder, filename=lcl_filename)
            new_files.append(lcl_file)
        #
        exp_df = pd.DataFrame({'Date': dates, 'File': new_files})
        exp_file = folder + '/' + filename + '.txt'
        exp_df.to_csv(exp_file, sep=';', index=False)
        #
        # view
        view_rasters(exp_file, mapvar='ETpat', mapid='etpat', vmin=0, vmax=1, tui=tui, dtype='float32', nodata=nodata)
    else:
        # just import rasters
        exp_file = import_map_series(fmapseries=finputseries, filename='calib_etpat_series', rasterfolder=rasterfolder,
                                     folder=folder, rasterfilename=rasterfilename, view=False)
        #
        # view
        view_rasters(exp_file, mapvar='ET', mapid='flow_v', vmin=0, vmax=10, tui=tui, dtype='float32', nodata=nodata)
    return exp_file


def view_rasters(fmapseries, mapvar='ET', mapid='etpat', vmin='local', vmax='local', tui=False, dtype='int16', nodata=-1):
    """
    Batch routine to plot raster maps from map series file

    :param fmapseries: string file path to map series txt file
    :param mapvar: string code of map variable
    :param mapid: string code of mapid
    :param vmin: float of min value on plot or string 'local' to get the local value
    :param vmax: float of max value on plot or string 'local' to get the local value
    :param tui: boolean of tui display
    :return: none
    """
    from visuals import plot_map_view
    import os
    # import main data
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
        # open map
        meta, lcl_map = inp.asc_raster(files[i], dtype=dtype)
        # set range
        if vmin == 'local':
            v_min = np.min(lcl_map)
        else:
            v_min = float(vmin)
        if vmax == 'local':
            v_max = np.max(lcl_map)
        else:
            v_max = float(vmax)
        ranges = [v_min, v_max]
        # plot local map
        plot_map_view(lcl_map, meta, ranges, mapid, mapttl='{} | {}'.format(mapvar, dates[i]),
                      filename=lcl_filename, folder=lcl_folder, nodata=nodata)


def compute_zmap_series(fvarseries, ftwi, fshru, fhistograms, var, filename='var_zmap_series', folder='C:/bin',
                        tui=False, dtype='int16', factor=1.0):
    """
    Batch routine to compute zmaps from raster series

    :param fvarseries: string file path to variable series
    :param ftwi: string file path to TWI asc raster map
    :param fshru: string file path to SHRU asc raster map
    :param fhistograms: string file path to histograms txt file
    :param filename: string of output series txt file name
    :param folder: string path to output folder
    :param tui: boolean to tui display
    :param dtype: string of input raster dtype
    :param factor: float of raster factor (to divide by during import)
    :return: string file path to map series txt file
    """
    from out import zmap
    from inp import histograms
    from hydrology import built_zmap
    import os
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
    meta, twi = inp.asc_raster(ftwi, dtype='float32')
    meta, shru = inp.asc_raster(fshru, dtype='float32')
    #
    if tui:
        status('loading histograms')
    count, twibins, shrubins = histograms(fhistograms=fhistograms)
    #
    # process data
    if tui:
        status('computing zmaps')
    new_files = list()
    for i in range(len(dates)):
        if tui:
            status('  builting zmap of {}'.format(dates[i]))
        #
        lcl_folder = os.path.dirname(files[i])
        lcl_filenm = os.path.basename(files[i])
        lcl_new_filename = 'zmap_' +  var + '_' + dates[i]
        # get raster
        meta, lcl_var = inp.asc_raster(files[i], dtype=dtype)
        # use factor to access values
        if factor != 1.0:
            lcl_var = lcl_var / factor
        lcl_zmap = built_zmap(varmap=lcl_var, twi=twi, shru=shru, twibins=twibins, shrubins=shrubins)
        exp_file = out.zmap(zmap=lcl_zmap, twibins=twibins, shrubins=shrubins, folder=lcl_folder,
                            filename=lcl_new_filename)
        new_files.append(exp_file)
    #
    # export data
    exp_df = pd.DataFrame({'Date': dates, 'File': new_files})
    exp_file = folder + '/' + filename + '.txt'
    exp_df.to_csv(exp_file, sep=';', index=False)
    return exp_file

# this may be deprecated
def import_shru_series(flulcseries, flulcparam, fsoils, fsoilsparam, fshruparam, rasterfolder='C:/bin', folder='C:/bin',
                       filename='shru_series', suff='', tui=False):
    """
    Batch routine to compute SHRU series maps

    :param flulcseries: string file path to LULC series txt file
    :param flulcparam: string file path to LULC parameters dataframe txt file
    :param fsoils: string file path to Soils series txt file
    :param fsoilsparam: string file path to Soils parameters dataframe txt file
    :param fshruparam: string file path to SHRU parameters dataframe txt file
    :param rasterfolder: string path to raster maps output folder
    :param folder: string path to series txt output folder
    :param filename: string output series file name
    :param suff: string file name suffix
    :param tui: boolean of tui display
    :return: string filepath to output series file
    """
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
                             fshruparam=fshruparam, folder=rasterfolder, filename=lcl_filename, viewlabel=lcl_date)
        # print(lcl_expf)
        new_files.append(shru_file)
    #
    # export data
    exp_df = pd.DataFrame({'Date': dates, 'File': new_files})
    exp_file = folder + '/' + filename + '.txt'
    exp_df.to_csv(exp_file, sep=';', index=False)
    return exp_file


def get_shru_param(flulcparam, fsoilsparam, folder='C:/bin', filename='shru_param'):
    """
    Compute the SHRU parameters .txt dataframe file

    :param flulcparam: string file path to LULC parameters dataframe txt file
    :param fsoilsparam: string file path to Soils parameters dataframe txt file
    :param folder: string path to output folder
    :param filename: string file name
    :return: string file path to output file
    """
    # extract data
    lulc_df = pd.read_csv(flulcparam, sep=';', engine='python')
    lulc_df = dataframe_prepro(lulc_df, strfields='LULCName,LULCAlias,CanopySeason,ConvertTo,ColorLULC')
    #print(lulc_df.to_string())
    soils_df = pd.read_csv(fsoilsparam, sep=';', engine='python')
    soils_df = dataframe_prepro(soils_df, strfields='SoilName,SoilAlias,ColorSoil')
    #print(soils_df.to_string())
    lulc_ids = lulc_df['IdLULC'].values
    soils_ids = soils_df['IdSoil'].values
    #
    # process
    shru_ids = list()
    shru_nm = list()
    shru_al = list()
    shru_lulc_ids = list()
    shru_soils_ids = list()
    for i in range(len(lulc_ids)):
        for j in range(len(soils_ids)):
            lcl_shru_id = lulc_ids[i] * 100 + soils_ids[j]
            lcl_shru_nm = lulc_df['LULCName'].values[i] + '_' + soils_df['SoilName'].values[j]
            lcl_shru_al = lulc_df['LULCAlias'].values[i] + '_' + soils_df['SoilAlias'].values[j]
            shru_ids.append(lcl_shru_id)
            shru_nm.append(lcl_shru_nm)
            shru_al.append(lcl_shru_al)
            shru_lulc_ids.append(lulc_ids[i])
            shru_soils_ids.append(soils_ids[j])
    shru_df = pd.DataFrame({'IdSHRU':shru_ids, 'SHRUName': shru_nm, 'SHRUAlias': shru_al,
                            'IdLULC':shru_lulc_ids, 'IdSoil':shru_soils_ids})
    # join parameters:
    shru_df = shru_df.join(lulc_df.set_index('IdLULC'), on='IdLULC')
    shru_df = shru_df.join(soils_df.set_index('IdSoil'), on='IdSoil')
    # cross root zone:
    shru_df['f_EfRootZone'] = shru_df['Porosity'].values * shru_df['f_RootDepth'].values
    # cross infiltration capacity:
    shru_df['f_Inf'] = shru_df['f_Ksat'].values * shru_df['f_Pervious'].values
    #print(shru_df.to_string())
    #
    # export
    exp_file = folder + '/' + filename + '.txt'
    shru_df.to_csv(exp_file, sep=';', index=False)
    return exp_file


def canopy_series(fseries, fshruparam, folder='C:/bin', filename='canopy_season'):
    """
    Routine to compute SHRU canopy seasonal factor pattern series in terms of peak factor f_Canopy
    and stores it in a txt file

    :param fseries: string file path to txt file of time series. Must have a 'Date' field
    :param fshruparam: string file path to txt file of shry parameters file provided in the get_shru_param() tool
    :param folder: string path to output folder
    :param filename: string file name without extension
    :return: string file path to txt file output
    """
    from resample import interpolate_gaps
    from backend import get_stringfields
    # import series
    season_df = pd.read_csv(fseries, sep=';')
    season_df = dataframe_prepro(season_df, strf=False, date=True)
    season_df = season_df[['Date']]
    # import parameters
    shru_df = pd.read_csv(fshruparam, sep=';')
    aux_str = get_stringfields(fshruparam.split('/')[-1])
    shru_df = dataframe_prepro(shru_df, aux_str)
    # insert month field in
    season_df['Month'] = season_df['Date'].dt.month_name(locale='English').str.slice(stop=3)
    #
    # loop across all shru classes
    for i in range(len(shru_df)):
        # access shru name
        shru_nm = shru_df['SHRUAlias'].values[i]
        #print(shru_nm)
        lcl_field = 'f_Canopy_{}'.format(shru_nm)
        # access the season type
        season = shru_df['CanopySeason'].values[i]
        if season == 'none':
            season_df[lcl_field] = np.ones(shape=np.shape(season_df['Date'].values))
        else:
            #
            # get background parameter
            cpy_bkg = shru_df['f_CanopyBackg'].values[i]
            season = season.replace(' ', '').lower().split('&')
            #
            # create and empty numeric field
            season_df[lcl_field] = np.zeros(shape=np.shape(season_df['Date'].values))
            #
            # insert dormancy period
            flag = False
            for t in range(len(season_df)):
                if season_df['Month'].values[t].lower() == season[3]:
                    flag = True
                if flag:
                    season_df[lcl_field].values[t] = cpy_bkg
                if season_df['Month'].values[t].lower() == season[0]:
                    flag = False
            flag = False
            for t in range(len(season_df) - 1, -1, -1):
                if season_df['Month'].values[t].lower() == season[0]:
                    flag = True
                if flag:
                    if season_df[lcl_field].values[t] != cpy_bkg:
                        season_df[lcl_field].values[t] = cpy_bkg
                if season_df['Month'].values[t].lower() == season[3]:
                    flag = False
            #
            # insert peak period
            flag = False
            for t in range(len(season_df)):
                if season_df['Month'].values[t].lower() == season[1]:
                    flag = True
                if flag:
                    season_df[lcl_field].values[t] = 1.0
                if season_df['Month'].values[t].lower() == season[2]:
                    flag = False
            flag = False
            for t in range(len(season_df) - 1, -1, -1):
                if season_df['Month'].values[t].lower() == season[2]:
                    flag = True
                if flag:
                    if season_df[lcl_field].values[t] != 1.0:
                        season_df[lcl_field].values[t] = 1.0
                if season_df['Month'].values[t].lower() == season[1]:
                    flag = False
            #
            # insert gaps:
            season_df[lcl_field] = season_df[lcl_field].replace(0, np.nan)
            #
            # fill gaps in periods:
            new_df = interpolate_gaps(season_df, var_field=lcl_field, size=365, type='linear')
            season_df[lcl_field] = new_df['Interpolation']
    #
    season_df.fillna(1.0, inplace=True)
    # export
    outfile = folder + '/' + filename + '.txt'
    season_df.to_csv(outfile, sep=';', index=False)
    #print(season_df.tail(60).to_string())
    return outfile


def sdiag(fseries, filename='sim_diagnostics', folder='C:/bin', tui=False):
    """
    Run Hydrology Simulation Diagnostics

    :param fseries: string file path to simulation series txt file
    :param filename: string output file name
    :param folder: string path to output folder
    :param tui: boolean to tui display
    :return: string file path to output file
    """

    from hydrology import extract_sim_diagnostics
    import time, datetime
    #
    # report setup
    t0 = time.time()
    report_lst = list()
    report_lst.append('Execution timestamp: {}\n'.format(datetime.datetime.now()))
    report_lst.append('Process: SIMULATION DIAGNOSTICS | SDIAG\n')
    input_files_df = pd.DataFrame({'Input files': (fseries,)})
    report_lst.append(input_files_df.to_string(index=False))
    #
    if tui:
        from tui import status
        status('performing simulation diagnostics')
    # import series
    series_df = pd.read_csv(fseries, sep=';', parse_dates=['Date'])
    # compute diag
    diag_df = extract_sim_diagnostics(series_df)
    #
    # export
    exp_file = '{}/{}.txt'.format(folder, filename)
    diag_df.to_csv(exp_file, sep=';', index=False)
    #
    # Report
    output_df = pd.DataFrame({'Output files': (exp_file,)})
    report_lst.append('\n\n')
    report_lst.append(output_df.to_string(index=False))
    report_lst.append('\n\n')
    #
    report_lst.append('\n\nDiagnostics Parameter Results:\n')
    report_lst.append('\nInput parameters:\n')
    report_lst.append(diag_df[['Parameter', 'Prec', 'Temp', 'IRA', 'IRI']].to_string(index=False))
    report_lst.append('\n\nStock parameters:\n')
    report_lst.append(diag_df[['Parameter', 'D', 'Cpy', 'Sfs', 'Unz']].to_string(index=False))
    report_lst.append('\n\nOutput Flow parameters:\n')
    report_lst.append(diag_df[['Parameter', 'Q', 'Qb', 'Qs']].to_string(index=False))
    report_lst.append('\n\nRunoff Flow parameters:\n')
    report_lst.append(diag_df[['Parameter', 'TF', 'R', 'RSE', 'RIE']].to_string(index=False))
    report_lst.append('\n\nET Flow parameters:\n')
    report_lst.append(diag_df[['Parameter', 'PET', 'ET', 'Evc', 'Evs', 'Tpun', 'Tpgw']].to_string(index=False))
    report_lst.append('\n\nRecharge Flow parameters:\n')
    report_lst.append(diag_df[['Parameter', 'Qv', 'Inf']].to_string(index=False))
    report_lst.append('\n\nNon-water parameters:\n')
    report_lst.append(diag_df[['Parameter', 'RC', 'VSA']].to_string(index=False))
    #
    tf = time.time()
    report_lst.insert(2, 'Execution enlapsed time: {:.3f} seconds\n'.format(tf - t0))
    export_report(report_lst, filename='REPORT__diagnostics', folder=folder, tui=tui)
    return exp_file


def qualmap_analyst(fmap, fparams, faoi='full', type='lulc', folder='C:/bin', wkpl=False, label=''):
    """
    Analyst of qualitative maps.
    :param fmap: string file path to raster map .asc file
    :param fparams: string file path to .txt map parameter file
    :param faoi: string file path to aoi raster map .asc file OR default by pass code='full' for full extension
    :param type: string code for type of map. Allowed types: 'lulc' and 'soils' . Default: 'lulc'
    :param folder: string file path to output directory
    :param wkpl: boolean to set the folder param as an workplace
    :param label: string label for output file naming
    :return:
    """
    from geo import areas
    from visuals import plot_lulc_view
    from backend import create_rundir
    #
    # Run Folder setup
    if wkpl:  # if the passed folder is a workplace, create a sub folder within it
        if label != '':
            label = label + '_'
        folder = create_rundir(label=label + 'QMAP', wkplc=folder)
    #
    # type setup
    if type == 'lulc':
        str_fields = 'LULCName,ColorLULC,ConvertTo'
        idfield = 'IdLULC'
        namefield = 'LULCName'
        colorfield = 'ColorLULC'
    elif type == 'soils':
        str_fields = 'SoilName,ColorSoil'
        idfield = 'IdSoil'
        namefield = 'SoilName'
        colorfield = 'ColorSoil'
    # imports
    meta, qmap = inp.asc_raster(fmap, dtype='float32')
    param_df = pd.read_csv(fparams, sep=';')
    param_df = inp.dataframe_prepro(param_df, strfields=str_fields)
    if faoi != 'full':
        meta, aoi = inp.asc_raster(faoi, dtype='float32')
    else:
        aoi = 1.0 + (0.0 * qmap)
    #
    # Compute areas
    areas_m2 = areas(qmap * aoi, meta['cellsize'], values=param_df[idfield])
    areas_ha = areas_m2 / (100 * 100)
    areas_km2 = areas_m2 / (1000 * 1000)
    areas_df = param_df[[idfield, namefield, colorfield]].copy()
    areas_df['Area_m2'] = areas_m2
    areas_df['Area_ha'] = areas_ha
    areas_df['Area_km2'] = areas_km2
    areas_df['Area_%'] = 100 * areas_m2 / np.sum(areas_m2)
    print(areas_df.to_string())
    #
    # Export areas
    exp_file1 = '{}/areas_{}.txt'.format(folder, type)
    areas_df.to_csv(exp_file1, sep=';', index=False)
    #
    # Export pannel
    if type == 'lulc':
        plot_lulc_view(qmap, param_df, areas_df, aoi, meta, folder=folder)


def osa_series(fseries,
               fld_obs='Qobs',
               fld_sim='Q',
               fld_date='Date',
               folder='C:/bin',
               tui=False,
               var=True,
               log=True):
    """
    Observed vs. Simulated Analyst routine for series

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
    report_lst.append('Process: OBS-SIM ANALYST | OSA\n')
    input_files_df = pd.DataFrame({'Input files': (fseries,)})
    report_lst.append(input_files_df.to_string(index=False))
    #
    if tui:
        from tui import status
        status('performing obs vs. sim analysis')
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
    # **** Series Analysis ****
    #
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
    # **** Frequency analysis ****
    #
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
    # final exports
    tf = time.time()
    output_df = pd.DataFrame({'Output files': (exp_file1, exp_file2, exp_file3, exp_file4)})
    report_lst.append(output_df.to_string(index=False))
    report_lst.append('\n\n')
    report_lst.insert(2, 'Execution enlapsed time: {:.3f} seconds\n'.format(tf - t0))
    export_report(report_lst, filename='REPORT__analyst', folder=folder, tui=tui)
    return (exp_file1, exp_file2, exp_file3, exp_file4)


def osa_zmaps(fobs_series, fsim_series, fhistograms, fseries, fshru, ftwi,
              var='ETPat',
              folder='C:/bin',
              tui=True,
              nodata=-1,
              raster=False,
              wkpl=False,
              label=''):
    """

    Observed-Simulation Analyst of Zmaps

    :param fobs_series: string filepath to observed timeseries
    :param fsim_series: string filepath to simulated timeseries
    :param fhistograms: string filepath to histograms dataframe
    :param fseries: string filepath to input timeseries
    :param fshru: string filepath to shru raster map
    :param ftwi: string filepath to twi raster map
    :param var: string code of variable
    :param folder: string path to output folder
    :param tui: boolean to TUI display
    :param nodata: float no data value
    :param raster: boolean to compute raster views
    :param wkpl: boolean to consider folder as workplace
    :param label: string label
    :return: none
    """
    import time, datetime
    import analyst
    from hydrology import map_back
    from inp import histograms, asc_raster
    from visuals import plot_zmap_analyst, pannel_global
    from backend import create_rundir
    #
    # report setup
    t0 = time.time()
    report_lst = list()
    report_lst.append('Execution timestamp: {}\n'.format(datetime.datetime.now()))
    report_lst.append('Process: OBS-SIM ZMAP ANALYST | OSA_ZMAPS\n')
    input_files_df = pd.DataFrame({'Input files': (fobs_series, fsim_series, fhistograms)})
    report_lst.append(input_files_df.to_string(index=False))
    #
    if tui:
        from tui import status
        status('performing obs vs. sim analysis for zmaps')
    #
    # Run Folder setup
    if tui:
        from tui import status
        status('setting folders')
    if wkpl:  # if the passed folder is a workplace, create a sub folder
        if label != '':
            label = label + '_'
        folder = create_rundir(label=label + 'OSA_ZMAP', wkplc=folder)
    # extract input data
    #
    # extract count matrix (full map extension)
    if tui:
        status('loading histograms')
    count, twibins, shrubins = histograms(fhistograms=fhistograms)
    #
    if tui:
        status('loading ZMAPS')
    # Import Observed Zmaps dataframe series
    zmaps_obs_df = pd.read_csv(fobs_series, sep=';')
    zmaps_obs_df = dataframe_prepro(zmaps_obs_df, strfields='File', date=True)
    #
    # Import Simulated Zmaps dataframe series
    zmaps_sim_df = pd.read_csv(fsim_series, sep=';')
    zmaps_sim_df = dataframe_prepro(zmaps_sim_df, strfields='File', date=True)
    #
    # Match dates
    obssim_df = pd.merge(zmaps_obs_df, zmaps_sim_df, how='inner', on='Date', suffixes=('_obs', '_sim'))
    #
    # now load zmaps
    obs_zmaps = list()
    sim_zmaps = list()
    for i in range(len(obssim_df)):
        zmap_file = obssim_df['File_obs'].values[i]
        zmap, ybins, xbins = inp.zmap(zmap_file)
        #print(np.shape(zmap))
        obs_zmaps.append(zmap)
        zmap_file = obssim_df['File_sim'].values[i]
        zmap, ybins, xbins = inp.zmap(zmap_file)
        #print(np.shape(zmap))
        sim_zmaps.append(zmap)
    obs_zmaps = np.array(obs_zmaps)
    sim_zmaps = np.array(sim_zmaps)
    # get max values
    vmax_zmaps = np.max([obs_zmaps, sim_zmaps])
    #
    #
    # load series
    # extract Dataframe
    if tui:
        status('loading series')
    series_df = pd.read_csv(fseries, sep=';', engine='python')
    series_df = dataframe_prepro(series_df, strf=False, date=True)
    #
    tf = time.time()
    if tui:
        status('loading enlapsed time: {:.3f} seconds'.format(tf - t0))
    #
    #
    # Analysis
    if tui:
        status('running series analyst')
    a_dct = analyst.zmaps_series(obs=obs_zmaps, sim=sim_zmaps, count=1.0 * count, nodata=nodata, full_return=True)
    #
    # retrieve to dataframe
    _aux_tpl = ('N', 'MSE', 'RMSE', 'W-MSE', 'W-RMSE', 'R', 'NSE', 'KGE', 'Mean-Obs', 'Mean-Sim', 'Mean-Error')
    for m in _aux_tpl:
        obssim_df[m] = a_dct['Metrics'][m]
    #
    # Export files
    #
    #
    # Analyst series
    exp_file1 = folder + '/' + 'analyst_zmaps_series.txt'
    obssim_df.to_csv(exp_file1, sep=';', index=False)
    #
    # Updates sim series
    series_df = pd.merge(series_df, obssim_df[['Date', 'Mean-Obs']], how='left', on='Date')
    series_df.rename(columns={'Mean-Obs':'ETobs'}, inplace=True)
    # export sim series
    exp_file2 = folder + '/' + 'analyst_sim_series.txt'
    series_df.to_csv(exp_file2, sep=';', index=False)
    #
    # Visuals export
    if tui:
        status('exporting global pannel')
    pannel_global(series_df,
                  qobs=True,
                  etobs=True,
                  folder=folder)
    #
    # Zmaps
    if tui:
        status('exporting zmaps pannels')
    for i in range(len(obssim_df)):
        if tui:
            status('::: zmaps pannel | {}'.format(obssim_df['Date'].astype(str).values[i]))
        lcl_ttl = 'ET24h (mm) | {}'.format(obssim_df['Date'].astype(str).values[i])
        # built metrics dict
        lcl_dct = dict()
        for m in _aux_tpl:
            lcl_dct[m] = a_dct['Metrics'][m][i]
        plot_zmap_analyst(obs=obs_zmaps[i],
                          sim=sim_zmaps[i],
                          count=count,
                          error=a_dct['Maps']['Error'][i],
                          w_error=a_dct['Maps']['WError'][i],
                          obs_sig=a_dct['Signals']['Obs'][i],
                          sim_sig=a_dct['Signals']['Sim'][i],
                          error_sig=a_dct['Signals']['Error'][i],
                          w_error_sig=a_dct['Signals']['WError'][i],
                          ranges=(0, vmax_zmaps),
                          metricranges=(-vmax_zmaps, vmax_zmaps),
                          metrics_dct=lcl_dct,
                          show=False,
                          folder=folder,
                          filename='analyst_zmaps_{}'.format(obssim_df['Date'].astype(str).values[i]),
                          ttl=lcl_ttl)
    #
    # Rmaps
    if raster:
        from visuals import plot_raster_analyst
        meta, shru = asc_raster(fshru, dtype='float32')
        meta, twi = asc_raster(ftwi, dtype='float32')
        if tui:
            status('exporting raster pannels')
        for i in range(len(obssim_df)):
            if tui:
                status('::: raster pannel | {}'.format(obssim_df['Date'].astype(str).values[i]))
            lcl_ttl = 'ET24h (mm) | {}'.format(obssim_df['Date'].astype(str).values[i])
            # built metrics dict
            lcl_dct = dict()
            for m in _aux_tpl:
                lcl_dct[m] = a_dct['Metrics'][m][i]
            # load raster
            obs_rmap = map_back(obs_zmaps[i], twi, shru, twibins, shrubins)
            sim_rmap = map_back(sim_zmaps[i], twi, shru, twibins, shrubins)
            plot_raster_analyst(obs=obs_rmap,
                                sim=sim_rmap,
                                ranges=(0, vmax_zmaps),
                                metricranges=(-vmax_zmaps, vmax_zmaps),
                                metrics_dct=lcl_dct,
                                metrics_txt=True,
                                show=False,
                                folder=folder,
                                filename = 'analyst_raster{}'.format(obssim_df['Date'].astype(str).values[i]),
                                ttl = lcl_ttl)
    #
    # final exports
    tf = time.time()
    output_df = pd.DataFrame({'Output files': (exp_file1, 'ok')})
    report_lst.append(output_df.to_string(index=False))
    report_lst.append('\n\n')
    report_lst.insert(2, 'Execution enlapsed time: {:.3f} seconds\n'.format(tf - t0))
    export_report(report_lst, filename='REPORT__analyst_zmaps', folder=folder, tui=tui)


def bat_slh(fmodels, fseries, fhydroparam, fshruparam, fhistograms, fbasinhists, fbasin, ftwi, fshru, fcanopy,
            model_id='Id',
            mapback=False,
            mapraster=False,
            mapvar='all',
            mapdates='all',
            integrate=False,
            slicedates='all',
            qobs=False,
            folder='C:/bin',
            wkpl=False,
            label='',
            tui=False,
            aoi=False,
            pannel=False,
            ensemble=False,
            averaging=False):
    """

    Batch of SLH

    :param fmodels: string filepath to model dataframe csv
    :param fseries: string filepath to simulation series
    :param fhydroparam: string filepath to parameters csv (txt file)
    :param fshruparam: string filepath to shru parameters csv (txt file)
    :param fhistograms: string filepath to histograms csv (txt file)
    :param fbasinhists: string filepath to basin histograms csv (txt file)
    :param fbasin: string filepath to basin raster map (asc file)
    :param ftwi: string filepath to twi raster map (asc file)
    :param fshru: string filepath to shru raster map (asc file)
    :param fcanopy: string filepath to canopy seasonal factor series (txt file)
    :param model_id: string field name of model ID
    :param mapback: boolean to map back variables
    :param mapraster: boolean to map back raster maps
    :param mapvar: string of variables to map. Pass concatenated by '-'. Ex: 'ET-TF-Inf'.
    Options: 'Prec-Temp-IRA-IRI-PET-D-Cpy-TF-Sfs-R-RSE-RIE-RC-Inf-Unz-Qv-Evc-Evs-Tpun-Tpgw-ET-VSA'
    :param mapdates: string of dates to map. Pass concatenated by ' & '. Ex: '2011-21-01 & 21-22-01'
    :param integrate: boolean to include variable integration over the simulation period
    :param slicedates: string code to slice series. 'all' or 'YYYY-MM-DD to YYYY-MM-DD'
    :param qobs: boolean to inclue Qobs in visuals
    :param folder: string to folderpath
    :param wkpl: boolean to set folder as workplace
    :param label: string label
    :param tui: boolean to TUI display
    :param aoi: boolean to consider AOI basin
    :param pannel: boolean to export simulation pannel
    :param ensemble: boolean to export ensemble of models
    :param averaging: boolean to export average maps (if integration is allowed)
    :return: none
    """
    import os
    from backend import get_mapid
    from visuals import plot_ensemble
    # Run Folder setup
    if wkpl:  # if the passed folder is a workplace, create a sub folder within it
        from backend import create_rundir
        if label != '':
            label = label + '_'
        folder = create_rundir(label=label + 'batSLH', wkplc=folder)
    #
    # read models dataframe
    models_df = pd.read_csv(fmodels, sep=';')
    #
    # read sample fhydroparam
    hydroparam_dct, hydroparam_df = inp.hydroparams(fhydroparam=fhydroparam)
    #
    # loop across models to simulation
    series_dct = dict() # dict for storing series
    lcl_folders = list()
    params = ('m', 'lamb', 'qo', 'cpmax', 'sfmax', 'erz', 'ksat', 'c', 'k', 'n')
    for i in range(len(models_df)):
        print('Model ID: {}'.format(models_df[model_id].values[i]))
        # create local folder
        lcl_folder = folder + '/model_{}'.format(models_df[model_id].values[i])
        lcl_folders.append(lcl_folder)
        os.mkdir(lcl_folder)
        #
        # create local hydroparam file
        lcl_hydroparam_df = hydroparam_df.copy()
        lcl_hydroparam_df.set_index('Parameter', inplace=True)
        #
        # change parameter values
        for p in params:
            lcl_hydroparam_df.at[p, 'Set'] = models_df[p].values[i]
        lcl_hydroparam_df.reset_index(inplace=True)
        print(lcl_hydroparam_df[['Parameter', 'Set']])
        lcl_fhydroparam = lcl_folder + '/hydro_param.txt'
        lcl_hydroparam_df.to_csv(lcl_fhydroparam, sep=';', index=False)
        #
        # run local model
        lcl_dct = slh(fseries=fseries,
                      fhydroparam=lcl_fhydroparam,
                      fshruparam=fshruparam,
                      fhistograms=fhistograms,
                      fbasinhists=fbasinhists,
                      fcanopy=fcanopy,
                      fbasin=fbasin,
                      fshru=fshru,
                      ftwi=ftwi,
                      mapback=mapback,
                      mapraster=mapraster,
                      mapvar=mapvar,
                      integrate=integrate,
                      slicedates=slicedates,
                      qobs=qobs,
                      folder=lcl_folder,
                      wkpl=False,
                      tui=tui,
                      aoi=aoi,
                      pannel=pannel)
        series_dct[models_df[model_id].values[i]] = lcl_dct['Series']
    #
    # compute ensemble
    if ensemble:
        ensemble_vars = backend.get_all_simvars().split('-')
        series_df = pd.read_csv(fseries, sep=';', parse_dates=['Date'])
        if slicedates != 'all':
            slicedates = slicedates.split(' to ')
            q_str = 'Date >= "{}" and Date <= "{}"'.format(slicedates[0].strip(), slicedates[1].strip())
            series_df = series_df.query(q_str)
        t_ensem = len(series_df)
        n_ensem = len(models_df)
        for v in ensemble_vars:
            # set up
            sim_grid = np.zeros(shape=(n_ensem, t_ensem))
            # models loop:
            for i in range(n_ensem):
                # read lcl series
                lcl_file = series_dct[models_df[model_id].values[i]]
                lcl_series_df = pd.read_csv(lcl_file, sep=';', parse_dates=['Date'])
                sim_grid[i] = lcl_series_df[v].values
            # append series
            series_df['{}_05'.format(v)] = np.quantile(sim_grid, 0.05, axis=0)
            series_df['{}_50'.format(v)] = np.quantile(sim_grid, 0.5, axis=0)
            series_df['{}_95'.format(v)] = np.quantile(sim_grid, 0.95, axis=0)
        #
        # export ensemble series
        fseries_ensemble = '{}/series_ensemble.txt'.format(folder)
        series_df.to_csv(fseries_ensemble, sep=';', index=False)
        #
        # export ensemble plots
        for v in ensemble_vars:
            lcl_filename = 'ensemble'
            lcl_ttl = '{} ensemble'.format(v)
            lcl_qobs = False
            if v == 'Q':
                lcl_qobs = qobs
            plot_ensemble(dataframe=series_df,
                          q_05_field='{}_05'.format(v),
                          q_50_field='{}_50'.format(v),
                          q_95_field='{}_95'.format(v),
                          ttl=lcl_ttl,
                          ttl1=v,
                          filename=lcl_filename,
                          folder=folder,
                          suff=v,
                          show=False,
                          qobs=lcl_qobs)
    #
    # averaging across integration
    if integrate and averaging:
        # compute averages
        if mapvar == 'all':
            mapvars = backend.get_all_lclvars()
        else:
            mapvars = mapvar.split('-')
        # variable loop
        for v in mapvars:
            # models loop
            for i in range(len(lcl_folders)):
                lcl_file = '{}/integration/raster_integral_{}.asc'.format(lcl_folders[i], v)
                # import map
                meta, lcl_map = inp.asc_raster(lcl_file, dtype='float32')
                # accumulate
                if i == 0:
                    acc_map = lcl_map
                else:
                    acc_map = acc_map + lcl_map
            # get average
            acc_map = acc_map / len(lcl_folders)
            filename = 'avg_{}'.format(v)
            print(filename)
            favg_map = out.asc_raster(acc_map, meta, folder=folder, filename=filename)
            mapid = get_mapid(v)
            visuals.plot_map_view(acc_map, meta,
                                  ranges=[0,  np.max(acc_map)],
                                  mapid=mapid, mapttl='Average {}'.format(v),
                                  filename=filename,
                                  folder=folder)


def slh(fseries, fhydroparam, fshruparam, fhistograms, fbasinhists, fbasin, ftwi, fshru, fcanopy,
        mapback=False,
        mapraster=False,
        mapvar='all',
        mapdates='all',
        integrate=False,
        slicedates='all',
        qobs=False,
        folder='C:/bin',
        wkpl=False,
        label='',
        tui=False,
        aoi=False,
        pannel=True):
    """

    SLH - Stable LULC Hydrology routine

    :param fseries: string filepath to simulation series
    :param fhydroparam: string filepath to parameters csv (txt file)
    :param fshruparam: string filepath to shru parameters csv (txt file)
    :param fhistograms: string filepath to histograms csv (txt file)
    :param fbasinhists: string filepath to basin histograms csv (txt file)
    :param fbasin: string filepath to basin raster map (asc file)
    :param ftwi: string filepath to twi raster map (asc file)
    :param fshru: string filepath to shru raster map (asc file)
    :param fcanopy: string filepath to canopy seasonal factor series (txt file)
    :param mapback: boolean to map back variables
    :param mapraster: boolean to map back raster maps
    :param mapvar: string of variables to map. Pass concatenated by '-'. Ex: 'ET-TF-Inf'.
    Options: 'Prec-Temp-IRA-IRI-PET-D-Cpy-TF-Sfs-R-RSE-RIE-RC-Inf-Unz-Qv-Evc-Evs-Tpun-Tpgw-ET-VSA'
    :param mapdates: string of dates to map. Pass concatenated by ' & '. Ex: '2011-21-01 & 21-22-01'
    :param integrate: boolean to include variable integration over the simulation period
    :param slicedates: string code to slice series. 'all' or 'YYYY-MM-DD to YYYY-MM-DD'
    :param qobs: boolean to inclue Qobs in visuals
    :param folder: string to folderpath
    :param wkpl: boolean to set folder as workplace
    :param label: string label
    :param tui: boolean to TUI display
    :param aoi: boolean to consider AOI basin
    :param pannel: boolean to export simulation pannel
    :return: dictionary with keys to output filepaths
                out_dct =  {'Series':exp_file1,
                            'Histograms':exp_file2,
                            'Parameters':exp_file3,
                            'Pannel':exp_file4,
                            'Folder':folder}
    """
    import time, datetime
    from shutil import copyfile
    from inp import zmap
    from hydrology import simulation, map_back
    from visuals import pannel_global
    from backend import create_rundir, get_stringfields, get_mapid
    #
    # Run Folder setup
    if wkpl:  # if the passed folder is a workplace, create a sub folder within it
        if label != '':
            label = label + '_'
        folder = create_rundir(label=label + 'SLH', wkplc=folder)
    #
    # time and report setup
    t0 = time.time()
    report_lst = list()
    report_lst.append('Execution timestamp: {}\n'.format(datetime.datetime.now()))
    report_lst.append('Process: STABLE LULC HYDROLOGY | SLH\n')
    input_files_df = pd.DataFrame({'Input files': (fseries, fhydroparam, fshruparam, fhistograms, fbasinhists,
                                                   fbasin, ftwi, fshru)})
    report_lst.append(input_files_df.to_string(index=False))
    #
    #
    # ****** IMPORT ******
    #
    #
    init = time.time()
    if tui:
        from tui import status
        print('\n\nLocal run folder: {}\n\n'.format(folder))
        status('loading time series')

    series_df =  pd.read_csv(fseries, sep=';')
    series_df = dataframe_prepro(series_df, strf=False, date=True, datefield='Date')
    if slicedates != 'all':
        slicedates = slicedates.split(' to ')
        q_str = 'Date >= "{}" and Date <= "{}"'.format(slicedates[0].strip(), slicedates[1].strip())
        series_df = series_df.query(q_str)
    #
    if tui:
        status('loading hydrology parameters')
    hydroparam_dct, hydroparam_df = inp.hydroparams(fhydroparam=fhydroparam)
    #
    # extract set values
    m = hydroparam_dct['m']['Set']
    lamb = hydroparam_dct['lamb']['Set']
    qo = hydroparam_dct['qo']['Set']
    cpmax = hydroparam_dct['cpmax']['Set']
    sfmax = hydroparam_dct['sfmax']['Set']
    erz = hydroparam_dct['erz']['Set']
    ksat = hydroparam_dct['ksat']['Set']
    c = hydroparam_dct['c']['Set']
    lat = hydroparam_dct['lat']['Set']
    k = hydroparam_dct['k']['Set']
    n = hydroparam_dct['n']['Set']
    # AOI factor correction
    if aoi:
        m = m * hydroparam_dct['m']['AOI_f']
        lamb = lamb * hydroparam_dct['lamb']['AOI_f']
        qo = qo * hydroparam_dct['qo']['AOI_f']
        cpmax = cpmax * hydroparam_dct['cpmax']['AOI_f']
        sfmax = sfmax * hydroparam_dct['sfmax']['AOI_f']
        erz = erz * hydroparam_dct['erz']['AOI_f']
        ksat = ksat * hydroparam_dct['ksat']['AOI_f']
        c = c * hydroparam_dct['c']['AOI_f']
        lat = lat * hydroparam_dct['lat']['AOI_f']
        k = k * hydroparam_dct['k']['AOI_f']
        n = n * hydroparam_dct['n']['AOI_f']
    #
    # Shru parameters
    if tui:
        status('loading SHRU parameters')
    shru_df = pd.read_csv(fshruparam, sep=';')
    shru_df = dataframe_prepro(shru_df, get_stringfields(fshruparam.split('/')[-1]))
    #
    # canopy series pattern
    if tui:
        status('loading Canopy series pattern')
    canopy_df = pd.read_csv(fcanopy, sep=';')
    #
    # extract count matrix (full map extension)
    if tui:
        status('loading histograms of map extension')
    count, twibins, shrubins = zmap(file=fhistograms)
    #
    # extract count matrix (basin)
    if tui:
        status('loading histograms of basin')
    basincount, twibins2, shrubins2 = zmap(file=fbasinhists)
    #
    # get basin boundary conditions
    if tui:
        status('loading boundary basin conditions')
    meta = inp.asc_raster_meta(fbasin)  # get just the metadata
    area = np.sum(basincount) * meta['cellsize'] * meta['cellsize']
    qt0 = 0.01  # fixed
    if qobs:
        qt0 = series_df['Q'].values[0]
    #print(qt0)
    #
    end = time.time()
    report_lst.append('\n\nLoading enlapsed time: {:.3f} seconds\n'.format(end - init))
    if tui:
        status('loading enlapsed time: {:.3f} seconds'.format(end - init), process=False)
    #
    #
    # ****** SIMULATION ******
    #
    #
    init = time.time()
    if tui:
        status('running simulation')
    sim_dct = simulation(series=series_df,
                         shruparam=shru_df,
                         canopy=canopy_df,
                         twibins=twibins,
                         countmatrix=count,
                         lamb=lamb,
                         qt0=qt0,
                         m=m,
                         qo=qo,
                         cpmax=cpmax,
                         sfmax=sfmax,
                         erz=erz,
                         ksat=ksat,
                         c=c,
                         lat=lat,
                         k=k,
                         n=n,
                         area=area,
                         basinshadow=basincount,
                         tui=False,
                         qobs=qobs,
                         mapback=mapback,
                         mapvar=mapvar,
                         mapdates=mapdates)
    sim_df = sim_dct['Series']
    if mapback:
        mapped = sim_dct['Maps']
    end = time.time()
    report_lst.append('Simulation enlapsed time: {:.3f} seconds\n'.format(end - init))
    if tui:
        status('simulation enlapsed time: {:.3f} seconds'.format(end - init), process=False)
    #
    #
    # ****** DEFAULT EXPORT ******
    #
    #
    init = time.time()
    if tui:
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
    #
    if tui:
        status('running simulation diagnostics')
    sdiag_file1 = sdiag(fseries=exp_file1, folder=folder, tui=False)
    #
    # export visual pannel
    if pannel:
        if tui:
            status('exporting visual results')
        exp_file4 = pannel_global(sim_df, grid=True, show=False, qobs=qobs, folder=folder)
    #
    #
    # ****** MAP EXPORT ******
    #
    #
    if mapback:
        if tui:
            status('exporting variable maps')
        from os import mkdir
        from backend import get_all_lclvars
        from out import zmap
        #
        if mapraster or integrate:
            from hydrology import map_back
            from visuals import plot_map_view
            #
            # heavy imports
            if tui:
                status('importing twi raster')
            meta, twi = inp.asc_raster(ftwi, dtype='float32')
            if tui:
                status('importing shru raster')
            meta, shru = inp.asc_raster(fshru, dtype='float32')
        #
        if integrate:
            # make integration directory
            int_folder = folder + '/integration'
            mkdir(int_folder)
        #
        if mapvar == 'all':
            mapvar = get_all_lclvars()
        mapvar_lst = mapvar.split('-')  # load string variables alias to list
        #
        # get map stamps
        stamp = sim_dct['MappedDates']
        #
        # ****** ZMAP EXPORT ******
        #
        mapfiles_lst = list()
        intmaps_files = list()
        zmaps_dct = dict()
        for var in mapvar_lst:  # loop across asked variables
            if tui:
                status('exporting zmaps | {}'.format(var))
            #
            # make var directory
            lcl_folder = folder + '/sim_' + var
            mkdir(lcl_folder)
            #
            if integrate:
                # initiate integration
                integration = mapped[var][0] * 0.0
            # loop across all mapped dates
            lcl_files = list()
            for t in range(len(stamp)):
                # get local file name
                lcl_filename ='zmap_{}_{}'.format(var, stamp[t])
                # export to CSV
                lcl_file = zmap(zmap=mapped[var][t], twibins=twibins, shrubins=shrubins,
                                folder=lcl_folder, filename=lcl_filename)
                # trace file path
                lcl_files.append(lcl_file)
                # conpute integration
                if integrate:
                    integration = integration + mapped[var][t]
            #
            # ***** INTEGRATE *****
            #
            if integrate:
                # stock variables
                lcl_label = 'Accumulation'
                if var in set(['D', 'Unz', 'Cpy', 'Sfs', 'RC', 'VSA', 'Temp']):
                    integration = integration / len(mapped[var])  # take the average
                    if var == 'VSA':
                        integration = integration * 100
                    lcl_label = 'Average'
                # export integral zmap
                lcl_filename = 'zmap_integral_{}'.format(var)
                lcl_file = zmap(zmap=integration, twibins=twibins, shrubins=shrubins,
                                folder=int_folder, filename=lcl_filename)
                intmaps_files.append(lcl_files)
                #
                # recover raster
                mp = map_back(integration, a1=twi, a2=shru, bins1=twibins, bins2=shrubins)
                # export raster map
                lcl_filename = 'raster_integral_{}'.format(var)
                lcl_file = out.asc_raster(mp, meta, int_folder, lcl_filename)
                intmaps_files.append(lcl_file)
                # export raster view
                mapid = get_mapid(var)
                #
                ranges = [np.min(integration), np.max(integration)]  # set global ranges
                # plot raster view
                lcl_ttl = '{} of {}\n {} to {} | {} days'.format(lcl_label, var, stamp[0], stamp[len(stamp) - 1], len(stamp))
                plot_map_view(mp, meta, ranges, mapid, mapttl=lcl_ttl,folder=int_folder,
                              filename=lcl_filename, show=False, integration=True)
            #
            # export map list file to main folder:
            lcl_exp_df = pd.DataFrame({'Date': stamp, 'File': lcl_files})
            lcl_file = folder + '/' + 'sim_zmaps_series_' + var + '.txt'
            lcl_exp_df.to_csv(lcl_file, sep=';', index=False)
            zmaps_dct[var] = lcl_file
            mapfiles_lst.append(lcl_file)
        #
        #
        # ****** RASTER EXPORT ******
        #
        #
        if mapraster:
            #
            if tui:
                status('raster map export section')
            #
            #
            # loop in asked variables
            raster_dct = dict()
            for var in mapvar_lst:
                if tui:
                    status('exporting raster maps | {}'.format(var))
                lcl_folder = folder + '/sim_' + var
                #
                # loop across all asked timesteps
                lcl_files = list()
                for t in range(len(stamp)):
                    #
                    lcl_filename = 'raster_{}_{}'.format(var, stamp[t])  # 'raster_' + var + '_' + stamp_str
                    #
                    # express the map
                    mp = map_back(zmatrix=mapped[var][t], a1=twi, a2=shru, bins1=twibins, bins2=shrubins)
                    #
                    # export raster map
                    lcl_file = out.asc_raster(mp, meta, lcl_folder, lcl_filename)
                    lcl_files.append(lcl_file)
                    #
                    # export raster view
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
                    #
                    ranges = [np.min(mapped[var]), np.max(mapped[var])] # set global ranges
                    # plot raster view
                    plot_map_view(mp, meta, ranges, mapid, mapttl='{} | {}'.format(var, stamp[t]),
                                  folder=lcl_folder, filename=lcl_filename, show=False)
                #
                # export map list file to main folder:
                lcl_exp_df = pd.DataFrame({'Date': stamp, 'File': lcl_files})
                lcl_file = folder + '/' + 'sim_raster_series_' + var + '.txt'
                lcl_exp_df.to_csv(lcl_file, sep=';', index=False)
                # allocate in dictionary
                raster_dct[var] = lcl_file
                # append to mapfiles list
                mapfiles_lst.append(lcl_file)
    #
    # report protocols
    end = time.time()
    report_lst.append('Exporting enlapsed time: {:.3f} seconds\n'.format(end - init))
    if tui:
        status('exporting enlapsed time: {:.3f} seconds'.format(end - init), process=False)
    #
    tf = time.time()
    if tui:
        status('Execution enlapsed time: {:.3f} seconds'.format(tf - t0), process=False)
    #
    report_lst.insert(2, 'Execution enlapsed time: {:.3f} seconds\n'.format(tf - t0))
    #
    # output files report
    if pannel:
        outfiles = [exp_file1, exp_file2, exp_file3, sdiag_file1, exp_file4]
    else:
        outfiles = [exp_file1, exp_file2, exp_file3, sdiag_file1]
    output_df = pd.DataFrame({'Main output files': outfiles})
    report_lst.append(output_df.to_string(index=False))
    report_lst.append('\n')
    if mapback:
        output_df = pd.DataFrame({'Map series output files': mapfiles_lst})
        report_lst.append(output_df.to_string(index=False))
        report_lst.append('\n')
    export_report(report_lst, filename='REPORT__simulation', folder=folder, tui=tui)
    #
    # return section
    if pannel:
        out_dct = {'Series':exp_file1,
                   'Histograms':exp_file2,
                   'Parameters':exp_file3,
                   'Pannel':exp_file4,
                   'Folder':folder}
    else:
        out_dct = {'Series': exp_file1,
                   'Histograms': exp_file2,
                   'Parameters': exp_file3,
                   'Folder': folder}
    if mapback:
        out_dct['ZMaps'] = zmaps_dct
        if mapraster:
            out_dct['Raster'] = raster_dct
    return out_dct


def hca(fseries, fhydroparam, fshruparam, fhistograms, fbasinhists, fbasin, ftwi, fshru, fcanopy,
        fzmaps='',
        etpat=False,
        cutdatef=0.3,
        tail=False,
        folder='C:/bin',
        wkpl=False,
        label='',
        tui=False,
        vars='ET-D-VSA-Unz'):
    """

    Hydrology Calibration Assessment for a calibration series

    :param fseries: string filepath to simulation series
    :param fhydroparam: string filepath to parameters csv (txt file)
    :param fshruparam: string filepath to shru parameters csv (txt file)
    :param fhistograms: string filepath to histograms csv (txt file)
    :param fbasinhists: string filepath to basin histograms csv (txt file)
    :param fbasin: string filepath to basin raster map (asc file)
    :param ftwi: string filepath to twi raster map (asc file)
    :param fshru: string filepath to shru raster map (asc file)
    :param fcanopy: string filepath to canopy seasonal factor series (txt file)
    :param fzmaps: string filepath to etpat series of zmaps (txt file)
    :param cutdatef: float fraction (0 to 1) of the validation period
    :param qobs: boolean to include Qobs in visuals
    :param etobs: boolean to include ETobs in visuals
    :param folder: string to folderpath
    :param wkpl: boolean to set folder as workplace
    :param label: string label
    :param tui: boolean to TUI display
    :return:
    """
    from backend import create_rundir
    from visuals import pannel_calib_valid
    from os import mkdir

    def extract_calib_valid(dataframe, fvalid=0.333, tail=False):
        size = len(dataframe)
        if tail:
            cut_id = int(size * (fvalid))
            cut_date = dataframe['Date'].values[cut_id]
            calib_df = dataframe.query('Date >= "{}"'.format(cut_date))
            valid_df = dataframe.query('Date < "{}"'.format(cut_date))
        else:
            cut_id = int(size * (1 - fvalid))
            cut_date = dataframe['Date'].values[cut_id]
            calib_df = dataframe.query('Date < "{}"'.format(cut_date))
            valid_df = dataframe.query('Date >= "{}"'.format(cut_date))
        return calib_df, valid_df, cut_date
    #
    # Run Folder setup
    if tui:
        from tui import status
        status('setting folders')
    if wkpl:  # if the passed folder is a workplace, create a sub folder
        if label != '':
            label = label + '_'
        folder = create_rundir(label=label + 'HCA', wkplc=folder)
    #
    # set up output folders
    calibration_folder = folder + '/calibration_period'
    mkdir(calibration_folder)
    validation_folder = folder + '/validation_period'
    mkdir(validation_folder)
    full_folder = folder + '/full_period'
    mkdir(full_folder)
    #
    # import series and split
    if tui:
        status('importing series')
    series_df = pd.read_csv(fseries, sep=';')
    series_df = dataframe_prepro(series_df, strf=False, date=True, datefield='Date')
    #
    ##### series_df = series_df.query('Date >= "2011-01-01" and Date < "2011-07-01"')
    calib_df, valid_df, cut_date = extract_calib_valid(series_df, fvalid=cutdatef)
    #
    # export separate series:
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
    # run SLH for calibration period
    if tui:
        status('running SLH for calibration period')
    calib_dct = slh(fseries=fcalib_series,
                    fhydroparam=fhydroparam,
                    fshruparam=fshruparam,
                    fhistograms=fhistograms,
                    fbasinhists=fbasinhists,
                    fbasin=fbasin,
                    ftwi=ftwi,
                    fshru=fshru,
                    fcanopy=fcanopy,
                    mapback=False,
                    mapraster=False,
                    integrate=False,
                    qobs=True,
                    folder=calibration_folder,
                    wkpl=False,
                    label=label,
                    tui=tui)
    fsim_calib = calib_dct['Series']
    # run OSA for calibration period
    if tui:
        status('running OSA for calibration period')
    osa_files1 = osa_series(fseries=fsim_calib,
                            fld_obs='Qobs',
                            fld_sim='Q',
                            fld_date='Date',
                            folder=calibration_folder,
                            tui=False)
    #
    #
    # run SLH for validation period
    if tui:
        status('running SLH for validation period')
    valid_dct = slh(fseries=fvalid_series,
                    fhydroparam=fhydroparam,
                    fshruparam=fshruparam,
                    fhistograms=fhistograms,
                    fbasinhists=fbasinhists,
                    fbasin=fbasin,
                    ftwi=ftwi,
                    fshru=fshru,
                    fcanopy=fcanopy,
                    mapback=False,
                    mapraster=False,
                    integrate=False,
                    qobs=True,
                    folder=validation_folder,
                    wkpl=False,
                    label=label,
                    tui=tui)
    fsim_valid = valid_dct['Series']
    # run OSA for validation period
    if tui:
        status('running OSA for validation period')
    osa_files2 = osa_series(fseries=fsim_valid,
                            fld_obs='Qobs',
                            fld_sim='Q',
                            fld_date='Date',
                            folder=validation_folder,
                            tui=False)
    #
    #
    # run SLH for full period
    if tui:
        status('running SLH for full period')
    # Import Observed Zmaps dataframe series
    zmaps_obs_df = pd.read_csv(fzmaps, sep=';')
    zmaps_obs_df = dataframe_prepro(zmaps_obs_df, strfields='File', date=True)
    mapdates = ' & '.join(zmaps_obs_df['Date'].astype('str'))
    full_dct = slh(fseries=ffull_series,
                    fhydroparam=fhydroparam,
                    fshruparam=fshruparam,
                    fhistograms=fhistograms,
                    fbasinhists=fbasinhists,
                    fbasin=fbasin,
                    ftwi=ftwi,
                    fshru=fshru,
                    fcanopy=fcanopy,
                    mapback=etpat,
                    mapraster=True,
                    integrate=False,
                    mapvar=vars,
                    mapdates=mapdates,
                    qobs=True,
                    folder=full_folder,
                    wkpl=False,
                    label=label,
                    tui=tui)
    fsim_full = full_dct['Series']
    # run OSA for full period
    if tui:
        status('running OSA for full period')
    osa_files3 = osa_series(fseries=fsim_full,
                            fld_obs='Qobs',
                            fld_sim='Q',
                            fld_date='Date',
                            folder=full_folder,
                            tui=tui)
    # run OSA on zmaps for full period
    if etpat and fzmaps != '':
        if tui:
            status('running ZMAP OSA for full period')
        fzmaps_sim = full_folder + '/' + 'sim_zmaps_series_ET.txt'
        osa_zmaps_folder = full_folder + '/osa_zmaps'
        mkdir(osa_zmaps_folder)
        osa_zmaps(fobs_series=fzmaps,
                  fsim_series=fzmaps_sim,
                  fhistograms=fhistograms,
                  fseries=fsim_full,
                  fshru=fshru,
                  ftwi=ftwi,
                  folder=osa_zmaps_folder,
                  wkpl=False,
                  tui=tui,
                  raster=True)
    #
    # exports
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
    return {'Folder':folder, 'CalibFolder':calibration_folder, 'ValidFolder':validation_folder, 'FullFolder':full_folder}


def calibrate(fseries, fhydroparam, fshruparam, fhistograms, fbasinhists, fbasin, ftwi, fshru, fcanopy,
              fetpatzmaps='',
              etpat=False,
              folder='C:/bin',
              tui=False,
              mapback=True,
              mapvar='ET-D-Unz-VSA',
              qobs=True,
              cutdatef=0.3,
              tail=False,
              generations=100,
              popsize=200,
              likelihood='KGE',
              label='',
              normalize=False):
    """
    Calibration tool
    :param fseries: string filepath to input series txt file
    :param fhydroparam: string filepath to parameters txt file
    :param fshruparam: string filepath to shru parameters txt file
    :param fhistograms: string filepath to histograms zmap txt file
    :param fbasinhists: string filepath to basin histograms zmap txt file
    :param fbasin: string filepath to basin asc file
    :param ftwi: string filepath to twi asc file
    :param fshru: string filepath to shru asc file
    :param fcanopy: string filepath to canopy series txt file
    :param fetpatzmaps: string filepath to etpat zmaps series txt file
    :param etpat: boolean to use etpat series in calibration
    :param folder: string filepath to workplace folder
    :param tui: boolean for display
    :param mapvar: string code of variables to mapback during MLM simulation (see SLH docstring)
    :param qobs: boolean to inform qobs during MLM simulation
    :param cutdatef: float cut date fraction (0 to 1) for validation period
    :param tail: boolean to use the series tail as the calibration set
    :param generations: int number of generations to run the calibration
    :param popsize: int number of models in each generation
    :param likelihood: string code for Q calibration.

    Options: 'NSE', 'NSElog', 'RMSE', 'RMSElog', 'KGE', 'KGElog', 'PBias', 'RMSE-CFC', 'RMSElog-CFC'

    :param label: string label to append in output folder
    :param normalize: boolean to normalize ET values into an fuzzy ET pattern
    :return: dictionary of run folder and files
    """
    from inp import histograms
    from hydrology import avg_2d, simulation, map_back, calibration
    from visuals import pannel_global
    from backend import create_rundir, get_stringfields
    import time
    from os import mkdir

    def clear_prec_by_dates(dseries, datesstr):
        def_df = dseries.copy()
        mapdates_df = pd.DataFrame({'Date': datesstr.split('&')})
        mapdates_df['Date'] = mapdates_df['Date'].str.strip()
        mapdates_df['Date'] = pd.to_datetime(mapdates_df['Date'])
        lookup_dates = mapdates_df['Date'].values  # it is coming as datetime!
        for i in range(len(lookup_dates)):
            index = def_df[def_df['Date'] == lookup_dates[i]].index
            def_df.loc[index, 'Prec'] = 0.0
        return def_df

    def extract_calib_valid(dataframe, fvalid=0.333, tail=False):
        size = len(dataframe)
        if tail:
            cut_id = int(size * (fvalid))
            cut_date = dataframe['Date'].values[cut_id]
            calib_df = dataframe.query('Date >= "{}"'.format(cut_date))
        else:
            cut_id = int(size * (1 - fvalid))
            cut_date = dataframe['Date'].values[cut_id]
            calib_df = dataframe.query('Date < "{}"'.format(cut_date))
        return calib_df, cut_date

    def extract_ranges(fhydroparam):
        _dct, hydroparam_df = inp.hydroparams(fhydroparam=fhydroparam)
        #
        # extract set range values
        out_dct = {'Params_df':hydroparam_df,
                   'm': (_dct['m']['Min'], _dct['m']['Max']),
                   'lamb': (_dct['lamb']['Min'], _dct['lamb']['Max']),
                   'qo': (_dct['qo']['Min'], _dct['qo']['Max']),
                   'cpmax': (_dct['cpmax']['Min'], _dct['cpmax']['Max']),
                   'sfmax': (_dct['sfmax']['Min'], _dct['sfmax']['Max']),
                   'erz': (_dct['erz']['Min'], _dct['erz']['Max']),
                   'ksat': (_dct['ksat']['Min'], _dct['ksat']['Max']),
                   'c': (_dct['c']['Min'], _dct['c']['Max']),
                   'k': (_dct['k']['Min'], _dct['k']['Max']),
                   'n': (_dct['n']['Min'], _dct['n']['Max']),
                   'lat':_dct['lat']['Set']}
        return out_dct

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
    # Run Folder setup
    if label != '':
        label = label + '_'
    folder = create_rundir(label=label + 'Hydrology_' + likelihood, wkplc=folder)
    #
    t0 = time.time()
    if tui:
        from tui import status
        init = time.time()
        print('\n\t**** Load Data Protocol ****\n')
        status('loading time series')
    #
    #
    # ******* IMPORT DATA *******
    #
    # Series
    series_df =  pd.read_csv(fseries, sep=';')
    series_df = dataframe_prepro(series_df, strf=False, date=True, datefield='Date')
    # get the cut date
    calib_df, cut_date = extract_calib_valid(series_df, fvalid=cutdatef, tail=tail)
    # Hydro param
    if tui:
        status('loading hydrology parameters') #print(' >>> loading hydrology parameters...')
    rng_dct = extract_ranges(fhydroparam=fhydroparam)
    hydroparam_df = rng_dct['Params_df']
    lat = rng_dct['lat']
    # SHRU param
    if tui:
        status('loading SHRU parameters')
    shru_df = pd.read_csv(fshruparam, sep=';')
    #aux_str = 'SHRUName,SHRUAlias,LULCName,LULCAlias,CanopySeason,ConvertTo,ColorLULC,SoilName,SoilAlias,ColorSoil'
    shru_df = dataframe_prepro(shru_df, get_stringfields(fshruparam.split('/')[-1]))
    #
    # canopy series pattern
    if tui:
        status('loading Canopy series pattern')
    canopy_df = pd.read_csv(fcanopy, sep=';')
    #
    #
    # extract count matrix (full map extension)
    if tui:
        status('loading histograms of full extension')
    count, twibins, shrubins = histograms(fhistograms=fhistograms)
    #
    # extract count matrix (basin)
    if tui:
        status('loading histograms of basin')
    basincount, twibins2, shrubins2 = histograms(fhistograms=fbasinhists)
    #
    # get boundary conditions
    if tui:
        status('loading boundary conditions')
    meta = inp.asc_raster_meta(fbasin)
    area = np.sum(basincount) * meta['cellsize'] * meta['cellsize']
    qt0 = 0.01  # fixed
    if qobs:
        qt0 = series_df['Q'].values[0]
    #
    #
    # ******* ET PAT ZMAPS *******
    if etpat:
        # extract etpat zmaps series for calibration
        if tui:
            status('loading OBS ETPat Z-maps')
        #
        # Import Observed Zmaps
        etpat_zmaps_obs_df = pd.read_csv(fetpatzmaps, sep=';')
        etpat_zmaps_obs_df = dataframe_prepro(etpat_zmaps_obs_df, strfields='File', date=True)
        # split dataframes for later
        if tail:
            etpat_zmaps_obs_calib_df = etpat_zmaps_obs_df.query('Date >= "{}"'.format(cut_date))
            etpat_zmaps_obs_valid_df = etpat_zmaps_obs_df.query('Date < "{}"'.format(cut_date))
        else:
            etpat_zmaps_obs_calib_df = etpat_zmaps_obs_df.query('Date < "{}"'.format(cut_date))
            etpat_zmaps_obs_valid_df = etpat_zmaps_obs_df.query('Date >= "{}"'.format(cut_date))
        #
        # get a dataframe to store each date series
        etpat_calib_dates = pd.DataFrame({'Date': etpat_zmaps_obs_calib_df['Date']})
        etpat_valid_dates = pd.DataFrame({'Date': etpat_zmaps_obs_valid_df['Date']})
        #
        # extract dates to string code for calibration
        etpat_dates_str_calib = ' & '.join(etpat_zmaps_obs_calib_df['Date'].astype('str').values)  # for calibration!!
        etpat_dates_str_full = ' & '.join(etpat_zmaps_obs_df['Date'].astype('str').values)
        #
        # clear prec by dates:
        series_df = clear_prec_by_dates(dseries=series_df, datesstr=etpat_dates_str_full)
        #
        # now load zmaps for calibration
        etpat_zmaps_obs_calib = list()
        for i in range(len(etpat_zmaps_obs_calib_df)):
            zmap_file = etpat_zmaps_obs_calib_df['File'].values[i]
            zmap, ybins, xbins = inp.zmap(zmap_file)
            etpat_zmaps_obs_calib.append(zmap)
        # array of zmaps
        etpat_zmaps_obs_calib = np.array(etpat_zmaps_obs_calib)
    else:
        etpat_dates_str_calib = None
        etpat_zmaps_obs_calib = None
    #
    # split series again (now for real)
    calib_df, cut_date = extract_calib_valid(series_df, fvalid=cutdatef, tail=tail)
    # export to file and update fseries
    fseries = '{}/calibration_series.txt'.format(folder)
    series_df.to_csv(fseries, sep=';', index=False)
    #
    #
    end = time.time()
    if tui:
        status('loading enlapsed time: {:.3f} seconds'.format(end - init))
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
    cal_dct = calibration(series=calib_df,
                          shruparam=shru_df,
                          twibins=twibins,
                          countmatrix=count,
                          canopy=canopy_df,
                          qt0=qt0,
                          lat=lat,
                          area=area,
                          basinshadow=basincount,
                          p_ranges=rng_dct,
                          etpatdates=etpat_dates_str_calib,
                          etpatzmaps=etpat_zmaps_obs_calib,
                          tui=tui,
                          generations=generations,
                          popsize=popsize,
                          likelihood=likelihood,
                          tracefrac=1,
                          tracepop=True,
                          normalize=normalize,
                          etpat=etpat)
    end = time.time()
    if tui:
        print('\nCalibration enlapsed time: {:.3f} seconds'.format(end - init))
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
    #
    # export traced parents
    exp_file5 = lcl_folder + '/' + 'traced_parents.txt'
    cal_dct['Traced'].to_csv(exp_file5, sep=';', index=False)
    #
    # export traced population
    exp_file6 = lcl_folder + '/' + 'population.txt'
    cal_dct['Population'].to_csv(exp_file6, sep=';', index=False)
    #
    #
    #
    # ********* MAXIMUM LIKELIHOOD MODEL ASSESSMENT *********
    #
    #
    # create MLM folder
    mlm_folder = folder + '/' + 'MLM'
    mkdir(mlm_folder)
    #
    # best set parameters export
    if tui:
        print(' >>> exporting best set run parameters...')
    fhydroparam_mlm = mlm_folder + '/' + 'mlm_parameters.txt'
    hydroparam_df['Set'] = [cal_dct['MLM'][0],
                            cal_dct['MLM'][1],
                            cal_dct['MLM'][2],
                            cal_dct['MLM'][3],
                            cal_dct['MLM'][4],
                            cal_dct['MLM'][5],
                            cal_dct['MLM'][6],
                            cal_dct['MLM'][7],
                            lat,
                            cal_dct['MLM'][8],
                            cal_dct['MLM'][9]]
    hydroparam_df.to_csv(fhydroparam_mlm, sep=';', index=False)
    #
    # run HCA for calibration basin
    slh_dct = hca(fseries=fseries,
                  fhydroparam=fhydroparam_mlm,
                  fshruparam=fshruparam,
                  fhistograms=fhistograms,
                  fbasinhists=fbasinhists,
                  fbasin=fbasin,
                  ftwi=ftwi,
                  fshru=fshru,
                  fcanopy=fcanopy,
                  fzmaps=fetpatzmaps,
                  tui=tui,
                  folder=mlm_folder,
                  vars=mapvar,
                  etpat=etpat,
                  tail=tail)
    # extract folders:
    calib_folder = slh_dct['CalibFolder']
    valid_folder = slh_dct['ValidFolder']
    full_folder = slh_dct['FullFolder']
    #
    #
    # return dictionary
    return {'Folder':folder}


def glue(fseries, fmodels, fhydroparam, fshruparam, fhistograms, fbasinhists, fbasin, fcanopy,
         nmodels='all',
         modelid='SetIds',
         likelihood='L',
         criteria='>',
         behavioural=0.1,
         sampling_grid=100,
         run_ensemble=True,
         folder='C:/bin',
         wkpl=False,
         tui=False,
         normalize=False,
         label=''):
    """

    GLUE tool

    :param fseries: string - filepath to input series txt file
    :param fmodels: string - filepath to models dataframe txt file
    :param fhydroparam: string - filepath
    :param fshruparam: string - filepath to shru param
    :param fhistograms: string - filepath
    :param fbasinhists: string - filepath
    :param fbasin: string - filepath
    :param nmodels: int or string - number of models or the string code 'all' for all models
    :param modelid: string - field of model id
    :param likelihood: string - field of likelihood
    :param criteria: string - query criteria for behavioural models '<', '==' or '>'
    :param behavioural: float - likelihood value to apply the behavioural criteria
    :param sampling_grid: int - ??
    :param folder: string - output folder
    :param wkpl: boolean to set folder as workplace
    :param tui: boolean to display
    :param label: string - label to append in folder name
    :return:
    """
    from backend import create_rundir, get_stringfields
    from visuals import glue_scattergram
    from hydrology import ensemble
    from visuals import glue_ensemble, glue_posterior
    import inp

    def extract_ranges(fhydroparam):
        dct, hydroparam_df = inp.hydroparams(fhydroparam=fhydroparam)
        #
        # extract set range values
        out_dct = {'Params_df':hydroparam_df,
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
                   'lat':dct['lat']['Set']}
        return out_dct

    def extract_histdata(fhistograms):
        dataframe = pd.read_csv(fhistograms, sep=';')
        dataframe = dataframe_prepro(dataframe, strf=False)
        dataframe = dataframe.set_index(dataframe.columns[0])
        shru_ids = dataframe.columns.astype('int')
        twi_bins = dataframe.index.values
        count_matrix = dataframe.values
        return count_matrix, twi_bins, shru_ids
    #
    #
    # folder setup
    if tui:
        from tui import status
        status('setting folders')
    if wkpl:  # if the passed folder is a workplace, create a sub folder
        if label != '':
            label = label + '_'
        folder = create_rundir(label=label + 'GLUE_{}'.format(likelihood), wkplc=folder)
    #
    #
    # ******* IMPORT DATA *******
    #
    # Series
    series_df = pd.read_csv(fseries, sep=';')
    series_df = dataframe_prepro(series_df, strf=False, date=True, datefield='Date')
    #
    if tui:
        status('loading hydrology parameters')  # print(' >>> loading hydrology parameters...')
    rng_dct = extract_ranges(fhydroparam=fhydroparam)
    hydroparam_df = rng_dct['Params_df']
    lat = rng_dct['lat']
    params = ('m', 'lamb', 'qo', 'cpmax', 'sfmax', 'erz', 'ksat', 'c', 'k', 'n')
    #
    if tui:
        status('loading SHRU parameters')
    shru_df = pd.read_csv(fshruparam, sep=';')
    shru_df = dataframe_prepro(shru_df, get_stringfields(fshruparam.split('/')[-1]))
    #
    # extract countmatrix (full map extension)
    if tui:
        status('loading histograms of full extension')
    count, twibins, shrubins = extract_histdata(fhistograms=fhistograms)
    #
    # extract count matrix (basin)
    if tui:
        status('loading histograms of basin')
    basincount, twibins2, shrubins2 = extract_histdata(fhistograms=fbasinhists)
    #
    # canopy series pattern
    if tui:
        status('loading Canopy series pattern')
    canopy_df = pd.read_csv(fcanopy, sep=';')
    #
    # import models dataframe:
    if tui:
        status('loading GLUE models')
    models_df = pd.read_csv(fmodels, sep=';')
    # get boundary conditions
    if tui:
        status('loading boundary conditions')
    meta = inp.asc_raster_meta(fbasin)
    area = np.sum(basincount) * meta['cellsize'] * meta['cellsize']
    qt0 = series_df['Qobs'].values[0]
    #
    #
    #
    # ******* SELECTION OF BEHAVIOURAL MODELS *******
    #
    #
    if tui:
        status('selecting behavioural models')
    # get unique models
    models_df = models_df.drop_duplicates(subset=[modelid])
    # filter behavioural models
    behav_df = models_df.query('{} {} {}'.format(likelihood, criteria, behavioural))
    if nmodels != 'all':
        # extract n models
        behav_df = behav_df.nlargest(nmodels, columns=[likelihood])
    if normalize:
        behav_df_norm = behav_df.copy()
        behav_df_norm[likelihood] = (behav_df_norm[likelihood] - behav_df_norm[likelihood].min()) / \
                                    (behav_df_norm[likelihood].max() - behav_df_norm[likelihood].min())
        behavioural_norm = 0.0
    #
    # export behaviroural models:
    if tui:
        status('exporting behavioural models datasets')
    exp_file0 = '{}/behaviroural.txt'.format(folder)
    behav_df = behav_df.sort_values(by=likelihood, ascending=False)
    behav_df.to_csv(exp_file0, sep=';', index=False)
    if normalize:
        exp_file0 = '{}/behaviroural_normalized.txt'.format(folder)
        behav_df_norm = behav_df_norm.sort_values(by=likelihood, ascending=False)
        behav_df_norm.to_csv(exp_file0, sep=';', index=False)
    #
    # export visual scattergrams:
    if tui:
        status('exporting behavioural models scattergram')
    exp_file1 = glue_scattergram(behav_df, rng_dct,
                                 likelihood=likelihood,
                                 criteria=criteria,
                                 behavioural=behavioural,
                                 folder=folder,
                                 filename='scattergrams')
    if normalize:
        exp_file1 = glue_scattergram(behav_df_norm, rng_dct,
                                     likelihood=likelihood,
                                     criteria=criteria,
                                     behavioural=behavioural_norm,
                                     folder=folder,
                                     filename='scattergrams_normalized')
    #
    # ******* POSTERIOR BAYESIAN ANALYSIS *******
    #
    #
    # compute posterior CDFs and posterior ranges (90%)
    #
    #
    if tui:
        status('performing posterior GLUE analysis')
    #
    # prior accumulated likelihood
    lo_acc = np.linspace(0, 1, sampling_grid)
    posterior_df = pd.DataFrame({'Lo':lo_acc[1], 'Lo_acc':lo_acc})
    for i in range(len(params)):
        lcl_param = params[i]
        lcl_param_rng = '{}_rng'.format(lcl_param)
        param_grid = rng_dct[lcl_param_rng][0] + (rng_dct[lcl_param_rng][1] - rng_dct[lcl_param_rng][0]) * lo_acc
        #
        # aggregate observed likelihood ly
        ly_agg = np.zeros(len(lo_acc))
        for i in range(len(param_grid)):
            if i == 0:
                ly_agg[i] = 0
            else:
                lcl_query = '{} > {} and {} <= {}'.format(lcl_param, param_grid[i - 1], lcl_param, param_grid[i])
                lcl_df = behav_df.query(lcl_query)
                ly_agg[i] = np.sum(lcl_df[likelihood].values - behavioural)
        #
        # compute posterior likelihood
        lp = lo_acc[1] * ly_agg / np.sum(lo_acc[1] * ly_agg)
        #
        # accumulate posterior likelihood
        lp_acc = np.zeros(len(lp))
        for i in range(len(param_grid)):
            if i == 0:
                lp_acc[i] = 0
            else:
                lp_acc[i] = lp_acc[i - 1] + lp[i]
        # append to dataframe
        posterior_df['{}'.format(lcl_param)] = param_grid
        posterior_df['{}_Ly_agg'.format(lcl_param)] = ly_agg
        posterior_df['{}_Lp_acc'.format(lcl_param)] = lp_acc
        q_str = '{}_Lp_acc <= 0.05'.format(lcl_param)
        posterior_df['{}_Lp_5'.format(lcl_param)] = posterior_df.query(q_str)[lcl_param].max()
        q_str = '{}_Lp_acc <= 0.95'.format(lcl_param)
        posterior_df['{}_Lp_95'.format(lcl_param)] = posterior_df.query(q_str)[lcl_param].max()
    # export posterior analysis:
    if tui:
        status('exporting posterior analysis')
    exp_file2 = '{}/posterior_analysis.txt'.format(folder)
    posterior_df.to_csv(exp_file2, sep=';', index=False)
    label = 'Criteria: {} {} {} | N = {}'.format(likelihood, criteria, behavioural, len(behav_df))
    glue_posterior(posterior_df,
                   rng_dct=rng_dct,
                   label=label,
                   folder=folder)

    #
    #
    #
    if run_ensemble:
        #
        # ******* FLOW ENSEMBLE *******
        #
        #
        if tui:
            status('computing ensemble datasets')
        ensb_dct = ensemble(series=series_df,
                            models_df=behav_df,
                            shruparam=shru_df,
                            twibins=twibins,
                            countmatrix=count,
                            canopy_df=canopy_df,
                            qt0=qt0,
                            lat=lat,
                            area=area,
                            basinshadow=basincount,
                            tui=tui)
        #
        # export ensemble dataframes
        if tui:
            status('exporting ensemble datasets')
        exp_en_q = '{}/ensemble_q.txt'.format(folder)
        ensb_dct['Q'].to_csv(exp_en_q, sep=';', index=False)
        exp_en_q = '{}/ensemble_qb.txt'.format(folder)
        ensb_dct['Qb'].to_csv(exp_en_q, sep=';', index=False)
        #
        # plot ensemble
        if tui:
            status('plotting ensemble datasets')
        glue_ensemble(sim_df=series_df,
                      ensemble_df=ensb_dct['Q'],
                      filename='ensemble_q_log',
                      folder=folder)
        glue_ensemble(sim_df=series_df,
                      ensemble_df=ensb_dct['Q'],
                      filename='ensemble_q_lin',
                      scale='lin',
                      folder=folder)
        glue_ensemble(sim_df=series_df,
                      ensemble_df=ensb_dct['Qb'],
                      baseflow=True,
                      filename='ensemble_qb_log',
                      folder=folder)
        glue_ensemble(sim_df=series_df,
                      ensemble_df=ensb_dct['Qb'],
                      baseflow=True,
                      filename='ensemble_qb_lin',
                      scale='lin',
                      folder=folder)
    #
    #
    #
    return 666


def sal_d_by_twi(ftwi1, ftwi2, m=10, dmax=100, size=100, label='', wkpl=False, folder='C:/bin'):
    """
    SAL of deficit by changing TWI
    :param ftwi1: string filepath to .asc raster map of TWI 1
    :param ftwi2: string filepath to .asc raster map of TWI 2
    :param m: int of m parameter
    :param dmax: int of max deficit
    :param size: int size of SAL
    :param label: string file label
    :param wkpl: boolen to set the output folder as workplace
    :param folder: string file path to output folder
    :return: none
    """
    from hydrology import topmodel_di, topmodel_vsai
    from visuals import sal_deficit_frame
    from backend import create_rundir

    def id_label(id):
        if id < 10:
            return '000' + str(id)
        elif id >= 10 and id < 100:
            return '00' + str(id)
        elif id >= 100 and id < 1000:
            return '0' + str(id)
        elif id >= 1000 and id < 10000:
            return  str(id)

    # folder setup
    if wkpl:  # if the passed folder is a workplace, create a sub folder
        if label != '':
            label = label + '_'
        folder = create_rundir(label=label + 'SAL_D_by_TWI', wkplc=folder)

    # load twi maps
    meta, twi1 = inp.asc_raster(file=ftwi1, dtype='float32')
    lamb1 = np.mean(twi1)
    meta, twi2 = inp.asc_raster(file=ftwi2, dtype='float32')
    lamb2 = np.mean(twi2)

    d = np.linspace(0, dmax, size)
    for i in range(len(d)):
        lcl_d = d[i]
        print(i)
        lcl_di_1 = topmodel_di(d=lcl_d, twi=twi1, m=m, lamb=lamb1)
        lcl_di_2 = topmodel_di(d=lcl_d, twi=twi2, m=m, lamb=lamb2)
        lcl_vsai_1 = topmodel_vsai(di=lcl_di_1)
        lcl_vsai_2 = topmodel_vsai(di=lcl_di_2)
        # plot frame
        lcl_flnm = 'sal_d_by_twi__{}'.format(id_label(id=i))
        sal_deficit_frame(dgbl=lcl_d, d1=lcl_di_1, d2=lcl_di_2, m1=m, m2=m, vsa1=lcl_vsai_1, vsa2=lcl_vsai_2,
                          dgbl_max=dmax, filename=lcl_flnm, folder=folder, supttl='Sensitivity to TWI')


def sal_d_by_m(ftwi, m1=10, m2=500, dmax=100, size=100, label='', wkpl=False, folder='C:/bin'):
    """
    SAL of deficit by changing m
    :param ftwi: string filepath to .asc raster map of TWI
    :param m1: float of m parameter 1
    :param m2: float of m parameter 2
    :param dmax: int of max deficit
    :param size: int size of SAL
    :param label: string file label
    :param wkpl: boolen to set the output folder as workplace
    :param folder: string file path to output folder
    :return: none
    """
    from hydrology import topmodel_di, topmodel_vsai
    from visuals import sal_deficit_frame
    from backend import create_rundir

    def id_label(id):
        if id < 10:
            return '000' + str(id)
        elif id >= 10 and id < 100:
            return '00' + str(id)
        elif id >= 100 and id < 1000:
            return '0' + str(id)
        elif id >= 1000 and id < 10000:
            return  str(id)

    # folder setup
    if wkpl:  # if the passed folder is a workplace, create a sub folder
        if label != '':
            label = label + '_'
        folder = create_rundir(label=label + 'SAL_D_by_m__{}_{}'.format(str(int(m1)), str(int(m1))), wkplc=folder)

    # load twi maps
    meta, twi = inp.asc_raster(file=ftwi, dtype='float32')
    # standard lambda:
    lamb_mean = np.mean(twi)

    d = np.linspace(0, dmax, size)
    for i in range(len(d)):
        lcl_d = d[i]
        print(i)
        lcl_di_1 = topmodel_di(d=lcl_d, twi=twi, m=m1, lamb=lamb_mean)
        lcl_di_2 = topmodel_di(d=lcl_d, twi=twi, m=m2, lamb=lamb_mean)
        lcl_vsai_1 = topmodel_vsai(di=lcl_di_1)
        lcl_vsai_2 = topmodel_vsai(di=lcl_di_2)
        # plot frame
        lcl_flnm = 'sal_d_by_m__{}'.format(id_label(id=i))
        sal_deficit_frame(dgbl=lcl_d, d1=lcl_di_1, d2=lcl_di_2, p1=m1, p2=m2, p_lbl='m', vsa1=lcl_vsai_1, vsa2=lcl_vsai_2,
                          dgbl_max=dmax, filename=lcl_flnm, folder=folder, supttl='Sensitivity to m | lamb={}'.format(lamb_mean))


def sal_d_by_lamb(ftwi, m=10, lamb1=5, lamb2=15, dmax=100, size=100, label='', wkpl=False, folder='C:/bin'):
    """
    SAL of deficit by changing Lambda
    :param ftwi: string filepath to .asc raster map of TWI
    :param lamb1: float of lamb parameter 1
    :param lamb2: float of lamb parameter 2
    :param dmax: int of max deficit
    :param size: int size of SAL
    :param label: string file label
    :param wkpl: boolen to set the output folder as workplace
    :param folder: string file path to output folder
    :return: none
    """
    from hydrology import topmodel_di, topmodel_vsai
    from visuals import sal_deficit_frame
    from backend import create_rundir

    def id_label(id):
        if id < 10:
            return '000' + str(id)
        elif id >= 10 and id < 100:
            return '00' + str(id)
        elif id >= 100 and id < 1000:
            return '0' + str(id)
        elif id >= 1000 and id < 10000:
            return  str(id)

    # folder setup
    if wkpl:  # if the passed folder is a workplace, create a sub folder
        if label != '':
            label = label + '_'
        folder = create_rundir(label=label + 'SAL_D_by_lamb__{}_{}'.format(str(int(lamb1)), str(int(lamb2))), wkplc=folder)

    # load twi maps
    meta, twi = inp.asc_raster(file=ftwi, dtype='float32')

    d = np.linspace(0, dmax, size)
    for i in range(len(d)):
        lcl_d = d[i]
        print(i)
        lcl_di_1 = topmodel_di(d=lcl_d, twi=twi, m=m, lamb=lamb1)
        lcl_di_2 = topmodel_di(d=lcl_d, twi=twi, m=m, lamb=lamb2)
        lcl_vsai_1 = topmodel_vsai(di=lcl_di_1)
        lcl_vsai_2 = topmodel_vsai(di=lcl_di_2)
        # plot frame
        lcl_flnm = 'sal_d_by_lamb__{}'.format(id_label(id=i))
        sal_deficit_frame(dgbl=lcl_d, d1=lcl_di_1, d2=lcl_di_2, p1=lamb1, p2=lamb2, p_lbl='lamb', vsa1=lcl_vsai_1, vsa2=lcl_vsai_2,
                          dgbl_max=dmax, vmax=dmax, filename=lcl_flnm, folder=folder, supttl='Sensitivity to lambda | m={}'.format(m))