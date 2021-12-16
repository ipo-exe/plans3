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
This module stores all hydrology functions of plans3.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # todo remove this after release

# auxiliar functions
import analyst


def avg_2d(var2d, weight):
    """
    average raster value based on weight mask
    :param var2d: 2d numpy array of raster values
    :param weight: 2d numpy array of weight mask
    :return: float average value
    """
    lcl_avg = np.sum(var2d * weight) / np.sum(weight)  # avg
    return lcl_avg


def flatten_clear(array, mask, nodata=0):
    """
    convert an nd numpy array to 1d of cleared non-nan values
    :param array: nd numpy array
    :param mask: nd pseudo-boolean numpy array of mask (0 and 1)
    :return: 1d numpy array of cleared non-nan values
    """
    masked = np.copy(array)  # get copy
    masked[masked == nodata] = np.nan  # set nodata in array
    masked[mask == 0] = np.nan  # set nodata to 0 in mask
    flatten = masked.flatten()  # flatten
    cleared = masked[~np.isnan(masked)]  # clear nan values from flat array
    return cleared


def convert_q2sq(q, area):
    """
    convert discharge to specific discharge
    :param q: 1d array of q in m3/s
    :param area: watershed area in m2
    :return: 1d array of Q in mm/d
    """
    lcl_sq = q * 1000 * 86400 / area  #  m3/s * 1000 mm/m * 86400 s/d / m2
    return lcl_sq


def convert_sq2q(sq, area):
    """
    convert specific discharge to discharge
    :param sq: 1d array of specific discharge in mm/d
    :param area: watershed area in m2
    :return: 1d array of q in m3/s
    """
    lcl_q = sq * area / (1000 * 86400)
    return lcl_q

#
#
#
# general silent functions
def nash_cascade(q, k, n):
    """
    Runoff routing model of multiple linear reservoirs (Nash Cascade)
    :param q: 1d numpy array of runoff
    :param k: float of residence time in time step units
    :param n: float of equivalent number of reservoirs in Nash Cascade
    :return: 1d numpy array of routed runoff
    """
    from scipy.special import gamma
    size = len(q)
    time = np.arange(0, size)
    nash = np.power((time / k), (n - 1)) * np.exp(- time / k) / (k * gamma(n))  # the Nash Cascade time function
    for t in range(0, len(time)):
        lcl_q = q[t]
        if t == 0:
            qs = lcl_q * nash
        else:
            qs[t:] = qs[t:] + lcl_q * nash[:size - t]
    return qs


def count_matrix(twi, shru, aoi, shrubins, twibins):
    """

    Countmatrix function. Note: maps must have the same dimensions!

    :param twi: 2d numpy array of TWI raster map
    :param shru: 2d numpy array of SHRU raster map
    :param aoi: 2d numpy array of AOI pseudo-boolean raster map
    :param shrubins: 1d numpy array of SHRU id bins
    :param twibins: 1d numpy array of TWI bins
    :return: 2d numpy array of Countmatrix, 1d array of TWI bins (echo) and 1d array of SHRU id bins (echo)
    """
    countmatrix = np.zeros(shape=(len(twibins), len(shrubins)), dtype='int32')
    for i in range(len(countmatrix)):
        for j in range(len(countmatrix[i])):
            if i == 0:
                lcl_mask =  (shru == shrubins[j]) * (twi < twibins[i]) * aoi
            elif i == len(countmatrix) - 1:
                lcl_mask = (shru == shrubins[j]) * (twi >= twibins[i - 1]) * aoi
            else:
                lcl_mask = (shru == shrubins[j]) * (twi >= twibins[i - 1]) * (twi < twibins[i]) * aoi
            countmatrix[i][j] = np.sum(lcl_mask)  # insert sum of pixels found in local HRU
    return countmatrix, twibins, shrubins


def built_zmap(varmap, twi, shru, twibins, shrubins, nodata=-1.0):
    """
    Built a ZMAP of a variable map . Note: maps must have the same dimensions!
    :param varmap: 2d numpy array of variable raster map
    :param twi: 2d numpy array of twi raster map
    :param shru: 2d numpy array of SHRU raster map
    :param twibins: 1d numpy array of TWI bins
    :param shrubins: 1d numpy array of SHRU id bins
    :param nodata: float of standard no data value
    :return: 2d numpy array of variable ZMAP
    """
    import matplotlib.pyplot as plt
    # first mask by the nodata value
    aoivar = 1.0 * (varmap != nodata)
    # deploy the zmap matrix
    zmap = np.zeros(shape=(len(twibins), len(shrubins)))
    # loop in zmap matrix
    for i in range(len(zmap)):
        for j in range(len(zmap[i])):
            if i == 0:
                lcl_mask =  (shru == shrubins[j]) * (twi < twibins[i]) * aoivar  # get the HRU mask
            elif i == len(varmap) - 1:
                lcl_mask = (shru == shrubins[j]) * (twi >= twibins[i - 1]) * aoivar
            else:
                lcl_mask = (shru == shrubins[j]) * (twi >= twibins[i - 1]) * (twi < twibins[i]) * aoivar
            #plt.imshow(lcl_mask)
            #plt.show()
            if np.sum(lcl_mask) == 0.0:  # not found any local HRU within the AOI of var
                zmap[i][j] = nodata
            else:
                # take the mean variable value at the local HRU mask
                zmap[i][j] = np.sum(varmap * lcl_mask) / np.sum(lcl_mask)
        #plt.imshow(zmap)
        #plt.show()
    return zmap


def extract_map_signal(zmap, mask, nodata=-1):
    """
    Extract the signal of a map array
    :param zmap: 2d array of zmap
    :param mask: 3d array of pseudo-boolean mask or AOI
    :param nodata: float of no data
    :return: 1d array of zmap signal
    """
    array_flat = zmap.flatten()
    mask_flat = (1 * (mask.flatten() > 0.0))
    return flatten_clear(array=array_flat, mask=mask_flat, nodata=nodata)


def extract_sim_diagnostics(simseries, vars='all'):
    """
    Extract a diagnostics from the simulated dataframe
    :param simseries: pandas DataFrame of simulated series - see output from hydrology.simulation()
    :param vars: string of variables alias joined by '-' or 'all'. Available:
    'Prec-Temp-IRA-IRI-PET-D-Cpy-TF-Sfs-R-RSE-RIE-RC-Inf-Unz-Qv-Evc-Evs-Tpun-Tpgw-ET-VSA-Q-Qs-Qb'
    :return: pandas DataFrame of Diagnostics parameter for each variable. Parameters:
    Sum: sum of values for flow variables
    Mean: mean of all variables
    SD: Standard deviation for all variables
    N: number of events higher than 0.0
    Mean_N: mean of events
    SD_N: standard deviation of events
    C_Prec: Flow coeficient related to Prec
    C_Input: Flow coeficient related to total input (Prec + IRI + IRA)

    """
    from backend import get_all_vars
    parameters = ('Sum', 'Mean', 'SD', 'N', 'Mean_N', 'SD_N', 'C_Prec', 'C_Input')
    if vars == 'all':
        variables = get_all_vars().split('-')
    else:
        variables = vars.split('-')
    stocks = ('Cpy', 'Sfs', 'Unz', 'D', 'Temp')
    vnonwater = ('RC', 'VSA', 'Temp')
    prec = simseries['Prec'].values
    tinput = simseries['Prec'].values + simseries['IRA'].values + simseries['IRI'].values
    columns = simseries.columns
    diags = dict()
    diags['Parameter'] = parameters
    for v in variables:
        # get vector
        lcl_v = simseries[v].values
        if v in set(stocks) or v in set(vnonwater):
            lcl_sum = np.nan
        else:
            lcl_sum = np.sum(lcl_v)
        lcl_mean = np.mean(lcl_v)
        # SD
        if lcl_sum == 0.0:
            lcl_sd = 0.0
        else:
            lcl_sd = np.std(lcl_v)
        # events
        lcl_n = np.sum(1 * lcl_v > 0.0)
        # events stats
        if lcl_sum == 0:
            lcl_mean_n = 0
            lcl_sd_n = 0
        elif lcl_n == len(lcl_v):
            lcl_mean_n = lcl_mean
            lcl_sd_n = lcl_sd
        else:
            lcl_mask = 1 * (lcl_v > 0)
            lcl_v_clear = flatten_clear(lcl_v, lcl_mask)
            lcl_mean_n = np.mean(lcl_v_clear)
            lcl_sd_n = np.std(lcl_v_clear)
        # Coefs
        if v in set(stocks) or v in set(vnonwater):
            lcl_cprec = np.nan
            lcl_cinput = np.nan
        else:
            lcl_cprec = 100 * lcl_sum / np.sum(prec)
            lcl_cinput = 100 * lcl_sum / np.sum(tinput)
        diags[v] = (lcl_sum, lcl_mean, lcl_sd, lcl_n, lcl_mean_n, lcl_sd_n, lcl_cprec, lcl_cinput)
    diag_df = pd.DataFrame(diags)
    #print(diag_df.to_string())
    return diag_df


def map_back(zmatrix, a1, a2, bins1, bins2):
    """
    Map back function using a Z-map
    :param zmatrix: 2d numpy array of z matrix of values
    :param a1: 2d numpy array reference array of rows (ex: TWI)
    :param a2: 2d numpy array reference array of columns  (ex: SHRU)
    :param bins1: 1d numpy array of array 1 histogram bins
    :param bins2: 1d numpy array of array 2 histogram bins
    :return: 2d numpy array
    """
    # initiate map array
    map = np.zeros(shape=np.shape(a1))
    for i in range(len(zmatrix)):
        # first iteration on bins
        if i == 0:
            # define a boolean mask
            mask1 = (a1 <= bins1[i])
        else:
            mask1 = (a1 > bins1[i - 1]) * (a1 <= bins1[i])
        for j in range(len(zmatrix[i])):
            if j == 0:
                mask2 = (a2 <= bins2[j])
            else:
                mask2 = (a2 > bins2[j - 1]) * (a2 <= bins2[j])
            lclmap = mask1 * mask2 * zmatrix[i][j]
            map = map + lclmap
    return map

#
#
#
# PET model silent functions
def pet_gsc():
    """
    PET model Solar Constant
    :return: float solar constant in [W/m2]
    """
    return 1360.0  # W/m2


def pet_g(day):
    """
    PET model Solar Radiation as a function of Julian Day
    :param day: int - julian day
    :return: solar radiation (float or array) in [W/m2]
    """
    return pet_gsc() * (1 + 0.033 * np.cos(2 * np.pi * day / 365))


def pet_declination(day):
    """
    PET model - Earth declination angle
    :param day: int - julian day
    :return: Earth declination angle in [radians]
    """
    return (2 * np.pi * 23.45 / 360) * np.sin(2 * np.pi * (284 + day) / 365)


def pet_zenital_angle(day, latitude, hour):
    """
    Zenital incidence angle in radians on a horizontal plane
    :param day: int julian day
    :param latitude: latitude angle in [radians]
    :param hour: hour angle in [radians]
    :return: zenital incidence angle in radians
    """
    dcl = pet_declination(day=day)
    return np.arccos((np.cos(latitude) * np.cos(dcl) * np.cos(hour)) + (np.sin(latitude) * np.sin(dcl)))


def pet_altitude_angle(day, latitude, hour):
    """
    Altitude incidence angle in radians on a horizontal plane
    :param day: int julian day
    :param latitude: latitude angle in [radians]
    :param hour: hour angle in [radians]
    :return: zenital incidence angle in radians
    """
    zenit = pet_zenital_angle(day=day, latitude=latitude, hour=hour)
    return (np.pi / 2) - zenit


def pet_hss(declination, latitude):
    """
    PET model - Sun Set Hour angle in radians
    :param declination: declination angle in [radians]
    :param latitude: latitude angle in [radians]
    :return: Sun Set Hour angle in [radians]
    """
    return np.arccos(-np.tan(latitude) * np.tan(declination))


def pet_daily_hetrad(day, latitude):
    """
    PET model - Daily integration of the instant Horizontal Extraterrestrial Radiation Equations
    :param day: int - Julian day
    :param latitude: float - latitude in [radians]
    :return: Horizontal Daily Extraterrestrial Radiation in [MJ/(d * m2)]
    """
    g = pet_g(day)  # Get instant solar radiation in W/m2
    declination = pet_declination(day)  # Get declination in radians
    hss = pet_hss(declination=declination, latitude=latitude)  # get sun set hour angle in radians
    het = (24 * 3600 / np.pi) * g * ((np.cos(latitude) * np.cos(declination) * np.sin(hss))
                                     + (hss * np.sin(latitude) * np.sin(declination)))  # J/(d * m2)
    return het / (1000000)  # MJ/(d * m2)


def pet_latent_heat_flux():
    """
    PET model - Latent Heat Flux of water in MJ/kg
    :return: float -  Latent Heat Flux of water in MJ/kg
    """
    return 2.45  # MJ/kg


def pet_water_spmass():
    """
    PET model - Water specific mass in kg/m3
    :return: float - Water specific mass in kg/m3
    """
    return 1000.0  # kg/m3


def pet_oudin(temperature, day, latitude, k1=100, k2=5):
    """
    PET Oudin Model - Radiation and Temperature based PET model of  Ref: Oudin et al (2005b)
    :param temperature: float or array of daily average temperature in [C]
    :param day: int or array of Julian day
    :param latitude: latitude angle in [radians]
    :param k1: Scalar parameter in [C * m/mm]
    :param k2: Minimum air temperature [C]
    :return: Potential Evapotranspiration in [mm/d]
    """
    het = pet_daily_hetrad(day, latitude)
    pet = (1000 * het / (pet_latent_heat_flux() * pet_water_spmass() * k1)) * (temperature + k2) * ((temperature + k2) > 0) * 1.0
    return pet

#
#
#
# topmodel functions
def topmodel_d0(qt0, qo, m):
    """
    TOPMODEL Deficit as a function of baseflow (Beven and Kirkby, 1979)
    :param qt0: flow at t=0 in mm/d
    :param qo: max baseflow when d=0 in mm/d
    :param m: decay parameter in mm
    :return: basin global deficit in mm
    """
    return - m * np.log(qt0/qo)


def topmodel_qb(d, qo, m):
    """
    TOPMODEL baseflow as a function of global deficit (Beven and Kirkby, 1979)
    :param d: basin global deficit in mm
    :param qo: max baseflow when d=0 in mm/d
    :param m: decay parameter in mm
    :return: baseflow in mm/d
    """
    return qo * np.exp(-d/m)


def topmodel_qv(d, unz, ksat):
    """
    Global and local vertical recharge rate function (Beven and Woods.. )

    :param d: deficit in mm
    :param unz: usaturated zone water in mm
    :param ksat: velocity parameter in mm/d
    :return: vertical recharge rate
    """
    return ksat * (unz / (d + 0.001)) * (d > 0)  # + 0.001 to avoid division by zero


def topmodel_di(d, twi, m, lamb):
    """
    local deficit di (Beven and Kirkby, 1979)
    :param d: global deficit float
    :param twi: TWI 1d bins array
    :param m: float
    :param lamb: average value of TWIi
    :return: 1d bins array of local deficit
    """
    lcl_di = d + m * (lamb - twi)
    #avg_di = avg_2d(lcl_di, aoi)
    mask = 1.0 * (lcl_di > 0)
    lcl_di2 =  np.abs(lcl_di * mask)  # set negative deficit to zero
    return lcl_di2


def topmodel_vsai(di):
    """
    Variable Source Area
    :param di: float or nd array of local deficit in mm
    :return: float or nd array of pseudo boolean of saturated areas
    """
    return ((di == 0) * 1)
#
#
# main functions
def simulation(series, shruparam, canopy, twibins, countmatrix, lamb, qt0, m, qo, cpmax, sfmax, erz, ksat, c,
               lat, k, n, area, basinshadow, mapback=False, mapvar='all', mapdates='all', qobs=False, tui=False):
    """

    Plans3 model simulation routine

    :param series: pandas dataframe of input time series.
    :param shruparam: pandas dataframe of SHRU hard parameters (a priori parameter factors)
    :param canopy: pandas dataframe of SHRU canopy seasonal factor pattern time series.
    :param twibins: 1d numpy array of TWI bins
    :param countmatrix: pandas dataframe
    :param lamb: float model parameter
    :param qt0: float model parameter
    :param m: float model parameter
    :param qo: float model parameter
    :param cpmax: float model parameter
    :param sfmax: float model parameter
    :param erz: float model parameter
    :param ksat: float model parameter
    :param c: float model parameter
    :param lat: float model parameter
    :param k: float model parameter
    :param n: float model parameter
    :param area: float model boundary condition
    :param basinshadow: pandas dataframe of basin shadow matrix
    :param mapback: boolean to map back variables
    :param mapvar: string of variables to map. Pass concatenated by '-'. Ex: 'ET-TF-Inf'.
    Options: 'Prec-Temp-IRA-IRI-PET-D-Cpy-TF-Sfs-R-RSE-RIE-RC-Inf-Unz-Qv-Evc-Evs-Tpun-Tpgw-ET-VSA' or 'all'
    :param mapdates: string of dates to map. Pass concatenated by ' & '. Ex: '2011-21-01 & 21-22-01'
    :param qobs: boolean
    :param tui: boolean for TUI displays
    :return: output dictionary
    """
    # extract data input
    ts_prec = series['Prec'].values
    ts_temp = series['Temp'].values
    ts_ira = series['IRA'].values
    ts_iri = series['IRI'].values
    size = len(ts_prec)
    #
    # compute PET
    days = series['Date'].dt.dayofyear
    ts_days = days.values
    lat = lat * np.pi / 180  # convet lat from degrees to radians
    ts_pet = pet_oudin(temperature=ts_temp, day=ts_days, latitude=lat, k1=c)  # Oudin model
    #
    # get zmap shape
    shape = np.shape(countmatrix)
    rows = shape[0]
    cols = shape[1]
    #
    # get local Lambda (TWI)
    lamb_i = np.reshape(twibins, (rows, 1)) * np.ones(shape=shape, dtype='float32')
    #
    #
    # get local surface parameters
    # local full cpmax
    cpmax_i = cpmax * shruparam['f_Canopy'].values * np.ones(shape=shape, dtype='float32')
    cpmax_season_i = np.ones(shape=shape)
    # get canopy seasonal factor timeseries:
    ts_cp_season = np.array(canopy.values[:, 2:], dtype='float32')  # skip date and month fields
    # local sfmax
    sfmax_i = sfmax * shruparam['f_Surface'].values * np.ones(shape=shape, dtype='float32')
    # local erz
    erz_i = erz * shruparam['f_EfRootZone'].values * np.ones(shape=shape, dtype='float32')
    # local ksat
    ksat_i = ksat * shruparam['f_Ksat'].values * np.ones(shape=shape, dtype='float32')
    # local infcap
    infcap_i = ksat * shruparam['f_Inf'].values * np.ones(shape=shape, dtype='float32')
    # local ira factor
    fira_i = shruparam['f_IRA'].values * np.ones(shape=shape, dtype='float32')
    # local iri factor
    firi_i = shruparam['f_IRI'].values * np.ones(shape=shape, dtype='float32')
    #
    # set global deficit initial conditions
    d0 = topmodel_d0(qt0=qt0, qo=qo, m=m)
    #
    #
    # set local variables initial conditions:
    # local stock variables:
    d_i = topmodel_di(d=d0, twi=lamb_i, m=m, lamb=lamb)  # use topmodel equation to derive initial d
    cpy_i = np.zeros(shape=shape)
    sfs_i = np.zeros(shape=shape)
    unz_i = np.zeros(shape=shape)
    vsa_i = topmodel_vsai(di=d_i)  # variable source area map
    temp_i = ts_temp[0] * np.ones(shape=shape)
    # local flow variables:
    prec_i = ts_prec[0] * np.ones(shape=shape)
    ira_i = ts_ira[0] * fira_i
    iri_i = ts_iri[0] * firi_i
    cpyin_i = np.zeros(shape=shape)
    sfsin_i = np.zeros(shape=shape)
    pet_i = ts_pet[0] * np.ones(shape=shape)
    tf_i = np.zeros(shape=shape)
    r_i = np.zeros(shape=shape)
    rse_i = np.zeros(shape=shape)
    rie_i = np.zeros(shape=shape)
    rc_i = np.zeros(shape=shape)
    inf_i = np.zeros(shape=shape)
    qv_i = np.zeros(shape=shape)
    evc_i = np.zeros(shape=shape)
    evs_i = np.zeros(shape=shape)
    tpun_i = np.zeros(shape=shape)
    tpgw_i = np.zeros(shape=shape)
    et_i = np.zeros(shape=shape)
    #
    # set global time series arrays:
    # stock variables time series:
    ts_cpy = np.zeros(shape=size, dtype='float32')
    ts_sfs = np.zeros(shape=size, dtype='float32')
    ts_unz = np.zeros(shape=size, dtype='float32')
    ts_d = np.zeros(shape=size, dtype='float32')
    ts_d[0] = d0
    # flow variables time series:
    ts_iri_out = np.zeros(shape=size, dtype='float32')
    ts_ira_out = np.zeros(shape=size, dtype='float32')
    ts_tf = np.zeros(shape=size, dtype='float32')
    ts_inf = np.zeros(shape=size, dtype='float32')
    ts_r = np.zeros(shape=size, dtype='float32')
    ts_rse = np.zeros(shape=size, dtype='float32')
    ts_rie = np.zeros(shape=size, dtype='float32')
    ts_rc = np.zeros(shape=size, dtype='float32')
    ts_qv = np.zeros(shape=size, dtype='float32')
    ts_evc = np.zeros(shape=size, dtype='float32')
    ts_evs = np.zeros(shape=size, dtype='float32')
    ts_tpun = np.zeros(shape=size, dtype='float32')
    ts_tpgw = np.zeros(shape=size, dtype='float32')
    ts_et = np.zeros(shape=size, dtype='float32')
    ts_qb = np.zeros(shape=size, dtype='float32')
    ts_qb[0] = qt0
    ts_qs = np.zeros(shape=size, dtype='float32')
    ts_q = np.zeros(shape=size, dtype='float32')
    ts_q[0] = qt0
    ts_vsa = np.zeros(shape=size, dtype='float32')
    ts_vsa[0] = np.sum(vsa_i * basinshadow) / np.sum(basinshadow)
    #
    #
    # Z-Map Trace setup
    if mapback:
        if mapvar == 'all':
            from backend import get_all_lclvars
            mapvar = get_all_lclvars()
        mapvar_lst = mapvar.split('-')
        #
        #
        # map dates protocol
        mapped_dates = list()  # list of actual mapped files dates
        if mapdates == 'all':
            mapsize = size
            mapped_dates = list(series['Date'].astype('str')) # old code: list(pd.to_datetime(series['Date'], format='%y-%m-%d'))
        else:
            # extract map dates to array
            mapid = 0
            #
            # prep map dates
            mapdates_df = pd.DataFrame({'Date': mapdates.split('&')})
            mapdates_df['DateStr'] = mapdates_df['Date'].str.strip()
            mapdates_df['Date'] = pd.to_datetime(mapdates_df['Date'])
            lookup_dates = mapdates_df['Date'].values  # it is coming as datetime!
            #
            # get series dates
            dates_series = series['Date'].values
            #
            # built the array of timestep index of dates to map
            map_timesteps = list()
            for i in range(len(lookup_dates)):
                lcl_series_df = series.query('Date == "{}"'.format(lookup_dates[i]))
                if len(lcl_series_df) == 1:
                    lcl_step = lcl_series_df.index[0]
                    map_timesteps.append(lcl_step)  # append index to list
                    mapped_dates.append(mapdates_df['DateStr'].values[i])
            mapsize = len(map_timesteps)
        #
        # load to dict object a time series of empty zmaps for each variable
        mapkeys_dct = dict()
        for e in mapvar_lst:
            mapkeys_dct[e] = np.zeros(shape=(mapsize, rows, cols), dtype='float32')
        # store initial zmaps in dict
        ext = 1 * (countmatrix > 0)  # filter by map extension
        mapback_dct = {'TF': tf_i * ext,
                       'Qv': qv_i * ext,
                       'R': r_i * ext,
                       'RSE': rse_i * ext,
                       'RIE': rie_i * ext,
                       'RC': rc_i * ext,
                       'ET': et_i * ext,
                       'Cpy': cpy_i * ext,
                       'Sfs': sfs_i * ext,
                       'Inf': inf_i * ext,
                       'Tpun': tpun_i * ext,
                       'Evc': evc_i * ext,
                       'Tpgw': tpgw_i * ext,
                       'Evs': evs_i * ext,
                       'VSA': vsa_i * ext,
                       'Prec': prec_i * ext,
                       'Temp': temp_i * ext,
                       'IRA': ira_i * ext,
                       'IRI': iri_i * ext,
                       'PET': pet_i * ext,
                       'D': d_i * ext,
                       'Unz': unz_i * ext}
        # append it to map the first record in the map time series
        for e in mapvar_lst:
            mapkeys_dct[e][0] = mapback_dct[e]
    #
    #
    #
    #
    # ***** ESMA loop by finite differences *****
    #
    #
    for t in range(1, size):
        #
        # ****** UPDATE local water balance ******
        #
        # update Canopy water stock
        cpy_i = cpy_i + cpyin_i - (tf_i + evc_i)
        ts_cpy[t] = avg_2d(var2d=cpy_i, weight=basinshadow)  # compute average
        #
        # update Surface water stock
        sfs_i = sfs_i + (sfsin_i - r_i) - (inf_i + evs_i)
        ts_sfs[t] = avg_2d(var2d=sfs_i, weight=basinshadow)  # compute average
        #
        # update Unsaturated water stock
        unz_i = unz_i + inf_i - (qv_i + tpun_i)
        ts_unz[t] = avg_2d(var2d=unz_i, weight=basinshadow)  # compute average
        #
        #
        # ****** COMPUTE Flows for usage in next time step ******
        #
        #
        # --- Canopy water balance
        #
        #
        # update PREC:
        prec_i = ts_prec[t] * np.ones(shape=shape)
        #
        # update IRA:
        ira_i = ts_ira[t] * fira_i
        ts_ira_out[t] = avg_2d(ira_i, basinshadow) # get effective ira
        #
        # canopy total input:
        cpyin_i = prec_i + ira_i
        #
        # compute throughfall (tf):
        cpmax_season_i = cpmax_i * ts_cp_season[t]  # local seasonal canopy capacity
        tf_i = ((cpyin_i - (cpmax_season_i - cpy_i)) * (cpyin_i > (cpmax_season_i - cpy_i)))
        ts_tf[t] = avg_2d(var2d=tf_i, weight=basinshadow)
        #
        # compute current evaporation from canopy (evc):
        #
        # update PET (0) - load constant PET layer
        pet_i = ts_pet[t] * np.ones(shape=shape)
        petfull_i = ts_pet[t] * np.ones(shape=shape)  # reserve full PET for mapping
        #
        # compute interceptation (icp):
        icp_i = cpy_i + cpyin_i - tf_i
        #
        # evaporation from canopy (evc):
        evc_i = (pet_i * (icp_i > pet_i)) + (icp_i * (icp_i <= pet_i))
        ts_evc[t] = avg_2d(var2d=evc_i, weight=basinshadow)  # compute average in the basin
        #
        # update PET (1) - discount evaporation from canopy (evc)
        pet_i = pet_i - evc_i
        #
        #
        #
        # --- Saturated Root Zone water balance - transpiration part
        #
        #
        # potential transpiration from groundwter (tpgw):
        p_tpgw_i = (erz_i - d_i) * ((erz_i - d_i) > 0)
        #
        # compute transpiration from groundwter (tpgw):
        tpgw_i = (pet_i * (p_tpgw_i >= pet_i)) + (p_tpgw_i * (p_tpgw_i < pet_i))
        ts_tpgw[t] = avg_2d(var2d=tpgw_i, weight=basinshadow)  # compute average in the basin
        #
        # update pet (2) - discount transpiration from groundwter (tpgw):
        pet_i = pet_i - tpgw_i
        #
        #
        #
        # --- Surface water balance (runoff and infiltration)
        #
        #
        # update irrigation by inundation (IRI):
        iri_i = ts_iri[t] * firi_i
        ts_iri_out[t] = avg_2d(iri_i, basinshadow) # get effective IRI
        #
        # compute surface total input:
        sfsin_i = tf_i + iri_i
        #
        # compute current runoff (excess water on surface):
        r_i = ((sfs_i + sfsin_i) - sfmax_i) * ((sfs_i + sfsin_i) > sfmax_i)
        #
        # separate runoff components:
        rse_i = r_i * vsa_i  # Saturation excess runoff - Dunnean runoff
        rie_i = r_i * (vsa_i == 0.0) # Infiltration excess runoff - Hortonian runoff
        rc_i = 100 * (prec_i > 0) * (r_i / ((prec_i == 0) + prec_i)) # runoff coeficient (in %)
        #
        # compute global runoff components:
        ts_r[t] = avg_2d(var2d=r_i, weight=basinshadow)  # compute average in the basin
        ts_rse[t] = avg_2d(var2d=rse_i, weight=basinshadow)
        ts_rie[t] = avg_2d(var2d=rie_i, weight=basinshadow)
        ts_rc[t] = avg_2d(var2d=rc_i, weight=basinshadow)
        '''
        # idea for multiple basin runoff routing -- do this off the ESMA loop
        for b in range(len(basins_list)):
            ts_r[b][t] = avg_2d(var2d=r_i, weight=basins_list[b])  # time series of Runoff as 2d array
        '''
        #
        # --- Infiltration
        #
        # potential infiltration rate allowed by the surface water content:
        p_sfs_inf_i = (sfs_i * (sfs_i < infcap_i)) + (infcap_i * (sfs_i >= infcap_i))
        #
        # potential infiltration allowed by the usaturated zone water content:
        p_unz_inf_i = (d_i - unz_i) * ((d_i - unz_i) > 0)
        #
        # compute surface water depletion by infiltration (inf):
        inf_i = (p_sfs_inf_i * (p_sfs_inf_i < p_unz_inf_i)) + (p_unz_inf_i * (p_sfs_inf_i >= p_unz_inf_i))
        ts_inf[t] = avg_2d(var2d=inf_i, weight=basinshadow)  # compute average in the basin
        #
        #
        #
        # --- Unsaturated Zone water balance
        #
        #
        # potential qv recharge rate without PET:
        p_qv_i = topmodel_qv(d=d_i, unz=unz_i, ksat=ksat_i)
        #
        # actual qv recharge rate with PET (proportional to ratio):
        qv_i = unz_i * (p_qv_i/ (pet_i + p_qv_i + 1)) * ((pet_i + p_qv_i) > 0)  # + 1 to avoid division by zero
        ts_qv[t] = avg_2d(var2d=qv_i, weight=basinshadow)  # compute average in the basin
        #
        # compute potential tpun transpiration rate (gated by erz_i and proportional to ratio):
        p_tpun_i = unz_i * (unz_i <= erz_i) + erz_i * (unz_i > erz_i)
        p_tpun_i = p_tpun_i * (pet_i / (pet_i + p_qv_i + 1)) * ((pet_i + p_qv_i) > 0)  # + 1 to avoid division by zero
        #
        # transpiration from the unsaturated zone
        tpun_i = (pet_i * (p_tpun_i >= pet_i)) + (p_tpun_i * (p_tpun_i < pet_i))
        ts_tpun[t] = avg_2d(var2d=tpun_i, weight=basinshadow)  # compute average in the basin
        #
        # update PET (3) - discount transpiration from the unsaturated zone (tpun)
        pet_i = pet_i - tpun_i
        #
        #
        # --- Surface (evaporation)
        #
        #
        # potential evaporation allowed by surface water stock:
        p_sfs_evs_i = sfs_i - inf_i
        # evaporation from surface:
        evs_i = (pet_i * (p_sfs_evs_i >= pet_i)) + (p_sfs_evs_i * (p_sfs_evs_i < pet_i))
        ts_evs[t] = avg_2d(var2d=evs_i, weight=basinshadow)  # compute average in the basin
        #
        #
        # --- ET
        #
        # compute ET
        et_i = evc_i + evs_i + tpun_i + tpgw_i
        ts_et[t] = avg_2d(var2d=et_i, weight=basinshadow)  # compute average in the basin
        #
        #
        #
        #
        # ****** UPDATE Global Water balance ******
        #
        # global water balance
        ts_d[t] = ts_d[t - 1] + ts_qb[t - 1] - ts_qv[t - 1] + ts_tpgw[t - 1]
        #
        # compute Qb - Baseflow
        ts_qb[t] = topmodel_qb(d=ts_d[t], qo=qo, m=m)

        #
        # Update Di
        d_i = topmodel_di(d=ts_d[t], twi=lamb_i, m=m, lamb=lamb)
        #
        # compute VSA
        vsa_i = topmodel_vsai(di=d_i)
        ts_vsa[t] = 100 * np.sum(vsa_i * basinshadow) / np.sum(basinshadow) # in %
        #
        # get temperature map:
        temp_i = ts_temp[t] * np.ones(shape=shape)
        #
        # trace section
        if mapback:
            # store timestep maps in dict
            mapback_dct = {'TF': tf_i * ext,
                           'Qv': qv_i * ext,
                           'R': r_i * ext,
                           'RSE': rse_i * ext,
                           'RIE': rie_i * ext,
                           'RC': rc_i * ext,
                           'ET': et_i * ext,
                           'Cpy': cpy_i * ext,
                           'Sfs': sfs_i * ext,
                           'Inf': inf_i * ext,
                           'Tpun': tpun_i * ext,
                           'Evc': evc_i * ext,
                           'Tpgw': tpgw_i * ext,
                           'Evs': evs_i * ext,
                           'VSA': vsa_i * ext,
                           'Prec': prec_i * ext,
                           'Temp': temp_i * ext,
                           'IRA': ira_i * ext,
                           'IRI': iri_i * ext,
                           'PET': petfull_i * ext,
                           'D': d_i * ext,
                           'Unz': unz_i * ext}
            if mapdates == 'all':
                # append it to map
                for e in mapvar_lst:
                    mapkeys_dct[e][t] = mapback_dct[e]
            else:
                # check if the current time step is a mapped time step
                if t in set(map_timesteps):
                    # append it to map
                    for e in mapvar_lst:
                        mapkeys_dct[e][mapid] = mapback_dct[e]
                    mapid = mapid + 1  # increment mapid
    #
    #
    # RUNOFF ROUTING by Nash Cascade of linear reservoirs
    if n < 1:
        n = 1.0
    ts_qs = nash_cascade(ts_r, k=k, n=n)
    #
    #
    # Compute full discharge Q = Qb + Qs
    ts_q = ts_qb + ts_qs
    ts_flow = convert_sq2q(sq=ts_q, area=area)
    #
    #
    # export data
    exp_df = pd.DataFrame({'Date':series['Date'].values,
                           'Prec':series['Prec'].values,
                           'Temp':series['Temp'].values,
                           'IRA': ts_ira,
                           'IRI': ts_iri,
                           'EIRA': ts_ira_out,
                           'EIRI': ts_iri_out,
                           'PET':ts_pet,
                           'D':ts_d,
                           'Cpy':ts_cpy,
                           'TF':ts_tf,
                           'Sfs':ts_sfs,
                           'R':ts_r,
                           'RSE': ts_rse,
                           'RIE': ts_rie,
                           'RC': ts_rc,
                           'Inf':ts_inf,
                           'Unz':ts_unz,
                           'Qv':ts_qv,
                           'Evc':ts_evc,
                           'Evs':ts_evs,
                           'Tpun':ts_tpun,
                           'Tpgw':ts_tpgw,
                           'ET':ts_et,
                           'Qb':ts_qb,
                           'Qs':ts_qs,
                           'Q':ts_q,
                           'Flow':ts_flow,
                           'VSA':ts_vsa})
    if qobs:
        exp_df['Qobs'] = series['Q'].values
        exp_df['Fobs'] = convert_sq2q(sq=series['Q'].values, area=area)
    #
    #
    out_dct = {'Series':exp_df}
    if mapback:
        out_dct['Maps'] = mapkeys_dct
        out_dct['MappedDates'] = mapped_dates
    return out_dct


def calibration(series, shruparam, canopy, twibins, countmatrix, qt0, lat, area, basinshadow,
                p_ranges, etpatdates, etpatzmaps,
                tui=True,
                normalize=False,
                grid=1000,
                generations=10,
                popsize=20,
                offsfrac=2,
                mutrate=0.5,
                puremutrate=0.5,
                cutfrac=0.4,
                tracefrac=1,
                tracepop=True,
                likelihood='NSE',
                nodata=-1,
                etpat=True):
    """

    Calibration procedure

    :param series: pandas dataframe
    :param shruparam: pandas dataframe
    :param canopy: pandas dataframe
    :param twibins: 1d numpy array
    :param countmatrix: 2d numpy array
    :param qt0: boolean
    :param lat: float
    :param area: float
    :param basinshadow: 2d numpy array
    :param p_ranges: dictionary of iterables of parameter ranges
    :param etpatdates: string code of dates
    :param etpatzmaps: 3d numpy array of zmaps
    :param tui: boolean for TUI print
    :param normalize: boolean to normalize values of simulated ET (0 and 1)
    :param grid: int for sampling grid
    :param generations: int number of generations
    :param popsize: int number of population size
    :param offsfrac: float fraction of offspring size relative to population
    :param mutrate: float probability of mutation during crossover
    :param puremutrate: float probability of pure mutation
    :param cutfrac: float fraction for cutting genes during crossover
    :param tracefrac: float fraction of parents to traceback
    :param tracepop: boolean to trace all population
    :param likelihood: string code of likelihood
    :param nodata: float of no data
    :return: dictionary of MLM set, traced parents and (optional) population dataframes
    """
    from evolution import generate_population, generate_offspring, recruitment
    import analyst
    from geo import fuzzy_transition
    from sys import getsizeof
    from datetime import datetime

    def wrapper(traced, lowerb, ranges, gridsize, etpat=True):
        """
        Wrapper function to return list of dataframes with expressed genes
        :param traced: list of dictionaries with encoded generations
        :param lowerb: numpy array of lowerbound values
        :param ranges: numpy array of range values
        :return: list of dataframes of each generation
        """
        df_lst = list()
        #
        # generation loop:
        for g in range(len(traced)):
            m_lst = list()
            lamb_lst = list()
            qo_lst = list()
            cpmax_lst = list()
            sfmax_lst = list()
            erz_lst = list()
            ksat_lst = list()
            c_lst = list()
            k_lst = list()
            n_lst = list()
            #
            # individual loop to extract :
            for i in range(len(traced[g]['DNAs'])):
                lcl_dna = traced[g]['DNAs'][i]
                lcl_pset = express_parameter_set(gene=lcl_dna[0], lowerb=lowerb, ranges=ranges, gridsize=gridsize)
                m_lst.append(lcl_pset[0])
                lamb_lst.append(lcl_pset[1])
                qo_lst.append(lcl_pset[2])
                cpmax_lst.append(lcl_pset[3])
                sfmax_lst.append(lcl_pset[4])
                erz_lst.append(lcl_pset[5])
                ksat_lst.append(lcl_pset[6])
                c_lst.append(lcl_pset[7])
                k_lst.append(lcl_pset[8])
                n_lst.append(lcl_pset[9])
            #
            gen = np.ones(len(m_lst)) * g
            #
            # built dict
            if etpat:
                lcl_dct = {'Gen': gen,
                           'Id':traced[g]['Ids'],
                           'SetIds': traced[g]['SetIds'],
                           'L':traced[g]['L'],
                           'L_Q':traced[g]['L_Q'],
                           'L_ET':traced[g]['L_ET'],
                           'm': m_lst,
                           'lamb':lamb_lst,
                           'qo': qo_lst,
                           'cpmax': cpmax_lst,
                           'sfmax': sfmax_lst,
                           'erz': erz_lst,
                           'ksat': ksat_lst,
                           'c': c_lst,
                           'k': k_lst,
                           'n': n_lst}
            else:
                lcl_dct = {'Gen': gen,
                           'Id': traced[g]['Ids'],
                           'SetIds': traced[g]['SetIds'],
                           'L': traced[g]['L'],
                           'm': m_lst,
                           'lamb': lamb_lst,
                           'qo': qo_lst,
                           'cpmax': cpmax_lst,
                           'sfmax': sfmax_lst,
                           'erz': erz_lst,
                           'ksat': ksat_lst,
                           'c': c_lst,
                           'k': k_lst,
                           'n': n_lst}
            # append Q metadata
            q_meta = ('NSE',
                         'NSElog',
                         'KGE',
                         'KGElog',
                         'RMSE',
                         'RMSElog',
                         'PBias',
                         'RMSE_CFC',
                         'RMSElog_CFC',
                         'Q_sum',
                         'Q_mean',
                         'Q_sd',
                         'Qb_sum',
                         'Qb_mean',
                         'Qb_sd',
                         'Q_C',
                         'Qb_C'
                         )
            for m in q_meta:
                lcl_dct[m] = traced[g][m]
            # append ET metadata
            if etpat:
                et_meta = ('ET_MSE_mean',
                              'ET_MSE_min',
                              'ET_MSE_max',
                              'ET_RMSE_mean',
                              'ET_RMSE_min',
                              'ET_RMSE_max',
                              'ET_W-MSE_mean',
                              'ET_W-MSE_min',
                              'ET_W-MSE_max',
                              'ET_W-RMSE_mean',
                              'ET_W-RMSE_min',
                              'ET_W-RMSE_max',
                              'ET_NSE_mean',
                              'ET_NSE_min',
                              'ET_NSE_max',
                              'ET_KGE_mean',
                              'ET_KGE_min',
                              'ET_KGE_max',
                              'ET_R_mean',
                              'ET_R_min',
                              'ET_R_max',
                              'ET_MeanObs_mean',
                              'ET_MeanObs_min',
                              'ET_MeanObs_max',
                              'ET_MeanSim_mean',
                              'ET_MeanSim_min',
                              'ET_MeanSim_max',
                              'ET_MeanErr_mean',
                              'ET_MeanErr_min',
                              'ET_MeanErr_max')
                for m in et_meta:
                    lcl_dct[m] = traced[g][m]
            #
            # built export dataframe
            lcl_df = pd.DataFrame(lcl_dct)
            #
            #
            if g == 0:
                out_df = lcl_df
            else:
                out_df = out_df.append(lcl_df, ignore_index=True)
        # compute Qb_R
        out_df['Qb_R'] = out_df['Qb_sum'].values / out_df['Q_sum'].values
        # return full dataframe
        return out_df

    def express_parameter_set(gene, lowerb, ranges, gridsize):
        """
        Expression of parameter set
        :param gene: gene tuple ex: (4, 60, 20, 123, 12, 11, 400, 303)
        :param lowerb: numpy array of lowerbound values ex: (1, 2, 0.2, 2, 1, 0.5, 0.5, 0.6)
        :param ranges: numpy array of range values ex: (20, 45, 10, 32, 45, 50, 100, 30)
        :return: numpy array of parameter set
        """
        return (np.array(gene) * ranges / gridsize) + lowerb
    #
    #
    # data setup
    #
    # extract observed data
    sobs = series['Q'].values
    #
    # get log10 of flow for calibration metrics
    loglim = 0.000001
    #
    # get OBS etpat zmaps
    if etpat:
        sobs_etpat = np.array(etpatzmaps)
        #
        # get weights
        w_flow = 1 #len(sobs)  # weight of flow
        w_etpat = 1 #len(sobs_etpat)  # weight of et
        if tui:
            print('ETPat series shape: {}'.format(np.shape(sobs_etpat)))
            print('Flow weight: {}'.format(w_flow))
            print('EtPat weight: {}'.format(w_etpat))
    #
    #
    # bounds setup
    params = ('m', 'lamb', 'qo', 'cpmax', 'sfmax', 'erz', 'ksat', 'c', 'k', 'n')
    lower_bound = list()
    upper_bound = list()
    for p in params:
        lower_bound.append(np.min(p_ranges[p]))
        upper_bound.append(np.max(p_ranges[p]))
    lower_bound = np.array(lower_bound)
    upper_bound = np.array(upper_bound)
    ranges = upper_bound - lower_bound
    #
    #
    # Evolution setup
    #
    # 1) deploy random state
    seed = int(str(datetime.now())[-6:])
    np.random.seed(seed)
    if tui:
        print('Random Seed: {}'.format(seed))
    # get integer nucleotides
    nucleotides = tuple(np.arange(0, grid + 1))
    # generate initial population
    parents = generate_population(nucleotides=(nucleotides,),
                                  genesizes=(10,),
                                  popsize=popsize)
    offsize = int(offsfrac * len(parents))
    # list to append best solutions
    trace = list()
    if tracepop:
        trace_pop = list()
    #
    #
    # 2) generation loop:
    # run setup
    runsize = popsize + (generations - 1) * offsize
    if tui:
        print('Runsize = {}'.format(runsize))
    counter = 0
    pop_dct = dict()  # population dictionary
    for g in range(generations):
        if tui:
            print('\n\nGeneration {}\n'.format(g))
        #
        # 3) REPRODUCE the fitting population
        if g == 0:
            population = parents
        else:
            # get offstring
            population = generate_offspring(parents,
                                            offsize=offsize,
                                            nucleotides=(nucleotides,),
                                            mutrate=mutrate,
                                            puremutrate=puremutrate,
                                            cutfrac=cutfrac)
        #
        if tui:
            print('Population size = {}'.format(len(population)))
            print('Population: {} KB       '.format(getsizeof(population)))
            print('                   | Set  ', end='\t')
            if etpat:
                print('{:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} | {:>8} {:>8} {:>8}'.format('m',
                                                                                                               'lamb',
                                                                                                               'qo',
                                                                                                               'cpmax',
                                                                                                               'sfmax',
                                                                                                               'erz',
                                                                                                               'ksat',
                                                                                                               'c',
                                                                                                               'k',
                                                                                                               'n',
                                                                                                               'L',
                                                                                                               'L_Q',
                                                                                                               'L_ET'))
            else:
                print('{:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} | {:>8}'.format('m',
                                                                                                   'lamb',
                                                                                                   'qo',
                                                                                                   'cpmax',
                                                                                                   'sfmax',
                                                                                                   'erz',
                                                                                                   'ksat',
                                                                                                   'c',
                                                                                                   'k',
                                                                                                   'n',
                                                                                                   'L'))
        #
        # 4) FIT new population
        ids_lst = list()
        ids_set_lst = list()
        scores_lst = np.zeros(len(population))
        scores_flow_lst = np.zeros(len(population))
        scores_etpat_lst = np.zeros(len(population))
        #
        # metadata metric arrays
        meta_nse = np.zeros(len(population))
        meta_nselog = np.zeros(len(population))
        meta_kge = np.zeros(len(population))
        meta_kgelog = np.zeros(len(population))
        meta_rmse = np.zeros(len(population))
        meta_rmselog = np.zeros(len(population))
        meta_pbias = np.zeros(len(population))
        meta_rmse_cfc = np.zeros(len(population))
        meta_rmselog_cfc = np.zeros(len(population))
        #
        # metadata diags arrays
        meta_q_sum = np.zeros(len(population))
        meta_q_mean = np.zeros(len(population))
        meta_q_sd = np.zeros(len(population))
        meta_qb_sum = np.zeros(len(population))
        meta_qb_mean = np.zeros(len(population))
        meta_qb_sd = np.zeros(len(population))
        meta_c_qprec = np.zeros(len(population))
        meta_c_qbprec = np.zeros(len(population))
        #
        # metadata metric of ETpat
        if etpat:
            meta_mean_mse_etpat = np.zeros(len(population))
            meta_min_mse_etpat = np.zeros(len(population))
            meta_max_mse_etpat = np.zeros(len(population))
            #
            meta_mean_rmse_etpat = np.zeros(len(population))
            meta_min_rmse_etpat = np.zeros(len(population))
            meta_max_rmse_etpat = np.zeros(len(population))
            #
            meta_mean_wmse_etpat = np.zeros(len(population))
            meta_min_wmse_etpat = np.zeros(len(population))
            meta_max_wmse_etpat = np.zeros(len(population))
            #
            meta_mean_wrmse_etpat = np.zeros(len(population))
            meta_min_wrmse_etpat = np.zeros(len(population))
            meta_max_wrmse_etpat = np.zeros(len(population))
            #
            meta_mean_nse_etpat = np.zeros(len(population))
            meta_min_nse_etpat = np.zeros(len(population))
            meta_max_nse_etpat = np.zeros(len(population))
            #
            meta_mean_kge_etpat = np.zeros(len(population))
            meta_min_kge_etpat = np.zeros(len(population))
            meta_max_kge_etpat = np.zeros(len(population))
            #
            meta_mean_r_etpat = np.zeros(len(population))
            meta_min_r_etpat = np.zeros(len(population))
            meta_max_r_etpat = np.zeros(len(population))
            #
            meta_mean_mean_obs_etpat = np.zeros(len(population))
            meta_min_mean_obs_etpat = np.zeros(len(population))
            meta_max_mean_obs_etpat = np.zeros(len(population))
            #
            meta_mean_mean_sim_etpat = np.zeros(len(population))
            meta_min_mean_sim_etpat = np.zeros(len(population))
            meta_max_mean_sim_etpat = np.zeros(len(population))
            #
            meta_mean_mean_err = np.zeros(len(population))
            meta_min_mean_err = np.zeros(len(population))
            meta_max_mean_err = np.zeros(len(population))
        #
        #
        if tracepop:
            dnas_lst = list()
        #
        #
        # loop in individuals
        for i in range(len(population)):
            runstatus = 100 * counter / runsize
            counter = counter + 1
            #
            # get local score and id:
            lcl_dna = population[i]  # local dna
            lcl_set_id = str(lcl_dna[0]) # id is the string conversion of DNA
            #
            #
            # express parameter set
            pset = express_parameter_set(gene=lcl_dna[0],
                                         lowerb=lower_bound,
                                         ranges=ranges,
                                         gridsize=grid)
            #
            #
            # run model
            sim_dct = simulation(series=series,
                                 shruparam=shruparam,
                                 canopy=canopy,
                                 twibins=twibins,
                                 countmatrix=countmatrix,
                                 qt0=qt0,
                                 m=pset[0],
                                 lamb=pset[1],
                                 qo=pset[2],
                                 cpmax=pset[3],
                                 sfmax=pset[4],
                                 erz=pset[5],
                                 ksat=pset[6],
                                 c=pset[7],
                                 lat=lat,
                                 k=pset[8],
                                 n=pset[9],
                                 area=area,
                                 basinshadow=basinshadow,
                                 tui=False,
                                 qobs=True,
                                 mapback=etpat,
                                 mapvar='ET',
                                 mapdates=etpatdates)
            sim_df = sim_dct['Series']
            #
            # compute Flow sim data
            flow_analysis = analyst.flow(obs=sobs, sim=sim_df['Q'].values)
            #
            #
            # Selector to get fitness score for Flow:
            if likelihood == 'NSE':
                lcl_flow_score = flow_analysis['Metrics']['NSE']
            elif likelihood == 'NSElog':
                lcl_flow_score = flow_analysis['Metrics']['NSElog']
            elif likelihood == 'KGE':
                lcl_flow_score = flow_analysis['Metrics']['KGE']
            elif likelihood == 'KGElog':
                lcl_flow_score = flow_analysis['Metrics']['KGElog']
            elif likelihood == 'RMSE':
                lcl_flow_score = 1 - flow_analysis['Metrics']['NRMSE']
            elif likelihood == 'RMSElog':
                lcl_flow_score = 1 - flow_analysis['Metrics']['NRMSElog']
            elif likelihood == 'PBias':
                lcl_flow_score = 1 - (np.abs(flow_analysis['Metrics']['PBias']) / 100)
            elif likelihood == 'RMSE-CFC':
                lcl_flow_score = 1 - flow_analysis['Metrics']['NRMSE_CFC']
            elif likelihood == 'RMSElog-CFC':
                lcl_flow_score = 1 - flow_analysis['Metrics']['NRMSElog_CFC']
            elif likelihood == 'NSElog x KGElog':
                lcl_flow_score = flow_analysis['Metrics']['NSElog'] * flow_analysis['Metrics']['KGElog']
            elif likelihood == 'NSElog x RMSElog-CFC':
                lcl_flow_score = flow_analysis['Metrics']['NSElog'] * (1 - flow_analysis['Metrics']['NRMSElog_CFC'])
            elif likelihood == 'KGElog x RMSElog-CFC':
                lcl_flow_score = flow_analysis['Metrics']['KGElog'] * (1 - flow_analysis['Metrics']['NRMSElog_CFC'])
            else:
                lcl_flow_score = flow_analysis['Metrics']['NSE']
            #
            #
            if etpat:
                #
                # extract ET array:
                if normalize:
                    ssim_et = np.array(sim_dct['Maps']['ET'])
                    ssim_etpat = ssim_et * 0.0
                    for e in range(len(ssim_et)):
                        # linear normalization for each date
                        ssim_etpat[e] = fuzzy_transition(ssim_et[e], np.min(ssim_et[e]), np.max(ssim_et[e]), type='senoid')
                else:
                    ssim_etpat = np.array(sim_dct['Maps']['ET'])
                #
                # get analyst dict
                etpat_analysis = analyst.zmaps_series(obs=sobs_etpat,
                                                      sim=ssim_etpat,
                                                      count=1.0 *countmatrix,
                                                      nodata=nodata,
                                                      full_return=False)
                #
                # get the ETpat score (mean of signals KGE)
                lcl_etpat_score = etpat_analysis['Stats']['KGE_mean']
                #
                #
                #
                # COMPUTE GLOBAL SCORE:
                dy = (1 - lcl_flow_score)
                dx = (1 - lcl_etpat_score) * (w_etpat / w_flow)
                euclid_d = np.sqrt(np.power(dx, 2) + np.power(dy, 2))  # euclidean distance
                lcl_dna_score = 1 - euclid_d
            else:
                lcl_dna_score = lcl_flow_score
            #
            #
            #
            # printing
            if tui:
                print('Status: {:8.4f} % | Set '.format(runstatus), end='\t')
                if etpat:
                    print('{:8.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f} '
                          '{:8.3f} {:8.3f} {:8.3f} | {:8.3f} {:8.3f} {:8.3f} '.format(pset[0],
                                                                                      pset[1],
                                                                                      pset[2],
                                                                                      pset[3],
                                                                                      pset[4],
                                                                                      pset[5],
                                                                                      pset[6],
                                                                                      pset[7],
                                                                                      pset[8],
                                                                                      pset[9],
                                                                                      lcl_dna_score,
                                                                                      lcl_flow_score,
                                                                                      lcl_etpat_score))
                else:
                    print('{:8.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f} {:8.3f} '
                          '{:8.3f} {:8.3f} {:8.3f} | {:8.3f} '.format(pset[0],
                                                                                      pset[1],
                                                                                      pset[2],
                                                                                      pset[3],
                                                                                      pset[4],
                                                                                      pset[5],
                                                                                      pset[6],
                                                                                      pset[7],
                                                                                      pset[8],
                                                                                      pset[9],
                                                                                      lcl_dna_score
                                                                                      ))
            #
            #
            lcl_dna_id = 'G' + str(g + 1) + '-' + str(i)  # stamp DNA
            #
            # store in retrieval system:
            pop_dct[lcl_dna_id] = lcl_dna  # dict of population
            #
            ids_lst.append(lcl_dna_id)
            ids_set_lst.append(lcl_set_id)
            scores_lst[i] = lcl_dna_score
            if etpat:
                scores_flow_lst[i] = lcl_flow_score
                scores_etpat_lst[i] = lcl_etpat_score
            if tracepop:
                dnas_lst.append(lcl_dna)
            #
            # store metadata metrics:
            meta_nse[i] = flow_analysis['Metrics']['NSE']
            meta_nselog[i] = flow_analysis['Metrics']['NSElog']
            meta_kge[i] = flow_analysis['Metrics']['KGE']
            meta_kgelog[i] = flow_analysis['Metrics']['KGElog']
            meta_rmse[i] = flow_analysis['Metrics']['RMSE']
            meta_rmselog[i] = flow_analysis['Metrics']['RMSElog']
            meta_pbias[i] = flow_analysis['Metrics']['PBias']
            meta_rmse_cfc[i] = flow_analysis['Metrics']['RMSE_CFC']
            meta_rmselog_cfc[i] = flow_analysis['Metrics']['RMSElog_CFC']
            #
            # store metadata diags
            meta_q_sum[i] = flow_analysis['Stats']['Q_sum']
            meta_q_mean[i] = flow_analysis['Stats']['Q_mean']
            meta_q_sd[i] = flow_analysis['Stats']['Q_sd']
            meta_qb_sum[i] = np.sum(sim_df['Qb'].values)
            meta_qb_mean[i] = np.mean(sim_df['Qb'].values)
            meta_qb_sd[i] = np.std(sim_df['Qb'].values + (loglim * (sim_df['Qb'].values <= 0)))
            meta_c_qprec[i] = flow_analysis['Stats']['Q_sum'] / np.sum(sim_df['Prec'].values)
            meta_c_qbprec[i] = np.sum(sim_df['Qb'].values) / np.sum(sim_df['Prec'].values)
            #
            # store metadata et
            if etpat:
                meta_mean_mse_etpat[i] = etpat_analysis['Stats']['MSE_mean']
                meta_min_mse_etpat[i] = etpat_analysis['Stats']['MSE_min']
                meta_max_mse_etpat[i] = etpat_analysis['Stats']['MSE_max']
                #
                meta_mean_rmse_etpat[i] = etpat_analysis['Stats']['RMSE_mean']
                meta_min_rmse_etpat[i] = etpat_analysis['Stats']['RMSE_min']
                meta_max_rmse_etpat[i] = etpat_analysis['Stats']['RMSE_max']
                #
                meta_mean_wmse_etpat[i] = etpat_analysis['Stats']['W-MSE_mean']
                meta_min_wmse_etpat[i] = etpat_analysis['Stats']['W-MSE_min']
                meta_max_wmse_etpat[i] = etpat_analysis['Stats']['W-MSE_max']
                #
                meta_mean_wrmse_etpat[i] = etpat_analysis['Stats']['W-RMSE_mean']
                meta_min_wrmse_etpat[i] = etpat_analysis['Stats']['W-RMSE_min']
                meta_max_wrmse_etpat[i] = etpat_analysis['Stats']['W-RMSE_max']
                #
                meta_mean_nse_etpat[i] = etpat_analysis['Stats']['NSE_mean']
                meta_min_nse_etpat[i] = etpat_analysis['Stats']['NSE_min']
                meta_max_nse_etpat[i] = etpat_analysis['Stats']['NSE_max']
                #
                meta_mean_kge_etpat[i] = etpat_analysis['Stats']['KGE_mean']
                meta_min_kge_etpat[i] = etpat_analysis['Stats']['KGE_min']
                meta_max_kge_etpat[i] = etpat_analysis['Stats']['KGE_max']
                #
                meta_mean_r_etpat[i] = etpat_analysis['Stats']['R_mean']
                meta_min_r_etpat[i] = etpat_analysis['Stats']['R_min']
                meta_max_r_etpat[i] = etpat_analysis['Stats']['R_max']
                #
                meta_mean_mean_obs_etpat[i] = etpat_analysis['Stats']['MeanObs_mean']
                meta_min_mean_obs_etpat[i] = etpat_analysis['Stats']['MeanObs_min']
                meta_max_mean_obs_etpat[i] = etpat_analysis['Stats']['MeanObs_max']
                #
                meta_mean_mean_sim_etpat[i] = etpat_analysis['Stats']['MeanSim_mean']
                meta_min_mean_sim_etpat[i] = etpat_analysis['Stats']['MeanSim_min']
                meta_max_mean_sim_etpat[i] = etpat_analysis['Stats']['MeanSim_max']
                #
                meta_mean_mean_err[i] = etpat_analysis['Stats']['MeanErr_mean']
                meta_min_mean_err[i] = etpat_analysis['Stats']['MeanErr_min']
                meta_max_mean_err[i] = etpat_analysis['Stats']['MeanErr_max']
        #
        # trace full population
        if tracepop:
            if etpat:
                trace_pop.append({'DNAs': dnas_lst[:],
                                  'Ids': ids_lst[:],
                                  'SetIds': ids_set_lst[:],
                                  'L': scores_lst[:],
                                  'L_Q':scores_flow_lst[:],
                                  'L_ET':scores_etpat_lst[:],
                                  'NSE':meta_nse[:],
                                  'NSElog':meta_nselog[:],
                                  'KGE':meta_kge[:],
                                  'KGElog':meta_kgelog[:],
                                  'RMSE':meta_rmse[:],
                                  'RMSElog':meta_rmselog[:],
                                  'PBias':meta_pbias[:],
                                  'RMSE_CFC':meta_rmse_cfc[:],
                                  'RMSElog_CFC':meta_rmselog_cfc[:],
                                  'Q_sum':meta_q_sum[:],
                                  'Q_mean':meta_q_mean[:],
                                  'Q_sd':meta_q_sd[:],
                                  'Qb_sum':meta_qb_sum[:],
                                  'Qb_mean':meta_qb_mean,
                                  'Qb_sd':meta_qb_sd[:],
                                  'Q_C':meta_c_qprec[:],
                                  'Qb_C':meta_c_qbprec[:],
                                  'ET_MSE_mean': meta_mean_mse_etpat[:],
                                  'ET_MSE_min': meta_min_mse_etpat[:],
                                  'ET_MSE_max': meta_max_mse_etpat[:],
                                  'ET_RMSE_mean': meta_mean_rmse_etpat[:],
                                  'ET_RMSE_min': meta_min_rmse_etpat[:],
                                  'ET_RMSE_max': meta_max_rmse_etpat[:],
                                  'ET_W-MSE_mean': meta_mean_wmse_etpat[:],
                                  'ET_W-MSE_min': meta_min_wmse_etpat[:],
                                  'ET_W-MSE_max': meta_max_wmse_etpat[:],
                                  'ET_W-RMSE_mean': meta_mean_wrmse_etpat[:],
                                  'ET_W-RMSE_min': meta_min_wrmse_etpat[:],
                                  'ET_W-RMSE_max': meta_max_wrmse_etpat[:],
                                  'ET_NSE_mean': meta_mean_nse_etpat[:],
                                  'ET_NSE_min': meta_min_nse_etpat[:],
                                  'ET_NSE_max': meta_max_nse_etpat[:],
                                  'ET_KGE_mean': meta_mean_kge_etpat[:],
                                  'ET_KGE_min': meta_min_kge_etpat[:],
                                  'ET_KGE_max': meta_max_kge_etpat[:],
                                  'ET_R_mean': meta_mean_r_etpat[:],
                                  'ET_R_min': meta_min_r_etpat[:],
                                  'ET_R_max': meta_max_r_etpat[:],
                                  'ET_MeanObs_mean': meta_mean_mean_obs_etpat[:],
                                  'ET_MeanObs_min': meta_min_mean_obs_etpat[:],
                                  'ET_MeanObs_max': meta_max_mean_obs_etpat[:],
                                  'ET_MeanSim_mean': meta_mean_mean_sim_etpat[:],
                                  'ET_MeanSim_min': meta_min_mean_sim_etpat[:],
                                  'ET_MeanSim_max': meta_max_mean_sim_etpat[:],
                                  'ET_MeanErr_mean': meta_mean_mean_err[:],
                                  'ET_MeanErr_min': meta_min_mean_err[:],
                                  'ET_MeanErr_max': meta_max_mean_err[:]
                                  })
            else:
                trace_pop.append({'DNAs': dnas_lst[:],
                                  'Ids': ids_lst[:],
                                  'SetIds': ids_set_lst[:],
                                  'L': scores_lst[:],
                                  'NSE': meta_nse[:],
                                  'NSElog': meta_nselog[:],
                                  'KGE': meta_kge[:],
                                  'KGElog': meta_kgelog[:],
                                  'RMSE': meta_rmse[:],
                                  'RMSElog': meta_rmselog[:],
                                  'PBias': meta_pbias[:],
                                  'RMSE_CFC': meta_rmse_cfc[:],
                                  'RMSElog_CFC': meta_rmselog_cfc[:],
                                  'Q_sum': meta_q_sum[:],
                                  'Q_mean': meta_q_mean[:],
                                  'Q_sd': meta_q_sd[:],
                                  'Qb_sum': meta_qb_sum[:],
                                  'Qb_mean': meta_qb_mean,
                                  'Qb_sd': meta_qb_sd[:],
                                  'Q_C': meta_c_qprec[:],
                                  'Qb_C': meta_c_qbprec[:],
                                  })
        #
        # 5) RECRUIT new population
        if g == 0:
            if etpat:
                df_parents_rank = pd.DataFrame({'Id': ids_lst[:],
                                               'SetIds': ids_set_lst[:],
                                               'L': scores_lst[:],
                                               'L_Q':scores_flow_lst[:],
                                               'L_ET':scores_etpat_lst[:],
                                               'NSE': meta_nse[:],
                                               'NSElog': meta_nselog[:],
                                               'KGE': meta_kge[:],
                                               'KGElog': meta_kgelog[:],
                                               'RMSE': meta_rmse[:],
                                               'RMSElog': meta_rmselog[:],
                                               'PBias': meta_pbias[:],
                                               'RMSE_CFC': meta_rmse_cfc[:],
                                               'RMSElog_CFC': meta_rmselog_cfc[:],
                                               'Q_sum': meta_q_sum[:],
                                               'Q_mean': meta_q_mean[:],
                                               'Q_sd': meta_q_sd[:],
                                               'Qb_sum': meta_qb_sum[:],
                                               'Qb_mean': meta_qb_mean[:],
                                               'Qb_sd': meta_qb_sd[:],
                                               'Q_C': meta_c_qprec[:],
                                               'Qb_C': meta_c_qbprec[:],
                                                'ET_MSE_mean': meta_mean_mse_etpat[:],
                                                'ET_MSE_min': meta_min_mse_etpat[:],
                                                'ET_MSE_max': meta_max_mse_etpat[:],
                                                'ET_RMSE_mean': meta_mean_rmse_etpat[:],
                                                'ET_RMSE_min': meta_min_rmse_etpat[:],
                                                'ET_RMSE_max': meta_max_rmse_etpat[:],
                                                'ET_W-MSE_mean': meta_mean_wmse_etpat[:],
                                                'ET_W-MSE_min': meta_min_wmse_etpat[:],
                                                'ET_W-MSE_max': meta_max_wmse_etpat[:],
                                                'ET_W-RMSE_mean': meta_mean_wrmse_etpat[:],
                                                'ET_W-RMSE_min': meta_min_wrmse_etpat[:],
                                                'ET_W-RMSE_max': meta_max_wrmse_etpat[:],
                                                'ET_NSE_mean': meta_mean_nse_etpat[:],
                                                'ET_NSE_min': meta_min_nse_etpat[:],
                                                'ET_NSE_max': meta_max_nse_etpat[:],
                                                'ET_KGE_mean': meta_mean_kge_etpat[:],
                                                'ET_KGE_min': meta_min_kge_etpat[:],
                                                'ET_KGE_max': meta_max_kge_etpat[:],
                                                'ET_R_mean': meta_mean_r_etpat[:],
                                                'ET_R_min': meta_min_r_etpat[:],
                                                'ET_R_max': meta_max_r_etpat[:],
                                                'ET_MeanObs_mean': meta_mean_mean_obs_etpat[:],
                                                'ET_MeanObs_min': meta_min_mean_obs_etpat[:],
                                                'ET_MeanObs_max': meta_max_mean_obs_etpat[:],
                                                'ET_MeanSim_mean': meta_mean_mean_sim_etpat[:],
                                                'ET_MeanSim_min': meta_min_mean_sim_etpat[:],
                                                'ET_MeanSim_max': meta_max_mean_sim_etpat[:],
                                                'ET_MeanErr_mean': meta_mean_mean_err[:],
                                                'ET_MeanErr_min': meta_min_mean_err[:],
                                                'ET_MeanErr_max': meta_max_mean_err[:]
                                                })
            else:
                df_parents_rank = pd.DataFrame({'Id': ids_lst[:],
                                                'SetIds': ids_set_lst[:],
                                                'L': scores_lst[:],
                                                'NSE': meta_nse[:],
                                                'NSElog': meta_nselog[:],
                                                'KGE': meta_kge[:],
                                                'KGElog': meta_kgelog[:],
                                                'RMSE': meta_rmse[:],
                                                'RMSElog': meta_rmselog[:],
                                                'PBias': meta_pbias[:],
                                                'RMSE_CFC': meta_rmse_cfc[:],
                                                'RMSElog_CFC': meta_rmselog_cfc[:],
                                                'Q_sum': meta_q_sum[:],
                                                'Q_mean': meta_q_mean[:],
                                                'Q_sd': meta_q_sd[:],
                                                'Qb_sum': meta_qb_sum[:],
                                                'Qb_mean': meta_qb_mean[:],
                                                'Qb_sd': meta_qb_sd[:],
                                                'Q_C': meta_c_qprec[:],
                                                'Qb_C': meta_c_qbprec[:]
                                                })
        else:
            if etpat:
                df_offspring_rank = pd.DataFrame({'Id': ids_lst[:],
                                                'SetIds': ids_set_lst[:],
                                                'L': scores_lst[:],
                                                'L_Q': scores_flow_lst[:],
                                                'L_ET': scores_etpat_lst[:],
                                                'NSE': meta_nse[:],
                                                'NSElog': meta_nselog[:],
                                                'KGE': meta_kge[:],
                                                'KGElog': meta_kgelog[:],
                                                'RMSE': meta_rmse[:],
                                                'RMSElog': meta_rmselog[:],
                                                'PBias': meta_pbias[:],
                                                'RMSE_CFC': meta_rmse_cfc[:],
                                                'RMSElog_CFC': meta_rmselog_cfc[:],
                                                'Q_sum': meta_q_sum[:],
                                                'Q_mean': meta_q_mean[:],
                                                'Q_sd': meta_q_sd[:],
                                                'Qb_sum': meta_qb_sum[:],
                                                'Qb_mean': meta_qb_mean[:],
                                                'Qb_sd': meta_qb_sd[:],
                                                'Q_C': meta_c_qprec[:],
                                                'Qb_C': meta_c_qbprec[:],
                                                  'ET_MSE_mean': meta_mean_mse_etpat[:],
                                                  'ET_MSE_min': meta_min_mse_etpat[:],
                                                  'ET_MSE_max': meta_max_mse_etpat[:],
                                                  'ET_RMSE_mean': meta_mean_rmse_etpat[:],
                                                  'ET_RMSE_min': meta_min_rmse_etpat[:],
                                                  'ET_RMSE_max': meta_max_rmse_etpat[:],
                                                  'ET_W-MSE_mean': meta_mean_wmse_etpat[:],
                                                  'ET_W-MSE_min': meta_min_wmse_etpat[:],
                                                  'ET_W-MSE_max': meta_max_wmse_etpat[:],
                                                  'ET_W-RMSE_mean': meta_mean_wrmse_etpat[:],
                                                  'ET_W-RMSE_min': meta_min_wrmse_etpat[:],
                                                  'ET_W-RMSE_max': meta_max_wrmse_etpat[:],
                                                  'ET_NSE_mean': meta_mean_nse_etpat[:],
                                                  'ET_NSE_min': meta_min_nse_etpat[:],
                                                  'ET_NSE_max': meta_max_nse_etpat[:],
                                                  'ET_KGE_mean': meta_mean_kge_etpat[:],
                                                  'ET_KGE_min': meta_min_kge_etpat[:],
                                                  'ET_KGE_max': meta_max_kge_etpat[:],
                                                  'ET_R_mean': meta_mean_r_etpat[:],
                                                  'ET_R_min': meta_min_r_etpat[:],
                                                  'ET_R_max': meta_max_r_etpat[:],
                                                  'ET_MeanObs_mean': meta_mean_mean_obs_etpat[:],
                                                  'ET_MeanObs_min': meta_min_mean_obs_etpat[:],
                                                  'ET_MeanObs_max': meta_max_mean_obs_etpat[:],
                                                  'ET_MeanSim_mean': meta_mean_mean_sim_etpat[:],
                                                  'ET_MeanSim_min': meta_min_mean_sim_etpat[:],
                                                  'ET_MeanSim_max': meta_max_mean_sim_etpat[:],
                                                  'ET_MeanErr_mean': meta_mean_mean_err[:],
                                                  'ET_MeanErr_min': meta_min_mean_err[:],
                                                  'ET_MeanErr_max': meta_max_mean_err[:]
                                                  })
            else:
                df_offspring_rank = pd.DataFrame({'Id': ids_lst[:],
                                                  'SetIds': ids_set_lst[:],
                                                  'L': scores_lst[:],
                                                  'NSE': meta_nse[:],
                                                  'NSElog': meta_nselog[:],
                                                  'KGE': meta_kge[:],
                                                  'KGElog': meta_kgelog[:],
                                                  'RMSE': meta_rmse[:],
                                                  'RMSElog': meta_rmselog[:],
                                                  'PBias': meta_pbias[:],
                                                  'RMSE_CFC': meta_rmse_cfc[:],
                                                  'RMSElog_CFC': meta_rmselog_cfc[:],
                                                  'Q_sum': meta_q_sum[:],
                                                  'Q_mean': meta_q_mean[:],
                                                  'Q_sd': meta_q_sd[:],
                                                  'Qb_sum': meta_qb_sum[:],
                                                  'Qb_mean': meta_qb_mean[:],
                                                  'Qb_sd': meta_qb_sd[:],
                                                  'Q_C': meta_c_qprec[:],
                                                  'Qb_C': meta_c_qbprec[:],
                                                  })
            df_parents_rank = df_parents_rank.append(df_offspring_rank, ignore_index=True)
        #
        #
        # 6) RANK population Maximization. Therefore ascending=False
        df_parents_rank.drop_duplicates(subset=['SetIds'], inplace=True, ignore_index=True)
        df_parents_rank.sort_values(by='L', ascending=False, inplace=True)
        #
        #
        # 7) SELECT mating pool (Parents) - N largest score values
        df_parents_rank = df_parents_rank.nlargest(len(parents), columns=['L'])
        if tui:
            print('\nTop 10 parents:')
            if etpat:
                print(df_parents_rank[['Id', 'L', 'L_Q', 'L_ET']].head(10).to_string())
            else:
                print(df_parents_rank[['Id', 'L']].head(10).to_string())
        #
        # Extract parents
        parents_ids = df_parents_rank['Id'].values  # numpy array of string IDs
        #
        # Retrieve from pop dict the DNA of each parent ID
        parents_lst = list()
        for i in range(len(parents_ids)):
            parents_lst.append(pop_dct[parents_ids[i]])
        #
        # update parents DNAs
        parents = tuple(parents_lst)
        #
        # recicle index
        pop_dct = dict()
        for i in range(len(parents_ids)):
            pop_dct[parents_ids[i]] = parents_lst[i]
        #
        # tracing len setup
        tr_len = int(len(parents) * tracefrac)
        # trace best parents
        if etpat:
            trace.append({'DNAs': parents[:tr_len],
                          'Ids': parents_ids[:tr_len],
                          'SetIds': df_parents_rank['SetIds'].values[:tr_len],
                          'L': df_parents_rank['L'].values[:tr_len],
                          'L_Q': df_parents_rank['L_Q'].values[:tr_len],
                          'L_ET':df_parents_rank['L_ET'].values[:tr_len],
                          'NSE': df_parents_rank['NSE'].values[:tr_len],
                          'NSElog': df_parents_rank['NSElog'].values[:tr_len],
                          'KGE': df_parents_rank['KGE'].values[:tr_len],
                          'KGElog': df_parents_rank['KGElog'].values[:tr_len],
                          'RMSE': df_parents_rank['RMSE'].values[:tr_len],
                          'RMSElog': df_parents_rank['RMSElog'].values[:tr_len],
                          'PBias': df_parents_rank['PBias'].values[:tr_len],
                          'RMSE_CFC': df_parents_rank['RMSE_CFC'].values[:tr_len],
                          'RMSElog_CFC': df_parents_rank['RMSElog_CFC'].values[:tr_len],
                          'Q_sum': df_parents_rank['Q_sum'].values[:tr_len],
                          'Q_mean': df_parents_rank['Q_mean'].values[:tr_len],
                          'Q_sd': df_parents_rank['Q_sd'].values[:tr_len],
                          'Qb_sum': df_parents_rank['Qb_sum'].values[:tr_len],
                          'Qb_mean': df_parents_rank['Qb_mean'].values[:tr_len],
                          'Qb_sd': df_parents_rank['Qb_sd'].values[:tr_len],
                          'Q_C': df_parents_rank['Q_C'].values[:tr_len],
                          'Qb_C': df_parents_rank['Qb_C'].values[:tr_len],
                          'ET_MSE_mean': meta_mean_mse_etpat[:tr_len],
                          'ET_MSE_min': meta_min_mse_etpat[:tr_len],
                          'ET_MSE_max': meta_max_mse_etpat[:tr_len],
                          'ET_RMSE_mean': meta_mean_rmse_etpat[:tr_len],
                          'ET_RMSE_min': meta_min_rmse_etpat[:tr_len],
                          'ET_RMSE_max': meta_max_rmse_etpat[:tr_len],
                          'ET_W-MSE_mean': meta_mean_wmse_etpat[:tr_len],
                          'ET_W-MSE_min': meta_min_wmse_etpat[:tr_len],
                          'ET_W-MSE_max': meta_max_wmse_etpat[:tr_len],
                          'ET_W-RMSE_mean': meta_mean_wrmse_etpat[:tr_len],
                          'ET_W-RMSE_min': meta_min_wrmse_etpat[:tr_len],
                          'ET_W-RMSE_max': meta_max_wrmse_etpat[:tr_len],
                          'ET_NSE_mean': meta_mean_nse_etpat[:tr_len],
                          'ET_NSE_min': meta_min_nse_etpat[:tr_len],
                          'ET_NSE_max': meta_max_nse_etpat[:tr_len],
                          'ET_KGE_mean': meta_mean_kge_etpat[:tr_len],
                          'ET_KGE_min': meta_min_kge_etpat[:tr_len],
                          'ET_KGE_max': meta_max_kge_etpat[:tr_len],
                          'ET_R_mean': meta_mean_r_etpat[:tr_len],
                          'ET_R_min': meta_min_r_etpat[:tr_len],
                          'ET_R_max': meta_max_r_etpat[:tr_len],
                          'ET_MeanObs_mean': meta_mean_mean_obs_etpat[:tr_len],
                          'ET_MeanObs_min': meta_min_mean_obs_etpat[:tr_len],
                          'ET_MeanObs_max': meta_max_mean_obs_etpat[:tr_len],
                          'ET_MeanSim_mean': meta_mean_mean_sim_etpat[:tr_len],
                          'ET_MeanSim_min': meta_min_mean_sim_etpat[:tr_len],
                          'ET_MeanSim_max': meta_max_mean_sim_etpat[:tr_len],
                          'ET_MeanErr_mean': meta_mean_mean_err[:tr_len],
                          'ET_MeanErr_min': meta_min_mean_err[:tr_len],
                          'ET_MeanErr_max': meta_max_mean_err[:tr_len]
                          })
        else:
            trace.append({'DNAs': parents[:tr_len],
                          'Ids': parents_ids[:tr_len],
                          'SetIds': df_parents_rank['SetIds'].values[:tr_len],
                          'L': df_parents_rank['L'].values[:tr_len],
                          'NSE': df_parents_rank['NSE'].values[:tr_len],
                          'NSElog': df_parents_rank['NSElog'].values[:tr_len],
                          'KGE': df_parents_rank['KGE'].values[:tr_len],
                          'KGElog': df_parents_rank['KGElog'].values[:tr_len],
                          'RMSE': df_parents_rank['RMSE'].values[:tr_len],
                          'RMSElog': df_parents_rank['RMSElog'].values[:tr_len],
                          'PBias': df_parents_rank['PBias'].values[:tr_len],
                          'RMSE_CFC': df_parents_rank['RMSE_CFC'].values[:tr_len],
                          'RMSElog_CFC': df_parents_rank['RMSElog_CFC'].values[:tr_len],
                          'Q_sum': df_parents_rank['Q_sum'].values[:tr_len],
                          'Q_mean': df_parents_rank['Q_mean'].values[:tr_len],
                          'Q_sd': df_parents_rank['Q_sd'].values[:tr_len],
                          'Qb_sum': df_parents_rank['Qb_sum'].values[:tr_len],
                          'Qb_mean': df_parents_rank['Qb_mean'].values[:tr_len],
                          'Qb_sd': df_parents_rank['Qb_sd'].values[:tr_len],
                          'Q_C': df_parents_rank['Q_C'].values[:tr_len],
                          'Qb_C': df_parents_rank['Qb_C'].values[:tr_len],
                          })
        if tui:
            print('Trace size: {} KB'.format(getsizeof(trace)))
            print('Index size: {} KB'.format(getsizeof(pop_dct)))
    #
    # retrieve last best solution
    last = trace[len(trace) - 1]
    last_dna = last['DNAs'][0]
    last_score = last['L'][0]
    if etpat:
        last_flow_score = last['L_Q'][0]
        last_etpat_score = last['L_ET'][0]
    pset = express_parameter_set(last_dna[0], lowerb=lower_bound, ranges=ranges, gridsize=grid)
    if tui:
        print('\n\nMaximum Likelihood Model:')
        tui_df = pd.DataFrame({'Parameter': ('m', 'lamb', 'qo', 'cpmax', 'sfmax', 'erz', 'ksat', 'c', 'k', 'n'),
                               'Set': pset})
        print(tui_df.to_string())
        print('L = {:.3f}'.format(last_score))
        if etpat:
            print('L_Q = {:.3f}'.format(last_flow_score))
            print('L_ET = {:.3f}'.format(last_etpat_score))
    #
    # wrap traced
    wtrace = wrapper(traced=trace, lowerb=lower_bound, ranges=ranges, gridsize=grid, etpat=etpat)
    out_dct = {'MLM':pset, 'Traced': wtrace}
    if tracepop:
        # wrap population
        wtrace_pop = wrapper(traced=trace_pop, lowerb=lower_bound, ranges=ranges, gridsize=grid, etpat=etpat)
        out_dct['Population'] = wtrace_pop
    return out_dct


def ensemble(series, models_df, shruparam, twibins, countmatrix, canopy_df, qt0, lat, area, basinshadow, tui=False):
    """
    Run the entire ensemble of models
    :param series: pandas Dataframe of input series
    :param models_df: pandas Dataframe of models (parameters in the fields)
    :param shruparam: pandas Dataframe of shruparam (see simulation() routine)
    :param twibins: same as in the simulation() routine
    :param countmatrix: same as in the  simulation() routine
    :param qt0: same as in the  simulation() routine
    :param lat: same as in the  simulation() routine
    :param area: same as in the  simulation() routine
    :param basinshadow: same as in the  simulation() routine
    :param tui: boolean to display
    :return: dictionary of pandas dataframes with the Lower (5%), Mid (50%) and Upper (95%) of streamflow ('Q')
     and baseflow ('Qb')
    """
    from analyst import frequency

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

    if tui:
        from tui import status
    #
    # set up
    n_ensem = len(models_df)
    t_ensem = len(series)
    sim_grid_q = np.zeros(shape=(n_ensem, t_ensem))
    sim_grid_qb = np.zeros(shape=(n_ensem, t_ensem))
    sim_grid_et = np.zeros(shape=(n_ensem, t_ensem))
    sim_ids = models_df['Id']
    m_vec = models_df['m'].values
    lamb_vec = models_df['lamb'].values
    qo_vec = models_df['qo'].values
    cpmax_vec = models_df['cpmax'].values
    sfmax_vec = models_df['sfmax'].values
    erz_vec = models_df['erz'].values
    ksat_vec = models_df['ksat'].values
    c_vec = models_df['c'].values
    k_vec = models_df['k'].values
    n_vec = models_df['n'].values
    #
    # simulate every model:
    for i in range(n_ensem):
        if tui:
            status(msg='Running model {} of {}'.format(i + 1, n_ensem))
        sim_dct = simulation(series=series,
                             shruparam=shruparam,
                             twibins=twibins,
                             countmatrix=countmatrix,
                             lamb=lamb_vec[i],
                             qt0=qt0,
                             area=area,
                             basinshadow=basinshadow,
                             canopy=canopy_df,
                             lat=lat,
                             m=m_vec[i],
                             qo=qo_vec[i],
                             cpmax=cpmax_vec[i],
                             sfmax=sfmax_vec[i],
                             erz=erz_vec[i],
                             ksat=ksat_vec[i],
                             c=c_vec[i],
                             k=k_vec[i],
                             n=n_vec[i],
                             mapback=False)
        sim_df = sim_dct['Series']
        sim_grid_q[i] = sim_df['Q'].values
        sim_grid_qb[i] = sim_df['Qb'].values
        sim_grid_et[i] = sim_df['ET'].values
    #
    # transpose grid
    sim_grid_q_t = np.transpose(sim_grid_q)
    sim_grid_qb_t = np.transpose(sim_grid_qb)
    sim_grid_et_t = np.transpose(sim_grid_et)
    #
    # compute lo-mid-hi bounds
    lo_bound_q = np.zeros(shape=np.shape(series['Prec'].values))
    mid_bound_q = np.zeros(shape=np.shape(series['Prec'].values))
    hi_bound_q = np.zeros(shape=np.shape(series['Prec'].values))
    lo_bound_qb = np.zeros(shape=np.shape(series['Prec'].values))
    mid_bound_qb = np.zeros(shape=np.shape(series['Prec'].values))
    hi_bound_qb = np.zeros(shape=np.shape(series['Prec'].values))
    lo_bound_et = np.zeros(shape=np.shape(series['Prec'].values))
    mid_bound_et = np.zeros(shape=np.shape(series['Prec'].values))
    hi_bound_et = np.zeros(shape=np.shape(series['Prec'].values))
    #
    for t in range(len(sim_grid_q_t)):
        #
        # full discharge
        lcl_df = pd.DataFrame({'Values': sim_grid_q_t[t]})
        lcl_freq = frequency(dataframe=lcl_df, var_field='Values')
        lo_bound_q[t] = lcl_freq['Values'][5]
        mid_bound_q[t] = lcl_freq['Values'][50]
        hi_bound_q[t] = lcl_freq['Values'][95]
        #
        # baseflow
        lcl_df = pd.DataFrame({'Values': sim_grid_qb_t[t]})
        lcl_freq = frequency(dataframe=lcl_df, var_field='Values')
        lo_bound_qb[t] = lcl_freq['Values'][5]
        mid_bound_qb[t] = lcl_freq['Values'][50]
        hi_bound_qb[t] = lcl_freq['Values'][95]
        #
        # et
        lcl_df = pd.DataFrame({'Values': sim_grid_et_t[t]})
        lcl_freq = frequency(dataframe=lcl_df, var_field='Values')
        lo_bound_et[t] = lcl_freq['Values'][5]
        mid_bound_et[t] = lcl_freq['Values'][50]
        hi_bound_et[t] = lcl_freq['Values'][95]
    #
    # built exports
    out_q_df = pd.DataFrame({'Date': series['Date'],
                             'Lo_5': lo_bound_q,
                             'Mid_50': mid_bound_q,
                             'Hi_95': hi_bound_q })
    app_q_df = pd.DataFrame(sim_grid_q_t, columns=sim_ids)
    out_q_df = pd.concat([out_q_df, app_q_df], axis=1)
    # qb:
    out_qb_df = pd.DataFrame({'Date': series['Date'],
                             'Lo_5': lo_bound_qb,
                              'Mid_50': mid_bound_qb,
                             'Hi_95': hi_bound_qb})
    app_qb_df = pd.DataFrame(sim_grid_qb_t, columns=sim_ids)
    out_qb_df = pd.concat([out_qb_df, app_qb_df], axis=1)
    # et
    out_et_df = pd.DataFrame({'Date': series['Date'],
                              'Lo_5': lo_bound_et,
                              'Mid_50': mid_bound_et,
                              'Hi_95': hi_bound_et})
    app_et_df = pd.DataFrame(sim_grid_et_t, columns=sim_ids)
    out_et_df = pd.concat([out_et_df, app_et_df], axis=1)
    #
    return {'Q':out_q_df, 'Qb':out_qb_df, 'ET': out_et_df}


