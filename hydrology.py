import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
def find_nse(qobs, qsim, type='lin'):
    """
    Nash-Sutcliffe efficiency of 2 arrays of same length
    :param qobs: observed array
    :param qsim: simulated array
    :param type: 'log' for NSElog10
    :return: float number of NSE
    """
    qavg = qobs * 0.0 + np.mean(qsim)
    if type == 'log':
        qobs = np.log10(qobs)
        qsim = np.log10(qsim)
        qavg = np.log10(qavg)
    nse = 1 - (np.sum(np.power(qobs - qsim, 2))/ np.sum((qobs - qavg)))
    return nse


def find_pbias(qobs, qsim):
    """
    Percent bias coefficient (PBIAS)
    :param qobs:
    :param qsim:
    :return:   
    
    """
       
    pbias = 100 * np.sum(qobs - qsim) / np.sum(qobs)
    return pbias


def find_rmse(qobs, qsim, type='lin'):
    """
    Root of mean squared error of 2 arrays of same length
    :param qobs: observed array
    :param qsim: simulated array
    :param type: log' for RMSElog10
    :return: float
    """
    if type == 'log':
        qobs = np.log10(qobs)
        qsim = np.log10(qsim)
    rmse = np.sqrt(np.mean(np.power(qobs - qsim, 2)))
    return rmse


def find_cfc(a):
    """

    :param a: array
    :return: tuple with exeedance probability (%) and CFC values from input array
    """
    ptles = np.arange(0, 101, 1)
    cfc = np.percentile(a, ptles)
    exeed = 100 - ptles
    return (exeed, cfc)


'''


# auxiliar functions
def avg_2d(var2d, weight):
    """
    average raster value based on weight mask
    :param var2d: 2d numpy array of raster values
    :param weight: 2d numpy array of weight mask
    :return: float average value
    """
    lcl_avg = np.sum(var2d * weight) / np.sum(weight)  # avg
    return lcl_avg


def flatten_clear(array, mask):
    """
    convert an nd numpy array to 1d of True values
    :param array: nd numpy array
    :param mask: nd pseudo-boolean numpy array of mask (0 and 1)
    :return: 1d numpy array of True values
    """
    masked = np.copy(array)
    masked[mask == 0] = np.nan
    flatten = masked.flatten()
    cleared = masked[~np.isnan(masked)]
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

# silent functions
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
    nash_unit = nash/np.sum(nash)  # normalise to get the unit hydrograph
    for t in range(0, len(time)):
        lcl_q = q[t]
        if t == 0:
            qs = lcl_q * nash_unit
        else:
            qs[t:] = qs[t:] + lcl_q * nash_unit[:size - t]
    return qs


def count_matrix_twi_shru(twi, shru, aoi, shruids, ntwibins=10):
    # flat and clear:
    twi_flat_clear = flatten_clear(twi, aoi)
    # extract histogram of TWI
    twi_hist, twi_bins = np.histogram(twi_flat_clear, bins=ntwibins)
    twi_bins = twi_bins[1:]
    #
    countmatrix = np.zeros(shape=(ntwibins, len(shruids)), dtype='int32')
    for i in range(len(countmatrix)):
        for j in range(len(countmatrix[i])):
            #print('{}\t{}'.format(twi_bins[i], shruids[j]))
            if i == 0:
                lcl_mask =  (shru == shruids[j]) * (twi < twi_bins[i]) * aoi
            elif i == len(countmatrix) - 1:
                lcl_mask = (shru == shruids[j]) * (twi >= twi_bins[i - 1]) * aoi
            else:
                lcl_mask = (shru == shruids[j]) * (twi >= twi_bins[i - 1]) * (twi < twi_bins[i])  * aoi #* (shru == shruids[j]) * aoi
            countmatrix[i][j] = np.sum(lcl_mask)
    return countmatrix, twi_bins, shruids

# deprecated
def count_matrix(array2d1, array2d2, bins1, bins2, aoi):
    """
    2D histogram computing of 2 raster layers
    :param array2d1: 2d numpy array of raster 1
    :param array2d2: 2d numpy array of raster 2
    :param bins1: int number of histograms bins for raster 1
    :param bins2: int number of histograms bins for raster 2
    :param aoi: 2d numpy Area Of Interest raster pseudo-boolean mask (0 and 1)
    :return:
    1) 2d histogram/count (rows - raster 1, columns - raster 2) matrix
    2) tuple of histogram of raster 1: fist: bins - 1d numpy array, second: count - 1d numpy array
    3) tuple of histogram of raster 2: fist: bins - 1d numpy array, second: count - 1d numpy array
    """
    np.warnings.filterwarnings('ignore')
    # verify bins size
    a1_flat_clear = flatten_clear(array=array2d1, mask=aoi)
    a1_unique = np.unique(a1_flat_clear)
    if len(a1_unique) < bins1:
        bins1 = len(a1_unique)
    a2_flat_clear = flatten_clear(array=array2d2, mask=aoi)
    a2_unique = np.unique(a2_flat_clear)
    if len(a2_unique) < bins2:
        bins2 = len(a2_unique)
    #
    # extract histograms
    a1_hist, a1_bins = np.histogram(a1_flat_clear, bins=bins1)
    a1_bins = a1_bins[1:]
    #
    a2_hist, a2_bins = np.histogram(a2_flat_clear, bins=bins2)
    a2_bins = a2_bins[1:]
    #
    # Cross histograms
    countmatrix = np.zeros(shape=(bins1, bins2), dtype='int32')
    for i in range(len(countmatrix)):
        if i == 0:
            lcl_a1 = 1.0 * (array2d1 <= a1_bins[i])
        else:
            lcl_a1 = 1.0 * (array2d1 > a1_bins[i - 1]) * (array2d1 <= a1_bins[i])
        for j in range(len(countmatrix[i])):
            if j == 0:
                lcl_a2 = 1.0 * (array2d2 <= a2_bins[j])
            else:
                lcl_a2 = 1.0 * (array2d2 > a2_bins[j - 1]) * (array2d2 <= a2_bins[j])
            countmatrix[i][j] = np.sum(lcl_a1 * lcl_a2 * aoi)
    return countmatrix, (a1_bins, a1_hist), (a2_bins, a2_hist)


def map_back(zmatrix, a1, a2, bins1, bins2):
    """
    Map back function using a Z-Matrix
    :param zmatrix: 2d numpy array of z matrix of values
    :param a1: 2d numpy array reference array of rows (ex: TWI)
    :param a2: 2d numpy array reference array of columns  (ex: CN)
    :param bins1: 1d numpy array of array 1 histogram bins
    :param bins2: 1d numpy array of array 2 histogram bins
    :return: 2d numpy array
    """
    # initiate map array
    map = np.zeros(shape=np.shape(a1))
    for i in range(len(zmatrix)):
        if i == 0:
            mask1 = (a1 <= bins1[i])
        else:
            mask1 = (a1 > bins1[i - 1]) * (a1 <= bins1[i])
        for j in range(len(zmatrix[i])):
            if j == 0:
                mask2 = (a2 <= bins2[j])
            else:
                mask2 = (a2 > bins2[j - 1]) * (a2 <= bins2[j])
            # compute local map:
            lclmap = mask1 * mask2 * zmatrix[i][j]
            map = map + lclmap
    return map

# PET model functions
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

# topmodel functions
# deprecated:
def topmodel_s0max(cn, a):
    """
    Plans 3 Model of S0max as a function of CN

    s0max = a * (100 - CN + b)

    :param cn: float or nd array of CN
    :param a: scaling parameter in mm/CN
    :return: float or nd array of S0max
    """
    lcl_s0max = a * (100 - cn + 0.1)
    return lcl_s0max


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
    local deficit di
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


def topmodel_sim(series, shruparam, twibins, countmatrix, lamb, qt0, m, qo, cpmax, sfmax, erz, ksat, c, lat, k, n,
                 area, mapback=False, mapvar='all', qobs=False, tui=False):
    #
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
    #shru_ids = shruparam['Id'].values
    # local cpmax
    cpmax_i = cpmax * shruparam['f_Canopy'].values * np.ones(shape=shape, dtype='float32')
    # local sfmax
    sfmax_i = sfmax * shruparam['f_Surface'].values * np.ones(shape=shape, dtype='float32')
    # local erz
    erz_i = erz * shruparam['f_EfRootZone'].values * np.ones(shape=shape, dtype='float32')
    # local ksat
    ksat_i = ksat * shruparam['f_Ksat'].values * np.ones(shape=shape, dtype='float32')
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
    ts_tf = np.zeros(shape=size, dtype='float32')
    ts_inf = np.zeros(shape=size, dtype='float32')
    ts_r = np.zeros(shape=size, dtype='float32')
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
    ts_vsa = np.zeros(shape=size, dtype='float32')
    ts_vsa[0] = np.sum(vsa_i * countmatrix) / np.sum(countmatrix)
    #
    # Trace setup
    if mapback:
        if mapvar == 'all':
            mapvar = 'P-Temp-IRA-IRI-PET-D-Cpy-TF-Sfs-R-Inf-Unz-Qv-Evc-Evs-Tpun-Tpgw-ET-VSA'
        mapvar_lst = mapvar.split('-')
        # load to dict object a time series of empty zmaps for each variable
        mapkeys_dct = dict()
        for e in mapvar_lst:
            mapkeys_dct[e] = np.zeros(shape=(size, rows, cols), dtype='float32')
        # store timestep maps in dict
        mapback_dct = {'TF': tf_i, 'Qv': qv_i, 'R': r_i, 'ET': et_i, 'Cpy': cpy_i, 'Sfs': sfs_i, 'Inf': inf_i,
                       'Tpun': tpun_i, 'Evc': evc_i, 'Tpgw': tpgw_i, 'Evs': evs_i, 'VSA': vsa_i, 'P': prec_i,
                       'Temp': temp_i, 'IRA': ira_i, 'IRI': iri_i, 'PET': pet_i, 'D':d_i, 'Unz':unz_i}
        # append it to map
        for e in mapvar_lst:
            mapkeys_dct[e][0] = mapback_dct[e]
    #
    #
    #
    # ***** ESMA loop by finite differences *****
    for t in range(1, size):
        #
        # ****** UPDATE local water balance ******
        #
        # update Canopy water stock
        cpy_i = cpy_i + cpyin_i - tf_i - evc_i
        ts_cpy[t] = avg_2d(var2d=cpy_i, weight=countmatrix)  # compute average
        #
        # update Surface water stock
        sfs_i = sfs_i + sfsin_i - r_i - inf_i - evs_i
        ts_sfs[t] = avg_2d(var2d=sfs_i, weight=countmatrix)  # compute average
        #
        # update Unsaturated water stock
        unz_i = unz_i - qv_i - tpun_i + inf_i
        ts_unz[t] = avg_2d(var2d=unz_i, weight=countmatrix)  # compute average
        #
        #
        # ****** COMPUTE Flows for usage in next time step ******
        #
        # --- Canopy
        # compute current TF - Throughfall (or "effective precipitation")
        prec_i = ts_prec[t] * np.ones(shape=shape) # update PREC
        ira_i = ts_ira[t] * fira_i # update IRA
        cpyin_i = prec_i + ira_i  # canopy total input
        #plt.imshow(cpyin_i)
        #plt.title('Total input canopy')
        #plt.show()
        tf_i = ((cpyin_i - (cpmax_i - cpy_i)) * (cpyin_i > (cpmax_i - cpy_i)))
        ts_tf[t] = avg_2d(var2d=tf_i, weight=countmatrix)
        #
        # compute current Evc from canopy:
        pet_i = ts_pet[t] * np.ones(shape=shape)  # update PET
        petfull_i = ts_pet[t] * np.ones(shape=shape)  # reserve PET for mapping
        icp_i = cpy_i + cpyin_i - tf_i
        evc_i = (pet_i * (icp_i > pet_i)) + (icp_i * (icp_i <= pet_i))
        ts_evc[t] = avg_2d(var2d=evc_i, weight=countmatrix)  # compute average
        #
        # --- Surface
        # compute current runoff
        iri_i = ts_iri[t] * firi_i  # update IRI
        sfsin_i = tf_i + iri_i  # surface total input
        #plt.title('Total input surface')

        r_i = ((sfs_i + sfsin_i) - sfmax_i) * ((sfs_i + sfsin_i) > sfmax_i)
        ts_r[t] = avg_2d(var2d=r_i, weight=countmatrix)  # compute average
        #
        # compute surface depletion
        pet_i = pet_i - evc_i  # update pet
        #
        # compute potential separate flows
        p_evs_i = sfs_i * (pet_i / (ksat_i + pet_i + 1)) * ((ksat_i + pet_i) > 0)  # propotional to overall rate
        p_sfs_inf_i = sfs_i * (ksat_i / (ksat_i + pet_i + 1)) * ((ksat_i + pet_i) > 0)  # proportional to overall rate
        #plt.imshow(p_sfs_inf_i)
        #plt.show()
        #
        evs_i = (pet_i * (p_evs_i >= pet_i)) + (p_evs_i * (p_evs_i < pet_i))
        ts_evs[t] = avg_2d(var2d=evs_i, weight=countmatrix)  # compute average
        #
        p_unz_inf_i = (d_i - unz_i) * ((d_i - unz_i) > 0)
        inf_i = (p_sfs_inf_i * (p_sfs_inf_i < p_unz_inf_i)) + (p_unz_inf_i * (p_sfs_inf_i >= p_unz_inf_i))
        # inf_i = ((p_inf_i * (p_inf_i < (d_i - unz_i))) + ((d_i - unz_i) * (d_i > 0) * (p_inf_i >= (d_i - unz_i))))
        #plt.imshow(inf_i)
        #plt.show()
        ts_inf[t] = avg_2d(var2d=inf_i, weight=countmatrix)  # compute average
        #
        # update PET
        pet_i = pet_i - evs_i
        #
        # --- Unsaturated zone
        # compute QV
        p_qv_i = topmodel_qv(d=d_i, unz=unz_i, ksat=ksat_i)
        qv_i = unz_i * (p_qv_i/ (pet_i + p_qv_i + 1)) * ((pet_i + p_qv_i) > 0)  # + 1 to avoid division by zero
        ts_qv[t] = avg_2d(var2d=qv_i, weight=countmatrix)  # compute average
        #
        # compute tpun:
        p_tpun_i = unz_i * (pet_i / (pet_i + p_qv_i + 1)) * ((pet_i + p_qv_i) > 0)  # + 1 to avoid division by zero
        #plt.imshow(p_tpun_i * (countmatrix > 0))
        #plt.show()
        tpun_i = (pet_i * (p_tpun_i >= pet_i)) + (p_tpun_i * (p_tpun_i < pet_i))

        ts_tpun[t] = avg_2d(var2d=tpun_i, weight=countmatrix)  # compute average
        #
        # update PET
        pet_i = pet_i - tpun_i
        #
        # --- Saturated Root zone
        # compute tpgw:
        p_tpgw_i = (erz_i - d_i) * ((erz_i - d_i) > 0)  # potential tpgw
        tpgw_i = (pet_i * (p_tpgw_i >= pet_i)) + (p_tpgw_i * (p_tpgw_i < pet_i))
        ts_tpgw[t] = avg_2d(var2d=tpgw_i, weight=countmatrix)  # compute average
        #
        # --- ET
        # compute ET
        et_i = evc_i + evs_i + tpun_i + tpgw_i
        ts_et[t] = avg_2d(var2d=et_i, weight=countmatrix)  # compute average
        #
        #
        # ****** UPDATE Global Water balance ******
        # global water balance
        ts_d[t] = ts_d[t - 1] + ts_qb[t - 1] - ts_qv[t - 1] + ts_tpgw[t - 1]
        #
        # compute Qb - Baseflow
        ts_qb[t] = topmodel_qb(d=ts_d[t], qo=qo, m=m)
        #
        # Update Di
        d_i = topmodel_di(d=ts_d[t], twi=lamb_i, m=m, lamb=lamb)
        #plt.imshow(d_i, cmap='jet')
        #plt.show()
        #
        # compute VSA
        vsa_i = topmodel_vsai(di=d_i)
        ts_vsa[t] = np.sum(vsa_i * countmatrix) / np.sum(countmatrix)
        #
        # get temperature map:
        temp_i = ts_temp[t] * np.ones(shape=shape)
        #
        # trace section
        if mapback:
            # store timestep maps in dict
            mapback_dct = {'TF': tf_i, 'Qv': qv_i, 'R': r_i, 'ET': et_i, 'Cpy': cpy_i, 'Sfs': sfs_i, 'Inf': inf_i,
                           'Tpun': tpun_i, 'Evc': evc_i, 'Tpgw': tpgw_i, 'Evs': evs_i, 'VSA': vsa_i, 'P': prec_i,
                           'Temp': temp_i, 'IRA': ira_i, 'IRI': iri_i, 'PET': petfull_i, 'D':d_i, 'Unz':unz_i}
            # append it to map
            for e in mapvar_lst:
                mapkeys_dct[e][t] = mapback_dct[e]
    #
    # RUNOFF ROUTING by Nash Cascade of linear reservoirs
    ts_qs = nash_cascade(ts_r, k=k, n=n)
    #
    # compute full discharge Q = Qb + Qs
    ts_q = ts_qb + ts_qs
    ts_flow = convert_sq2q(sq=ts_q, area=area)
    #
    # export data
    exp_df = pd.DataFrame({'Date':series['Date'].values,
                           'Prec':series['Prec'].values,
                           'Temp':series['Temp'].values, 'PET':ts_pet, 'D':ts_d, 'Cpy':ts_cpy, 'TF':ts_tf,
                           'Sfs':ts_sfs, 'R':ts_r, 'Inf':ts_inf, 'Unz':ts_unz, 'Qv':ts_qv, 'Evc':ts_evc,
                           'Evs':ts_evs, 'Tpun':ts_tpun, 'Tpgw':ts_tpgw, 'ET':ts_et, 'Qb':ts_qb,'Qs':ts_qs, 'Q':ts_q,
                           'Flow':ts_flow})
    if qobs:
        exp_df['Qobs'] = series['Q'].values
    #
    if mapback:
        return exp_df, mapkeys_dct
    else:
        return exp_df

# deprecated:
def topmodel_hist(twi, cn, aoi, twibins=20, cnbins=10):
    """
    2D histogram for TOPMODEL in PLANS 3. Crossed variables: TWI and CN
    :param twi: 2d numpy array of TWI
    :param cn: 2d numpy array of CN
    :param aoi: 2d numpy array of AOI (Area of Interest)
    :param twibins: int number of bins in TWI histogram
    :param cnbins: int number of bins in CN histogram
    :return:
    1) 2d histogram/count (rows - TWI, columns - CN) matrix
    2) tuple of histogram of TWI: fist: bins - 1d numpy array, second: count - 1d numpy array
    3) tuple of histogram of CN: fist: bins - 1d numpy array, second: count - 1d numpy array
    """
    countmatrix, twi_hist, cn_hist = count_matrix(array2d1=twi, array2d2=cn, bins1=twibins, bins2=cnbins, aoi=aoi)
    return countmatrix, twi_hist, cn_hist

# deprecated:
def topmodel_sim_deprec(series, twihist, cnhist, countmatrix, lamb, ksat, m, qo, a, c, lat, qt0, k, n,
                 mapback=False, mapvar='R-ET-S1-S2', tui=False, qobs=False):
    """

    PLANS 3 TOPMODEL simulation procedure

    :param series: Pandas DataFrame of input series.
    Required fields: 'Date', 'Prec', 'Temp'. Optional: 'Q' (Q obs, in mm)
    :param twihist: tuple of histogram of TWI
    :param cnhist: tuple of histogram of CN
    :param countmatrix: 2D histogram of TWI and CN
    :param lamb: positive float - average TWI value of the AOI
    :param ksat: positive float - effective saturated hydraulic conductivity in mm/d
    :param m: positive float - effective transmissivity decay coefficient in mm
    :param qo: positive float - max baseflow when d=0 in mm/d
    :param a: positive float - scaling parameter for S0max model
    :param c: positive float - scaling parameter for PET model in Celcius
    :param lat: float - latitude in degrees for PET model
    :param qt0: positive float - baseflow at t=0 in mm/d
    :param k: positive float - Nash Cascade residence time in days
    :param n: positive float - equivalent number of reservoirs in Nash Cascade
    :param mapback: boolean control to map back variables
    :param mapvar: string code of variables to map back. Available variables:
    'TF', Qv', 'R', 'ET', 'S1', 'S2', 'Inf', 'Tp', 'Ev', 'Tpgw' (see below the relation)
    :param tui: boolean to control terminal messages
    :return: Pandas DataFrame of simulated variables:

    'Date': date (from input)
    'Prec': precipitation (from input), mm
    'Temp': temperature (from input), deg C
    'PET': simulated potential evapotranspiration, mm
    'S1': simulated water in S1 stock (canopy interceptation), mm
    'TF': simulated throughfall, mm
    'Ev': simulated evaporation from canopy, mm
    'S2': simulated water in S2 stock (unsaturated zone), mm
    'Inf': simulated infiltration, mm
    'R': simulated overland runoff, mm
    'Tp': simulated transpiration from unsaturated zone (S2), mm
    'Tpgw': simulated transpiration from the saturated zone, mm
    'ET': simulated actual evapotranspiration, mm
    'D': simulated soil water deficit, mm
    'Qb': simualted baseflow, mm
    'Qv':simulated recharge, mm
    'Qs':simulated surface flow, mm
    'Q': simulated streamflow, mm  (Q = Qb + Qs)
    'VSA': simulated variable source area (saturated areas), in %

    And
    if mapback=True:
    Dictionary of encoded 2d numpy arrays maps
    Keys to access maps: 'TF', Qv', 'R', 'ET', 'S1', 'S2', 'Inf', 'Tp', 'Ev', 'Tpgw'
    Each key stores an array of 2d numpy arrays (i.e., 3d array) in the ascending order of the time series.

    """
    #
    # extract data input
    ts_prec = series['Prec'].values
    ts_temp = series['Temp'].values
    size = len(ts_prec)
    #
    # compute PET
    days = series['Date'].dt.dayofyear
    ts_days = days.values
    lat = lat * np.pi / 180  # convet lat to radians
    ts_pet = pet_oudin(temperature=ts_temp, day=ts_days, latitude=lat, k1=c)
    #
    # set initial conditions
    d0 = topmodel_d0(qt0=qt0, qo=qo, m=m)
    #
    twi_bins = twihist[0]
    twi_count = twihist[1]
    s0max_bins = topmodel_s0max(cn=cnhist[0], a=a)  # convert cn to S0max
    #print(s0max_bins)
    s0max_count = cnhist[1]
    shape = np.shape(countmatrix)
    rows = shape[0]
    cols = shape[1]
    #
    # set 2d count parameter arrays
    s1maxi = 0.2 * s0max_bins * np.ones(shape=shape, dtype='float32')  # canopy water
    s2maxi = 0.8 * s0max_bins * np.ones(shape=shape, dtype='float32')  # rootzone
    rzdi = s2maxi.copy()
    #
    # get local Lambda
    lambi = np.reshape(twi_bins, (rows, 1)) * np.ones(shape=shape, dtype='float32')
    #
    # set 2d count variable arrays
    preci = ts_prec[0] * np.ones(shape=shape)
    peti = ts_pet[0] * np.ones(shape=shape)
    s1i = np.zeros(shape=shape)  # initial condition
    tfi = np.zeros(shape=shape)
    evi = np.zeros(shape=shape)
    qri = np.zeros(shape=shape)
    #
    s2i = np.zeros(shape=shape)  # initial condition
    infi = np.zeros(shape=shape)
    ri = np.zeros(shape=shape)
    tpi = np.zeros(shape=shape)
    tpgwi = np.zeros(shape=shape)
    eti = evi + tpi + tpgwi
    #
    #s3i = np.zeros(shape=shape)  # initial condition
    di = topmodel_di(d=d0, twi=lambi, m=m, lamb=lamb)
    vsai = topmodel_vsai(di=di)
    qvi = np.zeros(shape=shape)
    #
    # set stocks time series arrays and initial conditions
    ts_s1 = np.zeros(shape=np.shape(ts_prec), dtype='float32')
    ts_s1[0] = avg_2d(var2d=s1i, weight=countmatrix)
    ts_s2 = np.zeros(shape=np.shape(ts_prec), dtype='float32')
    ts_s2[0] = avg_2d(var2d=s2i, weight=countmatrix)
    ts_d = np.zeros(shape=np.shape(ts_prec), dtype='float32')
    ts_d[0] = d0
    ts_qv = np.zeros(shape=np.shape(ts_prec), dtype='float32')
    ts_qv[0] = avg_2d(var2d=qvi, weight=countmatrix)
    ts_qb = np.zeros(shape=np.shape(ts_prec), dtype='float32')
    ts_qb[0] = qt0
    ts_vsa = np.zeros(shape=np.shape(ts_prec), dtype='float32')
    ts_vsa[0] = np.sum(vsai * countmatrix) / np.sum(countmatrix)
    #
    # set flows time series arrays
    ts_ev = np.zeros(shape=np.shape(ts_prec), dtype='float32')
    ts_tp = np.zeros(shape=np.shape(ts_prec), dtype='float32')
    ts_tpgw = np.zeros(shape=np.shape(ts_prec), dtype='float32')
    ts_et = np.zeros(shape=np.shape(ts_prec), dtype='float32')
    ts_r = np.zeros(shape=np.shape(ts_prec), dtype='float32')
    ts_inf = np.zeros(shape=np.shape(ts_prec), dtype='float32')
    ts_tf = np.zeros(shape=np.shape(ts_prec), dtype='float32')
    #
    # Trace setup
    if mapback:
        mapvar_lst = mapvar.split('-')
        map_dct = dict()
        for e in mapvar_lst:
            map_dct[e] = np.zeros(shape=(size, rows, cols), dtype='float32')
    #
    if tui:
        print('Soil moisture accounting simulation...')
    # ESMA loop
    for t in range(1, size):
        #if tui:
        #   print('Step {}'.format(t))
        #
        # update S1 - Canopy storage - interceptation
        s1i = s1i - evi - tfi + preci
        ts_s1[t] = avg_2d(s1i, countmatrix)
        #
        # compute current EV - Evaporation from canopy storage
        peti = ts_pet[t] * np.ones(shape=shape)  # update PET
        evi = ((peti) * (s1i >= peti)) + (s1i * (s1i < peti))
        ts_ev[t] = avg_2d(evi, countmatrix)
        #
        # compute current TF - Throughfall (or "effective preciputation")
        preci = ts_prec[t] * np.ones(shape=shape)  # update PREC
        tfi = ((preci + s1i - evi - s1maxi) * ((preci + s1i - evi) >= s1maxi))
        ts_tf[t] = avg_2d(tfi, countmatrix)
        #
        # update S2 - Unsaturated soil water - vadoze zone
        s2i = s2i - qvi - tpi + infi
        ts_s2[t] = avg_2d(s2i, countmatrix)
        #
        # compute TP - Plant transpiration from vadoze zone
        peti = peti - evi  # update peti
        di_aux = di + ((np.max(di) + 3) * (di <= 0.0))  # auxiliar Di to replace zero values by a higher positive value to avoid division by zero
        ptpi = ((s2i * (rzdi >= di)) + ((s2i * rzdi / di_aux) * (rzdi < di))) * (di > 0.0)  # compute potential TP
        tpi = (ptpi * (peti >= ptpi)) + (peti * (peti < ptpi))
        ts_tp[t] = avg_2d(tpi, countmatrix)
        #
        # compute QV - Vertical flow to saturated zone - Water table recharge
        pqvi = (ksat * s2i / di_aux) * (di > 0.0)  # potential QV
        qvi = ((pqvi) * (s2i - tpi >= pqvi)) + ((s2i - tpi) * (s2i - tpi < pqvi))
        ts_qv[t] = avg_2d(qvi, countmatrix)
        #
        # compute Inf - Infiltration from surface. Soil has a potential infiltration capacity equal to the rootzone
        pinfi = (rzdi * (tfi >= rzdi)) + (tfi * (tfi < rzdi))  # potential infiltration -- infiltration capacity
        infi = ((di - s2i + tpi + qvi) * (pinfi >= (di - s2i + tpi + qvi))) + (pinfi * (pinfi < (di - s2i + tpi + qvi)))
        ts_inf[t] = avg_2d(infi, countmatrix)
        #
        # compute R - Runoff water
        ri = tfi - infi
        ts_r[t] = avg_2d(ri, countmatrix)
        #
        # compute TP-GW - Plant transpiration directly from the water table (if within reach of the root zone)
        peti = peti - tpi  # update peti
        ptpgwi = (rzdi - di) * (di < rzdi)  # potential local TP
        #ptpgwi = (s2maxi - di) * (di < s2maxi)  # potential local TP
        tpgwi = (peti * (ptpgwi > peti)) + (ptpgwi * (ptpgwi <= peti))
        ts_tpgw[t] = avg_2d(tpgwi, countmatrix)
        #
        # compute ET - Actual Evapo-transpiration
        eti = evi + tpi + tpgwi
        ts_et[t] = avg_2d(eti, countmatrix)
        #
        # update D water balance
        ts_d[t] = ts_d[t - 1] + ts_qb[t - 1] - ts_qv[t - 1] + ts_tpgw[t - 1]
        #
        # compute Qb - Baseflow
        ts_qb[t] = topmodel_qb(d=ts_d[t], qo=qo, m=m)
        #
        # Update Di
        di = topmodel_di(d=ts_d[t], twi=lambi, m=m, lamb=lamb)
        #
        # compute VSA
        vsai = topmodel_vsai(di=di)
        ts_vsa[t] = np.sum(vsai * countmatrix) / np.sum(countmatrix)
        #
        # trace section
        if mapback:
            dct = {'TF':tfi, 'Qv':qvi, 'R': ri, 'ET':eti, 'S1':s1i, 'S2':s2i, 'Inf':infi, 'Tp':tpi, 'Ev':evi,
                   'Tpgw':tpgwi}
            for e in mapvar_lst:
                map_dct[e][t] = dct[e]
    #
    # RUNOFF ROUTING by Nash Cascade of linear reservoirs
    if tui:
        print('Runoff routing...')
    ts_qs = nash_cascade(ts_r, k=k, n=n)
    #
    # compute full discharge Q = Qb + Qs
    ts_q = ts_qb + ts_qs
    #
    # export data
    exp_df = pd.DataFrame({'Date': series['Date'].values,
                           'Prec': series['Prec'].values,
                           'Temp': series['Temp'].values,
                           'PET': ts_pet,
                           'S1': np.round(ts_s1, 3), 'TF': np.round(ts_tf, 3), 'Ev': np.round(ts_ev, 3),
                           'S2': ts_s2, 'Inf': ts_inf, 'R': ts_r, 'Tp': ts_tp, 'Tpgw': ts_tpgw, 'ET': ts_et,
                           'D': ts_d, 'Qb': ts_qb, 'Qv': ts_qv, 'Qs': ts_qs, 'Q': ts_q, 'VSA': ts_vsa})
    if qobs:
        exp_df['Qobs'] = series['Q'].values
    #
    if mapback:
        return exp_df, map_dct
    else:
        return exp_df

# todo docstring (redo)
def topmodel_calib(series, shruparam, twibins, countmatrix, lamb, qt0, lat, area, m_range, qo_range, cpmax_range,
                   sfmax_range, erz_range, ksat_range, c_range, k_range, n_range, mapback=False, mapvar='D-R-ET',
                   tui=True, grid=200, generations=100, popsize=200, offsfrac=1,
                   mutrate=0.4, puremutrate=0.1, cutfrac=0.33, tracefrac=0.1, tracepop=True, metric='NSE'):
    """
    PLANS 3 TOPMODEL calibration procedure

    :param series: Pandas DataFrame of input series.
    Required fields: 'Date', 'Prec', 'Temp', 'Q' (Observed Q, in mm)
    :param twihist: tuple of histogram of TWI
    :param cnhist: tuple of histogram of CN
    :param countmatrix: 2D histogram of TWI and CN
    :param lamb: positive float - average TWI value of the AOI
    :param lat: float - latitude in degrees for PET model
    :param qt0: positive float - baseflow at t=0 in mm/d
    :param ksat_range: tuple with min and max floats - search range of effective saturated hydraulic conductivity in mm/d
    :param m_range: tuple with min and max floats - search range of effective transmissivity decay coefficient in mm
    :param qo_range: tuple with min and max floats - search range of max baseflow when d=0 in mm/d
    :param a_range: tuple with min and max floats - search range of scaling parameter for S0max model
    :param c_range: tuple with min and max floats - search range of scaling parameter for PET model in Celcius
    :param k_range: tuple with min and max floats - search range of Nash Cascade residence time in days
    :param n_range: tuple with min and max floats - search range of equivalent number of reservoirs in Nash Cascade
    :param mapback: boolean control to map back variables (best of last generation)
    :param mapvar: string code of variables to map back. Available variables:
    'TF', Qv', 'R', 'ET', 'S1', 'S2', 'Inf', 'Tp', 'Ev', 'Tpgw' (see docstrig in topmodel_sim())
    :param tui: boolean to control terminal messages
    :param generations: int - number of generations
    :param popsize: int - number of initial population
    :param grid: int - grid mesh to partitionate the ranges
    :param offsfrac: positive float - offspring fraction to initial population
    :param mutrate: float (0 to 1) - rate of mutation
    :param puremutrate: float (0 to 1) - internal rate of pure mutations
    :param cutfrac: float (0 to 0.5) - fraction of cutsize in crossover
    :param tracefrac: float (0 to 1) - fraction of parents tracing
    :param tracepop: boolean to control full population tracing
    :param metric: string code to metric. Available codes:

    'NSE' - NSE of series values
    'NSElog' - NSE of log10 of series values
    'KGE' - KGE of series values
    'KGElog' - KGE of log10 of series values
    'PBias' - Pbias of series values
    'RMSE' - RMSE of series values
    'RMSElog' - RMSE of log10 of series values
    'RMSE-CFC' - RMSE of CFCs values
    'RMSElog-CFC' - RMSE of log10 of CFCs values

    :return: Pandas DataFrame of simulated variables: see topmodel_sim() docstring.
    And
    if mapback=True:
    Dictionary of encoded 2d numpy arrays maps
    Keys to access maps: 'TF', Qv', 'R', 'ET', 'S1', 'S2', 'Inf', 'Tp', 'Ev', 'Tpgw'
    Each key stores an array of 2d numpy arrays (i.e., 3d array) in the ascending order of the time series.

    """
    from evolution import generate_population, generate_offspring, recruitment
    from analyst import nse, kge, rmse, pbias, frequency
    from sys import getsizeof
    from datetime import datetime

    def wrapper(traced, lowerb, ranges):
        """
        Wrapper function to return list of dataframes with expressed genes
        :param traced: list of dictionaries with encoded generations
        :param lowerb: numpy array of lowerbound values
        :param ranges: numpy array of range values
        :return: list of dataframes of each generation
        """
        df_lst = list()
        # generation loop:
        for g in range(len(traced)):
            m_lst = list()
            qo_lst = list()
            cpmax_lst = list()
            sfmax_lst = list()
            erz_lst = list()
            ksat_lst = list()
            c_lst = list()
            k_lst = list()
            n_lst = list()
            # individual loop:
            for i in range(len(traced[g]['DNAs'])):
                lcl_dna = traced[g]['DNAs'][i]
                #print(lcl_dna[0], end='\t\t')
                lcl_pset = express_parameter_set(gene=lcl_dna[0], lowerb=lowerb, ranges=ranges)
                #print(lcl_pset)
                m_lst.append(lcl_pset[0])
                qo_lst.append(lcl_pset[1])
                cpmax_lst.append(lcl_pset[2])
                sfmax_lst.append(lcl_pset[3])
                erz_lst.append(lcl_pset[4])
                ksat_lst.append(lcl_pset[5])
                c_lst.append(lcl_pset[6])
                k_lst.append(lcl_pset[7])
                n_lst.append(lcl_pset[8])
            lcl_df = pd.DataFrame({'Id':traced[g]['Ids'], 'Score':traced[g]['Scores'],
                                    'm':m_lst, 'qo':qo_lst, 'cpmax':cpmax_lst, 'sfmax':sfmax_lst, 'erz':erz_lst,
                                   'ksat':ksat_lst, 'c':c_lst, 'k':k_lst, 'n':n_lst})
            #print(lcl_df.to_string())
            df_lst.append(lcl_df.copy())
        return df_lst

    def express_parameter_set(gene, lowerb, ranges):
        """
        Expression of parameter set
        :param gene: gene tuple
        :param lowerb: numpy array of lowerbound values
        :param ranges: numpy array of range values
        :return: numpy array of parameter set
        """
        return (np.array(gene) * ranges / 100) + lowerb

    # run setup
    runsize = generations * popsize * 2
    #
    #
    # reset random state using time
    seed = int(str(datetime.now())[-6:])
    np.random.seed(seed)
    if tui:
        print('Random Seed: {}'.format(seed))
    #
    #
    # bounds setup  # todo improve function of scales
    lowerbound = np.array((np.min(m_range), np.min(qo_range), np.min(cpmax_range), np.min(sfmax_range),
                           np.min(erz_range), np.min(ksat_range), np.min(c_range), np.min(k_range), np.min(n_range)))
    upperbound = np.array((np.max(m_range), np.max(qo_range), np.max(cpmax_range), np.max(sfmax_range),
                           np.max(erz_range), np.max(ksat_range), np.max(c_range), np.max(k_range), np.max(n_range)))
    ranges = upperbound - lowerbound
    #
    #
    # Evolution setup
    nucleotides = tuple(np.arange(0, grid + 1))
    parents = generate_population(nucleotides=(nucleotides,), genesizes=(9,), popsize=popsize)
    #for e in parents:
    #    print(e[0])
    trace = list()  # list to append best solutions
    if tracepop:
        trace_pop = list()
    #
    #
    # generation loop:
    counter = 0
    for g in range(generations):
        if tui:
            print('\n\nGeneration {}\n'.format(g + 1))
        # get offstring
        offspring = generate_offspring(parents, offsfrac=offsfrac, nucleotides=(nucleotides,), mutrate=mutrate,
                                       puremutrate=puremutrate, cutfrac=cutfrac)
        # recruit new population
        population = recruitment(parents, offspring)
        if tui:
            print('Population: {} KB       '.format(getsizeof(population)))
            print('                   | Set  ', end='\t  ')
            print('{:7} {:7} {:7} {:7} {:7} {:7} {:7} {:7} {:7}'.format('m', 'qo',
                                                                                                   'cpmax', 'sfmax',
                                                                                                   'erz', 'ksat',
                                                                                                   'c', 'k',
                                                                                                   'n'))
        # fit new population
        ids_lst = list()
        scores_lst = list()
        pop_dct = dict()
        if tracepop:
            dnas_lst = list()
        # loop in individuals
        for i in range(len(population)):
            runstatus = 100 * counter / runsize
            counter = counter + 1
            #
            # get local score and id:
            lcl_dna = population[i]  # local dna
            #
            #
            #
            # express parameter set
            pset = express_parameter_set(gene=lcl_dna[0], lowerb=lowerbound, ranges=ranges)  # express the parameter set
            #
            #
            # run topmodel
            sim_df = topmodel_sim(series=series, shruparam=shruparam, twibins=twibins, countmatrix=countmatrix,
                                  lamb=lamb, qt0=qt0, m=pset[0], qo=pset[1], cpmax=pset[2], sfmax=pset[3], erz=pset[4],
                                  ksat=pset[5], c=pset[6], lat=lat, k=pset[7], n=pset[8], area=area, tui=False,
                                  qobs=True, mapback=False)
            #
            #
            # Get fitness score:
            loglim = 0.000001
            sobs = series['Q'].values
            ssim = sim_df['Q'].values
            sobs_log = np.log10(sobs + (loglim * (sobs <= 0)))
            ssim_log = np.log10(ssim + (loglim * (ssim <= 0)))
            if metric == 'NSE':
                lcl_dna_score = nse(obs=sobs, sim=ssim)
            elif metric == 'NSElog':
                lcl_dna_score = nse(obs=sobs_log, sim=ssim_log)
            elif metric == 'KGE':
                lcl_dna_score = kge(obs=sobs, sim=ssim)
            elif metric == 'KGElog':
                lcl_dna_score = kge(obs=sobs_log, sim=ssim_log)
            elif metric == 'RMSE':
                lcl_dna_score = rmse(obs=sobs, sim=ssim) * -1
            elif metric == 'RMSElog':
                lcl_dna_score = rmse(obs=sobs_log, sim=ssim_log) * -1
            elif metric == 'PBias':
                lcl_dna_score = 1 / (np.abs(pbias(obs=sobs, sim=ssim)) + loglim)
            elif metric == 'RMSE-CFC':
                cfc_obs = frequency(series=sobs)['Values']
                cfc_sim = frequency(series=ssim)['Values']
                lcl_dna_score = rmse(obs=cfc_obs, sim=cfc_sim) * -1
            elif metric == 'RMSElog-CFC':
                cfc_obs = frequency(series=sobs)['Values']
                cfc_sim = frequency(series=ssim)['Values']
                lcl_dna_score = rmse(obs=np.log10(cfc_obs), sim=np.log10(cfc_sim)) * -1
            else:
                lcl_dna_score = nse(obs=sobs, sim=ssim)
            # printing
            if tui:
                print('Status: {:8.4f} % | Set '.format(runstatus), end='\t')
                print('{:7.3f} {:7.3f} {:7.3f} {:7.3f} {:7.3f} {:7.3f} {:7.3f} {:7.3f} {:7.3f}'.format(pset[0], pset[1],
                                                                                                       pset[2], pset[3],
                                                                                                       pset[4], pset[5],
                                                                                                       pset[6], pset[7],
                                                                                                       pset[8]), end='   ')
                print(' | Score = {:.3f}'.format(lcl_dna_score))
            #
            #
            lcl_dna_id = 'G' + str(g + 1) + '-' + str(i)
            #
            # store in retrieval system:
            pop_dct[lcl_dna_id] = lcl_dna
            ids_lst.append(lcl_dna_id)
            scores_lst.append(lcl_dna_score)
            if tracepop:
                dnas_lst.append(lcl_dna)
        #
        # trace full population
        if tracepop:
            trace_pop.append({'DNAs': dnas_lst[:], 'Ids': ids_lst[:], 'Scores': scores_lst[:]})
        #
        # rank new population (Survival)
        df_population_rank = pd.DataFrame({'Id': ids_lst, 'Score': scores_lst})
        df_population_rank.sort_values(by='Score', ascending=False, inplace=True)
        #
        # Selection of mating pool
        df_parents_rank = df_population_rank.nlargest(len(parents), columns=['Score'])
        #
        parents_ids = df_parents_rank['Id'].values  # numpy array of string IDs
        parents_scores = df_parents_rank['Score'].values  # numpy array of float scores
        #
        parents_lst = list()
        for i in range(len(parents_ids)):
            parents_lst.append(pop_dct[parents_ids[i]])
        parents = tuple(parents_lst)  # update parents DNAs
        #
        # tracing
        tr_len = int(len(parents) * tracefrac)
        # printing
        '''if tui:
            for i in range(tr_len):
                print('{}\t\t\tScore: {}'.format(parents[i], round(parents_scores[i], 3)))'''
        #
        # trace best parents
        trace.append({'DNAs': parents[:tr_len], 'Ids': parents_ids[:tr_len], 'Scores': parents_scores[:tr_len]})
    #
    # retrieve last best solution
    last = trace[len(trace) - 1]
    last_dna = last['DNAs'][0]
    last_score = last['Scores'][0]
    pset = express_parameter_set(last_dna[0], lowerb=lowerbound, ranges=ranges)
    if tui:
        print('\n\nBEST solution:')
        print(pset)
        print('Score = {}'.format(last_score))
    wtrace = wrapper(traced=trace, lowerb=lowerbound, ranges=ranges)
    wtrace_pop = wrapper(traced=trace_pop, lowerb=lowerbound, ranges=ranges)
    if tracepop:
        return pset, wtrace, wtrace_pop
    else:
        return pset, wtrace
