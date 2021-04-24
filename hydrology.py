import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # todo remove this after release

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
    nash_unit = nash/np.sum(nash)  # normalise to get the unit hydrograph
    for t in range(0, len(time)):
        lcl_q = q[t]
        if t == 0:
            qs = lcl_q * nash_unit
        else:
            qs[t:] = qs[t:] + lcl_q * nash_unit[:size - t]
    return qs


def count_matrix(twi, shru, aoi, shrubins, twibins):
    # todo docstring
    countmatrix = np.zeros(shape=(len(twibins), len(shrubins)), dtype='int32')
    for i in range(len(countmatrix)):
        for j in range(len(countmatrix[i])):
            #print('{}\t{}'.format(twi_bins[i], shruids[j]))
            if i == 0:
                lcl_mask =  (shru == shrubins[j]) * (twi < twibins[i]) * aoi
            elif i == len(countmatrix) - 1:
                lcl_mask = (shru == shrubins[j]) * (twi >= twibins[i - 1]) * aoi
            else:
                lcl_mask = (shru == shrubins[j]) * (twi >= twibins[i - 1]) * (twi < twibins[i])  * aoi #* (shru == shruids[j]) * aoi
            countmatrix[i][j] = np.sum(lcl_mask)
    return countmatrix, twibins, shrubins


def built_zmap(varmap, twi, shru, twibins, shrubins):
    # todo docstring
    zmap = np.zeros(shape=(len(twibins), len(shrubins)))
    for i in range(len(zmap)):
        for j in range(len(zmap[i])):
            if i == 0:
                lcl_mask =  (shru == shrubins[j]) * (twi < twibins[i])
            elif i == len(varmap) - 1:
                lcl_mask = (shru == shrubins[j]) * (twi >= twibins[i - 1])
            else:
                lcl_mask = (shru == shrubins[j]) * (twi >= twibins[i - 1]) * (twi < twibins[i])
            if np.sum(lcl_mask) == 0.0:
                zmap[i][j] = 0.0
            else:
                zmap[i][j] = np.sum(varmap * lcl_mask) / np.sum(lcl_mask)  # mean variable value at the local mask
    return zmap


def extract_zmap_signal(zmap, mask):
    array_flat = zmap.flatten()
    mask_flat = (1 * (mask.flatten() > 0))
    return flatten_clear(array=array_flat, mask=mask_flat)


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


def topmodel_sim(series, shruparam, twibins, countmatrix, lamb, qt0, m, qo, cpmax, sfmax, erz, ksat, c,
                 lat, k, n, area, basinshadow, mapback=False, mapvar='all', mapdates='all', qobs=False, tui=False):
    # todo docstring
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
    ts_vsa = np.zeros(shape=size, dtype='float32')
    ts_vsa[0] = np.sum(vsa_i * basinshadow) / np.sum(basinshadow)
    #
    #
    # Z-Map Trace setup
    if mapback:
        if mapvar == 'all':
            mapvar = 'P-Temp-IRA-IRI-PET-D-Cpy-TF-Sfs-R-RSE-RIE-RC-Inf-Unz-Qv-Evc-Evs-Tpun-Tpgw-ET-VSA'
        mapvar_lst = mapvar.split('-')
        # map dates protocol
        mapped_dates = list()  # list of actual mapped files
        if mapdates == 'all':
            mapsize = size
            mapped_dates = list(pd.to_datetime(series['Date'], format='%y-%m-%d'))
        else:
            # extract map dates to array
            mapid = 0
            #
            # prep map dates
            mapdates_df = pd.DataFrame({'Date': mapdates.split('&')})
            mapdates_df['DateStr'] = mapdates_df['Date'].str.strip()
            mapdates_df['Date'] = pd.to_datetime(mapdates_df['Date'])
            lookup_dates = mapdates_df['Date'].values  # it is coming as datetime!
            #print(lookup_dates)
            #print(len(lookup_dates))
            #
            # get series dates
            dates_series = series['Date'].values
            #
            # built the array of timestep index of dates to map
            map_timesteps = list()
            for i in range(len(lookup_dates)):
                lcl_series_df = series.query('Date == "{}"'.format(lookup_dates[i]))
                #print(lcl_series_df)
                #print(len(lcl_series_df))
                if len(lcl_series_df) == 1:
                    lcl_step = lcl_series_df.index[0]
                    #print(lcl_step)
                    map_timesteps.append(lcl_step)  # append index to list
                    mapped_dates.append(mapdates_df['DateStr'].values[i])
            mapsize = len(map_timesteps)
            #print(mapsize)
        #print(len(mapped_dates))
        #
        # load to dict object a time series of empty zmaps for each variable
        mapkeys_dct = dict()
        for e in mapvar_lst:
            mapkeys_dct[e] = np.zeros(shape=(mapsize, rows, cols), dtype='float32')
        # store initial zmaps in dict
        ext = 1 * (countmatrix > 0)  # filter by map extension
        mapback_dct = {'TF': tf_i * ext, 'Qv': qv_i * ext, 'R': r_i * ext,
                       'RSE': rse_i * ext, 'RIE': rie_i * ext,
                       'RC': rc_i * ext, 'ET': et_i * ext,
                       'Cpy': cpy_i * ext, 'Sfs': sfs_i * ext,
                       'Inf': inf_i * ext, 'Tpun': tpun_i * ext,
                       'Evc': evc_i * ext, 'Tpgw': tpgw_i * ext,
                       'Evs': evs_i * ext, 'VSA': vsa_i * ext,
                       'P': prec_i * ext, 'Temp': temp_i * ext,
                       'IRA': ira_i * ext, 'IRI': iri_i * ext,
                       'PET': pet_i * ext, 'D': d_i * ext,
                       'Unz': unz_i * ext}
        # append it to map the first record in the map time series
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
        ts_cpy[t] = avg_2d(var2d=cpy_i, weight=basinshadow)  # compute average
        #
        # update Surface water stock
        sfs_i = sfs_i + sfsin_i - r_i - inf_i - evs_i
        ts_sfs[t] = avg_2d(var2d=sfs_i, weight=basinshadow)  # compute average
        #
        # update Unsaturated water stock
        unz_i = unz_i - qv_i - tpun_i + inf_i
        ts_unz[t] = avg_2d(var2d=unz_i, weight=basinshadow)  # compute average
        #
        #
        # ****** COMPUTE Flows for usage in next time step ******
        #
        # --- Canopy
        # compute current TF - Throughfall (or "effective precipitation")
        prec_i = ts_prec[t] * np.ones(shape=shape) # update PREC
        ira_i = ts_ira[t] * fira_i # update IRA
        cpyin_i = prec_i + ira_i  # canopy total input
        tf_i = ((cpyin_i - (cpmax_i - cpy_i)) * (cpyin_i > (cpmax_i - cpy_i)))
        ts_tf[t] = avg_2d(var2d=tf_i, weight=basinshadow)
        #
        # compute current Evc from canopy:
        pet_i = ts_pet[t] * np.ones(shape=shape)  # update PET
        petfull_i = ts_pet[t] * np.ones(shape=shape)  # reserve PET for mapping
        icp_i = cpy_i + cpyin_i - tf_i
        evc_i = (pet_i * (icp_i > pet_i)) + (icp_i * (icp_i <= pet_i))
        ts_evc[t] = avg_2d(var2d=evc_i, weight=basinshadow)  # compute average in the basin
        #
        # --- Surface
        #
        # compute current runoff
        iri_i = ts_iri[t] * firi_i  # update IRI
        sfsin_i = tf_i + iri_i  # surface total input
        r_i = ((sfs_i + sfsin_i) - sfmax_i) * ((sfs_i + sfsin_i) > sfmax_i)
        # separate runoff
        rse_i = r_i * vsa_i  # Saturation excess runoff - Dunnean runoff
        rie_i = r_i * (vsa_i == 0.0) # Infiltration excess runoff - Hortonian runoff
        rc_i = (prec_i > 0) * (r_i / ((prec_i == 0) + prec_i))
        # compute global runoff:
        ts_r[t] = avg_2d(var2d=r_i, weight=basinshadow)  # compute average in the basin
        ts_rse[t] = avg_2d(var2d=rse_i, weight=basinshadow)
        ts_rie[t] = avg_2d(var2d=rie_i, weight=basinshadow)
        ts_rc[t] = avg_2d(var2d=rc_i, weight=basinshadow)
        '''
        # idea for multiple basin runoff routing
        for b in range(len(basins_list)):
            ts_r[b][t] = avg_2d(var2d=r_i, weight=basins_list[b])  # time series of Runoff as 2d array
        '''
        #
        # compute surface depletion
        pet_i = pet_i - evc_i  # update pet
        #
        # compute potential separate flows
        p_evs_i = sfs_i * (pet_i / (ksat_i + pet_i + 1)) * ((ksat_i + pet_i) > 0)  # propotional to overall rate
        p_sfs_inf_i = sfs_i * (ksat_i / (ksat_i + pet_i + 1)) * ((ksat_i + pet_i) > 0)  # proportional to overall rate
        #
        evs_i = (pet_i * (p_evs_i >= pet_i)) + (p_evs_i * (p_evs_i < pet_i))
        ts_evs[t] = avg_2d(var2d=evs_i, weight=basinshadow)  # compute average in the basin
        #
        p_unz_inf_i = (d_i - unz_i) * ((d_i - unz_i) > 0)
        inf_i = (p_sfs_inf_i * (p_sfs_inf_i < p_unz_inf_i)) + (p_unz_inf_i * (p_sfs_inf_i >= p_unz_inf_i))
        ts_inf[t] = avg_2d(var2d=inf_i, weight=basinshadow)  # compute average in the basin
        #
        # update PET
        pet_i = pet_i - evs_i
        #
        # --- Unsaturated zone
        # compute QV
        p_qv_i = topmodel_qv(d=d_i, unz=unz_i, ksat=ksat_i)
        qv_i = unz_i * (p_qv_i/ (pet_i + p_qv_i + 1)) * ((pet_i + p_qv_i) > 0)  # + 1 to avoid division by zero
        ts_qv[t] = avg_2d(var2d=qv_i, weight=basinshadow)  # compute average in the basin
        #
        # compute tpun:
        p_tpun_i = unz_i * (pet_i / (pet_i + p_qv_i + 1)) * ((pet_i + p_qv_i) > 0)  # + 1 to avoid division by zero
        tpun_i = (pet_i * (p_tpun_i >= pet_i)) + (p_tpun_i * (p_tpun_i < pet_i))

        ts_tpun[t] = avg_2d(var2d=tpun_i, weight=basinshadow)  # compute average in the basin
        #
        # update PET
        pet_i = pet_i - tpun_i
        #
        # --- Saturated Root zone
        # compute tpgw:
        p_tpgw_i = (erz_i - d_i) * ((erz_i - d_i) > 0)  # potential tpgw
        tpgw_i = (pet_i * (p_tpgw_i >= pet_i)) + (p_tpgw_i * (p_tpgw_i < pet_i))
        ts_tpgw[t] = avg_2d(var2d=tpgw_i, weight=basinshadow)  # compute average in the basin
        #
        # --- ET
        # compute ET
        et_i = evc_i + evs_i + tpun_i + tpgw_i
        ts_et[t] = avg_2d(var2d=et_i, weight=basinshadow)  # compute average in the basin
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
        ts_vsa[t] = np.sum(vsa_i * basinshadow) / np.sum(basinshadow)
        #
        # get temperature map:
        temp_i = ts_temp[t] * np.ones(shape=shape)
        #
        # trace section
        if mapback:
            # store timestep maps in dict
            mapback_dct = {'TF': tf_i * ext, 'Qv': qv_i * ext, 'R': r_i * ext,
                           'RSE': rse_i * ext, 'RIE': rie_i * ext,
                           'RC': rc_i * ext, 'ET': et_i * ext,
                           'Cpy': cpy_i * ext, 'Sfs': sfs_i * ext,
                           'Inf': inf_i * ext, 'Tpun': tpun_i * ext,
                           'Evc': evc_i * ext, 'Tpgw': tpgw_i * ext,
                           'Evs': evs_i * ext, 'VSA': vsa_i * ext,
                           'P': prec_i * ext, 'Temp': temp_i * ext,
                           'IRA': ira_i * ext, 'IRI': iri_i * ext,
                           'PET': petfull_i * ext, 'D': d_i * ext,
                           'Unz': unz_i * ext}
            if mapdates == 'all':
                # append it to map
                for e in mapvar_lst:
                    mapkeys_dct[e][t] = mapback_dct[e]
            else:
                if t in set(map_timesteps):
                    # append it to map
                    for e in mapvar_lst:
                        mapkeys_dct[e][mapid] = mapback_dct[e]
                    mapid = mapid + 1
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
                           'Temp':series['Temp'].values,
                           'IRA': series['IRA'].values,
                           'IRI': series['IRI'].values,
                           'PET':ts_pet,
                           'D':ts_d, 'Cpy':ts_cpy, 'TF':ts_tf,
                           'Sfs':ts_sfs, 'R':ts_r, 'RSE': ts_rse,
                           'RIE': ts_rie, 'RC': ts_rc, 'Inf':ts_inf,
                           'Unz':ts_unz, 'Qv':ts_qv, 'Evc':ts_evc,
                           'Evs':ts_evs, 'Tpun':ts_tpun, 'Tpgw':ts_tpgw,
                           'ET':ts_et, 'Qb':ts_qb,'Qs':ts_qs, 'Q':ts_q,
                           'Flow':ts_flow})
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


def topmodel_calib(series, shruparam, twibins, countmatrix, lamb, qt0, lat, area, basinshadow, m_range, qo_range, cpmax_range,
                   sfmax_range, erz_range, ksat_range, c_range, k_range, n_range, etpatdates, etpatzmaps,
                   tui=True, grid=200, generations=100, popsize=200, offsfrac=1,
                   mutrate=0.4, puremutrate=0.1, cutfrac=0.33, tracefrac=0.1, tracepop=True, metric='NSE'):
    # todo docstring (redo)
    from evolution import generate_population, generate_offspring, recruitment
    from analyst import nse, kge, rmse, pbias, frequency, error
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

    def clear_full_signal(array3d, count):
        out_lst = list()
        for i in range(len(array3d)):
            lcl_signal = extract_zmap_signal(array3d[i], count)
            out_lst.append(lcl_signal)
        out_array = np.array(out_lst)
        return out_array.flatten()
    #
    #
    # run setup
    runsize = generations * popsize * 2
    #
    # extract observed data
    series = clear_prec_by_dates(dseries=series, datesstr=etpatdates)
    sobs = series['Q'].values
    # get log10 of flow for calibration metrics
    loglim = 0.000001
    sobs_log = np.log10(sobs + (loglim * (sobs <= 0)))
    #
    # get etpat zmap signal
    etpat_zmaps_nd = np.array(etpatzmaps)
    #plt.imshow(etpat_zmaps_nd[0])
    #plt.show()
    sobs_etpat = clear_full_signal(etpat_zmaps_nd, countmatrix)
    sobs_etpat = sobs_etpat / np.max(sobs_etpat)  # normalize pattern by max value
    lcl_x = np.arange(0, len(sobs_etpat))
    #plt.plot(lcl_x, sobs_etpat)
    #plt.show()
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
            print('{:7} {:7} {:7} {:7} {:7} {:7} {:7} {:7} {:7}'.format('m', 'qo','cpmax', 'sfmax', 'erz', 'ksat','c', 'k', 'n'))
        # fit new population
        ids_lst = list()
        scores_lst = list()
        scores_flow_lst = list()
        scores_etpat_lst = list()
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
            sim_dct = topmodel_sim(series=series, shruparam=shruparam, twibins=twibins, countmatrix=countmatrix,
                                   lamb=lamb, qt0=qt0, m=pset[0], qo=pset[1], cpmax=pset[2], sfmax=pset[3], erz=pset[4],
                                   ksat=pset[5], c=pset[6], lat=lat, k=pset[7], n=pset[8],
                                   area=area, basinshadow=basinshadow, tui=False,
                                   qobs=True, mapback=True, mapvar='ET', mapdates=etpatdates)
            sim_df = sim_dct['Series']
            # compute Flow sim data
            ssim = sim_df['Q'].values
            ssim_log = np.log10(ssim + (loglim * (ssim <= 0)))
            #
            # Get fitness score for Flow:
            if metric == 'NSE':
                lcl_flow_score = nse(obs=sobs, sim=ssim)
            elif metric == 'NSElog':
                lcl_flow_score = nse(obs=sobs_log, sim=ssim_log)
            elif metric == 'KGE':
                lcl_flow_score = kge(obs=sobs, sim=ssim)
            elif metric == 'KGElog':
                lcl_flow_score = kge(obs=sobs_log, sim=ssim_log)
            elif metric == 'RMSE':
                lcl_flow_score = 1 - (1 / (rmse(obs=sobs, sim=ssim) + loglim))
            elif metric == 'RMSElog':
                lcl_flow_score = 1 - (1 / (rmse(obs=sobs_log, sim=ssim_log) + loglim))
            elif metric == 'PBias':
                lcl_flow_score = 1 / (np.abs(pbias(obs=sobs, sim=ssim)) + loglim)
            elif metric == 'RMSE-CFC':
                cfc_obs = frequency(series=sobs)['Values']
                cfc_sim = frequency(series=ssim)['Values']
                lcl_flow_score = 1 - (1 / (rmse(obs=cfc_obs, sim=cfc_sim) + loglim))
            elif metric == 'RMSElog-CFC':
                cfc_obs = frequency(series=sobs)['Values']
                cfc_sim = frequency(series=ssim)['Values']
                lcl_flow_score = 1 - (1 / (rmse(obs=np.log10(cfc_obs), sim=np.log10(cfc_sim)) + loglim))
            else:
                lcl_flow_score = nse(obs=sobs, sim=ssim)
            #
            # extract ETpat array:
            et_zmaps_nd = np.array(sim_dct['Maps']['ET'])
            #plt.imshow(et_zmaps_nd[2])
            #plt.show()
            ssim_etmaps =  clear_full_signal(et_zmaps_nd, countmatrix)
            ssim_etpat = ssim_etmaps / np.max(ssim_etmaps)  # Normalize simulated signal
            #plt.plot(lcl_x, ssim_etpat)
            #plt.plot(lcl_x, sobs_etpat)
            #plt.show()
            #
            # get fitness score for the ET pattern:
            lcl_etpat_score = kge(obs=sobs_etpat, sim=ssim_etpat)
            #
            #
            #
            # COMPUTE GLOBAL SCORE:
            w_flow = len(sobs)
            w_etpat = len(etpatzmaps)
            dx = (1 - lcl_flow_score)
            dy = (1 - lcl_etpat_score) * (w_etpat / w_flow)
            euclid_d = np.sqrt(np.power(dx, 2) + np.power(dy, 2))  # euclidean distance
            lcl_dna_score = 1 - euclid_d
            #
            #
            #
            #
            # printing
            if tui:
                print('Status: {:8.4f} % | Set '.format(runstatus), end='\t')
                print('{:7.3f} {:7.3f} {:7.3f} {:7.3f} {:7.3f} {:7.3f} {:7.3f} {:7.3f} {:7.3f}'.format(pset[0], pset[1],
                                                                                                       pset[2], pset[3],
                                                                                                       pset[4], pset[5],
                                                                                                       pset[6], pset[7],
                                                                                                       pset[8]), end='   ')
                print(' | Score: {:8.3f} | Flow: {:8.3f} | ETpat: {:8.3f}'.format(lcl_dna_score, lcl_flow_score, lcl_etpat_score))
            #
            #
            lcl_dna_id = 'G' + str(g + 1) + '-' + str(i)
            #
            # store in retrieval system:
            pop_dct[lcl_dna_id] = lcl_dna
            ids_lst.append(lcl_dna_id)
            scores_lst.append(lcl_dna_score)
            scores_flow_lst.append(lcl_flow_score)
            scores_etpat_lst.append(lcl_etpat_score)
            if tracepop:
                dnas_lst.append(lcl_dna)
        #
        # trace full population
        if tracepop:
            trace_pop.append({'DNAs': dnas_lst[:], 'Ids': ids_lst[:], 'Scores': scores_lst[:],
                              'FlowScores':scores_flow_lst[:], 'EtpatScores':scores_etpat_lst[:]})
        #
        # rank new population (Survival)
        df_population_rank = pd.DataFrame({'Id': ids_lst, 'Score': scores_lst,
                                           'FlowScores':scores_flow_lst[:], 'EtpatScores':scores_etpat_lst[:]})
        df_population_rank.sort_values(by='Score', ascending=False, inplace=True)
        #
        # Selection of mating pool
        df_parents_rank = df_population_rank.nlargest(len(parents), columns=['Score'])
        #
        parents_ids = df_parents_rank['Id'].values  # numpy array of string IDs
        parents_scores = df_parents_rank['Score'].values  # numpy array of float scores
        parents_scores_flow = df_parents_rank['FlowScores'].values
        parents_scores_etpat = df_parents_rank['EtpatScores'].values
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
        trace.append({'DNAs': parents[:tr_len], 'Ids': parents_ids[:tr_len], 'Scores': parents_scores[:tr_len],
                      'FlowScores': parents_scores_flow[:tr_len], 'EtpatScores':parents_scores_etpat[:tr_len]})
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

