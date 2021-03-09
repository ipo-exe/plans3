import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    Map back feature using a Z-Matrix
    :param zmatrix: 2d numpy array of z matrix of values
    :param a1: 2d numpy array reference array of rows
    :param a2: 2d numpy array reference array of columns
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
    :param day: julian day
    :return: solar radiation (float or array) in [W/m2]
    """
    return pet_gsc() * (1 + 0.033 * np.cos(2 * np.pi * day / 365))


def pet_declination(day):
    """
    PET model - Earth declination angle
    :param day: julian day
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
    PET Oudin Model - Radiation and Temperature based PET model of  Oudin et al (2005b)
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


def topmodel_sim(series, twihist, cnhist, countmatrix, lamb, ksat, m, qo, a, c, lat, qt0, k, n,
                 mapback=False, mapvar='R-ET-S1-S2', tui=False):
    """

    PLANS 3 TOPMODEL simulation procedure

    :param series: dataframe of input series. Required fields: 'Date', 'Prec', 'Temp'
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
    'Qv', 'R', 'ET', 'S1', 'S2', 'Inf', 'Tp', 'Ev', 'Tpgw' (see below the relation)
    :param tui: boolean to control terminal messages
    :return: dataframe of simulated variables:

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
    'Q': simulated streamflow, mm
    'VSA': simulated variable source area (saturated areas), in %
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
    #plt.imshow(s1maxi)
    #plt.title('s1max')
    #plt.show()
    s2maxi = 0.8 * s0max_bins * np.ones(shape=shape, dtype='float32')  # rootzone
    rzdi = s2maxi.copy()
    #plt.imshow(rzdi)
    #plt.title('s2max')
    #plt.show()

    lambi = np.reshape(twi_bins, (rows, 1)) * np.ones(shape=shape, dtype='float32')
    #
    # set 2d count variable arrays
    preci = ts_prec[0] * np.ones(shape=shape)
    peti = ts_pet[0] * np.ones(shape=shape)
    s1i = np.zeros(shape=shape)  # initial condition
    tfi = np.zeros(shape=shape)
    evi = np.zeros(shape=shape)
    #
    s2i = np.zeros(shape=shape)  # initial condition
    infi = np.zeros(shape=shape)
    ri = np.zeros(shape=shape)
    #peri = np.zeros(shape=shape)
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
    #ts_s3 = np.zeros(shape=np.shape(ts_prec), dtype='float32')
    #ts_s3[0] = avg_2d(var2d=s3i, weight=countmatrix)
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
    # ts_per = np.zeros(shape=np.shape(ts_prec), dtype='float32')
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
        if tui:
            print('Step {}'.format(t))
        #
        # update S1
        s1i = s1i - evi - tfi + preci
        ts_s1[t] = avg_2d(s1i, countmatrix)
        #
        # compute current EV
        peti = ts_pet[t] * np.ones(shape=shape)  # update PET
        evi = ((peti) * (s1i >= peti)) + (s1i * (s1i < peti))
        ts_ev[t] = avg_2d(evi, countmatrix)
        #
        # compute current TF
        preci = ts_prec[t] * np.ones(shape=shape)  # update PREC
        tfi = ((preci + s1i - evi - s1maxi) * ((preci + s1i - evi) >= s1maxi)) # + ((preci * 0.0) * ((preci + s1i - evi) < s1maxi))
        ts_tf[t] = avg_2d(tfi, countmatrix)
        #
        # update S2
        s2i = s2i - qvi - tpi + infi
        ts_s2[t] = avg_2d(s2i, countmatrix)
        #
        # compute TP
        peti = peti - evi  # update peti
        di_aux = di + ((np.max(di) + 3) * (di <= 0.0))  # auxiliar Di to replace zero values by a higher positive value to avoid division by zero
        ptpi = ((s2i * (rzdi >= di)) + ((s2i * rzdi / di_aux) * (rzdi < di))) * (di > 0.0)  # compute potential TP
        tpi = (ptpi * (peti >= ptpi)) + (peti * (peti < ptpi))
        ts_tp[t] = avg_2d(tpi, countmatrix)
        #
        # compute QV
        pqvi = (ksat * s2i / di_aux) * (di > 0.0)  # potential QV
        qvi = ((pqvi) * (s2i - tpi >= pqvi)) + ((s2i - tpi) * (s2i - tpi < pqvi))
        ts_qv[t] = avg_2d(qvi, countmatrix)
        #
        # compute Inf
        pinfi = (rzdi * (tfi >= rzdi)) + (tfi * (tfi < rzdi))  # potential infiltration -- infiltration capacity
        infi = ((di - s2i + tpi + qvi) * (pinfi >= (di - s2i + tpi + qvi))) + (pinfi * (pinfi < (di - s2i + tpi + qvi)))
        ts_inf[t] = avg_2d(infi, countmatrix)
        #
        # compute R
        ri = tfi - infi
        ts_r[t] = avg_2d(ri, countmatrix)
        #
        # compute TP-GW
        peti = peti - tpi  # remaining local PET
        ptpgwi = (s2maxi - di) * (di < s2maxi)  # potential local TP
        tpgwi = (peti * (ptpgwi > peti)) + (ptpgwi * (ptpgwi <= peti))
        ts_tpgw[t] = avg_2d(tpgwi, countmatrix)
        #
        # compute ET
        eti = evi + tpi + tpgwi
        ts_et[t] = avg_2d(eti, countmatrix)
        #
        # update D
        ts_d[t] = ts_d[t - 1] + ts_qb[t - 1] - ts_qv[t - 1] + ts_tpgw[t - 1]
        #
        # compute Qb
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
            dct = {'Qv':qvi, 'R': ri, 'ET':eti, 'S1':s1i, 'S2':s2i, 'Inf':infi, 'Tp':tpi, 'Ev':evi, 'Tpgw':tpgwi}
            for e in mapvar_lst:
                map_dct[e][t] = dct[e]
    #
    # RUNOFF ROUTING
    if tui:
        print('Runoff routing...')
    ts_qs = nash_cascade(ts_r, k=k, n=n)
    #
    # compute full discharge
    ts_q = ts_qb + ts_qs
    #
    # export data
    #
    exp_df = pd.DataFrame({'Date':series['Date'].values,
                           'Prec':series['Prec'].values,
                           'Temp':series['Temp'].values,
                           'PET': ts_pet,
                           'S1':np.round(ts_s1, 3), 'TF':np.round(ts_tf, 3), 'Ev':np.round(ts_ev, 3),
                           'S2':ts_s2, 'Inf':ts_inf, 'R':ts_r, 'Tp':ts_tp, 'Tpgw':ts_tpgw,  'ET':ts_et,
                           'D':ts_d, 'Qb': ts_qb, 'Qv':ts_qv, 'Qs':ts_qs, 'Q':ts_q, 'VSA':ts_vsa})
    #
    if mapback:
        return exp_df, map_dct
    else:
        return exp_df
