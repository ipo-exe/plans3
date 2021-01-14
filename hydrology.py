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
    #
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

# topmpdel functions
def topmodel_s0max(cn, a):
    """
    Plans 3 Model of S0max as a function of CN
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


def topmodel_sim(series, twihist, cnhist, countmatrix, lamb, ksat, m, qo, a, c, qt0, k=1.5, n=1.5):
    """
    PLANS 3 TOPMODEL simulation
    :param series: dataframe of input series. Required fields: 'Date', 'Prec', 'Temp'
    :param twihist: tuple of histogram of TWI
    :param cnhist: tuple of histogram of CN
    :param countmatrix: 2D histogram of TWI and CN
    :param lamb: positive float - average TWI value of the AOI
    :param ksat: positive float - effective saturated hydraulic conductivity in mm/d
    :param m: positive float - effective transmissivity decay coefficient in mm
    :param qo: positive float - max baseflow when d=0 in mm/d
    :param a: positive float - scaling parameter for S0max model
    :param c: positive float - scaling parameter for PET model
    :param qt0: positive float - baseflow at t=0 in mm/d
    :param k: positive float - Nash Cascade residence time in days
    :param n: positive float - equivalent number of reservoirs in Nash Cascade
    :return: dataframe of simulated variables
    """
    #
    # extract data input
    ts_prec = series['Prec'].values
    ts_temp = series['Temp'].values
    size = len(ts_prec)
    #
    # compute PET
    ts_pet = c * ts_temp
    #
    # set initial conditions
    d0 = topmodel_d0(qt0=qt0, qo=qo, m=m)
    #
    twi_bins = twihist[0]
    twi_count = twihist[1]
    s0max_bins = topmodel_s0max(cn=cnhist[0], a=a)  # convert cn to S0max
    s0max_count = cnhist[1]
    shape = np.shape(countmatrix)
    rows = shape[0]
    #
    # set 2d count parameter arrays
    s1maxi = 0.2 * s0max_bins * np.ones(shape=shape, dtype='float32')
    s2maxi = 0.8 * s0max_bins * np.ones(shape=shape, dtype='float32')
    lambi = np.reshape(twi_bins, (rows, 1)) * np.ones(shape=shape, dtype='float32')
    #
    # set 2d count variable arrays
    s1i = np.zeros(shape=shape)  # initial condition
    tfi = np.zeros(shape=shape)
    evi = np.zeros(shape=shape)
    #
    s2i = np.zeros(shape=shape)  # initial condition
    infi = np.zeros(shape=shape)
    ri = np.zeros(shape=shape)
    peri = np.zeros(shape=shape)
    tpi = np.zeros(shape=shape)
    eti = evi + tpi
    #
    s3i = np.zeros(shape=shape)  # initial condition
    di = topmodel_di(d=d0, twi=lambi, m=m, lamb=lamb)
    vsai = topmodel_vsai(di=di)
    qvi = np.zeros(shape=shape)
    #
    # set stocks time series arrays and initial conditions
    ts_s1 = np.zeros(shape=np.shape(ts_prec), dtype='float32')
    ts_s1[0] = avg_2d(var2d=s1i, weight=countmatrix)
    ts_s2 = np.zeros(shape=np.shape(ts_prec), dtype='float32')
    ts_s2[0] = avg_2d(var2d=s2i, weight=countmatrix)
    ts_s3 = np.zeros(shape=np.shape(ts_prec), dtype='float32')
    ts_s3[0] = avg_2d(var2d=s3i, weight=countmatrix)
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
    ts_et = np.zeros(shape=np.shape(ts_prec), dtype='float32')
    ts_r = np.zeros(shape=np.shape(ts_prec), dtype='float32')
    ts_per = np.zeros(shape=np.shape(ts_prec), dtype='float32')
    ts_inf = np.zeros(shape=np.shape(ts_prec), dtype='float32')
    ts_tf = np.zeros(shape=np.shape(ts_prec), dtype='float32')
    #
    # ESMA loop
    for t in range(1, size):
        #print('Step {}'.format(t))
        #
        # update S1
        s1i = s1i + ts_prec[t - 1] - tfi - evi
        ts_s1[t] = avg_2d(s1i, countmatrix)
        # compute current TF
        tfi = ((ts_prec[t] - (s1maxi - s1i)) * (ts_prec[t] > (s1maxi - s1i))) + (0.0 * (ts_prec[t] <= (s1maxi - s1i)))
        ts_tf[t] = avg_2d(tfi, countmatrix)
        # compute current EV
        evi = ((ts_pet[t] * np.ones(shape=shape)) * (s1i > ts_pet[t])) + (s1i * (s1i <= ts_pet[t]))
        ts_ev[t] = avg_2d(evi, countmatrix)
        #
        #
        # update S2
        s2i = s2i + infi - peri - tpi
        ts_s2[t] = avg_2d(s2i, countmatrix)
        # compute Inf
        infi = ((s2maxi - s2i) * ((tfi + s2i) > s2maxi)) + (tfi * ((tfi + s2i) <= s2maxi))
        ts_inf[t] = avg_2d(infi, countmatrix)
        # compute R
        ri = tfi - infi
        ts_r[t] = avg_2d(ri, countmatrix)
        # compute PER
        peri_potential = (s2i * ((ksat * s2i / s2maxi) > s2i)) + ((ksat * s2i / s2maxi) * ((ksat * s2i / s2maxi) <= s2i))
        peri = (di * (peri_potential > di)) + (peri_potential * (peri_potential <= di))
        ts_per[t] = avg_2d(peri, countmatrix)
        # compute TP
        peti_remain = ts_pet[t] - evi
        s2i_remain = s2i - peri
        tpi = (peti_remain * (s2i_remain > peti_remain)) + (s2i_remain * (s2i_remain <= peti_remain))
        ts_tp[t] = avg_2d(tpi, countmatrix)
        # compute ET
        eti = evi + tpi
        ts_et[t] = avg_2d(eti, countmatrix)
        #
        # update D
        ts_d[t] = ts_d[t - 1] + ts_qb[t - 1] - ts_qv[t - 1]
        # compute Qb
        ts_qb[t] = topmodel_qb(d=ts_d[t], qo=qo, m=m)
        # compute Di
        di = topmodel_di(d=ts_d[t], twi=lambi, m=m, lamb=lamb)
        # compute VSA
        vsai = topmodel_vsai(di=di)
        ts_vsa[t] = np.sum(vsai * countmatrix) / np.sum(countmatrix)
        #
        # Update S3
        s3i = s3i + peri - qvi
        ts_s3[t] = avg_2d(s3i, countmatrix)
        # compute Qv
        aux_const = np.max(di) + 3
        di_aux = di + (aux_const * (di <= 0.0))
        qvi = (di_aux != aux_const) * (((ksat * s3i / di_aux) * ((ksat * s3i / di_aux) < s3i)) + (s3i * ((ksat * s3i / di_aux) >= s3i)))
        ts_qv[t] = avg_2d(qvi, countmatrix)
    #
    # RUNOFF ROUTING
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
                           'S2':ts_s2, 'Inf':ts_inf, 'R':ts_r, 'Per':ts_per, 'Tp':ts_tp, 'ET':ts_et,
                           'D':ts_d, 'Qb': ts_qb, 'S3':ts_s3, 'Qv':ts_qv, 'Qs':ts_qs, 'Q':ts_q, 'VSA':ts_vsa})
    #
    return exp_df

