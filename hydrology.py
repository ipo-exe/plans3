import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def topmodel_y(twi, to, aoi):
    """
    Y = average value of [TWIi - ln(To)]
    :param twi:
    :param to:
    :param aoi:
    :return:
    """
    lcl_n = np.sum(aoi)
    lcl_diff = (twi - np.log(to)) * aoi
    lcl_y = lcl_diff/lcl_n
    return lcl_y


def topmodel_lambda(twi, aoi):
    lcl_n = np.sum(aoi)
    lcl_twi = twi * aoi
    lcl_lambda = np.sum(lcl_twi) / lcl_n
    return lcl_lambda


def topmodel_d0(qt0, qo, m):
    return - m * np.log(qt0/qo)


def topmodel_qb(d, qo, m):
    return qo * np.exp(-d/m)



def topmodel_di(d, twi, m, lamb):
    """
    local deficit di
    :param d: global deficit float
    :param twi: TWI 2d array
    :param to: To float
    :param m: float
    :param lamb: average value of TWIi
    :return: 2d array of local deficit
    """
    lcl_di = d + m * (lamb - twi)
    lcl_di = ((lcl_di > 0) * 1) * lcl_di  # set negative deficit to zero
    return lcl_di


def topmodel_vsai(di):
    return ((di == 0) * 1)


def topmodel_vsa(vsai, aoi, cellsize):
    lcl_vsa = np.sum(vsai * aoi) * cellsize * cellsize
    return lcl_vsa


def topmodel(series, twi, aoi, ppat, tpat, cellsize, qt0, qo, m, k):
    """

    :param series: pandas dataframe of the observed daily time series.
    Mandatory columns:
    'Date' - datetime in YYYY-MM-DD format
    'Prec' - precipitation in mm
    'Temp' - temperature in Celsius

    Warning: no gaps in series is allowed.

    :param twi: 2d numpy array of TWI
    :param aoi: 2d numpy array pseudo-boolean of watershed area (1 and 0)
    :param ppat: list of 12 2d numpy arrays of monthly precipitation pattern (multiplier)
    :param tpat: list of 12 2d numpy arrays of monthly temperature pattern (multiplier)
    :param cellsize:  cellsize in meters
    :param qt0: specific flow at t=0 (mm/d) -- must be lower than qo
    :param qo: maximum baseflow specific flow (mm/d)
    :param m: decay parameter in mm
    :return: dataframe of simulation
    """
    # todo end this
    # load dataframe
    df_ts = series.copy()
    df_ts['Date'] = pd.to_datetime(df_ts['Date'])  # enforce datetime
    # extract months
    df_ts['Month'] = df_ts['Date'].dt.month
    print(df_ts.head())
    #
    # load input data arrays
    ts_month = df_ts['Month'].values
    ts_pobs = df_ts['Prec'].values
    ts_tobs = df_ts['Temp'].values
    #
    # set size and steps
    size = len(ts_pobs)
    print('Size: {}'.format(size))
    steps = np.arange(0, size)
    #
    #
    # 2d prec
    ts_p = np.zeros(size)
    pattern = ppat[ts_month[0]].copy()
    pi = ts_pobs[0] * pattern
    ts_p[0] = np.sum(pi * aoi) / np.sum(aoi)
    #
    # 2d temp
    ts_t = np.zeros(size)
    pattern = tpat[ts_month[0]].copy()
    ti = ts_tobs[0] * pattern
    ts_t[0] = np.sum(ti * aoi) / np.sum(aoi)  # avg temp
    #
    # 2d pet
    ts_pet = np.zeros(size)
    peti = ti.copy() * 0.0  # todo function to get 2d pet
    ts_pet[0] = np.sum(peti * aoi) / np.sum(aoi)  # avg pet
    #
    # get lambda
    lamb = topmodel_lambda(twi=twi, aoi=aoi)
    print('Lambda: {}'.format(lamb))
    #
    # deficit array
    d0 = topmodel_d0(qt0=qt0, qo=qo, m=m)
    print('D0 = {} mm'.format(d0))
    ts_d = np.zeros(size)
    ts_d[0] = d0
    #
    # baseflow array
    ts_qb = np.zeros(size)
    ts_qb[0] = qt0
    #
    # recharge array:
    ts_qv = np.zeros(size)
    di = topmodel_di(d=d0, twi=twi, m=m, lamb=lamb)
    vsai = topmodel_vsai(di)
    plt.imshow(aoi)
    plt.show()
    qv0 = 4
    ts_qv[0] = qv0
    #
    # variable source area array:
    ts_vsa = np.zeros(size)
    ts_vsa[0] = topmodel_vsa(vsai, aoi, cellsize)
    print('VSA = {} km2'.format(ts_vsa[0]/ (1000 * 1000)))
    #
    # runoff array
    ts_rff = np.zeros(size)
    rffi = twi.copy() * 0.0  # 2d runoff
    ts_rff[0] = np.sum(rffi * aoi) / np.sum(aoi)  # avg rff
    #
    # Suz array
    ts_suz = np.zeros(size)
    suzi = twi.copy() * 0.0  # 2d suzi
    ts_suz[0] = np.sum(suzi * aoi) / np.sum(aoi)  # avg suz
    #
    # Srz array
    ts_srz = np.zeros(size)
    srzi = twi.copy() * 0.0  # 2d srzi
    ts_srz[0] = np.sum(srzi * aoi) / np.sum(aoi)  # avg srz
    #
    for t in range(1, 50):
        print('Step: {}'.format(t))
        #
        # 2d prec
        pattern = ppat[ts_month[t - 1]].copy()
        pi = ts_pobs[t] * pattern
        ts_p[t] = np.sum(pi * aoi) / np.sum(aoi)  # avg prec
        #
        # 2d temp
        pattern = tpat[ts_month[t - 1]].copy()
        ti = ts_tobs[t] * pattern
        ts_t[t] = np.sum(ti * aoi) / np.sum(aoi)  # avg temp
        #
        # 2d pet
        peti = ti.copy() * 0.0  # todo function to get 2d pet
        ts_pet[t] = np.sum(peti * aoi) / np.sum(aoi)  # avg pet
        #
        # todo 2d ESMA
        #
        # deficit accounting
        ts_d[t] = ts_d[t - 1] + ts_qb[t - 1] - ts_qv[t - 1]
        ts_qb[t] = topmodel_qb(d=ts_d[t], qo=qo, m=m)

    exp_dct = {'Date':df_ts['Date'], 'Step':steps,
               'Prec':ts_p, 'Temp':ts_t, 'PET':ts_pet,
               'D':ts_d, 'VSA':ts_vsa, 'Qb':ts_qb, 'Qv':ts_qv,
               'R':ts_rff, 'Suz':ts_suz, 'Srz':ts_srz}
    exp_df = pd.DataFrame(exp_dct)
    print(exp_df.head(30).to_string())






