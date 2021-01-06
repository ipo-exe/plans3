import numpy as np
import pandas as pd


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
    di = d + m * (lamb - twi)
    return di


def topmodel(series, twi, aoi, ppat, tpat, cellsize, qt0, qo, m):
    """

    :param series: pandas dataframe of the observed daily time series.
    Mandatory columns:
    'Date' - datetime in YYYY-MM-DD format
    'Prec' - precipitation in mm
    'Temp' - temperature in Celsius

    Warning: no gaps in series is allowed.

    :param twi: 2d numpy array of TWI
    :param aoi: 2d numpy array pseudo-boolean of watershed area (1 and 0)
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
    # load arrays
    dates = df_ts['Date'].values
    prec = df_ts['Prec'].values
    temp = df_ts['Temp'].values
    size = len(prec)
    steps = np.arange(0, size)
    #  deficit array
    d0 = topmodel_d0(qt0=qt0, qo=qo, m=m)
    ts_d = np.zeros(size)
    ts_d[0] = d0
    #  baseflow array
    ts_qb = np.zeros(size)
    ts_qb[0] = qt0
    # recharge array:
    ts_qv = np.zeros(size) + 1.0 # constant recharge
    for t in range(1, size):
        ts_d[t] = ts_d[t - 1] + ts_qb[t - 1] - ts_qv[t - 1]
        ts_qb[t] = topmodel_qb(d=ts_d[t], qo=qo, m=m)
    return {'Step':steps, 'D':ts_d, 'Qb':ts_qb, 'Qv':ts_qv}






