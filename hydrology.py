import numpy as np



def convert_q2sq(q, area):
    """
    convert discharge to specific discharge
    :param q: 1d array of q in m3/s
    :param area: watershed area in m2
    :return: 1d array of Q in mm/d
    """
    lcl_sq = q * 1000 * 86400 / area
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


def topmodel(prec, temp, twi, aoi, to, m, sqo, srzmax, k, sqto, cellsize):
    y = topmodel_y(twi, to, aoi)
    print('Y = {}'.format(y))
    area = np.sum(aoi) * cellsize * cellsize
    print('Area = {} m2'.format(area))






