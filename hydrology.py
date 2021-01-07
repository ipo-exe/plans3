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
    #avg_di = avg_2d(lcl_di, aoi)
    lcl_di = ((lcl_di > 0) * 1) * lcl_di  # set negative deficit to zero
    return lcl_di


def topmodel_vsai(di):
    return ((di == 0) * 1)


def topmodel_vsa(vsai, aoi, cellsize):
    lcl_vsa = np.sum(vsai * aoi) * cellsize * cellsize
    return lcl_vsa


def avg_2d(var, aoi):
    lcl_avg = np.sum(var * aoi) / np.sum(aoi)  # avg temp
    return lcl_avg


def get_srzi(srzi, pi, peti, srzmaxi):
    # increment
    lcl_s1 = srzi + pi
    # et and pet balance
    lcl_et = lcl_s1 - peti  # discount
    lcl_et = peti * (lcl_et >= 0.0) + lcl_s1 * (lcl_et < 0.0)  # positive condition
    lcl_s2 = lcl_s1 - lcl_et  #
    lcl_pet = peti - lcl_et
    # final storage balance
    lcl_s = srzmaxi * (lcl_s2 >= srzmaxi) + lcl_s2 * (lcl_s2 < srzmaxi)
    # excess balance
    lcl_excs = (lcl_s2 - srzmaxi) * (lcl_s2 >= srzmaxi) + 0.0 * (lcl_s2 < srzmaxi)
    return lcl_s, lcl_et, lcl_pet, lcl_excs


def get_suzi(suzi, srzi_ex, peti, suzimaxi, k):
    # increment
    lcl_s1 = suzi + srzi_ex
    # runoff balance
    lcl_rff = 0.0 * (lcl_s1 < suzimaxi) + (lcl_s1 - suzimaxi) * (lcl_s1 >= suzimaxi)
    # balance
    lcl_s2 = lcl_s1 * (lcl_s1 < suzimaxi) + suzimaxi * (lcl_s1 >= suzimaxi)
    # et and peti balance
    lcl_et = lcl_s2 - peti
    lcl_et = peti * (lcl_et >= 0.0) + lcl_s2 * (lcl_et < 0.0)  # positive condition
    lcl_s3 = lcl_s2 - lcl_et
    lcl_pet = peti - lcl_et
    # recharge balance
    suzimaxi1 = suzimaxi + (2.0 + (suzimaxi <=0))
    lcl_qvi1 = lcl_s3 / (suzimaxi1 * k)
    lcl_qvi = lcl_qvi1 * (suzimaxi > 0)
    # final storage balance
    lcl_s = lcl_s3 - lcl_qvi
    return lcl_s, lcl_rff, lcl_et, lcl_pet, lcl_qvi



def topmodel(series, twi, aoi, ppat, tpat, cellsize, qt0, qo, m, k, trace=False, tracestep=10):
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
    area = np.sum(aoi) * cellsize * cellsize
    spflow = convert_q2sq(q=df_ts['Flow'].values, area=area)
    #
    # prec array
    ts_p = np.zeros(size)
    # temp array
    ts_t = np.zeros(size)
    # pet array
    ts_pet = np.zeros(size)
    # et array
    ts_et = np.zeros(size)
    # output array
    ts_out = np.zeros(size)
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
    # get lambda
    lamb = topmodel_lambda(twi=twi, aoi=aoi)
    print('Lambda: {}'.format(lamb))
    #
    # recharge array:
    ts_qv = np.zeros(size)
    di_now = topmodel_di(d=d0, twi=twi, m=m, lamb=lamb)
    vsai_now = topmodel_vsai(di_now)
    qv0 = 4
    ts_qv[0] = qv0
    #
    # variable source area array:
    ts_vsa = np.zeros(size)
    ts_vsa[0] = topmodel_vsa(vsai_now, aoi, cellsize)
    print('VSA = {} km2'.format(ts_vsa[0]/ (1000 * 1000)))
    #
    # runoff array
    ts_rff = np.zeros(size)
    rff_now = twi.copy() * 0.0  # 2d runoff
    #
    # Suz array
    ts_suz = np.zeros(size)
    suzi_now = twi.copy() * 0.0  # 2d suzi
    ts_suz[0] = avg_2d(suzi_now, aoi)  # avg suz
    #
    # Srz array
    ts_srz = np.zeros(size)
    srzi_now = twi.copy() * 0.0  # 2d srzi
    srzmaxi = twi.copy() * 0.0 + 4.0  # todo method for estimate srzmaxi by CN
    ts_srz[0] = avg_2d(srzi_now, aoi)  # avg srz
    #
    for t in range(1, size):
        print('Step: {}'.format(t))
        # reset last 2d arrays
        di_last = di_now.copy()
        vsai_last = vsai_now.copy()
        rff_last = rff_now.copy()
        suzi_last = suzi_now.copy()
        srzi_last = srzi_now.copy()
        vsai_last = vsai_now.copy()
        #
        # 2d prec
        pattern = ppat[ts_month[t - 1] - 1].copy()
        pi_last = ts_pobs[t - 1] * pattern
        #plt.imshow(di_last, cmap='Blues')
        #plt.show()
        ts_p[t - 1] = avg_2d(pi_last, aoi)  # avg prec
        #
        # 2d temp
        pattern = tpat[ts_month[t - 1] - 1].copy()
        ti_last = ts_tobs[t - 1] * pattern
        ts_t[t - 1] = avg_2d(ti_last, aoi)  # avg temp
        #
        # 2d pet
        peti_last = ti_last.copy() * 0.1  # todo function to get 2d pet
        ts_pet[t - 1] = avg_2d(peti_last, aoi)  # avg pet
        #
        # 2d root zone accounting
        srzi_now, et1i_last, peti_last, srzi_ex_last = get_srzi(srzi_last, pi_last, peti_last, srzmaxi)
        ts_srz[t] = avg_2d(srzi_now, aoi)
        #
        # 2d unsaturated zone accounting
        suzi_now, rff_last, et2i_last, peti_last, qvi_last = get_suzi(suzi_last, srzi_ex_last, peti_last, di_last, k=k)
        ts_suz[t] = avg_2d(suzi_now, aoi)
        ts_rff[t - 1] = avg_2d(rff_last, aoi)
        ts_qv[t - 1] = avg_2d(qvi_last, aoi)
        #
        # 2d saturated zone accounting
        ts_d[t] = ts_d[t - 1] + ts_qb[t - 1] - ts_qv[t - 1]
        #
        ts_qb[t] = topmodel_qb(d=ts_d[t], qo=qo, m=m)
        #
        di_now = topmodel_di(d=ts_d[t], twi=twi, m=m, lamb=lamb)
        vsai_now = topmodel_vsai(di_now)
        ts_vsa[t] = topmodel_vsa(vsai_now, aoi, cellsize=cellsize)
        #
        # ET accounting
        eti_last = et1i_last + et2i_last
        ts_et[t - 1] = avg_2d(eti_last, aoi)
        #
        # output accounting
        ts_out[t - 1] = ts_rff[t - 1] + ts_qb[t - 1] + ts_et[t - 1]
    # discharge
    ts_qsurf = convert_sq2q(ts_rff, area)
    ts_qbase = convert_sq2q(ts_qb, area)
    ts_spq = ts_qb + ts_rff
    ts_q = ts_qsurf + ts_qbase
    exp_dct = {'Date':df_ts['Date'], 'Step':steps,
               'Prec':ts_p, 'Temp':ts_t, 'PET':ts_pet,
               'D':ts_d, 'VSA':ts_vsa, 'Qv':ts_qv, 'Qb':ts_qb,
               'R':ts_rff, 'Qsp':ts_spq, 'Q':ts_q, 'Qsurf':ts_qsurf, 'Qbase':ts_qbase,
               'ET':ts_et, 'Out':ts_out,
               'Suz':ts_suz, 'Srz':ts_srz}
    exp_df = pd.DataFrame(exp_dct)
    return exp_df






