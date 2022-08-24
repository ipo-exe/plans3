''' 
PLANS - Planning Nature-based Solutions

Module description:
This module stores all basic analyst functions of plans3.

Copyright (C) 2022 Iporã Brito Possantti

************ GNU GENERAL PUBLIC LICENSE ************

https://www.gnu.org/licenses/gpl-3.0.en.html

Permissions:
 - Commercial use
 - Distribution
 - Modification
 - Patent use
 - Private use
Conditions:
 - Disclose source
 - License and copyright notice
 - Same license
 - State changes
Limitations:
 - Liability
 - Warranty

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

'''

import numpy as np
import pandas as pd


def error(obs, sim):
    """
    Error function
    :param obs: float, int or numpy array of Observerd data
    :param sim: float, int or numpy array of Simulated data
    :return: float, int, or numpy array of Error signal
    """
    return obs - sim


def sq_error(obs, sim):
    """
    Squared Error function
    :param obs: float, int or numpy array of Observerd data
    :param sim: float, int or numpy array of Simulated data
    :return: float, int, or numpy array of Squared Error signal
    """
    return np.square(error(obs=obs, sim=sim))


def mse(obs, sim):
    """
    Mean Squared Error (MSE) function
    :param obs: float, int or numpy array of Observerd data
    :param sim: float, int or numpy array of Simulated data
    :return: float value of MSE
    """
    return np.mean(sq_error(obs=obs, sim=sim))


def rmse(obs, sim):
    """
    Root of Mean Squared Error (RMSE) function
    :param obs: float, int or numpy array of Observerd data
    :param sim: float, int or numpy array of Simulated data
    :return: float value of RMSE
    """
    return np.sqrt(mse(obs=obs, sim=sim))


def nrmse(obs, sim):
    """
    Normalized RMSE by the mean observed value
    :param obs: float, int or numpy array of Observerd data
    :param sim: float, int or numpy array of Simulated data
    :return: float value of NRMSE
    """
    return rmse(obs=obs, sim=sim) / np.mean(obs)


def nse(obs, sim):
    """
    Nash-Sutcliffe Efficiency (NSE) coeficient
    :param obs: numpy array of Observerd data
    :param sim: numpy array of Simulated data
    :return: float of NSE
    """
    qmean = np.mean(obs)
    se_sim = sq_error(obs=obs, sim=sim)
    se_mean = sq_error(obs=obs, sim=qmean)
    return 1 - (np.sum(se_sim) / np.sum(se_mean))


def nnse(obs, sim):
    """
    Normalized NSE function (NSE re-scaled to [0,1])
    :param obs: numpy array of Observerd data
    :param sim: numpy array of Simulated data
    :return: float of NNSE
    """
    return 1 / (2 - nse(obs=obs, sim=sim))


def kge(obs, sim):
    """
    Kling-Gupta Efficiency (KGE) coeficient Gupta et al. (2009)

    KGE = 1 - sqroot( (r - 1)^2 + (sd_sim/sd_obs - 1)^2 + (m_sim/m_obs - 1)^2)

    - Correlation
    - Dispersion
    - Mean value

    :param obs: numpy array of Observerd data
    :param sim: numpy array of Simulated data
    :return: float of KGE
    """
    linmodel = linreg(obs=obs, sim=sim)
    r = linmodel['R']
    sd_obs = np.std(obs)
    sd_sim = np.std(sim)
    mean_obs = np.mean(obs)
    mean_sim = np.mean(sim)
    return 1 - np.sqrt(np.square(r - 1) + np.square((sd_sim/sd_obs) - 1) + np.square((mean_sim/mean_obs) - 1))


def pbias(obs, sim):
    """
    P-Bias function
    Negative P-Bias ->> Too much water! - ET have to work harder to remove water from the system
    Positive P-Bias ->> Too less water! -  ET is draining too much water
    :param obs: numpy array of Observerd data
    :param sim: numpy array of Simulated data
    :return: float of P-bias in % (0 to 100)
    """
    return 100 * (np.sum(error(obs=obs, sim=sim))/np.sum(obs))


def linreg(obs, sim):
    """
    Linear regression model function
    :param obs: 1-d numpy array of Observerd data
    :param sim: 1-d numpy array of Simulated data
    :return: dictionary object of linear model y=Ax+B parameters:
    A, B, R, P, SD
    """
    from scipy.stats import linregress
    a, b, r, p, sd = linregress(x=obs, y=sim)
    return {'A':a, 'B':b, 'R':r, 'P':p, 'SD':sd}


def frequency(dataframe, var_field, zero=True, step=1):
    """

    This fuction performs a frequency analysis on a given time series.

    :param dataframe: pandas DataFrame object with time series
    :param var_field: string of variable field
    :param zero: boolean control to consider values of zero. Default: True
    :return: pandas DataFrame object with the following columns:

     'Pecentiles' - percentiles in % of array values (from 0 to 100 by steps of 1%)
     'Exceedance' - exeedance probability in % (reverse of percentiles)
     'Frequency' - count of values on the histogram bin defined by the percentiles
     'Probability'- local bin empirical probability defined by frequency/count
     'Values' - values percentiles of bins

    """
    # get dataframe right
    in_df = dataframe[[var_field]].copy()
    in_df = in_df.dropna()
    if zero:
        pass
    else:
        mask = in_df[var_field] != 0
        in_df = in_df[mask]
    def_v = in_df[var_field].values
    ptles = np.arange(0, 100 + step, step)
    cfc = np.percentile(def_v, ptles)
    exeed = 100 - ptles
    freq = np.histogram(def_v, bins=len(ptles))[0]
    prob = freq/np.sum(freq)
    out_dct = {'Percentiles': ptles, 'Exceedance':exeed, 'Frequency': freq, 'Probability': prob, 'Values':cfc}
    out_df = pd.DataFrame(out_dct)
    return out_df


def flow(obs, sim, loglim=0.00001):
    #
    # get log
    _sim_log = np.log10(sim + (loglim * (sim <= 0)))
    _obs_log = np.log10(obs + (loglim * (obs <= 0)))
    #
    # get cfcs
    _cfc_obs = frequency(dataframe=pd.DataFrame({'Obs':obs}), var_field='Obs')['Values']
    try:
        _cfc_sim = frequency(dataframe=pd.DataFrame({'Sim':sim}), var_field='Sim')['Values']
    except ValueError:
        #print('Value Error found in simulated CFC')
        _cfc_sim = np.ones(shape=np.shape(_cfc_obs))
    #
    # get stats
    _dct_stats = dict()
    _dct_stats['Q_sum'] = np.sum(sim)
    _dct_stats['Q_mean'] = np.mean(sim)
    _dct_stats['Q_sd'] = np.std(sim + (loglim * (sim <= 0)))
    #
    # get metrics
    _dct_metrics = dict()
    _dct_metrics['NSE'] = nse(obs=obs, sim=sim)
    _dct_metrics['NSElog'] = nse(obs=_obs_log, sim=_sim_log)
    _dct_metrics['KGE'] = kge(obs=obs, sim=sim)
    _dct_metrics['KGElog'] = kge(obs=_obs_log, sim=_sim_log)
    _dct_metrics['RMSE'] = rmse(obs=obs, sim=sim)
    _dct_metrics['RMSElog'] = rmse(obs=_obs_log, sim=_sim_log)
    _dct_metrics['NRMSE'] = nrmse(obs=obs, sim=sim)
    _dct_metrics['NRMSElog'] = nrmse(obs=_obs_log, sim=_sim_log)
    _dct_metrics['PBias'] = pbias(obs=obs, sim=sim)
    _dct_metrics['RMSE_CFC'] = rmse(obs=_cfc_obs, sim=_cfc_sim)
    _dct_metrics['RMSElog_CFC'] = rmse(obs=np.log10(_cfc_obs), sim=np.log10(_cfc_sim))
    _dct_metrics['NRMSE_CFC'] = nrmse(obs=_cfc_obs, sim=_cfc_sim)
    _dct_metrics['NRMSElog_CFC'] = nrmse(obs=np.log10(_cfc_obs), sim=np.log10(_cfc_sim))
    #
    return {'Stats':_dct_stats, 'Metrics':_dct_metrics}


def zmaps(obs, sim, count, nodata=-1):
    """
    Compute the basics analysis of obs vs sim ZMAP

    :param obs: 2d numpy array of observed zmap
    :param sim: 2d numpy array of simulated zmap
    :param count: 2d numpy array of counting matrix (2d histogram)
    :param nodata: float of no data value
    :return: dict of analyst products
    """
    import geo
    # get full boolean mask
    mask = (obs != nodata) * (sim != nodata) * (count > 0) * 1.0
    # number of data points
    n = np.sum(mask)
    # map of errors
    error_map = (obs * mask) - (sim * mask)
    # map of squared errors
    sq_error_map = error_map * error_map
    #
    # get signals
    obs_sig = geo.flatten_clear(obs, mask)
    sim_sig = geo.flatten_clear(sim, mask)
    his_sig = geo.flatten_clear(count, mask)
    # compute means
    obs_mean = np.sum(obs_sig * his_sig) / np.sum(his_sig)
    sim_mean = np.sum(sim_sig * his_sig) / np.sum(his_sig)
    error_mean = error(obs_mean, sim_mean)
    # compute weighted error
    werror_map = error_map * count / np.sum(his_sig)
    #
    # compute error signals
    error_sig = error(obs_sig, sim_sig)
    sq_error_sig = sq_error(obs_sig, sim_sig)
    # compute
    werror_sig = geo.flatten_clear(werror_map, mask)
    #
    # compute metrics
    mse_sig = mse(obs_sig, sim_sig)
    w_mse_sig = np.mean(werror_sig * werror_sig)
    w_rmse_sig = np.sqrt(w_mse_sig)
    rmse_sig = rmse(obs_sig, sim_sig)
    nse_sig = nse(obs_sig, sim_sig)
    kge_sig = kge(obs_sig, sim_sig)
    r_sig = linreg(obs_sig, sim_sig)

    return {'Maps':{'Error': error_map, 'SqError':sq_error_map, 'WError':werror_map},
            'Signals':{'Obs':obs_sig, 'Sim':sim_sig, 'Error':error_sig, 'SqError':sq_error_sig, 'Count':his_sig, 'WError':werror_sig},
            'Metrics':{'N':n, 'MSE':mse_sig, 'RMSE':rmse_sig,
                       'W-MSE':w_mse_sig, 'W-RMSE':w_rmse_sig,
                       'NSE':nse_sig, 'R':r_sig['R'], 'KGE':kge_sig,
                       'Mean-Obs':obs_mean, 'Mean-Sim':sim_mean, 'Mean-Error':error_mean}}


def zmaps_lite(obs, sim, count, nodata=-1):
    """
    Compute the basics analysis of obs vs sim ZMAP no maps or signals returned

    :param obs: 2d numpy array of observed zmap
    :param sim: 2d numpy array of simulated zmap
    :param count: 2d numpy array of counting matrix (2d histogram)
    :param nodata: float of no data value
    :return: dict of analyst products
    """
    import geo
    # get full boolean mask
    mask = (obs != nodata) * (sim != nodata) * (count > 0) * 1.0
    # number of data points
    n = np.sum(mask)
    # get signals
    obs_sig = geo.flatten_clear(obs, mask)
    sim_sig = geo.flatten_clear(sim, mask)
    his_sig = geo.flatten_clear(count, mask)
    # compute means
    obs_mean = np.sum(obs_sig * his_sig) / np.sum(his_sig)
    sim_mean = np.sum(sim_sig * his_sig) / np.sum(his_sig)
    error_mean = error(obs_mean, sim_mean)
    # compute error signals
    error_sig = error(obs_sig, sim_sig)
    sq_error_sig = sq_error(obs_sig, sim_sig)
    # compute
    # map of errors
    error_map = (obs * mask) - (sim * mask)
    # compute weighted error
    werror_map = error_map * count / np.sum(his_sig)
    werror_sig = geo.flatten_clear(werror_map, mask)
    #
    # compute metrics
    mse_sig = mse(obs_sig, sim_sig)
    w_mse_sig = np.mean(werror_sig * werror_sig)
    w_rmse_sig = np.sqrt(w_mse_sig)
    rmse_sig = rmse(obs_sig, sim_sig)
    nse_sig = nse(obs_sig, sim_sig)
    kge_sig = kge(obs_sig, sim_sig)
    r_sig = linreg(obs_sig, sim_sig)
    #
    return {'Metrics':{'N':n, 'MSE':mse_sig, 'RMSE':rmse_sig,
                       'W-MSE':w_mse_sig, 'W-RMSE':w_rmse_sig,
                       'NSE':nse_sig, 'R':r_sig['R'], 'KGE':kge_sig,
                       'Mean-Obs':obs_mean, 'Mean-Sim':sim_mean, 'Mean-Error':error_mean}}


def zmaps_series(obs, sim, count, nodata=-1, full_return=False):
    """
    Analyst of a series of ZMaps
    :param obs: 3d numpy array of observed zmaps
    :param sim: 3d numpy array of simulated zmaps
    :param count: 2d numpy array of couting matrix (2d histogram)
    :param nodata: float nodata value in observed zmap
    :param full_return: boolean to set full return (maps and signals)
    :return: dictionary of output objects
    """
    # deploy arrays
    _n = np.zeros(len(obs))
    _mse = np.zeros(len(obs))
    _rmse = np.zeros(len(obs))
    _w_mse = np.zeros(len(obs))
    _w_rmse = np.zeros(len(obs))
    _nse = np.zeros(len(obs))
    _r = np.zeros(len(obs))
    _kge = np.zeros(len(obs))
    _mean_obs = np.zeros(len(obs))
    _mean_sim = np.zeros(len(obs))
    _mean_error = np.zeros(len(obs))
    # get maps and signals
    if full_return:
        _maps_errors = np.zeros(shape=(len(obs), np.shape(obs[0])[0], np.shape(obs[0])[1]))
        _maps_sqerrors = np.zeros(shape=(len(obs), np.shape(obs[0])[0], np.shape(obs[0])[1]))
        _maps_werrors = np.zeros(shape=(len(obs), np.shape(obs[0])[0], np.shape(obs[0])[1]))
        _sigs_obs = list()
        _sigs_sim = list()
        _sigs_error = list()
        _sigs_sqerror = list()
        _sigs_werror = list()
        _sigs_hist = list()
    #
    # loop in maps
    for i in range(len(obs)):
        # analysis
        if full_return:
            lcl_dct = zmaps(obs[i], sim[i], count=count, nodata=nodata)
            # retrieve from dict
            _maps_errors[i] = lcl_dct['Maps']['Error']
            _maps_sqerrors[i] = lcl_dct['Maps']['SqError']
            _maps_werrors[i] = lcl_dct['Maps']['WError']
            _sigs_obs.append(lcl_dct['Signals']['Obs'])
            _sigs_sim.append(lcl_dct['Signals']['Sim'])
            _sigs_error.append(lcl_dct['Signals']['Error'])
            _sigs_sqerror.append(lcl_dct['Signals']['SqError'])
            _sigs_werror.append(lcl_dct['Signals']['WError'])
            _sigs_hist.append(lcl_dct['Signals']['Count'])
        else:
            lcl_dct = zmaps_lite(obs[i], sim[i], count=count, nodata=nodata)
        #
        # retrieve from local dict
        _n[i] = lcl_dct['Metrics']['N']
        _mse[i] = lcl_dct['Metrics']['MSE']
        _rmse[i] = lcl_dct['Metrics']['RMSE']
        _w_mse[i] = lcl_dct['Metrics']['W-MSE']
        _w_rmse[i] = lcl_dct['Metrics']['W-RMSE']
        _nse[i] = lcl_dct['Metrics']['NSE']
        _r[i] = lcl_dct['Metrics']['R']
        _kge[i] = lcl_dct['Metrics']['KGE']
        _mean_obs[i] = lcl_dct['Metrics']['Mean-Obs']
        _mean_sim[i] = lcl_dct['Metrics']['Mean-Sim']
        _mean_error[i] = lcl_dct['Metrics']['Mean-Error']
    # return metrics and maps and signals
    if full_return:
        return {'Metrics':
                    {'N': _n,
                     'MSE': _mse,
                     'RMSE': _rmse,
                     'W-MSE': _w_mse,
                     'W-RMSE': _w_rmse,
                     'NSE': _nse,
                     'KGE': _kge,
                     'R': _r,
                     'Mean-Obs': _mean_obs,
                     'Mean-Sim': _mean_sim,
                     'Mean-Error': _mean_error},
                'Stats':
                    {'MSE_mean': np.mean(_mse),
                     'MSE_min': np.min(_mse),
                     'MSE_max': np.max(_mse),
                     'RMSE_mean': np.mean(_rmse),
                     'RMSE_min': np.min(_rmse),
                     'RMSE_max': np.max(_rmse),
                     'W-MSE_mean': np.mean(_w_mse),
                     'W-MSE_min': np.min(_w_mse),
                     'W-MSE_max': np.max(_w_mse),
                     'W-RMSE_mean': np.mean(_w_rmse),
                     'W-RMSE_min': np.min(_w_rmse),
                     'W-RMSE_max': np.max(_w_rmse),
                     'NSE_mean': np.mean(_nse),
                     'NSE_min': np.min(_nse),
                     'NSE_max': np.max(_nse),
                     'KGE_mean': np.mean(_kge),
                     'KGE_min': np.min(_kge),
                     'KGE_max': np.max(_kge),
                     'R_mean': np.mean(_r),
                     'R_min': np.min(_r),
                     'R_max': np.max(_r),
                     'MeanObs_mean': np.mean(_mean_obs),
                     'MeanObs_min': np.min(_mean_obs),
                     'MeanObs_max': np.max(_mean_obs),
                     'MeanSim_mean': np.mean(_mean_sim),
                     'MeanSim_min': np.min(_mean_sim),
                     'MeanSim_max': np.max(_mean_sim),
                     'MeanErr_mean': np.mean(_mean_error),
                     'MeanErr_min': np.min(_mean_error),
                     'MeanErr_max': np.max(_mean_error)},
                'Maps':
                    {'Error':_maps_errors,
                     'SqError':_maps_sqerrors,
                     'WError':_maps_werrors},
                'Signals':
                    {'Obs':_sigs_obs,
                     'Sim':_sigs_sim,
                     'Error':_sigs_error,
                     'SqError':_sigs_sqerror,
                     'WError':_sigs_werror,
                     'Count':_sigs_hist}
                }
    # return only metrics series (arrays)
    else:
        return  {'Metrics':
                    {'N': _n,
                     'MSE': _mse,
                     'RMSE': _rmse,
                     'W-MSE': _w_mse,
                     'W-RMSE': _w_rmse,
                     'NSE': _nse,
                     'KGE': _kge,
                     'R': _r,
                     'Mean-Obs': _mean_obs,
                     'Mean-Sim': _mean_sim,
                     'Mean-Error': _mean_error},
                 'Stats':
                     {'MSE_mean': np.mean(_mse),
                      'MSE_min': np.min(_mse),
                      'MSE_max': np.max(_mse),
                      'RMSE_mean': np.mean(_rmse),
                      'RMSE_min': np.min(_rmse),
                      'RMSE_max': np.max(_rmse),
                      'W-MSE_mean': np.mean(_w_mse),
                      'W-MSE_min': np.min(_w_mse),
                      'W-MSE_max': np.max(_w_mse),
                      'W-RMSE_mean': np.mean(_w_rmse),
                      'W-RMSE_min': np.min(_w_rmse),
                      'W-RMSE_max': np.max(_w_rmse),
                      'NSE_mean': np.mean(_nse),
                      'NSE_min': np.min(_nse),
                      'NSE_max': np.max(_nse),
                      'KGE_mean': np.mean(_kge),
                      'KGE_min': np.min(_kge),
                      'KGE_max': np.max(_kge),
                      'R_mean': np.mean(_r),
                      'R_min': np.min(_r),
                      'R_max': np.max(_r),
                      'MeanObs_mean': np.mean(_mean_obs),
                      'MeanObs_min': np.min(_mean_obs),
                      'MeanObs_max': np.max(_mean_obs),
                      'MeanSim_mean': np.mean(_mean_sim),
                      'MeanSim_min': np.min(_mean_sim),
                      'MeanSim_max': np.max(_mean_sim),
                      'MeanErr_mean': np.mean(_mean_error),
                      'MeanErr_min': np.min(_mean_error),
                      'MeanErr_max': np.max(_mean_error)
                      }
                }
