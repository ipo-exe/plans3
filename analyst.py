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
This module stores all basic analyst functions of plans3. 
Input parameters are all strings and booleans.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import geo


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


def frequency(series):
    """
    Frequency analysis on a given time series.
    :param series: 1-d numpy array
    :return: dictionary object with the following keys:
     'Pecentiles' - percentiles in % of array values (from 0 to 100 by steps of 1%)
     'Exeedance' - exeedance probability in % (reverse of percentiles: 100 - percentiles)
     'Frequency' - count of values on the histogram bin defined by the percentiles
     'Probability'- local bin empirical probability defined by frequency/count
     'Values' - values percentiles of bins
    """
    ptles = np.arange(0, 101, 1)  # create bins
    cfc = np.percentile(series, ptles)  # get percentiles
    exeed = 100 - ptles  # get exeedance probs.
    freq = np.histogram(series, bins=101)[0]  # get histogram
    prob = freq/np.sum(freq)  # estimate probs.
    out_dct = {'Percentiles': ptles, 'Exeedance':exeed, 'Frequency': freq, 'Probability': prob, 'Values':cfc}
    return out_dct


def zmaps(obs, sim, count, nodata=-1):
    """
    Compute the basics analysis of obs vs sim ZMAP

    :param obs: 2d numpy array of observed zmap
    :param sim: 2d numpy array of simulated zmap
    :param count: 2d numpy array of counting matrix (2d histogram)
    :param nodata: float of no data value
    :return: dict of analyst products
    """
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
    #
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
                }
