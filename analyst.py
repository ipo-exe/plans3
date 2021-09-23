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


def zmaps(obs, sim, count, nodata=-1, full_return=False):
    """
    Analyst of a series of ZMaps
    :param obs: 3d numpy array of observed zmaps
    :param sim: 3d numpy array of simulated zmaps
    :param count: 2d numpy array of couting matrix (2d histogram)
    :param nodata: float nodata value in observed zmap
    :param full_return: boolean to set full return
    :return: dictionary of output objects
    """
    obs_mask = 1.0 * (obs != nodata)  # get mask
    #
    # deploy series arrays
    obs_wmean_series = np.zeros(shape=(len(obs)))
    sim_wmean_series = np.zeros(shape=(len(obs)))
    # compute series averages
    for i in range(len(obs)):
        obs_wmean_series[i] = np.sum(obs[i] * obs_mask[i] * count / np.sum(count))
        sim_wmean_series[i] = np.sum(sim[i] * obs_mask[i] * count / np.sum(count))
    #import matplotlib.pyplot as plt
    #plt.plot(np.arange(start=0, stop=len(obs_wmean_series)), obs_wmean_series)
    #plt.plot(np.arange(start=0, stop=len(obs_wmean_series)), sim_wmean_series)
    #plt.show()
    obs_mean = np.mean(obs_wmean_series)
    # now compute error series
    e_wmean_series = error(obs=obs_wmean_series, sim=sim_wmean_series)
    se_wmean_series = sq_error(obs=obs_wmean_series, sim=sim_wmean_series)
    nse_wmean = nse(obs=obs_wmean_series, sim=sim_wmean_series)
    #
    # get the square of the weighed mean error
    we_mean = obs * 0.0
    we_sim = obs * 0.0
    for i in range(len(we_mean)):
        we_mean[i] = (obs[i] - obs_mean) * count / np.sum(count)
        we_sim[i] = (obs[i] - sim[i]) * count / np.sum(count)
    swe_mean = np.power(we_mean, 2)
    swe_sim = np.power(we_sim, 2)
    #
    # compute NSE
    nse_zmaps = 1 - (np.sum(swe_sim * obs_mask) / np.sum(swe_mean * obs_mask))
    if full_return:
        out_dct = {'Obs_Mean_Series':obs_wmean_series,
                   'Sim_Mean_Series':sim_wmean_series,
                   'Error_Mean_Series':e_wmean_series,
                   'SE_Mean_Series':se_wmean_series,
                   'NSE_Mean_Series':nse_wmean,
                   'Count':count,
                   'Error_ZMap_Series':we_sim,
                   'SE_ZMap_Series':swe_sim,
                   'NSE_ZMap_Series':nse_zmaps}
    else:
        out_dct = {'NSE_ZMap_Series':nse_zmaps, 'NSE_Mean_Series':nse_wmean}
    return out_dct









