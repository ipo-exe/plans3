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
    Kling-Gupta Efficiency (KGE) coeficient
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