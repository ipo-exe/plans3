import numpy as np


def area_trapezoidal(y, b, m):
    return (b + m * y) * y


def area_circular(y, r):
    alpha = np.arcsin((y - r) / r)
    print(alpha, end='\t')
    theta = ((np.pi / 2) + alpha)
    print(theta)
    return (1 / 8) * ((2 * theta) - np.sin((2 * theta))) * np.square(2 * r)


def hydraulic_radius(a, p):
    """
    computes the hydraulic radius
    :param a: wetted cross section in m2
    :param p: wetted perimeter in m
    :return: hydraulic radius in m
    """
    return a / p


def q_manning(n, a, rh, s):
    """
    Computes the discharge Q of the Manning Equation
    :param n: Manning roughness parameter
    :param a: wetted cross section in m2
    :param rh: hydraulic radius in m
    :param s: slope in m/m
    :return: dicharge in m3/s
    """
    return (a * np.power(x1=rh, x2=2/3) * np.power(x1=s, x2=0.5)) / n


def n_manning(q, a, rh, s):
    """
    Computes the Manning roughness parameter
    :param q: discharge Q of the Manning Equation in m3/s
    :param a: wetted cross section in m2
    :param rh: hydraulic radius in m
    :param s: slope in m/m
    :return: Manning roughness parameter
    """
    return (a * np.power(x1=rh, x2=2 / 3) * np.power(x1=s, x2=0.5)) / q


h = 6
r0 = 3
full_area = np.pi * np.square(r0)
print(full_area)
print(full_area/2)
area = area_circular(y=h, r=r0)
print(area)

