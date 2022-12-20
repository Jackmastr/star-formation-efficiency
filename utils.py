import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import csv

from scipy.integrate import trapz
from scipy.interpolate import interp1d
import scipy.stats as sps
from astropy import constants as const, units as u
from astropy.cosmology import Planck18 as cosmo

LINESTYLE_ARR = ['-', '--', ':', '-.']
ONE_SIGMA = sps.norm.cdf(-1)

def my_mpl():
    plt.rc('font', family='serif', size=20)
    plt.rc('axes', grid=True)
    plt.rc('lines', lw=3)
    ts = 8
    plt.rc('xtick.minor', size=ts-2)
    plt.rc('ytick.minor', size=ts-2)
    plt.rc('xtick.major', size=ts) 
    plt.rc('ytick.major', size=ts)

def to_array(x, extend=1):
    try:
        return np.array(list(iter(x)))
    except TypeError:
        return np.array(extend*[x])

A_HE = 1.22
Y_P = (4/3) * (1 - 1/1.22)
C_HII = 3
F_GAMMA = 4000
OMEGA_M = cosmo.Om0
OMEGA_L = cosmo.Ode0
OMEGA_B = cosmo.Ob0
H0 = cosmo.H0
RHO_CRIT = (cosmo.critical_density0).to('Msun/Mpc^3')
RHO_M = OMEGA_M * RHO_CRIT
M_H = const.m_p
N_H0 = ((1-Y_P) * OMEGA_B * RHO_CRIT / M_H).to('cm^-3')
KAPPA_UV = 1.15e-28 # u.Msun / u.yr / (u.erg * u.s**-1 * u.Hz**-1)

def dt_dz(z):
    return 1/(cosmo.H(z)*(1+z))

def m_to_L(m):
    log10_L = 0.4 * (51.6 - m)
    return 10**log10_L

def L_to_m(L):
    return 51.6 - 2.5 * np.log10(L)

def dL_dm(m):
    return 0.4*np.log(10) * m_to_L(m)

def loglog_interp(x, y, fill_value='extrapolate'):
    sgn_x = np.sign(x[0])
    sgn_y = np.sign(y[0])
    if sgn_x != sgn_y:
        raise NotImplementedError
    logx = np.log(sgn_x*x)
    logy = np.log(sgn_y*y)
    lin_interpolator = interp1d(logx, logy, fill_value=fill_value)
    loglog_interpolator = lambda z: sgn_y * np.exp(lin_interpolator(np.log(sgn_x*z)))
    return loglog_interpolator

def loglin_interp(x, y, fill_value='extrapolate'):
    sgn_x = np.sign(x[0])
    sgn_y = np.sign(y[0])
    #if sgn_x != sgn_y:
    #    raise NotImplementedError
    logx = np.log(sgn_x*x)
    lin_interpolator = interp1d(logx, y, fill_value=fill_value)
    loglin_interpolator = lambda z: lin_interpolator(np.log(sgn_x*z))
    return loglin_interpolator

def linlog_interp(x, y, fill_value='extrapolate'):
    sgn_x = np.sign(x[0])
    sgn_y = np.sign(y[0])
    #if sgn_x != sgn_y:
    #    raise NotImplementedError
    logy = np.log(sgn_y*y)
    lin_interpolator = interp1d(x, logy, fill_value=fill_value)
    linlog_interpolator = lambda z: sgn_y * np.exp(lin_interpolator(z))
    return linlog_interpolator

def trapz_integrate(integrand, lo, hi, n_intervals=100, logspace=False):
    integrand_fn = integrand
    if logspace:
        lo, hi = np.log(lo), np.log(hi)
        integrand_fn = lambda logx: np.exp(logx) * integrand(np.exp(logx))
    x = np.linspace(lo, hi, n_intervals)
    y = integrand_fn(x)
    dx = x[1] - x[0]
    return trapz(y=y, x=x, dx=dx)

def chi_squared(measured, theory, sigma_plus, sigma_minus=None):
    sq_dif = (measured-theory)**2
    if sigma_minus is None: # symmetric error
        sigma_minus = sigma_plus
    if sigma_minus == 0: # upper bound only so assume half-gaussian
        half_normal_factor = np.sqrt(2)
        return sq_dif / (half_normal_factor * sigma_plus)**2
    sigma = 2 * sigma_plus * sigma_minus / (sigma_plus+sigma_minus)
    sigma_prime = (sigma_plus-sigma_minus)/(sigma_plus+sigma_minus)
    return sq_dif / (sigma + sigma_prime*(theory-measured))**2


