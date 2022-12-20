import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker
import scipy.optimize
import pickle
import utils
import sys
sys.path.append('../')
from myfuncs import load_mf, load_pe
from scipy.interpolate import interp1d

import emcee, corner

class DustCorrector:
    def __init__(self, delta_m=0.5):
        self.sig_beta = 0.34
        self.c0 = 4.54
        self.c1 = 2.07
        self.c = -2.33
        self.m0 = -19.5
        self.beta_m0 = {
            5 : -1.91,
            6 : -2,
            7 : -2.05,
            8 : -2.13,
            }
        self.dbeta_dm0 = {
            5 : -0.14,
            6 : -0.2,
            7 : -0.2,
            8 : -0.15,
            }
        self.delta_m_old = delta_m ### TODO: NOT ALWAYS 0.5!!

    def beta(self, m, z):
        return self.dbeta_dm0[z] * (m + 19.5) + self.beta_m0[z]

    def A_uv(self, m, z):
        try:
            beta_val = self.beta(m, z)
        except KeyError:
            return 0
        A_val = (self.c0 + 0.2*np.log(10)*self.sig_beta**2*self.c1**2
                + self.c1*beta_val)
        return max(A_val, 0)

    def m_new(self, m, z):
        return m - self.A_uv(m, z)

    def delta_m_new(self, m, z):
        return (self.delta_m_old + self.A_uv(m - self.delta_m_old/2, z)
                - self.A_uv(m + self.delta_m_old/2, z))

    def phi_new(self, phi, m, z):
        return phi * self.delta_m_old / self.delta_m_new(m, z)

    def sigma_new(self, sigma, m, z):
        return sigma * self.delta_m_old / self.delta_m_new(m, z)

    def dodc(self, measurements):
        pz, pm, pv, pl, ph = measurements.T
        new_pz = pz
        new_pm = self.m_new(pm, pz)
        new_pv = self.phi_new(pv, pm, pz)
        new_pl = self.sigma_new(pl, pm, pz)
        new_ph = self.sigma_new(ph, pm, pz)
        dc_measurements = np.array([new_pz, new_pm, new_pv, new_pl, new_ph]).T
        return dc_measurements

class LuminosityFunctionPlotter:
    def __init__(self, models=[], measurement_groups=[], use_pts_from_models=False):
        self.models = utils.to_array(models)
        self.measurement_groups = utils.to_array(measurement_groups)
        if use_pts_from_models:
            for model in self.models:
                self.measurement_groups = np.append(self.measurement_groups, model.measurements)
        self.filenames = [m.filename for m in self.measurement_groups]
        self.n_groups = len(self.measurement_groups)

    def get_pts(self, dc=True):
        all_pts = {}
        dc = utils.to_array(dc, extend=self.n_groups)
        for i, m in enumerate(self.measurement_groups):
            m_pts = m.get_pts(dc=dc[i], by_z=True)
            if len(all_pts) == 0:
                all_pts = m_pts.copy()
            else:
                for z, point_info in m_pts.items():
                    try:
                        all_pts[z].extend(point_info)
                    except KeyError:
                        all_pts[z] = point_info
        return all_pts

    def plot_points(self, dc=True, n_col=1):
        points = self.get_pts(dc)
        num_z = len(points)
        n_row = int(np.ceil(num_z/n_col))
        layout = [n_row, n_col]
        fig, axes = plt.subplots(layout[0], layout[1], sharex='all', sharey='all', figsize=[16, 5*n_row])
        axs = np.ravel(axes, order='F')
        plt.subplots_adjust(hspace=0, wspace=0)
        for i, (z, point_info) in enumerate(sorted(points.items())):
            ax = axs[i]
            point_info = np.array(point_info).T
            mags = point_info[0]
            phis = point_info[1]
            los = point_info[2]
            his = point_info[3]
            label = point_info[4]
            isdc = point_info[5]
            for model in self.models:
                #fname = model.measurements.filename
                #model_label = 'B' + fname[fname.rfind('_')+1:fname.rfind('.')] + ' Fit'
                ax.plot(mags, [model.phi_m(mag, z) for mag in mags], label=model.name, lw=3)
            for k in range(len(mags)):
                lab='B'+label[k][-6:-4] + str('' if isdc[k] else ' no dc')
                mark = 'o' if isdc[k] else 'x'
                if label[k] == self.filenames[0]:
                    ax.errorbar(mags[k], phis[k], yerr=[[los[k]], [his[k]]], ls='', marker=mark, capsize=10, color='k', label=lab)
                elif label[k] == self.filenames[1]:
                    ax.errorbar(mags[k], phis[k], yerr=[[los[k]], [his[k]]], ls='', marker=mark, capsize=10, color='r', label=lab)
                elif label[k] == self.filenames[2]:
                    ax.errorbar(mags[k], phis[k], yerr=[[los[k]], [his[k]]], ls='', marker=mark, capsize=10, color='b', label=lab)
                else:
                    raise NotImplementedError
            ax.set_xscale('linear')
            ax.set_yscale('log')
            if i < n_row: 
                ax.set_ylabel(r'$\phi(m_{AB})$ [Mpc$^{-3}$ mag$^{-1}$]')
            ax.set_xlabel(r'$m_{AB}$ [mag]')
            title = r'$z=$' + str(int(z))
            ax.text(0.6, 0.85, title, horizontalalignment='center', transform=ax.transAxes, size='x-large')
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper left')

    def plot_lum_mass_rltn(self, M_vals, z_vals, figsize=[9,9]):
        fig, ax = plt.subplots(figsize=figsize)
        for i, z in enumerate(utils.to_array(z_vals)):
            c = 'C{}'.format(i)
            for k, model in enumerate(self.models):
                ls = utils.LINESTYLE_ARR[k]
                label = r'$z={}$ {}'.format(z, model.name)
                ax.plot(M_vals, [model.m_c(M, z) for M in M_vals], label=label, color=c, ls=ls)
        ax.set_xscale('log')
        ax.set_yscale('linear')
        ax.set_xlabel(r'$M_h$ [M$_\odot$]')
        ax.set_ylabel(r'$m_{UV}$ [mag]')
        ax.legend()
        ax2_func = lambda m: utils.KAPPA_UV * utils.m_to_L(m)
        ax2_inv_func = lambda sfr: utils.L_to_m(sfr/utils.KAPPA_UV)
        ax2 = ax.secondary_yaxis(location='right', functions=(ax2_func, ax2_inv_func))
        ax2.set_ylabel(r'SFR [M$_\odot$ yr$^{-1}$]')
        ax2.set_yscale('log')
        ax2.invert_yaxis()
        #ax2.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1e'))
        ax2.grid(None)

    def plot_LM(self, z_vals, figsize=[9,9]):
        func = LuminosityFunctionModel.m_c
        ax_scale = ['log', 'linear']
        ax_lab = [r'$M_h$ [M$_\odot$]', r'$m_{UV}$ [mag]']
        def sfr_axis(ax):
            ax2_func = lambda m: utils.KAPPA_UV * utils.m_to_L(m)
            ax2_inv_func = lambda sfr: utils.L_to_m(sfr/utils.KAPPA_UV)
            ax2 = ax.secondary_yaxis(location='right', functions=(ax2_func, ax2_inv_func))
            ax2.set_ylabel(r'SFR [M$_\odot$ yr$^{-1}$]')
            ax2.set_yscale('log')
            ax2.invert_yaxis()
        self.plot_model_func(z_vals, func, ax_scale, ax_lab, figsize, second_axis_func=sfr_axis)
    
    def plot_f_star(self, z_vals, figsize=[9,9]):
        func = LuminosityFunctionModel.f_star
        ax_scale = ['log', 'log']
        ax_lab = [r'$M_h$ [M$_\odot$]', r'$f_*(M, z)$']
        self.plot_model_func_vs_M(z_vals, func, ax_scale, ax_lab, figsize)

    def plot_xi(self, z_range, figsize=[9,9]):
        func = LuminosityFunctionModel.xi
        ax_scale = ['linear', 'linear']
        ax_lab = [r'$z$', r'$\langle x_i(z)\rangle$']
        self.plot_model_func_vs_z(z_range, func, ax_scale, ax_lab, figsize)

    def kwargs_func(model, z, z_num, model_num):
            kw = {'label' : '{} z={}'.format(model.name, z)}
            kw.update({'color': 'C{}'.format(model_num)})
            kw.update({'ls' : utils.LINESTYLE_ARR[z_num]})
            return kw

    def plot_model_func_vs_M(self, z_vals, func, ax_scale, ax_lab, figsize=[12,9], kwargs_func=kwargs_func, second_axis_func=None):
        fig, ax = plt.subplots(figsize=figsize)
        for i, z in enumerate(utils.to_array(z_vals)):
            for k, model in enumerate(self.models):
                M_vals = np.logspace(*np.log10(model.min_max_mass(z)), 1000)
                kw = kwargs_func(model, z, i, k) if kwargs_func is not None else {}
                ax.plot(M_vals, [func(model, M, z) for M in M_vals], **kw)
        ax.set_xscale(ax_scale[0])
        ax.set_yscale(ax_scale[1])
        ax.set_xlabel(ax_lab[0])
        ax.set_ylabel(ax_lab[1])
        ax.margins(0.1,0.05)
        ax.legend()
        if second_axis_func is not None:
            second_axis_func(ax)
        plt.tight_layout()

    def plot_corner(self, backend_fn, burn_in=0, which_model=0):
        mpl.style.use('default')
        plt.rc('font', family='serif', size=9)
        reader = emcee.backends.HDFBackend(backend_fn)
        samples = reader.get_chain(discard=burn_in, flat=True)
        model = self.models[0]
        params = model.to_optimizer_input()
        n_dim = len(params)
        labels = model.param_names
        quants = [utils.ONE_SIGMA, 0.5, 1-utils.ONE_SIGMA]
        label_kwargs=dict(fontsize=20)
        fig = corner.corner(samples, show_titles=True, labels=labels, quantiles=quants, label_kwargs=label_kwargs)
        for i, model in enumerate(self.models):
            params = model.to_optimizer_input()
            corner.overplot_lines(fig, params, color='C{}'.format(i), lw=2)
            corner.overplot_points(fig, params[None], marker='s', markeredgecolor='w', markersize=7, markeredgewidth=1.5)
        utils.my_mpl()

    def plot_chain(self, backend_fn):
        reader = emcee.backends.HDFBackend(backend_fn)
        samples = reader.get_chain()
        raise NotImplementedError


class LuminosityFunctionMeasurements:
    def __init__(self, filename):
        self.filename = filename
        self.uc_measurements = np.array(load_pe(filename)).T
        self.dust_corrector = DustCorrector()
        self.dc_measurements = np.array([self.dust_corrector.dodc(m) for m in self.uc_measurements])

    def group_points_by_z(self, measurements, dc=True):
        grouped = {z:[] for z in np.unique(measurements.T[0])}
        for point in measurements:
            point_info = np.array([*point[1:], self.filename, dc], dtype=object) #prevent conversion to string
            grouped[point[0]].append(point_info)
        return grouped

    def get_pts(self, dc=True, by_z=False):
        m = self.dc_measurements
        if not dc:
            m = self.uc_measurements
        if by_z:
            return self.group_points_by_z(m, dc=dc)
        return m

    def get_pts2(self, dc=True):
        meas = np.array(self.dc_measurements)
        if not dc:
            meas = np.array(self.uc_measurements)
        all_pts = {
            'z': meas.T[0],
            'mag' : meas.T[1],
            'phi' : meas.T[2],
            'sigma_minus' : meas.T[3],
            'sigma_plus' : meas.T[4]
            }
        return all_pts

class HaloMassFunction:
    def __init__(self):
        self.D_g_0 = self.D_g(0, normalize=False)

    def f(self, sigma):
        pass

    def D_g(self, z, normalize=True):
        hi = 1000
        prefactor = 5 * utils.OMEGA_M * utils.cosmo.H(z) / (2*utils.H0) 
        integrand = lambda z_prime: (1+z_prime) / (utils.cosmo.H(z_prime)/utils.H0)**3
        integral = utils.trapz_integrate(integrand, z, hi)
        if normalize:
            integral = integral / self.D_g_0
        return prefactor * integral

    def sigma_M(self, M, z):
        pass

    def dn_dM(self, M, z):
        sigma = self.sigma_M(M,z)
        prefactor = -(utils.RHO_M.value/M) * 1/sigma * self.dsigma_dM(M,z)
        return prefactor * self.f(sigma)
        
class Sheth_Tormen(HaloMassFunction):
    def __init__(self):
        self.A_ST = 0.322
        self.a_ST = 0.707
        self.p_ST = 0.3
        self.delta_ST = 1.686

    def f(self, sigma):
        term1 = self.A_ST * np.sqrt(1*self.a_ST/np.pi)
        term2 = 1 + (sigma**2/(self.a_ST*self.delta_ST**2))**self.p_ST
        term3 = self.delta_ST/sigma
        term4 = np.exp(-self.a_ST*self.delta_ST**2/(2*sigma**2))
        return term1 * term2 * term3 * term4

class LuminosityFunctionModel:
    def __init__(self, measurements_fn, param_dict={}, optimizer_input=None, dc=True, name=None):
        self.measurements = LuminosityFunctionMeasurements(measurements_fn)
        fname = measurements_fn
        self.name = 'B' + fname[fname.rfind('_')+1:fname.rfind('.')] + ' Fit'
        if name is not None:
            self.name = name
        self.sigma = np.log(10)*0.16
        self.f_esc = 0.2
        self.param_dict = param_dict
        self.xi_func = None
        self.mass_fn = load_mf()
        self.M_min = {}
        self.M_max = {}
        self.dc = dc
        self.chi_dict = {}
        self.param_names = []

    def dn_dM(self, M, z):
        return self.mass_fn[z](M)

    def L_c(self, M, z):
        pass

    def m_c(self, M, z):
        return utils.L_to_m(self.L_c(M, z))

    def L_mean(self, M, z):
        return np.exp(self.sigma**2/2) * self.L_c(M, z) # lognormal

    def min_max_mass(self, z):
        try:
            return self.M_min[z], self.M_max[z]
        except KeyError:
            mags = np.array(self.measurements.get_pts(by_z=True)[z]).T[0]
            mag1, mag2 = min(mags), max(mags)
            lo, hi = 7, 15
            M_vals = np.logspace(lo, hi, 1000)
            m_from_M = [self.m_c(M, z) for M in M_vals]
            M_from_m = interp1d(m_from_M, M_vals) #utils.linlog_interp(m_from_M, M_vals)
            M_min, M_max = M_from_m(mag2), M_from_m(mag1)
            self.M_min.update({z : M_min})
            self.M_max.update({z: M_max})
        return self.M_min[z], self.M_max[z]

    def phi_L_given_M(self, L, M, z):
        prefactor = 1/(np.sqrt(2*np.pi)*self.sigma*L)
        exponent = -np.log(L/self.L_c(M,z))**2 / (2*self.sigma**2)
        return prefactor * np.exp(exponent)

    def phi_L(self, L, z):
        lo, hi = 1e7, 1e15
        integrand = lambda M: self.dn_dM(M, z) * self.phi_L_given_M(L, M, z)
        return utils.trapz_integrate(integrand, lo, hi, logspace=True)

    def phi_m(self, m, z):
        L = utils.m_to_L(m)
        return utils.dL_dm(m) * self.phi_L(L, z)

    def M_dot_acc(self, M, z, delta=1.127, eta=2.5, to_Gyr=False):
        retval = 3 * (M/1e10)**delta * ((1+z)/7)**eta # In M_sun/yr
        if to_Gyr:
            retval *= 1e9
        return retval

    def sfr(self, M, z):
        L = self.L_mean(M, z)
        return utils.KAPPA_UV * L

    def f_star(self, M, z):
        return self.sfr(M, z) / self.M_dot_acc(M, z)

    def n_dot_ion(self, z):
        lo, hi = 1e8, 1e15
        prefactor = utils.A_HE * utils.F_GAMMA * (utils.OMEGA_M/utils.OMEGA_B) / utils.RHO_M
        def integrand(M):
            return self.f_esc * self.f_star(M, z) * self.M_dot_acc(M, z, to_Gyr=True) \
                * self.dn_dM(M, z)
        integral = utils.trapz_integrate(integrand, lo, hi, logspace=True)
        return prefactor * integral

    def big_n_dot_ion(self, z):
        retval_in_inv_Gyr = self.n_dot_ion(z)*utils.RHO_M*utils.OMEGA_B/utils.OMEGA_M \
            * (1/M_H.to('M_sun').value) * u.Gyr**-1
        return retval_in_inv_Gyr.to('s^-1').value

    def t_rec(self, z, T_e=1e4):
        alpha_B = 2.6e-13 * (T_e/1e4)**0.76 * u.cm**3 * u.s**-1
        return (1/(alpha_B * utils.C_HII * (1+z)**3 * utils.N_H0)).to('Gyr').value

    def dxi_dz(self, z, x):
        return utils.dt_dz(z) * (self.n_dot_ion(z) - x/self.t_rec(z))

    def xi(self, z):
        if self.xi_func is None:
            lo, hi = 5, 100
            fun = lambda z_prime, x: self.dxi_dz(z_prime, x)
            z_span = [lo, hi]
            z_eval = np.linspace(lo, hi, 1000)
            x0 = [0]
            xi_z_eval = solve_ivp(fun=fun, t_span=z_span, y0=x0, t_eval=np.flip(z_eval))
            self.xi_z = log_interp1d(z_eval, xi_z_eval)
        return self.xi_func(z)

    def tau_e(self, z):
        numer = 3 * utils.H0 * utils.OMEGA_B * const.c * const.sigma_T
        denom = 8 * np.pi * const.G * const.m_p
        prefactor = (numer/denom).to('') # Cancel out units
        def integrand(z):
            N_He = 1 if z < 3 else 2
            num = self.xi(z) * (1+z)**2 + (1-utils.Y_P) + N_He*utils.Y_P/4
            den = np.sqrt(utils.OMEGA_M * (1+z)**3 + utils.OMEGA_L)
            return num/den
        eps = np.finfo(float).eps # loglog interpolation breaks at 0
        z = max(z, eps)
        integral = utils.trapz_integrate(integrand, eps, min(z,3))
        if z > 3:
            integral += utils.trapz_integrate(integrand, 3, z)
        return prefactor * integral

    def chi_sq_of_model(self):
        points = self.measurements.get_pts(self.dc, by_z=True)
        for i, (z, point_info) in enumerate(sorted(points.items())):
            point_info = np.array(point_info).T
            mags = point_info[0]
            phis = point_info[1]
            los = point_info[2]
            his = point_info[3]
            t = [self.phi_m(mag, z) for mag in mags]
            self.chi_dict.update({z : sum([utils.chi_squared(phis[i], t[i], his[i], los[i]) for i in range(len(t))])})
        return self.chi_dict

class FiveParameterCLF(LuminosityFunctionModel):
    def __init__(self, measurements_fn, param_dict={}, optimizer_input=None, dc=True, name=None):
        super().__init__(measurements_fn, param_dict, optimizer_input, dc, name)
        self.param_names = [r'$p$', r'$q$', r'$r$', r'$L_0$ [mag]', r'$\log_{10}(M_1/$M$_\odot)$']
        if len(param_dict) > 0:
            self.p = param_dict['p']
            self.q = param_dict['q']
            self.r = param_dict['r']
            self.L0 = param_dict['L0']
            self.M1 = param_dict['M1']
        else:
            self.p = optimizer_input[0]
            self.q = optimizer_input[1]
            self.r = optimizer_input[2]
            self.L0 = utils.m_to_L(optimizer_input[3])
            self.M1 = 10**optimizer_input[4]

    def to_optimizer_input(self):
        return np.array([self.p, self.q, self.r, utils.L_to_m(self.L0), np.log10(self.M1)])

    def L_c(self, M, z):
        return self.L0 * (M/self.M1)**self.p/(1+(M/self.M1)**self.q) * ((1+z)/4.8)**self.r

class LFModelOptimizer:
    def __init__(self, measurements_fn, ModelClass, x0, prior=None):
        self.ModelClass = ModelClass
        self.measurements_fn = measurements_fn
        self.measurements = LuminosityFunctionMeasurements(measurements_fn)
        self.x0 = x0
        self.prior = prior
        self.scipy_output = None
        self.prep_chi_sq_func()
        self.theory_prediction = None

    def log_prior(self, x):
        if self.prior is None:
            return 0
        for i, param in enumerate(x):
            if param < min(self.prior[i]) or param > max(self.prior[i]):
                return -np.inf
        return 0

    def optimize(self, load_fn=None, method='Nelder-Mead'):
        if load_fn is not None:
            with open(load_fn, 'rb') as handle:
                self.scipy_output = pickle.load(handle)
        else:
            self.scipy_output = scipy.optimize.minimize(self.chi_sq_of_model, self.x0, method=method)
        return self.scipy_output
        #if not self.scipy_output.success:
        #    print('FAILED OPTIMIZATION')
        #return opt_output.fun, opt_output.x

    def prep_chi_sq_func(self):
        meas_dict = self.measurements.get_pts2()
        self.mags = meas_dict['mag']
        self.m = meas_dict['phi']
        self.sp = meas_dict['sigma_plus']
        self.sm = meas_dict['sigma_minus']
        self.z_vals = meas_dict['z']
        self.n_pts = len(self.m)

    def chi_sq_of_model(self, x):
        model = self.ModelClass(self.measurements_fn, optimizer_input=x)
        t = [model.phi_m(self.mags[i], self.z_vals[i]) for i in range(self.n_pts)]
        self.theory_prediction = t
        chi_arr = [utils.chi_squared(self.m[i], t[i], self.sp[i], self.sm[i]) for i in range(self.n_pts)]
        return sum(chi_arr)

    def log_prob(self, x):
        ln_prob = -0.5 * self.chi_sq_of_model(x) + self.log_prior(x)
        return ln_prob

def mcmc_log_prob(x, optimizer):
    return optimizer.log_prob(x), optimizer.theory_prediction

# N_WALKERS=96 bc running on 24 threads
def mcmc(optimizer, backend_fn, n_steps=5000, n_walkers=96, n_dim=5, init_sigma=0.1):
    import os
    from multiprocessing import Pool
    os.environ['OMP_NUM_THREADS'] = '1'
    backend = emcee.backends.HDFBackend(backend_fn)
    backend.reset(n_walkers, n_dim)
    emcee_x0 = np.random.normal(loc=0, scale=init_sigma, size=(n_walkers, n_dim)) + optimizer.x0
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, mcmc_log_prob, args=(optimizer,), pool=pool, backend=backend)
        sampler.run_mcmc(emcee_x0, n_steps, progress=True)

