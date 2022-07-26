import emcee
import numpy as np
from scipy import stats
from scipy import optimize
from scipy import special
from scipy import constants
from scipy import signal
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.stats import norm

m_e = constants.physical_constants['electron mass energy equivalent in MeV'][0]*1000
r_e = constants.physical_constants['classical electron radius'][0]*1e15

def gaus(loc, scale, area):
    """        
    Assuming the peaks are Gaussian, 
    we return the expected number of counts 
    at x.

    :param x: point to evaluate at 
    :param loc: peak location 
    :param scale: peak width
    :param area: area of the peak
    :returns: predicted number of counts at each location x
    :rtype: array

    """

    def eval(x):
        f = stats.norm(loc=loc, scale=scale)
        return area*f.pdf(x)
        
    return eval

def linear(a, b):
    """Linear background term

    :param a: slope of the line
    :param b: intercept of the line
    :returns: closure of eval(x)
    :rtype: function

    """

    def eval(x):
        """Closure to evaluate points

        :param x: array of points to evaluate at
        :returns: array of predicted values
        :rtype: array

        """        
        return a*x + b

    return eval


def exp_decay(a, tau):
    """exponential decay:
    y = a*exp(-x/tau)

    :param tau:  
    :returns: closure of eval(x)
    :rtype: function

    """

    def eval(x):
        """Closure to evaluate points

        :param x: array of points to evaluate at
        :returns: array of predicted values
        :rtype: array

        """
        f =  a * np.exp(-x/tau)
        return f

    return eval


def gaus_exp(loc, scale, area, tau):
    """        
    Gaussian with an exponential tail for silicon detector peaks.
    Form taken from Bortels and Collaers 1987

    :param x: point to evaluate at 
    :param loc: peak location 
    :param scale: peak width
    :param area: area of the peak
    :param tau: decay constant of the left sided exponential
    :returns: predicted number of counts at each location x
    :rtype: array

    """

    def eval(x):
        first_term = area/(2.0*tau) * np.exp((x - loc)/scale + 
                                             (scale**2.0 / (2.0*tau**2.0)))
                                              
        second_term = special.erfc((1.0/np.sqrt(2.0)) *
                                   ((x - loc)/scale + scale/tau))
        return first_term * second_term
        
    return eval




def parameters_values(samples):

    """
    Given an array of samples returns each parameters
    16, 50, 84 percentile values. Returns list of tuples.
    """
    values = [(v[1], v[2]-v[1], v[1]-v[0]) for v in zip(*np.percentile(samples,
                                    [16, 50, 84],
                                    axis=0))]
    return values



class GaussPrior():

    def __init__(self, loc, scale):
        self.pdf = stats.norm(loc=loc, scale=scale)

    def lnprior(self, x):
        return np.sum(self.pdf.logpdf(x))

class ScalePrior():
    def __init__(self, scale):
        self.pdf = stats.halfnorm(loc=0.0, scale=scale)

    def lnprior(self, x):
        return np.sum(self.pdf.logpdf(x))

class AreaPrior():

    def __init__(self, upper, lower=0.0):
        self.pdf = stats.uniform(loc=lower, scale=(upper - lower))

    def lnprior(self, x):
        return np.sum(self.pdf.logpdf(x))

class InterceptPrior():

    def __init__(self, lower, upper):
        self.pdf = uniform(loc=lower, scale=(upper - lower))
        self.prior_len = lower.shape[0]

    def lnprior(self, x):
        return np.sum(self.pdf.logpdf(x))


class LnLike():

    def __init__(self, functions, data, slices, stat='chi2'):
        self.slices = slices
        self.functions = functions
        self.channels = data[0]
        self.counts = data[1]
        self.dcounts = np.sqrt(data[1])
        # take care of zero counts
        self.dcounts[self.dcounts == 0] = 1
        if stat == 'chi2':
            self.lnlike = self.lnlike_chi2
        elif stat == 'poisson':
            self.lnlike = self.lnlike_poisson

    def slice_parms(self, parms):
        s = [parms[i[0]:i[1]] for i in self.slices]
        return s
        
    def eval_functions(self, parms):
        # associate proposed values with their function
        parm_slices = self.slice_parms(parms)
        D = np.zeros(len(self.channels)) # array that we will add to
        for i, f in zip(parm_slices, self.functions):
            eval = f(*i)
            D += eval(self.channels)
        return D

    def eval_functions_separately(self, samples):
        # Given an array of samples from MCMC
        # give predictions for each function separately
        values = np.zeros((samples.shape[0], len(self.slices),
                           len(self.channels)))
        for i, ele in enumerate(samples):
            parm_slices = self.slice_parms(ele)
            j = 0
            for x, f in zip(parm_slices, self.functions):
                eval = f(*x)
                D = eval(self.channels)
                values[i, j, :] = D
                j += 1
        return values
                
    def lnlike_poisson(self, parms):
        """
        Sum all of the Gaussian functions and background
        and compare to a Poisson expectation.
        """
        # 
        D = self.eval_functions(parms)
        probs = stats.poisson.logpmf(self.counts, D)
        return np.sum(probs)

    def lnlike_chi2(self, parms):
        D = self.eval_functions(parms)
        probs = stats.norm.logpdf(self.counts, loc=D, scale=self.dcounts)
        return np.sum(probs)

class Model():

    def __init__(self):
        self.priors = []
        self.likelihood = []
        self.functions = []
        self.index_pairs = []
        self.start_index = 0
        self.x0 = []
        self.func_names = []

    def create_gaus_peak(self, loc, loc_unc, scale, scale_unc, A_max, A_unc=None):
        stop_index = self.start_index + 3
        self.index_pairs.append((self.start_index, stop_index))
        self.start_index = stop_index
        self.priors.append(GaussPrior(loc, loc_unc))
        self.priors.append(GaussPrior(scale, scale_unc))
        if A_unc:
            self.priors.append(GaussPrior(A_max, A_unc))
        else:
            self.priors.append(AreaPrior(A_max))
        self.functions.append(gaus)
        self.x0 += [loc, scale, A_max]
        self.func_names.append('gaus')

    def create_gaus_exp_peak(self, loc, loc_unc, scale, scale_unc,
                             tau, tau_unc, A_max, A_unc=None):
        stop_index = self.start_index + 4
        self.index_pairs.append((self.start_index, stop_index))
        self.start_index = stop_index
        self.priors.append(GaussPrior(loc, loc_unc))
        self.priors.append(GaussPrior(scale, scale_unc))
        self.priors.append(GaussPrior(tau, tau_unc))
        if A_unc:
            self.priors.append(GaussPrior(A_max, A_unc))
        else:
            self.priors.append(AreaPrior(A_max))
        self.functions.append(gaus_exp)
        self.x0 += [loc, scale, A_max, tau]
        self.func_names.append('gaus_exp')

    def create_exp_decay(self, a, a_unc, tau, tau_unc):
        stop_index = self.start_index + 2
        self.index_pairs.append((self.start_index, stop_index))
        self.start_index = stop_index
        self.priors.append(GaussPrior(a, a_unc))
        self.priors.append(GaussPrior(tau, tau_unc))
        self.functions.append(exp_decay)
        self.x0 += [a, tau]
        self.func_names.append('exp')

    def create_compton_edge(self, Eg, a, a_unc, b, b_unc, scale, scale_unc, k, k_unc):
        stop_index = self.start_index + 4
        self.index_pairs.append((self.start_index, stop_index))
        self.start_index = stop_index
        self.priors.append(GaussPrior(a, a_unc))
        self.priors.append(GaussPrior(b, b_unc))
        self.priors.append(GaussPrior(scale, scale_unc))
        self.priors.append(GaussPrior(k, k_unc))
        compton_edge = Compton(Eg).compton_edge
        self.functions.append(compton_edge)
        self.x0 += [a, b, scale, k]
        self.func_names.append('compton')

    def create_compton_edge2(self, Eg1, Eg2, a, a_unc, b, b_unc,
                             scale1, scale1_unc, k1, k1_unc, scale2, scale2_unc, k2, k2_unc):
        stop_index = self.start_index + 6
        self.index_pairs.append((self.start_index, stop_index))
        self.start_index = stop_index
        self.priors.append(GaussPrior(a, a_unc))
        self.priors.append(GaussPrior(b, b_unc))
        self.priors.append(GaussPrior(scale1, scale1_unc))
        self.priors.append(GaussPrior(k1, k1_unc))
        self.priors.append(GaussPrior(scale2, scale2_unc))
        self.priors.append(GaussPrior(k2, k2_unc))

        compton_edge = Compton2(Eg1, Eg2).compton_edge
        self.functions.append(compton_edge)
        self.x0 += [a, b, scale1, k1, scale2, k2]
        self.func_names.append('compton2')

        
    def create_background(self, slope, intercept, slope_width=100.0, intercept_width=100.0):
        stop_index = self.start_index + 2
        self.index_pairs.append((self.start_index, stop_index))
        self.start_index = stop_index
        self.priors.append(GaussPrior(slope, slope_width))
        self.priors.append(GaussPrior(intercept, intercept_width))
        self.functions.append(linear)
        self.x0 += [slope, intercept]
        self.func_names.append('linear')
        
    def create_likelihood(self, data, like_type='chi2'):
        self.x0 = np.asarray(self.x0)
        self.likelihood.append(LnLike(self.functions,
                                      data, self.index_pairs,
                                      stat=like_type))


    def lnlike(self, parms):
        likelihood = 0.0                   
        for ele in self.likelihood:        
            likelihood += ele.lnlike(parms)

        if np.isnan(likelihood):
            return -1.0 * np.inf             
        return likelihood
        
    def lnprob(self, parms):

        priors = 0.0
        for i, ele in zip(parms, self.priors):
            priors += ele.lnprior(i)

        likelihood = self.lnlike(parms)

        probability = priors + likelihood

        if np.isnan(probability):
            return -1.0 * np.inf
        return probability

class Sampler():

    def __init__(self, model):
        """
        Currently not implemented since, mcmc is overkill for now
        """
        self.model = model
        self.nwalker = 100
        self.nstep = 1000
        self.ndim = model.x0.shape[0]

    def ball_init(self, scatter=1.e-2):
        """
        Initialize a random ndim ball around model.x0 parameters
        according to 'scatter' parameter, i.e the larger scale is
        the larger the ball is (can be).
        """
        self.p0 = ((self.model.x0*np.ones([self.nwalker, self.ndim])) +
                   (scatter*np.random.randn(self.nwalker, self.ndim)))

    def run_ensemble(self):
        """
        Run the sampler with the default stretch move.
        """
        self.sampler = emcee.EnsembleSampler(self.nwalker,
                                             self.ndim,
                                             self.model.lnprob)
        self.sampler.run_mcmc(self.p0, self.nstep, progress=True)

class Fit():

    def __init__(self, model, lnlike=True, x0=None):
        # logic for type of minimization, likelihood or MAP 
        if lnlike:
            self.lnprob = self.min_lnlike            
        else:
            self.lnprob = self.min_lnprob
        

        # logic for starting paramters
        if np.any(x0):
            self.custom_x0 = x0
        else:
            self.custom_x0 = model.x0

        self.fit_result = None
        self.model = model

    def min_lnlike(self, x):
        min = self.model.lnlike(x) 
        return -1.0 * min

    def min_lnprob(self, x):
        min = self.model.lnprob(x)
        return -1.0 * min

    def fit(self):
        r = optimize.minimize(self.lnprob, self.custom_x0)
        self.fit_result = r
        return r

    def fit_nm(self):
        r = optimize.minimize(self.lnprob, self.custom_x0, method='Nelder-Mead', options={'maxiter': 1e5, 'maxfev': 1e5})
        self.fit_result = r
        return r

    def get_model_index(self, start_index):
        return self.model.index_pairs[start_index]

    def fit_global(self, factor=1.0):
        # create bounds based type of functions to be fit
        self.bounds = []
        for i, ele in enumerate(self.model.func_names):
            if ele == 'gaus':
                start, stop = self.get_model_index(i)
                # all parameters should be positive
                temp = [(0.0, self.model.x0[j] + self.model.x0[j]*factor)
                        for j in range(start, stop)]
                self.bounds += temp
            elif ele == 'gaus_exp':
                start, stop = self.get_model_index(i)
                # all parameters should be positive
                temp = [(0.0, self.model.x0[j] + self.model.x0[j]*factor)
                        for j in range(start, stop)]
                self.bounds += temp
            elif ele == 'linear':
                start, stop = self.get_model_index(i)
                # no contraint, just take the factor range
                temp = [(-1.0*self.model.x0[j]*factor,
                         self.model.x0[j]*factor)
                        for j in range(start, stop)]
                self.bounds += temp
            else:
                start, stop = self.get_model_index(i)
                temp = [(-1.0*self.model.x0[j]*factor,
                         self.model.x0[j]*factor)
                        for j in range(start, stop)]
                self.bounds += temp

        r = optimize.dual_annealing(self.lnprob, self.bounds)
        self.fit_result = r
        return r

    
class DataPoint():

    def __init__(self, value, unc):
        """
        Create an object with a value and unc
        """

        self.value = value
        self.unc = unc


def area(counts):
    T = counts.sum()
    dT = np.sqrt(T)
    return T, dT
        

def net_area(channels, counts, bg_func):

    # Totals
    T = counts.sum()
    # Background counts
    B = np.sum(bg_func(channels))
    # Signal counts
    S = T - B
    dS = np.sqrt(T + B)
    return S, dS


def compton_energy(Eg):
    Ec = (2.0*Eg**2.0)/(2.0*Eg + m_e)
    return Ec

def electron_T(phi, Eg):
    alpha = Eg/(m_e)
    T = Eg*((2*alpha*(np.cos(phi)**2.0)) /
            ((1.0 + alpha)**2.0 -
             (alpha**2.0 * (np.cos(phi))**2.0)))
    return T

def klein_nishina(x, Eg):

    # T is electron kinetic energy
    # this formula has a ton of shit constants
    T_max = compton_energy(Eg)
    T = np.copy(x)
    T[T > T_max] = -1.0*np.inf
    # if T > T_max:
    #     return 0.0
    alpha = Eg/(m_e)    
    s = (T/Eg)
    const = (np.pi*r_e**2.0)/(m_e*alpha**2.0)
    term1 = s**2.0/(alpha**2.0*(1.0 - s)**2.0)
    term2 = s/(1.0 - s)
    term3 = (s - (2.0/alpha))
    result = const*(2.0 + term1 + (term2*term3))
    result[np.isnan(result)] = 0.0
    return result

def gaussian_spread(T, sigma, mean):
    return np.exp(-1.0/2.0*(mean - T)**2.0/(sigma)**2.0)

def kn_gaus_conv(E_max, sigma, Eg, samples=1e4):
    delta = 1.0/float(samples)
    samples = int(samples)
    grid = np.linspace(0.0, E_max, samples)
    kn = klein_nishina(grid, Eg)
    g = gaussian_spread(grid, sigma, E_max/2.0)*(1.0/sigma*np.sqrt(2.0*np.pi))
    conv = signal.fftconvolve(kn, g, mode='same')
    return grid, conv



class Compton():

    def __init__(self, Eg):
        self.Eg = Eg

    def compton_edge(self, a, b, scale, k):
        """        
        
        Find the Compton edge of plastic or liquid scintillator. 

        :param x_max: maximum channel number to integrate up to 
        :param scale: detector resolution in channel number
        :param k: normalization for the curve
        :param Eg: Energy of the incident gamma ray
        :returns: predicted number of counts at each location x
        :rtype: array

        """

        def eval(x):
            E = a*x + b
            grid, conv = kn_gaus_conv(10000, scale, self.Eg)
            f = interpolate.CubicSpline(grid, conv)
            return k*f(E)
        
        return eval


class Compton2():

    def __init__(self, Eg1, Eg2):
        self.Eg1 = Eg1
        self.Eg2 = Eg2
                

    def compton_edge(self, a, b, scale1, k1, scale2, k2):
        """        
        
        Find the Compton edge of plastic or liquid scintillator. 

        :param x_max: maximum channel number to integrate up to 
        :param scale: detector resolution in channel number
        :param k: normalization for the curve
        :param Eg: Energy of the incident gamma ray
        :returns: predicted number of counts at each location x
        :rtype: array

        """

        def eval(x):
            E = a*x + b
            grid1, conv1 = kn_gaus_conv(10000, scale1, self.Eg1)
            grid2, conv2 = kn_gaus_conv(10000, scale2, self.Eg2)

            f1 = interpolate.CubicSpline(grid1, conv1)
            f2 = interpolate.CubicSpline(grid2, conv2)
            return k1*f1(E) + k2*f2(E)
        
        return eval
