import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import scipy.signal as signal
import scipy.stats as stats
from common.util import orthogonal_regression

class Trajectory:
    def __init__(s, rawx, rawy, rawt, dt = 0.005, smooth=None, filter_order=2, cut = None, interpolate_order=3):
        s.rawx = np.asarray(rawx)
        s.rawy = np.asarray(rawy)
        s.rawt = np.asarray(rawt)
        s.dt = dt
        s.xf = interpolate.UnivariateSpline(s.rawt, s.rawx, k=interpolate_order, s=0, ext=0)
        s.yf = interpolate.UnivariateSpline(s.rawt, s.rawy, k=interpolate_order, s=0, ext=0)
        s.t = np.arange(s.rawt[0], s.rawt[-1]+s.dt, s.dt) 
        s.x = s.xf(s.t)
        s.y = s.yf(s.t)
        if smooth: s.butterworth_filter(cutoff = smooth, filter_order=filter_order)
        if cut: s.cutit(cut)
        s.xvel = s.xf.derivative(1)(s.t)
        s.yvel = s.yf.derivative(1)(s.t)
        s.V = np.sqrt(s.xvel**2.0 + s.yvel**2.0)

        if interpolate_order < 2: return
        
        s.xacc = s.xf.derivative(2)(s.t)
        s.yacc = s.yf.derivative(2)(s.t)
        s.xjerk = s.xf.derivative(3)(s.t)
        s.yjerk = s.yf.derivative(3)(s.t)
        s.J = np.sum(np.sqrt(s.xjerk**2 + s.yjerk**2)) * s.dt
        s.alpha = np.unwrap(np.arctan2(s.yvel, s.xvel))
        D0 = np.abs(s.yacc * s.xvel - s.xacc * s.yvel)
        s.D = np.asarray([np.NaN if dd == 0.0 else dd for dd in D0])
        s.R = (s.V**3.0) / s.D
        s.C = 1.0 / s.R
        s.A = s.V / s.R
        s.ds = s.V * s.dt

    def butterworth_filter(s, cutoff, filter_order = 2):
        B, A = signal.butter(filter_order, cutoff  * 2 * s.dt, 'low')
        s.x = signal.filtfilt(B, A, s.x)
        s.y = signal.filtfilt(B, A, s.y)
        s.xf = interpolate.UnivariateSpline(s.t, s.x, k=3, s=0)
        s.yf = interpolate.UnivariateSpline(s.t, s.y, k=3, s=0)
        return s
    
    def cutit(s, cut):
        if type(cut) == type([1,1]):
            start_cut = cut[0]
            end_cut = cut[1]
        elif type(cut) == type(1):
            start_cut = cut
            end_cut = cut
        else:
            start_cut = 0
            end_cut = 0
        s.x = s.x[int(start_cut / s.dt):  len(s.x) - 1 - int(end_cut / s.dt) ]
        s.y = s.y[int(start_cut / s.dt):  len(s.y) - 1 - int(end_cut / s.dt) ]
        s.t = s.t[int(start_cut / s.dt):  len(s.t) - 1 - int(end_cut / s.dt) ]
        


    def calc_betas(s, rlim=None, orthogonal=False):
        
        # remove NaNs
        filt = np.isfinite(s.C) & np.isfinite(s.A) & np.isfinite(s.V) & np.isfinite(s.R)        
        
        # if rlim is set, remove extreme R's as they are likey very flat parts of the path
        if rlim:
            rfilt = s.R < rlim
            filt = filt & rfilt
        
        s.C = s.C[filt]
        s.V = s.V[filt]
        s.A = s.A[filt]
        s.R = s.R[filt]
        s.tf = s.t[filt]  ## time variable for ploting with excluded elements
        s.filt = filt

        s.logC = np.log10(s.C)
        s.logV = np.log10(s.V)
        s.logA = np.log10(s.A)
        s.logR = np.log10(s.R)
        
        if orthogonal:
            CA = orthogonal_regression(s.logC, s.logA)
            s.betaCA, s.offsetCA, s.r2CA = CA["beta"], CA["offset"], CA["r2"]
            
            CV = orthogonal_regression(s.logC, s.logV)
            s.betaCV, s.offsetCV, s.r2CV = CV["beta"], CV["offset"], CV["r2"]
            
            RV = orthogonal_regression(s.logR, s.logV)
            s.betaRV, s.offsetRV, s.r2RV = RV["beta"], RV["offset"], RV["r2"]

        else:
            
            s.betaCA, s.offsetCA, rCA, p_v, std_err = stats.linregress(s.logC, s.logA)
            s.r2CA = rCA ** 2 

            s.betaCV, s.offsetCV, rCV, p_v, std_err = stats.linregress(s.logC, s.logV)
            s.r2CV = rCV ** 2

            s.betaRV, s.offsetRV, rRV, p_v, std_err = stats.linregress(s.logR, s.logV)
            s.r2RV = rRV**2
        
        
        return s

    def retrack(s, target_betaCA=None, target_betaCV=None, target_time=None, dt=None):
        if target_betaCA is not None:
            dts = s.ds / (s.C ** (target_betaCA - 1))
        elif target_betaCV is not None:
            dts = s.ds / (s.C ** target_betaCV)
        if dt is None: 
            dt=s.dt
        if (target_time is None): 
            target_time = s.t[-1] - s.t[0]

        t0 = np.concatenate(([0], np.cumsum(dts)[:-1]))
        t = s.t[0] + target_time  * (t0  / t0[-1])
        new_trajectory = Trajectory(np.copy(s.x), np.copy(s.y), t, dt)
        return new_trajectory

    def logplot(s, ax=None, step=1):
        if (ax == None): fig, ax = plt.subplots()
        ax.plot(s.logC[::step], s.logA[::step], '.', color="gray")
        reg_line = [s.beta * i + s.offset for i in s.logC[::step]]
        ax.plot(s.logC[::step], reg_line, '-', color="black", label=r"$\beta$={:.3f}".format(s.beta))
        ax.plot([],[], color=(0,0,0,0), label="$r^2$={:.2f}".format(s.r2))
        ax.legend(loc="lower right", frameon=False)
        ax.set_xlabel("log C")
        ax.set_ylabel("log A")
        
  
    def logplotCV(s, ax=None, step=1):
        if (ax == None): fig, ax = plt.subplots()
        ax.plot(s.logC[::step], s.logV[::step], '.', color="gray")
        reg_line = [s.betaCV * i + s.offsetCV for i in s.logC[::step]]
        ax.plot(s.logC[::step], reg_line, '-', color="black", label=r"$\beta$={:.3f}".format(s.betaCV))
        ax.plot([],[], color=(0,0,0,0), label="$r^2$={:.2f}".format(s.r2CV))
        ax.legend(loc="lower left", frameon=False)
        ax.set_xlabel("log C")
        ax.set_ylabel("log V")    
