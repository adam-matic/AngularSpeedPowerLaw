import numpy as np
import scipy.interpolate
import scipy.signal as signal
from collections import deque
import scipy.fftpack as fp
from numba import njit
import scipy.odr as odr


def distance (x1, y1, x2, y2): return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def rmse(a, b):
	a = np.asarray(a)
	b = np.asarray(b)
	dif = a - b
	dif_squared = dif ** 2
	mean_of_dif = dif_squared.mean()
	rmse_val = np.sqrt(mean_of_dif)
	return rmse_val

def butter_filter(x, cutoff, samples_per_s=200, filter_order=2):
	xc = np.copy(x)
	B, A = signal.butter(filter_order, cutoff / (samples_per_s / 2), 'low')
	xs = signal.filtfilt(B, A, xc)
	return xs

def sm(x, cutoff=0.5, cut=500, samples_per_second=100):
	""" smoothing and cutting beginning and end """
	if cutoff > 0.0:
		x0 = butter_filter(x, cutoff, samples_per_second)
	else:
		x0 = x
	x1 = x0[cut: len(x0)-cut]
	return x1

def interpolate(ts_raw, xs, dt):
	ts = [t - ts_raw[0] for t in ts_raw]
	xc = np.copy(xs)
	x_spline = scipy.interpolate.UnivariateSpline(ts, xc, k=3, s=0)
	new_ts = np.arange(ts[0], ts[-1], dt)
	new_xs = x_spline(new_ts)
	return new_ts, new_xs

class empty(object):
	pass

def linear_func(p, x):
    m, c = p
    return m*x + c

def orthogonal_regression(x, y):
    res = odr.ODR(odr.RealData(x, y), odr.Model(linear_func), beta0=[0.,0.]).run()
    yfit = linear_func(res.beta, x)
    my = np.mean(y)
    SE_regr = np.sum((yfit -   my)**2)
    SS_tot  = np.sum((y    -   my)**2)
    SS_res  = np.sum((y    - yfit)**2)
    r2      = 1.0 - SS_res/SS_tot 
    beta, offset = res.beta
    # for y = beta*x + offset
    return {"beta": beta, "offset":offset, "r2": r2, "res":res }


class DelayLine():
	def __init__(self, length, init_value=0):
		self.delay_line = deque([init_value] * length)

	def __call__(self, x):
		self.delay_line.appendleft(x)
		return self.delay_line.pop()

class Delay():
	def __init__(s, length, init_value=0):
		s.write_to = length
		s.read_from = 0
		s.delay = length + 1
		s.delay_line = np.repeat(init_value, s.delay)

	def add(s, new_value):
		s.delay_line[s.write_to] = new_value
		out = s.delay_line[s.read_from]
		s.write_to = (s.write_to + 1) % s.delay
		s.read_from = (s.read_from + 1) % s.delay
		return out
    


def fft(w, sample_rate):
	n = len(w)
	k = np.arange(n)
	T = n / sample_rate
	frq = (k / T)[range(n // 2)]
	Y = abs(fp.fft(w)) / n
	Y = Y[range(n // 2)]
	return frq, Y


def rmsep(a, b):
	""" error in ratio of total range of a """
	a = np.asarray(a)
	b = np.asarray(b)
	dif = a - b
	dif_squared = dif ** 2
	mean_of_dif = dif_squared.mean()
	rmse_val = np.sqrt(mean_of_dif)

	rangea = np.max(a) - np.min(a)
	rmse_percent = rmse_val/rangea
	return rmse_percent

def rmse_percent(a, b):
	""" error in ratio of total range of a """
	## assume numpy array
	dif = a - b
	dif_squared = dif ** 2
	mean_of_dif = dif_squared.mean()
	rmse_val = np.sqrt(mean_of_dif)
	rangea = np.max(a) - np.min(a)
	rmse_ratio = (rmse_val/rangea)
	rmse_percents = "{:.3f}%".format(100.0 * rmse_ratio)
	return rmse_percents

#rmse_percent a, b = (np.asarray([10,10]), np.asarray([5,5]))

def get_vel(ts, xs):
	return scipy.interpolate.UnivariateSpline(ts, xs, k=3, s=0).derivative(1)(ts)


def resample(ts, xs, new_dt=0.005, smooth=None, cut=None, interpolate_order=3):
	nts = np.arange(ts[0], ts[-1], new_dt)
	xf  = interpolate.UnivariateSpline(ts, xs, k=interpolate_order, s=0)
	nx = xf(nts)
	if smooth:
		B, A = signal.butter(2, smooth  * 2 * new_dt, 'low')
		nx = signal.filtfilt(B, A, nx)
	if cut:
		i0, i1 = int(cut[0]/new_dt), int(cut[1]/new_dt) 
		N = len(nx)
		nx  =  nx[i0: N - 1 - i1]
		nts = nts[i0: N - 1 - i1]
	return nts, nx


