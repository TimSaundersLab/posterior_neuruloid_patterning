import numpy as np
from numba import jit
import collections
from scipy.signal import argrelextrema, peak_widths
from scipy import stats
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

@jit(nopython=True)
def find_index(rand, prob):
    cum_prob = np.cumsum(prob)
    j = 0
    while j<len(prob):
        if rand<cum_prob[j]:
            break
        else:
            j+=1
    return j

def frequency(data, dp):
    number_count = collections.Counter(data.round(dp))
    pos = np.array([key for key, val in sorted(number_count.items())])
    count = np.array([val for key, val in sorted(number_count.items())])
    return pos, count

def line_to_circle(data, R):
    x = R*np.cos(data/R)
    y = R*np.sin(data/R)
    return x, y

def get_lobe_centre(x, pdf, threshold):
    """Using gaussian kernel density estimate to identify centre of clusters"""
    peaks = argrelextrema(pdf, np.greater)[0]
    lobes_loc = x[peaks][np.argwhere(pdf[peaks]>threshold)]
    return lobes_loc.flatten()

def fwhm(space_data, peak_idx, L, dx, mode="periodic"):
    """
    Returns full-width half maximum of peak, scaled with domain length, L
    """
    nx = len(space_data)
    
    if mode=="periodic":
        pad_length = 0.5*L
        n_pad = int(pad_length/dx)
        padded_data = np.concatenate((space_data[-n_pad:], space_data, space_data[:n_pad]))
        padded_peak_idx = peak_idx + n_pad
        widths = peak_widths(padded_data, padded_peak_idx, rel_height=0.5)[0]
        
    elif mode=="hard":
        widths = peak_widths(space_data, peak_idx, rel_height=0.5)[0]
        
    fwhm = widths*L/nx
    return fwhm

def angle_between_cluster(cluster_loc):
    # case 1: only 2 peaks, only 1 angle between
    if len(cluster_loc)==2:
        angle = cluster_loc[1] - cluster_loc[0]
        if angle>np.pi:
            angle = 2*np.pi - angle
        return np.array([angle])

    # case 2: n(>2) peaks, n angle between
    elif len(cluster_loc)>2:
        move_right = np.insert(cluster_loc[1:], len(cluster_loc[1:]), cluster_loc[0])
        angles = move_right - cluster_loc
        for i in range(len(angles)):
            if angles[i]<0:
                angles[i] = 2*np.pi + angles[i]
            if angles[i]>np.pi:
                angles[i] = 2*np.pi - angles[i]
        return angles
    
    # case 3: no peak
    else:
        return np.empty(0)

def combine_angles(theta_pos, start_index):
    if int(len(theta_pos))>start_index:
        theta_all = np.zeros(np.sum([len(theta_pos[i]) for i in range(start_index, len(theta_pos))]))
        count = 0
        for i in range(start_index, len(theta_pos)):
            theta_all[count:count+len(theta_pos[i])] = theta_pos[i]
            count += len(theta_pos[i])
        theta = np.mod(theta_all, 2*np.pi)
        return theta

    else:
        return np.concatenate([theta_pos[i] for i in range(len(theta_pos))])

def plot_polarization(theta_pos, 
                      dist_pos, 
                      rad, 
                      figsize=(25, 30), 
                      row=5, 
                      col=3):
    """Plotting simulation results of ring"""
    fig, ax = plt.subplots(row,col,figsize=figsize)
    plt.suptitle(f"Random simulation for radius {rad}.", fontsize=25)
    ax = ax.flatten()

    s=1
    w=5
    t=(((w-1)/2)-0.5)/s
    for i in range(len(theta_pos)):
        theta = theta_pos[i]
        dist = dist_pos[i]
        data_x = dist*np.cos(theta)
        data_y = dist*np.sin(theta)
        heatmap, xedges, yedges = np.histogram2d(data_x, data_y, bins=100)
        filtered = gaussian_filter(heatmap, sigma=s, truncate=t)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax[i].imshow(filtered.T, extent=extent, origin="lower")
        ax[i].set_title(f"Time={i*0.5}", fontsize=20)
    fig.colorbar(im, ax=ax, location="right", shrink=0.6)

def plot_pdf(pdf, threshold=None):
    angle = np.linspace(0, 2*np.pi, len(pdf))
    uniform = 1/(2*np.pi)
    figure = plt.figure(figsize=(10,6))
    plt.title("Particle distribution from Gaussian KDE")
    plt.xlabel("Angle")
    plt.xlim(0, 2*np.pi)
    plt.ylim(0, np.max(pdf))
    plt.xticks(np.arange(0, 2*np.pi+np.pi/2, step=(np.pi/2)), ['0','π/2','π', '3π/2', '2π'])
    plt.plot(angle, pdf)
    plt.hlines(uniform, xmin=0, xmax=2*np.pi, color="orange", label="Uniform distribution")
    if not threshold==None:
        lobes_loc = get_lobe_centre(angle, pdf, threshold)
        plt.vlines(lobes_loc, ymin=0, ymax=np.max(pdf), color="purple", label="Centre of clusters")
        plt.hlines(threshold, xmin=0, xmax=2*np.pi, color="red", label="Threshold")
    plt.legend()

def random_int(Nx, N):
    """
    Randomly allocate a total of N particles in Nx compartments
    Args:
        Nx: (int) number of compartments
        N: (int) number of particles to allocate
    
    Returns:
        array of integers with size Nx which sums up to N
    """
    arr = np.zeros(Nx)
    for i in range(N):
        arr[np.random.randint(0, Nx)] += 1
    return arr

def sinusoidal_perturbations(x, N, waves, factor=(-1/20)):
    """
    Parameters:
        x: (array) domain 
        N: (int/float) total number of particles
    Returns a sinusoidal perturbation wave around a uniform concentration
    """
    L = x[-1]
    dx = x[1] - x[0]
    S_shape = factor*np.cos(2*waves*np.pi*x/L)
    k = ((N/dx) - np.sum(S_shape))/len(x)
    Si = S_shape + np.ones_like(x)*k
    return Si

def random_perturbations(x, N, noise):
    """
    Parameters:
        x: (array) domain
        N: (int) total number of molecules
        
    Returns perturbations around the uniform distribution 
    around the uniform concentration N/L depending on gaussian noise level
    """
    L = x[-1]
    uniform = N/L
    rand = np.random.normal(0, noise, size=len(x))
    return rand+uniform

def to_frequency(space_data, L):
    """
    From compartmental spatial data to frequency data (before kde)
    Args:
        space_data: array with number of molecules in each compartment
        L: (float) total length of the domain
     
    Return:
        frequency data (array)
    """
    freq = np.zeros(int(np.sum(space_data)))
    track = 0
    for i in range(len(space_data)):
        ni = int(space_data[i])
        try:
            freq[track : track+ni] = np.ones(ni)*((i*L)/len(space_data))
        except:
            # print("Contains negative values")
            ni = 0
        track += ni
    return freq
    

def fit_gaussian_kde(space_data, L, dx=0.05, mode="periodic", bw_method_factor=(-1/4)):
    """
    Perform non-parametric gaussian kde fit on data
    Args:
        space_data: (array) with number of molecules in each compartment
        mode: "periodic" or "normal"
        bw_method_factor: power of len(data) for bandwidth OR "default" for Scott's rule
    Return:
        Approximated PDF with same size of the space_data
    """
    x = np.arange(0, L+dx, dx)
    
    if mode == "periodic": 
        # pad the data at the front and back, then fit gaussian kde
        pad_length = 0.5*L
        n_pad = int(pad_length/dx)
        padded_data = np.concatenate((space_data[-n_pad:], space_data, space_data[:n_pad]))
        freq_data = to_frequency(padded_data, L+2*pad_length)
        freq_data = freq_data - pad_length
        if bw_method_factor=="default":
            kernel = gaussian_kde(freq_data)
        else:
            kernel = gaussian_kde(freq_data, bw_method=len(freq_data)**bw_method_factor)
        pdf = kernel(x)*2
        
    elif mode == "hard":
        freq_data = to_frequency(space_data, L)
        kernel = gaussian_kde(freq_data)
        pdf = kernel(x)
        
    return pdf

# function to smooth data by moving average
def moving_average(x, width):
    """ Perform a moving average with uniform weights in the sliding window
    Missing data (i.e. NaN values) will be treated in the same way as end points,
    computing the average over only the non-nan terms in the window.
    """
    return weighted_moving_average(x, weights=np.ones(width))


def weighted_moving_average(x, weights):
    """ Perform a weighted moving average
    Weights will be renormalised to sum to 1.
    Missing data (i.e. NaN values) will be treated in the same way as end points,
    computing the average over only the non-nan terms in the window.
    """
    # We convolve with a vector of ones to get the correct normalisation at each point.
    # This is necessary to avoid end effects, since the ends of the output will not be
    # calculated using the full set of weights. Where there are NaN values in x, we use
    # a zero in the vector of ones to indicate that the corresponding weight will not
    # contribute.
    is_nan = np.isnan(x)
    x = np.where(is_nan, 0, x)
    ones = np.ones_like(x)
    ones[is_nan] = 0

    # If the signal is shorter than the window, convolve returns an array the length of
    # the window. To get around this we pad the signal with zeros at the beginning if
    # necessary to make it at least as large as the window, and cut down the result at
    # the end.
    nelts = len(x)
    if nelts < len(weights):
        pad = len(weights) - nelts
        x = np.concatenate([x, np.zeros(pad)])
        ones = np.concatenate([ones, np.zeros(pad)])

    norm = np.convolve(ones, weights, "same")
    smoothed = np.convolve(x, weights, "same") / norm

    return smoothed[:nelts]
