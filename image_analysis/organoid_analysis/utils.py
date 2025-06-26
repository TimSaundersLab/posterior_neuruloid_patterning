import numpy as np
from scipy.interpolate import interp1d

def index_to_xdata(xdata, indices):
    "interpolate values obtained from index_space to xdata_space"
    ind = np.arange(len(xdata))
    f = interp1d(ind, xdata)
    return f(indices)

def overall_mean_std(means, stds, sizes):
    # Calculate the weighted mean
    total_size = np.sum(sizes)
    weighted_mean = np.sum(np.array(means) * np.array(sizes)) / total_size
    
    # Calculate the weighted variance
    variance = (np.sum((np.array(sizes) - 1) * np.array(stds) ** 2) + 
                np.sum(np.array(sizes) * (np.array(means) - weighted_mean) ** 2)) / total_size
    
    # The overall standard deviation is the square root of the variance
    overall_std = np.sqrt(variance)
    
    return weighted_mean, overall_std

def fit_linear_line_and_evaluate(x, y):
    """
    Fits a linear line (y = mx + b) to the provided data and evaluates the fit.
    
    Parameters:
    - x: array-like, the independent variable (e.g., radii).
    - y: array-like, the dependent variable (e.g., mean or variance of gamma).
    - plot: bool, if True, plots the data and the fitted line.
    
    Returns:
    - slope: float, the slope of the fitted line.
    - intercept: float, the y-intercept of the fitted line.
    - r_squared: float, the coefficient of determination (R^2) of the fit.
    """
    # Fit a linear line y = mx + b
    slope, intercept = np.polyfit(x, y, 1)
    
    # Predicted values from the linear model
    y_pred = slope * x + intercept
    
    # Calculate R^2 (coefficient of determination)
    ss_total = np.sum((y - np.mean(y))**2)  # Total sum of squares
    ss_residual = np.sum((y - y_pred)**2)  # Residual sum of squares
    r_squared = 1 - (ss_residual / ss_total)  # R^2 formula
    return slope, intercept, r_squared