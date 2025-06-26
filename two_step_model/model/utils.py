import pandas as pd
import numpy as np
from functools import reduce

def merge_dataframes(dataframe_list):
    """Merge all dataframes in a list on common columns"""
    if not dataframe_list:
        raise ValueError("The dataframes list is empty. Provide at least one DataFrame.")
    
    common_columns = list(set.intersection(*(set(df.columns) for df in dataframe_list)))
    if not common_columns:
        raise ValueError("No common columns found among the DataFrames.")
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=common_columns), dataframe_list)
    return merged_df

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

def fit_proportional_line_and_evaluate(x, y):
    """
    Fits a proportional linear line (y = mx) to the provided data and evaluates the fit.
    
    Parameters:
    - x: array-like, the independent variable (e.g., radii).
    - y: array-like, the dependent variable (e.g., mean or variance of gamma).
    - plot: bool, if True, plots the data and the fitted line.
    
    Returns:
    - slope: float, the slope of the fitted line.
    - r_squared: float, the coefficient of determination (R^2) of the fit.
    """
    # Reshape x to a 2D array as required by np.linalg.lstsq
    x_reshaped = x[:, np.newaxis]
    
    # Solve for the slope using np.linalg.lstsq, with intercept fixed to 0
    slope, _, _, _ = np.linalg.lstsq(x_reshaped, y, rcond=None)
    slope = slope[0]  # Extract the slope value from the result
    
    # Predicted values from the proportional linear model (y = mx)
    y_pred = slope * x
    
    # Calculate R^2 (coefficient of determination)
    ss_total = np.sum((y - np.mean(y))**2)  # Total sum of squares
    ss_residual = np.sum((y - y_pred)**2)  # Residual sum of squares
    r_squared = 1 - (ss_residual / ss_total)  # R^2 formula
    
    return slope, r_squared


def mean_intensity_segments_ignore_zero(input_array, n_output):
    """
    Obtain mean of n_output segments from an input array
    If there are zero elements, ignore zeroes
    """
    # Calculate segment boundaries
    indices = np.linspace(0, len(input_array), n_output+1).astype(int)
    # Create segments
    segments = [input_array[indices[i]:indices[i+1]] for i in range(n_output)]
    # Mask zero elements by setting them to NaN
    masked_segments = [np.where(segments[i]!=0, segments[i], np.nan) for i in range(len(segments))] 
    # Compute the mean of non-zero elements (ignoring NaNs) (if segment is all nans, replace with 0)
    segment_means = np.array([np.nanmean(segment) if not np.isnan(segment).all() else 0 for segment in masked_segments]) 
    return segment_means