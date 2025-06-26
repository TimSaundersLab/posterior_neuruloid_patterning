import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import pandas as pd
plt.style.use('seaborn-v0_8-bright')
plt.rcParams["font.family"] = "monospace"
from scipy import ndimage
from joblib import Parallel, delayed
# import utils as u

from skimage import measure, morphology
from skimage.measure import find_contours
from scipy import ndimage, interpolate
from scipy.spatial import Delaunay
from scipy.signal import medfilt
from sklearn.cluster import KMeans

from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree

### FUNCTIONS FOR OBTAINING INTENSITY
def get_intensities_array(image,
                          contour, 
                          rotation_angle, 
                          centre,
                          radius_array, 
                          angles_to_save):
    """
    Obtains intensity array with intensities from different angles
    Args:
        image (array): image array to obtain intensity from
        contour (array): contour of the shape
        rotation_angle (float): angle of rotation of the image shape
        centre (array): centre of shape
        radius_array (array): array of radius discretization
        angles_to_save (array): array of angles to find intensity
    """
    def process_angle(angle, image, contour, rotation_angle, centre, radius_array):
        coordinates = obtain_row_col_idx(radius_array, angle, rotation_angle, centre)
        return obtain_intensity_from_image(image, contour, coordinates, trim_zeros=False)
    # parallelizes for loop
    intensities = Parallel(n_jobs=-1)(
        delayed(process_angle)(angle, image, contour, rotation_angle, centre, radius_array)
        for angle in angles_to_save)
    return np.array(intensities)

def get_intensities_annulus(image,
                            large_contour,
                            small_contour,
                            centre,
                            radius_array, 
                            angles):
    """Obtain intensity from annulus shape"""
    intensities = np.zeros((len(angles), len(radius_array)))
    for i in range(len(angles)):
        coordinates = obtain_row_col_idx(radius_array, angles[i], 0, centre)
        intensities[i, :] = obtain_intensity_from_annulus(image, 
                                                          large_contour,
                                                          small_contour,
                                                          coordinates,
                                                          trim_zeros=False)
    return intensities

def obtain_intensity_from_image(image, 
                                contour, 
                                coordinates, 
                                trim_zeros=True):
    within_micropattern = are_points_in_contour(contour, coordinates)
    intensity = np.array([image[tuple(coordinates[i])] if within_micropattern[i] else 0 for i in range(len(coordinates))])
    if trim_zeros:
        intensity = np.trim_zeros(intensity, 'b')
    return intensity

def obtain_intensity_from_annulus(image,
                                  large_contour,
                                  small_contour,
                                  coordinates,
                                  trim_zeros=True):
    """To obtain intensity from annulus micropattern"""
    within_mircopattern = are_points_in_micropattern(large_contour, 
                                                     small_contour, 
                                                     coordinates)
    intensity = np.array([image[tuple(coordinates[i])] if within_mircopattern[i] else 0 for i in range(len(coordinates))])
    if trim_zeros:
        intensity = np.trim_zeros(intensity, 'fb')
    return intensity

def are_points_in_contour(contour, coordinates):
    contour_path = Path(contour)
    result = contour_path.contains_points(coordinates)
    return result.tolist()

def are_points_in_micropattern(large_contour,
                               small_contour,
                               coordinates):
    """For determining if coordinates are in micropattern for annulus"""
    within_large_contour = are_points_in_contour(large_contour, coordinates)
    within_small_contour = are_points_in_contour(small_contour, coordinates)
    # need to be outside of small contour but inside large contour
    return np.array(within_large_contour) & ~np.array(within_small_contour)

def obtain_row_col_idx(radius_array, 
                       angle, 
                       rotation_angle,
                       centre):
    """
    Obtain column and row of image given radius array and angle (0-2pi radians)
    Args:
        radius_array (array): array of radius
        angle (float): angle in radians 
        rotation_angle (float): rotation angle of shape in radians
        centre (array): centre of the shape
    """
    col_idx = radius_array * np.cos(angle-rotation_angle) + centre[1]
    col_idx = np.round(col_idx).astype(int)
    row_idx = radius_array * np.sin(angle-rotation_angle) + centre[0]
    row_idx = np.round(row_idx).astype(int)
    coordinates = np.column_stack((row_idx, col_idx))
    return coordinates

def subtract_from_nonzero(array, background):
    """
    Subtract constant background value from array
    If value of array is less that background value, returns 0
    """
    return np.where((array!=0) & (array>background), 
                    array-background, 
                    0)

def intensity_from_contours(n_output,
                            image,
                            distances, 
                            tolerance, 
                            t, 
                            tck):
    """
    Args:
        n_output (int): number of outputs from the intensity analysis
        image (array): image to obtain intensities from
        distances (array): discretised distance from boundary to centre
        tolerance (float): tolerance for lower bound of the distance
        t (array): between 0 and 1, param for evaluating contour function
        tck (tuple): parameters for contour function (from splprep)
    Returns:
        mean_intensity_segments (array of len n_output): 
        intensities by taking mean of segments
    """
    x_smooth, y_smooth = splev(t, tck)
    smooth_points = np.vstack((x_smooth, y_smooth)).T

    tree = cKDTree(smooth_points)
    # Derivative for tangent vector
    dx, dy = splev(t, tck, der=1)

    # Normalize tangent to get the unit normal vectors
    norm = np.sqrt(dx**2 + dy**2)
    # Rotate tangent by 90 degrees to get normal
    normal = np.array([dy, -dx]) / norm

    # Plot each inner contour by offsetting along the normal direction
    mean_intensity = np.zeros(len(distances))
    for i in range(len(distances)):
        d = distances[i]
        x_inner = x_smooth + d * normal[0]
        y_inner = y_smooth + d * normal[1]

        points = np.vstack((x_inner, y_inner)).T
        valid_points, _ = find_nearest_boundary(points, tree, d, tolerance)
        
        if len(valid_points)==0:
            mean_intensity[i] = 0
        else:
            row_idx, col_idx = np.round(valid_points[:, 0]).astype(int), np.round(valid_points[:, 1]).astype(int)
            coordinates = np.column_stack((row_idx, col_idx))
            # Obtain all intensities at the contours
            intensities = np.array([image[tuple(coordinates[j])] for j in range(len(coordinates))])
            mean_intensity[i] = np.mean(intensities)
        
    mean_intensity = np.trim_zeros(mean_intensity, 'b')
    mean_intensity_segments = mean_intensity_segments_ignore_zero(mean_intensity,
                                                                  n_output)
    return mean_intensity_segments

### MAIN FUNCTIONS FOR PRE-PROCESSING IMAGE
def get_binary_image(DAPI_image, 
                     intensity_threshold=150, 
                     area_threshold=10**4,
                     binary_dilation_iter=10,
                     binary_closing_iter=50):
    bin_image = DAPI_image > intensity_threshold
    bin_image = ndimage.binary_erosion(bin_image)
    for i in range(binary_dilation_iter):
        bin_image = ndimage.binary_dilation(bin_image)
    for i in range(binary_closing_iter):
        bin_image = ndimage.binary_closing(bin_image)
    bin_image = morphology.remove_small_objects(bin_image, min_size=area_threshold, connectivity=5)
    return bin_image

def get_contour_and_centroid(binary_image, 
                             area_threshold=10**4, 
                             smoothing_factor=10**3, 
                             annulus=False):
    contours = measure.find_contours(binary_image, 0)
    areas = np.array([contour_area(contour) for contour in contours])

    if annulus:
        # for annulus, save two contours
        large_contour = contours[np.where(areas>area_threshold)[0][0]]
        small_contour = contours[np.where(areas>area_threshold)[0][1]]
        large_contour = smoothing_contour(large_contour, smoothing_factor)
        centroid = calculate_area_centroid(large_contour)
        return large_contour, small_contour, centroid

    else:
        contour = contours[np.where(areas>area_threshold)[0][0]]
        contour = smoothing_contour(contour, smoothing_factor)
        centroid = calculate_area_centroid(contour)
        return contour, centroid
    
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

def mean_ignore_zeros(intensity_array, trim_zeros=False):
    """
    Obtain mean of the intensity
    Ignore any zeros in the arrays and only return mean of the non-zero values
    Args:
        intensity_array (array)
    """
    masked_arr = np.ma.masked_equal(intensity_array, 0)
    mean_values = np.ma.mean(masked_arr, axis=0).filled(0)
    # std_values = np.ma.std(masked_arr, axis=0).filled(0)
    # mask = np.ma.getmaskarray(masked_arr)
    # masked_count_axis_0 = np.sum(~mask, axis=0)
    if trim_zeros:
        mean_values = np.trim_zeros(mean_values, 'b')
    return mean_values

def safe_division(arr1, arr2):
    """
    Elementwise dividing two arrays
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(np.abs(arr2)>0, np.divide(arr1, arr2), 0)
    return result

def get_max_dist_min_dist(contour, centroid):
    distances = np.linalg.norm(contour - centroid, axis=1)
    max_dist = distances[np.argmax(distances)]
    min_dist = distances[np.argmin(distances)]
    return max_dist, min_dist
    
### SUPPORT FUNCTIONS FOR PRE-PROCESSING IMAGE
def contour_area(contour):
    x, y = contour[:, 1], contour[:, 0]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def smoothing_contour(contour, smoothing_factor=10**3):
    # Fit a spline to the contour points
    tck, u = interpolate.splprep([contour[:, 0], contour[:, 1]], s=smoothing_factor, per=True)
    u_new = np.linspace(u.min(), u.max(), len(contour))
    x_new, y_new = interpolate.splev(u_new, tck, der=0)
    smoothed_contour = np.stack([x_new, y_new], axis=1)
    return smoothed_contour

def calculate_area_centroid(points):
    """To calculate the area centroid of irregularly shaped polygon"""
    x0, y0 = points[:,0], points[:,1]
    x1, y1 = np.roll(points[:,0], -1), np.roll(points[:,1], -1)
    # calculate the cross product and areas
    cross_product = x0 * y1 - x1 * y0
    A = 0.5 * np.sum(cross_product)
    # compute centroid coordinates
    C_x = np.sum((x0+x1) * cross_product) / (6*A)
    C_y = np.sum((y0+y1) * cross_product) / (6*A)
    centre = np.array([C_x, C_y])
    return centre

def shrink_contour(contour, centroid, scale_factor):
    # Ensure scale factor is within the correct range
    if not (0 < scale_factor < 1):
        raise ValueError("scale_factor must be between 0 and 1")
    # Shift the contour points so that the centroid is at the origin
    shifted_contour = contour - centroid
    # Scale the shifted contour points
    shrunk_shifted_contour = shifted_contour * scale_factor
    # Shift the contour points back to their original location
    shrunk_contour = shrunk_shifted_contour + centroid
    return shrunk_contour

def simple_background(image, square_size=50):
    """Obtain simple background value of image"""
    if len(image.shape)>2:
        print("Image must be in 2D (e.i. use max projection)")
        return
    background_square = image[:square_size, :square_size]
    return background_square.mean()

# PLOTTING FUNCTIONS
def find_nearest_boundary(points, 
                          tree, 
                          distance_threshold, 
                          tolerance):
    """
    Function to filter out coordinates, 
    Remove the coordinates if there is a shorter distance to boundary
    Args:
        points (array): points on contour to filter
        tree (cKDTree): cKDTree data structure of the boundary coordinates
        distance_threshold (float): shortest distances needed for pointst to contour
        tolerance (float): tolerance for lower bound of the distance
    Returns:
        valid_points (array): points to keep
        distances (array)
    """
    # Query the tree for the nearest distances
    distances,_ = tree.query(points)
    # Filter points based on the distance threshold
    valid_points = points[distances >= distance_threshold - tolerance]
    return valid_points, distances[distances >= distance_threshold - tolerance]

def plot_contours(distances, tolerance, t, tck, axes):
    """
    Args:
        distances (array): discretised distance from boundary to centre
        tolerance (float): tolerance for lower bound of the distance
        t (array): between 0 and 1, param for evaluating contour function
        tck (tuple): parameters for contour function (from splprep)

    Returns:
        plots of contour
    """
    x_smooth, y_smooth = splev(t, tck)
    smooth_points = np.vstack((x_smooth, y_smooth)).T

    tree = cKDTree(smooth_points)
    dx, dy = splev(t, tck, der=1)  # Derivative for tangent vector

    # Normalize tangent to get the unit normal vectors
    tangent = np.array([dx, dy])
    norm = np.sqrt(dx**2 + dy**2)
    normal = np.array([dy, -dx]) / norm  # Rotate tangent by 90 degrees to get normal

    # Plot each inner contour by offsetting along the normal direction
    for d in distances:
        x_inner = x_smooth + d * normal[0]
        y_inner = y_smooth + d * normal[1]

        points = np.vstack((x_inner, y_inner)).T
        valid_points, _ = find_nearest_boundary(points, tree, d, tolerance)
        # print(len(valid_points))
        axes.plot(valid_points[:, 1], 
                  valid_points[:, 0], 
                  linestyle="--", 
                  label=f"Inner Contour (dist={d:.2f})")