from dolfin import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon as Poly
from tqdm import tqdm

def is_point_in_mesh(point, mesh):
    point_dolfin = Point(*point)
    """Check if the given point is within the mesh"""
    bb_tree = mesh.bounding_box_tree()
    # print(bb_tree.compute_first_entity_collision(point_dolfin))
    # Use compute_first_entity_collision to check if a point collides with any cell
    return bb_tree.compute_first_entity_collision(point_dolfin) < mesh.num_cells()

def evaluate_point(point, u, mesh):
    """
    Evaluate the solution, u at a point
    Return None if the point is outside the mesh
    Args:
        point(array): coordinates of point to evaluate at
        u (function): solution of concentration
        mesh: mesh where pde is solved
    Returns
    """
    if is_point_in_mesh(point, mesh):
        value = u([point])
    else:
        value = None
    return value

def evaluate_solution(points, u, mesh):
    """
    Evaluate the solution, u at many points
    """
    values = np.zeros(len(points))
    for i in range(len(points)):
        values[i] = evaluate_point(points[i], u, mesh)
    return values

def extract_intensity_geometry(radii,
                               angles,
                               centre,
                               solution,
                               mesh):
    """
    Extract concentration from different geometries
    Returns dataframe of intensity for different angles
    Args:

    """
    df = pd.DataFrame(data={'radii':radii})
    for i in range(len(angles)):
        theta = angles[i]
        x_coord = radii * np.cos(theta) + centre[0]
        y_coord = radii * np.sin(theta) + centre[1]
        points = np.column_stack((x_coord, y_coord))
        A_conc = evaluate_solution(points, solution, mesh)
        df[f"{np.round(theta, 3)}"] = A_conc  
    return df

def plot_intensity(dataframe, normalise=True, normalise_all=False):
    """
    Plot intensity data given dataframe
    Args:
        dataframe (pandas df): with columns [radii, angles...]
        (radii is distance from centre to micropattern edge)
        normalise (bool): if True, normalise so that distance is 0-1
                        if False, distance not normalised
    """
    columns = list(dataframe.columns)
    for i in range(1, len(columns)):
        radius = dataframe["radii"].to_numpy()
        conc = dataframe[columns[i]].to_numpy()
        conc = np.trim_zeros(conc, trim='b')
        if normalise:
            radius = radius / radius.max()
        elif normalise_all:
            radius = radius[:len(conc)] / radius[len(conc)-1]
        plt.plot(radius[:len(conc)], conc, label=columns[i])
    plt.legend(title="Angles (radians)")
    plt.xlabel("Distance from centre to micropattern edge")
    plt.show()

def mean_ignore_zeros(intensity_array, trim_zeros=False):
    """
    Obtain mean of the intensity
    Ignore any zeros in the arrays and only return mean of the non-zero values
    Args:
        intensity_array (array)
    """
    masked_arr = np.ma.masked_equal(intensity_array, 0)
    mean_values = np.ma.mean(masked_arr, axis=0).filled(0)
    if trim_zeros:
        mean_values = np.trim_zeros(mean_values, 'b')
    return mean_values

def shrink_polygon(vertices, distance):
    """
    Args:
        vertices (list of tuples): vertices of original polygon
        d (float): distance to shrink
    Return:
        shrink_vertices (list of tuples): vertices of the shrunken polygon
    """
    original_polygon = Poly(vertices)
    smaller_polygon = original_polygon.buffer(-distance)
    # Check if shrinking results in a valid polygon
    if smaller_polygon.is_empty:
        return False
    else:
        return list(smaller_polygon.exterior.coords)

def discretize_polygon(vertices, n):
    """
    Discretize a polygon by generating points along each edge.
    Args:
        vertices (list of tuples): Vertices of the polygon as (x, y) pairs.
        n (int): Number of discretization points along each edge.
    Returns:
        list of tuples: Discretized points along the polygon edges.
    """
    discretized_points = []
    # Loop through each edge of the polygon
    for i in range(len(vertices)):
        # Get the start and end points of the edge
        start = np.array(vertices[i])
        end = np.array(vertices[(i + 1) % len(vertices)])  # Wrap around to the first vertex
        # Calculate the points along the edge
        for j in range(n + 1):  # n + 1 to include the endpoint
            # Interpolate between start and end
            point = start + (end - start) * (j / n)
            discretized_points.append(tuple(point))
    return discretized_points

def intensities_from_contours(distances, 
                              vertices,
                              A,
                              mesh,
                              plot=False,
                              axes=False):
    """
    Obtain intensities from contours of smaller polygons
    Args:
        distances (array): distances of contours to evaluate at
        vertices (list of tuples): coordinates of polygon vertices
        A (function): solution of PDE
        mesh (mshr): mesh of domain
    Return: 
        intensities (array): array of intensities from edge to centre
    """
    intensities = np.zeros(len(distances))
    intensities[0] = 1
    for i in tqdm(range(1, len(distances))):
        d_vertices = shrink_polygon(list(vertices), distances[i])
        if not d_vertices:
            continue
        else:
            points_on_polygon = discretize_polygon(d_vertices, 20)
            points = np.array(points_on_polygon)
            if plot:
                axes.plot(points[:,0], points[:,1], alpha=0.5)
            intensity_values = evaluate_solution(points, A, mesh)
            intensities[i] = np.mean(intensity_values)
    intensities = np.trim_zeros(intensities, 'b')
    return intensities

def intensity_from_annulus(distances,
                           A,
                           mesh,
                           Rout,
                           Rin):
    intensities = np.zeros(len(distances))
    x_coord = np.linspace(Rin, Rout, len(distances))
    intensities[0] = 1
    intensities[-1] = 1
    for i in tqdm(range(1, len(distances)-1)):
        intensities[i] = evaluate_point(np.array([x_coord[i], 0]), A, mesh)
    return intensities
