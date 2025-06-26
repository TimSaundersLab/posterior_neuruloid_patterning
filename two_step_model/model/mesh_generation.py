# To make meshes of different shapes
from fenics import *
from mshr import *
import numpy as np
from scipy.interpolate import splprep, splev

def circle(R, res=20):
    domain = Circle(Point(0,0), R)
    mesh = generate_mesh(domain, res)
    return mesh

def annulus(Rout, Rin, res=20):
    large_circle = Circle(Point(0,0), Rout)
    small_circle = Circle(Point(0,0), Rin)
    domain = large_circle - small_circle
    mesh = generate_mesh(domain, res)
    return mesh

def equilateral_triangle(length, res=20, return_vertices=False):
    vertices = [(0, 0), (length, 0), (0.5*length, length*sqrt(3)/2)]
    domain = Polygon([Point(p) for p in vertices])
    mesh = generate_mesh(domain, res)
    if return_vertices:
        return mesh, vertices
    else:
        return mesh

def square(length, res=20, return_vertices=False):
    vertices = [(0, 0), (length, 0), (length, length), (0, length)]
    domain = Polygon([Point(p) for p in vertices])
    mesh = generate_mesh(domain, res)
    if return_vertices:
        return mesh, vertices
    else:
        return mesh

def rectangle(long_length, short_length, res=20, return_vertices=False):
    vertices = [(0,0), (short_length, 0), (short_length, long_length), (0, long_length)]
    domain = Polygon([Point(p) for p in vertices])
    mesh = generate_mesh(domain, res)
    if return_vertices:
        return mesh, vertices
    else:
        return mesh

def four_pointed_star(length, res=20, return_vertices=False):
    vertices = [(0,0), (length/2, -1.5*length), (length,0), 
                (2.5*length, length/2), (length,length), (length/2, 2.5*length),
                (0,length), (-1.5*length, length/2)]
    domain = Polygon([Point(p) for p in vertices])
    mesh = generate_mesh(domain, res)
    if return_vertices:
        return mesh, vertices
    else:
        return mesh 

def wavy_star(octagon_length, n=100, res=20, return_vertices=False):
    R = octagon_length
    x_coords = np.array([1.6*R, R, 0.9*R*np.cos(np.pi/4), R*np.tan(np.pi/8), 0, 
                         -R*np.tan(np.pi/8), -0.9*R*np.cos(np.pi/4), -R, -1.6*R,
                         -R, -0.9*R*np.cos(np.pi/4), -R*np.tan(np.pi/8), 0,
                         R*np.tan(np.pi/8), 0.9*R*np.cos(np.pi/4), R])
    x_coords = np.r_[x_coords, x_coords[0]]
    y_coords = np.array([0, R*np.tan(np.pi/8), 0.9*R*np.sin(np.pi/4), R, 1.6*R,
                         R, 0.9*R*np.sin(np.pi/4), R*np.tan(np.pi/8), 0, 
                         -R*np.tan(np.pi/8), -0.9*R*np.sin(np.pi/4), -R, -1.6*R,
                         -R, -0.9*R*np.sin(np.pi/4), -R*np.tan(np.pi/8)])
    y_coords = np.r_[y_coords, y_coords[0]]
    tck, u = splprep([x_coords,y_coords], s=0, per=True)
    t = np.linspace(0, 1, n)
    xi, yi = splev(t, tck)
    area = mesh_area(xi, yi)
    points = np.column_stack((xi, yi))
    vertices = [tuple(p) for p in points]
    domain = Polygon([Point(p) for p in vertices])
    mesh = generate_mesh(domain, res)

    if return_vertices:
        return mesh, vertices
    else:
        return area, mesh

def mesh_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

