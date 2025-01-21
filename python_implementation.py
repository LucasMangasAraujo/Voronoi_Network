import os, sys
sys.path.append('../')
import numpy as np
from utilities import *
from collections import defaultdict
del run_relaxation, runinc
import time
from lammps import lammps


def main():
    
    # Read geometry and provide chain parametes
    geomfile = '../Geometries/500_nodes.txt'
    Nodes, Bonds, Boundary, BondTypes = readGeometry(geomfile);
    chain_type = '2';
    computational_bKuhn, NKuhn = 0.0464159, 100;
    cell_K = 1;
    
    # Write data file considering periodic boundary conditions and generate periodic counterpart
    BondTypes = {idx: NKuhn for idx in BondTypes.keys()} ## make all chains with same length
    writePositions('test.dat',Nodes,Bonds,Boundary,BondTypes, chain_type, [computational_bKuhn]) ## rewrite the dat file
    write_periodic_file(Nodes, Bonds, Boundary, 'test.dat')
    periodic_Bonds, _ = getUpdatedBonds('periodic.dat', dim = 2);
    
    
    # Performing first voronoi calculation, obtain delaunay triangles and compute initial areas
    coords, box_bounds = read_data_file('periodic.dat');
    vor, delaunay, periodic_coords, periodic_indices = create_periodic_tessellations(coords, box_bounds);
    neighbours, periodic_neighbours = get_voronoi_neighbors_optimized(vor, periodic_indices);
    triangles_tuple = [tuple(sorted(triangle)) for triangle in delaunay.simplices];
    cell_vertices, cell_triangles = find_cell_vertices_and_triangles(triangles_tuple, periodic_coords, 
                                                                        len(coords), periodic_indices);
    initial_areas = find_cell_areas(cell_vertices);
    total_area = np.sum(tuple(initial_areas.values()));
    
    if np.isclose(total_area, 1.0):
        print("Cell areas add up to one :)");
    else:
        print("Cell areas do not add up to one!!!!!!!!")
        breakpoint()
    
    # Initialise lammps instance
    lmp = lammps();
    
    breakpoint()
    
    return

def run_increment(lmp, mainfile):
    
    
    return


if __name__ == "__main__":
    main()
