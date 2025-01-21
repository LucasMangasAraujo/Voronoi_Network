import sys
sys.path.append('../')
import numpy as np
from utilities import  create_periodic_tessellations
from math import pow

class VoronoiEnergyFix:
    
    
    def __init__(self, lmp, bulk_modulus, initial_areas):
        self.lmp = lmp;
        self.bulk_modulus = bulk_modulus;
        self.initial_areas = initial_areas;
        
        
    
    
    
    def get_positions(self):
        x = self.lmp.numpy.extract_atom("x", 3);
        nAtoms = self.lmp.get_natoms();
        twoD_positions = [tuple(x[i][0], x[i][1]) for i in range(nAtoms)];
        
        return twoD_positions
    
    def get_box_bounds(self):
        
        simulation_box = self.lmp.extract_box();
        box_bounds['x'] = [simulation_box[0][0], simulation_box[1][0]];
        box_bounds['y'] = [simulation_box[0][1], simulation_box[1][1]];
        box_bounds['z'] = [simulation_box[0][2], simulation_box[1][2]];
        
        
        return box_bounds
    
    
    
    def compute_voronoi_energy_forces(self):
        
        # Get the 2D position of the crosslinks and box bounds
        positions = self.get_positions();
        box_bounds = self.get_box_bounds();
        forces = np.zeros_like(positions);
        
        # Generate periodic tesselation
        vor, delaunay, periodic_coords, periodic_indices = create_periodic_tessellations(positions, box_bounds)
        
        # Get list of neighbors of each voronoi cell
        neighbours, periodic_neighbours = get_voronoi_neighbors_optimized(vor, periodic_indices);
        
        # Get triangles and cell vertices
        triangles_tuple = [tuple(sorted(triangle)) for triangle in delaunay.simplices];
        cell_vertices, cell_triangles = find_cell_vertices_and_triangles(triangles_tuple, periodic_coords, 
                                                                        len(coords), periodic_indices);
        
        # Calculare cell areas and declare normal vector to the plane
        cell_areas = self.polygon_area(cell_vertices);
        K = self.bulk_modulus; ## Variable to store the bulk modulus of the cells
        Nz = np.array([0.0, 0.0, 1.0]); ## Normal to the plane
        
        # Loop over the cell centres atoms
        for i in range(len(positions)):
            ## Initialise force to be applied on node i
            force_at_node_i = 0.0;
            
            ## Get vertices and neighbours of node i
            vertices_of_node_i = self.ccw_polygon_vertices(cell_vertices[i]);
            neighbours_of_node_i = neighbours[i];
            A_i = cell_areas[i];
            A_i0 = self.initial_areas[i];
            
            ## Loop over the vertices 
            area_diff_i = 0.5 * K * (A_i - A_i0); ## scalar factor of multiplying vector
            vector = np.zeros_like(Nz);
            cross_product = np.zeros_like(Nz);
            
            for idx, mu in vertices_of_node_i:
                ## Get adjcent nodes
                mu_minus = vertices_of_node_i[idx - 1];
                mu_plus = vertices_of_node_i[(idx + 1) % len(vertices_of_node_i)];
                vector[0] = mu_plus[0] - mu_minus[0];
                vector[1] = mu_plus[1] - mu_minus[1];
                cross_product[0] = vector[1];
                cross_product[1] = -vector[0];
                
        
        return forces
    
    
    
    
    @staticmethod
    def polygon_area(vertices):
        # matrix_with_vertices = np.array(vertices);
        # centroid = np.mean(matrix_with_vertices, axis = 0);
        # angles = np.arctan2(vertices[:, 1] - centroid[1], vertices[:, 0] - centroid[0])
        # ccw_vertices = vertices[np.matrix_with_vertices(angles)];
        ccw_vertices = self.ccw_polygon_vertices(vertices);
        x, y = ordered_vertices[:, 0], ordered_vertices[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)));
        
        return area
        
    @staticmethod
    def ccw_polygon_vertices(vertices):
        matrix_with_vertices = np.array(vertices);
        centroid = np.mean(matrix_with_vertices, axis = 0);
        angles = np.arctan2(vertices[:, 1] - centroid[1], vertices[:, 0] - centroid[0])
        ccw_vertices = vertices[np.matrix_with_vertices(angles)];
        
        return ccw_vertices