import sys
sys.path.append('../')
import numpy as np
from utilities import  create_periodic_tessellations
from scipy.spatial import KDTree
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
        energies = np.zeros(len(positions));
        
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
            force_at_node_i = np.zeros(3, );
            
            ## Get vertices,neighbours and triagles of node i
            vertices_of_node_i = self.ccw_polygon_vertices(cell_vertices[i]);
            node_i_tree = KDTree(vertices_of_node_i); ## Tree for efficient query
            neighbours_of_node_i = neighbours[i];
            triangles_of_node_i = cell_triangles[i];
            
            ## Precompute areal factor for the fprce
            A_i = cell_areas[i];
            A_i0 = self.initial_areas[i];
            area_diff = 0.5 * K * (A_i - A_i0); ## scalar factor of multiplying vector information of the force
            energies[i] = 0.5 * K * pow(A_i - A_i0, 2);
            
            ## Find correspondance between the cell vertices and their delaunay triangles
            vertices_triangles, triangles_edge_lengths, baricentric_parameters = self.vertices_triangles_correspondance(triangles_of_node_i, vertices_of_node_i, periodic_coords)
            
            ## Loop over the vertices of node i
            vector = np.zeros(3, );
            cross_product = np.zeros_like(vector);
            
            for idx, mu in vertices_of_node_i:
                ## Get adjcent nodes and perform cross product calculation
                mu_minus = vertices_of_node_i[idx - 1];
                mu_plus = vertices_of_node_i[(idx + 1) % len(vertices_of_node_i)];
                vector[0] = mu_plus[0] - mu_minus[0];
                vector[1] = mu_plus[1] - mu_minus[1];
                cross_product[0] = vector[1];
                cross_product[1] = -vector[0];
                
                
                ## Extract delaunay triangle information
                triangle = vertices_triangles[idx];
                edge_lengths = triangles_edge_lengths[idx];
                baricentric_par = baricentric_parameters[idx];
                
                ## Calculare jacobian
                drmu_dri = compute_jacobian(triangle, edge_lengths, baricentric_par, periodic_coords)
                
                ## Calculate force contribution, and add it to the net force in node i
                force_vector = np.zeros_like(vector);
                for m in range(3):
                    for p in range(3):
                        force_vector[m] += cross_product[p] * drmu_dri[p][m];
                        
                    force_at_node_i[m] += force_vector[m];
                
                force_at_node_i *= -area_diff;
            
            
            ## Loop over the neighbors of node i and find out share vertices

            for neighbour in neighbours_of_node_i:
                ## Extract the vertices of neighbour, and its triangles
                vertices_of_neighbour = cell_vertices[periodic_indices[neighbour]];
                triangles_of_neighbour = cell_triangles[periodic_indices[neighbour]];
                
                ## Precompute areal factors
                A_j = cell_areas[neighbour];
                A_j0 = self.initial_areas[neighbour];
                area_diff = 0.5 * K * (A_j - A_j0);
                force_contrinution_j = np.zeros_like(force_at_node_i); ## Neighbour contribution to net force at node i
                
                ## Find correspondance between the cell vertices and their delaunay triangles for neighbour
                vertices_triangles, triangles_edge_lengths, baricentric_parameters = self.vertices_triangles_correspondance(triangles_of_neighbour, vertices_of_neighbour, periodic_coords)
                
                for idx, mu in enumerate(vertices_of_neighbour):
                    ## Check if vertex is share by neighbour j and node id
                    distance, _ = node_i_tree.query(mu);
                    if distance > 1e-6: ## Vertex mu is not shared
                        continue
                        
                    ## If vertex mu is shared, procees with calculations
                    mu_minus = vertices_of_neighbour[idx - 1];
                    mu_plus = vertices_of_neighbour[(idx + 1) % len(vertices_of_neighbour)];
                    vector[0] = mu_plus[0] - mu_minus[0];
                    vector[1] = mu_plus[1] - mu_minus[1];
                    cross_product[0] = vector[1];
                    cross_product[1] = -vector[0];
                    
                    ## Extract delaunay triangle information
                    triangle = vertices_triangles[idx];
                    edge_lengths = triangles_edge_lengths[idx];
                    baricentric_par = baricentric_parameters[idx];
                    drmu_drj = compute_jacobian(triangle, edge_lengths, baricentric_par, periodic_coords);
                    
                    ## Calculate force contribution due to neighbour, and add it to the net force in node i
                    force_vector = np.zeros_like(vector);
                    for m in range(3):
                        for p in range(3):
                            force_vector[m] += cross_product[p] * drmu_drj[p][m];
                            
                        force_contrinution_j[m] += force_vector[m];
                    
                    force_contrinution_j *= -area_diff;
                    
                
                ## Add contribution of node j to net force at node id
                force_at_node_i += force_contrinution_j;
            
            forces[i] = force_at_node_i;
            
        return forces
    
    
    
    
    @staticmethod
    def compute_jacobian(triangle, edge_lengths, baricentric_par, periodic_coords):
        # Extract node numbering from the triangle tuple
        i, j, k = triangle;
        
        # Extract coordinates of triangle nodes
        r_i = np.array([periodic_coords[i][0], periodic_coords[i][1], 0.0]);
        r_j = np.array([periodic_coords[j][0], periodic_coords[j][1], 0.0]);
        r_k = np.array([periodic_coords[k][0], periodic_coords[k][1], 0.0]);
        
        # Calculate edge vectors
        r_ij = r_i - r_j;
        r_jk = r_j - r_k;
        r_ki = r_k - r_i;
        
        # Extract edge lengths and baricentric information
        l_i2, l_j2, l_k2 = edge_lengths;
        lambda_1, lambda_2, lambda_3, Lambda = baricentric_par;
        
        # Compute some auxiliar derivatives associated with node i
        dlambda1_dri = 2 * l_i2 * (-r_ki + r_ij);
        dlambda2_dri = (-2 * (l_i2 + l_k2 - (2 * l_j2)) * r_ki) + (2 * l_j2 * r_ij);
        dlambda3_dri = (2 * (l_i2 + l_j2 - (2 * l_k2)) * r_ij) - (2 * l_k2 * r_ki);
        
        # Compute some auxiliar derivatives associated with node j
        dlambda1_drj = (2 * (l_j2 + l_k2 - (2 * l_i2)) * r_jk) - (2 * l_i2 * r_ij);
        dlambda2_drj = 2 * l_j2 * (r_jk - r_ij);
        dlambda3_drj = (-2 * (l_i2 + l_j2 - (2 * l_k2)) * r_ij) + (2 * l_k2 * r_jk);
        
        # Compute some auxiliar derivatives associated with node k
        dlambda1_drk = (-2 * (l_j2 + l_k2 - (2 * l_i2)) * r_jk) + (2 * l_i2 * r_ki);
        dlambda2_drk = (2 * (l_i2 + l_k2 - (2 * l_j2)) * r_ki) - (2 * l_j2 * r_jk);
        dlambda3_drk = 2 * l_k2 * (-r_jk + r_ki);
        
        # Compute some auxiliar derivatives associated with Lambda
        dLamdbda_dri = (-4 * (l_i2 + l_k2 - l_j2) * r_ki) + (4 * (l_i2 + l_j2 - l_k2) * r_ij);
        dLamdbda_drj = (4 * (l_j2 + l_k2 - l_i2) * r_jk) - (4 * (l_i2 + l_j2 - l_k2) * r_ij);
        dLamdbda_drk = (-4 * (l_j2 + l_k2 - l_i2) * r_jk) + (4 * (l_i2 + l_k2 - l_j2) * r_ki);
        
        # Compute now auxiliar derivative d(lambda_q/Lambda)_drp, q = 1, 2, 3 and p = i, j, k
        dlambda1_over_Lambda_dri = ((Lambda * dlambda1_dri) - (lambda_1 * dLamdbda_dri)) / pow(Lambda, 2);
        dlambda2_over_Lambda_dri = ((Lambda * dlambda2_dri) - (lambda_2 * dLamdbda_dri)) / pow(Lambda, 2);
        dlambda3_over_Lambda_dri = ((Lambda * dlambda3_dri) - (lambda_3 * dLamdbda_dri)) / pow(Lambda, 2);
        
        
        # Finally compute drmu_dri
        drmu_dri = np.zeros((3, 3),)
        I = np.zeros_like(drmu_dri);
        I[0][0], I[1][1], I[2][2] = 1.0, 1.0, 1.0;
        for m in range(3):
            for p in range(3):
                drmu_dri[m][p] = r_i[m] * dlambda1_over_Lambda_dri[p] + (lambda_1 * I[m][p] / Lambda) + 
                                 r_j[m] * dlambda2_over_Lambda_dri[p] + r_k[m] * dlambda3_over_Lambda_dri[p];
        
        return drmu_dri
    
    @staticmethod
    def polygon_area(vertices):
        # matrix_with_vertices = np.array(vertices);
        # centroid = np.mean(matrix_with_vertices, axis = 0);
        # angles = np.arctan2(vertices[:, 1] - centroid[1], vertices[:, 0] - centroid[0])
        # ccw_vertices = vertices[np.matrix_with_vertices(angles)];
        ccw_vertices = self.ccw_polygon_vertices(vertices);
        x, y = ccw_vertices[:, 0], ccw_vertices[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)));
        
        return area
        
    @staticmethod
    def ccw_polygon_vertices(vertices):
        matrix_with_vertices = np.array(vertices);
        centroid = np.mean(matrix_with_vertices, axis = 0);
        angles = np.arctan2(vertices[:, 1] - centroid[1], vertices[:, 0] - centroid[0])
        ccw_vertices = vertices[np.argsort(angles)];
        
        return ccw_vertices
        
        
    @staticmethod
    def ccw_polygon_indices(vertices):
        matrix_with_vertices = np.array(vertices);
        centroid = np.mean(matrix_with_vertices, axis = 0);
        angles = np.arctan2(vertices[:, 1] - centroid[1], vertices[:, 0] - centroid[0])
        ccw_indices = np.argsort(angles);
        
        return ccw_indices
    
    @staticmethod
    def vertices_triangles_correspondance(triangles_of_node_i, vertices_of_node_i, periodic_coords):
        # Initialise some outputs
        triangles_edge_lengths = {}; ## Initialisation of the lengths of each triangle
        baricentric_parameters = {}; ## Intialisation of the parameters to compute baricentric centre.
        vertices_triangles = {}; ## Triangle-vertices association
        
        # Loop over the triangles and reoder its nodes in counterclock_wise fashion
        ccw_triangles = [];
        for triangle in triangles_of_node_i:
            triangle_coords = [periodic_coords[idx] for idx in triangle];
            ccw_indices = self.ccw_polygon_indices(triangle_coords);
            i, j, k = triangle[ccw_indices[0]], triangle[ccw_indices[1]], triangle[ccw_indices[2]];
            ccw_triangles.append((i, j, k));
            
        # Obtain the baricentres and store them in a KDTree
        triangle_barycentres = [];
        for triangle in ccw_triangles:
            ## Get coordinates of triangle nodes
            r_i, r_j, r_k = (periodic_coords[triangle[0]], 
                             periodic_coords[triangle[1]], 
                             periodic_coords[triangle[2]])
                             
            
            ## Obtain the edge lengths 
            l_i2 = np.linalg.norm(r_j - r_k)**2
            l_j2 = np.linalg.norm(r_k - r_i)**2
            l_k2 = np.linalg.norm(r_i - r_j)**2
            
            ## Obtain baricentric centers
            lambda_1 = l_i2 * (l_j2 + l_k2 - l_i2);
            lambda_2 = l_j2 * (l_k2 + l_i2 - l_j2);
            lambda_3 = l_k2 * (l_i2 + l_j2 - l_k2);
            Lambda = lambda_1 + lambda_2 + lambda_3;
            
            ## Obtain baricentre and store
            baricentre = (lambda_1 * r_i + lambda_2 * r_j + lambda_3 * r_k) / Lambda
            triangle_barycentres.append(baricentre);
        
        kdtree = KDTree(triangle_barycentres); ## KDTree with baricentres
        
        # Assign now triangles to their correspondinc vertices
        assigned_ids = set();
        for idx, vertex in enumerate(vertices_of_node_i):
            ## Check whether triangle has already been assigned to a vertex
            _, candidate_triangle_id = kdtree.query(vertex);
            if candidate_triangle_id in assigned_ids: 
                continue
            
            ## Assemble coordinates
            r_i = periodic_coords[triangle[0]];
            r_j = periodic_coords[triangle[1]];
            r_k = periodic_coords[triangle[2]];
            
            ## Obtain the edge lengths 
            l_i2 = np.linalg.norm(r_j - r_k)**2
            l_j2 = np.linalg.norm(r_k - r_i)**2
            l_k2 = np.linalg.norm(r_i - r_j)**2
            
            ## Obtain baricentric centers
            lambda_1 = l_i2 * (l_j2 + l_k2 - l_i2);
            lambda_2 = l_j2 * (l_k2 + l_i2 - l_j2);
            lambda_3 = l_k2 * (l_i2 + l_j2 - l_k2);
            Lambda = lambda_1 + lambda_2 + lambda_3;
            
            ## Assemble baricentre
            baricentre = (lambda_1 * r_i / Lambda) + (lambda_2 * r_j / Lambda) + (lambda_3 * r_k / Lambda)
            
            ## Check if baricentre and vertex are identical
            if np.linalg.norm(baricentre - vertex) < 1e-6:
                vertices_triangles[idx] = triangle;
                triangles_edge_lengths[idx] = l_i2, l_j2, l_k2;
                baricentric_parameters[idx] = lambda_1, lambda_2, lambda_3, Lambda;
                assigned_ids.add(candidate_triangle_id);
                
        
        return vertices_triangles, triangles_edge_lengths, baricentric_parameters;