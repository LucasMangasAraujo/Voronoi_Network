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
    
    
    @staticmethod
    def polygon_area(vertices):
        matrix_with_vertices = np.array(vertices);
        centroid = np.mean(matrix_with_vertices, axis = 0);
        angles = np.arctan2(vertices[:, 1] - centroid[1], vertices[:, 0] - centroid[0])
        ccw_vertices = vertices[np.matrix_with_vertices(angles)];
        x, y = ordered_vertices[:, 0], ordered_vertices[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)));
        
        return area