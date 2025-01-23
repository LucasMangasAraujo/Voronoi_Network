import os, sys
sys.path.append('../')
import numpy as np
from utilities import *
from collections import defaultdict
del run_relaxation, runinc
import time
from lammps import lammps, PyLammps
from voronoi_energy import VoronoiEnergyFix


def main():
    
    # Read geometry and provide chain parametes
    geomfile = '../Geometries/700_nodes.txt'
    Nodes, Bonds, Boundary, BondTypes = readGeometry(geomfile);
    chain_type = '2';
    computational_bKuhn, NKuhn = 0.0793701, 100;
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
    
    
    # Run withiut volumetric effects
    lmp = lammps(); ## Initialisation of lammps instance
    run_FJC(lmp, 'periodic.dat',chain_type);
    lmp.close()
    
    # Initialise external force call back
    #def external_force_callback(caller_ptr, timestep, nlocal, tag, x, f):
    def external_force_callback(lmp, timestep, nlocal, tag, x, f):
        forces, energies = voronoi_fix.compute_voronoi_energy_forces();
        
        for i in range(nlocal):
            f[i][0] += forces[i][0]
            f[i][1] += forces[i][1]
            f[i][2] += 0.0  # 2D system
            
        
        #lmp.fix_external_set_energy_peratom("voronoi",energies)
        
    # Run considering volumetric effects
    lmp = lammps();
    voronoi_fix = VoronoiEnergyFix(lmp, bulk_modulus = cell_K, initial_areas = initial_areas); ## Create fix
    run_commands(lmp, 'periodic.dat',chain_type, external_force_callback)
    lmp.close();
    breakpoint()
    
    
    
    return

def run_commands(lmp, posfile, chain_type, external_force_callback,dim = 2):
    
    min_algo='fire' #algorithm for minimization
    dmax = 0.05
    
    if chain_type == '2':
        chain_name = 'langevin';

    ## Write general information
    lmp.command('#Main input file for LAMMPS\n');
    lmp.command('units\tlj\n');
    lmp.command('dimension\t%d\n' %dim);
    
    ## Write boundary conditons
    lmp.command('boundary\tp p p\n')
    
    ## Write styles
    lmp.command('atom_style\tbond\n')
    lmp.command('bond_style\t %s\n' %chain_name)
    lmp.command('atom_modify\tsort 0 0\n')
    lmp.command('pair_style\tnone\n\n')

    lmp.command('read_data\t%s\n\n' %posfile)

    lmp.command('reset_timestep\t0\n');
    lmp.command('timestep\t0.0001\n');
    lmp.command('neighbor\t0.1 nsq\n');
    
    ## Write the information that will be printed in the log file
    lmp.command('thermo\t1\n');
    lmp.command('thermo_style\t custom step etotal pe ebond press pxx pyy pxy\n')
    
    ## Write minisiation files
    lmp.command('min_style\t%s\n' %(min_algo));
    lmp.command('min_modify\tdmax %s\n\n' %(dmax));
    
    
    ## Write dump informatio for real-time geometry during minimization
    lmp.command('# Dump command to output atomic configurations to the same file\n');
    lmp.command('dump\t 1 all custom 1 dump.minimization.xyz id type x y z\n')
    lmp.command('dump_modify\t 1 sort id\n')
    lmp.command('\n')
    
    ## Write the fix part, where the deformaito is done
    
    lmp.command("fix\t 1 all deform 1 x delta 0 0 y volume remap x units box\n");
    lmp.command("fix\t voronoi all external pf/callback 1 1")
    lmp.set_fix_external_callback("voronoi", external_force_callback, lmp);
    lmp.command("run\t 1\n");
    lmp.command("\n");
    
    lmp.command("minimize\t 0 1e-16 1000 10000\n");
    
    ##  Conclude and write to other file
    lmp.command("# Stop the dump after minimization\n");
    lmp.command("undump\t 1\n");
    lmp.command("\n");
    lmp.command("run\t 0\n");
    lmp.command("\n");
    lmp.command("write_data\t test_out.dat\n");
    
    return


def run_FJC(lmp, posfile, chain_type, dim = 2):
    
    min_algo='fire' #algorithm for minimization
    dmax = 0.05
    
    if chain_type == '2':
        chain_name = 'langevin';

    ## Write general information
    lmp.command('#Main input file for LAMMPS\n');
    lmp.command('units\tlj\n');
    lmp.command('dimension\t%d\n' %dim);
    
    ## Write boundary conditons
    lmp.command('boundary\tp p p\n')
    
    ## Write styles
    lmp.command('atom_style\tbond\n')
    lmp.command('bond_style\t %s\n' %chain_name)
    lmp.command('atom_modify\tsort 0 0\n')
    lmp.command('pair_style\tnone\n\n')

    lmp.command('read_data\t%s\n\n' %posfile)

    lmp.command('reset_timestep\t0\n');
    lmp.command('timestep\t0.0001\n');
    lmp.command('neighbor\t0.1 nsq\n');
    
    ## Write the information that will be printed in the log file
    lmp.command('thermo\t1\n');
    lmp.command('thermo_style\t custom step etotal pe ebond press pxx pyy pxy\n')
    
    ## Write minisiation files
    lmp.command('min_style\t%s\n' %(min_algo));
    lmp.command('min_modify\tdmax %s\n\n' %(dmax));
    
    
    ## Write dump informatio for real-time geometry during minimization
    lmp.command('# Dump command to output atomic configurations to the same file\n');
    lmp.command('dump\t 1 all custom 1 dump.minimization.xyz id type x y z\n')
    lmp.command('dump_modify\t 1 sort id\n')
    lmp.command('\n')
    
    ## Write the fix part, where the deformaito is done
    
    lmp.command("fix\t 1 all deform 1 x delta 0 0 y volume remap x units box\n");
    lmp.command("run\t 1\n");
    lmp.command("\n");
    
    lmp.command("minimize\t 0 1e-16 1000 10000\n");
    
    ##  Conclude and write to other file
    lmp.command("# Stop the dump after minimization\n");
    lmp.command("undump\t 1\n");
    lmp.command("\n");
    lmp.command("run\t 0\n");
    lmp.command("\n");
    lmp.command("write_data\t test_FJC.dat\n");
    
    return


if __name__ == "__main__":
    main()
