# e2e_v2

This project contains a single Python script that can determine the Curie or Neel Temperature and other relevant magnetic properties of 2D materials from the crystal structure information. The code has many requirements; the primary one is having access to the code [VASP](https://www.vasp.at/). The script is tightly integrated with VASP and relies upon it to perform all the energy calculations. However, with a bit of effort, it should be possible to interface this code with other *ab initio* codes as well. Besides VASP, the code relies heavily on other Python packages such as [pymatgen](https://pymatgen.org/), [custodian](http://materialsproject.github.io/custodian/), [ASE](https://wiki.fysik.dtu.dk/ase/) and [numba](https://numba.pydata.org/). These packages need to be available in your environment to run the script properly. The proper installation instructions for these packages can be found on their respective websites, which I have hyperlinked.
The detailed working principle of the code can be found in the following publications. Please cite these if you find e2e_v2 useful. 

1. Kabiraj, A., Kumar, M. & Mahapatra, S. High-throughput discovery of high Curie point two-dimensional ferromagnetic materials. *npj Comput Mater* **6**, 35 (2020). https://doi.org/10.1038/s41524-020-0300-2
2. Arnab Kabiraj, Tripti Jain, and Santanu Mahapatra. Massive Monte Carlo simulations-guided interpretable learning of two-dimensional Curie temperature. *manuscript under review*.


## Initial preparation

It is best to create a conda or virtual environment for this script. Then the required packages can be installed using pip or conda.
A proper installation of pymatgen is essential. The detailed instructions can be found at https://pymatgen.org/installation.html. You would need to install the enumlib and interface it with pymatgen. Moreover, proper installation of bader is required if you want to perform bader charge and magnetism partitioning. The POTCAR library also needs to be properly linked to pymatgen. See https://pymatgen.org/installation.html#step-4-optional-install-enumlib-and-bader-only-for-osx-and-linux and https://pymatgen.org/installation.html#potcar-setup.

Currently, there is a small but crucial bug in [pymatgen input sets module](https://github.com/materialsproject/pymatgen/blob/master/pymatgen/io/vasp/sets.py) which must be fixed to run e2e_v2. The lines:
```
if "MAGMOM" in self.prev_incar:
    del self.prev_incar["magmom"]
```
should be replaced with:
```
if "MAGMOM" in self.prev_incar:
    del self.prev_incar["MAGMOM"]
```
That's it! We are good to go!


## Inputs

The code needs two input files to run, optionally three if you want to customize the Monte Carlo part. These files must reside in the same directory where the script is being run. The required input files are:
1. input
2. a crystal structure information file that can be specified in the input
3. input_MC (optional)

Most of the operations of the script can be controlled using the input file. A typical input file looks like the following.
```
structure_file = POSCAR
XC_functional = PBE
DFT_supercell_size = 2 2 1
VASP_command_std = mpirun -np 24 vasp_std_631_acc
VASP_command_ncl = mpirun -np 12 vasp_ncl_631_acc
accuracy = high
mag_prec = 0.1
enum_prec = 1e-7
max_neighbors = 4
mag_from = OSZICAR
GPU_accel = True
more_than_2_metal_layers = False
NSIM = 4
KPAR = 2
NCORE = 1
LDAUTYPE = 1
LDAUU = Cr 2.7
LDAUJ = Cr 0.7
same_neighbor_thresh = 0.05
```
I will briefly discuss all the available options for the input file below. Note that there must be a whitespace between the tag,"=", and the value.

* **structure_file**, default = none, must be specified.
The structure file name. The structure files are read through the ase.io.read method and can handle any structure files supported by ASE.

* **DFT_supercell_size**, default = 2 2 1.
The size of the supercell on which all the calculations will be performed. Note that the number of generated magnetic configurations and, in turn, the included number of interacting neighbors directly depends on this parameter. The default might not be enough for all lattice types. For example, in the case of hexagonal cells with coordination 6-6-6-12 (prototype: 2H or 1T Transition Metal Dichalcogenides) I recommend `DFT_supercell_size = 2 4 1`.

* **vacuum**, default = 25, unit = angstrom.
The amount of total vacuum added to the cells to eliminate spurious interaction from the periodically repeated layers. The default is a good choice. If any of the lattice parameters is too large, I suggest watching out for the VASP warning about charge sloshing and reducing the value of AMIN.

* **strain**, default = none, unit = fraction of respective lattice parameter.
The amount of strain on the unit cell. For example, `strain = 0.02 0.02 1` results in a 2% increase of the lattice parameters *a* and *b*, and 
`strain = 0.01 -0.05 1` results in a 1% increase of the lattice parameter *a* and 5% reduction of *b*.

* **VASP_command_std**, default = none, must be specified.
The VASP execution command for the collinear calculations.

* **VASP_command_ncl**, default = none, must be specified.
The VASP execution command for the non-collinear calculations including the spin-orbit coupling effect. 

* **XC_functional**, default = PBE.
The employed exchange-correlation functional. The available options are: `XC_functional = PBE or LDA or SCAN or R2SCAN or SCAN+RVV10 or PBEsol`.

* **randomise_VASP_command**, default = False.
Add a random number after the VASP command while executing. Proper VASP executable symbolic links have to be set up before execution. It might help in identifying and killing processes running on the same machine.

* **skip_configurations**, default = none.
Skip specific configuration(s). It can be used to skip unreasonable configurations. For example, if there are 6 detected configurations, and you would like to skip configurations 2 and 4, simply do `skip_configurations = 2 4`. Note that the remaining configurations would be renamed, so you have to be careful about the respective directories.

* **relax_structures**, default = True.
Whether to relax the structures while locating the ground state configurations. If you are sure of the ground state beforehand, you can simply relax your structure to the ground state on your own, supply that structure to the code, and do `relax_structures = False`. This would perform one-step fake relaxations and speed up the computation. Useful for materials with CDW characteristics.

* **mag_prec**, default = 0.1.
symm_prec (under transformation_kwargs) option for the pymatgen class [MagneticStructureEnumerator](https://pymatgen.org/pymatgen.analysis.magnetism.analyzer.html#pymatgen.analysis.magnetism.analyzer.MagneticStructureEnumerator). Can affect the number of generated configurations.

* **enum_prec**, default = 0.001.
enum_precision_parameter (under transformation_kwargs) option for the pymatgen class [MagneticStructureEnumerator](https://pymatgen.org/pymatgen.analysis.magnetism.analyzer.html#pymatgen.analysis.magnetism.analyzer.MagneticStructureEnumerator). can affect the number of generated configurations.

* **ltol**, default = 0.4.
ltol option for the pymatgen class [StructureMatcher](https://pymatgen.org/pymatgen.analysis.structure_matcher.html#pymatgen.analysis.structure_matcher.StructureMatcher)

* **stol**, default = 0.6.
stol option for the pymatgen class [StructureMatcher](https://pymatgen.org/pymatgen.analysis.structure_matcher.html#pymatgen.analysis.structure_matcher.StructureMatcher)

* **atol**, default = 5, unit = degrees.
angle_tol option for the pymatgen class [StructureMatcher](https://pymatgen.org/pymatgen.analysis.structure_matcher.html#pymatgen.analysis.structure_matcher.StructureMatcher)

* **max_neighbors**, default = 4.
Interactions from how many neighboring shells should be included at maximum. The code can handle up to 4 neighboring shells, which is the default.

* **mag_from**, default = OSZICAR.
From which source the magnetic moment of each magnetic atom should be obtained. The default is `mag_from = OSZICAR`, which means the total magnetism from the OSZICAR file of the FM configuration is divided by the number of magnetic atoms in the supercell. The other option is `mag_from = Bader`, which performs an automated bader magnetism partitioning to determine the moment of the magnetic species.

* **GPU_accel**, default = True.
Whether to accelerate the neighbor mapping process using GPU, if available.

* **more_than_2_metal_layers**, default = True.
Whether the material has more than two metal layers. If True, a neighbor zero padding scheme ensues after the neighbor mapping.

* **dump_spins**, default = False.
Whether to dump the spin lattice state during the Monte Carlo process.

* **ISMEAR**, default = -5 for PBE, 0 for SCAN variants.
The [ISMEAR](https://www.vasp.at/wiki/index.php/ISMEAR) tag from VASP.

* **SIGMA**, default = 0.05.
The [SIGMA](https://www.vasp.at/wiki/index.php/SIGMA) tag from VASP.

* **NSIM**, default = 4.
The [NSIM](https://www.vasp.at/wiki/index.php/NSIM) tag from VASP.

* **KPAR**, default = 1.
The [KPAR](https://www.vasp.at/wiki/index.php/KPAR) tag from VASP.

* **NCORE**, default = 1.
The [NCORE](https://www.vasp.at/wiki/index.php/NCORE) tag from VASP.

* **ISIF**, default = 3.
The [ISIF](https://www.vasp.at/wiki/index.php/ISIF) tag from VASP.

* **IOPT**, default = 3.
The [IOPT](https://theory.cm.utexas.edu/vtsttools/optimizers.html) tag from VTST Code.

* **LUSENCCL**, default = False.
The [LUSENCCL](https://www.vasp.at/wiki/index.php/LUSENCCL) tag from VASP.

* **LDAUJ**, default = {'Co': 0, 'Cr': 0, 'Fe': 0, 'Mn': 0, 'Mo': 0, 'Ni': 0, 'V': 0, 'W': 0,
'Nb': 0, 'Sc': 0, 'Ru': 0, 'Rh': 0, 'Pd': 0, 'Cu': 0, 'Y': 0, 'Os': 0, 'Ti': 0, 'Zr': 0, 'Re': 0, 'Hf': 0, 'Pt':0, 'La':0}, unit = eV.
The [LDAUJ](https://www.vasp.at/wiki/index.php/LDAUJ) values supplied to VASP. For example, to apply *J* = 0.7 eV on an orbital of Cr sites, you have to write `LDAUJ = Cr 0.7`. Multiple such entries are allowed.

* **LDAUU**, default = {'Co': 3.32, 'Cr': 3.7, 'Fe': 5.3, 'Mn': 3.9, 'Mo': 4.38, 'Ni': 6.2, 'V': 3.25, 'W': 6.2,
'Nb': 1.45, 'Sc': 4.18, 'Ru': 4.29, 'Rh': 4.17, 'Pd': 2.96, 'Cu': 7.71, 'Y': 3.23, 'Os': 2.47, 'Ti': 5.89, 'Zr': 5.55,
'Re': 1.28, 'Hf': 4.77, 'Pt': 2.95, 'La':5.3}, unit = eV.
The [LDAUU](https://www.vasp.at/wiki/index.php/LDAUU) values supplied to VASP. For example, to apply *U* = 2.7 eV on an orbital of Cr sites, you have to write `LDAUU = Cr 2.7`. Multiple such entries are allowed.

* **LDAUL**, default = 2.
The [LDAUL](https://www.vasp.at/wiki/index.php/LDAUL) values supplied to VASP. For example, to apply on-site corrections on the d orbital of Cr sites, you have to write `LDAUL = Cr 2`. Multiple such entries are allowed.

* **LDAUTYPE**, default = 2.
The [LDAUTYPE](https://www.vasp.at/wiki/index.php/LDAUTYPE) tag from VASP.

* **POTCAR**, default = [pymatgen MPRelaxSet defaults](https://pymatgen.org/pymatgen.io.vasp.sets.html#pymatgen.io.vasp.sets.MPRelaxSet).
Which variants of POTCAR files to use. For example, to use Cr_pv and Mg_sv pseudopotential variants in the same calculation, you have to write `POTCAR = Cr Cr_pv Mg Mg_sv`. If there are additional species you do not explicitly specify using this tag, the default POTCARS for them will be used.

* **same_neighbor_thresh**, default = 0.05, unit = angstrom.
How far apart sites should be treated under the same neighbor shell. A significantly higher value than the default is required if there is symmetry-breaking in the structure.

* **same_neighbor_thresh_buffer**, default = 0.01, unit = angstrom.
A small buffer value to break any possible ties between the neighbors.

* **accuracy**, default = default.
Accuracy of the DFT calculations. The choices are `accuracy = default or high`.

* **log_filename**, default = log.
The name of the file where the log will be written.

* **kpoints_density_relax**, default = 150 for `accuracy = default` and 300 for `accuracy = high`, unit = angstrom<sup>-3</sup>.
The kpoints sampling density for relaxations

* **kpoints_density_static**, default = 300 for `accuracy = default` and 1000 for `accuracy = high`, unit = angstrom<sup>-3</sup>.
The kpoints sampling density for all static energy calculations


Additionally, the MC process can be controlled with the input_MC file. A default file is written by the e2e_v2 script, which can be further tweaked. A typical input_MC file looks like the following.
```
directory = MC_Heisenberg
repeat = 25 25 1
restart = 0
J1 (eV/link) = 0.00216527055555559
J2 (eV/link) = 0.000325236805555542
J3 (eV/link) = -4.20614814813863e-5
J4 (eV/link) = 0
K1x (eV/link) = 0.000130227638888675
K1y (eV/link) = 0.000130227638888675
K1z (eV/link) = 9.24354166665087e-5
K2x (eV/link) = 2.37510416666517e-5
K2y (eV/link) = 2.37510416666517e-5
K2z (eV/link) = 3.16604166666901e-6
K3x (eV/link) = -0.000119818379629752
K3y (eV/link) = -0.000119818379629752
K3z (eV/link) = 8.29953703526676e-7
K4x (eV/link) = 0.0
K4y (eV/link) = 0.0
K4z (eV/link) = 0.0
Ax (eV/mag_atom) = 0.185448137430556
Ay (eV/mag_atom) = 0.185448137430556
Az (eV/mag_atom) = 0.185570837708333
mu (mu_B/mag_atom) = 3.0
EMA = 2
T_start (K) = 1e-6
T_end (K) = 80
div_T = 81
MCS = 100000
thresh = 10000
```
I will briefly discuss the available options.

* **directory**, default = MC_Heisenberg.
The directory name where all the MC output files would be written.

* **repeat**, default = 25 25 1.
The size of the supercell on which the MC simulation will be performed.

* **restart** , defult = 0.
Whether to read the neighbor maps from dumped files. `restart = 0` make the code figure out the neighbor maps from scratch, and `restart = 1` makes it look for neighbor map dumps in a subdirectory named "MC_Heisenberg", and read it if found.

* **T_start (K)**, default = 1e-6.
Starting temperature for the MC simulation.

* **T_end (K)**, default = a mean field estimation of the Curie temperature.
Final temperature for the MC simulation.

* **div_T**, default = 25.
How many temperature points to run the MC on. The interval between T_start and T_end is equally divided by these many points.

* **MCS**, default = 100000.
How many total Monte Carlo steps to run for each temperature point.

* **thresh**, default = 10000.
For how many steps the system is allowed to thermalize. The properties like magnetism and susceptibility are calculated as the average of the last `MCS - thresh` steps.


## Outputs

The main outputs of the code are written to a log file (the name specified by the `log_filename` tag) in addition to the outputs printed to stdout and stderr. The full process of extraction of the magnetic parameters from the DFT data is detailed in this file. There would also be DFT calculation directories, namely *relaxations*, *static_runs*, and *MAE*. The *MC_Heisenberg* directory (or the directory specified by the `directory` tag of input_MC) contains all the outputs and dump files from the MC process. The main MC output data is written to a file with a name ending with *M-X.dat*. This file should have data with 7 columns. The first column would be the temperature data. The second and third columns contain the magnetization and susceptibility data of the regions with up spins at the ground state. The fourth and fifth columns contain the magnetization and susceptibility data of the regions with down spins at the ground state. Finally, the sixth and seventh columns contain the magnetization and susceptibility data of the whole lattice. To determine either Curie or Neel temperature, one simply has to find at what temperature the data at the 3rd column peaks.

A typical example of the calculation of Curie temperature of CrI<sub>3</sub> with *U* = 2.7 eV and *J* = 0.7 eV can be found in the [figshare repository of the 2nd paper](https://doi.org/10.6084/m9.figshare.20439309).


## License

e2e_v2 is released under the MIT License. The terms of the license are as follows:

>Copyright (c) 2020-2022 Arnab Kabiraj
>
>Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
>
>The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
>
>THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.