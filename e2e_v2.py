#!/home/arnab/atomate/pyenv395/bin/python

from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet, MPSOCSet
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.magnetism.analyzer import MagneticStructureEnumerator, CollinearMagneticStructureAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.vasp.inputs import Incar, Kpoints
from pymatgen.io.vasp.outputs import Vasprun, Chgcar, Oszicar, Outcar, Potcar
from pymatgen.command_line.bader_caller import bader_analysis_from_objects, bader_analysis_from_path
from pymatgen.io.ase import AseAtomsAdaptor
from custodian.custodian import Custodian
from custodian.vasp.handlers import VaspErrorHandler, UnconvergedErrorHandler, FrozenJobErrorHandler,\
    MeshSymmetryErrorHandler, PositiveEnergyErrorHandler, StdErrHandler, NonConvergingErrorHandler, PotimErrorHandler 
from custodian.vasp.jobs import VaspJob
from custodian.vasp.validators import VasprunXMLValidator
import sys
import os
from shutil import copyfile
import datetime
from time import time, sleep
from ase.io import read, write
from ase.build import make_supercell, sort
import numpy as np
from sympy import Symbol, linsolve
import math
from numba import jit, cuda
from pickle import load, dump


__author__ = "Arnab Kabiraj"
__copyright__ = "Copyright 2022, NSDRL, IISc Bengaluru"
__credits__ = ["Arnab Kabiraj"]


root_path = os.getcwd()
start_time_global = time()

xc = 'PBE'
vacuum = 25
rep_DFT = [2, 2, 1]
strain = []
mag_prec = 0.1
enum_prec = 0.001
max_neigh = 4
randomise_cmd = False
mag_from = 'OSZICAR'
relx = True
GPU_accel = False
padding = True
dump_spins = False
nsim = 4
kpar = 1
ncore = 1
ismear = -5
sigma = 0.05
isif = 3
iopt = 3
nccl = False
d_thresh = 0.05
d_buff = 0.01
acc = 'default'
LDAUJ_provided = {}
LDAUU_provided = {}
LDAUL_provided = {}
potcar_provided = {}
ldautype = 2
log_file = 'log'
skip = []
kpt_den_relx = None
kpt_den_stat = None
ltol = 0.4
stol = 0.6
atol = 5


with open('input') as f: 
    for line in f:
        row = line.split()
        if 'structure_file' in line:
            struct_file = row[-1]
        elif 'DFT_supercell_size' in line:
            rep_z = int(row[-1])
            rep_y = int(row[-2])
            rep_x = int(row[-3])
            rep_DFT = [rep_x, rep_y, rep_z]
        elif 'vacuum' in line:
            vacuum = float(row[-1])
        elif 'strain' in line:
            strain_c = float(row[-1])
            strain_b = float(row[-2])
            strain_a = float(row[-3])
            strain = [strain_a, strain_b, strain_c]
        elif 'XC_functional' in line:
            xc = row[-1]
        elif 'VASP_command_std' in line:
            cmd = line[len('VASP_command_std =')+1:-1].split()
        elif 'VASP_command_ncl' in line:
            cmd_ncl = line[len('VASP_command_ncl =')+1:-1].split()
        elif 'randomise_VASP_command' in line:
            randomise_cmd = row[-1]=='True'
        elif 'skip_configurations' in line:
            skip = line[len('skip_configurations =')+1:-1].split()
        elif 'relax_structures' in line:
            relx = row[-1]=='True'
        elif 'mag_prec' in line:
            mag_prec = float(row[-1])
        elif 'enum_prec' in line:
            enum_prec = float(row[-1])
        elif 'ltol' in line:
            ltol = float(row[-1])
        elif 'stol' in line:
            stol = float(row[-1])
        elif 'atol' in line:
            atol = float(row[-1])
        elif 'max_neighbors' in line:
            max_neigh = int(row[-1])
        elif 'mag_from' in line:
            mag_from = row[-1]
        elif 'GPU_accel' in line:
            GPU_accel = row[-1]=='True'
        elif 'more_than_2_metal_layers' in line:
            padding = row[-1]=='True'
        elif 'dump_spins' in line:
            dump_spins = row[-1]=='True'
        elif 'ISMEAR' in line:
            ismear = int(row[-1])
        elif 'SIGMA' in line:
            sigma = float(row[-1])
        elif 'NSIM' in line:
            nsim = int(row[-1])
        elif 'KPAR' in line:
            kpar = int(row[-1])
        elif 'NCORE' in line:
            ncore = int(row[-1])
        elif 'ISIF' in line:
            isif = int(row[-1])
        elif 'IOPT' in line:
            iopt = int(row[-1])
        elif 'LUSENCCL' in line:
            nccl = row[-1]=='True'
        elif 'LDAUJ' in line:
            num_spec = len(row)-2
            for i in range(2,num_spec+1,2):
                LDAUJ_provided[row[i]] = float(row[i+1])
        elif 'LDAUU' in line:
            num_spec = len(row)-2
            for i in range(2,num_spec+1,2):
                LDAUU_provided[row[i]] = float(row[i+1])
        elif 'LDAUL' in line:
            num_spec = len(row)-2
            for i in range(2,num_spec+1,2):
                LDAUL_provided[row[i]] = float(row[i+1])
        elif 'LDAUTYPE' in line:
            ldautype = int(row[-1])
        elif 'POTCAR' in line:
            num_spec = len(row)-2
            for i in range(2,num_spec+1,2):
                potcar_provided[row[i]] = row[i+1]
        elif 'same_neighbor_thresh' in line:
            d_thresh = float(row[-1])
        elif 'same_neighbor_thresh_buffer' in line:
            d_buff = float(row[-1])            
        elif 'accuracy' in line:
            acc = row[-1]
        elif 'log_filename' in line:
            log_file = row[-1]
        elif 'kpoints_density_relax' in line:
            kpt_den_relx = float(row[-1])
        elif 'kpoints_density_static' in line:
            kpt_den_stat = float(row[-1])

if not potcar_provided:
    potcar_provided = None


# all functions

def replace_text(fileName,toFind,replaceWith):
    s = open(fileName).read()
    s = s.replace(toFind, replaceWith)
    f = open(fileName, 'w')
    f.write(s)
    f.close()


def log(string):
    string = str(string)
    f = open(root_path+'/'+log_file,'a+')
    time = datetime.datetime.now()
    f.write('>>> '+str(time)+'    '+string+'\n')
    f.close()
    print('>>> '+string)


def sanitize(path):
    try:
        run = Vasprun(path+'/vasprun.xml')
        if run.converged:
            msg = 'found converged vasp run in '+path+', no sanitization required'
            log(msg)
            return True
        else:
            raise ValueError
    except Exception as e:
        msg = str(e)
        log(msg)
        msg = 'found unconverged, nonexistent or damaged vasp run in '+path+', starting sanitization'
        log(msg)
        try:
            try_struct = Structure.from_file(path+'/CONTCAR')
            copyfile(path+'/CONTCAR',path+'/CONTCAR.bk')
            msg = 'backed up CONTCAR'
            log(msg)
        except Exception as e:
            msg = str(e)
            log(msg)
            msg = 'no valid CONTCAR found in '+ path
            log(msg)
        try:
            os.remove(path+'/INCAR')
            os.remove(path+'/INCAR.orig')
            os.remove(path+'/KPOINTS')
            os.remove(path+'/KPOINTS.orig')
            os.remove(path+'/POTCAR')
            os.remove(path+'/POTCAR.orig')
            msg = 'removed old INCAR, KPOINTS and POTCAR'
            log(msg)
        except Exception as e:
            msg = str(e)
            log(msg)
            msg = 'no INCAR or KPOINTS or POTCAR found in '+ path
            log(msg)
        return False


def dist_neighbors(struct):
    struct_l = struct.copy()
    struct_l.make_supercell([20,20,1])
    distances = np.unique(np.sort(np.around(struct_l.distance_matrix[1],2)))[0:15]
    dr_max = 0.01
    for i in range(len(distances)):
        for j in range(len(distances)):
            dr = np.abs(distances[i]-distances[j])
            if distances[j]<distances[i] and dr<d_thresh:
                distances[i]=distances[j]
                if dr>dr_max:
                    dr_max = dr
    distances = np.unique(distances)
    msg = 'neighbor distances are: '+str(distances)+' ang'
    log(msg)
    msg = 'treating '+str(dr_max)+' ang separated atoms as same neighbors'
    log(msg)
    distances[0]=dr_max+d_buff
    return distances

    
def Nfinder(struct_mag,site,d_N,dr):
    N = len(struct_mag)
    coord_site = struct_mag.cart_coords[site]
    Ns = struct_mag.get_neighbors_in_shell(coord_site,d_N,dr)
    Ns_wrapped = Ns[:]
    candidates = Ns[:]
    for i in range(len(Ns)):
        Ns_wrapped[i] = Ns[i][0].to_unit_cell()
        for j in range(N):
            if struct_mag[j].distance(Ns_wrapped[i])<0.01:
                candidates[i] = j
                break
    return candidates


@cuda.jit
def my_kernel(all_coords,coord_N,index):
    """
    Code for kernel.
    """
    pos = cuda.grid(1)
    if pos <= all_coords.size:
        if math.sqrt((all_coords[pos]-coord_N[0])**2 + (all_coords[pos+1]-coord_N[1])**2 + (all_coords[pos+2]-coord_N[2])**2) < 0.01:
            index[0] = pos/3


def Nfinder_GPU(struc_mag,site, d_N, dr):
    coord_site = struc_mag.cart_coords[site]
    Ns = struc_mag.get_neighbors_in_shell(coord_site,d_N,dr)
    Ns_wrapped = Ns[:]
    candidates = Ns[:]
    for i in range(len(Ns)):
        Ns_wrapped[i] = Ns[i][0].to_unit_cell()
        coord_N = np.array([Ns_wrapped[i].x,Ns_wrapped[i].y,Ns_wrapped[i].z],dtype='float32')
        index = np.array([-5])
        threadsperblock = 1000
        blockspergrid = math.ceil(all_coords.shape[0] / threadsperblock)
        my_kernel[blockspergrid,threadsperblock](all_coords,coord_N,index)
        candidates[i]=index[0]
    return candidates


def find_max_len(lst): 
    maxList = max(lst, key = lambda i: len(i)) 
    maxLength = len(maxList)   
    return maxLength 


def make_homogenous(lst):
    msg = 'finding and padding neighbors'
    log(msg)
    max_len = find_max_len(lst)
    for i in range(len(lst)):
        if len(lst[i])<max_len:
            pad = [100000]*(max_len-len(lst[i]))
            lst[i] += pad
        print(str(i)+'p / '+str(len(lst)-1)+' padded')


@jit(nopython=True)
def MC_func(spins_init,T,J2flag,J3flag,J4flag):

    spins_abs = np.abs(np.copy(spins_init))
    if EMA==0:
        spins_x = np.copy(spins_init)
        spins_y = np.zeros(N)
        spins_z = np.zeros(N)
    elif EMA==1:
        spins_x = np.zeros(N)
        spins_y = np.copy(spins_init)
        spins_z = np.zeros(N)
    elif EMA==2:
        spins_x = np.zeros(N)
        spins_y = np.zeros(N)
        spins_z = np.copy(spins_init)

    mag_ups, mag_up_sqs, mag_downs, mag_down_sqs, mag_tots, mag_tot_sqs = np.zeros(trange-threshold), np.zeros(trange-threshold), np.zeros(
        trange-threshold), np.zeros(trange-threshold), np.zeros(trange-threshold), np.zeros(trange-threshold)

    for t in range(trange):
        
        mag_up = 0
        mag_down = 0
        mag_tot = 0
        
        for i in range(N):

            site = np.random.randint(0,N)
            N1s = N1list[site]
            N2s = N2list[site]
            N3s = N3list[site]
            N4s = N4list[site]       
            
            S_current = np.array([spins_x[site],spins_y[site],spins_z[site]])
            u, v = np.random.random(),np.random.random()
            phi = 2*np.pi*u
            theta = np.arccos(2*v-1)
            S_x = spins_abs[site]*np.sin(theta)*np.cos(phi)
            S_y = spins_abs[site]*np.sin(theta)*np.sin(phi)
            S_z = spins_abs[site]*np.cos(theta)
            S_after = np.array([S_x,S_y,S_z])
            E_current = 0
            E_after = 0
            
            for N1 in N1s:
                if N1!=100000 or N1!=-5:
                    S_N1 = np.array([spins_x[N1],spins_y[N1],spins_z[N1]])
                    E_current += -J1*np.dot(S_current,S_N1) + (-K1x*S_current[0]*S_N1[0]) + (-K1y*S_current[1]*S_N1[1]) + (
                        -K1z*S_current[2]*S_N1[2]) 
                    E_after += -J1*np.dot(S_after,S_N1) + (-K1x*S_after[0]*S_N1[0]) + (-K1y*S_after[1]*S_N1[1]) + (
                        -K1z*S_after[2]*S_N1[2])
            if J2flag:
                for N2 in N2s:
                    if N2!=100000 or N2!=-5:
                        S_N2 = np.array([spins_x[N2],spins_y[N2],spins_z[N2]])
                        E_current += -J2*np.dot(S_current,S_N2) + (-K2x*S_current[0]*S_N2[0]) + (-K2y*S_current[1]*S_N2[1]) + (
                        -K2z*S_current[2]*S_N2[2])
                        E_after += -J2*np.dot(S_after,S_N2) + (-K2x*S_after[0]*S_N2[0]) + (-K2y*S_after[1]*S_N2[1]) + (
                        -K2z*S_after[2]*S_N2[2])
            if J3flag: 
                for N3 in N3s:
                    if N3!=100000 or N3!=-5:
                        S_N3 = np.array([spins_x[N3],spins_y[N3],spins_z[N3]])
                        E_current += -J3*np.dot(S_current,S_N3) + (-K3x*S_current[0]*S_N3[0]) + (-K3y*S_current[1]*S_N3[1]) + (
                        -K3z*S_current[2]*S_N3[2])
                        E_after += -J3*np.dot(S_after,S_N3) + (-K3x*S_after[0]*S_N3[0]) + (-K3y*S_after[1]*S_N3[1]) + (
                        -K3z*S_after[2]*S_N3[2])
            if J4flag: 
                for N4 in N4s:
                    if N4!= 100000 or N4!=-5:
                        S_N4 = np.array([spins_x[N4],spins_y[N4],spins_z[N4]])
                        E_current += -J4*np.dot(S_current,S_N4) + (-K4x*S_current[0]*S_N4[0]) + (-K4y*S_current[1]*S_N4[1]) + (
                        -K4z*S_current[2]*S_N4[2])
                        E_after += -J4*np.dot(S_after,S_N4) + (-K4x*S_after[0]*S_N4[0]) + (-K4y*S_after[1]*S_N4[1]) + (
                        -K4z*S_after[2]*S_N4[2])            

            E_current += -Ax*np.square(S_current[0]) + (-Ay*np.square(S_current[1])) + (-Az*np.square(S_current[2])) 
            E_after += -Ax*np.square(S_after[0]) + (-Ay*np.square(S_after[1])) + (-Az*np.square(S_after[2]))
            
            del_E = E_after-E_current
                    
            if del_E < 0:
                spins_x[site],spins_y[site],spins_z[site] = S_x,S_y,S_z 
            else:
                samp = np.random.random()
                if samp <= np.exp(-del_E/(kB*T)):
                    spins_x[site],spins_y[site],spins_z[site] = S_x,S_y,S_z


        if t>=threshold:

            mag_vec_up = np.zeros(3)
            mag_vec_down = np.zeros(3)
            num_up_sites = 0

            for i in range(N):
                if spins_init[i]>0:
                    mag_vec_up += 2*np.array([spins_x[i], spins_y[i], spins_z[i]])
                else:
                    mag_vec_down += 2*np.array([spins_x[i], spins_y[i], spins_z[i]])

            mag_up = np.linalg.norm(mag_vec_up)
            mag_ups[t-threshold] = np.abs(mag_up)
            mag_up_sqs[t-threshold] = np.square(mag_up)
            mag_down = np.linalg.norm(mag_vec_down)
            mag_downs[t-threshold] = np.abs(mag_down)
            mag_down_sqs[t-threshold] = np.square(mag_down)

            mag_vec_tot = 2*np.array([np.sum(spins_x),np.sum(spins_y),np.sum(spins_z)])
            mag_tot = np.linalg.norm(mag_vec_tot)
            mag_tots[t-threshold] = np.abs(mag_tot)
            mag_tot_sqs[t-threshold] = np.square(mag_tot)

            
    (M_up,M_up_sq,M_down,M_down_sq,M_tot,M_tot_sq) = (np.mean(mag_ups),np.mean(mag_up_sqs),np.mean(mag_downs),np.mean(mag_down_sqs),
    np.mean(mag_tots),np.mean(mag_tot_sqs))

    if np.abs(np.sum(spins_init))<1e-3:
        num_up_sites = N/2
    else:
        num_up_sites = N
    X_up = (M_up_sq-np.square(M_up))/(num_up_sites*kB*T)
    X_down = (M_down_sq-np.square(M_down))/(num_up_sites*kB*T)
    X_tot = (M_tot_sq-np.square(M_tot))/(N*kB*T)
    M_up = M_up/num_up_sites
    M_down = M_down/num_up_sites
    M_tot = M_tot/N

    return M_up,X_up,M_down,X_down,M_tot,X_tot,spins_x,spins_y,spins_z



# main code

msg = '*'*150
log(msg)
msg = '*** this code have been developed by Arnab Kabiraj at Nano-Scale Device Research Laboratory (NSDRL), IISc, Bengaluru, India ***\n'
msg += '*** for any queries please contact the authors at kabiraj@iisc.ac.in or santanu@iisc.ac.in ***'
log(msg)
msg = '*'*150
log(msg)
if acc == 'high':
    msg = '* command for high accuracy detected, the calculations could take significantly more time than ususal\n'
    log(msg)
    

magnetic_list = [Element('Co'), Element('Cr'), Element('Fe'), Element('Mn'), Element('Mo'),
Element('Ni'), Element('V'), Element('W'), Element('Ce'), Element('Os'), Element('Sc'),
Element('Ti'), Element('Ag'), Element('Zr'), Element('Pd'), Element('Rh'), Element('Hf'),
Element('Nb'), Element('Y'), Element('Re'), Element('Cu'), Element('Ru'), Element('Pt'), Element('La')]

cell = read(struct_file)
c = cell.cell.cellpar()[2]

for i in range(len(cell)):
    if cell[i].z > c*0.75:
        cell[i].z = cell[i].z - c

cell.center(2.5,2)
ase_adopt = AseAtomsAdaptor()
struct = ase_adopt.get_structure(sort(cell))
if strain:
    struct.apply_strain(strain)
    msg = 'the structure is being starined with '+str(strain)+', will set ISIF = 2'
    log(msg)

mag_enum = MagneticStructureEnumerator(struct,transformation_kwargs={'symm_prec':mag_prec,'enum_precision_parameter':enum_prec})
mag_enum_structs = []
for mag_struct in mag_enum.ordered_structures:
    n = len(mag_struct)
    spins = [0]*n
    uneven_spins = False
    for j in range(n):
        try:
            spins[j] = mag_struct.species[j].spin
        except Exception:
            element = mag_struct[j].specie.element
            if element in magnetic_list:
                uneven_spins = True
                break
            else:
                spins[j] = 0.0
    if uneven_spins:
        msg = '** a config has uneven spins, continuing without it'
        log(msg)
        continue
    mag_struct.add_site_property('magmom',spins)
    mag_struct.remove_spin()
    mag_struct.sort()
    mag_enum_structs.append(mag_struct)

s1 = mag_enum_structs[0].copy()
s1.make_supercell(rep_DFT)
matcher = StructureMatcher(primitive_cell=False,attempt_supercell=True)

mag_structs = []
mag_structs_super = []
spins_configs = []
spins_configs_super = []
count = 0

for i in range(len(mag_enum_structs)):
    s_mag = mag_enum_structs[i].copy()
    if matcher.fit(s1,s_mag):
        mag_struct = matcher.get_s2_like_s1(s1,s_mag)
        spins = mag_struct.site_properties['magmom']
        mag_tot = np.sum(spins)
        if i>0 and mag_tot!=0:
            msg = '** a config has uneven spins, continuing without it'
            log(msg)
            continue
        mag_cell = ase_adopt.get_atoms(mag_struct,magmoms=spins)
        mag_cell.center(vacuum/2,2)
        mag_struct = ase_adopt.get_structure(mag_cell)
        mag_struct.add_spin_by_site(spins)
        mag_struct.to(filename='POSCAR.config_'+str(count)+'.supercell.vasp')
        mag_structs_super.append(mag_struct)
        spins_configs_super.append(spins)
        mag_struct.remove_spin()
        mag_struct.add_site_property('magmom',spins)
        mag_struct_prim = mag_struct.get_primitive_structure(use_site_props=True)
        spins_prim = mag_struct_prim.site_properties['magmom']
        mag_struct_prim.remove_site_property('magmom')
        mag_struct_prim.add_spin_by_site(spins_prim)
        mag_struct_prim.to(filename='POSCAR.config_'+str(count)+'.vasp')
        mag_structs.append(mag_struct_prim)
        spins_configs.append(spins_prim)
        count += 1

if skip:
    skip2 = [int(ind) for ind in skip]
    for i in range(len(skip)):
        ind = skip2[i]
        mag_structs.pop(ind)
        spins_configs.pop(ind)
        mag_structs_super.pop(ind)
        spins_configs_super.pop(ind)
        skip2 = [ind-1 for ind in skip2]
        msg = 'skipping config_'+str(ind)+' on user request, the remaining configs would be renumbered'
        log(msg)
        
num_struct = len(mag_structs)
if num_struct == 1:
    msg = '*** only one config could be generated, can not fit Hamiltonian, exiting,'
    msg +=' play with enum_prec and mag_prec and DFT_supercell_size to generate more configs or try out a new material'
    log(msg)
    sys.exit()
elif num_struct == 2:
    msg = '** only two configs could be generated, only first nearest neighbor interaction can be included,'
    msg +=' play with enum_prec and mag_prec and DFT_supercell_size to generate more configs'
    log(msg)

msg = 'total '+str(num_struct)+' configs generated'
log(msg)

num_atoms = []
for struct in mag_structs:
    num_atoms.append(len(struct))
lcm_atoms = np.lcm.reduce(num_atoms)

LDAUJ_dict = {'Co': 0, 'Cr': 0, 'Fe': 0, 'Mn': 0, 'Mo': 0, 'Ni': 0, 'V': 0, 'W': 0,
'Nb': 0, 'Sc': 0, 'Ru': 0, 'Rh': 0, 'Pd': 0, 'Cu': 0, 'Y': 0, 'Os': 0, 'Ti': 0, 'Zr': 0, 'Re': 0, 'Hf': 0, 'Pt':0, 'La':0}
if LDAUJ_provided:
    LDAUJ_dict.update(LDAUJ_provided)

LDAUU_dict = {'Co': 3.32, 'Cr': 3.7, 'Fe': 5.3, 'Mn': 3.9, 'Mo': 4.38, 'Ni': 6.2, 'V': 3.25, 'W': 6.2,
'Nb': 1.45, 'Sc': 4.18, 'Ru': 4.29, 'Rh': 4.17, 'Pd': 2.96, 'Cu': 7.71, 'Y': 3.23, 'Os': 2.47, 'Ti': 5.89, 'Zr': 5.55,
'Re': 1.28, 'Hf': 4.77, 'Pt': 2.95, 'La':5.3}
if LDAUU_provided:
    LDAUU_dict.update(LDAUU_provided)

LDAUL_dict = {'Co': 2, 'Cr': 2, 'Fe': 2, 'Mn': 2, 'Mo': 2, 'Ni': 2, 'V': 2, 'W': 2,
'Nb': 2, 'Sc': 2, 'Ru': 2, 'Rh': 2, 'Pd': 2, 'Cu': 2, 'Y': 2, 'Os': 2, 'Ti': 2, 'Zr': 2, 'Re': 2, 'Hf': 2, 'Pt':2, 'La':2}
if LDAUL_provided:
    LDAUL_dict.update(LDAUL_provided)

relx_dict = {'ALGO': 'Fast', 'ISMEAR': 0, 'SIGMA': 0.01, 'EDIFF': 1E-4, 'EDIFFG': -0.01, 
'KPAR': kpar,  'NCORE': ncore, 'NSIM': nsim, 'LCHARG': False, 'ICHARG': 2, 'LREAL': False,
'LDAU': True, 'LDAUJ': LDAUJ_dict, 'LDAUL': LDAUL_dict, 'LDAUU': LDAUU_dict, 'LWAVE': False,
'LDAUPRINT': 1, 'LDAUTYPE': ldautype, 'LASPH': True, 'LMAXMIX': 4,
'ISIF': isif, 'IBRION': 3, 'POTIM': 0, 'IOPT': iopt, 'LTWODIM': True, 'LUSENCCL': nccl}

relx_handlers = [VaspErrorHandler(), UnconvergedErrorHandler(),
    FrozenJobErrorHandler(timeout=900), MeshSymmetryErrorHandler(), PositiveEnergyErrorHandler(),
    StdErrHandler(), NonConvergingErrorHandler(nionic_steps=5), PotimErrorHandler(dE_threshold=0.5)]

stat_dict = {'ISMEAR': ismear, 'EDIFF': 1E-6, 'KPAR': kpar, 'NCORE': ncore, 'NSIM': nsim, 'LORBMOM': True, 'LAECHG': True, 'LREAL': False,
'LDAU': True, 'LDAUJ': LDAUJ_dict, 'LDAUL': LDAUL_dict, 'LDAUU': LDAUU_dict, 'NELMIN': 6, 'NELM': 250, 'LVHAR': False, 'SIGMA': sigma,
'LDAUPRINT': 1, 'LDAUTYPE': ldautype, 'LASPH': True, 'LMAXMIX': 4, 'LCHARG': True, 'LWAVE': True, 'ISYM': -1, 'LVTOT': False, 'LUSENCCL': nccl}

stat_handlers = [VaspErrorHandler(), UnconvergedErrorHandler(),
    FrozenJobErrorHandler(timeout=3600), MeshSymmetryErrorHandler(), PositiveEnergyErrorHandler(), StdErrHandler()]

validator = [VasprunXMLValidator()]

ortho_ab= (mag_enum.ordered_structures[0].lattice.gamma>88 and mag_enum.ordered_structures[0].lattice.gamma<92) and (mag_enum.ordered_structures[0].lattice.a/mag_enum.ordered_structures[0].lattice.b<0.9 or
    mag_enum.ordered_structures[0].lattice.a/mag_enum.ordered_structures[0].lattice.b>1.1)

if xc=='PBE':
    pot = 'PBE_54'
elif xc=='LDA':
    pot = 'LDA_54'
elif xc=='SCAN':
    pot = 'PBE_54'
    relx_dict['METAGGA'] = 'SCAN'
    relx_dict['LMIXTAU'] = True
    relx_dict['LDAU'] = False
    relx_dict['ALGO'] = 'All'
    stat_dict['METAGGA'] = 'SCAN'
    stat_dict['LMIXTAU'] = True
    stat_dict['LDAU'] = False
    stat_dict['ALGO'] = 'All'
elif xc=='R2SCAN':
    pot = 'PBE_54'
    relx_dict['METAGGA'] = 'R2SCAN'
    relx_dict['LMIXTAU'] = True
    relx_dict['LDAU'] = False
    relx_dict['ALGO'] = 'All'
    stat_dict['METAGGA'] = 'R2SCAN'
    stat_dict['LMIXTAU'] = True
    stat_dict['LDAU'] = False
    stat_dict['ALGO'] = 'All'
elif xc=='SCAN+RVV10':
    pot = 'PBE_54'
    relx_dict['METAGGA'] = 'SCAN'
    relx_dict['LMIXTAU'] = True
    relx_dict['LDAU'] = False
    relx_dict['ALGO'] = 'All'
    relx_dict['LUSE_VDW'] = True
    relx_dict['BPARAM'] = 6.3
    relx_dict['CPARAM'] = 0.0093
    stat_dict['METAGGA'] = 'SCAN'
    stat_dict['LMIXTAU'] = True
    stat_dict['LDAU'] = False
    stat_dict['ALGO'] = 'All'
    stat_dict['LUSE_VDW'] = True
    stat_dict['BPARAM'] = 6.3
    stat_dict['CPARAM'] = 0.0093
elif xc=='R2SCAN+RVV10':
    pot = 'PBE_54'
    relx_dict['METAGGA'] = 'R2SCAN'
    relx_dict['LMIXTAU'] = True
    relx_dict['LDAU'] = False
    relx_dict['ALGO'] = 'All'
    relx_dict['LUSE_VDW'] = True
    relx_dict['BPARAM'] = 6.3
    relx_dict['CPARAM'] = 0.0093
    stat_dict['METAGGA'] = 'R2SCAN'
    stat_dict['LMIXTAU'] = True
    stat_dict['LDAU'] = False
    stat_dict['ALGO'] = 'All'
    stat_dict['LUSE_VDW'] = True
    stat_dict['BPARAM'] = 6.3
    stat_dict['CPARAM'] = 0.0093
elif xc=='PBEsol':
    pot = 'PBE_54'
    relx_dict['GGA'] = 'PS'
    stat_dict['GGA'] = 'PS'

if acc=='high':
    relx_dict['EDIFF'] = 1E-5
    stat_dict['EDIFF'] = 1E-8
    if kpt_den_relx==None:
        kpt_den_relx = 300
    if kpt_den_stat==None:
        kpt_den_stat = 1000

if not relx:
    relx_dict['EDIFFG'] = -10.0
    msg = 'command detected for no relaxation, structures wont be relaxed, only a fake and fast relaxation will be performed'
    log(msg)

if strain:
    relx_dict['ISIF'] = 2

if kpt_den_relx==None and acc!='high':
    kpt_den_relx = 150
if kpt_den_stat==None and acc!='high':
    kpt_den_stat = 300

if ortho_ab:
    saxes = [(1,0,0),(0,1,0),(0,0,1)]
    msg = 'found orthogonal a and b vectors, will perform noncollinear calculations for '+str(saxes)
    log(msg)
else:
    saxes = [(1,0,0),(0,0,1)]
    msg = 'found non-orthogonal or orthogonal but equal a and b vectors, will perform noncollinear calculations for '+str(saxes)+ ' only'
    log(msg)

start_time_dft = time()
energies_relx = []

# relax the enumerated structures, no supercell 

for i in range(num_struct):

    spins = spins_configs[i]
    struct_current = mag_structs[i].copy()
    factor = float(lcm_atoms)/len(struct_current)
    
    if factor!=int(factor):
        msg = '*** factor is float, '+str(factor)+', exiting'
        log(msg)
        sys.exit()

    relx_path = root_path+'/relxations'+'/config_'+str(i)
    clean = sanitize(relx_path)

    if not clean:

        relx = MPRelaxSet(struct_current,user_incar_settings=relx_dict,user_kpoints_settings={'reciprocal_density':kpt_den_relx},
            force_gamma=True,user_potcar_functional=pot,sort_structure=False,user_potcar_settings=potcar_provided)
        relx.write_input(relx_path)
        if xc=='SCAN+RVV10' or xc=='R2SCAN+RVV10':
            copyfile(root_path+'/vdw_kernel.bindat',relx_path+'/vdw_kernel.bindat')
        try:
            try_struct = Structure.from_file(relx_path+'/CONTCAR.bk')
            try_struct.to(filename=relx_path+'/POSCAR')
            msg = 'copied backed up CONTCAR to POSCAR'
        except Exception as e:
            print(e)
            msg = 'no backed up CONTCAR found'
        log(msg)
        kpts = Kpoints.from_file(relx_path+'/KPOINTS')
        kpts.kpts[0][2] = 1
        kpts.write_file(relx_path+'/KPOINTS')

        if randomise_cmd:
            cmd_rand = cmd[:]
            cmd_rand[-1] = cmd_rand[-1]+'_'+str(np.random.randint(0,9999))
            job = [VaspJob(cmd_rand)]
        else:
            job = [VaspJob(cmd)]
        cust = Custodian(relx_handlers,job,validator,max_errors=20,polling_time_step=5,monitor_freq=10,
            gzipped_output=False,checkpoint=False)
        msg = 'running relaxtion for config '+str(i)
        log(msg)
        done = 0

        os.chdir(relx_path)
        for j in range(3):
            try:
                cust.run()
                done = 1
                sleep(10)
                break
            except:
                sleep(10)
                continue
        os.chdir(root_path)
        
        if done == 1:
            msg = 'relaxation job finished successfully for config '+str(i)
            log(msg)
        else:
            msg = 'relaxation failed for config '+str(i)+' after several attempts, exiting, you might want to manually handle this one,'
            msg += 'and then restart this code'
            log(msg)
            sys.exit()

    run = Vasprun(relx_path+'/vasprun.xml',parse_dos=False,parse_eigen=False)
    energy = float(run.final_energy)
    energy = energy*factor
    energies_relx.append(energy)


msg = 'all relaxations have finished gracefully'
log(msg)
msg = 'the configuration wise relaxation energies are: '+str(energies_relx)
log(msg)
most_stable = np.argmin(energies_relx)
msg = '### The most stable config = config_'+str(most_stable)
log(msg)

s_mag = Structure.from_file(root_path+'/relxations'+'/config_'+str(most_stable)+'/CONTCAR')
s1 = Structure.from_file(root_path+'/POSCAR.config_0.supercell.vasp')
matcher = StructureMatcher(primitive_cell=False,attempt_supercell=True,ltol=ltol,stol=stol,angle_tol=atol)
struct_ground_super = matcher.get_s2_like_s1(s1,s_mag)
if struct_ground_super==None:
    msg = 'can not make supercell with the most stable relaxed structure, '
    msg += 'carefully check the relaxation results and play with ltol, stol and angle_tol'
    msg += ' or relax manually and run this code without relaxations, exiting'
    log(msg)
    sys.exit()

mag_structs = []
for i in range(num_struct):
    mag_struct = struct_ground_super.copy()
    mag_struct.add_spin_by_site(spins_configs_super[i])
    mag_structs.append(mag_struct)

if most_stable<=max_neigh:
    mag_structs = mag_structs[:max_neigh+1]

num_struct = len(mag_structs)

for i in range(num_struct):

    spins = spins_configs[i]
    stat_struct = mag_structs[i].copy()

    stat_path = root_path+'/static_runs'+'/config_'+str(i)
    clean = sanitize(stat_path)

    if not clean:

        stat = MPStaticSet(stat_struct,user_incar_settings=stat_dict,reciprocal_density=kpt_den_stat,
            force_gamma=True,user_potcar_functional=pot,sort_structure=False,user_potcar_settings=potcar_provided)
        stat.write_input(stat_path)
        if xc=='SCAN+RVV10' or xc=='R2SCAN+RVV10':
            copyfile(root_path+'/vdw_kernel.bindat',stat_path+'/vdw_kernel.bindat')
        kpts = Kpoints.from_file(stat_path+'/KPOINTS')
        kpts.kpts[0][2] = 1
        kpts.write_file(stat_path+'/KPOINTS')

        if randomise_cmd:
            cmd_rand = cmd[:]
            cmd_rand[-1] = cmd_rand[-1]+'_'+str(np.random.randint(0,9999))
            job = [VaspJob(cmd_rand)]
        else:
            job = [VaspJob(cmd)]
        cust = Custodian(stat_handlers,job,validator,max_errors=7,polling_time_step=5,monitor_freq=10,
            gzipped_output=False,checkpoint=False)
        msg = 'running static run for config '+str(i)
        log(msg)
        done = 0

        os.chdir(stat_path)
        for j in range(3):
            try:
                cust.run()
                done = 1
                sleep(10)
                break
            except:
                sleep(10)
                continue
        os.chdir(root_path)
    
        if done == 1:
            msg = 'static run finished successfully for config '+str(i)
            log(msg)
        else:
            msg = 'static run failed for config '+str(i)
            msg += ' after several attempts, exiting, you might want to manually handle this one, and then restart this code'
            log(msg)
            sys.exit()

    for axis in saxes:
        
        mae_path = root_path+'/MAE/config_'+str(i)+'/'+str(axis).replace(' ','')
        clean = sanitize(mae_path)
        
        if not clean:

            soc = MPSOCSet.from_prev_calc(stat_path,saxis=axis,nbands_factor=2,reciprocal_density=kpt_den_stat,
                force_gamma=True,user_potcar_functional=pot,sort_structure=False,user_potcar_settings=potcar_provided)                     
            soc.write_input(mae_path)
            if xc=='SCAN+RVV10' or xc=='R2SCAN+RVV10':
                copyfile(root_path+'/vdw_kernel.bindat',mae_path+'/vdw_kernel.bindat')
            replace_text(mae_path+'/INCAR','LCHARG = True','LCHARG = False')
            replace_text(mae_path+'/INCAR','LWAVE = True','LWAVE = False')
            replace_text(mae_path+'/INCAR','LAECHG = True','LAECHG = False')
            if 'SCAN' in xc:
                replace_text(mae_path+'/INCAR','ICHARG = 11','ICHARG = 1')
            else:
                replace_text(mae_path+'/INCAR','EDIFF = 1e-06','EDIFF = 1e-08')
            with open(mae_path+'/INCAR','a') as inc:
                inc.write('\nKPAR = '+str(kpar)+'\nNCORE = '+str(ncore)+'\nLUSENCCL = '+str(nccl))
            kpts = Kpoints.from_file(mae_path+'/KPOINTS')
            kpts.kpts[0][2] = 1
            kpts.write_file(mae_path+'/KPOINTS')

            try:
                copyfile(stat_path+'/WAVECAR',mae_path+'/WAVECAR')
            except:
                msg = '*** no collinear WAVECAR found, exiting, generate this and restart'
                log(msg)
                sys.exit()
            if randomise_cmd:
                cmd_rand = cmd_ncl[:]
                cmd_rand[-1] = cmd_rand[-1]+'_'+str(np.random.randint(0,9999))
                job = [VaspJob(cmd_rand)]
            else:
                job = [VaspJob(cmd_ncl)]
            cust = Custodian(stat_handlers,job,validator,max_errors=7,polling_time_step=5,monitor_freq=10,
                gzipped_output=False,checkpoint=False)
            msg = 'running non-collinear run for config '+str(i)+' and direction '+str(axis)
            log(msg)
            done = 0

            os.chdir(mae_path)
            for j in range(3):
                try:
                    cust.run()
                    done = 1
                    sleep(10)
                    break
                except:
                    sleep(10)
                    continue
            os.chdir(root_path)
        
            if done == 1:
                msg = 'non-collinear run finished successfully for config '+str(i)+' and direction '+str(axis)
                log(msg)
            else:
                msg = 'non-collinear run failed for config '+str(i)+' and direction '+str(axis)
                msg += ' after several attempts, exiting, you might want to manually handle this one, and then restart this code'
                log(msg)
                sys.exit()
            os.remove(mae_path+'/CHGCAR')
            os.remove(mae_path+'/WAVECAR')
                

end_time_dft = time()
time_dft = np.around(end_time_dft - start_time_dft, 2)
msg = 'all static and non-collinear runs for anisotopies have finished gracefully'
log(msg)

msg = 'DFT energy calculations/check of all possible configurations took total '+str(time_dft)+' s'
log(msg)
msg = 'attempting to collect data and fit the Hamiltonian now'
log(msg)


num_neigh = min([max_neigh, num_struct-1])
msg = 'total '+str(num_struct)+' valid FM/AFM configs have been detected, including '
msg += str(num_neigh)+' nearest-neighbors in the fitting'
log(msg)


semifinal_list = []

for i in range(num_struct):
    
    msg = 'checking vasp run status of config_'+str(i)+' static and non-collinear runs'
    log(msg)
    config_info = []
    stat_path = root_path+'/static_runs'+'/config_'+str(i)
    #struct = mag_structs[i].copy()
    struct = Structure.from_file(root_path+'/static_runs/config_'+str(i)+'/POSCAR')
    inc = Incar.from_file(root_path+'/static_runs/config_'+str(i)+'/INCAR')
    struct.add_spin_by_site(inc.as_dict()['MAGMOM']) 
    run = Vasprun(stat_path+'/vasprun.xml',parse_dos=False,parse_eigen=False)
    if not run.converged_electronic:
        msg = '*** static run have not converged for config_'+str(i)+', exiting'
        log(msg)
        sys.exit()
    else:
        msg = 'found converged static run'
        log(msg)
        
    energy = float(run.final_energy)  
        
    config_info.append(i)
    config_info.append(struct)
    config_info.append(energy)

    for axis in saxes:

        mae_path = root_path+'/MAE/config_'+str(i)+'/'+str(axis).replace(' ','')
        run = Vasprun(mae_path+'/vasprun.xml',parse_dos=False,parse_eigen=False)
        struct = Structure.from_file(mae_path+'/POSCAR')
        if not run.converged_electronic:
            msg = '*** non-collinear run have not converged for config_'+str(i)+' and axis '+str(axis)+', exiting'
            log(msg)
            sys.exit()
        else:
            msg = 'found converged non-collinear run'
            log(msg)
        energy = float(run.final_energy)
        config_info.append(energy)
        if not ortho_ab and axis==(1,0,0):
            config_info.append(energy)
    
    semifinal_list.append(config_info)

semifinal_list = sorted(semifinal_list, key = lambda x : x[2])
most_stable = semifinal_list[0][0]

msg = '### The most stable config = config_'+str(most_stable)
log(msg)

energies_ncl = semifinal_list[0][3:]
EMA = np.argmin(energies_ncl)
saxes = [(1,0,0),(0,1,0),(0,0,1)]

msg = '### The easy magnetization axis (EMA) = '+str(saxes[EMA])
log(msg)

analyzer = CollinearMagneticStructureAnalyzer(semifinal_list[0][1],overwrite_magmom_mode='replace_all_if_undefined',
    make_primitive=False)
num_mag_atoms = analyzer.number_of_magnetic_sites
E_100_001 = (energies_ncl[0] - energies_ncl[2])/(num_mag_atoms)
E_010_001 = (energies_ncl[1] - energies_ncl[2])/(num_mag_atoms)
msg = '### magnetocrystalline anisotropic energies (MAE) are:'
log(msg)
msg = 'E[100]-E[001] = '+str(E_100_001*1e6)+' ueV/magnetic_atom'
log(msg)
msg = 'E[010]-E[001] = '+str(E_010_001*1e6)+' ueV/magnetic_atom'
log(msg)


for i in range(len(semifinal_list)):

    config = semifinal_list[i][0]
    stat_path = root_path+'/static_runs'+'/config_'+str(config)
    if mag_from=='Bader' and config==most_stable:
        if not os.path.exists(stat_path+'/bader.dat'):
            msg = 'starting bader analysis for config_'+str(config)
            log(msg)
            ba = bader_analysis_from_path(stat_path)
            msg = 'finished bader analysis successfully'
            log(msg)
            f = open(stat_path+'/bader.dat','wb')
            dump(ba,f)
            f.close()                         
        else:
            f = open(stat_path+'/bader.dat','rb')
            ba = load(f)
            f.close()
            msg = 'reading magmoms from bader file'
            log(msg)
        magmom_stable = max(ba['magmom'])
        S_stable = magmom_stable/2.0

    elif mag_from=='OSZICAR' and config==0:
        osz = Oszicar(stat_path+'/OSZICAR')
        config_magmom = float(osz.ionic_steps[-1]['mag'])
        analyzer = CollinearMagneticStructureAnalyzer(semifinal_list[i][1],overwrite_magmom_mode='replace_all_if_undefined',
            make_primitive=False)
        num_mag_atoms = analyzer.number_of_magnetic_sites
        magmom_stable = config_magmom/num_mag_atoms
        S_stable = magmom_stable/2.0


E0 = Symbol('E0')
J1 = Symbol('J1')
J2 = Symbol('J2')
J3 = Symbol('J3')
J4 = Symbol('J4')
K1x = Symbol('K1x')
K1y = Symbol('K1y')
K1z = Symbol('K1z')
K2x = Symbol('K2x')
K2y = Symbol('K2y')
K2z = Symbol('K2z')
K3x = Symbol('K3x')
K3y = Symbol('K3y')
K3z = Symbol('K3z')
K4x = Symbol('K4x')
K4y = Symbol('K4y')
K4z = Symbol('K4z')
Ax = Symbol('Ax')
Ay = Symbol('Ay')
Az = Symbol('Az')


kB = np.double(8.6173303e-5)

fitted = False

while num_neigh>0:
    
    final_list = semifinal_list[:(num_neigh+1)]

    num_config = len(final_list)
    eqn_set_iso = [0]*num_config
    eqn_set_x = [0]*num_config
    eqn_set_y = [0]*num_config
    eqn_set_z = [0]*num_config
    CN1s = []
    CN2s = []
    CN3s = []
    CN4s = []

    for i in range(num_config):
        
        config = final_list[i][0]
        struct = final_list[i][1]
        energy_iso = final_list[i][2]
        energies_ncl = final_list[i][3:]
        stat_path = root_path+'/static_runs'+'/config_'+str(config)
               
        out = Outcar(stat_path+'/OUTCAR')
            
        sites_mag = []
        magmoms_mag = []
        magmoms_out = []
        for j in range(len(struct)):
            element = struct[j].specie.element
            if element in magnetic_list:
                sign_magmom = np.sign(struct[j].specie.spin)
                magmom = sign_magmom*magmom_stable
                magmoms_mag.append(magmom)
                sites_mag.append(struct[j])
                magmoms_out.append(out.magnetization[j]['tot'])
        struct_mag = Structure.from_sites(sites_mag)
        struct_mag_out = Structure.from_sites(sites_mag)
        struct_mag.remove_spin()
        struct_mag.add_site_property('magmom',magmoms_mag)
        struct_mag_out.add_site_property('magmom',magmoms_out)
        N = len(struct_mag)
        msg = 'config_'+str(config)+' (only magnetic atoms) = '
        log(msg)
        log(struct_mag)
        msg = 'same config with magmoms from OUTCAR is printed below, make sure this does not deviate too much from above'
        log(msg)
        log(struct_mag_out)
        
        ds = dist_neighbors(struct_mag)
        dr = ds[0]

        eqn_iso = E0 - energy_iso
        eqn_x, eqn_y, eqn_z = energy_iso - energies_ncl[0], energy_iso - energies_ncl[1], energy_iso - energies_ncl[2]

        N1s = []
        N2s = []
        N3s = []
        N4s = []
        
        for j in range(N):
            site = j
            S_site = struct_mag.site_properties['magmom'][j]/2.0
            if num_config==2:
                N1s = Nfinder(struct_mag,site,ds[1],dr)
            elif num_config==3:
                N1s = Nfinder(struct_mag,site,ds[1],dr)
                N2s = Nfinder(struct_mag,site,ds[2],dr)
            elif num_config==4:
                N1s = Nfinder(struct_mag,site,ds[1],dr)
                N2s = Nfinder(struct_mag,site,ds[2],dr)
                N3s = Nfinder(struct_mag,site,ds[3],dr)
            elif num_config==5:
                N1s = Nfinder(struct_mag,site,ds[1],dr)
                N2s = Nfinder(struct_mag,site,ds[2],dr)
                N3s = Nfinder(struct_mag,site,ds[3],dr)
                N4s = Nfinder(struct_mag,site,ds[4],dr)
            
            for N1 in N1s:
                S_N1 = struct_mag.site_properties['magmom'][N1]/2.0
                eqn_iso += -0.5*J1*S_site*S_N1
                eqn_x += -0.5*K1x*S_site*S_N1
                eqn_y += -0.5*K1y*S_site*S_N1
                eqn_z += -0.5*K1z*S_site*S_N1
            if N2s:
                for N2 in N2s:
                    S_N2 = struct_mag.site_properties['magmom'][N2]/2.0
                    eqn_iso += -0.5*J2*S_site*S_N2
                    eqn_x += -0.5*K2x*S_site*S_N2
                    eqn_y += -0.5*K2y*S_site*S_N2
                    eqn_z += -0.5*K2z*S_site*S_N2
            if N3s:
                for N3 in N3s:
                    S_N3 = struct_mag.site_properties['magmom'][N3]/2.0
                    eqn_iso += -0.5*J3*S_site*S_N3
                    eqn_x += -0.5*K3x*S_site*S_N3
                    eqn_y += -0.5*K3y*S_site*S_N3
                    eqn_z += -0.5*K3z*S_site*S_N3
            if N4s:
                for N4 in N4s:
                    S_N4 = struct_mag.site_properties['magmom'][N4]/2.0
                    eqn_iso += -0.5*J4*S_site*S_N4
                    eqn_x += -0.5*K4x*S_site*S_N4
                    eqn_y += -0.5*K4y*S_site*S_N4
                    eqn_z += -0.5*K4z*S_site*S_N4
            eqn_x += -Ax*np.square(S_site)
            eqn_y += -Ay*np.square(S_site)
            eqn_z += -Az*np.square(S_site)
            CN1s.append(len(N1s))
            CN2s.append(len(N2s))
            CN3s.append(len(N3s))
            CN4s.append(len(N4s))

        eqn_set_iso[i] = eqn_iso
        eqn_set_x[i] = eqn_x
        eqn_set_y[i] = eqn_y
        eqn_set_z[i] = eqn_z

        if config==most_stable:
            struct_mag_stable = struct_mag
            ds_stable = ds
            struct_stable = struct_mag

    msg = '### mu = '+str(magmom_stable)+' bohr magnetron/magnetic atom'
    log(msg)
            
    msg = 'eqns are:'
    log(msg)
    
    for eqn in eqn_set_iso:
        msg = str(eqn)+' = 0'
        log(msg)
    for eqn in eqn_set_x:
        msg = str(eqn)+' = 0'
        log(msg)
    if ortho_ab:
        for eqn in eqn_set_y:
            msg = str(eqn)+' = 0'
            log(msg)        
    for eqn in eqn_set_z:
        msg = str(eqn)+' = 0'
        log(msg)        

    if num_config==2:
        soln_iso = linsolve(eqn_set_iso, E0, J1)
        soln_x = linsolve(eqn_set_x, K1x, Ax)
        if ortho_ab:
            soln_y = linsolve(eqn_set_y, K1y, Ay)
        soln_z = linsolve(eqn_set_z, K1z, Az)
    elif num_config==3:
        soln_iso = linsolve(eqn_set_iso, E0, J1, J2)
        soln_x = linsolve(eqn_set_x, K1x, K2x, Ax)
        if ortho_ab:
            soln_y= linsolve(eqn_set_y, K1y, K2y, Ay)
        soln_z = linsolve(eqn_set_z, K1z, K2z, Az)
    elif num_config==4:
        soln_iso = linsolve(eqn_set_iso, E0, J1, J2, J3)
        soln_x = linsolve(eqn_set_x, K1x, K2x, K3x, Ax)
        if ortho_ab:
            soln_y = linsolve(eqn_set_y, K1y, K2y, K3y, Ay)
        soln_z = linsolve(eqn_set_z, K1z, K2z, K3z, Az)
    elif num_config==5:
        soln_iso = linsolve(eqn_set_iso, E0, J1, J2, J3, J4)
        soln_x = linsolve(eqn_set_x, K1x, K2x, K3x, K4x, Ax)
        if ortho_ab:
            soln_y = linsolve(eqn_set_y, K1y, K2y, K3y, K4y, Ay)
        soln_z = linsolve(eqn_set_z, K1z, K2z, K3z, K4z, Az)
    soln_iso = list(soln_iso)
    soln_x = list(soln_x)
    if ortho_ab:
        soln_y = list(soln_y)
    else:
        soln_y = [0]
    soln_z = list(soln_z)
    msg = 'the solutions are:'
    log(msg)
    log(soln_iso)
    log(soln_x)
    if ortho_ab:
        log(soln_y)
    log(soln_z)

    try:
        if (soln_iso and soln_x and soln_y and soln_z and np.max(np.abs(soln_iso[0]))<5e3 and np.max(np.abs(soln_x[0]))<5e3 and
         np.max(np.abs(soln_y[0]))<5e3 and np.max(np.abs(soln_z[0]))<5e3):
            fitted = True
            break
    except Exception as e:
        log(e)
        fitted = False

    if not fitted:
        num_neigh -= 1
        msg = 'looks like these set of equations are either not solvable or yielding unphysical values'
        log(msg)
        msg = 'reducing the number of included NNs to '+str(num_neigh)
        log(msg)

        
if not fitted:
    msg = '*** could not fit the Hamiltonian after several tries, exiting'
    log(msg)
    sys.exit()

CN1 = np.mean(CN1s)
CN2 = np.mean(CN2s)
CN3 = np.mean(CN3s)
CN4 = np.mean(CN4s)

if ortho_ab:
    msg = 'orthogonal a and b vectors found for the lattice, using the XYZ model'
    log(msg)
else:
    soln_y = soln_x
    msg = 'non-orthogonal a and b vectors found for the lattice, using XXZ model'
    log(msg)

if num_config==2:
    E0, J1 = soln_iso[0][0], soln_iso[0][1]
    J2, J3, J4 = 0, 0, 0
    K1 = np.array([soln_x[0][0], soln_y[0][0], soln_z[0][0]])
    K2, K3, K4 = np.zeros(3), np.zeros(3), np.zeros(3)
    A = np.array([soln_x[0][1], soln_y[0][1], soln_z[0][1]])
    msg = 'the NN corordinations for all configs and sites are: '+str(CN1s)
    log(msg)
    msg = '### the solutions are:'
    log(msg)
    msg = 'E0 = '+str(E0)+' eV'
    log(msg)
    msg = 'J1 = '+str(J1*1e3)+' meV/link with d1 = '+str(ds_stable[1])+' ang and avg. NN coordination = '+str(CN1)
    log(msg)
    msg = 'K1 = '+str(K1*1e3)+' meV/link'
    log(msg)
    msg = 'A = '+str(A*1e3)+' meV/magnetic_atom'
    log(msg)
    
elif num_config==3:
    E0, J1, J2 = soln_iso[0][0], soln_iso[0][1], soln_iso[0][2]
    J3, J4 = 0, 0
    K1 = np.array([soln_x[0][0], soln_y[0][0], soln_z[0][0]])
    K2 = np.array([soln_x[0][1], soln_y[0][1], soln_z[0][1]])
    K3, K4 = np.zeros(3), np.zeros(3)
    A =np.array([soln_x[0][2], soln_y[0][2], soln_z[0][2]])
    msg = 'the NN corordinations for all configs and sites are: '+str(CN1s)
    log(msg)
    msg = 'the NNN corordinations for all configs and sites are: '+str(CN2s)
    log(msg)
    msg = '### the solutions are:'
    log(msg)
    msg = 'E0 = '+str(E0)+' eV'
    log(msg)
    msg = 'J1 = '+str(J1*1e3)+' meV/link with d1 = '+str(ds_stable[1])+' ang and avg. NN coordination = '+str(CN1)
    log(msg)
    msg = 'K1 = '+str(K1*1e3)+' meV/link'
    log(msg)
    msg = 'J2 = '+str(J2*1e3)+' meV/link with d2 = '+str(ds_stable[2])+' ang and avg. NNN coordination = '+str(CN2)
    log(msg)
    msg = 'K2 = '+str(K2*1e3)+' meV/link'
    log(msg)
    msg = 'A = '+str(A*1e3)+' meV/magnetic_atom'
    log(msg)
    
elif num_config==4:
    E0, J1, J2, J3 = soln_iso[0][0], soln_iso[0][1], soln_iso[0][2], soln_iso[0][3]
    J4 = 0
    K1 = np.array([soln_x[0][0], soln_y[0][0], soln_z[0][0]])
    K2 = np.array([soln_x[0][1], soln_y[0][1], soln_z[0][1]])
    K3 = np.array([soln_x[0][2], soln_y[0][2], soln_z[0][2]])
    K4 = np.zeros(3)
    A = np.array([soln_x[0][3], soln_y[0][3], soln_z[0][3]])
    msg = 'the NN corordinations for all configs and sites are: '+str(CN1s)
    log(msg)
    msg = 'the NNN corordinations for all configs and sites are: '+str(CN2s)
    log(msg)
    msg = 'the NNNN corordinations for all configs and sites are: '+str(CN3s)
    log(msg)
    msg = '### the solutions are:'
    log(msg)
    msg = 'E0 = '+str(E0)+' eV'
    log(msg)
    msg = 'J1 = '+str(J1*1e3)+' meV/link with d1 = '+str(ds_stable[1])+' ang and avg. NN coordination = '+str(CN1)
    log(msg)
    msg = 'K1 = '+str(K1*1e3)+' meV/link'
    log(msg)
    msg = 'J2 = '+str(J2*1e3)+' meV/link with d2 = '+str(ds_stable[2])+' ang and avg. NNN coordination = '+str(CN2)
    log(msg)
    msg = 'K2 = '+str(K2*1e3)+' meV/link'
    log(msg)
    msg = 'J3 = '+str(J3*1e3)+' meV/link with d3 = '+str(ds_stable[3])+' ang and avg. NNNN coordination = '+str(CN3)
    log(msg)
    msg = 'K3 = '+str(K3*1e3)+' meV/link'
    log(msg)
    msg = 'A = '+str(A*1e3)+' meV/magnetic_atom'
    log(msg)
    
elif num_config==5:
    E0, J1, J2, J3, J4 = soln_iso[0][0], soln_iso[0][1], soln_iso[0][2], soln_iso[0][3], soln_iso[0][4]
    K1 = np.array([soln_x[0][0], soln_y[0][0], soln_z[0][0]])
    K2 = np.array([soln_x[0][1], soln_y[0][1], soln_z[0][1]])
    K3 = np.array([soln_x[0][2], soln_y[0][2], soln_z[0][2]])
    K4 = np.array([soln_x[0][3], soln_y[0][3], soln_z[0][3]])
    A = np.array([soln_x[0][4], soln_y[0][4], soln_z[0][4]])
    msg = 'the NN corordinations for all configs and sites are: '+str(CN1s)
    log(msg)
    msg = 'the NNN corordinations for all configs and sites are: '+str(CN2s)
    log(msg)
    msg = 'the NNNN corordinations for all configs and sites are: '+str(CN3s)
    log(msg)
    msg = 'the NNNNN corordinations for all configs and sites are: '+str(CN4s)
    log(msg)
    msg = 'the solutions are:'
    log(msg)
    msg = 'E0 = '+str(E0)+' eV'
    log(msg)
    msg = 'J1 = '+str(J1*1e3)+' meV/link with d1 = '+str(ds_stable[1])+' ang and avg. NN coordination = '+str(CN1)
    log(msg)
    msg = 'K1 = '+str(K1*1e3)+' meV/link'
    log(msg)    
    msg = 'J2 = '+str(J2*1e3)+' meV/link with d2 = '+str(ds_stable[2])+' ang and avg. NNN coordination = '+str(CN2)
    log(msg)
    msg = 'K2 = '+str(K2*1e3)+' meV/link'
    log(msg)
    msg = 'J3 = '+str(J3*1e3)+' meV/link with d3 = '+str(ds_stable[3])+' ang and avg. NNNN coordination = '+str(CN3)
    log(msg)
    msg = 'K3 = '+str(K3*1e3)+' meV/link'
    log(msg)   
    msg = 'J4 = '+str(J4*1e3)+' meV/link with d4 = '+str(ds_stable[4])+' ang and avg. NNNNN coordination = '+str(CN4)
    log(msg)
    msg = 'K4 = '+str(K4*1e3)+' meV/link'
    log(msg)
    msg = 'A = '+str(A*1e3)+' meV/magnetic_atom'
    log(msg)
    

if ds_stable[1]/ds_stable[2] >= 0.8:
    msg = '** d1/d2 is greater than 0.8, consider adding the 2nd neighbor for accurate results'
    log(msg)
    
elif ds_stable[1]/ds_stable[3] >= 0.7:
    msg = '** d1/d3 is greater than 0.7, consider adding the 3rd neighbor for accurate results'
    log(msg)

msg = 'the Hamiltonian fitting procedure finished successfullly, now starting the Monte-Carlo simulation'
log(msg)

if not os.path.exists(root_path+'/input_MC'):
    msg = 'no input_MC file detected, writing this'
    log(msg)
    T_MF = np.abs((S_stable*(S_stable+1)/(3*kB))*(J1*len(N1s)) + (S_stable*(S_stable+1)/(3*kB))*(J2*len(N2s)) + (
        S_stable*(S_stable+1)/(3*kB))*(J3*len(N3s)) + (S_stable*(S_stable+1)/(3*kB))*(J4*len(N4s)))

    f = open('input_MC','w+')
    f.write('directory = MC_Heisenberg\n')
    f.write('repeat = 25 25 1\n')
    f.write('restart = 0\n')
    f.write('J1 (eV/link) = '+str(J1)+'\n')
    f.write('J2 (eV/link) = '+str(J2)+'\n')
    f.write('J3 (eV/link) = '+str(J3)+'\n')
    f.write('J4 (eV/link) = '+str(J4)+'\n')
    f.write('K1x (eV/link) = '+str(K1[0])+'\n')
    f.write('K1y (eV/link) = '+str(K1[1])+'\n')
    f.write('K1z (eV/link) = '+str(K1[2])+'\n')
    f.write('K2x (eV/link) = '+str(K2[0])+'\n')
    f.write('K2y (eV/link) = '+str(K2[1])+'\n')
    f.write('K2z (eV/link) = '+str(K2[2])+'\n')
    f.write('K3x (eV/link) = '+str(K3[0])+'\n')
    f.write('K3y (eV/link) = '+str(K3[1])+'\n')
    f.write('K3z (eV/link) = '+str(K3[2])+'\n')
    f.write('K4x (eV/link) = '+str(K4[0])+'\n')
    f.write('K4y (eV/link) = '+str(K4[1])+'\n')
    f.write('K4z (eV/link) = '+str(K4[2])+'\n')
    f.write('Ax (eV/mag_atom) = '+str(A[0])+'\n')
    f.write('Ay (eV/mag_atom) = '+str(A[1])+'\n')
    f.write('Az (eV/mag_atom) = '+str(A[2])+'\n')
    f.write('mu (mu_B/mag_atom) = '+str(magmom_stable)+'\n')
    f.write('EMA = '+str(EMA)+'\n')
    f.write('T_start (K) = '+str(1e-6)+'\n')
    f.write('T_end (K) = '+str(T_MF)+'\n')
    f.write('div_T = 25\n')
    f.write('MCS = 100000\n')
    f.write('thresh = 10000\n')
    f.close()

    msg = 'successfully written input_MC, now will try to run Monte-Carlo based on this'
    log(msg)
    msg = 'if you want to run the MC with some other settings, make the neccesarry changes in input_MC and stop and re-run this script'
    log(msg)
    sleep(3)

else:
    msg = 'existing input_MC detected, will try to run the MC based on this'
    log(msg)
    sleep(3)


with open('input_MC') as f: 
    for line in f:
        row = line.split()
        if 'directory' in line:
            path = root_path+'/'+row[-1]
        elif 'restart' in line:
            restart = int(row[-1])
        elif 'repeat' in line:
            rep_z = int(row[-1])
            rep_y = int(row[-2])
            rep_x = int(row[-3])
        elif 'J1' in line:
            J1 = np.double(row[-1])
        elif 'J2' in line:
            J2 = np.double(row[-1])
        elif 'J3' in line:
            J3 = np.double(row[-1])
        elif 'J4' in line:
            J4 = np.double(row[-1])
        elif 'K1x' in line:
            K1x = np.double(row[-1])
        elif 'K1y' in line:
            K1y = np.double(row[-1])
        elif 'K1z' in line:
            K1z = np.double(row[-1])
        elif 'K2x' in line:
            K2x = np.double(row[-1])
        elif 'K2y' in line:
            K2y = np.double(row[-1])
        elif 'K2z' in line:
            K2z = np.double(row[-1])
        elif 'K3x' in line:
            K3x = np.double(row[-1])
        elif 'K3y' in line:
            K3y = np.double(row[-1])
        elif 'K3z' in line:
            K3z = np.double(row[-1])
        elif 'K4x' in line:
            K4x = np.double(row[-1])
        elif 'K4y' in line:
            K4y = np.double(row[-1])
        elif 'K4z' in line:
            K4z = np.double(row[-1])
        elif 'Ax' in line:
            Ax = np.double(row[-1])
        elif 'Ay' in line:
            Ay = np.double(row[-1])
        elif 'Az' in line:
            Az = np.double(row[-1])
        elif 'EMA' in line:
            EMA = int(row[-1])
        elif 'T_start' in line:
            Tstart = float(row[-1])
        elif 'T_end' in line:
            Trange = float(row[-1])
        elif 'div_T' in line:
            div_T = int(row[-1])
        elif 'mu' in line:
            mu = float(row[-1])
        elif 'MCS' in line:
            trange = int(row[-1])
        elif 'thresh' in line:
            threshold = int(row[-1])

if os.path.exists(path):
    new_name = path+'_'+str(time())
    os.rename(path,new_name)
    msg = 'found an old MC directory, renaming it to '+new_name
    log(msg)

os.makedirs(path)
try:
    copyfile(new_name+'/N1list',path+'/N1list')
    copyfile(new_name+'/N2list',path+'/N2list')
    copyfile(new_name+'/N3list',path+'/N3list')
    copyfile(new_name+'/N4list',path+'/N4list')
except:
    pass

        
repeat = [rep_x,rep_y,rep_z]
S = mu/2
os.chdir(path)

struct_mag_stable.make_supercell(repeat)
N = len(struct_mag_stable)
spins_init = np.array(struct_mag_stable.site_properties['magmom'][:])/2.0

if restart==0:
    dr_max = ds_stable[0]
    d_N1 = ds_stable[1]
    d_N2 = ds_stable[2]
    d_N3 = ds_stable[3]
    d_N4 = ds_stable[4]
    all_coords = [0]*N
    for i in range(N):
        all_coords[i] = [struct_mag_stable[i].x,struct_mag_stable[i].y,struct_mag_stable[i].z]
    all_coords = np.array(all_coords,dtype='float32')
    all_coords = all_coords.flatten()
    N1list = [[1,2]]*N
    N2list = [[1,2]]*N
    N3list = [[1,2]]*N
    N4list = [[1,2]]*N

    if GPU_accel:
        nf = Nfinder_GPU
        msg = 'neighbor mapping will try to use GPU acceleration'
        log(msg)
    else:
        nf = Nfinder
        msg = 'neighbor mapping will be sequentially done in CPU, can be quite slow'
        log(msg)
        
    start_time_map = time()
    for i in range(N):
        N1list[i] = nf(struct_mag_stable,i,d_N1,dr_max)
        if J2!=0:
            N2list[i] = nf(struct_mag_stable,i,d_N2,dr_max)
        if J3!=0:
            N3list[i] = nf(struct_mag_stable,i,d_N3,dr_max)
        if J4!=0:
            N4list[i] = nf(struct_mag_stable,i,d_N4,dr_max)
        print(str(i)+' / '+str(N-1)+ ' mapped')

    if padding:
        msg = 'anticipating inhomogenous number of neighbors for some atoms, trying padding'
        log(msg)
        make_homogenous(N1list)
        make_homogenous(N2list)
        make_homogenous(N3list)
        make_homogenous(N4list)

    end_time_map = time()
    time_map = np.around(end_time_map - start_time_map, 2)
    with open('N1list', 'wb') as f:
        dump(N1list, f)
    with open('N2list', 'wb') as f:
        dump(N2list, f)
    with open('N3list', 'wb') as f:
        dump(N3list, f)
    with open('N4list', 'wb') as f:
        dump(N4list, f)
    msg = 'neighbor mapping finished and dumped'
    log(msg)
    msg = 'the neighbor mapping process for a '+str(N)+' site lattice took '+str(time_map)+' s'
    log(msg)    

else:
    with open('N1list', 'rb') as f:
        N1list = load(f)
    with open('N2list', 'rb') as f:
        N2list = load(f)
    with open('N3list', 'rb') as f:
        N3list = load(f)
    with open('N4list', 'rb') as f:
        N4list = load(f)
    N = len(N1list)
    log('neighbor mapping successfully read')

N1list = np.array(N1list)
N2list = np.array(N2list)
N3list = np.array(N3list)
N4list = np.array(N4list)

temp = N1list.flatten()
corrupt = np.count_nonzero(temp == -5)
msg = 'the amount of site corruption in N1s is ' + str(corrupt) + ' / ' + str(len(temp)) + ', or ' + str(100.0*corrupt/len(temp)) + '%'
log(msg)
if J2!=0:
    temp = N2list.flatten()
    corrupt = np.count_nonzero(temp == -5)
    msg = 'the amount of site corruption in N2s is ' + str(corrupt) + ' / ' + str(len(temp)) + ', or ' + str(100.0*corrupt/len(temp)) + '%'
    log(msg)    
if J3!=0:
    temp = N3list.flatten()
    corrupt = np.count_nonzero(temp == -5)
    msg = 'the amount of site corruption in N3s is ' + str(corrupt) + ' / ' + str(len(temp)) + ', or ' + str(100.0*corrupt/len(temp)) + '%'
    log(msg)
if J4!=0:
    temp = N4list.flatten()
    corrupt = np.count_nonzero(temp == -5)
    msg = 'the amount of site corruption in N4s is ' + str(corrupt) + ' / ' + str(len(temp)) + ', or ' + str(100.0*corrupt/len(temp)) + '%'
    log(msg)
    

Ts = np.linspace(Tstart,Trange,div_T)
M_ups = []
X_ups = []
M_downs = []
X_downs = []
M_tots = []
X_tots = []

start_time_mc = time()

for T in Ts:

    M_up,X_up,M_down,X_down,M_tot,X_tot,spins_x,spins_y,spins_z = MC_func(spins_init,T,J2!=0,J3!=0,J4!=0)

    results = str(T)+'    '+str(M_up)+'    '+str(X_up)+'    '+str(M_down)+'    '+str(X_down)+'    '+str(M_tot)+'    '+str(X_tot)
    print(results)
    material = final_list[0][1].composition.reduced_formula
    f = open(material+'_'+str(int(np.floor(Tstart)))+'K-'+str(int(np.floor(Trange)))+'K_M-X.dat','a+')
    f.write(results+'\n')
    f.close()
    if dump_spins:
        f = open(material+'_'+
            str(int(np.floor(Tstart)))+'K-'+str(int(np.floor(Trange)))+'K_spinsdump.dat','a+')
        f.write('T = '+str(T)+' K'+'\n\n')
        f.write('Sx = \n')
        f.write(str(spins_x.tolist()))
        f.write('\n\n')
        f.write('Sy = \n')
        f.write(str(spins_y.tolist()))
        f.write('\n\n')
        f.write('Sz = \n')
        f.write(str(spins_z.tolist()))
        f.write('\n\n')
        f.write('-'*100+'\n\n')
        f.close()

end_time_mc = time()
time_mc = np.around(end_time_mc - start_time_mc, 2)
msg = 'MC simulation have finished, analyse the output to determine the Curire/Neel temp.'
log(msg)
msg = 'the MC simulation took '+str(time_mc)+' s'
log(msg)

end_time_global = time()
time_global = np.around(end_time_global - start_time_global, 2)
msg = 'the whole end-to-end process took '+str(time_global)+' s'
log(msg)