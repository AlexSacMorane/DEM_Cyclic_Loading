# once YADE is installed (https://yade-dem.org/doc/)
# command yade -j n_processors -n -x Rock_Cyclic_Isotropic.py
# the flag -n deactivate the GUI of YADE
# the flag -x induce the close of the yade environnement at the end of the simulation
# n_processors is the number of processors used for the simulation
# see the online doc available

#-------------------------------------------------------------------------------
#Librairies
#-------------------------------------------------------------------------------

from yade import pack, plot, export
import numpy as np
import matplotlib.pyplot as plt
import os, shutil, time, math, random
from pathlib import Path

#-------------------------------------------------------------------------------
#User
#-------------------------------------------------------------------------------

# PSD
n_grains = 3000 # 1600 is the minimum
L_r = []

# Particles
rMean = 0.000100  # m
rRelFuzz = .0

# Box
Dz_on_Dx = 1 # ratio Dz / Dxy
Dz = 0.0028 # m
Dx = Dz/Dz_on_Dx
Dy = Dx

# IC
n_steps_ic = 100
unbalancedForce_criterion_ic = 0.01

# Walls
P_load = 1e5 # Pa
P_load0 = P_load # save it
kp = 5e-9 # m.N-1
k_v_max = 0.00002 #-
flag_plot_vWalls = True # generate a plotfor debug

# cementation
P_cementation = P_load*0.1 # Pa
# 2T   : f_cemented (0.13), m_log (6.79), s_log (0.70), E ( 300MPa), Ab_mean (148-12)
# 2MB  : f_cemented (0.88), m_log (7.69), s_log (0.60), E ( 320MPa), Ab_mean (2303e-12)
# 11BB : f_cemented (0.98), m_log (8.01), s_log (0.88), E ( 760MPa), Ab_mean (4267e-12)
# 13BT : f_cemented (1.00), m_log (8.44), s_log (0.92), E ( 860MPa), Ab_mean (6577e-12)
# 13MB : f_cemented (1.00), m_log (8.77), s_log (0.73), E (1000MPa), Ab_mean (8137e-12)
type_cementation = 'user' # only for the report
f_cemented = 1. # -
#m_log = 8.77 # -
#s_log = 0.73 # -
YoungModulus = 1000e6 # Pa # elastic and for the contact
Ab_mean = 8137e-12 # m2
# mechanical rupture
factor_strength = 1
tensileCohesion = 2.75e6*factor_strength # Pa
shearCohesion = 6.6e6*factor_strength # Pa

# Cyclic loading
number_cycles = 1
amplitude_P_cycle = 0.9*P_load
d_P_d_iteration = P_load/1e5 # try to verify unbalancedForce < 0.1 (the smaller, the better, the more time consuming)

# time step
factor_dt_crit_ic = 0.6
factor_dt_crit_main = 0.2

# Report
simulation_report_name = O.tags['d.id']+'_report.txt'
simulation_report = open(simulation_report_name, 'w')
simulation_report.write('Isotropic Loading test\n')
simulation_report.write('Type of sample: Rock\n')
simulation_report.write('Cementation at '+str(int(P_cementation))+' Pa\n')
simulation_report.write('Type of cementation: '+type_cementation+'\n')
simulation_report.write('Confinement at '+str(int(P_load))+' Pa\n')
simulation_report.close()

#-------------------------------------------------------------------------------
#Initialisation
#-------------------------------------------------------------------------------

# clock to show performances
tic = time.perf_counter()
tic_0 = tic
iter_0 = 0

# plan simulation
if Path('plot').exists():
    shutil.rmtree('plot')
os.mkdir('plot')
if Path('data').exists():
    shutil.rmtree('data')
os.mkdir('data')

# define wall material (no friction)
O.materials.append(CohFrictMat(young=80e6, poisson=0.25, frictionAngle=0, density=2650, isCohesive=False, momentRotationLaw=False))

# create box with infinite wall and grains
O.bodies.append(aabbWalls([Vector3(0,0,0),Vector3(Dx,Dy,Dz)], thickness=0.,oversizeFactor=10))

# global names
plate_x_min = O.bodies[0]
plate_x_max = O.bodies[1]
plate_y_min = O.bodies[2]
plate_y_max = O.bodies[3]
plate_z_min = O.bodies[4]
plate_z_max = O.bodies[5]

# define grain material
O.materials.append(CohFrictMat(young=80e6, poisson=0.25, frictionAngle=atan(0.05), density=2650,\
                               isCohesive=True, normalCohesion=tensileCohesion, shearCohesion=shearCohesion,\
                               momentRotationLaw=True, alphaKr=0, alphaKtw=0))
# frictionAngle, alphaKr, alphaKtw are set to 0 during IC. The real value is set after IC.
frictionAngleReal = radians(20)
alphaKrReal = 0.5
alphaKtwReal = 0.5

# generate grain
for i in range(n_grains):
    radius = random.uniform(rMean*(1-rRelFuzz),rMean*(1+rRelFuzz))
    center_x = random.uniform(0+radius/n_steps_ic, Dx-radius/n_steps_ic)
    center_y = random.uniform(0+radius/n_steps_ic, Dy-radius/n_steps_ic)
    center_z = random.uniform(0+radius/n_steps_ic, Dz-radius/n_steps_ic)
    O.bodies.append(sphere(center=[center_x, center_y, center_z], radius=radius/n_steps_ic))
    # can use b.state.blockedDOFs = 'xyzXYZ' to block translation of rotation of a body
    L_r.append(radius)
O.tags['Step ic'] = '1'

# yade algorithm
O.engines = [
        PyRunner(command='grain_in_box()', iterPeriod = 1000),
        ForceResetter(),
        # sphere, wall
        InsertionSortCollider([Bo1_Sphere_Aabb(), Bo1_Box_Aabb()]),
        InteractionLoop(
                # need to handle sphere+sphere and sphere+wall
                # Ig : compute contact point. Ig2_Sphere (3DOF) or Ig2_Sphere6D (6DOF)
                # Ip : compute parameters needed
                # Law : compute contact law with parameters from Ip
                [Ig2_Sphere_Sphere_ScGeom6D(), Ig2_Box_Sphere_ScGeom6D()],
                [Ip2_CohFrictMat_CohFrictMat_CohFrictPhys()],
                [Law2_ScGeom6D_CohFrictPhys_CohesionMoment(always_use_moment_law=True)]
        ),
        NewtonIntegrator(gravity=(0, 0, 0), damping=0.001, label = 'Newton'),
        PyRunner(command='checkUnbalanced_ir_ic()', iterPeriod = 200, label='checker')
]
# time step
O.dt = factor_dt_crit_ic * PWaveTimeStep()

#-------------------------------------------------------------------------------

def grain_in_box():
    '''
    Delete grains outside the box.
    '''
    #detect grain outside the box
    L_id_to_delete = []
    for b in O.bodies :
        if isinstance(b.shape, Sphere):
            #limit x \ limit y \ limit z
            if b.state.pos[0] < plate_x_min.state.pos[0] or plate_x_max.state.pos[0] < b.state.pos[0] or \
            b.state.pos[1] < plate_y_min.state.pos[1] or plate_y_max.state.pos[1] < b.state.pos[1] or \
            b.state.pos[2] < plate_z_min.state.pos[2] or plate_z_max.state.pos[2] < b.state.pos[2] :
                L_id_to_delete.append(b.id)
    if L_id_to_delete != []:
        #delete grain detected
        for id in L_id_to_delete:
            O.bodies.erase(id)
        #print and report
        simulation_report = open(simulation_report_name, 'a')
        simulation_report.write(str(len(L_id_to_delete))+" grains erased (outside of the box)\n")
        simulation_report.close()
        print("\n"+str(len(L_id_to_delete))+" grains erased (outside of the box)\n")

#-------------------------------------------------------------------------------

def checkUnbalanced_ir_ic():
    '''
    Increase particle radius until a steady-state is found.
    '''
    # the rest will be run only if unbalanced is < .1 (stabilized packing)
    # Compute the ratio of mean summary force on bodies and mean force magnitude on interactions.
    if unbalancedForce() > .1:
        return
    # increase the radius of particles
    if int(O.tags['Step ic']) < n_steps_ic :
        print('IC step '+O.tags['Step ic']+'/'+str(n_steps_ic)+' done')
        O.tags['Step ic'] = str(int(O.tags['Step ic'])+1)
        i_L_r = 0
        for b in O.bodies :
            if isinstance(b.shape, Sphere):
                growParticle(b.id, int(O.tags['Step ic'])/n_steps_ic*L_r[i_L_r]/b.shape.radius)
                i_L_r = i_L_r + 1
        # update the dt as the radii change
        O.dt = factor_dt_crit_ic * PWaveTimeStep()
        return
    # characterize the ic algorithm
    global tic
    global iter_0
    tac = time.perf_counter()
    hours = (tac-tic)//(60*60)
    minutes = (tac-tic -hours*60*60)//(60)
    seconds = int(tac-tic -hours*60*60 -minutes*60)
    tic = tac
    # report
    simulation_report = open(simulation_report_name, 'a')
    simulation_report.write("IC Generated : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds\n")
    simulation_report.write(str(O.iter-iter_0)+' Iterations\n')
    simulation_report.write(str(n_grains)+' grains\n\n')
    simulation_report.close()
    print("\nIC Generated : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds\n")
    # next time, do not call this function anymore, but the next one instead
    iter_0 = O.iter
    checker.command = 'checkUnbalanced_load_cementation_ic()'
    checker.iterPeriod = 500
    # control top wall
    O.engines = O.engines + [PyRunner(command='controlWalls_ic()', iterPeriod = 1)]
    # switch on the gravity
    Newton.gravity = [0, 0, -9.81]

#-------------------------------------------------------------------------------

def controlWalls_ic():
    '''
    Control the walls to applied a defined confinement force.

    The displacement of the wall depends on the force difference. A maximum value is defined.
    '''
    Fx = (O.forces.f(plate_x_max.id)[0] - O.forces.f(plate_x_min.id)[0])/2
    if Fx == 0:
        plate_x_min.state.pos =  (min([b.state.pos[0]-0.995*b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)]),
                                       (plate_y_min.state.pos[1]+plate_y_max.state.pos[1])/2,
                                       (plate_z_min.state.pos[2]+plate_z_max.state.pos[2])/2)
        plate_x_max.state.pos =  (max([b.state.pos[0]+0.995*b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)]),
                                      (plate_y_min.state.pos[1]+plate_y_max.state.pos[1])/2,
                                      (plate_z_min.state.pos[2]+plate_z_max.state.pos[2])/2)
    else :
        dF = Fx - P_cementation*(plate_y_max.state.pos[1]-plate_y_min.state.pos[1])*(plate_z_max.state.pos[2]-plate_z_min.state.pos[2])
        v_plate_max = rMean*k_v_max/O.dt
        v_try_abs = abs(kp*dF)/O.dt
        # maximal speed is applied to lateral wall
        if v_try_abs < v_plate_max :
            plate_x_min.state.vel = (-np.sign(dF)*v_try_abs, 0, 0)
            plate_x_max.state.vel = ( np.sign(dF)*v_try_abs, 0, 0)
        else :
            plate_x_min.state.vel = (-np.sign(dF)*v_plate_max, 0, 0)
            plate_x_max.state.vel = ( np.sign(dF)*v_plate_max, 0, 0)
    
    Fy = (O.forces.f(plate_y_max.id)[1] - O.forces.f(plate_y_min.id)[1])/2
    if Fy == 0:
        plate_y_min.state.pos =  ((plate_x_min.state.pos[0]+plate_x_max.state.pos[0])/2,
                                  min([b.state.pos[1]-0.995*b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)]),
                                  (plate_z_min.state.pos[2]+plate_z_max.state.pos[2])/2)                    
        plate_y_max.state.pos =  ((plate_x_min.state.pos[0]+plate_x_max.state.pos[0])/2,
                                  max([b.state.pos[1]+0.995*b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)]),
                                  (plate_z_min.state.pos[2]+plate_z_max.state.pos[2])/2)
    else :
        dF = Fy - P_cementation*(plate_x_max.state.pos[0]-plate_x_min.state.pos[0])*(plate_z_max.state.pos[2]-plate_z_min.state.pos[2])
        v_plate_max = rMean*k_v_max/O.dt
        v_try_abs = abs(kp*dF)/O.dt
        # maximal speed is applied to lateral wall
        if v_try_abs < v_plate_max :
            plate_y_min.state.vel = (0, -np.sign(dF)*v_try_abs, 0)
            plate_y_max.state.vel = (0,  np.sign(dF)*v_try_abs, 0)
        else :
            plate_y_min.state.vel = (0, -np.sign(dF)*v_plate_max, 0)
            plate_y_max.state.vel = (0,  np.sign(dF)*v_plate_max, 0)

    Fz = (O.forces.f(plate_z_max.id)[2] - O.forces.f(plate_z_min.id)[2])/2
    if Fz == 0:
        plate_z_min.state.pos =  ((plate_x_min.state.pos[0]+plate_x_max.state.pos[0])/2,
                                  (plate_y_min.state.pos[1]+plate_y_max.state.pos[1])/2,
                                  min([b.state.pos[2]-0.995*b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)]))
        plate_z_max.state.pos =  ((plate_x_min.state.pos[0]+plate_x_max.state.pos[0])/2,
                                  (plate_y_min.state.pos[1]+plate_y_max.state.pos[1])/2,
                                  max([b.state.pos[2]+0.995*b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)]))
    else :
        dF = Fz - P_cementation*(plate_x_max.state.pos[0]-plate_x_min.state.pos[0])*(plate_y_max.state.pos[1]-plate_y_min.state.pos[1])
        v_plate_max = rMean*k_v_max/O.dt
        v_try_abs = abs(kp*dF)/O.dt
        # maximal speed is applied to top wall
        if v_try_abs < v_plate_max :
            plate_z_min.state.vel = (0, 0, -np.sign(dF)*v_try_abs)
            plate_z_max.state.vel = (0, 0,  np.sign(dF)*v_try_abs)
        else :
            plate_z_min.state.vel = (0, 0, -np.sign(dF)*v_plate_max)
            plate_z_max.state.vel = (0, 0,  np.sign(dF)*v_plate_max)

#-------------------------------------------------------------------------------

def checkUnbalanced_load_cementation_ic():
    '''
    Wait to reach the confining pressure targetted for cementation.
    '''
    addPlotData_cementation_ic()
    saveData_ic()
    # check the force applied
    if abs((O.forces.f(plate_x_max.id)[0]-O.forces.f(plate_x_min.id)[0])/2 - P_cementation*(plate_y_max.state.pos[1]-plate_y_min.state.pos[1])*(plate_z_max.state.pos[2]-plate_z_min.state.pos[2]))/\
        (P_cementation*(plate_y_max.state.pos[1]-plate_y_min.state.pos[1])*(plate_z_max.state.pos[2]-plate_z_min.state.pos[2])) > 0.005 :
        return
    if abs((O.forces.f(plate_y_max.id)[1]-O.forces.f(plate_y_min.id)[1])/2 - P_cementation*(plate_x_max.state.pos[0]-plate_x_min.state.pos[0])*(plate_z_max.state.pos[2]-plate_z_min.state.pos[2]))/\
        (P_cementation*(plate_x_max.state.pos[0]-plate_x_min.state.pos[0])*(plate_z_max.state.pos[2]-plate_z_min.state.pos[2])) > 0.005 :
        return
    if abs((O.forces.f(plate_z_max.id)[2]-O.forces.f(plate_z_min.id)[2])/2 - P_cementation*(plate_x_max.state.pos[0]-plate_x_min.state.pos[0])*(plate_y_max.state.pos[1]-plate_y_min.state.pos[1]))/\
        (P_cementation*(plate_x_max.state.pos[0]-plate_x_min.state.pos[0])*(plate_y_max.state.pos[1]-plate_y_min.state.pos[1])) > 0.005 :
        return
    if unbalancedForce() > unbalancedForce_criterion_ic :
        return
    # characterize the ic algorithm
    global tic
    tac = time.perf_counter()
    hours = (tac-tic)//(60*60)
    minutes = (tac-tic -hours*60*60)//(60)
    seconds = int(tac-tic -hours*60*60 -minutes*60)
    tic = tac
    # report
    simulation_report = open(simulation_report_name, 'a')
    simulation_report.write("Pressure (Cementation) applied : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds\n")
    simulation_report.write(str(n_grains)+' grains\n\n')
    simulation_report.close()
    print("\nPressure (Cementation) applied : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds\n")
    # switch on friction, bending resistance and twisting resistance between particles
    O.materials[-1].frictionAngle = frictionAngleReal
    O.materials[-1].alphaKr = alphaKrReal
    O.materials[-1].alphaKtw = alphaKtwReal
    # for existing contacts, clear them
    O.interactions.clear()
    # calm down particles
    for b in O.bodies:
        if isinstance(b.shape,Sphere):
            b.state.angVel = Vector3(0,0,0)
            b.state.vel = Vector3(0,0,0)
    # switch off damping
    #Newton.damping = 0
    # next time, do not call this function anymore, but the next one instead
    checker.command = 'checkUnbalanced_param_ic()'

#-------------------------------------------------------------------------------

def checkUnbalanced_param_ic():
    '''
    Wait to reach the equilibrium after switching on the friction and the rolling resistances.
    '''
    addPlotData_cementation_ic()
    saveData_ic()
    # check the force applied
    if abs((O.forces.f(plate_x_max.id)[0]-O.forces.f(plate_x_min.id)[0])/2 - P_cementation*(plate_y_max.state.pos[1]-plate_y_min.state.pos[1])*(plate_z_max.state.pos[2]-plate_z_min.state.pos[2]))/\
            (P_cementation*(plate_y_max.state.pos[1]-plate_y_min.state.pos[1])*(plate_z_max.state.pos[2]-plate_z_min.state.pos[2])) > 0.005 :
        return
    if abs((O.forces.f(plate_y_max.id)[1]-O.forces.f(plate_y_min.id)[1])/2 - P_cementation*(plate_x_max.state.pos[0]-plate_x_min.state.pos[0])*(plate_z_max.state.pos[2]-plate_z_min.state.pos[2]))/\
            (P_cementation*(plate_x_max.state.pos[0]-plate_x_min.state.pos[0])*(plate_z_max.state.pos[2]-plate_z_min.state.pos[2])) > 0.005 :
        return
    if abs((O.forces.f(plate_z_max.id)[2]-O.forces.f(plate_z_min.id)[2])/2 - P_cementation*(plate_x_max.state.pos[0]-plate_x_min.state.pos[0])*(plate_y_max.state.pos[1]-plate_y_min.state.pos[1]))/\
            (P_cementation*(plate_x_max.state.pos[0]-plate_x_min.state.pos[0])*(plate_y_max.state.pos[1]-plate_y_min.state.pos[1])) > 0.005 :
        return
    if unbalancedForce() > unbalancedForce_criterion_ic :
        return
    # characterize the ic algorithm
    global tic
    tac = time.perf_counter()
    hours = (tac-tic)//(60*60)
    minutes = (tac-tic -hours*60*60)//(60)
    seconds = int(tac-tic -hours*60*60 -minutes*60)
    tic = tac
    # report
    simulation_report = open(simulation_report_name, 'a')
    simulation_report.write("Parameters applied : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds\n\n")
    simulation_report.close()
    print("\nParameters applied : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds\n")
    # next time, do not call this function anymore, but the next one instead
    checker.command = 'cementation()'
    checker.iterPeriod = 10

#------------------------------------------------------------------------------

def pdf_lognormal(x,m_log,s_log):
    '''
    Return the probability of a value x with a log normal function defined by the mean m_log and the variance s_log.
    '''
    p = np.exp(-(np.log(x)-m_log)**2/(2*s_log**2))/(x*s_log*np.sqrt(2*np.pi))
    return p

#-------------------------------------------------------------------------------

def cementation():
    '''
    Generate cementation between grains.
    '''
    # generate the list of cohesive surface area and its list of weight
    #x_min = 1e2 # µm2
    #x_max = 4e4 # µm2
    #n_x = 20000
    #x_L = np.linspace(x_min, x_max, n_x)
    #p_x_L = []
    #for x in x_L :
    #    p_x_L.append(pdf_lognormal(x,m_log,s_log))
    # counter
    global counter_bond0
    counter_bond0 = 0
    # iterate on interactions
    for i in O.interactions:
        # only grain-grain contact can be cemented
        if isinstance(O.bodies[i.id1].shape, Sphere) and isinstance(O.bodies[i.id2].shape, Sphere) :
            # only a fraction of the contact is cemented
            if random.uniform(0,1) < f_cemented :
                counter_bond0 = counter_bond0 + 1
                # creation of cohesion
                i.phys.cohesionBroken = False
                # determine the cohesive surface
                #cohesiveSurface = (random.choices(x_L,p_x_L)[0])*1e-12 # m2
                cohesiveSurface = Ab_mean #m2 # uniform cementation 
                # set normal and shear adhesions
                i.phys.normalAdhesion = tensileCohesion*cohesiveSurface
                i.phys.shearAdhesion = shearCohesion*cohesiveSurface
                # set normal and shear adhesions
                i.phys.normalAdhesion = tensileCohesion*cohesiveSurface
                i.phys.shearAdhesion = shearCohesion*cohesiveSurface
                # local law E(Ab)
                localYoungModulus = (YoungModulus-80e6)*cohesiveSurface/Ab_mean + 80e6
                i.phys.kn = localYoungModulus*(O.bodies[i.id1].shape.radius*2*O.bodies[i.id2].shape.radius*2)/(O.bodies[i.id1].shape.radius*2+O.bodies[i.id2].shape.radius*2)
                i.phys.ks = 0.25*localYoungModulus*(O.bodies[i.id1].shape.radius*2*O.bodies[i.id2].shape.radius*2)/(O.bodies[i.id1].shape.radius*2+O.bodies[i.id2].shape.radius*2) # 0.25 is the Poisson ratio
                i.phys.kr = i.phys.ks*alphaKrReal*O.bodies[i.id1].shape.radius*O.bodies[i.id2].shape.radius
                i.phys.ktw = i.phys.ks*alphaKtwReal*O.bodies[i.id1].shape.radius*O.bodies[i.id2].shape.radius
    # write in the report
    simulation_report = open(simulation_report_name, 'a')
    simulation_report.write(str(counter_bond0)+" contacts cemented initially\n\n")
    simulation_report.close()
    print('\n'+str(counter_bond0)+" contacts cemented\n")

    # update of the time step
    O.dt = factor_dt_crit_main * PWaveTimeStep()

    # next time, do not call this function anymore, but the next one instead
    checker.command = 'checkUnbalanced_load_confinement_ic()'
    checker.iterPeriod = 200
    # change the vertical pressure applied
    O.engines = O.engines[:-1] + [PyRunner(command='controlWalls()', iterPeriod = 1)]

#-------------------------------------------------------------------------------

def checkUnbalanced_load_confinement_ic():
    '''
    Wait to reach the vertical/lateral pressure targetted for confinement.
    '''
    addPlotData_confinement_ic()
    saveData_ic()
    # check the force applied
    if abs((O.forces.f(plate_x_max.id)[0]-O.forces.f(plate_x_min.id)[0])/2 - P_load*(plate_y_max.state.pos[1]-plate_y_min.state.pos[1])*(plate_z_max.state.pos[2]-plate_z_min.state.pos[2]))/\
            (P_load*(plate_y_max.state.pos[1]-plate_y_min.state.pos[1])*(plate_z_max.state.pos[2]-plate_z_min.state.pos[2])) > 0.005 :
        return
    if abs((O.forces.f(plate_y_max.id)[1]-O.forces.f(plate_y_min.id)[1])/2 - P_load*(plate_x_max.state.pos[0]-plate_x_min.state.pos[0])*(plate_z_max.state.pos[2]-plate_z_min.state.pos[2]))/\
            (P_load*(plate_x_max.state.pos[0]-plate_x_min.state.pos[0])*(plate_z_max.state.pos[2]-plate_z_min.state.pos[2])) > 0.005 :
        return
    if abs((O.forces.f(plate_z_max.id)[2]-O.forces.f(plate_z_min.id)[2])/2 - P_load*(plate_x_max.state.pos[0]-plate_x_min.state.pos[0])*(plate_y_max.state.pos[1]-plate_y_min.state.pos[1]))/\
            (P_load*(plate_x_max.state.pos[0]-plate_x_min.state.pos[0])*(plate_y_max.state.pos[1]-plate_y_min.state.pos[1])) > 0.005 :
        return
    if unbalancedForce() > unbalancedForce_criterion_ic :
        return
    # characterize the ic algorithm
    global tic, iter_0
    tac = time.perf_counter()
    hours = (tac-tic)//(60*60)
    minutes = (tac-tic -hours*60*60)//(60)
    seconds = int(tac-tic -hours*60*60 -minutes*60)
    tic = tac

    # compute mean overlap/diameter
    L_over_diam = []
    for contact in O.interactions:
        if isinstance(O.bodies[contact.id1].shape, Sphere) and isinstance(O.bodies[contact.id2].shape, Sphere):
            b1_x = O.bodies[contact.id1].state.pos[0]
            b1_y = O.bodies[contact.id1].state.pos[1]
            b1_z = O.bodies[contact.id1].state.pos[2]
            b2_x = O.bodies[contact.id2].state.pos[0]
            b2_y = O.bodies[contact.id2].state.pos[1]
            b2_z = O.bodies[contact.id2].state.pos[2]
            dist = math.sqrt((b1_x-b2_x)**2+(b1_y-b2_y)**2+(b1_z-b2_z)**2)
            over = O.bodies[contact.id1].shape.radius + O.bodies[contact.id2].shape.radius - dist
            diam = 1/(1/(O.bodies[contact.id1].shape.radius*2)+1/(O.bodies[contact.id2].shape.radius*2))
            L_over_diam.append(over/diam)
    m_over_diam = np.mean(L_over_diam)
    print('Mean Overlap/Diameter', m_over_diam)

    # report
    simulation_report = open(simulation_report_name, 'a')
    simulation_report.write("Pressure (Confinement) applied : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds\n")
    simulation_report.write(str(O.iter-iter_0)+' Iterations\n')
    simulation_report.write(str(n_grains)+' grains\n')
    simulation_report.write('Mean Overlap/Diameter ' + str(m_over_diam) + '\n')
    simulation_report.write('IC generation ends\n\n')
    simulation_report.close()
    print("\nPressure (Confinement) applied : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds\n")

    # reset plot (IC done, simulation starts)
    plot.reset()
    # save new reference position for walls
    plate_x_min.state.refPos = plate_x_min.state.pos
    plate_x_max.state.refPos = plate_x_max.state.pos
    plate_y_min.state.refPos = plate_y_min.state.pos
    plate_y_max.state.refPos = plate_y_max.state.pos
    plate_z_min.state.refPos = plate_z_min.state.pos
    plate_z_max.state.refPos = plate_z_max.state.pos

    # next time, do not call this function anymore, but the next one instead
    iter_0 = O.iter
    checker.command = 'checkUnbalanced()'
    checker.iterPeriod = 500

    # add a control of the loading force
    O.engines = O.engines[:-1] + [PyRunner(command='control_Pload()', iterPeriod = 1), 
                                  PyRunner(command='controlWalls()', iterPeriod = 1)]

    # trackers    
    O.tags['i_cycle'] = '1'
    O.tags['cycle_step'] = 'unload' # start with 'unload' and then 'load' 

    global L_unbalanced_cycle, L_confinement_x_cycle, L_confinement_y_cycle, L_confinement_z_cycle, L_count_bond_cycle, L_margin_bond_cycle, L_p_cycle, L_eps_v_cycle
    L_unbalanced_cycle = []
    L_confinement_x_cycle = []
    L_confinement_y_cycle = []
    L_confinement_z_cycle = []
    L_count_bond_cycle = []
    L_margin_bond_cycle = []
    L_p_cycle = []
    L_eps_v_cycle = []

    if flag_plot_vWalls:
        global L_speed_wall
        L_speed_wall = []
        
    # user print
    print('Start cycle number '+O.tags['i_cycle']+' /', number_cycles, '('+O.tags['cycle_step']+')')

#-------------------------------------------------------------------------------

def addPlotData_cementation_ic():
    """
    Save data in plot.
    """
    # add forces applied on walls
    sx = (O.forces.f(plate_x_max.id)[0]-O.forces.f(plate_x_min.id)[0])/2/((plate_y_max.state.pos[1]-plate_y_min.state.pos[1])*(plate_z_max.state.pos[2]-plate_z_min.state.pos[2]))
    sy = (O.forces.f(plate_y_max.id)[1]-O.forces.f(plate_y_min.id)[1])/2/((plate_x_max.state.pos[0]-plate_x_min.state.pos[0])*(plate_z_max.state.pos[2]-plate_z_min.state.pos[2]))
    sz = (O.forces.f(plate_z_max.id)[2]-O.forces.f(plate_z_min.id)[2])/2/((plate_x_max.state.pos[0]-plate_x_min.state.pos[0])*(plate_y_max.state.pos[1]-plate_y_min.state.pos[1]))
    # add data
    plot.addData(i=O.iter-iter_0, porosity=porosity(), coordination=avgNumInteractions(), unbalanced=unbalancedForce(), counter_bond=0,\
                 Sx=sx, Sy=sy, Sz=sz,\
                 conf_verified= 1/3*sx/P_cementation*100 + 1/3*sy/P_cementation*100 + 1/3*sz/P_cementation*100,\
                 strain_x=100*((plate_x_max.state.pos[0]-plate_x_min.state.pos[0])-(plate_x_max.state.refPos[0]-plate_x_min.state.refPos[0]))/(plate_x_max.state.refPos[0]-plate_x_min.state.refPos[0]),
                 strain_y=100*((plate_y_max.state.pos[1]-plate_y_min.state.pos[1])-(plate_y_max.state.refPos[1]-plate_y_min.state.refPos[1]))/(plate_y_max.state.refPos[1]-plate_y_min.state.refPos[1]),
                 strain_z=100*((plate_z_max.state.pos[2]-plate_z_min.state.pos[2])-(plate_z_max.state.refPos[2]-plate_z_min.state.refPos[2]))/(plate_z_max.state.refPos[2]-plate_z_min.state.refPos[2]))

#-------------------------------------------------------------------------------

def addPlotData_confinement_ic():
    """
    Save data in plot.
    """
    # add forces applied on walls
    sx = (O.forces.f(plate_x_max.id)[0]-O.forces.f(plate_x_min.id)[0])/2/((plate_y_max.state.pos[1]-plate_y_min.state.pos[1])*(plate_z_max.state.pos[2]-plate_z_min.state.pos[2]))
    sy = (O.forces.f(plate_y_max.id)[1]-O.forces.f(plate_y_min.id)[1])/2/((plate_x_max.state.pos[0]-plate_x_min.state.pos[0])*(plate_z_max.state.pos[2]-plate_z_min.state.pos[2]))
    sz = (O.forces.f(plate_z_max.id)[2]-O.forces.f(plate_z_min.id)[2])/2/((plate_x_max.state.pos[0]-plate_x_min.state.pos[0])*(plate_y_max.state.pos[1]-plate_y_min.state.pos[1]))
    # add data
    plot.addData(i=O.iter-iter_0, porosity=porosity(), coordination=avgNumInteractions(), unbalanced=unbalancedForce(), counter_bond=count_bond(),\
                 Sx=sx, Sy=sy, Sz=sz,\
                 conf_verified= 1/3*sx/P_load*100 + 1/3*sy/P_load*100 + 1/3*sz/P_load*100, \
                 strain_x=100*((plate_x_max.state.pos[0]-plate_x_min.state.pos[0])-(plate_x_max.state.refPos[0]-plate_x_min.state.refPos[0]))/(plate_x_max.state.refPos[0]-plate_x_min.state.refPos[0]),
                 strain_y=100*((plate_y_max.state.pos[1]-plate_y_min.state.pos[1])-(plate_y_max.state.refPos[1]-plate_y_min.state.refPos[1]))/(plate_y_max.state.refPos[1]-plate_y_min.state.refPos[1]),
                 strain_z=100*((plate_z_max.state.pos[2]-plate_z_min.state.pos[2])-(plate_z_max.state.refPos[2]-plate_z_min.state.refPos[2]))/(plate_z_max.state.refPos[2]-plate_z_min.state.refPos[2]))

#-------------------------------------------------------------------------------

def saveData_ic():
    """
    Save data in .txt file during the ic.
    """
    plot.saveDataTxt('data/IC_'+O.tags['d.id']+'.txt')
    # post-proccess
    L_sigma_x = []
    L_sigma_y = []
    L_sigma_z = []
    L_sigma_mean = []
    L_confinement = []
    L_coordination = []
    L_unbalanced = []
    L_ite  = []
    L_strain_x = []
    L_strain_y = []
    L_strain_z = []
    L_strain_vol = []
    L_n_bond = []
    file = 'data/IC_'+O.tags['d.id']+'.txt'
    data = np.genfromtxt(file, skip_header=1)
    file_read = open(file, 'r')
    lines = file_read.readlines()
    file_read.close()
    if len(lines) >= 3:
        for i in range(len(data)):
            L_sigma_x.append(abs(data[i][0]))
            L_sigma_y.append(abs(data[i][1]))
            L_sigma_z.append(abs(data[i][2]))
            L_sigma_mean.append((L_sigma_x[-1]+L_sigma_y[-1]+L_sigma_z[-1])/3)
            L_confinement.append(data[i][3])
            L_coordination.append(data[i][4])
            L_n_bond.append(data[i][5])
            L_ite.append(data[i][6])
            L_strain_x.append(data[i][8])
            L_strain_y.append(data[i][9])
            L_strain_z.append(data[i][10])
            L_strain_vol.append(L_strain_x[-1]+L_strain_y[-1]+L_strain_z[-1])
            L_unbalanced.append(data[i][11])

        # plot
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize=(20,10),num=1)

        ax1.plot(L_ite, L_sigma_x, label = r'$\sigma_x$')
        ax1.plot(L_ite, L_sigma_y, label = r'$\sigma_y$')
        ax1.plot(L_ite, L_sigma_z, label = r'$\sigma_z$')
        ax1.plot(L_ite, L_sigma_mean, label = r'$\sigma_{mean}$')
        ax1.legend()
        ax1.set_title('Stresses (Pa)')

        ax2.plot(L_ite, L_unbalanced, 'b')
        ax2.set_ylabel('Unbalanced (-)', color='b')
        ax2.set_ylim(ymin=0, ymax=2*unbalancedForce_criterion_ic)
        ax2b = ax2.twinx()
        ax2b.plot(L_ite, L_confinement, 'r')
        ax2b.set_ylabel('Confinement (%)', color='r')
        ax2b.set_ylim(ymin=0, ymax=150)
        ax2b.set_title('Steady-state indices')

        ax3.plot(L_ite, L_n_bond)
        ax3.set_title('Number of bond (-)')

        ax4.plot(L_ite, L_strain_x, label=r'$\epsilon_x$ (%)')
        ax4.plot(L_ite, L_strain_y, label=r'$\epsilon_y$ (%)')
        ax4.plot(L_ite, L_strain_z, label=r'$\epsilon_z$ (%)')
        ax4.legend()
        ax4.set_title('Strains (%)')

        ax5.plot(L_ite, L_strain_vol)
        ax5.set_title('Volumeric strain (%)')

        ax6.plot(L_ite, L_coordination)
        ax6.set_title('Coordination number (-)')

        plt.savefig('plot/IC_'+O.tags['d.id']+'.png')

        plt.close()

#-------------------------------------------------------------------------------
#Load
#-------------------------------------------------------------------------------

def control_Pload():
    '''
    Control the loading force applied during the cycling loading.
    '''
    global P_load
    # unloading
    if O.tags['cycle_step'] == 'unload':
        P_load = P_load - d_P_d_iteration
    # loading
    if O.tags['cycle_step'] == 'load':
        P_load = P_load + d_P_d_iteration

#-------------------------------------------------------------------------------

def controlWalls():
    '''
    Control the upper wall to applied a defined confinement force.

    The displacement of the wall depends on the force difference. A maximum value is defined.
    '''
    Fx = (O.forces.f(plate_x_max.id)[0] - O.forces.f(plate_x_min.id)[0])/2
    if Fx == 0:
        plate_x_min.state.pos =  (min([b.state.pos[0]-0.995*b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)]),
                                        (plate_y_min.state.pos[1]+plate_y_max.state.pos[1])/2,
                                        (plate_z_min.state.pos[2]+plate_z_max.state.pos[2])/2)
        plate_x_max.state.pos =  (max([b.state.pos[0]+0.995*b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)]),
                                        (plate_y_min.state.pos[1]+plate_y_max.state.pos[1])/2,
                                        (plate_z_min.state.pos[2]+plate_z_max.state.pos[2])/2)
    else :
        dF = Fx - P_load*(plate_y_max.state.pos[1]-plate_y_min.state.pos[1])*(plate_z_max.state.pos[2]-plate_z_min.state.pos[2])
        v_plate_max = rMean*k_v_max/O.dt
        v_try_abs = abs(kp*dF)/O.dt
        # maximal speed is applied to lateral wall
        if v_try_abs < v_plate_max :
            plate_x_min.state.vel = (-np.sign(dF)*v_try_abs, 0, 0)
            plate_x_max.state.vel = ( np.sign(dF)*v_try_abs, 0, 0)
        else :
            plate_x_min.state.vel = (-np.sign(dF)*v_plate_max, 0, 0)
            plate_x_max.state.vel = ( np.sign(dF)*v_plate_max, 0, 0)

    Fy = (O.forces.f(plate_y_max.id)[1] - O.forces.f(plate_y_min.id)[1])/2
    if Fy == 0:
        plate_y_min.state.pos =  ((plate_x_min.state.pos[0]+plate_x_max.state.pos[0])/2,
                                    min([b.state.pos[1]-0.995*b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)]),
                                    (plate_z_min.state.pos[2]+plate_z_max.state.pos[2])/2)                    
        plate_y_max.state.pos =  ((plate_x_min.state.pos[0]+plate_x_max.state.pos[0])/2,
                                    max([b.state.pos[1]+0.995*b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)]),
                                    (plate_z_min.state.pos[2]+plate_z_max.state.pos[2])/2)
    else :
        dF = Fy - P_load*(plate_x_max.state.pos[0]-plate_x_min.state.pos[0])*(plate_z_max.state.pos[2]-plate_z_min.state.pos[2])
        v_plate_max = rMean*k_v_max/O.dt
        v_try_abs = abs(kp*dF)/O.dt
        # maximal speed is applied to lateral wall
        if v_try_abs < v_plate_max :
            plate_y_min.state.vel = (0, -np.sign(dF)*v_try_abs, 0)
            plate_y_max.state.vel = (0,  np.sign(dF)*v_try_abs, 0)
        else :
            plate_y_min.state.vel = (0, -np.sign(dF)*v_plate_max, 0)
            plate_y_max.state.vel = (0,  np.sign(dF)*v_plate_max, 0)

    Fz = (O.forces.f(plate_z_max.id)[2] - O.forces.f(plate_z_min.id)[2])/2
    if Fz == 0:
        plate_z_min.state.pos =  ((plate_x_min.state.pos[0]+plate_x_max.state.pos[0])/2,
                                  (plate_y_min.state.pos[1]+plate_y_max.state.pos[1])/2,
                                  min([b.state.pos[2]-0.995*b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)]))
        plate_z_max.state.pos =  ((plate_x_min.state.pos[0]+plate_x_max.state.pos[0])/2,
                                  (plate_y_min.state.pos[1]+plate_y_max.state.pos[1])/2,
                                  max([b.state.pos[2]+0.995*b.shape.radius for b in O.bodies if isinstance(b.shape, Sphere)]))
    else :
        dF = Fz - P_load*(plate_x_max.state.pos[0]-plate_x_min.state.pos[0])*(plate_y_max.state.pos[1]-plate_y_min.state.pos[1])
        v_plate_max = rMean*k_v_max/O.dt
        v_try_abs = abs(kp*dF)/O.dt
        # maximal speed is applied to top wall
        if v_try_abs < v_plate_max :
            plate_z_min.state.vel = (0, 0, -np.sign(dF)*v_try_abs)
            plate_z_max.state.vel = (0, 0,  np.sign(dF)*v_try_abs)
        else :
            plate_z_min.state.vel = (0, 0, -np.sign(dF)*v_plate_max)
            plate_z_max.state.vel = (0, 0,  np.sign(dF)*v_plate_max)

#-------------------------------------------------------------------------------

def count_bond():
    '''
    Count the number of bond.
    '''
    counter_bond = 0
    for i in O.interactions:
        if isinstance(O.bodies[i.id1].shape, Sphere) and isinstance(O.bodies[i.id2].shape, Sphere):
            if not i.phys.cohesionBroken :
                counter_bond = counter_bond + 1
            # correct Young modulus if bond broken 
            else :
                if i.phys.kn != 80*1e6*(O.bodies[i.id1].shape.radius*2*O.bodies[i.id2].shape.radius*2)/(O.bodies[i.id1].shape.radius*2+O.bodies[i.id2].shape.radius*2):
                    localYoungModulus = 80*1e6 # Pa
                    i.phys.kn = localYoungModulus*(O.bodies[i.id1].shape.radius*2*O.bodies[i.id2].shape.radius*2)/(O.bodies[i.id1].shape.radius*2+O.bodies[i.id2].shape.radius*2)
                    i.phys.ks = 0.25*localYoungModulus*(O.bodies[i.id1].shape.radius*2*O.bodies[i.id2].shape.radius*2)/(O.bodies[i.id1].shape.radius*2+O.bodies[i.id2].shape.radius*2) # 0.25 is the Poisson ratio
                    i.phys.kr = i.phys.ks*alphaKrReal*O.bodies[i.id1].shape.radius*O.bodies[i.id2].shape.radius
                    i.phys.ktw = i.phys.ks*alphaKtwReal*O.bodies[i.id1].shape.radius*O.bodies[i.id2].shape.radius
    return counter_bond

#-------------------------------------------------------------------------------

def compute_margin():
    '''
    Compute the margin of the bonds before the rupture.

    The maximum between the tensile and shear is considered.
    '''
    # compute the margin for the bonds (force/stiffness) 
    margin_bond = 0
    counter_margin = 0
    for i in O.interactions:
        if isinstance(O.bodies[i.id1].shape, Sphere) and isinstance(O.bodies[i.id2].shape, Sphere):
            if not i.phys.cohesionBroken :
                # tensile margin
                if i.geom.penetrationDepth < 0:
                    tensile_margin = np.linalg.norm(i.phys.normalForce)/i.phys.normalAdhesion
                else :
                    tensile_margin = 0
                # shear margin 
                shear_margin = np.linalg.norm(i.phys.shearForce)/(i.phys.shearAdhesion+i.phys.tangensOfFrictionAngle*np.linalg.norm(i.phys.normalForce))
                # take the maximum (largest potential to crack)
                margin_bond = margin_bond + max(tensile_margin, shear_margin)
                counter_margin = counter_margin + 1
    return margin_bond/counter_margin
        
#-------------------------------------------------------------------------------

def checkUnbalanced():
    """
    Look for the equilibrium during the loading phase.
    """
    # save data
    saveData()

    if flag_plot_vWalls:
        L_speed_wall.append(np.mean([abs(plate_x_min.state.vel[0]), abs(plate_x_max.state.vel[0]),\
                                    abs(plate_y_min.state.vel[1]), abs(plate_y_max.state.vel[1]),\
                                    abs(plate_z_min.state.vel[1]), abs(plate_z_max.state.vel[2])]))
        fig, (ax1) = plt.subplots(1,1, figsize=(16,9),num=1)
        ax1.plot(L_speed_wall)
        ax1.plot([0, len(L_speed_wall)-1], [rMean*k_v_max/O.dt, rMean*k_v_max/O.dt], color='k')
        fig.savefig('plot/debug_speed_wall.png')
        plt.close()

    # save data but only for the current cycle (unloading-loading)
    global L_unbalanced_cycle, L_confinement_x_cycle, L_confinement_y_cycle, L_confinement_z_cycle, L_count_bond_cycle, L_margin_bond_cycle, L_p_cycle, L_eps_v_cycle
    # track unbalanced in the cycle (should be < 0.1-0.01)
    L_unbalanced_cycle.append(unbalancedForce())
    # track confinement (should be ~ 100%)
    L_confinement_x_cycle.append((O.forces.f(plate_x_max.id)[0]-O.forces.f(plate_x_min.id)[0])/2/(P_load*(plate_y_max.state.pos[1]-plate_y_min.state.pos[1])*(plate_z_max.state.pos[2]-plate_z_min.state.pos[2]))*100)
    L_confinement_y_cycle.append((O.forces.f(plate_y_max.id)[1]-O.forces.f(plate_y_min.id)[1])/2/(P_load*(plate_x_max.state.pos[0]-plate_x_min.state.pos[0])*(plate_z_max.state.pos[2]-plate_z_min.state.pos[2]))*100)
    L_confinement_z_cycle.append((O.forces.f(plate_z_max.id)[2]-O.forces.f(plate_z_min.id)[2])/2/(P_load*(plate_x_max.state.pos[0]-plate_x_min.state.pos[0])*(plate_y_max.state.pos[1]-plate_y_min.state.pos[1]))*100)
    # track bonds number
    L_count_bond_cycle.append(count_bond())
    # track mean stress p
    L_p_cycle.append(1/3*(O.forces.f(plate_x_max.id)[0]-O.forces.f(plate_x_min.id)[0])/2/((plate_y_max.state.pos[1]-plate_y_min.state.pos[1])*(plate_z_max.state.pos[2]-plate_z_min.state.pos[2]))+\
                     1/3*(O.forces.f(plate_y_max.id)[1]-O.forces.f(plate_y_min.id)[1])/2/((plate_x_max.state.pos[0]-plate_x_min.state.pos[0])*(plate_z_max.state.pos[2]-plate_z_min.state.pos[2]))+\
                     1/3*(O.forces.f(plate_z_max.id)[2]-O.forces.f(plate_z_min.id)[2])/2/((plate_x_max.state.pos[0]-plate_x_min.state.pos[0])*(plate_y_max.state.pos[1]-plate_y_min.state.pos[1])))
    # track volumetric strain
    L_eps_v_cycle.append(100*((plate_x_max.state.pos[0]-plate_x_min.state.pos[0])-(plate_x_max.state.refPos[0]-plate_x_min.state.refPos[0]))/(plate_x_max.state.refPos[0]-plate_x_min.state.refPos[0])+\
                         100*((plate_y_max.state.pos[1]-plate_y_min.state.pos[1])-(plate_y_max.state.refPos[1]-plate_y_min.state.refPos[1]))/(plate_y_max.state.refPos[1]-plate_y_min.state.refPos[1])+\
                         100*((plate_z_max.state.pos[2]-plate_z_min.state.pos[2])-(plate_z_max.state.refPos[2]-plate_z_min.state.refPos[2]))/(plate_z_max.state.refPos[2]-plate_z_min.state.refPos[2]))
    # track the margin for the bonds  
    L_margin_bond_cycle.append(compute_margin())

    # plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(16,9),num=1)
    # unbalanced
    ax1.plot(L_unbalanced_cycle)
    ax1.set_title('unbalanced force (-)')
    ax1.set_ylim(ymin=0, ymax=2*0.01)
    # confinement
    ax2.plot(L_confinement_x_cycle)
    ax2.plot(L_confinement_y_cycle)
    ax2.plot(L_confinement_z_cycle)
    ax2.set_ylim(ymin=90, ymax=110)
    ax2.set_title('confinements (%)')
    # number of bond
    ax3.plot(L_count_bond_cycle, color='b')
    ax3.set_title('Bonds')
    ax3.set_ylabel('Number of bonds (-)', color='b')
    ax3b = ax3.twinx()
    ax3b.plot(L_margin_bond_cycle, color='r')
    ax3b.set_ylabel('Bond margin (-)', color='r')
    # eps_v - p
    ax4.plot(L_eps_v_cycle, L_p_cycle)
    ax4.set_title(r'p (Pa) - $\epsilon_v$ (%)')
    # close
    fig.savefig('plot/tracking_cycle_'+O.tags['i_cycle']+'.png')
    plt.close()

    # the unload of the cycle is verified
    if O.tags['cycle_step'] == 'unload' and P_load < P_load0 - amplitude_P_cycle:
        # next step of the cycle, the loading
        O.tags['cycle_step'] = 'load'

    # the load of the cycle is verified
    elif O.tags['cycle_step'] == 'load' and P_load0 < P_load:
        # characterize the cycle
        global tic
        tac = time.perf_counter()
        hours = (tac-tic)//(60*60)
        minutes = (tac-tic -hours*60*60)//(60)
        seconds = int(tac-tic -hours*60*60 -minutes*60)
        tic = tac
        # report
        simulation_report = open(simulation_report_name, 'a')
        simulation_report.write("Cycle "+O.tags['i_cycle']+" : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds\n")
        simulation_report.write(str(n_grains)+' grains\n')
        simulation_report.write(str(L_count_bond_cycle[0]-L_count_bond_cycle[-1])+' bond breakages\n')
        simulation_report.close()

        # next cycle, start with the unloading  
        O.tags['cycle_step'] = 'unload'
        O.tags['i_cycle'] = str(int(O.tags['i_cycle']) + 1)

        # check if the simulation should continue (number of cycles to do)
        if int(O.tags['i_cycle']) > number_cycles:
            stopLoad()
            return
        else:
            # reset the trackers
            L_unbalanced_cycle = []
            L_confinement_x_cycle = []
            L_confinement_y_cycle = []
            L_confinement_z_cycle = []
            L_count_bond_cycle = []
            L_margin_bond_cycle = []
            L_p_cycle = []
            L_eps_v_cycle = []

    # the step of the cycle (unload or load) should continue
    else :
        return

    # user print
    print('Start cycle number '+O.tags['i_cycle']+' /', number_cycles, '('+O.tags['cycle_step']+')')
        
#-------------------------------------------------------------------------------

def stopLoad():
    """
    Close simulation.
    """
    # close yade
    O.pause()
    # characterize the simulation
    tac = time.perf_counter()
    hours = (tac-tic_0)//(60*60)
    minutes = (tac-tic_0 -hours*60*60)//(60)
    seconds = int(tac-tic_0 -hours*60*60 -minutes*60)
    # report
    simulation_report = open(simulation_report_name, 'a')
    simulation_report.write("Simulation time : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds\n\n")
    simulation_report.write(str(count_bond())+" contacts cemented finally\n\n")
    simulation_report.close()
    print("\nSimulation time : "+str(hours)+" hours "+str(minutes)+" minutes "+str(seconds)+" seconds\n")
    print('\n'+str(count_bond())+" contacts cemented\n")

    # save simulation
    os.mkdir('../Data_Rock_Cyclic_Isotropic/'+O.tags['d.id'])
    shutil.copytree('data','../Data_Rock_Cyclic_Isotropic/'+O.tags['d.id']+'/data')
    shutil.copytree('plot','../Data_Rock_Cyclic_Isotropic/'+O.tags['d.id']+'/plot')
    shutil.copy('Rock_Cyclic_Isotropic.py','../Data_Rock_Cyclic_Isotropic/'+O.tags['d.id']+'/Rock_Cyclic_Isotropic.py')
    shutil.move(O.tags['d.id']+'_report.txt','../Data_Rock_Cyclic_Isotropic/'+O.tags['d.id']+'/'+O.tags['d.id']+'_report.txt')

#-------------------------------------------------------------------------------

def addPlotData():
    """
    Save data in plot.
    """
    # add forces applied on wall x and z
    sx = O.forces.f(plate_x_max.id)[0]/(plate_y_max.state.pos[1]*plate_z_max.state.pos[2])
    sy = O.forces.f(plate_y_max.id)[1]/(plate_x_max.state.pos[0]*plate_z_max.state.pos[2])
    sz = O.forces.f(plate_z_max.id)[2]/(plate_x_max.state.pos[0]*plate_y_max.state.pos[1])
    # add data
    plot.addData(i=O.iter-iter_0, porosity=porosity(), coordination=avgNumInteractions(), unbalanced=unbalancedForce(),\
                counter_bond=count_bond(), bond_margin=compute_margin(),\
                Sx=sx, Sy=sy, Sz=sz, \
                X_plate=plate_x_max.state.pos[0], Y_plate=plate_y_max.state.pos[1], Z_plate=plate_z_max.state.pos[2],\
                conf_verified=1/3*sz/P_load*100 + 1/3*sx/(P_load)*100 + 1/3*sy/(P_load)*100, \
                strain_x=100*(plate_x_max.state.pos[0]-plate_x_max.state.refPos[0])/plate_x_max.state.refPos[0],\
                strain_y=100*(plate_y_max.state.pos[1]-plate_y_max.state.refPos[1])/plate_y_max.state.refPos[1],\
                strain_z=100*(plate_z_max.state.pos[2]-plate_z_max.state.refPos[2])/plate_z_max.state.refPos[2])

#-------------------------------------------------------------------------------

def saveData():
    """
    Save data in .txt file during the steps.
    """
    addPlotData()
    plot.saveDataTxt('data/'+O.tags['d.id']+'.txt')
    # post-proccess
    L_coordination = []
    L_n_bond = []
    L_margin_bond = []
    L_sigma_x = []
    L_sigma_y = []
    L_sigma_z = []
    L_sigma_mean = []
    L_strain_z = []
    L_strain_x = []
    L_strain_y = []
    L_vol_strain = []
    file = 'data/'+O.tags['d.id']+'.txt'
    data = np.genfromtxt(file, skip_header=1)
    file_read = open(file, 'r')
    lines = file_read.readlines()
    file_read.close()
    if len(lines) >= 3:
        for i in range(len(data)):
            L_sigma_x.append(data[i][0])
            L_sigma_y.append(data[i][1])
            L_sigma_z.append(data[i][2])
            L_sigma_mean.append(1/3*L_sigma_x[-1]+1/3*L_sigma_y[-1]+1/3*L_sigma_z[-1])
            L_margin_bond.append(data[i][6])
            L_coordination.append(data[i][8])
            L_n_bond.append(data[i][9])
            L_strain_x.append(data[i][12])
            L_strain_y.append(data[i][13])
            L_strain_z.append(data[i][14])
            L_vol_strain.append(abs(L_strain_x[-1]+L_strain_y[-1]+L_strain_z[-1]))

        # plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(16,9),num=1)

        ax1.plot(L_vol_strain, L_sigma_mean)
        ax1.set_title(r'p vs. $\epsilon_v$')
        ax1.set_xlabel(r'$\epsilon_v$ (%)')
        ax1.set_ylabel(r'Mean stress (Pa)')

        ax2.plot(L_coordination)
        ax2.set_title('Coordination (-)')

        ax3.plot(L_n_bond)
        ax3.set_title('Number of bond (-)')

        ax4.plot(L_margin_bond_cycle)
        ax4.set_ylabel('Bond margin (-)')

        plt.savefig('plot/'+O.tags['d.id']+'.png')
        plt.close()

#-------------------------------------------------------------------------------
# start simulation
#-------------------------------------------------------------------------------

O.run()
waitIfBatch()
