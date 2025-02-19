#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import opengate as gate
import opengate.contrib.phantoms.nemaiec as gate_iec
#import opengate.contrib.phantoms.jaszczak as gate_jaszczak

import picopet_source as gate_iec

from box import Box

from pathlib import Path
##change
#import opengate.contrib.pet.philipsvereos as pet_vereos

import picopet as pet_vereos

from pet_helpers import add_vereos_digitizer_v1
from opengate.geometry.utility import get_circular_repetition
from opengate.sources.base import get_rad_yield

if __name__ == "__main__":
    sim = gate.Simulation() #initializes new simulation

    # options
    # warning the visualisation is slow !
    sim.visu = True
    sim.visu_type = "qt"
    sim.random_seed = "auto"
    sim.number_of_threads = 1
    sim.progress_bar = True
    sim.output_dir = "./output"
    data_path = Path("data")

    # units
    m = gate.g4_units.m
    mm = gate.g4_units.mm
    cm = gate.g4_units.cm
    cm3 = gate.g4_units.cm3
    sec = gate.g4_units.s
    ps = gate.g4_units.ps
    keV = gate.g4_units.keV
    Bq = gate.g4_units.Bq
    gcm3 = gate.g4_units.g_cm3
    BqmL = Bq / cm3


    # world
    world = sim.world
    ## change
    #world.size = [0.5 * m, 0.5 * m, 0.5 * m]
    world.size = [1.2 * m, 1.2 * m, 1.2 * m]

    world.material = "G4_AIR" #setting the world material to air 

    # add the Philips Vereos PET
    pet = pet_vereos.add_pet(sim, "pet", create_housing=False)

    # If visu is enabled, we simplified the PET system, otherwise it is too slow
    #if sim.visu:
    #    module = sim.volume_manager.get_volume("pet_module")
    #    # only 2 repetition instead of 18
    #    translations_ring, rotations_ring = get_circular_repetition(
    #        2, [391.5 * mm, 0, 0], start_angle_deg=190, axis=[0, 0, 1]
    #    )
    #    module.translation = translations_ring
    #    module.rotation = rotations_ring

    # add table
    #bed = pet_vereos.add_table(sim, "pet")

    iec_phantom = gate_iec.add_iec_phantom(sim, 'iec_phantom')
    activities = [3 * Bq, 4 * Bq, 5 * Bq, 6 * Bq, 9 * Bq, 12 * Bq]
    #iec_source = gate_iec.add_spheres_sources(sim, 'iec_phantom', 'iec_source', 'all', activities)

    # Add positron-emitting sources to all spheres
    sphere_names = [
    "iec_phantom_sphere_37mm", "iec_phantom_sphere_28mm",
    "iec_phantom_sphere_22mm", "iec_phantom_sphere_17mm",
    "iec_phantom_sphere_13mm", "iec_phantom_sphere_10mm"
    ]

    for sphere, activity in zip(sphere_names, activities):
        source = sim.add_source("GenericSource", f"{sphere}_source")
        source.attached_to = sphere
        source.position.type = "sphere"
        source.position.radius = 0.5 * cm
        source.particle = "e+"  # Positron emission
        source.energy.type = "F18"
        source.activity = activity
        source.half_life = 6586.26 * sec  # F18 half-life



    cylinder_source = gate_iec.add_central_cylinder_source(sim, "iec_phantom", "cylinder_source", activity_Bq_mL=0.000000001, verbose=False)
    iec_bg_source = gate_iec.add_background_source(sim, 'iec_phantom', 'iec_bg_source', 0.1 * Bq)

    # List of IEC phantom sphere names

    #jaszczak_phantom = gate_jaszczak.add_jaszczak_phantom(sim)
    #jaszczak_source = gate_jaszczak.add_background_source(sim, jaszczak_name="jaszczak", src_name="source", activity_bqml=1e-14)
    
   # add a simple waterbox with a hot sphere inside
    #waterbox = sim.add_volume("Box", "waterbox") #ORIGINAL
    #waterbox.size = [12 * cm, 12 * cm, 12 * cm]

    '''
    #creating a cylinder waterbox
    waterbox = sim.add_volume("TubsVolume", "waterbox")
    waterbox.rmin = 0.0 * cm
    waterbox.rmax = 0.5 * cm
    waterbox.dz = 1 * cm
    '''

    #waterbox.translation = [4 * cm, 2 * cm, 0 * cm]
    #waterbox.material = "G4_WATER"
    #waterbox.color = [0, 0, 1, 1]
    

    #hot_sphere = sim.add_volume("Sphere", "hot_sphere")
    #hot_sphere.mother = waterbox.name
    #hot_sphere.rmax = 2 * cm
    #hot_sphere.translation = [4 * cm, 2 * cm , 0]
    #hot_sphere.material = "G4_WATER"
    #hot_sphere.color = [1, 0, 0, 1]

    #Adding a source to the hot sphere
    #hot_source = sim.add_source("GenericSource", "hot_sphere_source")
    #hot_source.attached_to = "hot_sphere"
    #hot_source.position.type = "sphere"
    #hot_source.position.radius = 2 * cm #Same as the hot sphere size
    #hot_source.particle = "e+"
    #hot_source.energy.type = "F18"
    #hot_source.activity = 5e6 * Bq #Higher activity than the waterbox
    #hot_source.half_life = 6586.26 * sec

 
    # source for tests
    #source = sim.add_source("GenericSource", "waterbox_source")
    #total_yield = get_rad_yield("F18")
    #print("Yield for F18 (nb of e+ per decay) : ", total_yield)
    #source.attached_to = "waterbox"

    #creating spherical source
    #source.distribution_type = "Volume"
    #source.shape = "Sphere"
    #source.radius = 0.4 * cm #Positron source spread over sphere

    #source.particle = "e+"
    #source.energy.type = "F18"
    #source.activity = 1e6 * Bq * total_yield
    #if sim.visu:
    #    source.activity = 1e5 * Bq * total_yield
    #source.half_life = 6586.26 * sec
 

    # add IEC phantom
    #iec_phantom = gate_iec.add_iec_phantom(sim, "iec")
    #iec_phantom.translation = [0 * cm, 0 * cm, 0 * cm]

    # source for tests
    #total_yield = get_rad_yield("Ga68")
    #print("Yield for Ga68 (nb of e+ per decay) : ", total_yield)
    #a = 1e3* BqmL * total_yield
    #if sim.visu:
    #    a = 1e2 * BqmL * total_yield
    #activities = [a] * 6
    #sources = gate_iec.add_spheres_sources(sim, "iec", "iec_source", "all", activities)
    #for source in sources:
    #    source.particle = "e+"
    #    source.energy.type = "Ga68"
    #    source.half_life = 67.71 * 60 * sec


    # physics
    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option3"
    sim.physics_manager.enable_decay = True
    sim.physics_manager.set_production_cut("world", "all", 1 * m)
    #sim.physics_manager.set_production_cut("waterbox", "all", 1 * mm)

    # add the PET digitizer
    add_vereos_digitizer_v1(sim, pet, f"output_vereos.root")

    # add stat actor
    stats = sim.add_actor("SimulationStatisticsActor", "Stats")
    stats.track_types_flag = True
    stats.output_filename = "stats_vereos.txt"

    # timing
    sim.run_timing_intervals = [[0, 10 * sec]]

    # go
    sim.run()

    # end
    """print(f"Output statistics are in {stats.output}")
    print(f"Output edep map is in {dose.output}")
    print(f"vv {ct.image} --fusion {dose.output}")
    stats = sim.output.get_actor("Stats")
    print(stats)"""
