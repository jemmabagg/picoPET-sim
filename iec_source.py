import opengate as gate
import opengate.contrib.phantoms.nemaiec as gate_iec
from pathlib import Path

#Initialise the simulation
sim = gate.Simulation()

#Simulation Options
sim.visu = True #enable visualisation
sim.visu_type = "qt"
sim.random_seed = "auto"
sim.number_of_threads = 1
sim.progress_bar = True 
sim.output_dir = "./output"

m = gate.g4_units.m
cm = gate.g4_units.cm 
sec = gate.g4_units.s 
Bq = gate.g4_units.Bq 

#Define world
world = sim.world
world.size = [1.2 * m, 1.2 * m, 1.2 * m] # Large enough for the PET scanner
world.material = "G4_AIR"

#Add the Philips Vereos PET Scanner
import picopet as pet_vereos
pet = pet_vereos.add_pet(sim, "pet", create_housing=False)

#Add the IEC phantom
iec_phantom = gate_iec.add_iec_phantom(sim, "iec_phantom")

#Define the phantom sphere names
sphere_names = [
    "iec_phantom_sphere_37mm", "iec_phantom_sphere_28mm",
    "iec_phantom_sphere_22mm", "iec_phantom_sphere_17mm",
    "iec_phantom_sphere_13mm", "iec_phantom_sphere_10mm"
]

#Define activity levels for each sphere
#activities = [3e7 *Bq, 4e7 * Bq, 5e7 * Bq, 6e7 * Bq, 9e7 * Bq, 12e7 * Bq]
activities = [1e6 * Bq] * len(sphere_names)  # Reduce activity to 1 million Bq per sphere - to make simulation run faster


#Add a hot positron-emitting source to each sphere
for sphere, acitvity in zip(sphere_names, activities):
    source = sim.add_source("GenericSource", f"{sphere}_source")
    source.attached_to = sphere
    source.position.type = "sphere"
    source.position.radius = 0.5 * cm 
    source.particle = "e+" #Emit positrons
    source.energy.type = "F18"
    source.activity = acitvity
    source.half_life = 6586.26 * sec

#Add in a background source 
iec_bg_source = gate_iec.add_background_source(sim, "iec_phantom", "iec_bg_source", 0.1 * Bq)

# Define physics settings
sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option3"
sim.physics_manager.enable_decay = True
sim.physics_manager.set_production_cut("world", "all", 1 * m)
sim.physics_manager.set_production_cut("iec_phantom", "all", 1 * cm)

# Add PET digitizer
from pet_helpers import add_vereos_digitizer_v1
add_vereos_digitizer_v1(sim, pet, "output_vereos.root")

# Add simulation statistics actor
stats = sim.add_actor("SimulationStatisticsActor", "Stats")
stats.track_types_flag = True
stats.output_filename = "stats_vereos.txt"

# Define simulation timing
sim.run_timing_intervals = [[0, 10 * sec]]

# Run the simulation
sim.run()